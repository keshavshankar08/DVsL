import os
import io
import time
import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from fvcore.nn import FlopCountAnalysis

from model_registry import BaseSDNN

# ==============================================================================
# Configuration
# ==============================================================================
DATA_DIR = "data"
IMG_SIZE = 128
DEVICE   = torch.device("cpu")

# ==============================================================================
# Data
# ==============================================================================
def get_dataloaders(batch_size: int = 8):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    classes      = full_dataset.classes

    train_size = int(0.80 * len(full_dataset))
    val_size   = int(0.10 * len(full_dataset))
    test_size  = len(full_dataset) - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, classes


# ==============================================================================
# Quantization Extraction (SLAYER Native)
# ==============================================================================
def extract_quantized_weights(slayer_model: nn.Module, save_path: str = "quantized_sdnn.npz") -> float:
    """
    Extracts weights from a trained SLAYER model, applies QAT scaling, 
    converts them to 8-bit integers, and saves to a compressed .npz file.
    Returns the file size in MB.
    """
    quantized_state = {}
    print("\nExtracting and converting INT8 weights...")
    
    # 1. Strip weight_norm so 'weight' becomes a normal accessible tensor again
    for module in slayer_model.modules():
        if hasattr(module, 'weight_g') and hasattr(module, 'weight_v'):
            try:
                torch.nn.utils.remove_weight_norm(module)
            except ValueError:
                pass

    # 2. Extract weights specifically from SLAYER's synapse submodules
    extracted_count = 0
    for name, module in slayer_model.named_modules():
        # Look specifically for 'synapse' modules that hold the weights (Conv/Dense)
        if 'synapse' in name and hasattr(module, 'weight'):
            raw_weights = module.weight.detach().cpu().numpy()
            
            # Retrieve the weight scale. If SLAYER doesn't expose it on this module, 
            # we default to 2 (since you set weight_scale=2 in BaseSDNN parameters)
            w_scale = getattr(module, 'weight_scale', 2)
            
            # SLAYER scales weights by 2^(weight_scale) -> equivalent to a bitwise shift
            scaled_weights = raw_weights * (1 << w_scale)
            
            # Round to nearest integer and clip to standard INT8 bounds (-128 to 127)
            int8_weights = np.clip(np.round(scaled_weights), -128, 127).astype(np.int8)
            
            quantized_state[name] = int8_weights
            print(f"  {name:<25} -> shape: {str(int8_weights.shape):<20} dtype: {int8_weights.dtype}")
            extracted_count += 1
            
    if extracted_count == 0:
        print("  [ERROR] No synapse weights were found! The extraction logic missed the layers.")
            
    np.savez_compressed(save_path, **quantized_state)
    print(f"  Saved {extracted_count} quantized weight tensors -> {save_path}")
    
    return os.path.getsize(save_path) / (1024 ** 2)

# ==============================================================================
# Metrics
# ==============================================================================
def get_model_size_mb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue()) / (1024 ** 2)


def evaluate_model(model: nn.Module, loader: DataLoader) -> tuple[float, list, list]:
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.unsqueeze(-1) # [B, C, H, W] -> [B, C, H, W, T=1]
            outputs = model(inputs)

            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            logits = logits.flatten(start_dim=1).float()
            preds   = logits.argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    return 100 * correct / total, all_preds, all_labels


def measure_latency(model: nn.Module, dummy_input: torch.Tensor,
                    n_warmup: int = 20, n_runs: int = 200) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            times.append((time.perf_counter() - t0) * 1000)

    t = torch.tensor(times)
    return t.mean().item(), t.std().item()


def measure_flops(model: nn.Module, dummy_input: torch.Tensor) -> str:
    try:
        analyzer = FlopCountAnalysis(model, dummy_input)
        analyzer.unsupported_ops_warnings(False)
        total = analyzer.total()
        if total == 0:
            return "N/A"
        return f"{total / 1e6:.1f}M"
    except Exception:
        return "N/A"


def clear_hooks(model: nn.Module) -> None:
    for m in model.modules():
        m._forward_hooks.clear()
        m._forward_pre_hooks.clear()
        m._backward_hooks.clear()


def plot_confusion_matrix(all_labels, all_preds, classes, title, filename):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"  Confusion matrix saved → {filename}")


def load_sdnn_weights(model: nn.Module, weight_path: str) -> nn.Module:
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
    for name, module in model.named_modules():
        for attr in ("running_mean", "running_var"):
            ckpt_key = f"{name}.{attr}"
            if hasattr(module, attr) and ckpt_key in checkpoint:
                ckpt_shape = checkpoint[ckpt_key].shape
                current_shape = getattr(module, attr).shape
                if ckpt_shape != current_shape:
                    print(f"  Resizing {ckpt_key}: {current_shape} → {ckpt_shape}")
                    module.register_buffer(attr, torch.zeros(ckpt_shape))

    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    weight_path = "sdnn_v1.pth"
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"{weight_path} not found. Train your SDNN first and save weights there."
        )

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    num_classes = len(classes)

    # ── Load Model ───────────────────────────────────────────────────────────
    print("Loading SDNN weights …")
    base_sdnn = BaseSDNN(num_classes=num_classes)
    base_sdnn = load_sdnn_weights(base_sdnn, weight_path)
    print("Weights loaded.")

    # ── Dummy inputs ─────────────────────────────────────────────────────────
    dummy_input = torch.randn(1, 1, IMG_SIZE, IMG_SIZE, 1)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print("\nEvaluating Model (QAT Baseline) ...")
    acc, preds, labels_list = evaluate_model(base_sdnn, test_loader)
    pt_size_mb = get_model_size_mb(base_sdnn)
    flops      = measure_flops(base_sdnn, dummy_input)
    lat_mean, lat_std = measure_latency(base_sdnn, dummy_input)

    col = 22
    print("\n" + "=" * 88)
    print(f"{'Model':<{col}} | {'Acc (%)':>8} | {'PyTorch Size':>12} | {'FLOPs':>10} | {'Latency (ms/img)':>18}")
    print("-" * 88)
    lat_str = f"{lat_mean:.2f} ± {lat_std:.2f}"
    print(f"{'SDNN (QAT Baseline)':<{col}} | {acc:>8.2f} | {pt_size_mb:>9.3f} MB | {flops:>10} | {lat_str:>18}")
    print("=" * 88)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    plot_confusion_matrix(labels_list, preds, classes,
                          title="Confusion Matrix — SDNN",
                          filename="sdnn_qat_confusion_matrix.png")
    
    # Clean profiling hooks just in case
    clear_hooks(base_sdnn)

    # ── Extract Quantized Weights for Edge Deployment ─────────────────────────
    npz_save_path = "quantized_sdnn_weights.npz"
    int8_size_mb = extract_quantized_weights(base_sdnn, save_path=npz_save_path)

    # ── Size reduction summary ────────────────────────────────────────────────
    print(f"\nSize Reduction Summary:")
    print(f"  PyTorch Checkpoint (FP32/Metadata): {pt_size_mb:.3f} MB")
    print(f"  Raw INT8 Weights (.npz):            {int8_size_mb:.3f} MB")
    if int8_size_mb > 0:
        print(f"  (Approx. {pt_size_mb / int8_size_mb:.2f}× smaller parameter footprint)")