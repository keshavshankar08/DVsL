import os
import time
import io
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as quant
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from fvcore.nn import FlopCountAnalysis

from model_registry import MODEL_REGISTRY, BaseCNN

# ==============================================================================
# Global Configuration
# ==============================================================================
TARGET_ARCH = "cnn_v1"
DATA_DIR    = "data"
IMG_SIZE    = 128
EPOCHS      = 15
DEVICE      = torch.device("cpu")   # PTQ INT8 requires CPU / fbgemm

# ==============================================================================
# Data
# ==============================================================================
def get_dataloaders(batch_size: int = 32):
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
# Training
# ==============================================================================
def train_cnn(train_loader, val_loader, test_loader, classes):
    if TARGET_ARCH not in MODEL_REGISTRY:
        raise ValueError(f"Architecture '{TARGET_ARCH}' not in MODEL_REGISTRY.")

    num_classes = len(classes)
    model_path  = f"{TARGET_ARCH}.pth"

    model     = MODEL_REGISTRY[TARGET_ARCH]["class"](num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    print(f"\n--- Training {TARGET_ARCH} ---")
    for epoch in range(EPOCHS):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            logits = model(inputs).flatten(start_dim=1)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct      += (logits.argmax(1) == labels).sum().item()
            total        += labels.size(0)

        train_losses.append(running_loss / len(train_loader))
        train_accs.append(100 * correct / total)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                logits    = model(inputs).flatten(start_dim=1)
                val_loss += criterion(logits, labels).item()
                correct  += (logits.argmax(1) == labels).sum().item()
                total    += labels.size(0)

        val_losses.append(val_loss / len(val_loader))
        val_accs.append(100 * correct / total)
        scheduler.step()

        print(
            f"Epoch {epoch+1:>3}/{EPOCHS}  |  "
            f"Train Loss {train_losses[-1]:.4f}  Acc {train_accs[-1]:.2f}%  |  "
            f"Val Loss {val_losses[-1]:.4f}  Acc {val_accs[-1]:.2f}%"
        )

    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved → {model_path}")
    _plot_training(train_losses, val_losses, train_accs, val_accs)
    evaluate_and_plot(model, test_loader, classes)
    return model


def _plot_training(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="Train"); ax1.plot(val_losses, label="Val")
    ax1.set_title("Loss"); ax1.legend()
    ax2.plot(train_accs,   label="Train"); ax2.plot(val_accs,   label="Val")
    ax2.set_title("Accuracy (%)"); ax2.legend()
    plt.tight_layout()
    plt.savefig(f"{TARGET_ARCH}_training_curves.png")
    print(f"Training curves saved → {TARGET_ARCH}_training_curves.png")


def evaluate_and_plot(model, test_loader, classes):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            preds = model(inputs).flatten(start_dim=1).argmax(1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    acc = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"Test Accuracy: {acc:.3f}%")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {TARGET_ARCH}")
    plt.tight_layout()
    plt.savefig(f"{TARGET_ARCH}_confusion_matrix.png")
    print(f"Confusion matrix saved → {TARGET_ARCH}_confusion_matrix.png")


# ==============================================================================
# Quantization
# ==============================================================================
class QuantWrapper(nn.Module):
    """Wraps a CNN with QuantStub / DeQuantStub for INT8 PTQ."""
    def __init__(self, model):
        super().__init__()
        self.quant   = quant.QuantStub()
        self.model   = model
        self.dequant = quant.DeQuantStub()

    def forward(self, x):
        return self.dequant(self.model(self.quant(x)))


def fold_bn_into_linear(model: nn.Module) -> nn.Module:
    """
    Absorbs every BatchNorm1d that immediately follows a Linear layer
    into that Linear's weight and bias, then replaces the BN with Identity.

    Why this is necessary for INT8 PTQ
    ------------------------------------
    PyTorch's fuse_modules() supports  Conv+BN+ReLU  and  Linear+ReLU,
    but NOT  Linear+BN+ReLU.  If a BN1d is left in the graph after the
    QuantStub fires, it receives a QuantizedCPU tensor and crashes with:
        NotImplementedError: 'aten::native_batch_norm' not implemented
        for backend QuantizedCPU.

    The fix is to eliminate the BN entirely by absorbing its parameters
    into the preceding Linear before wrapping with QuantWrapper.

    Math
    ----
    Given  Linear: y = W·x + b
    and    BN:     z = γ · (y − μ) / √(σ² + ε) + β

    The composed operation is:
        z = (γ/√(σ²+ε)) · W · x  +  γ·(b−μ)/√(σ²+ε) + β

    So the folded parameters are:
        W' = W  ·  diag(γ/√(σ²+ε))          [scale each output row]
        b' = (b − μ) · γ/√(σ²+ε)  +  β
    """
    model = copy.deepcopy(model)

    # Walk every Sequential in the model and look for (Linear, BN1d) pairs
    for seq in model.modules():
        if not isinstance(seq, nn.Sequential):
            continue

        children = list(seq.named_children())   # [(name, module), ...]
        for idx in range(len(children) - 1):
            name_lin, lin = children[idx]
            name_bn,  bn  = children[idx + 1]

            if not (isinstance(lin, nn.Linear) and isinstance(bn, nn.BatchNorm1d)):
                continue

            # Ensure BN is in eval mode so running stats are frozen
            bn.eval()

            with torch.no_grad():
                # Scale factor per output feature: γ / √(σ² + ε)
                scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)  # [out]

                # Fold into Linear weight  [out, in] → row-wise scale
                lin.weight.data = lin.weight.data * scale.unsqueeze(1)

                # Fold into Linear bias (create it if the layer had none)
                if lin.bias is None:
                    lin.bias = nn.Parameter(torch.zeros(lin.out_features))
                lin.bias.data = (lin.bias.data - bn.running_mean) * scale + bn.bias

            # Replace BN with a no-op so the graph is clean for quantisation
            setattr(seq, name_bn, nn.Identity())
            print(f"  Folded BN1d '{name_bn}' into Linear '{name_lin}'")

    return model


def _conv_fusion_list():
    """
    Conv+BN+ReLU triples for fuse_modules() — conv_layers only.
    (FC fusions are handled by fold_bn_into_linear + Linear+ReLU pairs below.)

    BaseCNN conv_layers layout (assumed):
        0  Conv2d  1  BN2d  2  ReLU
        3  Conv2d  4  BN2d  5  ReLU
        6  Conv2d  7  BN2d  8  ReLU
        9  Conv2d  10 BN2d  11 ReLU
    """
    return [
        ["model.conv_layers.0",  "model.conv_layers.1",  "model.conv_layers.2"],
        ["model.conv_layers.3",  "model.conv_layers.4",  "model.conv_layers.5"],
        ["model.conv_layers.6",  "model.conv_layers.7",  "model.conv_layers.8"],
        ["model.conv_layers.9",  "model.conv_layers.10", "model.conv_layers.11"],
        # After BN folding, fc_layers looks like:
        #   0 Flatten  1 Linear  2 Identity  3 ReLU  4 Dropout
        #   5 Linear   6 Identity 7 ReLU     8 Linear
        # Identity is transparent to fuse_modules, so Linear+ReLU works fine:
        ["model.fc_layers.1", "model.fc_layers.3"],
        ["model.fc_layers.5", "model.fc_layers.7"],
    ]


def apply_ptq_int8(model: nn.Module, calibration_loader, n_calibration_batches: int = 10) -> nn.Module:
    """
    Static INT8 post-training quantisation via fbgemm.

    Steps:
      1. Fold BN1d into preceding Linear (eliminates QuantizedCPU crash)
      2. Wrap with QuantStub / DeQuantStub
      3. Fuse Conv+BN+ReLU and Linear+ReLU
      4. Insert observers (prepare)
      5. Calibrate on real data
      6. Convert to quantised ops
    """
    # ── 1. Pre-process: absorb BN1d into Linear ───────────────────────────
    print("  Folding BN1d layers into preceding Linear layers …")
    clean_model = fold_bn_into_linear(model)

    ptq_model = QuantWrapper(clean_model)
    ptq_model.eval()

    # ── 2. Fuse ────────────────────────────────────────────────────────────
    fusion_list = _conv_fusion_list()
    try:
        quant.fuse_modules(ptq_model, fusion_list, inplace=True)
        print("  Layer fusion succeeded.")
    except Exception as e:
        print(f"  Layer fusion warning ({e}). Continuing.")

    # ── 3. Quantisation config ────────────────────────────────────────────
    torch.backends.quantized.engine = "fbgemm"
    ptq_model.qconfig = quant.get_default_qconfig("fbgemm")
    quant.prepare(ptq_model, inplace=True)

    # ── 4. Calibrate on real data ─────────────────────────────────────────
    print(f"  Calibrating on {n_calibration_batches} batches …")
    with torch.no_grad():
        for i, (inputs, _) in enumerate(calibration_loader):
            if i >= n_calibration_batches:
                break
            ptq_model(inputs)

    # ── 5. Convert ────────────────────────────────────────────────────────
    quant.convert(ptq_model, inplace=True)
    print("  INT8 conversion complete.")
    return ptq_model


def apply_ptq_kmeans(model: nn.Module, n_clusters: int = 16) -> nn.Module:
    """
    K-Means weight clustering (simulates 4-bit quantisation).
    Cluster centres are stored as FP32; the theoretical size
    is reported separately via get_kmeans_theoretical_size_mb().
    """
    km_model = copy.deepcopy(model)
    km_model.eval()

    with torch.no_grad():
        for _, module in km_model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue

            w      = module.weight.data.numpy()
            shape  = w.shape
            flat   = w.reshape(-1, 1)

            km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
            km.fit(flat)

            module.weight.data = torch.from_numpy(
                km.cluster_centers_[km.labels_].reshape(shape)
            ).float()

    return km_model


# ==============================================================================
# Metrics
# ==============================================================================
def get_model_size_mb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue()) / (1024 ** 2)


def get_kmeans_theoretical_size_mb(model: nn.Module, n_clusters: int = 16) -> float:
    """
    Theoretical size if indices were stored at ceil(log2(n_clusters)) bits
    and cluster centres at FP32.
    """
    bits_per_index = (n_clusters - 1).bit_length()   # 16 clusters → 4 bits
    total_bits = 0
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() > 1:
            total_bits += param.numel() * bits_per_index + n_clusters * 32
        else:
            total_bits += param.numel() * 32
    return total_bits / (8 * 1024 ** 2)


def evaluate_model(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            logits  = model(inputs)
            # QuantWrapper returns FP32 after dequant; plain CNN returns tensor
            logits  = logits.flatten(start_dim=1)
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return 100 * correct / total


def measure_flops(model: nn.Module, dummy_input: torch.Tensor,
                  fallback: str = "N/A") -> str:
    """
    fvcore traces the model and matches op names to known FLOP counters.
    After quant.convert() all ops become  quantized::conv2d / quantized::linear,
    which fvcore has no handlers for → returns 0 or raises.

    Pass fallback= to reuse a previously measured value (e.g. from the FP32
    model) since quantization doesn't change the op count, only the precision.
    """
    try:
        analyzer = FlopCountAnalysis(model, dummy_input)
        analyzer.unsupported_ops_warnings(False)
        total = analyzer.total()
        if total == 0:
            return fallback
        return f"{total / 1e6:.1f}M"
    except Exception:
        return fallback


def measure_latency(model: nn.Module, dummy_input: torch.Tensor,
                    n_warmup: int = 20, n_runs: int = 200) -> tuple[float, float]:
    """
    Measures per-image CPU inference latency in milliseconds.

    Warmup runs are discarded — they prime the CPU branch predictor,
    fill caches, and (for INT8) let fbgemm reach steady-state throughput.
    Then n_runs timed passes are collected and we report:
        mean ± std  (ms / image)

    Why per-image and not per-batch?
    ---------------------------------
    Batch size inflates raw latency numbers and makes FP32 vs INT8
    comparisons misleading because INT8 batching efficiency differs.
    Dividing by batch size gives a hardware-agnostic unit.
    """
    model.eval()
    batch = dummy_input  # shape [1, 1, H, W]  → 1 image

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(batch)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(batch)
            times.append((time.perf_counter() - t0) * 1000)   # → ms

    t = torch.tensor(times)
    return t.mean().item(), t.std().item()


def clear_hooks(model: nn.Module) -> None:
    """Remove all forward/backward hooks left by profiling tools (e.g. fvcore)."""
    for m in model.modules():
        m._forward_hooks.clear()
        m._forward_pre_hooks.clear()
        m._backward_hooks.clear()


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    num_classes = len(classes)
    dummy_input = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)

    # ── Train (or load) ───────────────────────────────────────────────────────
    weight_path = f"{TARGET_ARCH}.pth"
    base_cnn    = BaseCNN(num_classes=num_classes)

    if os.path.exists(weight_path):
        print(f"Found {weight_path} — loading weights, skipping training.")
        base_cnn.load_state_dict(
            torch.load(weight_path, map_location="cpu", weights_only=True)
        )
    else:
        print(f"{weight_path} not found — training from scratch.")
        base_cnn = train_cnn(train_loader, val_loader, test_loader, classes)

    base_cnn.eval()

    # ── Quantise ──────────────────────────────────────────────────────────────
    print("\n--- Post-Training Quantisation ---")

    print("\n[1/2] INT8 static PTQ (fbgemm) …")
    cnn_int8   = apply_ptq_int8(base_cnn, calibration_loader=train_loader)

    print("\n[2/2] K-Means 4-bit clustering …")
    cnn_kmeans = apply_ptq_kmeans(base_cnn, n_clusters=16)

    # ── Evaluate & report ────────────────────────────────────────────────────
    models = {
        "Base CNN (FP32)":    (base_cnn,   False),
        "CNN PTQ INT8":       (cnn_int8,   True),
        "CNN K-Means 4-bit":  (cnn_kmeans, False),
    }

    col = 26
    print("\n" + "=" * 88)
    print(f"{'Model':<{col}} | {'Acc (%)':>8} | {'Size (MB)':>10} | {'FLOPs':>10} | {'Latency (ms/img)':>18}")
    print("-" * 88)

    last_known_flops = "N/A"   # carries forward the most recent valid count
    for name, (model, is_int8) in models.items():
        acc     = evaluate_model(model, test_loader)
        size_mb = (get_kmeans_theoretical_size_mb(model)
                   if "K-Means" in name else get_model_size_mb(model))
        flops   = measure_flops(model, dummy_input, fallback=last_known_flops)
        if flops != "N/A":
            last_known_flops = flops

        lat_mean, lat_std = measure_latency(model, dummy_input)
        lat_str = f"{lat_mean:.2f} ± {lat_std:.2f}"
        print(f"{name:<{col}} | {acc:>8.2f} | {size_mb:>10.3f} | {flops:>10} | {lat_str:>18}")

        clear_hooks(model)   # drop fvcore ScopePopHook before pickling
        # ── Export ────────────────────────────────────────────────────────
        save_path = f"exported_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pth"
        if is_int8:
            # torch.jit.trace is incompatible with quantized models that carry
            # residual observer hooks (e.g. from fvcore FlopCountAnalysis).
            # torch.save(model) serialises the full quantized object — the
            # recommended approach for PTQ INT8 deployment.
            # Reload with: model = torch.load("file.pth"); model.eval()
            torch.save(model, save_path)
            print(f"  Saved quantized model → {save_path}")
        else:
            torch.save(model.state_dict(), save_path)
            print(f"  Saved state_dict      → {save_path}")

    print("=" * 88)