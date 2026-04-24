import os
import time
import io
import copy
import torch
import torch.nn as nn
import torch.ao.quantization as quant
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model_registry import MODEL_REGISTRY, BaseCNN, QuantWrapper

# ==============================================================================
# Global Configuration
# ==============================================================================
TARGET_ARCH = "cnn_v1"
DATA_DIR    = "data"
IMG_SIZE    = 128
DEVICE      = torch.device("cpu")   # PTQ INT8 requires CPU / fbgemm

# ==============================================================================
# Data (Kept for PTQ Calibration & Evaluation)
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

    # train_loader is used to calibrate the INT8 model
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader, classes


# ==============================================================================
# Quantization
# ==============================================================================
def fold_bn_into_linear(model: nn.Module) -> nn.Module:
    model = copy.deepcopy(model)

    for seq in model.modules():
        if not isinstance(seq, nn.Sequential):
            continue

        children = list(seq.named_children())
        for idx in range(len(children) - 1):
            name_lin, lin = children[idx]
            name_bn,  bn  = children[idx + 1]

            if not (isinstance(lin, nn.Linear) and isinstance(bn, nn.BatchNorm1d)):
                continue

            bn.eval()

            with torch.no_grad():
                scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
                lin.weight.data = lin.weight.data * scale.unsqueeze(1)

                if lin.bias is None:
                    lin.bias = nn.Parameter(torch.zeros(lin.out_features))
                lin.bias.data = (lin.bias.data - bn.running_mean) * scale + bn.bias

            setattr(seq, name_bn, nn.Identity())
            print(f"  Folded BN1d '{name_bn}' into Linear '{name_lin}'")

    return model


def _conv_fusion_list():
    return [
        ["model.conv_layers.0",  "model.conv_layers.1",  "model.conv_layers.2"],
        ["model.conv_layers.3",  "model.conv_layers.4",  "model.conv_layers.5"],
        ["model.conv_layers.6",  "model.conv_layers.7",  "model.conv_layers.8"],
        ["model.conv_layers.9",  "model.conv_layers.10", "model.conv_layers.11"],
        ["model.fc_layers.1", "model.fc_layers.3"],
        ["model.fc_layers.5", "model.fc_layers.7"],
    ]


def apply_ptq_int8(model: nn.Module, calibration_loader, n_calibration_batches: int = 10) -> nn.Module:
    print("  Folding BN1d layers into preceding Linear layers …")
    clean_model = fold_bn_into_linear(model)

    ptq_model = QuantWrapper(clean_model)
    ptq_model.eval()

    fusion_list = _conv_fusion_list()
    try:
        quant.fuse_modules(ptq_model, fusion_list, inplace=True)
        print("  Layer fusion succeeded.")
    except Exception as e:
        print(f"  Layer fusion warning ({e}). Continuing.")

    torch.backends.quantized.engine = "fbgemm"
    ptq_model.qconfig = quant.get_default_qconfig("fbgemm")
    quant.prepare(ptq_model, inplace=True)

    print(f"  Calibrating on {n_calibration_batches} batches …")
    with torch.no_grad():
        for i, (inputs, _) in enumerate(calibration_loader):
            if i >= n_calibration_batches:
                break
            ptq_model(inputs)

    quant.convert(ptq_model, inplace=True)
    print("  INT8 conversion complete.")
    return ptq_model


# ==============================================================================
# Metrics
# ==============================================================================
def get_model_size_mb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue()) / (1024 ** 2)


def evaluate_model(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            logits  = model(inputs)
            logits  = logits.flatten(start_dim=1)
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return 100 * correct / total


def measure_latency(model: nn.Module, dummy_input: torch.Tensor,
                    n_warmup: int = 20, n_runs: int = 200) -> tuple[float, float]:
    model.eval()
    batch = dummy_input

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(batch)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(batch)
            times.append((time.perf_counter() - t0) * 1000)

    t = torch.tensor(times)
    return t.mean().item(), t.std().item()