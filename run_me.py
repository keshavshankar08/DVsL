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
import torch.nn.utils.prune as prune

from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

from model_registry import MODEL_REGISTRY, BaseCNN, QuantWrapper

# ==============================================================================
# Global Configuration
# ==============================================================================
TARGET_ARCH = "cnn_v1"
DATA_DIR    = "data"
IMG_SIZE    = 128
EPOCHS      = 15
DEVICE      = torch.device("cpu") 

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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False) # BS=1 for latency
    return train_loader, val_loader, test_loader, classes

# ==============================================================================
# Training & Evaluation
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

    print(f"\n--- Training {TARGET_ARCH} ---")
    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            logits = model(inputs).flatten(start_dim=1)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        print(f"Epoch {epoch+1}/{EPOCHS} complete.")

    torch.save(model.state_dict(), model_path)
    return model

def evaluate_model_with_latency(model: nn.Module, loader: DataLoader):
    """
    Measures accuracy and end-to-end latency per image on the provided loader.
    """
    model.eval()
    correct, total = 0, 0
    latencies = []

    with torch.no_grad():
        for inputs, labels in loader:
            # End-to-end timing: input to model -> result
            start_time = time.perf_counter_ns()
            logits = model(inputs).flatten(start_dim=1)
            end_time = time.perf_counter_ns()
            
            # Record latency in milliseconds
            latencies.append((end_time - start_time))
            
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)

    acc = 100 * correct / total
    lat_tensor = torch.tensor(latencies).double()
    mean_ms = (lat_tensor.mean().item()) / 1e6
    return acc, mean_ms
# ==============================================================================
# Quantization Logic
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
            if isinstance(lin, nn.Linear) and isinstance(bn, nn.BatchNorm1d):
                bn.eval()
                with torch.no_grad():
                    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
                    lin.weight.data = lin.weight.data * scale.unsqueeze(1)
                    if lin.bias is None:
                        lin.bias = nn.Parameter(torch.zeros(lin.out_features))
                    lin.bias.data = (lin.bias.data - bn.running_mean) * scale + bn.bias
                setattr(seq, name_bn, nn.Identity())
    return model

def apply_ptq_int8(model: nn.Module, calibration_loader) -> nn.Module:
    clean_model = fold_bn_into_linear(model)
    ptq_model = QuantWrapper(clean_model)
    ptq_model.eval()
    
    # Simple Fusion (adjust indices based on your model_registry definition)
    # Note: In a production script, you'd automate this list generation
    torch.backends.quantized.engine = "fbgemm"
    ptq_model.qconfig = quant.get_default_qconfig("fbgemm")
    quant.prepare(ptq_model, inplace=True)

    with torch.no_grad():
        for i, (inputs, _) in enumerate(calibration_loader):
            if i >= 10: break
            ptq_model(inputs)

    quant.convert(ptq_model, inplace=True)
    return ptq_model

def apply_ptq_kmeans(model: nn.Module, n_clusters: int = 16) -> nn.Module:
    km_model = copy.deepcopy(model)
    km_model.eval()
    with torch.no_grad():
        for module in km_model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                w = module.weight.data.numpy()
                shape = w.shape
                km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
                km.fit(w.reshape(-1, 1))
                module.weight.data = torch.from_numpy(
                    km.cluster_centers_[km.labels_].reshape(shape)
                ).float()
    return km_model

def apply_pruning(model: nn.Module, amount: float = 0.4) -> nn.Module:
    """
    Applies global L1 unstructured pruning to the model.
    'amount' is the fraction of total connections to prune (0.4 = 40%).
    """
    sparse_model = copy.deepcopy(model)
    
    # Identify which modules to prune (Conv and Linear layers)
    parameters_to_prune = []
    for module in sparse_model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))

    # Apply global pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # Make the pruning permanent (removes the pruning hooks/masks and 
    # replaces weight with zeroed weight)
    for module, name in parameters_to_prune:
        prune.remove(module, name)

    return sparse_model

# ==============================================================================
# Model Size
# ==============================================================================
def get_model_size_mb(model: nn.Module, mode: str = "none", n_clusters=16) -> float:
    if mode == "none":
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        return len(buf.getvalue()) / (1024 ** 2)
    
    elif mode == "kmeans":
        bits_per_index = (n_clusters - 1).bit_length()
        total_bits = 0
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() > 1:
                total_bits += param.numel() * bits_per_index + n_clusters * 32
            else:
                total_bits += param.numel() * 32
        return total_bits / (8 * 1024 ** 2)

    elif mode == "pruned":
        total_bits = 0
        for name, param in model.named_parameters():
            if "weight" in name:
                non_zero = torch.count_nonzero(param).item()
                total = param.numel()
                # Theoretical: 32 bits per non-zero weight + 1 bit mask per total weight
                total_bits += (non_zero * 32) + total 
            else:
                total_bits += param.numel() * 32
        return total_bits / (8 * 1024 ** 2)
    
def count_sparse_macs(model: nn.Module, input_size=(1, 1, 128, 128)):
    """
    Estimates MACs, accounting for zeroed-out (pruned) weights.
    """
    total_macs = 0
    
    # We need to know the spatial size of the feature maps
    # We'll do a dummy forward pass to track shapes
    input_data = torch.randn(input_size)
    
    def hook_fn(module, input, output):
        nonlocal total_macs
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Get the number of non-zero elements in the weight
            non_zero_params = torch.count_nonzero(module.weight).item()
            
            if isinstance(module, nn.Conv2d):
                # MACs = non-zero weights * number of output pixels
                output_pixels = output.shape[2] * output.shape[3]
                total_macs += non_zero_params * output_pixels
            else:
                # For Linear layers: non-zero weights * batch size
                total_macs += non_zero_params * input[0].shape[0]

    # Register hooks to catch every layer
    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Trigger the hooks
    model(input_data)
    
    # Clean up
    for h in hooks:
        h.remove()
        
    return total_macs

# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    weight_path = f"{TARGET_ARCH}.pth"
    
    base_cnn = BaseCNN(num_classes=len(classes))
    if os.path.exists(weight_path):
        base_cnn.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
    else:
        base_cnn = train_cnn(train_loader, val_loader, test_loader, classes)

    # --- Quantization & Pruning ---
    cnn_int8   = apply_ptq_int8(base_cnn, calibration_loader=train_loader)
    cnn_kmeans = apply_ptq_kmeans(base_cnn, n_clusters=16)
    cnn_pruned_25 = apply_pruning(base_cnn, amount=0.25)
    cnn_pruned_50 = apply_pruning(base_cnn, amount=0.50)
    cnn_pruned_75 = apply_pruning(base_cnn, amount=0.75)

    models = {
        "Base CNN (FP32)":    (base_cnn,   "none"),
        "CNN PTQ INT8":       (cnn_int8,   "none"),
        "CNN K-Means 4-bit":  (cnn_kmeans, "kmeans"),
        "CNN Pruned (25%)":   (cnn_pruned_25, "pruned"),
        "CNN Pruned (50%)":   (cnn_pruned_50, "pruned"),
        "CNN Pruned (75%)":   (cnn_pruned_75, "pruned"),
    }

    print("\n" + "=" * 85)
    print(f"{'Model':<25} | {'Acc (%)':>8} | {'Size (MB)':>10} | {'Latency (ms/img)':>18} | {'MACs (M)':>10}")
    print("-" * 75)

    for name, (model, size_mode) in models.items():
        acc, l_mean,  = evaluate_model_with_latency(model, test_loader)
        size = get_model_size_mb(model, mode=size_mode)
        macs = count_sparse_macs(model) / 1e6
        
        print(f"{name:<25} | {acc:>8.2f} | {size:>10.3f} | {l_mean:>18.3} | {macs:>10.2f}")
    
    print("=" * 85)