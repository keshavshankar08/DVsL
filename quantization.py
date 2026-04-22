import torch
import torch.nn as nn
import torch.ao.quantization as quant
from sklearn.cluster import KMeans
import copy
import os
import tempfile
from model_registry import *
import io

# ==============================================================================
# 1. Linear Post-Training Quantization (PTQ - INT8)
# ==============================================================================
def apply_ptq_linear(model: nn.Module, dummy_input_shape=(1, 1, 128, 128)) -> nn.Module:
    """Applies standard PyTorch INT8 static quantization with Module Fusion."""
    ptq_model = QuantWrapper(copy.deepcopy(model))
    ptq_model.eval()
    
    # Standard engine for x86 CPU
    torch.backends.quantized.engine = 'fbgemm'

    # --- MODULE FUSION ---
    # We must fuse Conv+BN+ReLU or Linear+BN+ReLU triplets.
    # The strings refer to the attribute paths within 'ptq_model'.
    
    fusion_list = [
        # conv_layers fusion (Triplets of Conv, BN, ReLU)
        ['model.conv_layers.0', 'model.conv_layers.1', 'model.conv_layers.2'],
        ['model.conv_layers.3', 'model.conv_layers.4', 'model.conv_layers.5'],
        ['model.conv_layers.6', 'model.conv_layers.7', 'model.conv_layers.8'],
        ['model.conv_layers.9', 'model.conv_layers.10', 'model.conv_layers.11'],
        
        # fc_layers fusion (Triplets of Linear, BN, ReLU)
        ['model.fc_layers.1', 'model.fc_layers.2', 'model.fc_layers.3'],
        ['model.fc_layers.5', 'model.fc_layers.6', 'model.fc_layers.7'],
    ]
    
    # Perform the fusion
    ptq_model = torch.ao.quantization.fuse_modules(ptq_model, fusion_list)
    # ---------------------

    # Set quantization configuration
    ptq_model.qconfig = quant.get_default_qconfig('fbgemm')
    
    # Prepare inserts observers
    quant.prepare(ptq_model, inplace=True)
    
    # Calibration: Pass a few batches through the model so observers can pick up ranges
    with torch.no_grad():
        for _ in range(10):
            dummy_data = torch.randn(8, *dummy_input_shape[1:])
            ptq_model(dummy_data)
            
    # Convert: Turn observers into actual quantized operations
    quant.convert(ptq_model, inplace=True)
    
    return ptq_model

# ==============================================================================
# 2. K-Means Post-Training Quantization (PTQ - 4-bit Simulation)
# ==============================================================================
def apply_ptq_kmeans(model: nn.Module, n_clusters: int = 16) -> nn.Module:
    """
    Clusters the weights of Conv2d and Linear layers into 'n_clusters' using K-Means.
    16 clusters effectively simulates 4-bit weight storage.
    """
    kmeans_model = copy.deepcopy(model)
    kmeans_model.eval()
    
    with torch.no_grad():
        for name, module in kmeans_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Extract and flatten weights
                weights = module.weight.data.cpu().numpy()
                original_shape = weights.shape
                flattened_weights = weights.reshape(-1, 1)
                
                # Fit K-Means
                kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
                kmeans.fit(flattened_weights)
                
                # Map original weights to their nearest calculated cluster center
                cluster_centers = kmeans.cluster_centers_
                labels = kmeans.labels_
                quantized_weights = cluster_centers[labels].reshape(original_shape)
                
                # Overwrite the model weights
                module.weight.data = torch.from_numpy(quantized_weights).float().to(module.weight.device)
                
    return kmeans_model

# ==============================================================================
# Size Evaluation & Comparison Functions
# ==============================================================================
def get_model_size_mb(model: nn.Module) -> float:
    """
    Saves the model to a temporary file and returns its size in MB.
    This works perfectly for standard FP32 models and PyTorch native INT8 models.
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = len(buffer.getvalue()) / (1024 * 1024)
    return size_mb

def get_kmeans_theoretical_size_mb(model: nn.Module, n_clusters: int = 16) -> float:
    """
    Calculates the theoretical compressed size of a K-Means clustered model.
    Assume 16 clusters = 4 bits per weight to store the index.
    """
    bits_per_weight = n_clusters.bit_length() - 1 
    total_bits = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) > 1: 
            num_weights = param.numel()
            total_bits += (num_weights * bits_per_weight) + (n_clusters * 32)
        else: 
            total_bits += param.numel() * 32
            
    size_mb = total_bits / (8 * 1024 * 1024)
    return size_mb

def compare_architectures(base_sdnn: nn.Module, base_cnn: nn.Module, 
                          cnn_ptq_linear: nn.Module, cnn_ptq_kmeans: nn.Module):
    """
    Prints a clean comparison table of model sizes.
    """
    print("-" * 65)
    print(f"{'Model Architecture':<25} | {'Parameters':<15} | {'Size (MB)':<15}")
    print("-" * 65)
    
    sdnn_params = sum(p.numel() for p in base_sdnn.parameters())
    sdnn_size = get_model_size_mb(base_sdnn)
    print(f"{'Base SDNN (FP32)':<25} | {sdnn_params:<15,} | {sdnn_size:<15.3f}")
    
    cnn_params = sum(p.numel() for p in base_cnn.parameters())
    cnn_size = get_model_size_mb(base_cnn)
    print(f"{'Base CNN (FP32)':<25} | {cnn_params:<15,} | {cnn_size:<15.3f}")
    
    cnn_linear_size = get_model_size_mb(cnn_ptq_linear)
    print(f"{'CNN PTQ Linear (INT8)':<25} | {'--':<15} | {cnn_linear_size:<15.3f}")
    
    cnn_kmeans_size = get_kmeans_theoretical_size_mb(cnn_ptq_kmeans, n_clusters=16)
    print(f"{'CNN PTQ K-Means (~4-bit)':<25} | {cnn_params:<15,} | {cnn_kmeans_size:<15.3f}")
    print("-" * 65)

def save_quantized_models(cnn_ptq_linear, cnn_ptq_kmeans):
    # Save K-Means (it's still a standard FP32 model structure, just with clustered values)
    torch.save(cnn_ptq_kmeans.state_dict(), "cnn_ptq_kmeans.pth")
    
    # Save Linear PTQ (INT8) - Using TorchScript is safest for quantized models
    try:
        # We use a dummy input to trace the model for export
        dummy_input = torch.randn(1, 1, 128, 128)
        traced_model = torch.jit.trace(cnn_ptq_linear, dummy_input)
        torch.jit.save(traced_model, "cnn_ptq_linear_int8.pt")
        print("Linear PTQ saved as TorchScript (.pt)")
    except Exception as e:
        print(f"TorchScript export failed, saving state_dict instead: {e}")
        torch.save(cnn_ptq_linear.state_dict(), "cnn_ptq_linear_stat_dict.pth")


from fvcore.nn import FlopCountAnalysis

def get_model_flops(model, dummy_input):
    model.eval()
    flops = FlopCountAnalysis(model, dummy_input)
    return flops.total()

def evaluate_model(model, loader, device, is_sdnn=False, is_int8=False):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            # 1. Handle SDNN input shape
            if is_sdnn:
                inputs = inputs.unsqueeze(-1)
            
            # 2. Move to device (Note: INT8 PTQ models usually run on CPU)
            if not is_int8:
                inputs, labels = inputs.to(device), labels.to(device)
            else:
                inputs, labels = inputs.cpu(), labels.cpu()
                model.cpu()

            outputs = model(inputs)
            
            # 3. Handle SDNN output tuple
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            logits = logits.flatten(start_dim=1)
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100 * correct / total


# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == "__main__":
    num_classes = 26
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, 1, 128, 128)
    
    # 1. Initialize Models
    base_sdnn = BaseSDNN(num_classes=num_classes)
    base_cnn = BaseCNN(num_classes=num_classes)

    # 2. Load Weights (Your existing logic)
    print("Loading pre-trained weights...")
    if os.path.exists("sdnn_v1.pth") and os.path.exists("cnn_v1.pth"):
        sdnn_checkpoint = torch.load("sdnn_v1.pth", map_location='cpu', weights_only=True)
        cnn_checkpoint = torch.load("cnn_v1.pth", map_location='cpu', weights_only=True)

        for name, module in base_sdnn.named_modules():
            for attr in ['running_mean', 'running_var']:
                if hasattr(module, attr) and f"{name}.{attr}" in sdnn_checkpoint:
                    target_shape = sdnn_checkpoint[f"{name}.{attr}"].shape
                    module.register_buffer(attr, torch.zeros(target_shape))
        
        base_sdnn.load_state_dict(sdnn_checkpoint)
        base_cnn.load_state_dict(cnn_checkpoint)
        print("Weights loaded successfully!")
    else:
        print("Warning: Weights not found. Using random init.")

    # 3. Apply Quantization
    print("\nGenerating INT8 Linear PTQ Model...")
    # Note: PTQ Linear usually stays on CPU for inference simulation
    cnn_ptq_linear = apply_ptq_linear(base_cnn, dummy_input_shape=(1, 1, 128, 128))

    print("Generating 4-bit K-Means PTQ Model...")
    cnn_ptq_kmeans = apply_ptq_kmeans(base_cnn, n_clusters=16)

    # 4. Prepare Evaluation Dictionary
    # Mapping: { "Name": (model_instance, is_sdnn, is_int8) }
    models_to_test = {
        "Base SDNN (FP32)": (base_sdnn.to(device), True, False),
        "Base CNN (FP32)": (base_cnn.to(device), False, False),
        "CNN PTQ (INT8)": (cnn_ptq_linear.cpu(), False, True), # INT8 runs on CPU
        "CNN K-Means (4-bit)": (cnn_ptq_kmeans.to(device), False, False),
    }

    # 5. Export, Measure FLOPs, and Evaluate Accuracy
    print("\n" + "="*80)
    print(f"{'Model Architecture':<25} | {'Acc (%)':<10} | {'Size (MB)':<10} | {'FLOPs':<10}")
    print("-" * 80)

    for name, (model, is_sdnn, is_int8) in models_to_test.items():
        # A. Measure Accuracy (using the evaluate_model function from previous response)
        # Assuming test_loader is defined globally or passed in
        accuracy = evaluate_model(model, test_loader, device, is_sdnn, is_int8)

        # B. Measure Size
        if "K-Means" in name:
            size_mb = get_kmeans_theoretical_size_mb(model)
        else:
            size_mb = get_model_size_mb(model)

        # C. Measure FLOPs (Theoretical)
        # We use CPU for FLOP counting to avoid device mismatch issues
        model_cpu = copy.deepcopy(model).cpu()
        flops = FlopCountAnalysis(model_cpu, dummy_input).total()
        
        print(f"{name:<25} | {accuracy:<10.2f} | {size_mb:<10.3f} | {flops/1e6:<9.1f}M")

        # D. Export to disk
        save_path = f"exported_{name.replace(' ', '_').lower()}.pth"
        if is_int8:
            # Use TorchScript for the converted INT8 model
            traced = torch.jit.trace(model, dummy_input)
            torch.jit.save(traced, save_path.replace(".pth", ".pt"))
        else:
            torch.save(model.state_dict(), save_path)

    print("="*80)
    print("Evaluation and Export complete.")