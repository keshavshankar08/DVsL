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
    """
    Applies standard PyTorch INT8 static quantization.
    Requires a calibration step with representative data to calculate activation scales.
    """
    ptq_model = copy.deepcopy(model)
    ptq_model.eval()
    
    # Set the quantization engine (qnnpack is standard for edge/mobile CPU simulation)
    torch.backends.quantized.engine = 'fbgemm'
    ptq_model.qconfig = quant.get_default_qconfig('fbgemm')
    
    # Insert observers to calibrate activations
    quant.prepare(ptq_model, inplace=True)
    
    # CALIBRATION: Pass representative data through the model.
    with torch.no_grad():
        for _ in range(10): # 10 calibration batches is usually enough
            dummy_data = torch.randn(8, *dummy_input_shape[1:]) # Batch size 8
            ptq_model(dummy_data)
            
    # Convert observed model to quantized INT8 model
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

# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == "__main__":
    num_classes = 26
    base_sdnn = BaseSDNN(num_classes=num_classes)
    base_cnn = BaseCNN(num_classes=num_classes)

    print("Loading pre-trained weights...")
    if os.path.exists("sdnn_v1.pth") and os.path.exists("cnn_v1.pth"):
        # Load checkpoints to CPU memory first
        sdnn_checkpoint = torch.load("sdnn_v1.pth", map_location='cpu', weights_only=True)
        cnn_checkpoint = torch.load("cnn_v1.pth", map_location='cpu', weights_only=True)

        # --- Dynamic SLAYER Buffer Patch ---
        for name, module in base_sdnn.named_modules():
            for attr in ['running_mean', 'running_var']:
                if hasattr(module, attr) and f"{name}.{attr}" in sdnn_checkpoint:
                    target_shape = sdnn_checkpoint[f"{name}.{attr}"].shape
                    module.register_buffer(attr, torch.zeros(target_shape))
        
        # Load the patched states
        try:
            base_sdnn.load_state_dict(sdnn_checkpoint, strict=True)
            base_cnn.load_state_dict(cnn_checkpoint, strict=True)
            base_sdnn.eval()
            base_cnn.eval()
            print("Weights loaded successfully!")
        except RuntimeError as e:
            print(f"Failed to load weights: {e}")
    else:
        print("Warning: .pth files not found in the current directory. Using randomly initialized weights.")

    # Generate Quantized CNNs
    print("\nGenerating INT8 Linear PTQ Model...")
    cnn_ptq_linear = apply_ptq_linear(base_cnn, dummy_input_shape=(1, 1, 128, 128))

    print("Generating 4-bit K-Means PTQ Model...\n")
    cnn_ptq_kmeans = apply_ptq_kmeans(base_cnn, n_clusters=16)

    # Run size comparison script
    compare_architectures(base_sdnn, base_cnn, cnn_ptq_linear, cnn_ptq_kmeans)