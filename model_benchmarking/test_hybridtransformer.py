import os
import time
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets as tv_datasets, transforms

DATASETS = ["MNIST_DVS", "N_MNIST", "TCASL", "ASL_DVS"]

# --- Keep your exact model architecture ---
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        mlp_output = self.mlp(x)
        x = self.norm2(x + self.dropout(mlp_output))
        return x

class HybridTransformer(nn.Module):
    def __init__(self, num_classes: int, embed_dim=128):
        super().__init__()
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.adaptive_bridge = nn.AdaptiveAvgPool2d((4, 4))
        self.proj = nn.Linear(64, embed_dim)
        self.transformer = TransformerBlock(embed_dim=embed_dim, num_heads=4, ff_dim=256)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_backbone(x)
        x = self.adaptive_bridge(x)
        B, C, H, W = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        x = self.proj(x)
        x = self.transformer(x)
        x = x.mean(dim=0)
        return self.classifier(x)

def main():
    BASE_DATA_DIR = '' # Update to your local Mac path if needed
    ARCH_NAME = "HybridTransformer"
    
    # Force CPU for standard hardware benchmarking (removed GPU flags)
    device = torch.device("cpu")
    print(f"Device: {device} | Architecture: {ARCH_NAME} | Mode: Inference Benchmark\n")

    log_file = f"{ARCH_NAME}_inference_metrics.csv"
    with open(log_file, mode='w', newline='') as file:
        csv.writer(file).writerow(["Dataset", "Avg_Latency_ms", "FPS", "Model_Size_MB"])

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) 
    ])

    for dataset_name in DATASETS:
        DATA_DIR = os.path.join(BASE_DATA_DIR, dataset_name)
        best_model_path = f"{ARCH_NAME}_{dataset_name}_best.pth"

        if not os.path.exists(DATA_DIR) or not os.path.exists(best_model_path):
            print(f"[Skip] Data or .pth file for '{dataset_name}' not found.")
            continue

        full_dataset = tv_datasets.ImageFolder(root=DATA_DIR, transform=transform)
        num_classes = len(full_dataset.classes)
        
        # Grab exactly 10% of the dataset deterministically
        subset_size = int(0.1 * len(full_dataset))
        indices = torch.randperm(len(full_dataset), generator=torch.Generator().manual_seed(42))[:subset_size]
        test_subset = Subset(full_dataset, indices)

        # Batch size 1 is required to measure real-time, per-frame latency
        test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

        # Load Model
        model = HybridTransformer(num_classes).to(device)
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        model.eval()

        # Get file size in MB
        model_size_mb = os.path.getsize(best_model_path) / (1024 * 1024)

        print(f"Benchmarking {dataset_name} ({subset_size} images)...")

        # --- WARM UP ---
        # Run a few dummy passes so the CPU cache and memory allocator spin up
        with torch.no_grad():
            for i, (inputs, _) in enumerate(test_loader):
                if i >= 10: break
                _ = model(inputs.to(device))

        # --- ACTUAL BENCHMARK ---
        latencies = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                
                start_time = time.perf_counter()
                _ = model(inputs)
                end_time = time.perf_counter()
                
                # Convert to milliseconds
                latencies.append((end_time - start_time) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        fps = 1000 / avg_latency  # Since we are doing 1 frame at a time

        print(f"  -> Avg Latency: {avg_latency:.2f} ms/image")
        print(f"  -> Throughput:  {fps:.2f} FPS")
        print(f"  -> Model Size:  {model_size_mb:.2f} MB\n")

        with open(log_file, mode='a', newline='') as file:
            csv.writer(file).writerow([dataset_name, f"{avg_latency:.2f}", f"{fps:.2f}", f"{model_size_mb:.2f}"])

if __name__ == "__main__":
    main()