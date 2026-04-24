import os
import time
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets as tv_datasets, transforms

DATASETS = ["MNIST_DVS", "N_MNIST", "TCASL", "ASL_DVS"]

# --- Keep your exact model architecture ---
class ResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.bn(self.conv(x))
        res = self.shortcut(x)
        return torch.relu(out + res)

class RGCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(ResBlock2D(32, 64), nn.MaxPool2d(kernel_size=2))
        self.layer3 = nn.Sequential(ResBlock2D(64, 128), nn.MaxPool2d(kernel_size=2))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def main():
    BASE_DATA_DIR = '' # Update to your local Mac path if needed
    ARCH_NAME = "RGCNN"
    
    # Force CPU for standard hardware benchmarking
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
        model = RGCNN(num_classes).to(device)
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