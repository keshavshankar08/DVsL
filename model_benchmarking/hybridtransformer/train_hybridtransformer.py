import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets as tv_datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

DATASETS = ["MNIST_DVS", "N_MNIST", "TCASL", "ASL_DVS"]

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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    BASE_DATA_DIR = '/mnt/data1/workspace/kes298/DVsL'
    ARCH_NAME = "HybridTransformer"
    BATCH_SIZE = 64
    MAX_EPOCHS = 100
    PATIENCE = 10
    LEARNING_RATE = 0.0001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Architecture: {ARCH_NAME}")

    log_file = f"{ARCH_NAME}_results.csv"
    with open(log_file, mode='w', newline='') as file:
        csv.writer(file).writerow(["Dataset", "Parameters", "Test_Acc_Top1"])

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    for dataset_name in DATASETS:
        DATA_DIR = os.path.join(BASE_DATA_DIR, dataset_name)
        if not os.path.exists(DATA_DIR):
            print(f"[Skip] '{DATA_DIR}' not found.")
            continue

        full_dataset = tv_datasets.ImageFolder(root=DATA_DIR, transform=transform)
        classes = full_dataset.classes
        num_classes = len(classes)
        
        train_size = int(0.8 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        loader_kwargs = {"batch_size": BATCH_SIZE, "num_workers": 4, "pin_memory": True}
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

        model = HybridTransformer(num_classes).to(device)
        total_params = count_parameters(model)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = f"{ARCH_NAME}_{dataset_name}_best.pth"

        print(f"\nTraining {dataset_name} ({len(full_dataset)} samples, {num_classes} classes)...")

        for epoch in range(MAX_EPOCHS):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            scheduler.step()
            
            train_losses.append(running_loss / len(train_loader))
            train_accs.append(100 * correct / total)

            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_losses.append(val_loss / len(val_loader))
            val_accs.append(100 * correct / total)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:02d}/{MAX_EPOCHS} | Train Acc: {train_accs[-1]:.2f}% | Val Acc: {val_accs[-1]:.2f}%")

            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}.")
                    break

        # --- Testing ---
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        model.eval()
        test_correct, test_total = 0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                        
        top1_test_acc = 100 * test_correct / test_total
        print(f"[{dataset_name} RESULTS] Top-1 Test Acc: {top1_test_acc:.2f}%")

        with open(log_file, mode='a', newline='') as file:
            csv.writer(file).writerow([dataset_name, total_params, f"{top1_test_acc:.2f}"])
        
        # --- Plotting ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Val Loss')
        ax1.set_title(f'Loss - {dataset_name}')
        ax1.legend()
        ax2.plot(train_accs, label='Train Acc')
        ax2.plot(val_accs, label='Val Acc')
        ax2.set_title(f'Accuracy - {dataset_name}')
        ax2.legend()
        plt.savefig(f'{ARCH_NAME}_{dataset_name}_curves.png')
        plt.close()

        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix(all_labels, all_preds), annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f'{ARCH_NAME} - {dataset_name}')
        plt.savefig(f'{ARCH_NAME}_{dataset_name}_cm.png')
        plt.close()

if __name__ == "__main__":
    main()