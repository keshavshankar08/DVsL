import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as fnc
from torch.utils.data import DataLoader, random_split
from torchvision import datasets as tv_datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import lava.lib.dl.slayer as slayer

DATASETS = ["MNIST_DVS", "N_MNIST", "TCASL", "ASL_DVS"]

class SpikingAdaptivePool(nn.Module):
    def __init__(self, target_size=(6, 6)):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(target_size)

    def forward(self, x):
        B, C, H, W, T = x.shape
        x = x.permute(0, 4, 1, 2, 3).reshape(B * T, C, H, W)
        x = self.pool(x)
        H_out, W_out = self.pool.output_size
        return x.reshape(B, T, C, H_out, W_out).permute(0, 2, 3, 4, 1)

class BaseSDNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        sdnn_params = {
            'threshold': 0.1, 'tau_grad': 0.5, 'scale_grad': 1,
            'requires_grad': True, 'shared_param': True, 'activation': fnc.relu,
        }
        sdnn_cnn_params = {**sdnn_params, 'norm': slayer.neuron.norm.MeanOnlyBatchNorm}
        sdnn_dense_params = {**sdnn_cnn_params, 'dropout': slayer.neuron.Dropout(p=0.2)}
        
        self.blocks = torch.nn.ModuleList([
            slayer.block.sigma_delta.Input(sdnn_params), 
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 1, 24, 3, padding=0, stride=2, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 24, 36, 3, padding=0, stride=2, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 36, 64, 3, padding=(1, 0), stride=(2, 1), weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 64, 64, 3, padding=0, stride=1, weight_scale=2, weight_norm=True),
            SpikingAdaptivePool(target_size=(6, 6)),
            slayer.block.sigma_delta.Flatten(),
            slayer.block.sigma_delta.Dense(sdnn_dense_params, 2304, 100, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Dense(sdnn_dense_params, 100, 50, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Dense(sdnn_dense_params, 50, num_classes, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Output(sdnn_dense_params, num_classes, num_classes, weight_scale=2, weight_norm=True)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks: 
            x = block(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    BASE_DATA_DIR = '/mnt/data1/workspace/kes298/DVsL'    
    ARCH_NAME = "SDNN"
    BATCH_SIZE = 256
    MAX_EPOCHS = 100               
    PATIENCE = 10    
    LEARNING_RATE = 0.0001
    LAM = 0.01

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Architecture: {ARCH_NAME}")

    log_file = f"{ARCH_NAME}_results.csv"
    with open(log_file, mode='w', newline='') as file:
        csv.writer(file).writerow(["Dataset", "Parameters", "Test_Acc_Top1"])

    transform = transforms.Compose([
        transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST),
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

        loader_kwargs = {"batch_size": BATCH_SIZE, "num_workers": 8, "pin_memory": True, "prefetch_factor": 2}
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

        model = BaseSDNN(num_classes).to(device)
        total_params = count_parameters(model)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.RAdam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = f"{ARCH_NAME}_{dataset_name}_best.pth"

        print(f"\nTraining {dataset_name} ({len(full_dataset)} samples, {num_classes} classes)...")

        for epoch in range(MAX_EPOCHS):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            
            for inputs, labels in train_loader:
                inputs = inputs.unsqueeze(-1).to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)

                if isinstance(outputs, tuple):
                    logits, event_cost, _ = outputs
                    logits = logits.flatten(start_dim=1) 
                    loss = criterion(logits, labels) + LAM * event_cost
                else:
                    logits = outputs.flatten(start_dim=1)
                    loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            train_losses.append(running_loss / len(train_loader))
            train_accs.append(100 * correct / total)

            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.unsqueeze(-1).to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    logits = logits.flatten(start_dim=1)
                    
                    val_loss += criterion(logits, labels).item()
                    _, predicted = torch.max(logits, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_losses.append(val_loss / len(val_loader))
            val_accs.append(100 * val_correct / val_total)
            
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
                inputs = inputs.unsqueeze(-1).to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                logits = logits.flatten(start_dim=1)
                
                _, predicted = torch.max(logits, 1)
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
        ax1.plot(train_losses, label='Train')
        ax1.plot(val_losses, label='Val')
        ax1.set_title(f'Loss ({ARCH_NAME} on {dataset_name})')
        ax1.legend()
        ax2.plot(train_accs, label='Train')
        ax2.plot(val_accs, label='Val')
        ax2.set_title(f'Accuracy ({ARCH_NAME} on {dataset_name})')
        ax2.legend()
        plt.savefig(f'{ARCH_NAME}_{dataset_name}_curves.png')
        plt.close()

        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix(all_labels, all_preds), annot=True, fmt='d', cmap='Greens', xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix: {ARCH_NAME} on {dataset_name}')
        plt.tight_layout()
        plt.savefig(f'{ARCH_NAME}_{dataset_name}_cm.png')
        plt.close()

if __name__ == "__main__":
    main()