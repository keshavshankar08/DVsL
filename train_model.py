import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from model_registry import LOCAL_MODEL_REGISTRY

TARGET_ARCH = "cnn_v1"

def main() -> None:
    """Executes the complete model training, evaluation, and saving pipeline."""
    DATA_DIR = 'data'
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 0.001
    IMG_SIZE = 128
    MODEL_PATH = f"{TARGET_ARCH}.pth"

    if TARGET_ARCH not in LOCAL_MODEL_REGISTRY:
        raise ValueError(f"Architecture '{TARGET_ARCH}' not found in model_registry.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Training architecture: {TARGET_ARCH}")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    classes = full_dataset.classes
    num_classes = len(classes)
    print(f"Found {len(full_dataset)} samples across {num_classes} classes.")

    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    ModelClass = LOCAL_MODEL_REGISTRY[TARGET_ARCH]
    model = ModelClass(num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print("Starting training...")
    for epoch in range(EPOCHS):
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
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_losses.append(running_loss / len(train_loader))
        train_accs.append(100 * correct / total)
        
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(100 * correct / total)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_losses[-1]:.4f}, Acc: {train_accs[-1]:.2f}% | Val Loss: {val_losses[-1]:.4f}, Acc: {val_accs[-1]:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Loss over Epochs')
    ax1.legend()
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_title('Accuracy over Epochs')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f'{TARGET_ARCH}_training_curves.png')

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Test Set Confusion Matrix ({TARGET_ARCH})')
    plt.tight_layout()
    plt.savefig(f'{TARGET_ARCH}_confusion_matrix.png')

if __name__ == "__main__":
    main()