# Note to self: Use 2168PROJECT
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics

# read in data #

# set up tensor transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.ImageFolder(root='data/', transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

class GestureCNN(nn.Module):
    def __init__(self, num_classes):
        super(GestureCNN, self).__init__()
        # Conv 1: Input (3, 128, 128) -> Output (32, 128, 128)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv 2: Input (32, 64, 64) -> Output (64, 64, 64)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Second Pool: Output (64, 32, 32)
        
        # Conv 3: Adding a 3rd layer is often better for 128x128 resolution
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # Third Pool: Output (128, 16, 16)

        # Calculation: 128 filters * 16 * 16 = 32,768
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 128 -> 64
        x = self.pool(F.relu(self.conv2(x))) # 64 -> 32
        x = self.pool(F.relu(self.conv3(x))) # 32 -> 16
        
        x = x.view(-1, 128 * 16 * 16) 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize model (replace '10' with your actual number of gesture classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GestureCNN(num_classes=len(dataset.classes)).to(device)
train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=len(dataset.classes)).to(device)

import torch.optim as optim

# Hyperparameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10

def train_model():
    model.train()
    for epoch in range(epochs):
        train_acc.reset()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + Backward + Optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_acc.update(outputs, labels)
            

        total_acc = train_acc.compute()
        print(f"Epoch {epoch+1} Accuracy: {total_acc:.2%}")
    print("Done training")

# Run the training
train_model()

torch.save(model.state_dict(), "ASL_Model")