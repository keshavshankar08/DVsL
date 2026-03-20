import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# 1. Re-define the Architecture (Must match your training script exactly)
class GestureCNN(nn.Module):
    def __init__(self, num_classes):
        super(GestureCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 2. Setup Device and Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replace 27 with the exact number of classes you had during training
# (a-z = 26, plus 'none' = 27)
NUM_CLASSES = 27 
model = GestureCNN(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load("ASL_Model", map_location=device))
model.eval()

# 3. Define the labels mapping
# ImageFolder usually maps folders alphabetically: 0='a', 1='b', etc.
# We'll create a list to turn the predicted index back into a letter.
class_names = [chr(i) for i in range(ord('a'), ord('z') + 1)] + ['none']

# 4. Transform for the incoming .png files
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 5. Loop through the test_data folder
test_folder = 'test_data/'
files = sorted([f for f in os.listdir(test_folder) if f.endswith('.png')])

print(f"{'File Name':<12} | {'Predicted Label':<15} | {'Confidence'}")
print("-" * 45)

with torch.no_grad():
    for filename in files:
        img_path = os.path.join(test_folder, filename)
        
        # Load and transform the image
        img = Image.open(img_path).convert('L') # Convert to Grayscale
        img_tensor = test_transform(img).unsqueeze(0).to(device) # Add batch dimension
        
        # Inference
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        conf, predicted = torch.max(probabilities, 1)
        
        pred_label = class_names[predicted.item()]
        confidence_percent = conf.item() * 100
        
        print(f"{filename:<12} | {pred_label:<15} | {confidence_percent:>8.2f}%")