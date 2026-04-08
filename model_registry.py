import torch.nn as nn
from torch import Tensor
from typing import Dict, Type

class BaseCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        """
        Initializes the CNN layers.

        :param num_classes: The number of output classes for prediction.
        """
        super(BaseCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass of the network.

        :param x: Input tensor containing image data.
        :return: Output tensor with class logits.
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Add your new architectures here
LOCAL_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "cnn_v1": BaseCNN,
}