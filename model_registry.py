import torch
import torch.nn as nn
from typing import Dict
import torch.ao.quantization as quant

class QuantWrapper(nn.Module):
    """Wraps a CNN with QuantStub / DeQuantStub for INT8 PTQ."""
    def __init__(self, model):
        super().__init__()
        self.quant   = quant.QuantStub()
        self.model   = model
        self.dequant = quant.DeQuantStub()

    def forward(self, x):
        return self.dequant(self.model(self.quant(x)))


class BaseCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        """
        Initializes the CNN layers.

        :param num_classes: The number of output classes for prediction.
        """
        super(BaseCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 24, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(24), 
            nn.ReLU(),
            
            # Block 2
            nn.Conv2d(24, 36, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            
            # Block 3 (Asymmetrical padding/stride)
            nn.Conv2d(36, 64, kernel_size=3, padding=(1, 0), stride=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Block 4
            nn.Conv2d(64, 64, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            
            # Block 5: Dense 1
            nn.Linear(24192, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Block 6: Dense 2
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Block 7: Output Layer
            nn.Linear(50, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

MODEL_REGISTRY: Dict[str, Dict] = {
    "cnn_v1": {
        "class": BaseCNN,
    },
    "cnn_k-means_4-bit": {
        "class": BaseCNN,
    },
    "cnn_ptq_int8": {
        "class": BaseCNN,
    }
}