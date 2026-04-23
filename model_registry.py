import torch.nn as nn
from typing import Dict
import torch
import torch.nn.functional as fnc
import lava.lib.dl.slayer as slayer

import torch.ao.quantization as quant

# Add the class definition
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
            # Block 1: Matches SDNN Conv1
            nn.Conv2d(1, 24, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(24), # Mirrors Slayer's MeanOnlyBatchNorm
            nn.ReLU(),
            
            # Block 2: Matches SDNN Conv2
            nn.Conv2d(24, 36, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            
            # Block 3: Matches SDNN Conv3 (Asymmetrical padding/stride)
            nn.Conv2d(36, 64, kernel_size=3, padding=(1, 0), stride=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Block 4: Matches SDNN Conv4
            nn.Conv2d(64, 64, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            
            # Block 5: Matches SDNN Dense 1
            nn.Linear(24192, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.2), # Matched to sdnn_dense_params
            
            # Block 6: Matches SDNN Dense 2
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

class BaseSDNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(BaseSDNN, self).__init__()
        
        sdnn_params = {
            'threshold'     : 0.1,
            'tau_grad'      : 0.5,
            'scale_grad'    : 1,
            'requires_grad' : True,
            'shared_param'  : True,
            'activation'    : fnc.relu,
        }
        sdnn_cnn_params = {
                **sdnn_params,
                'norm' : slayer.neuron.norm.MeanOnlyBatchNorm,
        }
        sdnn_dense_params = {
                **sdnn_cnn_params,
                'dropout' : slayer.neuron.Dropout(p=0.2),
        }
        
        self.blocks = torch.nn.ModuleList([
            slayer.block.sigma_delta.Input(sdnn_params), 
            slayer.block.sigma_delta.Conv(sdnn_cnn_params,  1, 24, 3, padding=0, stride=2, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 24, 36, 3, padding=0, stride=2, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 36, 64, 3, padding=(1, 0), stride=(2, 1), weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 64, 64, 3, padding=0, stride=1, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Flatten(),
            slayer.block.sigma_delta.Dense(sdnn_dense_params, 24192, 100, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Dense(sdnn_dense_params,   100,  50, weight_scale=2, weight_norm=True),
            
            slayer.block.sigma_delta.Dense(sdnn_dense_params,    50, num_classes, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Output(sdnn_dense_params, num_classes, num_classes, weight_scale=2, weight_norm=True)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks: 
            x = block(x)
        return x

MODEL_REGISTRY: Dict[str, Dict] = {
    "sdnn_v1": {
        "class": BaseSDNN,
        "url": "https://github.com/keshavshankar08/TCASLCore/releases/download/v1.0.0/sdnn_v1.pth"
        },
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