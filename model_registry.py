import torch.nn as nn
from typing import Dict
import torch
import torch.nn.functional as fnc
import lava.lib.dl.slayer as slayer

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
    "sdnn_v2": {
        "class": BaseSDNN,
        "url": "https://github.com/keshavshankar08/TCASLCore/releases/download/v1.0.1/sdnn_v2.pth" 
    }
}