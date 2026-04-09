import torch.nn as nn
from torch import Tensor
from typing import Dict, Type
import torch
import torch.nn.functional as F

# New libraries @keshavshankar08
import lava.lib.dl.slayer as slayer
# h5py



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
    
# TODO
class SDNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(SDNN, self).__init__()
    
        sdnn_params = { # sigma-delta neuron parameters
            'threshold'     : 0.1,    # delta unit threshold
            'tau_grad'      : 0.5,    # delta unit surrogate gradient relaxation parameter
            'scale_grad'    : 1,      # delta unit surrogate gradient scale parameter
            'requires_grad' : True,   # trainable threshold
            'shared_param'  : True,   # layer wise threshold
            'activation'    : F.relu, # activation function
        }
        sdnn_cnn_params = { # conv layer has additional mean only batch norm
                **sdnn_params,                                 # copy all sdnn_params
                'norm' : slayer.neuron.norm.MeanOnlyBatchNorm, # mean only quantized batch normalizaton
            }
        sdnn_dense_params = { # dense layers have additional dropout units enabled
                **sdnn_cnn_params,                        # copy all sdnn_cnn_params
                'dropout' : slayer.neuron.Dropout(p=0.2), # neuron dropout
            }
        
        self.blocks = torch.nn.ModuleList([# sequential network blocks 
                # delta encoding of the input
                slayer.block.sigma_delta.Input(sdnn_params), 
                # convolution layers
                slayer.block.sigma_delta.Conv(sdnn_cnn_params,  1, 24, 3, padding=0, stride=2, weight_scale=2, weight_norm=True),
                slayer.block.sigma_delta.Conv(sdnn_cnn_params, 24, 36, 3, padding=0, stride=2, weight_scale=2, weight_norm=True),
                slayer.block.sigma_delta.Conv(sdnn_cnn_params, 36, 64, 3, padding=(1, 0), stride=(2, 1), weight_scale=2, weight_norm=True),
                slayer.block.sigma_delta.Conv(sdnn_cnn_params, 64, 64, 3, padding=0, stride=1, weight_scale=2, weight_norm=True),
                # flatten layer
                slayer.block.sigma_delta.Flatten(),
                # dense layers
                slayer.block.sigma_delta.Dense(sdnn_dense_params, 24192, 100, weight_scale=2, weight_norm=True),
                slayer.block.sigma_delta.Dense(sdnn_dense_params,   100,  50, weight_scale=2, weight_norm=True),
                slayer.block.sigma_delta.Dense(sdnn_dense_params,    50,  27, weight_scale=2, weight_norm=True),
                # linear readout with sigma decoding of output
                slayer.block.sigma_delta.Output(sdnn_dense_params,   27,   27, weight_scale=2, weight_norm=True)
            ])
        
    
        # Event sparsity loss: penalizes network for high-rate events
    def event_rate_loss(self, x, max_rate=0.01):
        mean_event_rate = torch.mean(torch.abs(x))
        return F.mse_loss(F.relu(mean_event_rate - max_rate), torch.zeros_like(mean_event_rate))
        
    def forward(self, x):
        count = []
        event_cost = 0
    
        for block in self.blocks: 
            x = block(x)
            if hasattr(block, 'neuron'):
                event_cost += self.event_rate_loss(x)
                # Count non-zero spikes in this layer
                layer_spikes = torch.sum((x[..., 1:] != 0).to(torch.float32)).item()
                count.append(layer_spikes)
    
        return x, event_cost, torch.FloatTensor(count).reshape((1, -1)).to(x.device)



    

# Add your new architectures here
LOCAL_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "cnn_v1": BaseCNN,
    "sdnn_v1": SDNN
}