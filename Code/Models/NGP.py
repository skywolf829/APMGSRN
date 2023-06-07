import torch
import torch.nn as nn
from math import exp, log
import tinycudann as tcnn

class NGP_TCNN(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        # hash grid metadata
        self.max_resolution = opt['hash_max_resolution']
        self.base_resolution = opt['hash_base_resolution']
        self.n_grids = opt['n_grids']
        self.table_size = 1 << opt['hash_log2_size']
        self.feat_dim = opt['n_features']
        per_level_scale = exp(
            (log(self.max_resolution) - log(self.base_resolution))/(self.n_grids-1)
        )  # growth factor
        
        self.decoder_dim = opt['nodes_per_layer']
        self.decoder_outdim = opt['n_outputs']
        self.decoder_layers = opt['n_layers']
        
        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=opt['n_dims'],
            n_output_dims=self.decoder_outdim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": self.n_grids,
                "n_features_per_level": self.feat_dim,
                "log2_hashmap_size": opt['hash_log2_size'],
                "base_resolution": self.base_resolution,
                "per_level_scale": per_level_scale,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.decoder_dim,
                "n_hidden_layers": self.decoder_layers,
            },
        )
        self.register_buffer(
            "volume_min",
            torch.tensor([self.opt['data_min']], requires_grad=False, dtype=torch.float32),
            persistent=False
        )
        self.register_buffer(
            "volume_max",
            torch.tensor([self.opt['data_max']], requires_grad=False, dtype=torch.float32),
            persistent=False
        )
    
    def min(self):
        return self.volume_min

    def max(self):
        return self.volume_max
    
    def get_volume_extents(self):
        return self.opt['full_shape']
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # HashGrid seems to perform better with input scaled [0,1],
        # as I believe the negative input is clipped to 0
        y = self.model((x+1)/2).float()
        y = y * (self.volume_max - self.volume_min) + self.volume_min
        return y
