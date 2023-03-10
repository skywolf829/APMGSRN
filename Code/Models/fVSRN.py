import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.layers import ReLULayer, PositionalEncoding
from typing import List
import math

class fVSRN(nn.Module):
    def __init__(self, n_features: int, 
        feature_grid_shape: List[int], n_dims : int, 
        n_outputs: int, nodes_per_layer: int, n_layers: int, 
        num_positional_encoding_terms, use_tcnn:bool, use_bias:bool,
        requires_padded_feats:bool,data_min:float, data_max:float,
        full_shape:List[int]):
        super().__init__()
        
        
        self.requires_padded_feats : bool = requires_padded_feats
        self.padding_size : int = 0
        self.full_shape = full_shape

        if(requires_padded_feats):
            self.padding_size : int = \
                16*int(math.ceil(max(1, (n_features+num_positional_encoding_terms*n_dims*2)/16))) - \
                    (n_features+num_positional_encoding_terms*n_dims*2)
            
        self.pe = PositionalEncoding(num_positional_encoding_terms, n_dims)        
        feat_shape : List[int] = [1, n_features] + feature_grid_shape
        self.full_shape = full_shape
        self.feature_grid = torch.rand(feat_shape, 
            dtype=torch.float32)
        self.feature_grid = torch.nn.Parameter(self.feature_grid, 
            requires_grad=True)

        def init_decoder_tcnn():
            import tinycudann as tcnn 
            
            input_size:int = n_features+num_positional_encoding_terms*n_dims*2
            if(requires_padded_feats):
                input_size = n_features+num_positional_encoding_terms*n_dims*2 + self.padding_size
                
            decoder = tcnn.Network(
                n_input_dims=input_size,
                n_output_dims=n_outputs,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": nodes_per_layer,
                    "n_hidden_layers": n_layers,
                }
            )
            return decoder
        
        def init_decoder_pytorch():
            decoder = nn.ModuleList()
            
            input_size:int = n_features+num_positional_encoding_terms*n_dims*2
            if(requires_padded_feats):
                input_size = n_features+num_positional_encoding_terms*n_dims*2 + self.padding_size
                                    
            layer = ReLULayer(input_size, 
                nodes_per_layer, bias=use_bias)
            decoder.append(layer)
            
            for i in range(n_layers):
                if i == n_layers - 1:
                    layer = nn.Linear(nodes_per_layer, n_outputs, bias=use_bias)
                    decoder.append(layer)
                else:
                    layer = ReLULayer(nodes_per_layer, nodes_per_layer, bias=use_bias)
                    decoder.append(layer)
            decoder = torch.nn.Sequential(*decoder)
            return decoder

        if(use_tcnn):
            try:
                self.decoder = init_decoder_tcnn()
            except ImportError:
                print(f"Tried to use TinyCUDANN but found it was not installed - reverting to PyTorch layers.")
                self.decoder = init_decoder_pytorch()
        else:
            self.decoder = init_decoder_pytorch()

        self.register_buffer(
            "volume_min",
            torch.tensor([data_min], requires_grad=False, dtype=torch.float32),
            persistent=False
        )
        self.register_buffer(
            "volume_max",
            torch.tensor([data_max], requires_grad=False, dtype=torch.float32),
            persistent=False
        )
    
    def min(self):
        return self.volume_min

    def max(self):
        return self.volume_max
    
    def get_volume_extents(self):
        return self.full_shape
                       
    def forward(self, x):     
        
        feats = F.grid_sample(self.feature_grid,
                x.reshape(([1]*x.shape[-1]) + list(x.shape)),
                mode='bilinear', align_corners=True) 
        pe = self.pe(x)  
        
        feats = feats.flatten(0, -2).permute(1, 0)
        feats = torch.cat([pe, feats], dim=1)
        if(self.requires_padded_feats):
            feats = F.pad(feats, (0, self.padding_size), value=1.0) 
        y = self.decoder(feats).float()
        y = y * (self.volume_max - self.volume_min) + self.volume_min
        return y

        