import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.layers import ReLULayer, PositionalEncoding
from typing import List
import math



class fVSRN_NGP(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        
        self.requires_padded_feats : bool = opt['requires_padded_feats']
        self.padding_size : int = 0
        self.full_shape = opt['full_shape']
            
        res = 1
        res_grid = [eval(i) for i in opt['feature_grid_shape'].split(',')]
        for i in range(len(res_grid)):
            res *= res_grid[i]

        import tinycudann as tcnn
        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=opt['n_dims']*2,
            n_output_dims=opt['n_outputs'],
            encoding_config={
                "otype": "Composite",
                "nested": [
                    {
                        "otype": "Grid",
                        "type": "Dense",
                        "n_levels": 1,
                        "n_features_per_level": int(opt['n_features']),
                        "base_resolution": int(res**(1.0/len(res_grid)))+1,
                        "interpolation": "Linear",
                        "n_dims_to_encode": opt['n_dims']
                    },
                    {
                        "n_frequencies": int(opt['num_positional_encoding_terms']), 
                        "otype": "Frequency",
                        "n_dims_to_encode": opt['n_dims']
                    }
                ]                
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": opt['nodes_per_layer'],
                "n_hidden_layers": opt['n_layers'],
            },
        )
        
        self.register_buffer(
            "volume_min",
            torch.tensor([opt['data_min']], requires_grad=False, dtype=torch.float32),
            persistent=False
        )
        self.register_buffer(
            "volume_max",
            torch.tensor([opt['data_max']], requires_grad=False, dtype=torch.float32),
            persistent=False
        )
    
    def min(self):
        return self.volume_min

    def max(self):
        return self.volume_max
    
    def get_volume_extents(self):
        return self.full_shape
                       
    def forward(self, x):     
        
        y = self.model((x.repeat(1, 2)+1)/2).float()
        y = y * (self.volume_max - self.volume_min) + self.volume_min
        return y

        

class fVSRN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        
        self.requires_padded_feats : bool = opt['requires_padded_feats']
        self.padding_size : int = 0
        self.full_shape = opt['full_shape']
        feature_grid_shape = [eval(i) for i in opt['feature_grid_shape'].split(",")]


        if(opt['requires_padded_feats']):
            self.padding_size : int = \
                16*int(math.ceil(max(1, (opt['n_features'] +opt['num_positional_encoding_terms']*opt['n_dims']*2)/16))) - \
                    (opt['n_features'] +opt['num_positional_encoding_terms']*opt['n_dims']*2)
            
        self.pe = PositionalEncoding(opt['num_positional_encoding_terms'], opt['n_dims'])        
        feat_shape : List[int] = [1, opt['n_features'] ] + feature_grid_shape
        self.full_shape = opt['full_shape']
        self.feature_grid = torch.rand(feat_shape, 
            dtype=torch.float32)
        self.feature_grid = torch.nn.Parameter(self.feature_grid, 
            requires_grad=True)

        def init_decoder_tcnn():
            import tinycudann as tcnn 
            
            input_size:int = opt['n_features'] +opt['num_positional_encoding_terms']*opt['n_dims']*2
            if(opt['requires_padded_feats']):
                input_size = opt['n_features'] +opt['num_positional_encoding_terms']*opt['n_dims']*2 + self.padding_size
                
            decoder = tcnn.Network(
                n_input_dims=input_size,
                n_output_dims=opt['n_outputs'],
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": opt['nodes_per_layer'],
                    "n_hidden_layers": opt['n_layers'] ,
                }
            )
            return decoder
        
        def init_decoder_pytorch():
            decoder = nn.ModuleList()
            
            input_size:int = opt['n_features'] +opt['num_positional_encoding_terms']*opt['n_dims']*2
            if(opt['requires_padded_feats']):
                input_size = opt['n_features'] +opt['num_positional_encoding_terms']*opt['n_dims']*2 + self.padding_size
                                    
            layer = ReLULayer(input_size, 
                opt['nodes_per_layer'], bias=False)
            decoder.append(layer)
            
            for i in range(opt['n_layers'] ):
                if i == opt['n_layers']  - 1:
                    layer = nn.Linear(opt['nodes_per_layer'], opt['n_outputs'], bias=False)
                    decoder.append(layer)
                else:
                    layer = ReLULayer(opt['nodes_per_layer'], opt['nodes_per_layer'], bias=False)
                    decoder.append(layer)
            decoder = torch.nn.Sequential(*decoder)
            return decoder

        if(opt['use_tcnn_if_available']):
            try:
                self.decoder = init_decoder_tcnn()
            except ImportError:
                print(f"Tried to use TinyCUDANN but found it was not installed - reverting to PyTorch layers.")
                self.decoder = init_decoder_pytorch()
        else:
            self.decoder = init_decoder_pytorch()

        self.register_buffer(
            "volume_min",
            torch.tensor([opt['data_min']], requires_grad=False, dtype=torch.float32),
            persistent=False
        )
        self.register_buffer(
            "volume_max",
            torch.tensor([opt['data_max']], requires_grad=False, dtype=torch.float32),
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

        