from random import gauss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Other.utility_functions import make_coord_grid    
from Models.layers import LReLULayer, SineLayer, SnakeAltLayer, PositionalEncoding

class fVSRN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.opt = opt
        self.recently_added_layer = False

        if(opt['extents'] is not None):
            ext = opt['extents'].split(',')
            self.ext = [eval(i) for i in ext]
            dim_size_voxels = [
                self.ext[1]-self.ext[0], self.ext[3]-self.ext[2], self.ext[5]-self.ext[4]
            ]
            dim_start = [
                (self.ext[0] / self.opt['full_shape'][0])*2 - 1,
                (self.ext[1] / self.opt['full_shape'][1])*2 - 1,
                (self.ext[2] / self.opt['full_shape'][2])*2 - 1
            ]
            dim_global_proportions = [
                2*dim_size_voxels[0]/self.opt['full_shape'][0],
                2*dim_size_voxels[1]/self.opt['full_shape'][1],
                2*dim_size_voxels[2]/self.opt['full_shape'][2]
            ]

            self.register_buffer("dim_start", 
                torch.tensor(dim_start), persistent=False)
            self.register_buffer("dim_global_proportions", 
                torch.tensor(dim_global_proportions), persistent=False)

        self.pe = PositionalEncoding(opt)
        
        feat_shape = [1, opt['n_features']] + [eval(i) for i in opt['feature_grid_shape'].split(",")]

        self.feature_grid = torch.rand(feat_shape, 
            device=self.opt['device'], dtype=torch.float32)
        self.feature_grid = torch.nn.Parameter(self.feature_grid, 
            requires_grad=True)



        try:
            import tinycudann as tcnn 
            print(f"Using TinyCUDANN (tcnn) since it is installed for performance gains.")
            print(f"WARNING: This model will be incompatible with non-tcnn compatible systems")
            self.decoder = tcnn.Network(
                n_input_dims=opt['num_positional_encoding_terms']*opt['n_dims']*2 + opt['n_features'],
                n_output_dims=opt['n_outputs'],
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": opt['nodes_per_layer'],
                    "n_hidden_layers": opt['n_layers'],
                }
            )
        except ImportError:
            print(f"TinyCUDANN (tcnn) not installed: falling back to normal PyTorch")
            self.decoder = nn.ModuleList()
            
            first_layer_input_size = opt['n_features']*opt['n_grids']# + opt['num_positional_encoding_terms']*opt['n_dims']*2
                    
            layer = LReLULayer(first_layer_input_size, 
                                opt['nodes_per_layer'])
            self.decoder.append(layer)
            
            for i in range(opt['n_layers']):
                if i == opt['n_layers'] - 1:
                    layer = nn.Linear(opt['nodes_per_layer'], opt['n_outputs'])
                    self.decoder.append(layer)
                else:
                    layer = LReLULayer(opt['nodes_per_layer'], opt['nodes_per_layer'])
                    self.decoder.append(layer)
            self.decoder = torch.nn.Sequential(*self.decoder)
                    
        
    def forward(self, x):     
        
        feats = F.grid_sample(self.feature_grid,
                x.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                mode='bilinear', align_corners=True) 
        if(self.opt['use_global_position']):
            x = x + 1.0
            x = x / 2.0
            x = x * self.dim_global_proportions
            x = x + self.dim_start
        pe = self.pe(x)  
        feats = feats.flatten(0,1).permute(1, 0)
        y = torch.cat([pe, feats], dim=1)
        y = self.decoder(y).float()
        return y

        