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
        
        feat_shape = [1, opt['n_features']]
        grid_shape = opt['feature_grid_shape'].split(",")
        for i in range(len(grid_shape)):
            feat_shape.append(int(grid_shape[i]))
        #print(f'Feature grid shape: {feat_shape}')
        self.feature_grid = torch.rand(feat_shape, 
            device=self.opt['device'], dtype=torch.float32)
        self.feature_grid = torch.nn.Parameter(self.feature_grid, 
            requires_grad=True)

        self.decoder = nn.ModuleList()
        first_layer_input_size = opt['num_positional_encoding_terms']*opt['n_dims']*2 + opt['n_features']
        layer = SnakeAltLayer(first_layer_input_size, 
                            opt['nodes_per_layer'])
        self.decoder.append(layer)
        
        for i in range(opt['n_layers']):
            if i == opt['n_layers'] - 1:
                layer = nn.Linear(opt['nodes_per_layer'], opt['n_outputs'])
                nn.init.xavier_normal_(layer.weight)
                self.decoder.append(layer)
            else:
                layer = SnakeAltLayer(opt['nodes_per_layer'], opt['nodes_per_layer'])
                self.decoder.append(layer)
            
    def add_layer(self):
        self.recently_added_layer = True
        self.previous_last_layer = self.decoder[-1]
        self.decoder.pop(-1)
        self.decoder.append(SnakeAltLayer(
            self.opt['nodes_per_layer'], 
            self.opt['nodes_per_layer'])
        )
        new_last_layer = nn.Linear(self.opt['nodes_per_layer'], 
                                   self.opt['n_outputs'])
        nn.init.normal_(new_last_layer.weight, 0, 0.001)
        self.decoder.append(new_last_layer)
        
        
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
        feats = feats.squeeze().permute(1, 0)
        y = torch.cat([pe, feats], dim=1)
        
        i = 0
        while i < len(self.decoder):
            if(self.recently_added_layer and i == len(self.decoder - 2)):
                
                y1 = self.previous_last_layer(y.clone())
                y2 = self.decoder[i](y.clone())
                y2 = self.decoder[i+1](y2)
                
                a = self.opt['iters_since_new_layer'] / self.opt['iters_to_train_new_layer']
                b = 1 - a
                y = b*y1 + a*y2
                i = i + 2
            else:
                y = self.decoder[i](y)
                i = i + 1
            
        return y

        