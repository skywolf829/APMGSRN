from random import gauss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Other.utility_functions import make_coord_grid    
from Models.layers import LReLULayer, SineLayer, SnakeAltLayer, PositionalEncoding

       
class afVSRN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.opt = opt
        
        self.total_n_features = 1
        feat_grid_shape = opt['feature_grid_shape'].split(',')
        feat_grid_shape = [eval(i) for i in feat_grid_shape]
        for i in feat_grid_shape:
            self.total_n_features *= i
        # Generate random centers for the gaussians
        self.feature_locations = torch.nn.parameter.Parameter(
            torch.rand(
                [self.total_n_features, opt['n_dims']],
                device = opt['device']
            ) * 2 - 1
        )

        # Generate random starting features for each gaussian        
        self.feature_vectors = torch.nn.parameter.Parameter(
            torch.ones(
                [self.total_n_features, opt['n_features']],
                device = opt['device']
            ).normal_(-1, 1)
        )
        self.pe = PositionalEncoding(opt)
        
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
    
        
    def forward(self, x):     
        
        feat_distance = x.unsqueeze(1).repeat(1, self.total_n_features, 1)
        feat_distance = feat_distance - self.feature_locations
        feat_distance = feat_distance ** 2
        feat_distance = feat_distance.sum(dim=-1)
        feat_distance = feat_distance ** 0.5
        feat_distance = 1 / (feat_distance + 1e-6)
        feat_distance = F.softmax(feat_distance, dim=1)
        
        feats = torch.matmul(feat_distance, self.feature_vectors)
        
        if(self.opt['use_global_position']):
            x = x + 1.0
            x = x / 2.0
            x = x * self.dim_global_proportions
            x = x + self.dim_start
        
        pe = self.pe(x)  
        y = torch.cat([pe, feats], dim=1)
        
        i = 0
        while i < len(self.decoder):
            y = self.decoder[i](y)
            i = i + 1
            
        return y

        