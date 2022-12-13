from random import gauss
import torch
import torch.nn as nn
import torch.nn.functional as F
from Other.utility_functions import make_coord_grid    
from Models.layers import LReLULayer

       
class SigmoidNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.opt = opt
        
        feat_grid_shape = opt['feature_grid_shape'].split(',')
        feat_grid_shape = [eval(i) for i in feat_grid_shape]
        
        init_translations = torch.ones(
                [self.opt['n_grids'], 3],
                device = opt['device']
            ).uniform_(-0.1, 0.1)

        init_scales = torch.zeros(
                [self.opt['n_grids'], 3],
                device = opt['device']
            ).uniform_(3, 4)

        self.grid_scales = torch.nn.Parameter(
            init_scales,
            requires_grad=True
        )
        self.grid_translations = torch.nn.Parameter(
            init_translations,
            requires_grad=True
        )
        
        
        self.feature_grids =  torch.nn.parameter.Parameter(
            torch.ones(
                [self.opt['n_grids'], self.opt['n_features'], 
                feat_grid_shape[0], feat_grid_shape[1], feat_grid_shape[2]],
                device = opt['device']
            ).uniform_(-0.001, 0.001),
            requires_grad=True
        )
        
        
        self.decoder = nn.ModuleList()
        
        first_layer_input_size = opt['n_features']*opt['n_grids']
                 
        layer = LReLULayer(first_layer_input_size, 
                            opt['nodes_per_layer'])
        self.decoder.append(layer)
        
        for i in range(opt['n_layers']):
            if i == opt['n_layers'] - 1:
                layer = nn.Linear(opt['nodes_per_layer'], opt['n_outputs'])
                nn.init.xavier_normal_(layer.weight)
                self.decoder.append(layer)
            else:
                layer = LReLULayer(opt['nodes_per_layer'], opt['nodes_per_layer'])
                self.decoder.append(layer)
    
    def transform(self, x):
        
        transformed_points = x.unsqueeze(0).repeat(
            self.opt['n_grids'], 1, 1)
        transformed_points = -1.1 + (2.2 / \
            (1 + \
                torch.exp(-self.grid_scales.unsqueeze(1)*(transformed_points - self.grid_translations.unsqueeze(1)))
                )
            )
        return transformed_points
    
    def inverse_transform(self, x):
        transformed_points = x.unsqueeze(0).repeat(
            self.opt['n_grids'], 1, 1)
        transformed_points = self.grid_translations.unsqueeze(1) - (1/self.grid_scales.unsqueeze(1)) * \
            torch.log((2.2/(transformed_points + 1.1))-1)
        return transformed_points

    def fix_params(self):
        return

    def forward(self, x):   
        
        transformed_points = self.transform(x)       
        
        transformed_points = transformed_points.unsqueeze(1).unsqueeze(1)
        feats = F.grid_sample(self.feature_grids,
                transformed_points,
                mode='bilinear', align_corners=True,
                padding_mode="border")[:,:,0,0,:]
        
        feats = feats.flatten(0,1).permute(1, 0)

        if(self.opt['use_global_position']):
            x = x + 1.0
            x = x / 2.0
            x = x * self.dim_global_proportions
            x = x + self.dim_start
        
        y = feats
        
        i = 0
        while i < len(self.decoder):
            y = self.decoder[i](y)
            i = i + 1
            
        return y

        