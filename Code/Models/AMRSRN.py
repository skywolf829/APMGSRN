from random import gauss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Other.utility_functions import make_coord_grid    
from Models.layers import LReLULayer, SineLayer, SnakeAltLayer, PositionalEncoding

       
class AMRSRN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.opt = opt
        
        feat_grid_shape = opt['feature_grid_shape'].split(',')
        feat_grid_shape = [eval(i) for i in feat_grid_shape]
        
        init_grid_transforms = torch.zeros(
                [self.opt['n_grids'], 4, 4],
                device = opt['device']
            ).normal_(1, 1)
        
        init_grid_transforms[:,-1,:] = 0
        init_grid_transforms[:,-1,-1] = 1
        init_grid_transforms[:] = torch.eye(4,
                            dtype=torch.float32,
                            device=opt['device'])
        
        '''
        level = 0
        grid = 0
        translate_start = 0
        while grid < opt['n_grids']:
            grids_remaining = opt['n_grids'] - grid
            scale = 0.5 ** level
            
            grids_this_level = 8**level
            # if all can fit uniformly
            if(grids_remaining >= grids_this_level):
                grids_per_dim = int(2 ** level) 
                
                for x in range(grids_per_dim):
                    for y in range(grids_per_dim):
                        for z in range(grids_per_dim):
                            x_trans = translate_start + x*scale
                            y_trans = translate_start + y*scale
                            z_trans = translate_start + z*scale
                            
                            init_grid_transforms[grid,0,0] = scale
                            init_grid_transforms[grid,1,1] = scale
                            init_grid_transforms[grid,2,2] = scale
                            
                            init_grid_transforms[grid,0,-1] = x_trans
                            init_grid_transforms[grid,1,-1] = y_trans
                            init_grid_transforms[grid,2,-1] = z_trans                            
                            
                            init_grid_transforms[grid,-1,-1] = 1
                            
                            grid += 1
            else:
                init_grid_transforms[grid:,0,0] = scale
                init_grid_transforms[grid:,1,1] = scale          
                init_grid_transforms[grid:,2,2] = scale
                
                init_grid_transforms[grid:,0,-1] = torch.rand([opt['n_grids']-grid], 
                                                              device=opt['device'],
                                                              dtype=torch.float32)
                init_grid_transforms[grid:,1,-1] = torch.rand([opt['n_grids']-grid], 
                                                              device=opt['device'],
                                                              dtype=torch.float32)
                init_grid_transforms[grid:,2,-1] = torch.rand([opt['n_grids']-grid], 
                                                              device=opt['device'],
                                                              dtype=torch.float32)              
                
                init_grid_transforms[grid:,-1,-1] = 1
                grid += opt['n_grids'] - grid
            
            translate_start -= 0.5**(level+1)
            level += 1
        '''
        
        self.feature_grid_transform_matrices =  torch.nn.parameter.Parameter(
            init_grid_transforms,
            requires_grad=True
        )
        
        self.feature_grids =  torch.nn.parameter.Parameter(
            torch.ones(
                [self.opt['n_grids'], self.opt['n_features'], 
                feat_grid_shape[0], feat_grid_shape[1], feat_grid_shape[2]],
                device = opt['device']
            ).normal_(0, 1),
            requires_grad=True
        )
        
        self.pe = PositionalEncoding(opt)
        
        self.decoder = nn.ModuleList()
        
        first_layer_input_size = opt['num_positional_encoding_terms']*opt['n_dims']*2 + \
            opt['n_features']*opt['n_grids']        
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
        
        transformed_points = torch.cat([x, torch.ones([x.shape[0], 1], 
            device=self.opt['device'],
            dtype=torch.float32)], 
            dim=1)
        transformed_points = transformed_points.unsqueeze(0).expand(
            self.opt['n_grids'], 
            transformed_points.shape[0], 
            transformed_points.shape[1])
        
        transformed_points = torch.bmm(transformed_points, 
                            self.feature_grid_transform_matrices.transpose(-1, -2))
        transformed_points = transformed_points[...,0:3]
       
        
        transformed_points = transformed_points.unsqueeze(1).unsqueeze(1)
        feats = F.grid_sample(self.feature_grids,
                transformed_points,
                mode='bilinear', align_corners=True,
                padding_mode="zeros") 
        feats = feats.squeeze().flatten(0,1).permute(1, 0)
        
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

        