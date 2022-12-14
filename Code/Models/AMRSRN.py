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
        
        init_scales = torch.ones(
                [self.opt['n_grids'], 3],
                device = opt['device']
            ).uniform_(1,1.5)
            #.uniform_(1.9,2.1)

        init_translations = torch.zeros(
                [self.opt['n_grids'], 3],
                device = opt['device']
            ).uniform_(-1, 1) * (init_scales-1)
            #.uniform_(0.85, 1) * (init_scales-1)
        #init_translations[:,-1].uniform_(-1, -0.85) 
        #init_translations[:,-1] *= (init_scales[:,-1]-1)
    
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
        
        self.pe = PositionalEncoding(opt)
        
        self.decoder = nn.ModuleList()
        
        first_layer_input_size = opt['n_features']*opt['n_grids'] #+ opt['num_positional_encoding_terms']*opt['n_dims']*2
                 
        layer = SnakeAltLayer(first_layer_input_size, 
                            opt['nodes_per_layer'])
        self.decoder.append(layer)
        
        for i in range(opt['n_layers']):
            if i == opt['n_layers'] - 1:
                layer = nn.Linear(opt['nodes_per_layer'], opt['n_outputs'])
                nn.init.xavier_normal_(layer.weight)
                self.decoder.append(layer)
            else:
                #layer = SnakeAltLayer(opt['nodes_per_layer'] + opt['n_features']*opt['n_grids'], opt['nodes_per_layer'])
                layer = LReLULayer(opt['nodes_per_layer'], opt['nodes_per_layer'])
                self.decoder.append(layer)
    
    def get_transformation_matrices(self):
        transformation_matrices = torch.zeros(
                [self.opt['n_grids'], 4, 4],
                device = self.opt['device']
            )
        transformation_matrices[:,-1,-1] = 1
        transformation_matrices[:,0,0] = self.grid_scales[:,0]
        transformation_matrices[:,1,1] = self.grid_scales[:,1]
        transformation_matrices[:,2,2] = self.grid_scales[:,2]
        transformation_matrices[:,0:3,-1] = self.grid_translations
        return transformation_matrices

    def transform(self, x):
        transformed_points = torch.cat([x, torch.ones([x.shape[0], 1], 
            device=self.opt['device'],
            dtype=torch.float32)], 
            dim=1)
        transformed_points = transformed_points.unsqueeze(0).repeat(
            self.opt['n_grids'], 1, 1)
        transformation_matrices = self.get_transformation_matrices()
        
        transformed_points = torch.bmm(transformation_matrices, 
                            transformed_points.transpose(-1, -2)).transpose(-1, -2)
        transformed_points = transformed_points[...,0:3]
        return transformed_points

    def inverse_transform(self, x):
        transformed_points = torch.cat([x, torch.ones(
            [x.shape[0], 1], 
            device=self.opt['device'],
            dtype=torch.float32)], 
            dim=1)
        transformed_points = transformed_points.unsqueeze(0).expand(
            self.opt['n_grids'], transformed_points.shape[0], transformed_points.shape[1])
        local_to_global_matrices = torch.inverse(self.get_transformation_matrices())
        
        transformed_points = torch.bmm(local_to_global_matrices,
                                    transformed_points.transpose(-1,-2)).transpose(-1, -2)
        transformed_points = transformed_points[...,0:3].detach().cpu()
        return transformed_points
    
    def feature_density_gaussian(self, x):
       
       
        x = x.unsqueeze(1).repeat(1,self.opt['n_grids'],1)
        local_to_globals = torch.inverse(self.get_transformation_matrices())
        grid_centers = local_to_globals[:,0:3,-1]
        grid_stds = torch.diagonal(local_to_globals, 0, 1, 2)[:,0:3] * 0.5
        
        exps = ((x - grid_centers.unsqueeze(0))**2) / (2 * (grid_stds.unsqueeze(0)**2))
        exps = -torch.sum(exps,dim=2)
        exps = torch.exp(exps)
        
        exps = exps / (torch.prod(grid_stds.unsqueeze(0), dim=2) ** 0.5)
        exps = torch.sum(exps, dim=1)
        return exps / exps.max()
        

    def feature_density_box(self, volume_shape):
        feat_density = torch.zeros(volume_shape, 
            device=self.opt['device'], dtype=torch.float32)
        
        starts = torch.tensor([-1, -1, -1], device=self.opt['device'], dtype=torch.float32).unsqueeze(0)
        stops = torch.tensor([1, 1, 1], device=self.opt['device'], dtype=torch.float32).unsqueeze(0)

        ends = torch.cat([starts, stops], dim=0)
        transformed_points = self.inverse_transform(ends)
        transformed_points += 1
        transformed_points *= 0.5 * (torch.tensor(volume_shape)-1)
        transformed_points = transformed_points.type(torch.LongTensor)
        print(transformed_points.shape)

        for i in range(transformed_points.shape[0]):
            feat_density[transformed_points[i,0,0]:transformed_points[i,1,0],
                transformed_points[i,0,1]:transformed_points[i,1,1],
                transformed_points[i,0,2]:transformed_points[i,1,2]] += 1


        return feat_density.permute(2,1,0)

    def fix_params(self):
        with torch.no_grad():
            self.grid_scales.clamp_(1, 32)
            max_deviation = self.grid_scales-1
            self.grid_translations.clamp_(-max_deviation, max_deviation)

    def forward(self, x):   
        
        transformed_points = self.transform(x)       
        
        transformed_points = transformed_points.unsqueeze(1).unsqueeze(1)
        feats = F.grid_sample(self.feature_grids,
                transformed_points.detach(),
                mode='bilinear', align_corners=True,
                padding_mode="zeros")[:,:,0,0,:]
        
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

        