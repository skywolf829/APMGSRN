from random import gauss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Other.utility_functions import make_coord_grid    
from Models.layers import LReLULayer, SineLayer, SnakeAltLayer, PositionalEncoding
import tinycudann as tcnn
from math import log, exp

       
class AMRSRN_TCNN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.opt = opt
        
        init_scales = torch.ones(
                [self.opt['n_grids'], 3],
                device = opt['device']
            ).uniform_(1,2)
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
        
        self.register_buffer("ROOT_TWO", 
                torch.tensor([2.0 ** 0.5]),
                persistent=False)            
        self.register_buffer("FLAT_TOP_GAUSSIAN_EXP", 
                torch.tensor([2.0 * 10.0]),
                persistent=False)
        self.register_buffer("DIM_COEFF", 
                torch.tensor([(2.0 * torch.pi) **(self.opt['n_dims']/2)]),
                persistent=False)
        
        
        self.max_resolution = opt['hash_max_resolution']
        self.base_resolution = opt['hash_base_resolution']
        self.n_grids = opt['n_grids']
        self.table_size = 1 << opt['hash_log2_size']
        self.feat_dim = opt['n_features']
        per_level_scale = exp(
            (log(self.max_resolution) - log(self.base_resolution))/(self.n_grids-1)
        )  # growth 
        self.resolution = torch.floor(
            torch.tensor([self.base_resolution*per_level_scale**i for i in range(self.n_grids)])
        ).long().tolist()
        
        self.feature_grids = []
        for resolution in self.resolution:
            grid = tcnn.Encoding(
                n_input_dims=opt['n_dims'],
                encoding_config={
                    "otype": "Grid",
                    "type": "Hash",
                    "n_levels": 1,
                    "n_features_per_level": self.feat_dim,
                    "log2_hashmap_size": opt['hash_log2_size'],
                    "base_resolution": resolution,
                    "per_level_scale": per_level_scale,
                }
            )
            self.feature_grids.append(grid)
        self.feature_grids = nn.ModuleList(self.feature_grids)
            
        self.decoder_dim = opt['nodes_per_layer']
        self.decoder_outdim = opt['n_outputs']
        self.decoder_layers = opt['n_layers']
        
        self.decoder = tcnn.Network(
            n_input_dims=self.feat_dim*self.n_grids,
            n_output_dims=self.decoder_outdim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.decoder_dim,
                "n_hidden_layers": self.decoder_layers,
            }
        )
    
    def get_transformation_matrices(self):
        transformation_matrices = torch.zeros(
                [self.opt['n_grids'], 4, 4],
                device = self.opt['device']
            )
        transformation_matrices[:,0,0] = self.grid_scales[:,0]
        transformation_matrices[:,1,1] = self.grid_scales[:,1]
        transformation_matrices[:,2,2] = self.grid_scales[:,2]
        transformation_matrices[:,0:3,-1] = self.grid_translations      
        transformation_matrices[:,-1,-1] = 1
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
        return transformed_points * (0.6)

    def inverse_transform(self, x):
        transformed_points = torch.cat([x * (1/0.6), torch.ones(
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
       
        x = x.unsqueeze(1).repeat(1,self.opt['n_grids'],1).detach()
        local_to_globals = torch.inverse(self.get_transformation_matrices())
        grid_centers = local_to_globals[:,0:3,-1]
        grid_stds = torch.diagonal(local_to_globals, 0, 1, 2)[:,0:3]
        
        coeffs = 1 / \
        (torch.prod(grid_stds, dim=-1).unsqueeze(0) * \
            self.DIM_COEFF)
        
        exps = torch.exp(-1 * \
            torch.sum(
                (((x - grid_centers.unsqueeze(0))) / \
                (self.ROOT_TWO * grid_stds.unsqueeze(0)))**self.FLAT_TOP_GAUSSIAN_EXP, 
            dim=-1))
        
        return torch.sum(coeffs * exps, dim=-1, keepdim=True)
       
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
        #with torch.no_grad():            
            #self.grid_scales.clamp_(1, 32)
            #max_deviation = self.grid_scales-1
            #self.grid_translations.clamp_(-max_deviation, max_deviation)
        return

    def forward(self, x):
        transformed_points = self.transform(x)
        
        # rescale [-1, 1] to [0, 1] for TCNN's hash grid
        transformed_points = (transformed_points + 1) / 2 
        feats = []
        for grid, points in zip(self.feature_grids, transformed_points):
            # zero out out-of-bound points' features for each local grid
            inbound_selector = ((points > 0.0) & (points < 1.0)).all(dim=-1)
            feat = grid(points)*inbound_selector[...,None]
            feats.append(feat)
        feats = torch.cat(feats, dim=-1)
        
        y = self.decoder(feats).float()
        return y