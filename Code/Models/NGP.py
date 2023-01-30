import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import exp, log
from Other.utility_functions import make_coord_grid
from Models.layers import LinearLayer
import tinycudann as tcnn
# from Models.layers import LReLULayer, SineLayer, SnakeAltLayer, PositionalEncoding

# feature grid from fVSRN.py, but initialize with Uniform(-1e-4, 1e-4) as instant-NGP paper
class FeatureGrid(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        
        feat_shape = [1, opt['n_features']]
        grid_shape = opt['feature_grid_shape'].split(",")
        for i in range(len(grid_shape)):
            feat_shape.append(int(grid_shape[i]))

        self.feature_grid = torch.rand(feat_shape,
                                       device=self.opt['device'], dtype=torch.float32)
        self.feature_grid = torch.nn.Parameter(torch.Tensor(*feat_shape),
                                               requires_grad=True)
        nn.init.uniform_(self.feature_grid, a=-0.0001, b=0.0001) # initialize with Uniform(-1e-4, 1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = F.grid_sample(self.feature_grid,
                              x.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                              mode='bilinear', align_corners=True)
        return feats

# TODO: for grid size < table size, use FeatureGrid to avoid inefficient hashing + interpolation implemnentaion -ty
class MultiHashGrid(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        # interpolation metadata
        self.VOXEL_VTX_OFFSET = torch.tensor([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]],
                                             device=self.opt['device']).long()
        self.bbox = torch.tensor([[-1,-1,-1],[1,1,1]], device=self.opt['device'])
        # hash grid metadata
        self.PRIMES = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737] # from tiny-cuda-nn implementation
        self.max_resolution = opt['hash_max_resolution']
        self.base_resolution = opt['hash_base_resolution']
        self.n_grids = opt['n_grids']
        self.table_size = 1 << opt['hash_log2_size']
        self.feat_dim = opt['n_features']
        per_level_scale = exp(
            (log(self.max_resolution) - log(self.base_resolution))/(self.n_grids-1)
        )  # growth factor for resolution from base to max, eq(3)
        
        # resolution for grid on each level, eq(2)
        self.resolution = torch.floor(
            torch.tensor([self.base_resolution*per_level_scale**i for i in range(self.n_grids)])
        ).long().tolist()
        
        self.feature_grid = nn.ModuleList([nn.Embedding(self.table_size, self.feat_dim) for i in range(self.n_grids)])
        for grid in self.feature_grid:
            nn.init.uniform_(grid.weight, a=-0.0001, b=0.0001)
            
        self.table_size = torch.tensor(self.table_size, dtype=torch.int32, device=self.opt['device'])

    def spatial_hashing(self, x:torch.LongTensor):
        xor_result = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
        with torch.no_grad():
            for i in range(x.shape[-1]):
                xor_result ^= x[..., i]*self.PRIMES[i]
            return xor_result % self.table_size
        
    def find_voxel(self, x:torch.Tensor, grid_resolution):
        '''
        given input coordinates and the current grid resolution,
        return:
            the interpolation weights (B, 3)
            hash table idx for voxel vtx of each coord (B, 3)
        '''
        bbox_min, bbox_max = self.bbox
        with torch.no_grad():
            x_normed = (x - bbox_min) / (bbox_max - bbox_min) * grid_resolution
            # map coordinate from bbox scale to [0, grid_resolution] for easier interpolation weight calc
        voxel_idx_000 = torch.floor(x_normed).long()
        interp_w = x_normed - voxel_idx_000
        
        voxel_idx_all = voxel_idx_000.unsqueeze(1) + self.VOXEL_VTX_OFFSET
        hash_idx_all = self.spatial_hashing(voxel_idx_all)
        
        return interp_w, hash_idx_all

    def trilerp_per_voxel(self, interp_w, vox_feats):
        '''
        computer trilerp results given a set of coordinates, their voxel min/max, and vtx values.
        x: (B, 3)
        vox_weights: (B, 3) interpolation weights in each dimension for x in its voxel
        vox_vals: (B, 8, value_dim)
        '''
        c00 = vox_feats[:,0]*(1-interp_w[:,0][:,None]) + vox_feats[:,4]*interp_w[:,0][:,None]
        c01 = vox_feats[:,1]*(1-interp_w[:,0][:,None]) + vox_feats[:,5]*interp_w[:,0][:,None]
        c10 = vox_feats[:,2]*(1-interp_w[:,0][:,None]) + vox_feats[:,6]*interp_w[:,0][:,None]
        c11 = vox_feats[:,3]*(1-interp_w[:,0][:,None]) + vox_feats[:,7]*interp_w[:,0][:,None]
        c0 = c00*(1-interp_w[:,1][:,None]) + c10*interp_w[:,1][:,None]
        c1 = c01*(1-interp_w[:,1][:,None]) + c11*interp_w[:,1][:,None]
        c = c0*(1-interp_w[:,2][:,None]) + c1*interp_w[:,2][:,None]
        return c
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inbox_selector = ((x >= -1.0) & (x <= 1.0)).all(dim=-1)
        
        feats = []
        for grid, resolution in zip(self.feature_grid, self.resolution):
            x_feat = torch.zeros(x.shape[0], self.feat_dim).to(x)
            x_inbox = x[inbox_selector]
            interp_w, hash_idx = self.find_voxel(x_inbox, resolution)
            
            vox_feat = grid(hash_idx) # (B_inbox,8,3)
            x_feat[inbox_selector] = self.trilerp_per_voxel(interp_w, vox_feat)
            feats.append(x_feat)
        
        feats = torch.concatenate(feats, -1)
        return feats
    
class NGP(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
    
        self.opt = opt
        
        self.hash_grid = MultiHashGrid(opt)
        
        self.decoder_indim = self.hash_grid.feat_dim * self.hash_grid.n_grids
        self.decoder_dim = opt['nodes_per_layer']
        self.decoder_outdim = opt['n_outputs']
        self.decoder_layers = opt['n_layers']
        self.decoder = nn.Sequential(
            LinearLayer(self.decoder_indim, self.decoder_dim),
            *[LinearLayer(self.decoder_dim, self.decoder_dim) for i in range(self.decoder_layers-1)],
            nn.Linear(self.decoder_dim, self.decoder_outdim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.hash_grid(x))

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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).float()
