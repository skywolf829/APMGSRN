from random import gauss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Other.utility_functions import make_coord_grid    
from Models.layers import LReLULayer, ReLULayer, SineLayer, SnakeAltLayer, PositionalEncoding
  
       
class AMRSRN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.opt = opt

        feat_grid_shape = opt['feature_grid_shape'].split(',')
        feat_grid_shape = [eval(i) for i in feat_grid_shape]
        
        init_scales = torch.ones(
                [self.opt['n_grids'], 3],
                device = opt['device']
            ).uniform_(1.5,2.5)
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
        
        self.feature_grids =  torch.nn.parameter.Parameter(
            torch.ones(
                [self.opt['n_grids'], self.opt['n_features'], 
                feat_grid_shape[0], feat_grid_shape[1], feat_grid_shape[2]],
                device = opt['device']
            ).uniform_(-0.001, 0.001),
            requires_grad=True
        )
        
        self.pe = PositionalEncoding(opt)
        
        try:
            import tinycudann as tcnn 
            print(f"Using TinyCUDANN (tcnn) since it is installed for performance gains.")
            self.decoder = tcnn.Network(
                n_input_dims=self.feat_dim*self.n_grids,
                n_output_dims=self.decoder_outdim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": opt['nodes_per_layer'],
                    "n_hidden_layers": opt['n_layers'],
                }
            )
        except ImportError:
            
            self.decoder = nn.ModuleList()
            
            first_layer_input_size = opt['n_features']*opt['n_grids']# + opt['num_positional_encoding_terms']*opt['n_dims']*2
                    
            layer = ReLULayer(first_layer_input_size, 
                                opt['nodes_per_layer'])
            self.decoder.append(layer)
            
            for i in range(opt['n_layers']):
                if i == opt['n_layers'] - 1:
                    layer = nn.Linear(opt['nodes_per_layer'], opt['n_outputs'])
                    nn.init.xavier_normal_(layer.weight)
                    self.decoder.append(layer)
                else:
                    layer = ReLULayer(opt['nodes_per_layer'], opt['nodes_per_layer'])
                    self.decoder.append(layer)
    
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

    '''
    Transforms global coordinates x to local coordinates within
    each feature grid, where feature grids are assumed to be on
    the boundary of [-1, 1]^3 in their local coordinate system.
    Scales the grid by a factor to match the gaussian shape
    (see feature_density_gaussian())
    
    x: Input coordinates with shape [batch, 3]
    returns: local coordinates in a shape [batch, n_grids, 3]
    '''
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
        return transformed_points * (1/1.48)

    '''
    Transforms local coordinates within each feature grid x to 
    global coordinates. Scales local coordinates by a factor
    so as to be consistent with the transform() method, which
    attempts to align feature grids with the guassian density 
    calculated in feature_density_gaussian()
    
    x: Input coordinates with shape [batch, 3]
    returns: local coordinates in a shape [batch, n_grids, 3]
    '''
    def inverse_transform(self, x):
        transformed_points = torch.cat([x * 1.48, torch.ones(
            [x.shape[0], 1], 
            device=self.opt['device'],
            dtype=torch.float32)], 
            dim=1)
        transformed_points = transformed_points.unsqueeze(0).expand(
            self.opt['n_grids'], transformed_points.shape[0], transformed_points.shape[1])
        # Slow
        #local_to_global_matrices = torch.inverse(self.get_transformation_matrices())
        # fast        
        local_to_global_matrices = torch.zeros([self.opt['n_grids'], 4, 4],
                                               device=self.opt['device'],
                                               dtype=torch.float32)
        local_to_global_matrices[:,0,0] = 1/self.grid_scales[:,0]
        local_to_global_matrices[:,1,1] = 1/self.grid_scales[:,1]
        local_to_global_matrices[:,2,2] = 1/self.grid_scales[:,2]
        local_to_global_matrices[:,0:3,-1] = -self.grid_translations/self.grid_scales
        local_to_global_matrices[:,-1,-1] = 1
        
        transformed_points = torch.bmm(local_to_global_matrices,
                                    transformed_points.transpose(-1,-2)).transpose(-1, -2)
        transformed_points = transformed_points[...,0:3].detach().cpu()
        return transformed_points
    
    def feature_density_gaussian(self, x):
       
        x = x.unsqueeze(1).repeat(1,self.opt['n_grids'],1).detach()
        
        # The expensive (but general) way
        #local_to_globals = torch.inverse(self.get_transformation_matrices())
        #grid_centers = local_to_globals[:,0:3,-1]
        #grid_stds = torch.diagonal(local_to_globals, 0, 1, 2)[:,0:3]
        
        # The cheap way if only a translation/scale
        grid_stds = 1/self.grid_scales
        grid_centers = -self.grid_translations*grid_stds
        
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
        
        transformed_points = transformed_points.unsqueeze(1).unsqueeze(1)
        feats = F.grid_sample(self.feature_grids,
                transformed_points.detach(),
                mode='bilinear', align_corners=True,
                padding_mode="zeros")[:,:,0,0,:]
        feats = feats.flatten(0,1).permute(1, 0)
        
        
        y = feats
        i = 0
        while i < len(self.decoder):
            y = self.decoder[i](y)
            i = i + 1
        
        return y

        