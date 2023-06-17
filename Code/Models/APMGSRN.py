import torch
import torch.nn as nn
import torch.nn.functional as F 
from Models.layers import ReLULayer
from typing import List, Dict, Optional
import math

def weights_init(m):
    classname = m.__class__.__name__
    if classname.lower().find('linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if(m.bias is not None):
            torch.nn.init.normal_(m.bias, 0, 0.001) 

class APMG_encoder(nn.Module):
    def __init__(self, n_grids:int, n_features:int,
                 feat_grid_shape:List[int], n_dims:int,
                 grid_initializaton:str):
        super().__init__()
             
        self.transformation_matrices = torch.nn.Parameter(
            torch.zeros(
                [n_grids, n_dims+1, n_dims+1],
                dtype=torch.float32
            ),
            requires_grad=True
        )
        self.feature_grids =  torch.nn.parameter.Parameter(
            torch.ones(
                [n_grids, n_features] + feat_grid_shape,
                dtype=torch.float32
            ).uniform_(-0.0001, 0.0001),
            requires_grad=True
        )
    
        if("default" in grid_initializaton):
            self.randomize_grids()
        elif("small" in grid_initializaton):
            self.init_grids_small()
        elif("large" in grid_initializaton):
            self.init_grids_large()
        else:
            self.randomize_grids()
        

    def init_grids_large(self):  
        with torch.no_grad():     
            d = self.feature_grids.device
            n_dims = len(self.feature_grids.shape[2:])
            n_grids = self.feature_grids.shape[0]
            tm = torch.eye(n_dims+1, 
                device=d, dtype=torch.float32).unsqueeze(0).repeat(n_grids,1,1) * 0.8
            tm[:,0:n_dims,:] += torch.randn_like(
                tm[:,0:n_dims,:],
                device=d, dtype=torch.float32) * 0.05
            #tm @= tm.transpose(-1, -2)           
            tm[:,n_dims,0:n_dims] = 0.0
            tm[:,-1,-1] = 1.0
            
        self.transformation_matrices = torch.nn.Parameter(
            tm,                
            requires_grad=True)

    def init_grids_small(self):  
        with torch.no_grad():     
            d = self.feature_grids.device
            n_dims = len(self.feature_grids.shape[2:])
            n_grids = self.feature_grids.shape[0]
            tm = torch.eye(n_dims+1, 
                device=d, dtype=torch.float32).unsqueeze(0).repeat(n_grids,1,1) * 8
            tm[:,0:n_dims,:] += torch.randn_like(
                tm[:,0:n_dims,:],
                device=d, dtype=torch.float32) * 0.05
            tm[:,0:3,-1] += (torch.rand_like(
                tm[:,0:3,-1],
                device = d, dtype=torch.float32
            ) *2 - 1) * tm.diagonal(0, 1, 2)[:,0:-1]
            #tm @= tm.transpose(-1, -2)           
            tm[:,n_dims,0:n_dims] = 0.0
            tm[:,-1,-1] = 1.0
            
        self.transformation_matrices = torch.nn.Parameter(
            tm,                
            requires_grad=True)

    def randomize_grids(self):  
        with torch.no_grad():     
            d = self.feature_grids.device
            n_dims = len(self.feature_grids.shape[2:])
            n_grids = self.feature_grids.shape[0]
            tm = torch.eye(n_dims+1, 
                device=d, dtype=torch.float32).unsqueeze(0).repeat(n_grids,1,1)
            tm[:,0:n_dims,:] += torch.randn_like(
                tm[:,0:n_dims,:],
                device=d, dtype=torch.float32) * 0.05
            #tm[:,0,0] += (torch.rand_like(tm[:,0,0])-0.01)*0.5
            #tm[:,1,1] += (torch.rand_like(tm[:,1,1])-0.01)*0.5
            #tm[:,2,2] += (torch.rand_like(tm[:,2,2])-0.01)*0.5
            #tm @= tm.transpose(-1, -2)           
            tm[:,n_dims,0:n_dims] = 0.0
            tm[:,-1,-1] = 1.0
            
        self.transformation_matrices = torch.nn.Parameter(
            tm,                
            requires_grad=True)

    def transform(self, x):
        '''
        Transforms global coordinates x to local coordinates within
        each feature grid, where feature grids are assumed to be on
        the boundary of [-1, 1]^n_dims in their local coordinate system.
        Scales the grid by a factor to match the gaussian shape
        (see feature_density_gaussian()). Assumes S*R*T order
        
        x: Input coordinates with shape [batch, n_dims]
        returns: local coordinates in a shape [n_grids, batch, n_dims]
        '''
                
        # x starts [batch,n_dims], this changes it to [n_grids,batch,n_dims+1]
        # by appending 1 to the xy(z(t)) and repeating it n_grids times
        
        batch : int = x.shape[0]
        dims : int = x.shape[1]
        ones = torch.ones([batch, 1], 
            device=x.device,
            dtype=torch.float32)
            
        x = torch.cat([x, ones], dim=1)
        #x = x.unsqueeze(0)
        #x = x.repeat(self.feature_grids.shape[0], 1, 1)
        
        # BMM will result in [n_grids,n_dims+1,n_dims+1] x [n_grids,n_dims+1,batch]
        # which returns [n_grids,n_dims+1,batch], which is then transposed
        # to [n_grids,batch,n_dims+1]
        #transformed_points = torch.bmm(self.transformation_matrices, 
        #                    x.transpose(1, 2)).transpose(1, 2)
        transformed_points = torch.matmul(self.transformation_matrices, 
                            x.transpose(0, 1)).transpose(1, 2)
        transformed_points = transformed_points[...,0:dims]
        
        # return [n_grids,batch,n_dims]
        return transformed_points
   
    def inverse_transform(self, x):
        '''
        Transforms local coordinates within each feature grid x to 
        global coordinates. Scales local coordinates by a factor
        so as to be consistent with the transform() method, which
        attempts to align feature grids with the guassian density 
        calculated in feature_density_gaussian().Assumes S*R*T order,
        so inverse is T^(-1)*R^T*(1/S)
        
        x: Input coordinates with shape [batch, n_dims]
        returns: local coordinates in a shape [n_grids, batch, n_dims]
        '''

        local_to_global_matrices = torch.linalg.inv(self.transformation_matrices)
       
        batch : int = x.shape[0]
        dims : int = x.shape[1]
        ones = torch.ones([batch, 1], 
            device=x.device,
            dtype=torch.float32)
        
        x = torch.cat([x, ones], dim=1)
        #x = x.unsqueeze(0)
        #x = x.repeat(n_grids, 1, 1)
        
        transformed_points = torch.matmul(local_to_global_matrices,
            x.transpose(0,1)).transpose(1, 2)
        transformed_points = transformed_points[...,0:dims]

        return transformed_points
    
    def feature_density_pre_transformed(self, x):
        transformed_points = x
            
        # get the coeffs of shape [n_grids], then unsqueeze to [1,n_grids] for broadcasting
        coeffs = torch.linalg.det(self.transformation_matrices[:,0:-1,0:-1]).unsqueeze(0) \
            / (2.0*torch.pi)**(x.shape[-1]/2)
        #coeffs = torch.prod(self.grid_scales,dim=1).unsqueeze(0) \
        #    / (2.0*torch.pi)**(x.shape[-1]/2)

        # sum the exp part to [batch,n_grids]
        exps = torch.exp(-0.5 * \
            torch.sum(
                transformed_points.transpose(0,1)**20, 
            dim=-1))
        # Another similar solution, may or may not be more efficient gradient computation 
        #exps = 1 / (1 + torch.sum(local_positions**20,dim=-1))
        
        result = torch.sum(coeffs * exps, dim=-1, keepdim=True)
        return result
    
    def feature_density(self, x):
        # Transform the points to local grid spaces first
        transformed_points = self.transform(x)
        return self.feature_density_pre_transformed(transformed_points)     
    
    def forward_pre_transformed(self, x):
        
        # Reshape to proper grid sampling size
        grids : int = x.shape[0]
        batch : int = x.shape[1]
        dims : int = x.shape[2]        
        x = x.reshape(grids, 1, 1, batch, dims)
        
        
        # Sample the grids at the batch of transformed point locations
        # Uses zero padding, so any point outside of [-1,1]^n_dims will be a 0 feature vector
        feats = F.grid_sample(self.feature_grids,
            x.detach() if self.training else x,
            mode='bilinear', align_corners=True,
            padding_mode="zeros").flatten(0, dims).permute(1,0)
        
        return feats
    
    def forward(self, x):
        x = self.transform(x)
        return self.forward_pre_transformed(x)

class APMGSRN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.n_grids : int = opt['n_grids']
        self.n_features : int = opt['n_features'] 
        self.feature_grid_shape : List[int] = [eval(i) for i in opt['feature_grid_shape'].split(',')]
        self.n_dims : int = opt['n_dims']
        self.n_outputs : int = opt['n_outputs'] 
        self.nodes_per_layer : int = opt['nodes_per_layer']
        self.n_layers : int = opt['n_layers'] 
        self.requires_padded_feats : bool = opt['requires_padded_feats']
        self.padding_size : int = 0
        self.full_shape = opt['full_shape']
        if(opt['requires_padded_feats']):
            self.padding_size : int = 16*int(math.ceil(max(1, (opt['n_grids']*opt['n_features'] )/16))) - \
                opt['n_grids']*opt['n_features'] 
            
        self.encoder = APMG_encoder(opt['n_grids'], opt['n_features'] , 
            self.feature_grid_shape, opt['n_dims'], opt['grid_initialization'])
        
        def init_decoder_tcnn():
            import tinycudann as tcnn 
            input_size:int = opt['n_features'] *opt['n_grids'] # + 6*3*2
            if(opt['requires_padded_feats']):
                input_size = opt['n_features'] *opt['n_grids'] + self.padding_size
                
            decoder = tcnn.Network(
                n_input_dims=input_size,
                n_output_dims=opt['n_outputs'] ,
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
            
            first_layer_input_size:int = opt['n_features'] *opt['n_grids'] # + 6*3*2
            if(opt['requires_padded_feats']):
                first_layer_input_size = opt['n_features'] *opt['n_grids'] + self.padding_size
                                           
            layer = ReLULayer(first_layer_input_size, 
                opt['nodes_per_layer'], bias=opt['use_bias'], dtype=torch.float32)
            decoder.append(layer)
            
            for i in range(opt['n_layers'] ):
                if i == opt['n_layers']  - 1:
                    layer = nn.Linear(opt['nodes_per_layer'], opt['n_outputs'] , 
                        bias=opt['use_bias'], dtype=torch.float32)
                    decoder.append(layer)
                else:
                    layer = ReLULayer(opt['nodes_per_layer'], opt['nodes_per_layer'], 
                        bias=opt['use_bias'], dtype=torch.float32)
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

        self.reset_parameters()
    
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

    def get_transform_parameters(self):
        return [{"params": self.encoder.transformation_matrices}]
    
    def get_model_parameters(self):
        return [
            {"params": [self.encoder.feature_grids]},
            {"params": self.decoder.parameters()}
        ]
    
    def get_volume_extents(self):
        return self.full_shape
    
    def reset_parameters(self):
        with torch.no_grad():
            feat_grid_shape = self.feature_grid_shape
            self.encoder.feature_grids =  torch.nn.parameter.Parameter(
                torch.ones(
                    [self.n_grids, self.n_features] + feat_grid_shape,
                    device = self.encoder.feature_grids.device
                ).uniform_(-0.0001, 0.0001),
                requires_grad=True
            )
            self.decoder.apply(weights_init)   

    def feature_density_pre_transformed(self, x):
        return self.encoder.feature_density_pre_transformed(x)

    @torch.jit.export
    def feature_density(self, x):
        return self.encoder.feature_density(x)

    @torch.jit.export
    def transform(self, x):
        return self.encoder.transform(x)
    
    @torch.jit.export
    def inverse_transform(self, x):
        return self.encoder.inverse_transform(x)
    
    @torch.jit.export
    def grad_at(self, x):
        x.requires_grad_(True)
        y = self(x)

        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y),]

        grad_x = torch.autograd.grad([y], [x],
            grad_outputs=grad_outputs)[0]
        return grad_x

    def min(self):
        return self.volume_min

    def max(self):
        return self.volume_max

    def forward_pre_transformed(self, x):
        feats = self.encoder.forward_pre_transformed(x)    
        if(self.requires_padded_feats):
            feats = F.pad(feats, (0, self.padding_size), value=1.0) 
        y = self.decoder(feats)
        y = y.float() * (self.volume_max - self.volume_min) + self.volume_min     
        return y

    def forward(self, x):
        x_t = self.encoder.transform(x)
        return self.forward_pre_transformed(x_t)   

# Works, but is slower and less accurate the the PyTorch version. 
# May provide a useful speed-up in the future if fixed.
class APMGSRN_NGP(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.n_grids = opt['n_grids']
        self.n_dims = opt['n_dims']
        self.feature_grid_shape = [eval(i) for i in opt['feature_grid_shape'].split(',')]
        self.n_features = opt['n_features']
        feat_grid_res = 1
        for i in range(len(self.feature_grid_shape)):
            feat_grid_res *= self.feature_grid_shape[i]
        feat_grid_res = int(feat_grid_res**(1/opt['n_dims']))+1

        self.transformation_matrices = torch.nn.Parameter(
            torch.zeros(
                [opt['n_grids'], opt['n_dims']+1, opt['n_dims']+1],
                dtype=torch.float32
            ),
            requires_grad=True
        )
        encoding_params = []
        for i in range(opt['n_grids']):
            encoding_params.append(
                {
                    "otype": "Grid",
                    "type": "Dense",
                    "n_levels": 1,
                    "n_features_per_level": opt['n_features'],
                    "base_resolution": feat_grid_res,
                    "interpolation": "Linear",
                    "n_dims_to_encode": opt['n_dims']
                }
            )
        import tinycudann as tcnn
        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=opt['n_dims']*opt['n_grids'],
            n_output_dims=opt['n_outputs'],
            encoding_config={
                "otype": "Composite",
                "nested": encoding_params
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": opt['nodes_per_layer'],
                "n_hidden_layers": opt['n_layers'],
            },
        )
        
        if("default" in opt['grid_initialization']):
            self.randomize_grids()
        elif("small" in opt['grid_initialization']):
            self.init_grids_small()
        elif("large" in opt['grid_initialization']):
            self.init_grids_large()
        else:
            self.randomize_grids()

        self.reset_parameters()
    
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

    def get_transform_parameters(self):
        return [{"params": self.transformation_matrices}]

    def get_model_parameters(self):
        return [{"params":self.model.parameters()}]

    def init_grids_large(self):  
        with torch.no_grad():     
            d = self.transformation_matrices.device
            tm = torch.eye(self.n_dims+1, 
                device=d, dtype=torch.float32).unsqueeze(0).repeat(self.n_grids,1,1) * 0.8
            tm[:,0:self.n_dims,:] += torch.randn_like(
                tm[:,0:self.n_dims,:],
                device=d, dtype=torch.float32) * 0.05
            #tm @= tm.transpose(-1, -2)           
            tm[:,self.n_dims,0:self.n_dims] = 0.0
            tm[:,-1,-1] = 1.0
            
        self.transformation_matrices = torch.nn.Parameter(
            tm,                
            requires_grad=True)

    def init_grids_small(self):  
        with torch.no_grad():     
            d = self.transformation_matrices.device
            tm = torch.eye(self.n_dims+1, 
                device=d, dtype=torch.float32).unsqueeze(0).repeat(self.n_grids,1,1) * 8
            tm[:,0:self.n_dims,:] += torch.randn_like(
                tm[:,0:self.n_dims,:],
                device=d, dtype=torch.float32) * 0.05
            tm[:,0:3,-1] += (torch.rand_like(
                tm[:,0:3,-1],
                device = d, dtype=torch.float32
            ) *2 - 1) * tm.diagonal(0, 1, 2)[:,0:-1]
            #tm @= tm.transpose(-1, -2)           
            tm[:,self.n_dims,0:self.n_dims] = 0.0
            tm[:,-1,-1] = 1.0
            
        self.transformation_matrices = torch.nn.Parameter(
            tm,                
            requires_grad=True)

    def randomize_grids(self):  
        with torch.no_grad():     
            d = self.transformation_matrices.device
            tm = torch.eye(self.n_dims+1, 
                device=d, dtype=torch.float32).unsqueeze(0).repeat(self.n_grids,1,1)
            tm[:,0:self.n_dims,:] += torch.randn_like(
                tm[:,0:self.n_dims,:],
                device=d, dtype=torch.float32) * 0.05
            #tm[:,0,0] += (torch.rand_like(tm[:,0,0])-0.01)*0.5
            #tm[:,1,1] += (torch.rand_like(tm[:,1,1])-0.01)*0.5
            #tm[:,2,2] += (torch.rand_like(tm[:,2,2])-0.01)*0.5
            #tm @= tm.transpose(-1, -2)           
            tm[:,self.n_dims,0:self.n_dims] = 0.0
            tm[:,-1,-1] = 1.0
            
        self.transformation_matrices = torch.nn.Parameter(
            tm,                
            requires_grad=True)

    def transform(self, x):
        '''
        Transforms global coordinates x to local coordinates within
        each feature grid, where feature grids are assumed to be on
        the boundary of [-1, 1]^n_dims in their local coordinate system.
        Scales the grid by a factor to match the gaussian shape
        (see feature_density_gaussian()). Assumes S*R*T order
        
        x: Input coordinates with shape [batch, n_dims]
        returns: local coordinates in a shape [n_grids, batch, n_dims]
        '''
                
        # x starts [batch,n_dims], this changes it to [n_grids,batch,n_dims+1]
        # by appending 1 to the xy(z(t)) and repeating it n_grids times
        
        batch : int = x.shape[0]
        dims : int = x.shape[1]
        ones = torch.ones([batch, 1], 
            device=x.device,
            dtype=torch.float32)
            
        x = torch.cat([x, ones], dim=1)
        transformed_points = torch.matmul(self.transformation_matrices, 
                            x.transpose(0, 1)).transpose(1, 2)
        transformed_points = transformed_points[...,0:dims]
        
        # return [n_grids,batch,n_dims]
        return transformed_points
   
    def inverse_transform(self, x):
        '''
        Transforms local coordinates within each feature grid x to 
        global coordinates. Scales local coordinates by a factor
        so as to be consistent with the transform() method, which
        attempts to align feature grids with the guassian density 
        calculated in feature_density_gaussian().Assumes S*R*T order,
        so inverse is T^(-1)*R^T*(1/S)
        
        x: Input coordinates with shape [batch, n_dims]
        returns: local coordinates in a shape [n_grids, batch, n_dims]
        '''

        local_to_global_matrices = torch.linalg.inv(self.transformation_matrices)
       
        batch : int = x.shape[0]
        dims : int = x.shape[1]
        ones = torch.ones([batch, 1], 
            device=x.device,
            dtype=torch.float32)
        
        x = torch.cat([x, ones], dim=1)
        #x = x.unsqueeze(0)
        #x = x.repeat(n_grids, 1, 1)
        
        transformed_points = torch.matmul(local_to_global_matrices,
            x.transpose(0,1)).transpose(1, 2)
        transformed_points = transformed_points[...,0:dims]

        return transformed_points
    
    def feature_density_pre_transformed(self, x):
        transformed_points = x
            
        # get the coeffs of shape [n_grids], then unsqueeze to [1,n_grids] for broadcasting
        coeffs = torch.linalg.det(self.transformation_matrices[:,0:-1,0:-1]).unsqueeze(0) \
            / (2.0*torch.pi)**(x.shape[-1]/2)
        #coeffs = torch.prod(self.grid_scales,dim=1).unsqueeze(0) \
        #    / (2.0*torch.pi)**(x.shape[-1]/2)

        # sum the exp part to [batch,n_grids]
        exps = torch.exp(-0.5 * \
            torch.sum(
                transformed_points.transpose(0,1)**20, 
            dim=-1))
        # Another similar solution, may or may not be more efficient gradient computation 
        #exps = 1 / (1 + torch.sum(local_positions**20,dim=-1))
        
        result = torch.sum(coeffs * exps, dim=-1, keepdim=True)
        return result
    
    def feature_density(self, x):
        # Transform the points to local grid spaces first
        transformed_points = self.transform(x)
        return self.feature_density_pre_transformed(transformed_points)     
            
    def get_volume_extents(self):
        return self.full_shape
    
    def reset_parameters(self):
        with torch.no_grad():
            self.model.apply(weights_init)   

    def grad_at(self, x):
        x.requires_grad_(True)
        y = self(x)

        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y),]

        grad_x = torch.autograd.grad([y], [x],
            grad_outputs=grad_outputs)[0]
        return grad_x

    def min(self):
        return self.volume_min

    def max(self):
        return self.volume_max

    def forward_pre_transformed(self, x):
        out = self.model((x.transpose(1,2).flatten(0,1).transpose(0,1).detach()+1)/2).float()
        return out 

    def forward(self, x):
        x_t = self.transform(x)
        return self.forward_pre_transformed(x_t)

        
