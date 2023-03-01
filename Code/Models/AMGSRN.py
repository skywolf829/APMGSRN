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

class AMG_encoder(nn.Module):
    def __init__(self, n_grids:int, n_features:int,
                 feat_grid_shape:List[int], n_dims:int):
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
    
        self.randomize_grids()
    
    def get_transform_parameters(self) -> List[Dict[str, torch.Tensor]]:
        return [{"params": self.transformation_matrices}]
        
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
         
class AMGSRN(nn.Module):
    def __init__(self, n_grids: int, n_features: int, 
        feature_grid_shape: List[int], n_dims : int, 
        n_outputs: int, nodes_per_layer: int, n_layers: int, 
        use_tcnn:bool,use_bias:bool,requires_padded_feats:bool,
        data_min:float, data_max:float):
        super().__init__()
        
        self.n_grids : int = n_grids
        self.n_features : int = n_features
        self.feature_grid_shape : List[int] = feature_grid_shape
        self.n_dims : int = n_dims
        self.n_outputs : int = n_outputs
        self.nodes_per_layer : int = nodes_per_layer
        self.n_layers : int = n_layers
        self.requires_padded_feats : bool = requires_padded_feats
        self.padding_size : int = 0
        if(requires_padded_feats):
            self.padding_size : int = 16*int(math.ceil(max(1, (n_grids*n_features)/16))) - n_grids*n_features
            
        self.encoder = AMG_encoder(n_grids, n_features, 
            feature_grid_shape, n_dims)
        
        def init_decoder_tcnn():
            import tinycudann as tcnn 
            input_size:int = n_features*n_grids
            if(requires_padded_feats):
                input_size = n_features*n_grids + self.padding_size
                
            decoder = tcnn.Network(
                n_input_dims=input_size,
                n_output_dims=n_outputs,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": nodes_per_layer,
                    "n_hidden_layers": n_layers,
                }
            )
                    
            return decoder
        
        def init_decoder_pytorch():
            decoder = nn.ModuleList()   
            
            first_layer_input_size:int = n_features*n_grids
            if(requires_padded_feats):
                first_layer_input_size = n_features*n_grids + self.padding_size
                                           
            layer = ReLULayer(first_layer_input_size, 
                nodes_per_layer, bias=use_bias)
            decoder.append(layer)
            
            for i in range(n_layers):
                if i == n_layers - 1:
                    layer = nn.Linear(nodes_per_layer, n_outputs, bias=use_bias)
                    decoder.append(layer)
                else:
                    layer = ReLULayer(nodes_per_layer, nodes_per_layer, bias=use_bias)
                    decoder.append(layer)
            decoder = torch.nn.Sequential(*decoder)
            return decoder
             
        if(use_tcnn):
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
            torch.tensor([data_min], requires_grad=False, dtype=torch.float32),
            persistent=False
        )
        self.register_buffer(
            "volume_max",
            torch.tensor([data_max], requires_grad=False, dtype=torch.float32),
            persistent=False
        )

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
    
    '''
    def precodition_grids(self, dataset, writer, logging):
        
        # First, train the params with fixed grids
        self.encoder.uniform_grids()
        self.encoder.feature_grids.requires_grad_(True)
        self.decoder.requires_grad_(True)
        self.encoder.grid_scales.requires_grad_(False)
        self.encoder.grid_translations.requires_grad_(False)
        param_optimizer = torch.optim.Adam([
            {"params": [self.encoder.feature_grids], "lr": 0.03},
            {"params": self.decoder.parameters(), "lr": 0.03}
        ], betas=[self.opt['beta_1'], self.opt['beta_2']], eps = 10e-15)        
        param_scheduler = torch.optim.lr_scheduler.StepLR(param_optimizer, 
                step_size=5000, gamma=0.1)
        for iteration in range(10000):
            param_optimizer.zero_grad()
            
            x,y = dataset.get_random_points(self.opt['points_per_iteration'])
            x = x.to(self.opt['device'])
            y = y.to(self.opt['device'])
            
            model_output = self(x)
            loss = F.mse_loss(model_output, y, reduction='none')
            loss = loss.sum(dim=1, keepdim=True)
            loss.mean().backward()
            
            param_optimizer.step()
            param_scheduler.step()                 
            if(self.opt['log_every'] != 0):
                logging(writer, iteration, 
                    {"Preconditioning loss": loss}, 
                    self, self.opt, dataset.data.shape[2:], dataset, 
                    preconditioning="model")

        # Second, train the density to match the current error
        # First, create map of current error
        with torch.no_grad():
            grid = list(dataset.data.shape[2:])
            error = sample_grid(self, grid, max_points=1000000,
                                device=self.opt['device'],
                                data_device=self.opt['data_device'])
            error = error.to(self.opt['data_device'])
            error = error.permute(3, 0, 1, 2).unsqueeze(0)
            error -= dataset.data
            # Use squared error
            error **= 2
            # Add all dims together
            error = torch.sum(error, dim=1, keepdim=True)      
            
            # Normalize by sum
            error /= error.sum() 
            
        self.encoder.randomize_grids()
        self.encoder.feature_grids.requires_grad_(False)
        self.decoder.requires_grad_(False)
        self.encoder.grid_scales.requires_grad_(True)
        self.encoder.grid_translations.requires_grad_(True)
        grid_optimizer = torch.optim.Adam([
            {"params": [self.encoder.grid_translations, 
                        self.encoder.grid_scales], "lr": 0.001}
        ], betas=[self.opt['beta_1'], self.opt['beta_2']], eps = 10e-15)        
        grid_scheduler = torch.optim.lr_scheduler.StepLR(grid_optimizer, 
                step_size=9000, gamma=0.1)
        for iteration in range(10000):
            grid_optimizer.zero_grad()
            
            x = torch.rand([1, 1, 1, 10000, self.opt['n_dims']], 
                device=self.opt['data_device']) * 2 - 1            
            y = F.grid_sample(error,
                x, mode='bilinear', 
                align_corners=self.opt['align_corners'])
            x = x.squeeze()
            y = y.squeeze()
            if(len(y.shape) == 1):
                y = y.unsqueeze(0)                
            y = y.permute(1,0)
            
            density = self.encoder.feature_density_gaussian(x)       
            density /= density.sum().detach()     
            
            target = torch.exp(torch.log(density+1e-16) / \
                (y/y.mean()))
            target /= target.sum()
                
            density_loss = F.kl_div(
                torch.log(density+1e-16), 
                    torch.log(target.detach()+1e-16), 
                    reduction='none', 
                    log_target=True)
            density_loss.mean().backward()
            
            grid_optimizer.step()
            grid_scheduler.step()                 
            if(self.opt['log_every'] != 0):
                logging(writer, iteration, 
                    {"Grid fitting loss": density_loss}, 
                    self, self.opt, dataset.data.shape[2:], dataset, 
                    preconditioning="grid")
    '''

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
        y = self.decoder(feats).float()
        y = y * (self.volume_max - self.volume_min + 1e-8) + self.volume_min        
        return y

    def forward(self, x):        
        feats = self.encoder(x)    
        if(self.requires_padded_feats):
            feats = F.pad(feats, (0, self.padding_size), value=1.0) 
        y = self.decoder(feats).float()
        y = y * (self.volume_max - self.volume_min) + self.volume_min
        return y

        