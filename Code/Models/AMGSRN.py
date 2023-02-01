import torch
import torch.nn as nn
import torch.nn.functional as F 
from Models.layers import LReLULayer
from Models.models import sample_grid
from typing import List, Dict

def weights_init(m):
    classname = m.__class__.__name__
    if classname.lower().find('linear') != -1:
        print(f"Found {classname}, initializing to xavier normal")
        nn.init.xavier_normal_(m.weight)
        torch.nn.init.normal_(m.bias, 0, 0.001)
    else:
        print(f"Found {classname}, not initializing")     

@torch.jit.script
def get_transformation_matrices(num_grids:int,grid_translations,
        grid_scales,grid_rotations,device:str):
    '''
    Creates the transformation matrices for the grids
    given the defined translation, scale, and rotation values.
    Computes M = TRS, where T = translation matrix, R = rotation
    matrix, S = scale matrix
    '''
    c = torch.cos(grid_rotations)
    s = torch.sin(grid_rotations)
    m = torch.zeros([
        num_grids,4,4],
        device = device)
    # rotation is applied yaw-pitch-roll (left-to-right) order
    # https://en.wikipedia.org/wiki/Rotation_matrix
    m[:,0,0] = grid_scales[:,0]*c[:,1]*c[:,2]   
    m[:,0,1] = grid_scales[:,1]*(s[:,0]*s[:,1]*c[:,2] - c[:,0]*s[:,2])
    m[:,0,2] = grid_scales[:,2]*(c[:,0]*s[:,1]*c[:,2] + s[:,0]*s[:,2])
    m[:,0,3] = grid_translations[:,0]      
    m[:,1,0] = grid_scales[:,0]*c[:,1]*s[:,2]
    m[:,1,1] = grid_scales[:,1]*(s[:,0]*s[:,1]*s[:,2]+c[:,0]*c[:,2])
    m[:,1,2] = grid_scales[:,2]*(c[:,0]*s[:,1]*s[:,2] - s[:,0]*c[:,2])
    m[:,1,3] = grid_translations[:,1]
    m[:,2,0] = grid_scales[:,0]*-s[:,1]
    m[:,2,1] = grid_scales[:,1]*s[:,0]*c[:,1]
    m[:,2,2] = grid_scales[:,2]*c[:,0]*c[:,1]
    m[:,2,3] = grid_translations[:,2]
    m[:,3,0] = 0    
    m[:,3,1] = 0    
    m[:,3,2] = 0    
    m[:,3,3] = 1    
    return m

@torch.jit.script
def get_inverse_transformation_matrices(num_grids:int,grid_translations,
        grid_scales,grid_rotations,device:str):
    '''
    Creates the inverse transformation matrices for the grids
    defined with the translation, scale, and rotation values.
    Computes M^-1 = (TRS)^-1 = (S^-1)(R^-1)(T^-1) = (1/S)(R^T)(-T), 
    where T = translation matrix, R = rotation
    matrix, S = scale matrix
    '''
    c = torch.cos(grid_rotations)
    s = torch.sin(grid_rotations)
    m = torch.zeros([
        num_grids,4,4],
        device = device)
    # rotation is applied yaw-pitch-roll (left-to-right) order
    # https://en.wikipedia.org/wiki/Rotation_matrix
    m[:,0,0] = (1/grid_scales[:,0])*c[:,1]*c[:,2]     
    m[:,0,1] = (1/grid_scales[:,0])*c[:,1]*s[:,2] 
    m[:,0,2] = (1/grid_scales[:,0])*-s[:,1]
    m[:,0,3] = (-grid_translations[:,0]/grid_scales[:,0])*c[:,1]*c[:,2] + \
        (-grid_translations[:,1]/grid_scales[:,0])*c[:,1]*s[:,2] + \
        (-grid_translations[:,2]/grid_scales[:,0])*-s[:,1]
    m[:,1,0] = (1/grid_scales[:,1])*(s[:,0]*s[:,1]*c[:,2] - c[:,0]*s[:,2])
    m[:,1,1] = (1/grid_scales[:,1])*(s[:,0]*s[:,1]*s[:,2]+c[:,0]*c[:,2])
    m[:,1,2] = (1/grid_scales[:,1])*s[:,0]*c[:,1]
    m[:,1,3] = (-grid_translations[:,0]/grid_scales[:,1])*(s[:,0]*s[:,1]*c[:,2] - c[:,0]*s[:,2]) + \
        (-grid_translations[:,1]/grid_scales[:,1])*(s[:,0]*s[:,1]*s[:,2]+c[:,0]*c[:,2]) + \
        (-grid_translations[:,2]/grid_scales[:,1])*s[:,0]*c[:,1] 
    m[:,2,0] = (1/grid_scales[:,2])*(c[:,0]*s[:,1]*c[:,2] + s[:,0]*s[:,2])  
    m[:,2,1] = (1/grid_scales[:,2])*(c[:,0]*s[:,1]*s[:,2] - s[:,0]*c[:,2])
    m[:,2,2] = (1/grid_scales[:,2])*c[:,0]*c[:,1]
    m[:,2,3] = (-grid_translations[:,0]/grid_scales[:,2])*(c[:,0]*s[:,1]*c[:,2] + s[:,0]*s[:,2]) + \
        (-grid_translations[:,1]/grid_scales[:,2])*(c[:,0]*s[:,1]*s[:,2] - s[:,0]*c[:,2]) + \
        (-grid_translations[:,2]/grid_scales[:,2])*c[:,0]*c[:,1]     
    m[:,3,0] = 0    
    m[:,3,1] = 0    
    m[:,3,2] = 0    
    m[:,3,3] = 1    
    return m
      
class AMG_encoder(nn.Module):
    def __init__(self, n_grids:int, n_features:int,
                 feat_grid_shape:List[int], n_dims:int, device:str):
        super().__init__()
             
        self.register_buffer("DIM_COEFF", 
                torch.tensor([(2.0*torch.pi)**(n_dims/2)]),
                persistent=False)    
        
        self.grid_scales = torch.nn.Parameter(
            torch.ones(
                [n_grids, 3],
                device = device
            ),
            requires_grad=True
        )
        self.grid_translations = torch.nn.Parameter(
            torch.zeros(
                [n_grids, 3],
                device = device
            ),
            requires_grad=True
        )
        self.grid_rotations = torch.nn.Parameter(
            torch.zeros(
                [n_grids, 3],
                device = device
            ),
            requires_grad=True
        )        
        
        self.feature_grids =  torch.nn.parameter.Parameter(
            torch.ones(
                [n_grids, n_features, 
                feat_grid_shape[0], feat_grid_shape[1], feat_grid_shape[2]],
                device = device
            ).uniform_(-0.0001, 0.0001),
            requires_grad=True
        )
    
        self.randomize_grids()
    
    def randomize_grids(self):  
        with torch.no_grad():     
            self.grid_scales.uniform_(1.0,1.2)
            self.grid_translations.uniform_(-0.1, 0.1)
            self.grid_rotations.uniform_(-torch.pi/16, torch.pi/16)

    @torch.jit.export
    def get_transformation_matrices(self):
        c = torch.cos(self.grid_rotations)
        s = torch.sin(self.grid_rotations)
        m = torch.empty([
            self.grid_rotations.shape[0],4,4],
            device = self.grid_rotations.device)
        # rotation is applied yaw-pitch-roll (left-to-right) order
        # https://en.wikipedia.org/wiki/Rotation_matrix
        
        m[:,0,0] = self.grid_scales[:,0]*c[:,1]*c[:,2]   
        m[:,0,1] = self.grid_scales[:,1]*(s[:,0]*s[:,1]*c[:,2] - c[:,0]*s[:,2])
        m[:,0,2] = self.grid_scales[:,2]*(c[:,0]*s[:,1]*c[:,2] + s[:,0]*s[:,2])
        m[:,0,3] = self.grid_translations[:,0]      
        m[:,1,0] = self.grid_scales[:,0]*c[:,1]*s[:,2]
        m[:,1,1] = self.grid_scales[:,1]*(s[:,0]*s[:,1]*s[:,2]+c[:,0]*c[:,2])
        m[:,1,2] = self.grid_scales[:,2]*(c[:,0]*s[:,1]*s[:,2] - s[:,0]*c[:,2])
        m[:,1,3] = self.grid_translations[:,1]
        m[:,2,0] = self.grid_scales[:,0]*-s[:,1]
        m[:,2,1] = self.grid_scales[:,1]*s[:,0]*c[:,1]
        m[:,2,2] = self.grid_scales[:,2]*c[:,0]*c[:,1]
        m[:,2,3] = self.grid_translations[:,2]
        m[:,3,0] = 0    
        m[:,3,1] = 0    
        m[:,3,2] = 0    
        m[:,3,3] = 1    
        return m
  
    @torch.jit.export
    def get_inverse_transformation_matrices(self):
        '''
        Creates the inverse transformation matrices for the grids
        defined with the translation, scale, and rotation values.
        Computes M^-1 = (TRS)^-1 = (S^-1)(R^-1)(T^-1) = (1/S)(R^T)(-T), 
        where T = translation matrix, R = rotation
        matrix, S = scale matrix
        '''
        c = torch.cos(self.grid_rotations)
        s = torch.sin(self.grid_rotations)
        m = torch.zeros([
            self.grid_rotations.shape[0],4,4],
            device = self.grid_rotations.device)
        # rotation is applied yaw-pitch-roll (left-to-right) order
        # https://en.wikipedia.org/wiki/Rotation_matrix
        m[:,0,0] = (1/self.grid_scales[:,0])*c[:,1]*c[:,2]     
        m[:,0,1] = (1/self.grid_scales[:,0])*c[:,1]*s[:,2] 
        m[:,0,2] = (1/self.grid_scales[:,0])*-s[:,1]
        m[:,0,3] = (-self.grid_translations[:,0]/self.grid_scales[:,0])*c[:,1]*c[:,2] + \
            (-self.grid_translations[:,1]/self.grid_scales[:,0])*c[:,1]*s[:,2] + \
            (-self.grid_translations[:,2]/self.grid_scales[:,0])*-s[:,1]
        m[:,1,0] = (1/self.grid_scales[:,1])*(s[:,0]*s[:,1]*c[:,2] - c[:,0]*s[:,2])
        m[:,1,1] = (1/self.grid_scales[:,1])*(s[:,0]*s[:,1]*s[:,2]+c[:,0]*c[:,2])
        m[:,1,2] = (1/self.grid_scales[:,1])*s[:,0]*c[:,1]
        m[:,1,3] = (-self.grid_translations[:,0]/self.grid_scales[:,1])*(s[:,0]*s[:,1]*c[:,2] - c[:,0]*s[:,2]) + \
            (-self.grid_translations[:,1]/self.grid_scales[:,1])*(s[:,0]*s[:,1]*s[:,2]+c[:,0]*c[:,2]) + \
            (-self.grid_translations[:,2]/self.grid_scales[:,1])*s[:,0]*c[:,1] 
        m[:,2,0] = (1/self.grid_scales[:,2])*(c[:,0]*s[:,1]*c[:,2] + s[:,0]*s[:,2])  
        m[:,2,1] = (1/self.grid_scales[:,2])*(c[:,0]*s[:,1]*s[:,2] - s[:,0]*c[:,2])
        m[:,2,2] = (1/self.grid_scales[:,2])*c[:,0]*c[:,1]
        m[:,2,3] = (-self.grid_translations[:,0]/self.grid_scales[:,2])*(c[:,0]*s[:,1]*c[:,2] + s[:,0]*s[:,2]) + \
            (-self.grid_translations[:,1]/self.grid_scales[:,2])*(c[:,0]*s[:,1]*s[:,2] - s[:,0]*c[:,2]) + \
            (-self.grid_translations[:,2]/self.grid_scales[:,2])*c[:,0]*c[:,1]     
        m[:,3,0] = 0    
        m[:,3,1] = 0    
        m[:,3,2] = 0    
        m[:,3,3] = 1    
        return m
  
    @torch.jit.export
    def transform(self, x):
        '''
        Transforms global coordinates x to local coordinates within
        each feature grid, where feature grids are assumed to be on
        the boundary of [-1, 1]^3 in their local coordinate system.
        Scales the grid by a factor to match the gaussian shape
        (see feature_density_gaussian())
        
        x: Input coordinates with shape [batch, 3]
        returns: local coordinates in a shape [n_grids, batch, 3]
        '''

        # x starts [batch,3], this changes it to [n_grids,batch,4]#
        # by appending 1 to the xyz and repeating it n_grids times
        transformed_points = torch.cat(
            [x, torch.ones([x.shape[0], 1], 
            device=x.device,
            dtype=torch.float32)], 
            dim=1).unsqueeze(0).repeat(
                self.grid_scales.shape[0], 1, 1
            )

        
        #transformation_matrices = get_transformation_matrices(
        #    self.opt['n_grids'], self.grid_translations, 
        #    self.grid_scales, self.grid_rotations, 
        #    self.opt['device']
        #    )
        transformation_matrices = self.get_transformation_matrices()
        
        # BMM will result in [n_grids,4,4] x [n_grids,4,batch]
        # which returns [n_grids,4,batch], which is then transposed
        # to [n_grids,batch,4]
        transformed_points = torch.bmm(transformation_matrices, 
                            transformed_points.transpose(-1, -2)).transpose(-1, -2)

        # Finally, only the xyz coordinates are taken
        transformed_points = transformed_points[...,0:3]
                    
        # return [n_grids,batch,3]
        return transformed_points
   
    @torch.jit.export
    def inverse_transform(self, x):
        '''
        Transforms local coordinates within each feature grid x to 
        global coordinates. Scales local coordinates by a factor
        so as to be consistent with the transform() method, which
        attempts to align feature grids with the guassian density 
        calculated in feature_density_gaussian()
        
        x: Input coordinates with shape [batch, 3]
        returns: local coordinates in a shape [n_grids, batch, 3]
        '''
        transformed_points = torch.cat([x, 
            torch.ones(
                [x.shape[0], 1], 
                device=x.device,
                dtype=torch.float32
                )],
            dim=1).unsqueeze(0).repeat(
                self.grid_rotations.shape[0], 1, 1
            )
        
        #local_to_global_matrices = get_inverse_transformation_matrices(
        #    self.opt['n_grids'], self.grid_translations, 
        #    self.grid_scales, self.grid_rotations, 
        #    self.opt['device']
        #    )
        local_to_global_matrices = self.get_inverse_transformation_matrices()
        transformed_points = torch.bmm(local_to_global_matrices,
                                    transformed_points.transpose(-1,-2)).transpose(-1, -2)
        transformed_points = transformed_points[...,0:3].detach().cpu()
        return transformed_points
    
    @torch.jit.export
    def feature_density_gaussian(self, x):
       
        local_positions = self.transform(x).transpose(0,1)

        coeffs = torch.prod(self.grid_scales, dim=-1).unsqueeze(0) / self.DIM_COEFF
        
        exps = torch.exp(-0.5 * \
            torch.sum(
                local_positions**20, 
            dim=-1))
        return torch.sum(coeffs * exps, dim=-1, keepdim=True)
    
    def forward(self, x):
        transformed_points = self.transform(x)       
        
        transformed_points = transformed_points.unsqueeze(1).unsqueeze(1)
        feats = F.grid_sample(self.feature_grids,
                transformed_points.detach(),
                mode='bilinear', align_corners=True,
                padding_mode="zeros")[:,:,0,0,:]
        feats = feats.flatten(0,1).permute(1, 0)
        return feats
      
    
class AMGSRN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.opt = opt

        feat_grid_shape = opt['feature_grid_shape'].split(',')
        feat_grid_shape = [eval(i) for i in feat_grid_shape]
        
        #self.encoder = AMG_encoder(opt)
        self.encoder = torch.jit.script(AMG_encoder(opt['n_grids'], opt['n_features'], 
            [eval(i) for i in opt['feature_grid_shape'].split(',')], opt['n_dims'], 
            opt['device']))
        
        try:
            import tinycudann as tcnn 
            print(f"Using TinyCUDANN (tcnn) since it is installed for performance gains.")
            print(f"WARNING: This model will be incompatible with non-tcnn compatible systems")
            self.decoder = tcnn.Network(
                n_input_dims=opt['n_features']*opt['n_grids'],
                n_output_dims=opt['n_outputs'],
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": opt['nodes_per_layer'],
                    "n_hidden_layers": opt['n_layers'],
                }
            )
        except ImportError:
            print(f"TinyCUDANN (tcnn) not installed: falling back to normal PyTorch")
            self.decoder = nn.ModuleList()
            
            first_layer_input_size = opt['n_features']*opt['n_grids']# + opt['num_positional_encoding_terms']*opt['n_dims']*2
                    
            layer = LReLULayer(first_layer_input_size, 
                                opt['nodes_per_layer'])
            self.decoder.append(layer)
            
            for i in range(opt['n_layers']):
                if i == opt['n_layers'] - 1:
                    layer = nn.Linear(opt['nodes_per_layer'], opt['n_outputs'])
                    self.decoder.append(layer)
                else:
                    layer = LReLULayer(opt['nodes_per_layer'], opt['nodes_per_layer'])
                    self.decoder.append(layer)
            self.decoder = torch.nn.Sequential(*self.decoder)
            
        self.reset_parameters()
        
        
       
    def reset_parameters(self):
        with torch.no_grad():
            feat_grid_shape = self.opt['feature_grid_shape'].split(',')
            feat_grid_shape = [eval(i) for i in feat_grid_shape]
            self.encoder.feature_grids =  torch.nn.parameter.Parameter(
                torch.ones(
                    [self.opt['n_grids'], self.opt['n_features'], 
                    feat_grid_shape[0], feat_grid_shape[1], feat_grid_shape[2]],
                    device = self.opt['device']
                ).uniform_(-0.0001, 0.0001),
                requires_grad=True
            )
            self.decoder.apply(weights_init)   
            
    def get_transformation_matrices(self):        
        return self.encoder.get_transformation_matrices()

    def feature_density_gaussian(self, x):
        return self.encoder.feature_density_gaussian(x)

    def transform(self, x):
        return self.encoder.transform(x)
    
    def inverse_transform(self, x):
        return self.encoder.inverse_transform(x)
    
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
                   
    def forward(self, x):        
        feats = self.encoder(x)        
        y = self.decoder(feats).float()
        
        return y

        