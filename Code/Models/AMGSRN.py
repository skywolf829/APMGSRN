import torch
import torch.nn as nn
import torch.nn.functional as F 
from Models.layers import LReLULayer
from Models.models import sample_grid
 
def weights_init(m):
    classname = m.__class__.__name__
    if classname.lower().find('linear') != -1:
        print(f"Found {classname}, initializing to xavier normal")
        nn.init.xavier_normal_(m.weight)
        torch.nn.init.normal_(m.bias, 0, 0.001)
    else:
        print(f"Found {classname}, not initializing")     
 
         
class AMG_encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        feat_grid_shape = opt['feature_grid_shape'].split(',')
        feat_grid_shape = [eval(i) for i in feat_grid_shape]
    
        self.register_buffer("ROOT_TWO", 
                torch.tensor([2.0 ** 0.5]),
                persistent=False)            
        self.register_buffer("FLAT_TOP_GAUSSIAN_EXP", 
                torch.tensor([2.0 * 10.0]),
                persistent=False)
        self.register_buffer("DIM_COEFF", 
                torch.tensor([(2.0 * torch.pi) **(self.opt['n_dims']/2)]),
                persistent=False)
        self.register_buffer("GRID_SCALING", 
                torch.tensor([1.48]),
                persistent=False)
        self.register_buffer("INV_GRID_SCALING", 
                torch.tensor([1/1.48]),
                persistent=False)
        
        self.randomize_grids()        
        
        self.feature_grids =  torch.nn.parameter.Parameter(
            torch.ones(
                [self.opt['n_grids'], self.opt['n_features'], 
                feat_grid_shape[0], feat_grid_shape[1], feat_grid_shape[2]],
                device = opt['device']
            ).uniform_(-0.0001, 0.0001),
            requires_grad=True
        )
    
    def uniform_grids(self):
        init_scales = torch.ones(
                [self.opt['n_grids'], 3],
                device = self.opt['device']
            ) * (1.48)

        init_translations = torch.zeros(
                [self.opt['n_grids'], 3],
                device = self.opt['device']
            )
    
        self.grid_scales = torch.nn.Parameter(
            init_scales,
            requires_grad=True
        )
        self.grid_translations = torch.nn.Parameter(
            init_translations,
            requires_grad=True
        )
    
    def randomize_grids(self):
        init_scales = torch.ones(
                [self.opt['n_grids'], 3],
                device = self.opt['device']
            ).uniform_(1.0,1.25) * 1.48
        init_translations = torch.zeros(
                [self.opt['n_grids'], 3],
                device = self.opt['device']
            ).uniform_(-1, 1) * (init_scales-1)
        with torch.no_grad():
            self.grid_scales = torch.nn.Parameter(
                init_scales,
                requires_grad=True
            )
            self.grid_translations = torch.nn.Parameter(
                init_translations,
                requires_grad=True
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
        '''
        Transforms global coordinates x to local coordinates within
        each feature grid, where feature grids are assumed to be on
        the boundary of [-1, 1]^3 in their local coordinate system.
        Scales the grid by a factor to match the gaussian shape
        (see feature_density_gaussian())
        
        x: Input coordinates with shape [batch, 3]
        returns: local coordinates in a shape [batch, n_grids, 3]
        '''
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
        return transformed_points * self.INV_GRID_SCALING
   
    def inverse_transform(self, x):
        '''
        Transforms local coordinates within each feature grid x to 
        global coordinates. Scales local coordinates by a factor
        so as to be consistent with the transform() method, which
        attempts to align feature grids with the guassian density 
        calculated in feature_density_gaussian()
        
        x: Input coordinates with shape [batch, 3]
        returns: local coordinates in a shape [batch, n_grids, 3]
        '''
        transformed_points = torch.cat([x * self.GRID_SCALING, 
            torch.ones(
                [x.shape[0], 1], 
                device=self.opt['device'],
                dtype=torch.float32
                )],
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
        
        self.encoder = AMG_encoder(opt)
        
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

        