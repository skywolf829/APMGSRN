import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearLayer(nn.Module):
  def __init__(self, in_features, out_features=256, act_fn=nn.ReLU(), use_norm=False) -> None:
    super(LinearLayer, self).__init__()
    self.layer = nn.ModuleList([ nn.Linear(in_features, out_features) ])
    if use_norm:
      self.layer.append(nn.LayerNorm(in_features))
    if act_fn is not None:
      self.layer.append(act_fn)
    self.layer = nn.Sequential(*self.layer)
    
  def forward(self, x):
    return self.layer(x)

class ReLULayer(nn.Module):
    def __init__(self, in_features, out_features, 
                bias=True, dtype=torch.float32):
        super().__init__()
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, 
            bias=bias, dtype=dtype)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            nn.init.xavier_normal_(self.linear.weight)
            if(self.linear.bias is not None):
                nn.init.zeros_(self.linear.bias)

    def forward(self, input):
        return F.relu(self.linear(input))
    
class LReLULayer(nn.Module):
    def __init__(self, in_features, out_features, 
                 bias=True, dtype=torch.float32):
        super().__init__()
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, 
            bias=bias, dtype=dtype)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.linear.weight)

    def forward(self, input):
        return F.leaky_relu(self.linear(input), 0.2)

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, 
            bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class SnakeAltLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            nn.init.xavier_normal_(self.linear.weight)
        
    def forward(self, input):
        x = self.linear(input)
        return 0.5*x + torch.sin(x)**2

class PositionalEncoding(nn.Module):
    def __init__(self, num_terms:int, n_dims:int):
        super(PositionalEncoding, self).__init__()  
        self.n_dims = n_dims      
        
        self.L = num_terms
        L_terms = torch.arange(0, num_terms, 
            dtype=torch.float32).repeat_interleave(2*n_dims)
        L_terms = torch.pow(2, L_terms) * torch.pi
        
        self.register_buffer("L_terms", L_terms, persistent=False)

    def forward(self, locations):
        repeats = len(locations.shape) * [1]
        repeats[-1] = self.L*2
        locations = locations.repeat(repeats)
        
        locations = locations * self.L_terms# + self.phase_shift
        if(self.n_dims == 2):
            locations[..., 0::4] = torch.sin(locations[..., 0::4])
            locations[..., 1::4] = torch.sin(locations[..., 1::4])
            locations[..., 2::4] = torch.cos(locations[..., 2::4])
            locations[..., 3::4] = torch.cos(locations[..., 3::4])
        else:
            locations[..., 0::6] = torch.sin(locations[..., 0::6])
            locations[..., 1::6] = torch.sin(locations[..., 1::6])
            locations[..., 2::6] = torch.sin(locations[..., 2::6])
            locations[..., 3::6] = torch.cos(locations[..., 3::6])
            locations[..., 4::6] = torch.cos(locations[..., 4::6])
            locations[..., 5::6] = torch.cos(locations[..., 5::6])
        return locations
       