from __future__ import absolute_import, division, print_function
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import os
from math import pi
from Models.options import *
from Other.utility_functions import create_folder, make_coord_grid
import math

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..", "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

def next_highest_multiple(num:int, base:int):
    return base*int(math.ceil(max(1, num/base)))

def convert_tcnn_to_pytorch(ckpt, opt):
    base = 16
    requires_padding = False
    if(opt['model'] == "fVSRN"):
        input_dim = opt['n_features']+opt['num_positional_encoding_terms']*opt['n_dims']*2
        input_dim_padded = next_highest_multiple(input_dim, base)
        if(input_dim != input_dim_padded):
            requires_padding = True
        
        output_dim = opt['n_outputs']
        output_dim_padded = next_highest_multiple(output_dim, base)
        
    elif(opt['model'] == "APMGSRN" or opt['model'] == "AMGSRN"):
        input_dim = opt['n_features']*opt['n_grids']
        input_dim_padded = next_highest_multiple(input_dim, base)
        if(input_dim != input_dim_padded):
            requires_padding = True
        
        output_dim = opt['n_outputs']
        output_dim_padded = next_highest_multiple(output_dim, base)
    
    else:
        #print(f"Currently we do not support converting model type {opt['model']} to pure PyTorch.")
        quit()
    opt['requires_padded_feats'] = requires_padding  
        
    layer_weight_shape = []
    
    first_layer_shape = [opt['nodes_per_layer'], input_dim_padded]
    layer_weight_shape.append(first_layer_shape)
    
    for i in range(opt['n_layers']-1):
        layer_shape = [opt['nodes_per_layer'], opt['nodes_per_layer']]
        layer_weight_shape.append(layer_shape)
        
    last_layer_shape = [output_dim_padded,opt['nodes_per_layer']]
    layer_weight_shape.append(last_layer_shape)
    
    weights = ckpt['state_dict']['decoder.params']
    new_weights = []
    current_weight_index = 0
    for i in range(len(layer_weight_shape)):
        #print(f"Layer {i}: {layer_weight_shape[i]}")
        
        weight_shape_this_layer = layer_weight_shape[i]        
        num_weights_this_layer = weight_shape_this_layer[0]*weight_shape_this_layer[1]
        
        if(current_weight_index+num_weights_this_layer > weights.shape[0]):
            #print(f"Number of expected weights {current_weight_index+num_weights_this_layer} is larger than the number of weights saved {weights.shape}.")
            quit()
        this_layer_weights = weights[current_weight_index:current_weight_index+num_weights_this_layer].clone()
        this_layer_weights = this_layer_weights.reshape(weight_shape_this_layer)
        new_weights.append(this_layer_weights)
        current_weight_index += num_weights_this_layer
        
    del ckpt['state_dict']['decoder.params']
    for i in range(len(new_weights)):
        if(i == len(new_weights) - 1):
            # In the last layer, we can actually prune the unused weights when
            # moving back to PyTorch
            name = f"decoder.{i}.weight"
            ckpt['state_dict'][name] = new_weights[i][0:1]
        else:
            # The first layer is not pruned, even if it was padded, as TCNN learned
            # to use those padded weights as a bias essentially
            # All other layers are not changed            
            name = f"decoder.{i}.linear.weight"
            ckpt['state_dict'][name] = new_weights[i]
    
    return ckpt

def save_model(model,opt):
    folder = create_folder(save_folder, opt["save_name"])
    path_to_save = os.path.join(save_folder, folder)
    
    torch.save({'state_dict': model.state_dict()}, 
        os.path.join(path_to_save, "model.ckpt.tar"),
        pickle_protocol=4
    )
    save_options(opt, path_to_save)

def load_model(opt, device):
    path_to_load = os.path.join(save_folder, opt["save_name"])
    
    try:
        import tinycudann
        tcnn_installed = True
    except ImportError:
        tcnn_installed = False

    if(not opt['ensemble']):
        ckpt = torch.load(os.path.join(path_to_load, 'model.ckpt.tar'), 
            map_location = device)   

        if('decoder.params' in ckpt['state_dict']):
            if(tcnn_installed):
                print(f"Model was trained with TCNN and TCNN is available.")
                model = create_model(opt)
                model.load_state_dict(ckpt['state_dict'])
            else:
                print(f"Model was trained with TCNN and TCNN is not available. Converting to PyTorch linear layers.")
                new_ckpt = convert_tcnn_to_pytorch(ckpt, opt)
                model = create_model(opt)
                model.load_state_dict(new_ckpt['state_dict'])
        else:
            if(tcnn_installed):
                print(f"Model was trained without TCNN and TCNN is available. Keeping model in pure PyTorch")
                opt['use_tcnn_if_available'] = False
            else:
                print(f"Model was trained without TCNN and is being loaded without TCNN.")
        
            model = create_model(opt)
            model.load_state_dict(ckpt['state_dict'])
    else:
        model = create_model(opt)
    return model

def create_model(opt):

    if(opt['ensemble']):
        from Models.ensemble_SRN import Ensemble_SRN
        return Ensemble_SRN(opt)
    else:
        if(opt['model'] == "fVSRN"):
            from Models.fVSRN import fVSRN_NGP
            return fVSRN_NGP(opt)
        elif(opt['model'] == "APMGSRN" or opt['model'] == "AMGSRN"):
            from Models.APMGSRN import APMGSRN
            return APMGSRN(opt)
        elif(opt['model'] == "NGP"):
            from Models.NGP import NGP
            return NGP(opt)
        elif(opt['model'] == "NGP_TCNN"):        
            from Models.NGP import NGP_TCNN
            return NGP_TCNN(opt)
   

def sample_grid(model, grid, align_corners:bool = False,
                device:str="cuda", data_device:str="cuda", max_points:int = 100000):
    coord_grid = make_coord_grid(grid, 
        data_device, flatten=False,
        align_corners=align_corners)
    coord_grid_shape = list(coord_grid.shape)
    coord_grid = coord_grid.view(-1, coord_grid.shape[-1])
    vals = forward_maxpoints(model, coord_grid, 
                             max_points = max_points,
                             data_device=data_device,
                             device=device
                             )
    coord_grid_shape[-1] = -1
    vals = vals.reshape(coord_grid_shape)
    return vals

def forward_maxpoints(model, coords, out_dim=1, max_points=100000, 
                      data_device="cuda", device="cuda"):
    output_shape = list(coords.shape)
    output_shape[-1] = out_dim
    output = torch.empty(output_shape, 
        dtype=torch.float32, 
        device=data_device)
    
    for start in range(0, coords.shape[0], max_points):
        output[start:min(start+max_points, coords.shape[0])] = \
            model(coords[start:min(start+max_points, coords.shape[0])].to(device)).to(data_device)
    return output

