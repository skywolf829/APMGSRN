from __future__ import absolute_import, division, print_function
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import os
from math import pi
from Models.options import *

from Other.utility_functions import create_folder, make_coord_grid

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..", "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

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
    model = create_model(opt)

    ckpt = torch.load(os.path.join(path_to_load, 'model.ckpt.tar'), 
        map_location = device)
    
    model.load_state_dict(ckpt['state_dict'])

    return model

def create_model(opt):
    from Models.fVSRN import fVSRN
    from Models.afVSRN import afVSRN
    from Models.AMGSRN import AMGSRN
    from Models.SigmoidNet import SigmoidNet
    from Models.ExpNet import ExpNet
    if(opt['model'] == "fVSRN"):
        return fVSRN(opt)
    elif(opt['model'] == "afVSRN"):
        return afVSRN(opt)
    elif(opt['model'] == "AMGSRN"):
        return AMGSRN(opt['n_grids']. opt['n_features'], 
        [eval(i) for i in opt['feature_grid_shape'].split(",")], 
        opt['n_dims'], opt['n_outputs'], opt['nodes_per_layer'], 
        opt['n_layers'], opt['device'])
    elif(opt['model'] == "AMRSRN_TCNN"):
        from Models.AMRSRN_TCNN import AMRSRN_TCNN
        return AMRSRN_TCNN(opt)
    elif(opt['model'] == "NGP"):
        from Models.NGP import NGP
        return NGP(opt)
    elif(opt['model'] == "NGP_TCNN"):        
        from Models.NGP import NGP_TCNN
        return NGP_TCNN(opt)
    elif(opt['model'] == "SigmoidNet"):
        return SigmoidNet(opt)
    elif(opt['model'] == "ExpNet"):
        return ExpNet(opt)

def sample_grid(model, grid, device="cuda", data_device="cuda", max_points = 100000):
    coord_grid = make_coord_grid(grid, 
        data_device, flatten=False,
        align_corners=model.opt['align_corners'])
    coord_grid_shape = list(coord_grid.shape)
    coord_grid = coord_grid.view(-1, coord_grid.shape[-1])
    vals = forward_maxpoints(model, coord_grid, 
                             max_points = max_points,
                             data_device=data_device,
                             device=device
                             )
    coord_grid_shape[-1] = model.opt['n_outputs']
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
            model(coords[start:min(start+max_points, coords.shape[0])].to(device))
    return output

