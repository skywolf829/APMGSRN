from __future__ import absolute_import, division, print_function
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import os
from math import pi
from Models.options import *
from Models.fVSRN import fVSRN
from Other.utility_functions import create_folder
from Other.utility_functions import make_coord_grid

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
    return fVSRN(opt)

def sample_grid(model, grid, max_points = 100000):
    coord_grid = make_coord_grid(grid, 
        model.opt['device'], flatten=False,
        align_corners=model.opt['align_corners'])
    coord_grid_shape = list(coord_grid.shape)
    coord_grid = coord_grid.view(-1, coord_grid.shape[-1])
    vals = forward_maxpoints(model, coord_grid, max_points = max_points)
    coord_grid_shape[-1] = model.opt['n_outputs']
    vals = vals.reshape(coord_grid_shape)
    return vals

def sample_grad_grid(model, grid, 
    output_dim = 0, max_points=1000):
    
    coord_grid = make_coord_grid(grid, 
        model.opt['device'], flatten=False,
        align_corners=model.opt['align_corners'])
    
    coord_grid_shape = list(coord_grid.shape)
    coord_grid = coord_grid.view(-1, coord_grid.shape[-1]).requires_grad_(True)       

    output_shape = list(coord_grid.shape)
    output_shape[-1] = model.opt['n_dims']
    
    output = torch.empty(output_shape, 
        dtype=torch.float32, device=model.opt['device'], 
        requires_grad=False)

    for start in range(0, coord_grid.shape[0], max_points):
        vals = model(
            coord_grid[start:min(start+max_points, coord_grid.shape[0])])
        grad = torch.autograd.grad(vals[:,output_dim], 
            coord_grid, 
            grad_outputs=torch.ones_like(vals[:,output_dim])
            )[0][start:min(start+max_points, coord_grid.shape[0])]
        
        output[start:min(start+max_points, coord_grid.shape[0])] = grad

    output = output.reshape(coord_grid_shape)
    
    return output

def sample_grid_for_image(model, grid, 
    boundary_scaling = 1.0):
    coord_grid = make_coord_grid(grid, 
        model.opt['device'], flatten=False,
        align_corners=model.opt['align_corners'])
    if(len(coord_grid.shape) == 4):
        coord_grid = coord_grid[:,:,int(coord_grid.shape[2]/2),:]
    
    coord_grid *= boundary_scaling

    coord_grid_shape = list(coord_grid.shape)
    coord_grid = coord_grid.view(-1, coord_grid.shape[-1])
    vals = forward_maxpoints(model, coord_grid)
    coord_grid_shape[-1] = model.opt['n_outputs']
    vals = vals.reshape(coord_grid_shape)
    return vals

def sample_occupancy_grid_for_image(model, grid, opt, boundary_scaling = 1.0):
    coord_grid = make_coord_grid(grid, 
        opt['device'], flatten=False,
        align_corners=opt['align_corners'])
    if(len(coord_grid.shape) == 4):
        coord_grid = coord_grid[:,:,int(coord_grid.shape[2]/2),:]
    
    coord_grid *= boundary_scaling

    coord_grid_shape = list(coord_grid.shape)
    coord_grid = coord_grid.view(-1, coord_grid.shape[-1])
    vals = forward_maxpoints(model, coord_grid)
    coord_grid_shape[-1] = model.opt['n_outputs']
    vals = vals.reshape(coord_grid_shape)
    if(model.opt['loss'] == "l1occupancy"):
        vals = vals[...,-1]
    return vals

def sample_grad_grid_for_image(model, grid, boundary_scaling = 1.0, 
    input_dim = 0, output_dim = 0):

    coord_grid = make_coord_grid(grid, 
        model.opt['device'], flatten=False,
        align_corners=model.opt['align_corners'])      
    if(len(coord_grid.shape) == 4):
        coord_grid = coord_grid[:,:,int(coord_grid.shape[2]/2),:]
    coord_grid *= boundary_scaling
    
    coord_grid_shape = list(coord_grid.shape)
    coord_grid = coord_grid.view(-1, coord_grid.shape[-1]).requires_grad_(True)
    vals = forward_maxpoints(model, coord_grid)    


    grad = torch.autograd.grad(vals[:,output_dim], 
        coord_grid,#[:,input_dim], 
        grad_outputs=torch.ones_like(vals[:, output_dim]),
        allow_unused=True)
    

    grad = grad[0][:,input_dim]
    coord_grid_shape[-1] = 1
    grad = grad.reshape(coord_grid_shape)
    
    return grad

def sample_rect(model, starts, widths, samples):
    positions = []
    for i in range(len(starts)):
        positions.append(
            torch.arange(starts[i], starts[i] + widths[i], widths[i] / samples[i], 
                dtype=torch.float32, device=model.opt['device'])
        )
    grid_to_sample = torch.stack(torch.meshgrid(*positions), dim=-1)
    vals = model.forward(grid_to_sample)
    return vals

def sample_grad_rect(model, starts, widths, samples, input_dim, output_dim):
    positions = []
    for i in range(len(starts)):
        positions.append(
            torch.arange(starts[i], starts[i] + widths[i], widths[i] / samples[i], 
                dtype=torch.float32, device=model.opt['device'])
        )
    grid_to_sample = torch.stack(torch.meshgrid(*positions), dim=-1).requires_grad_(True)
    vals = model.forward(grid_to_sample)
    
    grad = torch.autograd.grad(vals[:,output_dim], 
        grid_to_sample,#[:,input_dim], 
        grad_outputs=torch.ones_like(vals[:, output_dim]),
        allow_unused=True)
    

    grad = grad[0][:,input_dim]
    grid_to_sample[-1] = 1
    grad = grad.reshape(grid_to_sample)

    return vals

def forward_w_grad(model, coords):
    coords = coords.requires_grad_(True)
    output = model(coords)
    return output, coords

def forward_maxpoints(model, coords, max_points=100000):
    output_shape = list(coords.shape)
    output_shape[-1] = model.opt['n_outputs']
    output = torch.empty(output_shape, 
        dtype=torch.float32, device=model.opt['device'])

    for start in range(0, coords.shape[0], max_points):
        output[start:min(start+max_points, coords.shape[0])] = \
            model(coords[start:min(start+max_points, coords.shape[0])])
    return output

