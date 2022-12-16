import torch
import os
import argparse
import json
import time
import subprocess
import shlex
from Other.utility_functions import create_path, nc_to_tensor, tensor_to_cdf, make_coord_grid
import h5py
import numpy as np

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

def gaussian_test(x):
    n_grids = 1
    transformation_matrices = torch.tensor(
        [[10, 0, 0, 0],
         [0, 10, 0, 0],
         [0, 0, 10, 0],
         [0, 0, 0, 1]], dtype=torch.float64, device='cuda'
    ).unsqueeze(0).repeat(n_grids, 1, 1)
    
    x = x.unsqueeze(1).repeat(1,n_grids,1)

    local_to_globals = torch.inverse(transformation_matrices)

    grid_centers = local_to_globals[:,0:-1,-1]
    grid_stds = torch.diagonal(local_to_globals, 0, 1, 2)[:,0:-1]


    
    coeffs = 1 / \
        (torch.prod(grid_stds, dim=-1).unsqueeze(0) * \
            ((2 * torch.pi)**(grid_centers.shape[-1]/2.0)))
        
    exps = torch.exp(-0.5 * \
        torch.sum(((x - grid_centers.unsqueeze(0))**2) / \
        ((grid_stds.unsqueeze(0)**2)), dim=-1))
    
    
    return torch.sum(coeffs * exps, dim=-1)
    
if __name__ == '__main__':
    size = 500
    x = make_coord_grid([size,size,size], device='cuda', 
                        flatten=True, align_corners=True)
    gaussian_densities = gaussian_test(x)
    gaussian_densities = gaussian_densities.reshape(size, size, size, 1)
    gaussian_densities = gaussian_densities.permute(3, 0, 1, 2).unsqueeze(0)

    gaussian_densities /= (1/8)*size*size*size
    tensor_to_cdf(gaussian_densities, "guassian_test.nc")
    print(gaussian_densities.sum())
    
    
    quit()