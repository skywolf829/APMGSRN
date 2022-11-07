import torch
import os
import argparse
import json
import time
import subprocess
import shlex
from Other.utility_functions import create_path, nc_to_tensor, tensor_to_cdf

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")


if __name__ == '__main__':
    
    d = nc_to_tensor(os.path.join(data_folder, "Isotropic.nc"))
    d = torch.linalg.norm(d,dim=1).unsqueeze(0)
    print(d.shape)
    tensor_to_cdf(d, os.path.join(data_folder, "IsotropicMag.nc"))

    quit()