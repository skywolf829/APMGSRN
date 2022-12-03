import torch
import os
import argparse
import json
import time
import subprocess
import shlex
from Other.utility_functions import create_path, nc_to_tensor, tensor_to_cdf
import h5py
import numpy as np

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")


if __name__ == '__main__':
    
    d, full_shape = nc_to_tensor(os.path.join(data_folder, "Supernova.nc"))
    new_d = torch.zeros([1, 1, 432*2, 432*2, 432*2], dtype=torch.float32)
    new_d[:,:,432:, 0:432, 0:432] = d
    #new_d[:,:,432:, 0:432, 0:432] = d
    new_d = new_d[:,:,::2,::2,::2]
    tensor_to_cdf(new_d, os.path.join(data_folder, "Supernova_test.nc"))
    quit()