from __future__ import absolute_import, division, print_function
import argparse
import datetime
from Other.utility_functions import str2bool
from Models.models import load_model, create_model, save_model
import torch
import time
import os
from Models.options import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains an implicit model on data.')

    parser.add_argument('--model_name',default=None,type=str,
        help='Saved model to load and convert to torchscript')

    args = vars(parser.parse_args())

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..")
    data_folder = os.path.join(project_folder_path, "Data")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")
    
    opt = load_options(os.path.join(save_folder, args["model_name"]))
    opt["device"] = "cpu"
    opt["save_name"] = args["model_name"]
    
    model = load_model(opt, opt['device'])

        
    model_jit = torch.jit.script(model)
    torch.jit.save(model_jit, 
        os.path.join(save_folder, args["model_name"], "traced_model.pt"))

