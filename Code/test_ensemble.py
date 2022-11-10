from __future__ import absolute_import, division, print_function
import argparse
import os
from Other.utility_functions import PSNR, tensor_to_cdf, create_path
from Models.models import load_model, sample_grid
from Models.options import load_options
from Datasets.datasets import Dataset
import torch
import copy

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")



def model_reconstruction(models, dataset, options, save_name):
    full_data = torch.zeros_like(dataset.data)
    for i in range(len(models)):        
        model = models[i]
        opt = options[i]
        print(f"Reconstructing for {opt['save_name']}")
        
        extents = opt['extents'].split(',')
        e = [int(i) for i in extents] 
        grid = [e[1]-e[0],e[3]-e[2],e[5]-e[4]]
        
        with torch.no_grad():
            result = sample_grid(model, grid, 1000000)
            
        result = result.to(opt['data_device'])
        result = result.permute(3, 0, 1, 2).unsqueeze(0)
    
        p = PSNR(dataset.data[:,:,e[0]:e[1],e[2]:e[3],e[4]:e[5]], result, in_place=False)
        print(f"Model {opt['save_name']} PSNR: {p : 0.03f}")
        
        full_data[:,:,e[0]:e[1],e[2]:e[3],e[4]:e[5]] = result
        
    create_path(os.path.join(output_folder, "Reconstruction"))
    tensor_to_cdf(full_data, os.path.join(output_folder, "Reconstruction", save_name+".nc"))
    
    p = PSNR(dataset.data, full_data, in_place=True)
    print(f"Total PSNR: {p : 0.03f}")
    
def perform_tests(model, data, tests, opt, save_name):
    if("reconstruction" in tests):
        model_reconstruction(model, data, opt, save_name)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model on some tests')

    parser.add_argument('--load_from',default=None,type=str,help="Model name to load")
    parser.add_argument('--tests_to_run',default=None,type=str,
                        help="A set of tests to run, separated by commas")
    parser.add_argument('--device',default=None,type=str,
                        help="Device to load model to")
    parser.add_argument('--data_device',default=None,type=str,
                        help="Device to load data to")
    args = vars(parser.parse_args())

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..")
    data_folder = os.path.join(project_folder_path, "Data")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")
    
    tests_to_run = args['tests_to_run'].split(',')
    
    # Load the models
    models = []
    options = []
    for fold in os.listdir(save_folder):
        if(args['load_from'] in fold and \
            os.path.exists(os.path.join(save_folder, fold, "model.ckpt.tar")) and \
            os.path.exists(os.path.join(save_folder, fold, "options.json"))):         
            
            print(f"Loding from {fold}")   
            opt = load_options(os.path.join(save_folder, fold))
            opt['device'] = args['device']
            opt['data_device'] = args['data_device']
            model = load_model(opt, args['device']).to(args['device'])
            model.train(False)
            model.eval()
            
            models.append(model)
            options.append(opt)
            
    general_opt = copy.deepcopy(opt)
    general_opt['extents'] = None
    
    # Load the reference data
    data = Dataset(general_opt)
    
    # Perform tests
    perform_tests(models, data, tests_to_run, options, args['load_from'])
    
        
    
        



        

