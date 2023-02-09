from __future__ import absolute_import, division, print_function
import torch
import os
import argparse
import timeit
from Models.options import load_options
from Models.models import load_model
import time

def inference_speed_evaluation(model, input_batch, num_iters):

    for _ in range(num_iters):
        model_output = model(input_batch)


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True    
    torch.backends.cuda.benchmark = True
    parser = argparse.ArgumentParser(description='Inference speed test with different batch sizes')
    parser.add_argument('--load_from',default=None,type=str,help="Model name to load")
    parser.add_argument('--device',default="cpu",type=str,help="Device to use")

    args = vars(parser.parse_args())

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..")
    data_folder = os.path.join(project_folder_path, "Data")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")
    
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Load the model
    opt = load_options(os.path.join(save_folder, args['load_from']))
    opt['device'] = args['device']
    opt['data_device'] = args['device']
    model = load_model(opt, "cpu").to(opt['device']).eval()
    
    # Perform tests
    with torch.no_grad():
        
    
        input_batch = torch.rand([1000000, 3], dtype=torch.float32, device=args['device'])
        torch.cuda.synchronize()
        start_time = time.time()
        inference_speed_evaluation(model,input_batch,1000)        
        torch.cuda.synchronize()
        end_time = time.time()
        print(f"Total time: {end_time - start_time}")
        