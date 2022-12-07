from __future__ import absolute_import, division, print_function
import argparse
from random import gauss
from Datasets.datasets import Dataset
import datetime
from Other.utility_functions import str2bool
from Models.models import load_model, create_model, save_model
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import os
from Models.options import *
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from Models.losses import *
import shutil
from Models.models import sample_grid_for_image
from Other.utility_functions import make_coord_grid, create_path
import numpy as np

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

def log_to_writer(iteration, losses, writer, opt):
    with torch.no_grad():   
        print_str = f"Iteration {iteration}/{opt['iterations']}, "
        for key in losses.keys():    
            print_str = print_str + str(key) + f": {losses[key].item() : 0.05f} " 
            writer.add_scalar(str(key), losses[key].item(), iteration)
        print(print_str)
        if("cuda" in opt['device']):
            GBytes = (torch.cuda.max_memory_allocated(device=opt['device']) \
                / (1024**3))
            writer.add_scalar('GPU memory (GB)', GBytes, iteration)

def log_image(model, grid_to_sample, writer, iteration, dataset):
    with torch.no_grad():
        img = sample_grid_for_image(model, grid_to_sample)
        writer.add_image('Reconstruction', img.clamp(0, 1), 
            iteration, dataformats='HWC')

def log_grad_image(model, grid_to_sample, writer, iteration):
    grad_img = model.sample_grad_grid_for_image(grid_to_sample)
    for output_index in range(len(grad_img)):
        for input_index in range(grad_img[output_index].shape[-1]):
            grad_img[output_index][...,input_index] -= \
                grad_img[output_index][...,input_index].min()
            grad_img[output_index][...,input_index] /= \
                grad_img[output_index][...,input_index].max()

            writer.add_image('Gradient_outputdim'+str(output_index)+\
                "_wrt_inpudim_"+str(input_index), 
                grad_img[output_index][...,input_index:input_index+1].clamp(0, 1), 
                iteration, dataformats='HWC')

def logging(writer, iteration, losses, model, opt, grid_to_sample, dataset):
    if(iteration % opt['log_every'] == 0):
        log_to_writer(iteration, losses, writer, opt)
    if(iteration % 50 == 0 and "AMRSRN" in opt['model']):
        log_feature_points(model, dataset, opt, iteration)

def log_feature_points(model, dataset, opt, iteration):
    feat_grid_shape = opt['feature_grid_shape'].split(',')
    feat_grid_shape = [eval(i) for i in feat_grid_shape]
    
    global_points = make_coord_grid(feat_grid_shape, opt['device'], 
                    flatten=True, align_corners=True)
    transformed_points = torch.cat([global_points, torch.ones(
        [global_points.shape[0], 1], 
        device=opt['device'],
        dtype=torch.float32)], 
        dim=1)
    transformed_points = transformed_points.unsqueeze(0).expand(
        opt['n_grids'], transformed_points.shape[0], transformed_points.shape[1])
    local_to_global_matrices = torch.inverse(model.get_transformation_matrices())
    
    transformed_points = torch.bmm(local_to_global_matrices,
                                   transformed_points.transpose(-1,-2)).transpose(-1, -2)
    transformed_points = transformed_points[...,0:3].detach().cpu()
    transformed_points[...,0] += 1
    transformed_points[...,1] += 1
    transformed_points[...,2] += 1
    transformed_points[...,0] *= 0.5 * dataset.data.shape[2]
    transformed_points[...,1] *= 0.5 * dataset.data.shape[3]
    transformed_points[...,2] *= 0.5 * dataset.data.shape[4]
    ids = torch.arange(transformed_points.shape[0])
    ids = ids.unsqueeze(1).unsqueeze(1)
    ids = ids.repeat([1, transformed_points.shape[1], 1])
    transformed_points = torch.cat((transformed_points, ids), dim=2)
    transformed_points = transformed_points.flatten(0,1).numpy()
    
    create_path(os.path.join(output_folder, "FeatureLocations", opt['save_name']))
    np.savetxt(os.path.join(output_folder, "FeatureLocations", 
        opt['save_name'], opt['save_name']+"_"+str(iteration)+".csv"),
        transformed_points, delimiter=",", header="x,y,z,id")

def train( model, dataset, opt):
      
    model = model.to(opt['device'])        
    print("Training on %s" % (opt["device"]), 
        os.path.join(save_folder, opt["save_name"]))
    if("AMRSRN" in opt['model']):
        optimizer = optim.Adam([
            {
            "params": [model.grid_scales], "lr": opt["lr"]*1
            },
            {
            "params": [model.grid_translations], "lr": opt["lr"]*0.01
            },
            {
            "params": [model.feature_grids], "lr": opt["lr"]
            },
            {
            "params": model.decoder.parameters(), "lr": opt["lr"]
            }
        ], betas=[opt['beta_1'], opt['beta_2']]) 
    else:
        optimizer = optim.Adam(model.parameters(), lr=opt["lr"], 
        betas=[opt['beta_1'], opt['beta_2']]) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
        step_size=(opt['iterations']*3)//4, gamma=0.1)

    if(os.path.exists(os.path.join(project_folder_path, "tensorboard", opt['save_name']))):
        shutil.rmtree(os.path.join(project_folder_path, "tensorboard", opt['save_name']))
        
    writer = SummaryWriter(os.path.join('tensorboard',opt['save_name']))
    gt_img = dataset.get_2D_slice()
    writer.add_image("Ground Truth", gt_img, 0, dataformats="CHW")
    
    model.train(True)

    loss_func = get_loss_func(opt)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=8,
            active=1,
            repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_modules=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            os.path.join('tensorboard',opt['save_name'])),
    with_stack=True) as profiler:
        for iteration in range(0, opt['iterations']):
            opt['iteration_number'] = iteration
            optimizer.zero_grad()
            
            x, y = dataset.get_random_points(opt['points_per_iteration'],
                    device=opt['device'])
            
            model_output = model(x, detach_features=(iteration/10000)%2 == 1)
            loss = loss_func(y, model_output)

            loss.backward()                   
            
            optimizer.step()
            scheduler.step()        
            profiler.step()
            if("AMRSRN" in opt['model']):
                with torch.no_grad():
                    model.grid_scales.clamp_(1, 32)
                    max_deviation = model.grid_scales-1
                    model.grid_translations.clamp_(-max_deviation, max_deviation)

            logging(writer, iteration, {"Fitting loss": loss}, 
                model, opt, dataset.data.shape[2:], dataset)
            
    writer.close()
    save_model(model, opt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains an implicit model on data.')

    parser.add_argument('--n_dims',default=None,type=int,
        help='Number of dimensions in the data')
    parser.add_argument('--n_outputs',default=None,type=int,
        help='Number of output channels for the data (ex. 1 for scalar field, 3 for image or vector field)')
    parser.add_argument('--feature_grid_shape',default=None,type=str,
        help='Resolution for feature grid')
    parser.add_argument('--n_features',default=None,type=int,
        help='Number of features in the feature grid')       
    parser.add_argument('--n_grids',default=None,type=int,
        help='Number of grids for AMRSRN')
    parser.add_argument('--num_positional_encoding_terms',default=None,type=int,
        help='Number of positional encoding terms')   
    parser.add_argument('--extents',default=None,type=str,
        help='Spatial extents to use for this model from the data')   
    parser.add_argument('--use_global_position',default=None,type=str2bool,
        help='For the fourier featuers, whether to use the global position or local.')   
    

    parser.add_argument('--data',default=None,type=str,
        help='Data file name')
    parser.add_argument('--model',default=None,type=str,
        help='The model architecture to use')
    parser.add_argument('--save_name',default=None,type=str,
        help='Save name for the model')
    parser.add_argument('--align_corners',default=None,type=str2bool,
        help='Aligns corners in implicit model.')    
    parser.add_argument('--n_layers',default=None,type=int,
        help='Number of layers in the model')
    parser.add_argument('--nodes_per_layer',default=None,type=int,
        help='Nodes per layer in the model')    
    parser.add_argument('--interpolate',default=None,type=str2bool,
        help='Whether or not to use interpolation during training')    
    
    parser.add_argument('--iters_to_train_new_layer',default=None,type=int,
        help='Number of iterations to fine tune a new layer')    
    parser.add_argument('--iters_since_new_layer',default=None,type=int,
        help='To track the number of iterations since a new layer was added')    
    
    
    parser.add_argument('--device',default=None,type=str,
        help='Which device to train on')
    parser.add_argument('--data_device',default=None,type=str,
        help='Which device to keep the data on')

    parser.add_argument('--iterations',default=None, type=int,
        help='Number of iterations to train')
    parser.add_argument('--points_per_iteration',default=None, type=int,
        help='Number of points to sample per training loop update')
    parser.add_argument('--lr',default=None, type=float,
        help='Learning rate for the adam optimizer')
    parser.add_argument('--beta_1',default=None, type=float,
        help='Beta1 for the adam optimizer')
    parser.add_argument('--beta_2',default=None, type=float,
        help='Beta2 for the adam optimizer')

    parser.add_argument('--iteration_number',default=None, type=int,
        help="Not used.")
    parser.add_argument('--save_every',default=None, type=int,
        help='How often to save the model')
    parser.add_argument('--log_every',default=None, type=int,
        help='How often to log the loss')
    parser.add_argument('--log_image_every',default=None, type=int,
        help='How often to log the image')
    parser.add_argument('--load_from',default=None, type=str,
        help='Model to load to start training from')
    parser.add_argument('--log_image',default=None, type=str2bool,
        help='Whether or not to log an image. Slows down training.')


    args = vars(parser.parse_args())

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..")
    data_folder = os.path.join(project_folder_path, "Data")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    torch.manual_seed(42)

    if(args['load_from'] is None):
        # Init models
        model = None
        opt = Options.get_default()

        # Read arguments and update our options
        for k in args.keys():
            if args[k] is not None:
                opt[k] = args[k]

        dataset = Dataset(opt)
        model = create_model(opt)
    else:        
        opt = load_options(os.path.join(save_folder, args["load_from"]))
        opt["device"] = args["device"]
        opt["save_name"] = args["load_from"]
        for k in args.keys():
            if args[k] is not None:
                opt[k] = args[k]
        dataset = Dataset(opt)
        model = load_model(opt, opt['device'])

    now = datetime.datetime.now()
    start_time = time.time()
    
    train(model, dataset, opt)
        
    opt['iteration_number'] = 0
    save_model(model, opt)
    

