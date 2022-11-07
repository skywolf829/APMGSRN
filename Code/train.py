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

def logging(writer, iteration, losses, opt, grid_to_sample, dataset):
    if(iteration % opt['log_every'] == 0):
        log_to_writer(iteration, losses, writer, opt)
    if(opt['log_image'] and iteration % opt['log_image_every'] == 0):
        log_image(model, grid_to_sample, writer, iteration, dataset)
                    
def train(rank, model, dataset, opt):
      
    model = model.to(opt['device'])        
    print("Training on %s" % (opt["device"]), 
        os.path.join(save_folder, opt["save_name"]))

    for name, param in model.named_parameters(): 
        if param.requires_grad: 
            print(name)

    optimizer = optim.Adam(model.parameters(), lr=opt["lr"],
        betas=[opt['beta_1'], opt['beta_2']]) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
        step_size=opt['iterations']//3, gamma=0.1)

    if(os.path.exists(os.path.join(project_folder_path, "tensorboard", opt['save_name']))):
        shutil.rmtree(os.path.join(project_folder_path, "tensorboard", opt['save_name']))
        
    writer = SummaryWriter(os.path.join('tensorboard',opt['save_name']))
    gt_img = dataset.get_2D_slice()
    writer.add_image("Ground Truth", gt_img, 0, dataformats="CHW")
    
    model.train(True)
    print(optimizer.param_groups)
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
            
            model_output = model(x)
            loss = loss_func(y, model_output)

            loss.backward()        
            optimizer.step()
            scheduler.step()        
            profiler.step()

            logging(writer, iteration, {"Fitting loss": loss}, opt, dataset.data.shape[2:], dataset)
            if(iteration % opt['save_every'] == 0):
                save_model(model, opt)

            #if(iteration == 100):
                #model.add_layer()
                #optimizer.param_groups.pop(0)
                #optimizer.param_groups.append(model.parameters())

            
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
    parser.add_argument('--num_positional_encoding_terms',default=None,type=int,
        help='Number of positional encoding terms')   

    parser.add_argument('--data',default=None,type=str,
        help='Data file name')
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
    
    train(opt['device'], model, dataset,opt)
        
    opt['iteration_number'] = 0
    save_model(model, opt)
    

