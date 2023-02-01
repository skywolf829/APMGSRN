from __future__ import absolute_import, division, print_function
import argparse
from Datasets.datasets import Dataset
import datetime
from Other.utility_functions import str2bool
from Models.models import load_model, create_model, save_model
import torch
import torch.optim as optim
import time
import os
from Models.options import *
from torch.utils.tensorboard import SummaryWriter
from Models.losses import *
import shutil
from Other.utility_functions import make_coord_grid, create_path, tensor_to_cdf
from Other.vis_io import get_vts, write_vts, write_pvd, write_vtm
from vtk import vtkMultiBlockDataSet
import glob
import numpy as np
from torch.utils.data import DataLoader

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

def log_to_writer(iteration, losses, writer, opt, preconditioning=None):
    with torch.no_grad():   
        print_str = f"Iteration {iteration}/{opt['iterations']}, "
        for key in losses.keys():
            if(losses[key] is not None):    
                print_str = print_str + str(key) + f": {losses[key].mean().item() : 0.07f} " 
                writer.add_scalar(str(key), losses[key].mean().item(), iteration)
        print(print_str)
        if("cuda" in opt['device']):
            GBytes = (torch.cuda.max_memory_allocated(device=opt['device']) \
                / (1024**3))
            if preconditioning is None:
                writer.add_scalar('GPU memory (GB)', GBytes, iteration)
            elif "model" in preconditioning:
                writer.add_scalar('Preconditioning model GPU memory (GB)', GBytes, iteration)
            elif "grid" in preconditioning:
                writer.add_scalar('Preconditioning grid GPU memory (GB)', GBytes, iteration)

def logging(writer, iteration, losses, model, opt, grid_to_sample, dataset, 
            preconditioning=None):
    if(opt['log_every'] > 0 and iteration % opt['log_every'] == 0):
        log_to_writer(iteration, losses, writer, opt, preconditioning)
    if(opt['log_features_every'] > 0 and \
        iteration % opt['log_features_every'] == 0 and \
        preconditioning is not None and "grid" in preconditioning):
        log_feature_points(model, dataset, opt, iteration)

def log_feature_density(model, dataset, opt):
    feat_density = model.feature_density_box(list(dataset.data.shape[2:]))
    tensor_to_cdf(feat_density.unsqueeze(0).unsqueeze(0), os.path.join(output_folder, "FeatureLocations", 
        opt['save_name'], "density.nc"))
    coord_grid = make_coord_grid(list(dataset.data.shape[2:]), 
        opt['device'], flatten=False,
        align_corners=True)
    print(coord_grid.shape)
    coord_grid_shape = list(coord_grid.shape)
    coord_grid = coord_grid.view(-1, coord_grid.shape[-1])
    gaussian_densities = model.feature_density_gaussian(coord_grid)
    coord_grid_shape[-1] = 1
    
    gaussian_densities = gaussian_densities.reshape(coord_grid_shape)
    print(gaussian_densities.shape)

    gaussian_densities = gaussian_densities.permute(3, 0, 1, 2).unsqueeze(0)
    tensor_to_cdf(gaussian_densities, os.path.join(output_folder, "FeatureLocations", 
        opt['save_name'], "gaussian_density.nc"))

def log_feature_points(model, dataset, opt, iteration):
    feat_grid_shape = [eval(i) for i in opt['feature_grid_shape'].split(',')]
    feat_grid_shape = [2,2,2]
    
    # Dont use feature grid shape - too much overhead
    # Just use [2,2,2]
    global_points = make_coord_grid(feat_grid_shape, opt['device'], 
                    flatten=True, align_corners=True)
    transformed_points = model.inverse_transform(global_points)

    transformed_points += 1.0
    transformed_points *= 0.5 
    transformed_points *= (torch.tensor(list(dataset.data.shape[2:]))-1)
    transformed_points = transformed_points.detach().cpu()

    ids = torch.arange(transformed_points.shape[0])
    ids = ids.unsqueeze(1).unsqueeze(1)
    ids = ids.repeat([1, transformed_points.shape[1], 1])
    transformed_points = torch.cat((transformed_points, ids), dim=2)
    transformed_points = transformed_points.flatten(0,1).numpy()
    
    create_path(os.path.join(output_folder, "FeatureLocations", opt['save_name']))
    np.savetxt(os.path.join(output_folder, "FeatureLocations", 
        opt['save_name'], opt['save_name']+"_"+ f"{iteration:05}" +".csv"),
        transformed_points, delimiter=",", header="x,y,z,id")

def log_feature_grids(model, dataset, opt, iteration):
    feat_grid_shape = opt['feature_grid_shape'].split(',')
    feat_grid_shape = [eval(i) for i in feat_grid_shape]
    feat_grid_shape = [2,2,2]
    
    global_points = make_coord_grid(feat_grid_shape, opt['device'], 
                    flatten=True, align_corners=True)
    transformed_points = model.transform(global_points)

    transformed_points += 1.0
    transformed_points *= 0.5 * (torch.tensor(dataset.data.shape)-1)
    ids = torch.arange(transformed_points.shape[0])
    ids = ids.unsqueeze(1).unsqueeze(1)
    ids = ids.repeat([1, transformed_points.shape[1], 1])
    transformed_points = torch.cat((transformed_points, ids), dim=2)
    
    # use zyx point ordering for vtk files
    feat_grid_shape_zyx = np.flip(feat_grid_shape)

    # write each grid as a vts file, and aggregate their info in one .pvd file
    grid_dir = os.path.join(output_folder, "FeatureLocations", opt['save_name'], f"iter{iteration}")
    create_path(grid_dir)
    vtsNames = []
    for i, grid in enumerate(transformed_points):
        grid_points = grid[:, :3]
        grid_ids = grid[:, -1]
        vts = get_vts(feat_grid_shape_zyx, grid_points, scalar_fields={"id": grid_ids})
        vtsName = f"grid{i:02}_ts{iteration:05}.vts"
        write_vts(os.path.join(grid_dir, vtsName), vts)
        vtsNames.append(vtsName)
    write_pvd(vtsNames, outPath=os.path.join(grid_dir, f"grids_ts{iteration:05}.pvd"))

def log_feature_grids_from_points(opt):
    logdir = os.path.join(output_folder, "FeatureLocations", opt['save_name'])
    csvPaths = sorted(glob.glob(os.path.join(logdir, f"*.csv")))
    grids_iters = np.array([np.genfromtxt(csvPath, delimiter=',') for csvPath in csvPaths])
    
    feat_grid_shape = np.array([2,2,2], dtype=int)
    feat_grid_shape_zyx = np.flip(feat_grid_shape)
    grids_iters = grids_iters.reshape(len(grids_iters), opt['n_grids'], feat_grid_shape.prod(), 4)
    
    vtm_dir = os.path.join(logdir, "vtms")
    create_path(vtm_dir)
    for i, grids_iter in enumerate(grids_iters):
        vtm = vtkMultiBlockDataSet()
        vtm.SetNumberOfBlocks(len(grids_iter))
        for j, grid in enumerate(grids_iter):
            grid_points = grid[:, :3]
            grid_ids = grid[:, -1]
            vts = get_vts(feat_grid_shape_zyx, grid_points, scalar_fields={"id": grid_ids})
            vtm.SetBlock(j, vts)
        write_vtm(os.path.join(vtm_dir, f"grids_{i:03}.vtm", ), vtm)

def train_step_AMGSRN_precondition(opt, iteration, batch, dataset, model, optimizer, scheduler, profiler, writer):
    opt['iteration_number'] = iteration
    optimizer.zero_grad() 
                 
    x, y = batch
    x = x.to(opt['device'])
    y = y.to(opt['device'])
        
    model_output = model(x)
    loss = F.mse_loss(model_output, y, reduction='none')
    loss = loss.sum(dim=1, keepdim=True)
    loss.mean().backward()

    optimizer.step()
    scheduler.step()        
    profiler.step()
    
    if(opt['log_every'] != 0):
        logging(writer, iteration, 
            {"Fitting loss": loss}, 
            model, opt, dataset.data.shape[2:], dataset)

def train_step_AMGSRN(opt, iteration, batch, dataset, model, optimizer, scheduler, profiler, writer):
    opt['iteration_number'] = iteration
    optimizer[0].zero_grad() 
                 
    x, y = batch
    x = x.to(opt['device'])
    y = y.to(opt['device'])
        
    model_output = model(x)
    loss = F.mse_loss(model_output, y, reduction='none')
    loss = loss.sum(dim=1, keepdim=True)
    loss.mean().backward()
       
    
    if(iteration < opt['iterations']*0.9):
        optimizer[1].zero_grad() 
        density = model.feature_density_gaussian(x) 
        density /= density.sum().detach()  
        target = torch.exp(torch.log(density+1e-16) / \
            (loss/loss.mean()))
        target /= target.sum()
            
        density_loss = F.kl_div(
            torch.log(density+1e-16), 
                torch.log(target.detach()+1e-16), 
                reduction='none', 
                log_target=True)
        density_loss.mean().backward()
        
        optimizer[1].step()
        scheduler[1].step()   
    else:
        density_loss = None
         
    regularization_loss = 10e-6 * (torch.cat([x.view(-1) for x in model.parameters()])**2).mean()
    regularization_loss.backward()
    
    optimizer[0].step()
    scheduler[0].step()   
         
    profiler.step()
    
    if(opt['log_every'] != 0):
        logging(writer, iteration, 
            {"Fitting loss": loss, 
             "Grid loss": density_loss,
             "L1 Regularization": regularization_loss}, 
            model, opt, dataset.data.shape[2:], dataset, preconditioning='grid')

def train_step_vanilla(opt, iteration, batch, dataset, model, optimizer, scheduler, profiler, writer):
    opt['iteration_number'] = iteration
    optimizer.zero_grad()
       
    x, y = batch
    x = x.to(opt['device'])
    y = y.to(opt['device'])
    
    model_output = model(x)
    loss = F.mse_loss(model_output, y, reduction='none')
    loss.mean().backward()                   

    optimizer.step()
    scheduler.step()        
    profiler.step()
    
    logging(writer, iteration, 
        {"Fitting loss": loss.mean()}, 
        model, opt, dataset.data.shape[2:], dataset)

def train( model, dataset, opt):
    model = model.to(opt['device'])
    print(model)
    print("Training on %s" % (opt["device"]), 
        os.path.join(save_folder, opt["save_name"]))
    

    if(os.path.exists(os.path.join(project_folder_path, "tensorboard", opt['save_name']))):
        shutil.rmtree(os.path.join(project_folder_path, "tensorboard", opt['save_name']))
    
    if(os.path.exists(os.path.join(output_folder, "FeatureLocations", opt['save_name']))):
        shutil.rmtree(os.path.join(output_folder, "FeatureLocations", opt['save_name']))
        
    writer = SummaryWriter(os.path.join('tensorboard',opt['save_name']))
    dataloader = DataLoader(dataset, 
                            batch_size=None, 
                            num_workers=0 if "cuda" in opt['data_device'] else 4,
                            pin_memory=True if "cpu" in opt['data_device'] else False,
                            pin_memory_device=opt['device'])
    
    model.train(True)

    # choose the specific training iteration function based on the model
    train_step = train_step_vanilla
    if 'AMGSRN' in opt['model']:
        if(opt['precondition']):
            train_step = train_step_AMGSRN_precondition
            model.precodition_grids(dataset, writer, logging)
            model.zero_grad()            
            # Finally, reset the parameters necessary, and keep the grids
            model.reset_parameters()
            model.feature_grids.requires_grad_(True)
            model.decoder.requires_grad_(True)
            model.grid_scales.requires_grad_(False)
            model.grid_translations.requires_grad_(False)
        else:
            train_step = train_step_AMGSRN
                
    if("AMGSRN" in opt['model']):
        if(opt['precondition']):
            optimizer = optim.Adam([
                {"params": [model.encoder.feature_grids], "lr": opt["lr"]},
                {"params": model.decoder.parameters(), "lr": opt["lr"]}
            ],betas=[opt['beta_1'], opt['beta_2']], eps = 10e-15) 
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                [opt['iterations']*(2/5), opt['iterations']*(4/5)],
                gamma=0.1)
        else:
            optimizer = [optim.Adam([
                {"params": [model.encoder.feature_grids], "lr": opt["lr"]},
                {"params": model.decoder.parameters(), "lr": opt["lr"]}
            ], betas=[opt['beta_1'], opt['beta_2']], eps = 10e-15),
                optim.Adam([
                {"params": [model.encoder.grid_translations], "lr": opt['lr'] * 0.1}, 
                {"params": [model.encoder.grid_scales],"lr": opt['lr'] * 0.1},
                {"params": [model.encoder.grid_rotations], "lr": opt["lr"] * 1}
            ], betas=[opt['beta_1'], opt['beta_2']], eps = 10e-15)
            ]        
            scheduler = [
                torch.optim.lr_scheduler.LinearLR(optimizer[0],
                    start_factor=1, end_factor=1),
                torch.optim.lr_scheduler.LinearLR(optimizer[1], 
                    start_factor=1, end_factor=1)
            ]      
    else:
        optimizer = optim.Adam(model.parameters(), lr=opt["lr"], 
            betas=[opt['beta_1'], opt['beta_2']]) 
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
            [opt['iterations']*(2/5), opt['iterations']*(3/5), opt['iterations']*(4/5)],
            gamma=0.33)
        
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
        for (iteration, batch) in enumerate(dataloader):
            train_step(opt,
                       iteration,
                       batch,
                       dataset,
                       model,
                       optimizer,
                       scheduler,
                       profiler,
                       writer)
            
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
        help='Number of grids for AMGSRN')
    parser.add_argument('--num_positional_encoding_terms',default=None,type=int,
        help='Number of positional encoding terms')   
    parser.add_argument('--extents',default=None,type=str,
        help='Spatial extents to use for this model from the data')   
    parser.add_argument('--use_global_position',default=None,type=str2bool,
        help='For the fourier featuers, whether to use the global position or local.')
    
    # Hash Grid (NGP model) hyperparameters
    parser.add_argument('--hash_log2_size',default=None,type=int,
        help='Size of hash table')
    parser.add_argument('--hash_base_resolution',default=None,type=int,
        help='Minimum resolution of a single dimension')
    parser.add_argument('--hash_max_resolution',default=None,type=int,
        help='Maximum resolution of a single dimension') 
    

    parser.add_argument('--data',default=None,type=str,
        help='Data file name')
    parser.add_argument('--model',default=None,type=str,
        help='The model architecture to use')
    parser.add_argument('--save_name',default=None,type=str,
        help='Save name for the model')
    parser.add_argument('--align_corners',default=None,type=str2bool,
        help='Aligns corners in implicit model.')    
    parser.add_argument('--precondition',default=None,type=str2bool,
        help='Preconditions the grid transformations.')    
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
    parser.add_argument('--log_features_every',default=None, type=int,
        help='How often to log the feature positions')
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
    torch.jit.enable_onednn_fusion(True)

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
    if(opt['log_features_every'] > 0):
       log_feature_grids_from_points(opt)
        
    opt['iteration_number'] = 0
    save_model(model, opt)
    

