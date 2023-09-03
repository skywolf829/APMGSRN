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
        iteration % opt['log_features_every'] == 0):
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
    transformed_points = model.inverse_transform(global_points).detach().cpu()

    transformed_points += 1.0
    transformed_points *= 0.5 
    transformed_points *= (torch.tensor(list(dataset.data.shape[2:]))-1).flip(0)
    transformed_points = transformed_points.detach().cpu()#.flip(-1)

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
    transformed_points *= 0.5 * (torch.tensor(dataset.data.shape)-1).flip(0)
    #transformed_points = transformed_points.flip(-1)
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

def train_step_APMGSRN_precondition(opt, iteration, batch, dataset, model, optimizer, scheduler, profiler, writer):
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

def train_step_APMGSRN(opt, iteration, batch, dataset, model, optimizer, scheduler, writer, 
                      early_stopping_data=None):
    early_stop_reconstruction = early_stopping_data[0]
    early_stop_grid = early_stopping_data[1]
    early_stopping_reconstruction_losses = early_stopping_data[2]
    early_stopping_grid_losses = early_stopping_data[3]
    if(early_stop_reconstruction and early_stop_grid):
        return (early_stop_reconstruction, early_stop_grid, 
            early_stopping_reconstruction_losses,
            early_stopping_grid_losses)
    optimizer[0].zero_grad()                  
    x, y = batch
    
    x = x.to(opt['device'])
    y = y.to(opt['device'])
    
    transformed_x = model.transform(x)    
    model_output = model.forward_pre_transformed(transformed_x)
    
    loss = F.mse_loss(model_output, y, reduction='none')
    loss = loss.sum(dim=1, keepdim=True)
    
    loss.mean().backward()
    early_stopping_reconstruction_losses[iteration] = loss.mean().detach()
    early_stop_reconstruction = optimizer[0].param_groups[0]['lr'] < opt['lr'] * 1e-2

    if(iteration > 500 and  # let the network learn a bit first
        iteration < opt['iterations']*0.8 and  # stop the grid moving to adequately learn at the end
        not early_stop_grid):
        optimizer[1].zero_grad() 
        
        density = model.feature_density_pre_transformed(transformed_x) 
        
        density /= density.sum().detach()
        target = torch.exp(torch.log(density+1e-16) * \
            (loss.mean()/(loss+1e-16)))
        target /= target.sum()
        
        density_loss = F.kl_div(
           torch.log(density+1e-16), 
           torch.log(target.detach()+1e-16), reduction='none', 
            log_target=True)
        
        density_loss.mean().backward()
        
        optimizer[1].step()
        scheduler[1].step()   

        early_stopping_grid_losses[iteration] = density_loss.mean().detach()
        if(iteration >= 2500):
            prev_avg = early_stopping_grid_losses[iteration-2000:iteration-1000].mean()
            current_avg = early_stopping_grid_losses[iteration-1000:iteration].mean()
            
            thresh = prev_avg * 1e-4
            momentum_needed = 1
            
            # See if the slope is under the threshold
            thresh_met = prev_avg - current_avg < thresh
            
            # a let the momentum of the grids finish for 1k more iterations
            if(thresh_met):
                early_stopping_grid_losses[-1] += 1
            else:
                early_stopping_grid_losses[-1] = 0
                
            early_stop_grid = thresh_met and early_stopping_grid_losses[-1] > momentum_needed 
            if(early_stop_grid):
                print(f"Grid has converged. Setting early stopping flag.")

    else:
        density_loss = None
    
    optimizer[0].step()
    if(early_stop_grid):
        scheduler[0].step(early_stopping_reconstruction_losses[iteration-1000:iteration].mean())   
    
    if(opt['log_every'] != 0):
        logging(writer, iteration, 
            {"Fitting loss": loss, 
             "Grid loss": density_loss}, 
            model, opt, dataset.data.shape[2:], dataset, preconditioning='grid')
    return (early_stop_reconstruction, early_stop_grid, 
            early_stopping_reconstruction_losses,
            early_stopping_grid_losses)

def train_step_vanilla(opt, iteration, batch, dataset, model, optimizer, scheduler, writer,
                       early_stopping_data=None):
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
    
    if(opt['log_every'] > 0):
        writer = SummaryWriter(os.path.join('tensorboard',opt['save_name']))
    else: 
        writer = None
    dataloader = DataLoader(dataset, 
                            batch_size=None, 
                            num_workers=4 if ("cpu" in opt['data_device'] and "cuda" in opt['device']) else 0,
                            pin_memory=True if ("cpu" in opt['data_device'] and "cuda" in opt['device']) else False,
                            pin_memory_device=opt['device'] if ("cpu" in opt['data_device'] and "cuda" in opt['device']) else "")
    
    model.train(True)

    # choose the specific training iteration function based on the model
    
    if("APMGSRN" in opt['model']):
        train_step = train_step_APMGSRN
        optimizer = [
            optim.Adam(
                model.get_model_parameters(), 
                lr=opt["lr"],
                betas=[opt['beta_1'], opt['beta_2']], eps = 10e-15
                ),
            optim.Adam(
                model.get_transform_parameters(), 
                lr=opt['lr'] * 0.05, 
                betas=[opt['beta_1'], opt['beta_2']], eps = 10e-15
                )
        ]        
        scheduler = [
            torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer[0],
                mode="min", patience=500, threshold=1e-4, threshold_mode="rel",
                cooldown=250,factor=0.1,verbose=True),
            torch.optim.lr_scheduler.LinearLR(optimizer[1], 
                start_factor=1, end_factor=0.5)
        ]      
        early_stopping_data = (False, False,
            torch.zeros([opt['iterations']], 
                dtype=torch.float32, device=opt['device']),
            torch.zeros([opt['iterations']], 
                dtype=torch.float32, device=opt['device'])
            )
    else:
        train_step = train_step_vanilla
        optimizer = optim.Adam(model.parameters(), lr=opt["lr"], 
            betas=[opt['beta_1'], opt['beta_2']]) 
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
            [opt['iterations']*(2/5), opt['iterations']*(3/5), opt['iterations']*(4/5)],
            gamma=0.33)
        early_stopping_data = (False,
            torch.zeros([opt['iterations']], 
            dtype=torch.float32, device=opt['device'])
            )
    
    start_time = time.time()
    for (iteration, batch) in enumerate(dataloader):
        early_stopping_data = train_step(opt,
                iteration,
                batch,
                dataset,
                model,
                optimizer,
                scheduler,
                writer,
                early_stopping_data=early_stopping_data)
    end_time = time.time()
    sec_passed = end_time-start_time
    mins = sec_passed / 60
    
    print(f"Model completed training after {int(mins)}m {sec_passed%60:0.02f}s")

    
    #writer.add_graph(model, torch.zeros([1, 3], device=opt['device'], dtype=torch.float32))
    if(opt['log_every'] > 0):
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
        help='Number of grids for APMGSRN')
    parser.add_argument('--num_positional_encoding_terms',default=None,type=int,
        help='Number of positional encoding terms')   
    parser.add_argument('--extents',default=None,type=str,
        help='Spatial extents to use for this model from the data')   
    parser.add_argument('--bias',default=None,type=str2bool,
        help='Use bias in linear layers or not')
    parser.add_argument('--use_global_position',default=None,type=str2bool,
        help='For the fourier featuers, whether to use the global position or local.')
    parser.add_argument('--use_tcnn_if_available',default=None,type=str2bool,
        help='Whether to use TCNN if available on the machine training.')
    
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
    parser.add_argument('--grid_initialization',default=None,type=str,
        help='How to initialize APMGSRN grids. choices: default, large, small')
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
    parser.add_argument('--requires_padded_feats',default=None,type=str2bool,
        help='Pads features to next multiple of 16 for TCNN.')      
    parser.add_argument('--grid_index',default=None,type=str,
        help='Index for this network in an ensemble of networks')      
    
    
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
    torch.backends.cuda.matmul.allow_tf32 = True

    if(args['load_from'] is None):
        # Init models
        model = None
        opt = Options.get_default()

        # Read arguments and update our options
        for k in args.keys():
            if args[k] is not None:
                opt[k] = args[k]

        dataset = Dataset(opt)
        opt['data_min'] = dataset.min().item()
        opt['data_max'] = dataset.max().item()
        
        #opt['data_min'] = max(dataset.min(), dataset.data.mean() - dataset.data.std()*3).item()
        #opt['data_max'] = min(dataset.max(), dataset.data.mean() + dataset.data.std()*3).item()
        #opt['data_min'] = dataset.data.mean().item()
        #opt['data_max'] = max(dataset.data.mean()-dataset.data.min(), dataset.data.max() -dataset.data.mean()).item()
        model = create_model(opt)
        model = model.to(opt['device'])
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
    

