from __future__ import absolute_import, division, print_function
import argparse
import os
from Other.utility_functions import PSNR, tensor_to_cdf, create_path, make_coord_grid
from Models.models import load_model, sample_grid, forward_maxpoints
from Models.options import load_options
from Datasets.datasets import Dataset
import torch
import numpy as np
import torch.nn.functional as F

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

def model_reconstruction(model, opt):
    
    # Load the reference data
    with torch.no_grad():
        result = sample_grid(model, opt['full_shape'], max_points=1000000,
                             align_corners=opt['align_corners'],
                             device=opt['device'],
                             data_device=opt['data_device'])
    result = result.to(opt['data_device'])
    result = result.permute(3, 0, 1, 2).unsqueeze(0)
    create_path(os.path.join(output_folder, "Reconstruction"))
    tensor_to_cdf(result, 
        os.path.join(output_folder, 
        "Reconstruction", opt['save_name']+".nc"))

def model_reconstruction_chunked(model, opt):
    
    chunk_size = 512
    full_shape = opt['full_shape']
    
    output = torch.empty(opt['full_shape'], 
        dtype=torch.float32, 
        device=opt['data_device']).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        for z_ind in range(0, full_shape[0], chunk_size):
            z_ind_end = min(full_shape[0], z_ind+chunk_size)
            z_range = z_ind_end-z_ind
            for y_ind in range(0, full_shape[1], chunk_size):
                y_ind_end = min(full_shape[1], y_ind+chunk_size)
                y_range = y_ind_end-y_ind            
                for x_ind in range(0, full_shape[2], chunk_size):
                    x_ind_end = min(full_shape[2], x_ind+chunk_size)
                    x_range = x_ind_end-x_ind
                    
                    opt['extents'] = f"{z_ind},{z_ind_end},{y_ind},{y_ind_end},{x_ind},{x_ind_end}"
                    print(f"Extents: {z_ind},{z_ind_end},{y_ind},{y_ind_end},{x_ind},{x_ind_end}")
                                                                
                    grid = [z_range, y_range, x_range]
                    coord_grid = make_coord_grid(grid, 
                        opt['data_device'], flatten=True,
                        align_corners=opt['align_corners'],
                        use_half=False)
                    
                    coord_grid += 1.0
                    coord_grid /= 2.0
                    
                    coord_grid[:,0] *= (x_range-1) / (full_shape[2]-1)
                    coord_grid[:,1] *= (y_range-1) / (full_shape[1]-1)
                    coord_grid[:,2] *= (z_range-1) / (full_shape[0]-1)
                    
                    coord_grid[:,0] += x_ind / (full_shape[2]-1)
                    coord_grid[:,1] += y_ind / (full_shape[1]-1)
                    coord_grid[:,2] += z_ind / (full_shape[0]-1)
                    
                    coord_grid *= 2.0
                    coord_grid -= 1.0
                    
                    out_tmp = forward_maxpoints(model, 
                                                coord_grid, max_points=2**20, 
                                                data_device=opt['data_device'],
                                                device=opt['device'])
                    out_tmp = out_tmp.permute(1,0)
                    out_tmp = out_tmp.view([out_tmp.shape[0]] + grid)
                    output[0,:,z_ind:z_ind_end,y_ind:y_ind_end,x_ind:x_ind_end] = out_tmp

                    print(f"Chunk {z_ind},{z_ind_end},{y_ind},{y_ind_end},{x_ind},{x_ind_end}")
        
    create_path(os.path.join(output_folder, "Reconstruction"))
    tensor_to_cdf(output, 
        os.path.join(output_folder, 
        "Reconstruction", opt['save_name']+".nc"))
    
def test_psnr(model, opt):
    
    # Load the reference data
    data = Dataset(opt).data
    
    grid = list(data.shape[2:])
    
    data = data[0].flatten(1,-1).permute(1,0)
    data_max = data.max()
    data_min = data.min()
        
    with torch.no_grad():
        coord_grid = make_coord_grid(grid, 
        opt['data_device'], flatten=True,
        align_corners=opt['align_corners'],
        use_half=False)
        
        for start in range(0, coord_grid.shape[0], 2**20):
            end_ind = min(coord_grid.shape[0], start+2**20)
            output = model(coord_grid[start:end_ind].to(opt['device']).float()).to(opt['data_device'])
            data[start:end_ind] -= output
        
        data **= 2
        SSE : torch.Tensor = data.sum()
        MSE = SSE / data.numel()
        y = 10*torch.log10(MSE)
        y = 20.0 * torch.log10(data_max-data_min) - y
    
    print(f"PSNR: {y : 0.03f}")
    return y, SSE, MSE, data.numel()

def test_psnr_chunked(model, opt):
    
    data_max = None
    data_min = None
    
    SSE = torch.tensor([0.0], dtype=torch.float32, device=opt['data_device'])
    
    chunk_size = 768
    full_shape = opt['full_shape']
    with torch.no_grad():
        for z_ind in range(0, full_shape[0], chunk_size):
            z_ind_end = min(full_shape[0], z_ind+chunk_size)
            z_range = z_ind_end-z_ind
            for y_ind in range(0, full_shape[1], chunk_size):
                y_ind_end = min(full_shape[1], y_ind+chunk_size)
                y_range = y_ind_end-y_ind            
                for x_ind in range(0, full_shape[2], chunk_size):
                    x_ind_end = min(full_shape[2], x_ind+chunk_size)
                    x_range = x_ind_end-x_ind
                    
                    opt['extents'] = f"{z_ind},{z_ind_end},{y_ind},{y_ind_end},{x_ind},{x_ind_end}"
                    print(f"Extents: {z_ind},{z_ind_end},{y_ind},{y_ind_end},{x_ind},{x_ind_end}")
                    data = Dataset(opt).data
                    data = data[0].flatten(1,-1).permute(1,0)
                    
                    if(data_max is None):
                        data_max = data.max()
                    else:
                        data_max = max(data.max(), data_max)
                    if(data_min is None):
                        data_min = data.min()
                    else:
                        data_min = min(data.min(), data_min)
                        
                    grid = [z_range, y_range, x_range]
                    coord_grid = make_coord_grid(grid, 
                        opt['data_device'], flatten=True,
                        align_corners=opt['align_corners'],
                        use_half=False)
                    
                    coord_grid += 1.0
                    coord_grid /= 2.0
                    
                    coord_grid[:,0] *= (x_range-1) / (full_shape[2]-1)
                    coord_grid[:,1] *= (y_range-1) / (full_shape[1]-1)
                    coord_grid[:,2] *= (z_range-1) / (full_shape[0]-1)
                    
                    coord_grid[:,0] += x_ind / (full_shape[2]-1)
                    coord_grid[:,1] += y_ind / (full_shape[1]-1)
                    coord_grid[:,2] += z_ind / (full_shape[0]-1)
                    
                    coord_grid *= 2.0
                    coord_grid -= 1.0
                    
                    for start in range(0, coord_grid.shape[0], 2**20):
                        end_ind = min(coord_grid.shape[0], start+2**20)
                        output = model(coord_grid[start:end_ind].to(opt['device']).float()).to(opt['data_device'])
                        data[start:end_ind] -= output
        
                    data **= 2
                    SSE += data.sum()
                    print(f"Chunk {z_ind},{z_ind_end},{y_ind},{y_ind_end},{x_ind},{x_ind_end} SSE: {data.sum()}")
        
        MSE = SSE / (full_shape[0]*full_shape[1]*full_shape[2])
        print(f"MSE: {MSE}, shape {full_shape}")
        y = 10 * torch.log10(MSE)
        y = 20.0 * torch.log10(data_max-data_min) - y
    print(f"Data min/max: {data_min}/{data_max}")
    print(f"PSNR: {y.item() : 0.03f}")

def error_volume(model, opt):
    
    # Load the reference data
    dataset = Dataset(opt)
    
    grid = list(dataset.data.shape[2:])
    
    
    with torch.no_grad():
        result = sample_grid(model, grid, max_points=1000000,
                             device=opt['device'],
                             data_device=opt['data_device'])
    result = result.to(opt['data_device'])
    result = result.permute(3, 0, 1, 2).unsqueeze(0)
    create_path(os.path.join(output_folder, "ErrorVolume"))
    
    result -= dataset.data
    result **= 2
    tensor_to_cdf(result, 
        os.path.join(output_folder, "ErrorVolume",
        opt['save_name'] + "_error.nc"))

def data_hist(model, opt):
    grid = list(opt['full_shape'])
    with torch.no_grad():
        result = sample_grid(model, grid, max_points=1000000,
            align_corners=opt['align_corners'],
            device=opt['device'],
            data_device=opt['data_device'])
        result = result.cpu().numpy().flatten()
    import matplotlib.pyplot as plt
    plt.hist(result, bins=100)
    plt.show()
    
def scale_distribution(model, opt):
    import matplotlib.pyplot as plt
    grid_scales = torch.diagonal(model.get_transformation_matrices()[:,], 0, 1, 2)[0:3]
    x_scales = grid_scales[:,0].detach().cpu().numpy()
    y_scales = grid_scales[:,1].detach().cpu().numpy()
    z_scales = grid_scales[:,2].detach().cpu().numpy()
    plt.hist(x_scales, alpha=0.4, bins=20, label="X scales")
    plt.hist(y_scales, alpha=0.4, bins=20, label="Y scales")
    plt.hist(z_scales, alpha=0.4, bins=20, label="Z scales")
    plt.legend(loc='upper right')
    plt.title("Scale distributions")
    create_path(os.path.join(output_folder, "ScaleDistributions"))
    plt.savefig(os.path.join(output_folder, "ScaleDistributions", opt['save_name']+'.png'))

def test_throughput(model, opt):

    batch = 2**23
    num_forward = 1000

    with torch.no_grad():
        input_data :torch.Tensor = torch.rand([batch, 3], device=opt['device'], dtype=torch.float32)

        import time
        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(num_forward):
            input_data.random_()
            model(input_data)

        torch.cuda.synchronize()
        t1 = time.time()
    passed_time = t1 - t0
    points_queried = batch * num_forward
    print(f"Time for {num_forward} passes with batch size {batch}: {passed_time}")
    print(f"Throughput: {points_queried/passed_time} points per second")
    GBytes = (torch.cuda.max_memory_allocated(device=opt['device']) \
                / (1024**3))
    print(f"{GBytes : 0.02f}GB of memory used (max reserved) during test.")

def feature_density(model, opt):
    
    # Load the reference data
    dataset = Dataset(opt)
    
    create_path(os.path.join(output_folder, "FeatureDensity"))
    
    data_shape = list(dataset.data.shape[2:])
    grid = make_coord_grid(data_shape, opt['device'], 
                           flatten=True, align_corners=opt['align_corners'])
    with torch.no_grad():
        print(grid.device)
        
        density = forward_maxpoints(model.feature_density, grid, 
                                    data_device=opt['data_device'], 
                                    device=opt['device'],
                                    max_points=1000000)
        density = density.reshape(data_shape)
        density = density.unsqueeze(0).unsqueeze(0)
        density = density / density.sum()
        
        tensor_to_cdf(density, 
            os.path.join(output_folder, 
            "FeatureDensity", opt['save_name']+"_density.nc"))
        
        result = sample_grid(model, list(dataset.data.shape[2:]), 
                             max_points=1000000,
                             device=opt['device'],
                             data_device=opt['data_device'])
        result = result.to(opt['data_device'])
        result = result.permute(3, 0, 1, 2).unsqueeze(0)
        result -= dataset.data
        result **= 2
        result /= result.mean()
        result = torch.exp(torch.log(density+1e-16) / torch.exp(result))
        result /= result.sum()
        tensor_to_cdf(result, 
            os.path.join(output_folder, 
            "FeatureDensity", opt['save_name']+"_targetdensity.nc"))     
        
        result = F.kl_div(torch.log(density+1e-16), 
                          torch.log(result+1e-16), 
                               reduction="none", 
                               log_target=True)           
        tensor_to_cdf(result, 
            os.path.join(output_folder, 
            "FeatureDensity", opt['save_name']+"_kl.nc"))    
        
def feature_locations(model, opt):
    if(opt['model'] == "afVSRN"):
        feat_locations = model.feature_locations.detach().cpu().numpy()
        np.savetxt(os.path.join(output_folder, "feature_locations", opt['save_name']+".csv"),
                feat_locations, delimiter=",")
    elif(opt['model'] == "AMRSRN"):
        feat_grid_shape = opt['feature_grid_shape'].split(',')
        feat_grid_shape = [eval(i) for i in feat_grid_shape]
        with torch.no_grad():
            global_points = make_coord_grid(feat_grid_shape, opt['device'], 
                                flatten=True, align_corners=opt['align_corners'])
            
            transformed_points = model.transform(global_points)
            ids = torch.arange(transformed_points.shape[0])
            ids = ids.unsqueeze(1).unsqueeze(1)
            ids = ids.repeat([1, transformed_points.shape[1], 1])
            transformed_points = torch.cat((transformed_points, ids), dim=2)
            transformed_points = transformed_points.flatten(0,1).numpy()
        
        create_path(os.path.join(output_folder, "FeatureLocations"))

        np.savetxt(os.path.join(output_folder, "FeatureLocations", opt['save_name']+".csv"),
                transformed_points, delimiter=",", header="x,y,z,id")
        
        print(f"Largest/smallest transformed points: {transformed_points.min()} {transformed_points.max()}")
    
def perform_tests(model, tests, opt):
    if("reconstruction" in tests):
        model_reconstruction_chunked(model, opt),
    if("feature_locations" in tests):
        feature_locations(model, opt)
    if("error_volume" in tests):
        error_volume(model, opt)
    if("scale_distribution" in tests):
        scale_distribution(model, opt)
    if("feature_density" in tests):
        feature_density(model, opt)
    if("psnr" in tests):
        test_psnr_chunked(model, opt)
    if("histogram" in tests):
        data_hist(model, opt)
    if("throughput" in tests):
        test_throughput(model, opt)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model on some tests')

    parser.add_argument('--load_from',default=None,type=str,help="Model name to load")
    parser.add_argument('--tests_to_run',default=None,type=str,
                        help="A set of tests to run, separated by commas. Options are psnr, reconstruction, error_volume, histogram, throughput, and feature_locations.")
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
    
    # Load the model
    opt = load_options(os.path.join(save_folder, args['load_from']))
    opt['device'] = args['device']
    opt['data_device'] = args['data_device']
    model = load_model(opt, args['device'])
    model = model.to(opt['device'])
    model.train(False)
    model.eval()
    print(model)
    
    # Perform tests
    perform_tests(model, tests_to_run, opt)
    
        
    
        



        

