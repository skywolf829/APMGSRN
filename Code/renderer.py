import torch
from nerfacc import ray_marching, rendering, OccupancyGrid, Grid
import argparse
import os
from Models.models import load_model
from Models.options import load_options
import matplotlib.pyplot as plt
import numpy as np
from Other.utility_functions import make_coord_grid, str2bool
import time
import torch.nn.functional as F

def sync_time():
    torch.cuda.synchronize()
    return time.time()

class TransferFunction():
    def __init__(self, device):
        self.device = device
        
        self.num_dict_entries = 4096
        self.coolwarm()
             
    def coolwarm(self):
        self.color_control_points = torch.tensor([0.0, 0.5, 1.0],
                                dtype=torch.float32,
                                device=self.device)
        self.opacity_control_points = torch.tensor([0.0, 1.0],
                                dtype=torch.float32,
                                device=self.device)
        
        self.color_values = torch.tensor([[59/255.0,76/255.0,192/255.0],
                             [221/255.0,221/255.0,221/255.0],
                             [180/255.0,4/255.0,38/255.0]], 
                                dtype=torch.float32,
                                device=self.device)
        self.opacity_values = torch.tensor([0.0, 1.0],
                                dtype=torch.float32,
                                device=self.device)
        
        self.precomputed_color_map = torch.zeros([self.num_dict_entries, 3],
                                dtype=torch.float32,
                                device=self.device)
        self.precomputed_opacity_map = torch.zeros([self.num_dict_entries, 1],
                                dtype=torch.float32,
                                device=self.device)
        
        for ind in range(self.color_control_points.shape[0]-1):
            color_a = self.color_values[ind]
            color_b = self.color_values[ind+1]
            
            section_range = self.color_control_points[ind+1]-self.color_control_points[ind]
            start_ind = int(self.num_dict_entries*self.color_control_points[ind])
            num_elements = int(self.num_dict_entries*section_range)
            
            color_a = color_a.unsqueeze(0).repeat(num_elements, 1)
            color_b = color_b.unsqueeze(0).repeat(num_elements, 1)
            
            lerp_values = torch.arange(0.0, 1.0, step=(1/num_elements),
                            dtype=torch.float32,
                            device=self.device).unsqueeze(1).repeat(1, 3)
            self.precomputed_color_map[start_ind:start_ind+num_elements] =\
                color_a * (1-lerp_values) + color_b*lerp_values
        
        for ind in range(self.opacity_control_points.shape[0]-1):
            opacity_a = self.opacity_values[ind]
            opacity_b = self.opacity_values[ind+1]
            
            section_range = self.opacity_control_points[ind+1]-self.opacity_control_points[ind]
            start_ind = int(self.num_dict_entries*self.opacity_control_points[ind])
            num_elements = int(self.num_dict_entries*section_range)
            
            opacity_a = opacity_a.unsqueeze(0).repeat(num_elements, 1)
            opacity_b = opacity_b.unsqueeze(0).repeat(num_elements, 1)
            
            lerp_values = torch.arange(0.0, 1.0, step=(1/num_elements),
                            dtype=torch.float32,
                            device=self.device).unsqueeze(1)
            
            self.precomputed_opacity_map[start_ind:start_ind+num_elements] =\
                opacity_a * (1-lerp_values) + opacity_b*lerp_values
        
    def color_at_value(self, value):
        value_ind = (value[:,0]*self.num_dict_entries).long().clamp(0,self.num_dict_entries)
        return torch.index_select(self.precomputed_color_map, dim=0, index=value_ind)
    
    def opacity_at_value(self, value):
        value_ind = (value[:,0]*self.num_dict_entries).long().clamp(0,self.num_dict_entries)
        return torch.index_select(self.precomputed_opacity_map, dim=0, index=value_ind)
    
class Scene():
    def __init__(self, model, opt, image_resolution, batch_size):
        self.model = model
        self.opt = opt
        self.device = self.opt['device']
        self.scene_aabb = \
            torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], 
            device=self.device)
        self.image_resolution = image_resolution
        self.batch_size = batch_size
        
        self.transfer_function = TransferFunction(self.device)
        self.occpancy_grid = self.precompute_occupancy_grid()
        torch.cuda.empty_cache()
    
    def precompute_occupancy_grid(self, grid_res=[64, 64, 64]):
        # pre-allocate an occupancy grid from a dense sampling that gets max-pooled
        sample_grid = [grid_res[0]*4, grid_res[1]*4, grid_res[2]*4]
        with torch.no_grad():
            grid = OccupancyGrid(self.scene_aabb, grid_res)
            query_points = make_coord_grid(sample_grid, device=device)
            output = self.forward_maxpoints(model, query_points)
            output_density = self.transfer_function.opacity_at_value(output)
            output_density = output_density.reshape(sample_grid)
            output_density = F.max_pool3d(output_density.unsqueeze(0).unsqueeze(0), kernel_size=4)
            filter_size = 16
            output_density = F.max_pool3d(output_density,
                kernel_size = filter_size, stride=1, padding = int(filter_size/2))
            output_density = output_density.squeeze()
            output_density = (output_density>0.01)
            grid._binary = output_density.clone()
        del output_density
        print(f"{100*((grid._binary.numel()-grid._binary.sum())/grid._binary.numel()):0.02f}% of space empty for skipping!")
        return grid
    
    def generate_viewpoint_rays(self, camera=None):
        batch_size = self.image_resolution[0]*self.image_resolution[1]
        self.rays_o = torch.cat([1*torch.ones([batch_size, 1], device=device), 
                            1*make_coord_grid([self.image_resolution[0], self.image_resolution[1]], device=device)], dim=1)
        self.rays_d = torch.tensor([-1.0, 0, 0], device=device).unsqueeze(0).repeat(batch_size, 1)
        self.rays_d = self.rays_d / self.rays_d.norm(dim=-1, keepdim=True)
        
        # Ray marching with near far plane.
        ray_indices, t_starts, t_ends = ray_marching(
            self.rays_o, self.rays_d, 
            scene_aabb=self.scene_aabb, 
            render_step_size = 5e-3,
            grid=self.occpancy_grid
        ) 
        return ray_indices, t_starts, t_ends
    
    def alpha_fn(self, t_starts, t_ends, ray_indices):
        sample_locs = self.rays_o[ray_indices] + self.rays_d[ray_indices] * (t_starts + t_ends) / 2.0
        densities = self.forward_maxpoints(model, sample_locs)
        alphas = self.transfer_function.opacity_at_value(densities)
        return alphas
    
    def rgb_alpha_fn(self, t_starts, t_ends, ray_indices):
        sample_locs = self.rays_o[ray_indices] + self.rays_d[ray_indices] * (t_starts + t_ends) / 2.0
        densities = self.forward_maxpoints(model, sample_locs)
        rgbs = self.transfer_function.color_at_value(densities)
        alphas = self.transfer_function.opacity_at_value(densities)
        return rgbs, torch.log(1+alphas)

    def forward_maxpoints(self, model, coords):
        output_shape = list(coords.shape)
        output_shape[-1] = 1
        output = torch.empty(output_shape, 
            dtype=torch.float32, 
            device=self.device)
        
        for start in range(0, coords.shape[0], self.batch_size):
            output[start:min(start+self.batch_size, coords.shape[0])] = \
                model(coords[start:min(start+self.batch_size, coords.shape[0])])
        return output

    def render_rays(self, t_starts, t_ends, ray_indices):
        colors, opacities, depths = rendering(
            t_starts, t_ends, ray_indices, n_rays=self.image_resolution[0]*self.image_resolution[1], 
            rgb_alpha_fn=self.rgb_alpha_fn,
            render_bkgd=torch.tensor([1.0, 1.0, 1.0],dtype=torch.float32,device=self.device))
        colors = colors.reshape(self.image_resolution[0], self.image_resolution[1], 3).clip(0.0,1.0)
        colors = colors.cpu().numpy()
        #print(f"Renderer {ray_indices.shape[0]} samples on {self.image_resolution[0]*self.image_resolution[1]} rays.")
        return colors
        
    def render(self):
        with torch.no_grad():
            ray_indices, t_starts, t_ends = self.generate_viewpoint_rays()
            colors = self.render_rays(t_starts, t_ends, ray_indices)
        return colors
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model on some tests')

    parser.add_argument('--load_from',default=None,type=str,help="Model name to load")
    parser.add_argument('--device',default="cuda:0",type=str,
                        help="Device to load model to")
    parser.add_argument('--tensorrt',default=False,type=str2bool,
                        help="Use TensorRT acceleration")
    args = vars(parser.parse_args())

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..")
    data_folder = os.path.join(project_folder_path, "Data")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")

    # Load the model
    opt = load_options(os.path.join(save_folder, args['load_from']))
    opt['device'] = args['device']
    model = load_model(opt, args['device']).to(opt['device'])
    model.eval()
    
    batch_size = 2**20 # just over 1 million, 1048576
    
    if(args['tensorrt']):
        import torch_tensorrt as torchtrt
        # Convert model to torch.jit.scriptmodule
        if("NGP" in opt['model']):
            print(f"Cannot convert model type {opt['model']} to torchscript for ONNX conversion. Exiting.")
            quit()
        # Check if TCNN was used in this model and convert if necessary
        if("decoder.params" in model.state_dict().keys()):
            print(f"TCNN decoder used in model. Converting to pytorch for tracing.")
            from model_to_torchscript import convert_tcnn_to_pytorch
            new_model_name = convert_tcnn_to_pytorch(opt['save_name'])
            opt = load_options(os.path.join(save_folder, new_model_name))
            opt["device"] = args['device']    
            model = load_model(opt, opt['device'])
            model = model.to(opt['device'])
        model = model.eval()
        print(model)
        #model = torch.jit.script(model)
        
        '''  
        # Convert model to onnx
        onnx_file_path = os.path.join(save_folder, opt['save_name'], "model.onnx")
        torch.onnx.export(model, torch.rand([1, 3],
                                dtype=torch.float32, 
                                device=opt['device']),
                          onnx_file_path,
                          export_params=True, 
                          input_names=["input"],
                          output_names=['output'],
                          opset_version=16 #needed for grid_sampler
                          )
        model = onnx.load(onnx_file_path)
        onnx.checker.check_model(model)
        '''
        
        os.environ["CUDA_MODULE_LOADING"] = "LAZY"
        # Convert to torch_tensorrt
        inputs = [
            torchtrt.Input([batch_size, 3],
                dtype=torch.float32
            )
        ]
        enabled_precisions = {torch.float}#, torch.half}
        model = torchtrt.compile(model, 
                inputs = inputs, 
                enabled_precisions = enabled_precisions,
                workspace_size = 1 << 33)
        
        
    if("cuda" in args['device']):        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    device = args['device']
    
    scene = Scene(model, opt, [512,512], batch_size=batch_size)
    # One warm up is always slower    
    scene.render()
    
    timesteps = 10
    times = np.zeros([timesteps])
    for i in range(timesteps):
        t0 = sync_time()
        img = scene.render()      
        t1 = sync_time()
        times[i] = t1-t0
    print(times)
    
    print(f"Average frame time: {times.mean():0.04f}")
    print(f"Min frame time: {times.min():0.04f}")
    print(f"Max frame time: {times.max():0.04f}")
    print(f"Average FPS: {1/times.mean():0.02f}")
    plt.imshow(np.flip(img, 0))
    plt.show()