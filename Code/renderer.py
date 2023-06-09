import torch
import nerfacc
import argparse
import os
from Models.models import load_model
from Models.options import load_options
import numpy as np
from Other.utility_functions import make_coord_grid, str2bool
import time
import torch.nn.functional as F
from typing import List, Tuple
from math import ceil

def sync_time():
    torch.cuda.synchronize()
    return time.time()

def imgs_to_video(out_path:str, imgs: np.ndarray, fps:int=15):
    '''
    output a img sequence (N, width, height, 3) to an avi video
    '''
    import cv2
    height, width = imgs.shape[1:3]
    video=cv2.VideoWriter(
                    os.path.join(out_path+".avi"),
                    cv2.VideoWriter_fourcc(*'DIVX'),
                    fps,
                    (height, width)
                )
    for img in imgs:
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(frame)
    video.release()

def imgs_to_video_imageio(save_location, stacked_imgs, fps=10):
    import imageio.v3 as imageio
    imageio.imwrite(save_location, stacked_imgs,
                    extension=".mp4", fps=fps)

class RawData(torch.nn.Module):
    def __init__(self, data_name, device):
        super().__init__()
        self.device = device
        project_folder_path = os.path.dirname(os.path.abspath(__file__))
        project_folder_path = os.path.join(project_folder_path, "..")
        data_folder = os.path.join(project_folder_path, "Data")
        from Other.utility_functions import nc_to_tensor
        self.data, self.shape = nc_to_tensor(os.path.join(data_folder, data_name))
        self.data = self.data.to(device)
        
        
    def min(self):
        return self.data.min()

    def max(self):
        return self.data.max()
    
    def forward(self, x):
        x_device = x.device
        x = x.to(self.device)
        y = F.grid_sample(self.data,
                x.reshape(([1]*x.shape[-1]) + list(x.shape)),
                mode='bilinear', align_corners=True).squeeze().unsqueeze(1)
        return y.to(x_device)

class TransferFunction():
    def __init__(self, device, 
                 min_value :float = 0.0, max_value:float = 1.0, colormap=None):
        self.device = device
        
        self.min_value = min_value
        self.max_value = max_value
        self.mapping_minmax = torch.tensor([0.0, 1.0], device=self.device)
        
        self.num_dict_entries = 4096
        if(colormap is None):
            self.coolwarm()
        else:
            self.loadColormap(colormap)
    
    def loadColormap(self, colormapname):
        '''
        Loads a colormap exported from Paraview. Assumes colormapname is a 
        file path to the json to be loaded
        '''
        project_folder_path = os.path.dirname(os.path.abspath(__file__))
        project_folder_path = os.path.join(project_folder_path, "..")
        colormaps_folder = os.path.join(project_folder_path, "Colormaps")
        file_location = os.path.join(colormaps_folder, colormapname)
        import json
        if(os.path.exists(file_location)):
            with open(file_location) as f:
                color_data = json.load(f)[0]
        else:
            print("Colormap file doesn't exist, reverting to coolwarm")
            self.coolwarm()
            return
        
        # Load all RGB data
        rgb_data = color_data['RGBPoints']
        self.color_control_points = torch.tensor(rgb_data[0::4],
                                dtype=torch.float32,
                                device=self.device)
        self.color_control_points = self.color_control_points - self.color_control_points[0]
        self.color_control_points = self.color_control_points / self.color_control_points[-1]
        r = torch.tensor(rgb_data[1::4],
                        dtype=torch.float32,
                        device=self.device)
        g = torch.tensor(rgb_data[2::4],
                        dtype=torch.float32,
                        device=self.device)
        b = torch.tensor(rgb_data[3::4],
                        dtype=torch.float32,
                        device=self.device)
        self.color_values = torch.stack([r,g,b], dim=1)
        
        # If alpha points set, load those, otherwise ramp opacity  
        if("Points" in color_data.keys()):
            a_data = color_data['Points']
            self.opacity_control_points = torch.tensor(a_data[0::4],
                                    dtype=torch.float32,
                                    device=self.device)            
            self.opacity_control_points = self.opacity_control_points - self.opacity_control_points[0]
            self.opacity_control_points = self.opacity_control_points / self.opacity_control_points[-1]
            self.opacity_values = torch.tensor(a_data[1::4],
                                    dtype=torch.float32,
                                    device=self.device)

        else:
            self.opacity_control_points = torch.tensor([0.0, 1.0],
                                dtype=torch.float32,
                                device=self.device)
            self.opacity_values = torch.tensor([0.0, 1.0],
                                dtype=torch.float32,
                                device=self.device)
            
        self.precompute_maps()
             
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
        self.precompute_maps()
      
    def precompute_maps(self):
        self.precompute_color_map()
        self.precompute_opacity_map()
    
    def precompute_color_map(self):
        self.precomputed_color_map = torch.zeros([self.num_dict_entries, 3],
                            dtype=torch.float32,
                            device=self.device)
        for ind in range(self.color_control_points.shape[0]-1):
            color_a = self.color_values[ind]
            color_b = self.color_values[ind+1]
            
            start_ind = int(self.num_dict_entries*self.color_control_points[ind])
            end_ind = int(self.num_dict_entries*self.color_control_points[ind+1])
            num_elements = end_ind - start_ind
            if(num_elements > 0):
                color_a = color_a.unsqueeze(0).repeat(num_elements, 1)
                color_b = color_b.unsqueeze(0).repeat(num_elements, 1)
                
                lerp_values = torch.arange(0.0, 1.0, step=(1/num_elements),
                                dtype=torch.float32,
                                device=self.device).unsqueeze(1).repeat(1, 3)
                self.precomputed_color_map[start_ind:end_ind] =\
                    color_a * (1-lerp_values) + color_b*lerp_values
                    
    def precompute_opacity_map(self):       
        self.precomputed_opacity_map = torch.zeros([self.num_dict_entries, 1],
                                dtype=torch.float32,
                                device=self.device)
        
        for ind in range(self.opacity_control_points.shape[0]-1):
            opacity_a = self.opacity_values[ind]
            opacity_b = self.opacity_values[ind+1]
            
            start_ind = int(self.num_dict_entries*self.opacity_control_points[ind])
            end_ind = int(self.num_dict_entries*self.opacity_control_points[ind+1])
            num_elements = end_ind - start_ind
            if(num_elements > 0):
                opacity_a = opacity_a.unsqueeze(0).repeat(num_elements, 1)
                opacity_b = opacity_b.unsqueeze(0).repeat(num_elements, 1)
                
                lerp_values = torch.arange(0.0, 1.0, step=(1/num_elements),
                                dtype=torch.float32,
                                device=self.device).unsqueeze(1)[0:num_elements]
                self.precomputed_opacity_map[start_ind:end_ind] =\
                    opacity_a * (1-lerp_values) + opacity_b*lerp_values
        
    def set_minmax(self, min, max):
        self.min_value = min
        self.max_value = max
    
    def set_mapping_minmax(self, min, max):
        self.mapping_minmax = torch.tensor([min, max], device=self.device)
        
    def remap_value(self, values):
        new_min = -(self.mapping_minmax[0])
        new_max = 2 - self.mapping_minmax[1]
        values = values * (new_max - new_min)
        values += new_min
        return values
        
    def remap_value_inplace(self, values):
        new_min = -(self.mapping_minmax[0])
        new_max = 2 - self.mapping_minmax[1]
        values *= (new_max - new_min)
        values += new_min
    
    def update_opacities(self, opacity_control_points, opacity_values):
        self.opacity_control_points = torch.tensor(opacity_control_points,
                                    dtype=torch.float32,
                                    device=self.device)  
        self.opacity_values = torch.tensor(opacity_values,
                                    dtype=torch.float32,
                                    device=self.device)
        self.precompute_opacity_map()
        
    def color_at_value(self, value:torch.Tensor):
        value_device = value.device
        value = value.to(self.device)
        value_ind = (
            (value[:,0] - self.min_value) / (self.max_value - self.min_value) *
            (self.mapping_minmax[1] - self.mapping_minmax[0]) + self.mapping_minmax[0]
        )
        value_ind *= self.num_dict_entries
        value_ind = value_ind.long()
        value_ind.clamp_(0,self.num_dict_entries-1)
        return torch.index_select(self.precomputed_color_map, dim=0, index=value_ind).to(value_device)
    
    def opacity_at_value(self, value:torch.Tensor):
        value_device = value.device
        value = value.to(self.device)
        value_ind = (
            (value[:,0] - self.min_value) / (self.max_value - self.min_value) *
            (self.mapping_minmax[1] - self.mapping_minmax[0]) + self.mapping_minmax[0]
        )
        value_ind *= self.num_dict_entries
        value_ind = value_ind.type(torch.long)
        value_ind.clamp_(0,self.num_dict_entries-1)
        return torch.index_select(self.precomputed_opacity_map, dim=0, index=value_ind).to(value_device)

    def color_opacity_at_value(self, value:torch.Tensor):
        value_device = value.device
        value = value.to(self.device)
        value -= self.min_value
        value /= (self.max_value-self.min_value)
        self.remap_value_inplace(value)
        value *= self.num_dict_entries
        value = value.type(torch.long)
        value.clamp_(0,self.num_dict_entries-1)
        return (torch.index_select(self.precomputed_color_map, dim=0, index=value).to(value_device),
            torch.index_select(self.precomputed_opacity_map, dim=0, index=value).to(value_device))
    
    def color_opacity_at_value_inplace(self, value:torch.Tensor, rgbs, alphas, start_ind):
        value_device = value.device
        value = value.to(self.device)
        value_ind = self.remap_value((value[:,0] - self.min_value) / (self.max_value - self.min_value))
        value_ind *= self.num_dict_entries
        value_ind = value_ind.type(torch.long)
        value_ind.clamp_(0,self.num_dict_entries-1)
        rgbs[start_ind:start_ind+value.shape[0]] = \
            torch.index_select(self.precomputed_color_map, dim=0, index=value_ind).to(value_device)
        alphas[start_ind:start_ind+value.shape[0]] = \
            torch.index_select(self.precomputed_opacity_map, dim=0, index=value_ind).to(value_device)

class Camera():
    def __init__(self, device,
                 scene_aabb:torch.Tensor,
                 coi:torch.Tensor=torch.Tensor([0.,0.,0.]),
                 azi_deg:float=0.,
                 polar_deg:float=90.,
                 dist:float=200.,
                ):
        self.device = device
        self.fov = torch.tensor([60.0], device=self.device)
        self.transformation_matrix = torch.tensor([[1,0,0,200],
                                                   [0,1,0,200],
                                                   [0,0,1,-400],
                                                   [0,0,0,1]],
                                                  dtype=torch.float32,
                                                  device=self.device)
        self.look_at = torch.tensor([0.0,0.0,0.0], device=self.device)
        # argball camera args
        self.azi = torch.zeros(1, device=coi.device)
        self.polar = torch.zeros(1, device=coi.device)
        self.dist = torch.zeros(1, device=coi.device)
        self.coi = torch.zeros(3, device=coi.device)
        self.set_azi(azi_deg, device=coi.device)
        self.set_polar(polar_deg, device=coi.device)
        self.set_dist(dist, device=coi.device)
        self.set_coi(coi)
        self.set_eye(self.calc_eye())
        # inital camera: looking at the z direction, with right be x axis, up be y axis
        self.vMat = self.get_view()
    
    def position(self):
        # return self.transformation_matrix[0:3,3]
        return self.eye
    
    def set_azi(self, azi_deg:float, device="cuda:0"):
        self.azi = torch.deg2rad(torch.tensor(azi_deg, device=device))
        self.set_eye(self.calc_eye())
        
    def set_polar(self, polar_deg:float, device="cuda:0"):
        self.polar = torch.deg2rad(torch.tensor(polar_deg, device=device))
        self.set_eye(self.calc_eye())
        
    def set_dist(self, dist:float, device="cuda:0"):
        self.dist = torch.tensor(dist, device=device)
        self.set_eye(self.calc_eye())

    def set_coi(self, coi:torch.Tensor):
        self.coi = coi
        self.set_eye(self.calc_eye())
    
    def set_eye(self, eye:torch.Tensor):
        self.eye = eye
    
    def calc_eye(self):
        '''
        need 4 vars set: self.azi, self.polar, self.dist, self.coi. (Done once in constructor)
        '''
        y = self.dist*torch.cos(self.polar)
        dist_project_xz = self.dist*torch.sin(self.polar)
        x = torch.sin(self.azi)*dist_project_xz
        z = torch.cos(self.azi)*dist_project_xz
        eye_origin = torch.stack([x,y,z]).to(self.coi) # sync devices
        return self.coi + eye_origin
    
    def get_c2w(self):
        # print("PRINT C2W and eye: ", self.get_view(), self.eye, sep="\n")
        return torch.linalg.inv(self.get_view())
    
    def get_view(self):
        '''
        need 2 vars set: self.coi, self.eye
        '''
        normalize = lambda x: x / torch.norm(x, dim=-1 , keepdim=True)
        zaxis = normalize(self.eye - self.coi)
        
        up = torch.tensor([0., 1., 0.]).to(zaxis)
            
        xaxis = torch.cross(normalize(up), zaxis)
        xaxis = normalize(xaxis) if xaxis.sum() != 0. else xaxis
        yaxis = torch.cross(zaxis, xaxis)
        
        vMat = torch.tensor([
            [xaxis[0], xaxis[1], xaxis[2], -torch.dot(self.eye, xaxis)],
            [yaxis[0], yaxis[1], yaxis[2], -torch.dot(self.eye, yaxis)],
            [zaxis[0], zaxis[1], zaxis[2], -torch.dot(self.eye, zaxis)],
            [0.,0.,0.,1.]
        ], device=self.device)
        return vMat

    def get_rotate_2d(self, degree:torch.Tensor):
        cos_val = torch.cos(torch.deg2rad(degree))
        sin_val = torch.sin(torch.deg2rad(degree))
        return torch.tensor(
            [
                [cos_val, -sin_val],
                [sin_val, cos_val],
            ],
            dtype=torch.float32,
            device=self.device
        )

    def generate_dirs(self, width, height):
        '''
        generate viewing ray directions of number width x height.
        instance vars need:
            self.fov
        '''
        x, y = torch.meshgrid(
            torch.arange(width),
            torch.arange(height),
            indexing="xy",
        )
        x = x.flatten().to(self.device)
        y = y.flatten().to(self.device)
        
        # move x, y to pixel center, rescale to [-1, 1] and invert y
        x = (2*(x+0.5)/width - 1) *  torch.tan(torch.deg2rad(self.fov/2))*(width/height)
        y = (1 - 2*(y+0.5)/height) * torch.tan(torch.deg2rad(self.fov/2))
        z = -torch.ones(x.shape).to(self.device)
        camera_dirs = torch.stack([x,y,z],-1) # (height*width, 3)
        # map camera space dirs to world space
        c2w = self.get_c2w() # (4,4)
        directions = (c2w[:3,:3] @ camera_dirs.T).T
        directions = directions / torch.linalg.norm(directions, dim=-1, keepdims=True)
        
        return directions.reshape(height, width, 3)
              
class Scene(torch.nn.Module):
    def __init__(self, model, camera, full_shape, 
        image_resolution:Tuple[int], 
        batch_size : int, spp : int,
        transfer_function:TransferFunction,
        device="cuda:0", data_device="cuda:0"):
        super().__init__()
        self.model = model
        self.device : str = device
        self.data_device : str = data_device
        self.swapxyz = False
        if(self.swapxyz):
            self.scene_aabb = \
                torch.tensor([0.0, 0.0, 0.0, 
                            full_shape[2]-1,
                            full_shape[1]-1,
                            full_shape[0]-1], 
                device=self.device)
        else:
            self.scene_aabb = \
                torch.tensor([0.0, 0.0, 0.0, 
                            full_shape[0]-1,
                            full_shape[1]-1,
                            full_shape[2]-1], 
                device=self.device)
        self.image_resolution : Tuple[int]= image_resolution
        self.batch_size : int = batch_size
        self.spp = spp
        self.estimator = nerfacc.OccGridEstimator(self.scene_aabb,
            resolution=1, levels=1).to(self.device)
        # Overwrite the binary to be 1 (meaning full) everywhere
        self.estimator.binaries = torch.ones_like(self.estimator.binaries)
        self.transfer_function = transfer_function
        self.amount_empty = 0.0
        self.camera = camera
        self.on_setting_change()
    
    def get_mem_use(self):
        return torch.cuda.max_memory_allocated(device=self.device) \
                / (1024**3)
        
    def set_aabb(self, full_shape : np.ndarray):
        if(self.swapxyz):
            self.scene_aabb = \
                torch.tensor([0.0, 0.0, 0.0, 
                            full_shape[2]-1,
                            full_shape[1]-1,
                            full_shape[0]-1], 
                device=self.device)
        else:
            self.scene_aabb = \
                torch.tensor([0.0, 0.0, 0.0, 
                            full_shape[0]-1,
                            full_shape[1]-1,
                            full_shape[2]-1], 
                device=self.device)
        self.estimator = nerfacc.OccGridEstimator(self.scene_aabb,
            resolution=1, levels=1).to(self.device)
        # Overwrite the binary to be 1 (meaning full) everywhere
        self.estimator.binaries = torch.ones_like(self.estimator.binaries)
                
    def generate_viewpoint_rays(self, camera: Camera):
        height, width = self.image_resolution[:2]
        batch_size = self.image_resolution[0]*self.image_resolution[1]

        self.rays_d = camera.generate_dirs(width, height).view(-1, 3)
        self.rays_o = camera.position().unsqueeze(0).expand(batch_size, 3)
        #self.rays_o = torch.cat([1*torch.ones([batch_size, 1], device=self.device), 
        #                    1*make_coord_grid([self.image_resolution[0], self.image_resolution[1]], device=self.device)], dim=1)
        #self.rays_d = torch.tensor([-1.0, 0, 0], device=device).unsqueeze(0).repeat(batch_size, 1)
        
        # print(self.rays_d.shape)
        # print(self.rays_o.shape)
        
        max_view_dist = (self.scene_aabb[3]**2 + self.scene_aabb[4]**2 + self.scene_aabb[5]**2)**0.5
        ray_indices, t_starts, t_ends = self.estimator.sampling(
            self.rays_o, self.rays_d,
            render_step_size = max_view_dist/self.spp
        )
        return ray_indices, t_starts, t_ends
        
    def rgb_alpha_fn(self, t_starts, t_ends, ray_indices):
        sample_locs = self.rays_o[ray_indices] + self.rays_d[ray_indices] * (t_starts + t_ends)[:,None] / 2.0
        sample_locs /= self.scene_aabb[3:]
        sample_locs *= 2 
        sample_locs -= 1
        densities = self.model(sample_locs.to(self.data_device)).to(self.device)
        rgbs, alphas = self.transfer_function.color_opacity_at_value(densities[:,0])
        alphas += 1
        alphas.log_()
        
        return rgbs, alphas[:,0]
    
    def rgb_alpha_fn_batch(self, t_starts, t_ends, ray_indices):
        '''
        A batched version of rgb_alpha_fn that may help not expand memory use by
        only creating the sample locations as they are needed for evaluation.
        '''
        rgbs = torch.empty([t_starts.shape[0], 3], device=self.device, dtype=torch.float32)
        alphas = torch.empty([t_starts.shape[0], 1], device=self.device, dtype=torch.float32)
        for ray_ind_start in range(0, t_starts.shape[0], self.batch_size):
            ray_ind_end = min(ray_ind_start+self.batch_size, t_starts.shape[0])
            sample_locs = self.rays_o[ray_indices[ray_ind_start:ray_ind_end]] + \
                self.rays_d[ray_indices[ray_ind_start:ray_ind_end]] * \
                    (t_starts[ray_ind_start:ray_ind_end] + t_ends[ray_ind_start:ray_ind_end])[:,None] / 2
            sample_locs /= (self.scene_aabb[3:]-1)
            sample_locs *= 2
            sample_locs -= 1
            densities = self.model(sample_locs)
            self.transfer_function.color_opacity_at_value_inplace(densities, rgbs, alphas, ray_ind_start)
        alphas += 1
        alphas.log_()
        return rgbs, alphas[:,0]

    def forward_maxpoints(self, model, coords):
        '''
        Batches forward passes in chunks to reduce memory overhead.
        '''
        output_shape = list(coords.shape)
        output_shape[-1] = 1
        output = torch.empty(output_shape, 
            dtype=torch.float32, 
            device=self.device)
        with torch.no_grad():
            for start in range(0, coords.shape[0], self.batch_size):
                output[start:min(start+self.batch_size, coords.shape[0])] = \
                    model(coords[start:min(start+self.batch_size, coords.shape[0])].to(self.device))
        return output

    def render_rays(self, t_starts, t_ends, ray_indices, n_rays):
        
        colors, _, _, _ = nerfacc.rendering(
            t_starts, t_ends, ray_indices, n_rays, 
            rgb_alpha_fn=self.rgb_alpha_fn,
            render_bkgd=torch.tensor([1.0, 1.0, 1.0],dtype=torch.float32,device=self.device)
            )
        colors.clip_(0.0, 1.0)
        
        #print(f"Renderer {ray_indices.shape[0]} samples on {self.image_resolution[0]*self.image_resolution[1]} rays.")
        return colors
      
    def render(self, camera):
        with torch.no_grad():
            ray_indices, t_starts, t_ends = self.generate_viewpoint_rays(camera)
            colors = self.render_rays(t_starts, t_ends, ray_indices, self.image_resolution[0]*self.image_resolution[1])
        return colors.reshape(self.image_resolution[0], self.image_resolution[1], 3)

    def generate_checkerboard_render_order(self):
        class Rect():
            def __init__(self, x, y, w, h):
                self.x = x
                self.y = y
                self.w = w
                self.h = h
                if w > 1 and h > 1:
                    self.queue = [
                        #(x,y),
                        (x+w//2,y+h//2),
                        (x+w//2, y),
                        (x,y+h//2)
                    ]
                elif w > 1:
                    self.queue = [
                        #(x,y),
                        (x+w//2,y)
                    ]
                elif h > 1:
                    self.queue = [
                        #(x,y),
                        (x,y+h//2)
                    ]
                else:
                    self.queue = [
                        #(x,y)
                    ]

            def subdivide(self):
                if(self.w > 1 and self.h > 1):
                    return [
                        Rect(self.x, self.y, self.w//2, self.h//2),
                        Rect(self.x+self.w//2, self.y+self.h//2, self.w-self.w//2, self.h-self.h//2),
                        Rect(self.x+self.w//2, self.y, self.w-self.w//2, self.h//2),
                        Rect(self.x, self.y+self.h//2, self.w//2, self.h-self.h//2)
                    ]
                elif(self.w > 1):
                    return [
                        Rect(self.x, self.y, self.w//2, self.h),
                        Rect(self.x+self.w//2, self.y, self.w-self.w//2, self.h),
                    ]
                elif(self.h > 1):
                    return [
                        Rect(self.x, self.y, self.w, self.h//2),
                        Rect(self.x, self.y+self.h//2, self.w, self.h-self.h//2),
                    ]
                return []
            
            def get_next(self):
                if(len(self.queue) > 0):
                    return self.queue.pop(0)
                return None
            
            def needs_subdivide(self):
                return len(self.queue) == 0

        def checkerboard_render_order(w,h):
            rects = [Rect(0,0,w,h)]

            offset_order = [(0,0)]
            rects_to_add = []
            # continue until all rects are done
            while len(rects) > 0:
                # loop through current rects 1 at a time
                indices_to_remove = []
                # Get the next spot for each rect
                for i in range(len(rects)):
                    spot = rects[i].get_next()
                    # make sure it is valid
                    if spot is not None:
                        offset_order.append(spot)
                    # see if the rect need subdivision
                    if rects[i].needs_subdivide():
                        indices_to_remove.append(i)
                        rects_to_add += rects[i].subdivide()
                # remove finished rects
                for i in range(len(indices_to_remove)):
                    rects.pop(indices_to_remove[len(indices_to_remove)-i-1])
                
                # put new rects into queue
                if(len(rects) == 0):
                    while(len(rects_to_add) > 0):
                        rects.append(rects_to_add.pop(0))

            return offset_order

        return checkerboard_render_order(self.strides, self.strides)

    def generate_normal_render_order(self):
        order = []
        for x in range(self.strides):
            for y in range(self.strides):
                order.append((y,x))
        return order

    def on_setting_change(self):
        self.max_view_dist = (self.scene_aabb[3]**2 + self.scene_aabb[4]**2 + self.scene_aabb[5]**2)**0.5
        self.height, self.width = self.image_resolution[:2]
        n_point_evals = self.width * self.height * self.spp * (1-self.amount_empty)
        n_passes = n_point_evals / self.batch_size
        self.strides = int(n_passes**0.5) + 1
        self.image = torch.empty([self.height, self.width, 3], device=self.device, dtype=torch.float32) 
        self.mip = torch.zeros([ceil(self.height/self.strides), 
                                ceil(self.width/self.strides), 3],
                              device=self.device, dtype=torch.float32)
        self.mask = torch.zeros_like(self.image, dtype=torch.bool)
        self.temp_image = torch.empty_like(self.image)
        self.render_order = self.generate_checkerboard_render_order()
        self.current_order_spot = 0
        self.all_rays = torch.tensor(self.camera.generate_dirs(self.width, self.height), 
                                     device=self.device)
        self.cam_origin = torch.tensor(self.camera.position(), device=self.device).unsqueeze(0)
        self.y_leftover = self.height % self.strides
        self.x_leftover = self.width % self.strides
        self.passes = 0
        self.mip_level = 0
        torch.cuda.empty_cache()
   
    def on_tf_change(self):
        self.image.zero_()
        self.mask.zero_()
        self.temp_image.zero_()
        self.mip = torch.zeros([ceil(self.height/self.strides), 
                                ceil(self.width/self.strides), 3],
                              device=self.device, dtype=torch.float32)
        self.current_order_spot = 0       
        self.passes = 0
        self.mip_level = 0
        torch.cuda.empty_cache() 
         
    def on_rotate_zoom_pan(self):
        self.image.zero_()
        self.mask.zero_()
        self.temp_image.zero_()
        # Only mips need to get reset
        self.mip = torch.zeros([ceil(self.height/self.strides), 
                                ceil(self.width/self.strides), 3],
                              device=self.device, dtype=torch.float32)
        
        self.current_order_spot = 0
        self.all_rays = torch.tensor(self.camera.generate_dirs(self.width, self.height), 
                                     device=self.device)
        self.cam_origin = torch.tensor(self.camera.position(), device=self.device).unsqueeze(0)
        self.passes = 0
        self.mip_level = 0
        torch.cuda.empty_cache()
        
    def on_resize(self):
        self.height, self.width = self.image_resolution[:2]
        n_point_evals = self.width * self.height * self.spp * (1-self.amount_empty)
        n_passes = n_point_evals / self.batch_size
        self.strides = int(n_passes**0.5) + 1
        self.image = torch.empty([self.height, self.width, 3], device=self.device, dtype=torch.float32) 
        self.mip = torch.zeros([ceil(self.height/self.strides), 
            ceil(self.width/self.strides), 3],
            device=self.device, dtype=torch.float32)
        self.mask = torch.zeros_like(self.image, dtype=torch.bool)
        self.temp_image = torch.empty_like(self.image)
        self.render_order = self.generate_checkerboard_render_order()
        self.current_order_spot = 0
        self.camera.resize(self.width, self.height)
        self.all_rays = torch.tensor(self.camera.generate_dirs(self.width, self.height), 
            device=self.device)
        self.cam_origin = torch.tensor(self.camera.position(), device=self.device).unsqueeze(0)
        self.y_leftover = self.height % self.strides
        self.x_leftover = self.width % self.strides        
        self.passes = 0
        self.mip_level = 0
        torch.cuda.empty_cache()
    
    def one_step_update(self):
        if(self.current_order_spot == len(self.render_order)):
            return
        with torch.no_grad():
            x,y = self.render_order[self.current_order_spot]
                        
            mip_stride = min(self.strides, int(2**self.mip_level))
            mip_x = round(mip_stride*(x/(self.strides)))
            mip_y = round(mip_stride*(y/(self.strides)))
            y_extra = 1 if y < self.y_leftover else 0
            x_extra = 1 if x < self.x_leftover else 0
            
            rays_this_iter = self.all_rays[y::self.strides,x::self.strides].clone().view(-1, 3)
            self.rays_d = rays_this_iter
            num_rays = rays_this_iter.shape[0]
            self.rays_o = self.cam_origin.expand(num_rays, 3)
            
            ray_indices, t_starts, t_ends = self.estimator.sampling(
                self.rays_o, self.rays_d,
                render_step_size = self.max_view_dist/self.spp
            )
            new_colors = self.render_rays(
                t_starts, t_ends, 
                ray_indices, 
                num_rays).view(
                self.height//self.strides + y_extra, 
                self.width//self.strides + x_extra,
                3)

            self.image[y::self.strides,x::self.strides,:] = new_colors
            self.mask[y::self.strides,x::self.strides,:] = 1
            self.mip[mip_y:new_colors.shape[0]*mip_stride:mip_stride,
                mip_x:new_colors.shape[1]*mip_stride:mip_stride,:] = new_colors
            
            self.temp_image = self.image * self.mask + \
                F.interpolate(self.mip.permute(2,0,1).unsqueeze(0), 
                    size=[self.height,self.width],mode='bilinear')[0].permute(1,2,0) *~self.mask
            
            self.passes += 1
            if self.passes == int(4**self.mip_level):
                self.mip_level += 1
                if(self.mip.shape[0]*2 > self.height or self.mip.shape[1]*2 > self.width):
                    upscale_shape = [self.height, self.width]
                else:
                    effective_stride = self.strides/(2**self.mip_level)
                    new_h = ceil(self.height/effective_stride)
                    new_w = ceil(self.width/effective_stride)
                    upscale_shape = [new_h, new_w]
                self.mip = F.interpolate(self.mip.permute(2,0,1).unsqueeze(0), 
                    size=upscale_shape,mode='nearest')[0].permute(1,2,0)
        
        self.current_order_spot += 1
        
    def render_checkerboard(self):
        imgs = []
        
        while(self.current_order_spot < len(self.render_order)):
            self.one_step_update()

                    
        return self.image, imgs
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model on some tests')

    parser.add_argument('--load_from',default=None,type=str,help="Model name to load")
    parser.add_argument('--device',default="cuda:0",type=str,
                        help="Device to perform rendering on (requires CUDA)")
    parser.add_argument('--data_device',default="cuda:0",type=str,
                        help="Device to load and perform model/data inference/sampling")
    parser.add_argument('--colormap',default=None,type=str,
                        help="The colormap file to use for visualization.")
    parser.add_argument('--raw_data',default="false",type=str2bool,
                        help="Render raw data instead of neural rendering")
    # rendering args ********* ->
    parser.add_argument(
        '--azi',
        default=0.,
        type=float,
        help="the azimuth angle around y-axis from 0-360 degree. 0 aligns positive z-axis."
    )
    parser.add_argument(
        '--polar',
        default=90.,
        type=float,
        help="the elevation angle from 0-180 degree. 0 aligns positive y-axis, 180 aligns negative y-axis"
    )
    # default radius can be a ratio of AABB's extent
    parser.add_argument(
        '--dist',
        default=None,
        type=float,
        help="distance from center of AABB (i.e. COI) to camera"
    )
    parser.add_argument(
        '--spp',
        default=256,
        type=int,
        help="(max) samples per pixel"
    )
    parser.add_argument(
        '--hw',
        default="512,512",
        type=lambda s: [int(item) for item in s.split(",")],
        help="comma seperated height and width of image. Ex: --hw=512,512"
    )
    parser.add_argument(
        '--batch_size',
        default=2**23,
        type=int,
        help="batch size for feedforward. Larger batch size renders faster. Use smaller values for smaller VRAM. Typically between 2^19 - 2^25."
    )

    parser.add_argument(
        '--img_name',
        default="render.png",
        type=str,
        help="The save name for the rendered image."
    )
    
    # rendering args ********* <-
    
    args = vars(parser.parse_args())

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..")
    data_folder = os.path.join(project_folder_path, "Data")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")

    # Load the model
    if(not args['raw_data']):
        opt = load_options(os.path.join(save_folder, args['load_from']))
        opt['data_min'] = 0
        opt['data_max'] = 1
        opt['device'] = args['data_device']
        opt['data_device'] = args['data_device']
        model = load_model(opt, args['data_device'])
        print(opt['data_device'])
        model = model.to(opt['data_device'])
        full_shape = opt['full_shape']
        model.eval()
    else:
        model = RawData(args['load_from'], args['data_device'])
        full_shape = model.shape
    
    batch_size = args['batch_size']
            
    if("cuda" in args['device'] or "cuda" in args['data_device']):        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    device = args['device']
    
    tf = TransferFunction(device,
                          model.min(), model.max(), 
                          args['colormap'])
    aabb = torch.tensor([0.0, 0.0, 0.0, 
                        full_shape[0]-1,
                        full_shape[1]-1,
                        full_shape[2]-1])
    if(args['dist'] is None):
        args['dist'] = (aabb[3]**2 + aabb[4]**2 + aabb[5]**2)**0.5
    camera = Camera(
        device,
        scene_aabb=aabb,
        coi=aabb.reshape(2,3).mean(dim=0), # camera lookat center of aabb,
        azi_deg=args['azi'],
        polar_deg=args['polar'],
        dist=args['dist']
    )
        
    scene = Scene(model, camera, full_shape, args['hw'], 
                  batch_size, args['spp'], tf, device, 
                  args['data_device'])
    if args['dist'] is None:
        # set default camera distance to COI by a ratio to AABB
        args['dist'] = (scene.scene_aabb.max(0)[0] - scene.scene_aabb.min(0)[0])*1.8
        print("Camera distance to center of AABB:", args['dist'])
        
    #print(camera.get_c2w())
    #free_mem, total_mem = torch.cuda.mem_get_info(device)
    #free_mem /= (1024)**3
    #total_mem /= (1024)**3
    #print(f"GPU memory free/total {free_mem:0.02f}GB/{total_mem:0.02f}GB")
    
    # One warm up is always slower    
    img, seq = scene.render_checkerboard()

    from imageio import imsave
    imsave(os.path.join(output_folder, args['img_name']), 
           (img*255).cpu().numpy().astype(np.uint8))

    '''
    total_time = 1.43
    import cv2
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1.2
    fontColor              = (0,0,0)
    thickness              = 3
    lineType               = cv2.LINE_AA
    
    seq.insert(0, np.zeros_like(seq[0])+1)
    
    for i in range(len(seq)):
        current_time = i*(total_time / len(seq))
        cv2.putText(seq[i], f"Frame {i}/{len(seq)-1}",
                            (700, 50), font, fontScale,
                            fontColor, thickness, lineType) 
        cv2.putText(seq[i], f"Time: {current_time :0.03f} sec.",
                            (700, 110), font, fontScale,
                            fontColor, thickness, lineType)  
        
    imgs_to_video_imageio(os.path.join(output_folder, "Render_sequence_mask_blend.mp4"), 
                          np.array(seq), fps=15)

    from imageio import imsave, imread
    from Other.utility_functions import PSNR, ssim
    
    gt_im = torch.tensor(imread(os.path.join(output_folder, "gt.png")),dtype=torch.float32)/255
    img = img.cpu()
    p = PSNR(img, gt_im)
    s = ssim(img.permute(2,0,1).unsqueeze(0), gt_im.permute(2,0,1).unsqueeze(0))
    print(f"PSNR (image): {p:0.03f} dB")
    print(f"SSIM (image): {s: 0.03f} ")
    #img = img.astype(np.uint8)
    
    imsave("Output/model.png", img.cpu().numpy())
    '''
    
    
    timesteps = 10
    times = np.zeros([timesteps])
    for i in range(timesteps):
        torch.cuda.empty_cache()
        t0 = sync_time()
        scene.current_order_spot = 0
        img = scene.render_checkerboard()   
        t1 = sync_time()
        times[i] = t1-t0
    print(times)
    
    print(f"Average frame time: {times.mean():0.04f}")
    print(f"Min frame time: {times.min():0.04f}")
    print(f"Max frame time: {times.max():0.04f}")
    print(f"Average FPS: {1/times.mean():0.02f}")
    GBytes = (torch.cuda.max_memory_allocated(device=device) \
                / (1024**3))
    print(f"{GBytes : 0.02f}GB of memory used (max reserved) during render.")
    