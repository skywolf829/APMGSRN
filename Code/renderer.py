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
from typing import Dict, List, Tuple
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QLabel
import sys

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

class TransferFunction():
    def __init__(self, device, 
                 min_value = 0.0, max_value = 1.0, colormap=None):
        self.device = device
        
        self.min_value = min_value
        self.max_value = max_value
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
            color_data = json.load(file_location)
        else:
            print("Colormap file doesn't exist, reverting to coolwarm")
            self.coolwarm()
            return
        
        # Load all RGB data
        rgb_data = color_data['RGBPoints']
        self.rgb_conrtol_points = torch.tensor(rgb_data[0::4],
                                dtype=torch.float32,
                                device=self.device)
        self.rgb_conrtol_points -= self.rgb_conrtol_points[0]
        self.rgb_conrtol_points /= self.rgb_conrtol_points[-1]
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
            self.opacity_conrtol_points = torch.tensor(a_data[0::4],
                                    dtype=torch.float32,
                                    device=self.device)            
            self.opacity_conrtol_points -= self.opacity_conrtol_points[0]
            self.opacity_conrtol_points /= self.opacity_conrtol_points[-1]
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
        value = (value - self.min_value) / (self.max_value - self.min_value)
        value_ind = (value[:,0]*self.num_dict_entries).long().clamp(0,self.num_dict_entries-1)
        return torch.index_select(self.precomputed_color_map, dim=0, index=value_ind)
    
    def opacity_at_value(self, value):
        value = (value - self.min_value) / (self.max_value - self.min_value)
        value_ind = (value[:,0]*self.num_dict_entries).long().clamp(0,self.num_dict_entries-1)
        return torch.index_select(self.precomputed_opacity_map, dim=0, index=value_ind)

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
    
    def up(self):
        dir=self.transformation_matrix@torch.tensor([0,1.0,0,1.0],
                                                       device=self.device)
        return dir[0:3] / dir[0:3].norm()
    
    def forward(self):
        dir = self.transformation_matrix@torch.tensor([0,0,1.0,1.0],
                                                       device=self.device)
        return dir[0:3] / dir[0:3].norm()
    
    def right(self):
        dir = self.transformation_matrix@torch.tensor([1.0,0,0,1.0],
                                                       device=self.device)
        return dir[0:3] / dir[0:3].norm()
  
    # def screen_to_ray_dirs(self, screen_coords):
        
    #     im_width = screen_coords.shape[1]
    #     im_height = screen_coords.shape[0]
    #     aspect_ratio = im_width/im_height
        
    #     z = 1/torch.tan(self.fov*torch.pi/360)
    #     screen_coords[:,:,0] *= aspect_ratio
    #     screen_coords = torch.cat([screen_coords, 
    #         z.unsqueeze(0).unsqueeze(0).repeat(im_height, im_width, 1)],dim=-1)
    #     return screen_coords
    
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
    
    def update_view_eye(self, dx, dy):
        '''
        TODO:
        arcball camera model: update view matrix and eye based on x,y change in screen
        '''
        pass
    
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
    def __init__(self, model, opt, 
        image_resolution:Tuple[int], 
        batch_size : int, transfer_function:TransferFunction):
        super().__init__()
        self.model = model
        self.opt : Dict = opt
        self.device : str = opt['device']
        self.scene_aabb = \
            torch.tensor([0.0, 0.0, 0.0, 
                        self.opt['full_shape'][0],
                        self.opt['full_shape'][1],
                        self.opt['full_shape'][2]], 
            device=self.device)
        print(f"Bounding box: {self.scene_aabb}")
        self.image_resolution : Tuple[int]= image_resolution
        self.batch_size : int= batch_size
        
        self.transfer_function = transfer_function
        # self.occpancy_grid = self.precompute_occupancy_grid()
        torch.cuda.empty_cache()
   
    def precompute_occupancy_grid(self, grid_res:List[int]=[64, 64, 64]):
        # pre-allocate an occupancy grid from a dense sampling that gets max-pooled
        sample_grid : List[int] = [grid_res[0]*4, grid_res[1]*4, grid_res[2]*4]
        with torch.no_grad():
            grid = OccupancyGrid(self.scene_aabb, grid_res)
            query_points = make_coord_grid(sample_grid, device=device)
            output = self.forward_maxpoints(model, query_points)
            output_density = self.transfer_function.opacity_at_value(output)
            output_density = output_density.reshape(sample_grid)
            output_density = F.max_pool3d(output_density.unsqueeze(0).unsqueeze(0), kernel_size=4)
            filter_size : int = 16
            output_density = F.max_pool3d(output_density,
                kernel_size = filter_size, stride=1, padding = int(filter_size/2))
            output_density = output_density.squeeze()
            output_density = (output_density>0.01)
            grid._binary = output_density.clone()
        print(f"{100*((grid._binary.numel()-grid._binary.sum())/grid._binary.numel()):0.02f}% of space empty for skipping!")
        return grid
    
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
        
        # Ray marching with near far plane.
        ray_indices, t_starts, t_ends = ray_marching(
            self.rays_o, self.rays_d,
            scene_aabb=self.scene_aabb, 
            render_step_size = 2,
            # grid=self.occpancy_grid
            grid=None
        )
        return ray_indices, t_starts, t_ends
    
    def alpha_fn(self, t_starts, t_ends, ray_indices):
        sample_locs = self.rays_o[ray_indices] + self.rays_d[ray_indices] * (t_starts + t_ends) / 2.0
        densities = self.forward_maxpoints(model, sample_locs)
        alphas = self.transfer_function.opacity_at_value(densities)
        return alphas
    
    def rgb_alpha_fn(self, t_starts, t_ends, ray_indices):
        sample_locs = self.rays_o[ray_indices] + self.rays_d[ray_indices] * (t_starts + t_ends) / 2.0
        sample_locs /= self.scene_aabb[3:]
        sample_locs *= 2 
        sample_locs -= 1
        densities = self.forward_maxpoints(model, sample_locs)
        rgbs = self.transfer_function.color_at_value(densities)
        alphas = self.transfer_function.opacity_at_value(densities)
        return rgbs, torch.log(1+alphas)

    def forward_maxpoints(self, model, coords):
        print(coords.shape)
        output_shape = list(coords.shape)
        output_shape[-1] = 1
        output = torch.empty(output_shape, 
            dtype=torch.float32, 
            device=self.device)
        with torch.no_grad():
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
        #print(f"Renderer {ray_indices.shape[0]} samples on {self.image_resolution[0]*self.image_resolution[1]} rays.")
        return colors
      
    def render(self, camera):
        with torch.no_grad():
            ray_indices, t_starts, t_ends = self.generate_viewpoint_rays(camera)
            colors = self.render_rays(t_starts, t_ends, ray_indices)
        return colors

# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")
    
        self.render_view = QLabel()
    
        # Set the central widget of the Window.
        self.setCentralWidget(self.render_view)
        
    def set_render_image(self, img):    
        height, width, channel = img.shape
        bytesPerLine = channel * width
        qImg = QImage(img, width, height, bytesPerLine, QImage.Format_RGB888)
        self.render_view.setPixmap(QPixmap(qImg))


            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model on some tests')

    parser.add_argument('--load_from',default=None,type=str,help="Model name to load")
    parser.add_argument('--device',default="cuda:0",type=str,
                        help="Device to load model to")
    parser.add_argument('--tensorrt',default=False,type=str2bool,
                        help="Use TensorRT acceleration")
    parser.add_argument('--colormap',default=None,type=str,
                        help="The colormap file to use for visualization.")
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
        '--hw',
        default="512,512",
        type=lambda s: [int(item) for item in s.split(",")],
        help="comma seperated height and width of image. Ex: --hw=512,512"
    )
    
    # rendering args ********* <-
    
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
    
    tf = TransferFunction(device, model.min(), model.max(), args['colormap'])
    scene = Scene(model, opt, args['hw'], batch_size, tf)
    if args['dist'] is None:
        # set default camera distance to COI by a ratio to AABB
        args['dist'] = (scene.scene_aabb.max(0)[0] - scene.scene_aabb.min(0)[0])*1.8
        print("Camera distance to center of AABB:", args['dist'])
    camera = Camera(
        device,
        scene_aabb=scene.scene_aabb,
        coi=scene.scene_aabb.reshape(2,3).mean(0), # camera lookat center of aabb,
        azi_deg=args['azi'],
        polar_deg=args['polar'],
        dist=args['dist']
    )
    
    # One warm up is always slower    
    img = scene.render(camera)
    from imageio import imsave
    img = img.cpu().numpy()*255
    img = img.astype(np.uint8)
    imsave("Output/gt.png", img)
    
    # timesteps = 10
    # times = np.zeros([timesteps])
    # for i in range(timesteps):
    #     t0 = sync_time()
    #     img = scene.render(camera).cpu().numpy()     
    #     t1 = sync_time()
    #     times[i] = t1-t0
    # print(times)
    
    # print(f"Average frame time: {times.mean():0.04f}")
    # print(f"Min frame time: {times.min():0.04f}")
    # print(f"Max frame time: {times.max():0.04f}")
    # print(f"Average FPS: {1/times.mean():0.02f}")
    # #plt.imshow(np.flip(img, 0))
    # #plt.show()
    
    # app = QApplication([])

    # window = MainWindow()
    # img = img*255
    # img = img.astype(np.uint8)
    # window.set_render_image(img)
    # window.show()

    # app.exec()