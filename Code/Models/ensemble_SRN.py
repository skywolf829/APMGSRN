import torch
import torch.nn as nn
import os
from Models.options import load_options
from Models.models import load_model
from Other.utility_functions import get_data_size

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..", "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

class Ensemble_SRN(nn.Module):
    '''
    This class is purely for inference (not training). Our
    start_jobs script will ensure that the ensemble models
    are trained disjointly, but to query them efficiently
    after training, this ensemble model "wraps" the 
    many smaller models for inference/visualization tasks.
    '''
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        ensemble_grid = [eval(i) for i in opt['ensemble_grid'].split(',')]
        full_shape = opt['full_shape']
        
        models = []
        unsorted_model_dict = {}
        local_extents = []
        model_folders = os.listdir(os.path.join(save_folder, opt['save_name']))
        model_folders.sort()
        for submodel in model_folders:
            if(os.path.isdir(os.path.join(save_folder, opt['save_name'], submodel))):
                sub_opt = load_options(os.path.join(save_folder, 
                    opt['save_name'], submodel))
                sub_opt['device'] = opt['device']
                submodel_model = load_model(sub_opt, opt['device'])
                model_extents = sub_opt['extents']
                model_extents = [float(i) for i in model_extents.split(',')]
                indices = [eval(i) for i in sub_opt['grid_index'].split(",")]
                ind = indices[2] + indices[1]*ensemble_grid[2] + indices[0]*ensemble_grid[2]*ensemble_grid[1]
                #print(f"Ensemble grid index {sub_opt['grid_index']} -> {ind}")
                model_extents[0] = ((model_extents[0] / (full_shape[0]-1)) - 0.5) * 2
                model_extents[1] = (((model_extents[1]-1) / (full_shape[0]-1)) - 0.5) * 2
                model_extents[2] = ((model_extents[2] / (full_shape[1]-1)) - 0.5) * 2
                model_extents[3] = (((model_extents[3]-1) / (full_shape[1]-1)) - 0.5) * 2
                model_extents[4] = ((model_extents[4] / (full_shape[2]-1)) - 0.5) * 2
                model_extents[5] = (((model_extents[5]-1) / (full_shape[2]-1)) - 0.5) * 2
                unsorted_model_dict[ind] = [submodel_model, model_extents]
                
        # print("ensemble_grid", ensemble_grid, ensemble_grid[0]*ensemble_grid[1]*ensemble_grid[2])
        # print("unsorted model dict", len(unsorted_model_dict), unsorted_model_dict.items())
        # extent_dict = sorted([[k, v[1]] for k,v in unsorted_model_dict.items()])
        # for k, v in extent_dict:
        #     print(k, v)
            
        for i in range(ensemble_grid[0]*ensemble_grid[1]*ensemble_grid[2]):
            models.append(unsorted_model_dict[i][0])
            local_extents.append(unsorted_model_dict[i][1])
        
        local_extents = torch.tensor(local_extents)

        #print(f"Loaded {len(models)} models in ensemble model")

        self.register_buffer("model_grid_shape",
            torch.tensor(ensemble_grid, dtype=torch.long).flip(0),
            persistent=False)
        self.register_buffer("full_data_shape",
            torch.tensor(full_shape, dtype=torch.long),
            persistent=False)
        self.register_buffer("local_min_extents",
            local_extents[:,0::2].clone().detach().flip(-1),
            persistent=False)
        self.register_buffer("local_max_extents",
            local_extents[:,1::2].clone().detach().flip(-1),
            persistent=False)
        self.models = torch.nn.ModuleList(models)

    def min(self):
        val = self.models[0].min()
        for i in range(1, len(self.models)):
            val = torch.min(val, self.models[i].min())
        return val

    def max(self):
        val = self.models[0].max()
        for i in range(1, len(self.models)):
            val = torch.max(val, self.models[i].max())
        return val
    
    def get_volume_extents(self):
        return self.opt['full_shape']
    
    def forward(self, x):    

        # divide by slightly larger than 2
        # to avoid the result equaling 1
        
        indices = (x+1)/2.0
        indices.clamp_(0.0, 0.99)
        indices *= self.model_grid_shape     
        indices = indices.long()
        
        indices = indices[:,0] + indices[:,1]*self.model_grid_shape[0] + \
            indices[:,2]*(self.model_grid_shape[0]*self.model_grid_shape[1])
        
        y = torch.empty([x.shape[0], 1], 
            device=x.device, dtype=x.dtype)

        for i in range(len(self.models)):
            mask = (indices == i)
            x_i = x[mask]
            y[mask] = self.models[i](-1+2*((x_i-self.local_min_extents[i])/\
                (self.local_max_extents[i]-self.local_min_extents[i])))
        return y

        