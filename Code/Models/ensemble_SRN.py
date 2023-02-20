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
        full_shape = get_data_size(os.path.join(data_folder, opt['data']))
        steps = [full_shape[0] / ensemble_grid[0], 
            full_shape[1] / ensemble_grid[1],
            full_shape[2] / ensemble_grid[2]]
        
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
                ind_1 = int(model_extents[0] / steps[0])
                ind_2 = int(model_extents[2] / steps[1])
                ind_3 = int(model_extents[4] / steps[2])
                ind = ind_1 + ensemble_grid[0]*ind_2 + ind_3*ensemble_grid[0]*ensemble_grid[1]
                model_extents[0] = ((model_extents[0] / full_shape[0]) - 0.5) * 2
                model_extents[1] = ((model_extents[1] / full_shape[0]) - 0.5) * 2
                model_extents[2] = ((model_extents[2] / full_shape[1]) - 0.5) * 2
                model_extents[3] = ((model_extents[3] / full_shape[1]) - 0.5) * 2
                model_extents[4] = ((model_extents[4] / full_shape[2]) - 0.5) * 2
                model_extents[5] = ((model_extents[5] / full_shape[2]) - 0.5) * 2
                unsorted_model_dict[ind] = [submodel_model, model_extents]
        print(unsorted_model_dict.keys())
        for i in range(ensemble_grid[0]*ensemble_grid[1]*ensemble_grid[2]):
            models.append(unsorted_model_dict[i][0])
            local_extents.append(unsorted_model_dict[i][1])
        
        local_extents = torch.tensor(local_extents)

        print(f"Loaded {len(models)} models in ensemble model")

        self.register_buffer("model_grid_shape",
            torch.tensor(ensemble_grid, dtype=torch.long),
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


    def forward(self, x):    

        indices = (x+1)/(2+1e-6)
        indices = indices.flip(-1)*self.model_grid_shape 
        indices = indices.type(torch.long)
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

        