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
        full_shape = get_data_size(os.path.join(data_folder, opt['data']))
        models = []
        local_extents = []
        for submodel in os.listdir(os.path.join(save_folder, opt['save_name'])):
            if(os.path.isdir(os.path.join(save_folder, opt['save_name'], submodel))):
                sub_opt = load_options(os.path.join(save_folder, 
                    opt['save_name'], submodel))
                sub_opt['device'] = opt['device']
                models.append(load_model(sub_opt, opt['device']))
                model_extents = sub_opt['extents']
                model_extents = [float(i) for i in model_extents.split(',')]
                model_extents[0] = ((model_extents[0] / full_shape[0]) - 0.5) * 2
                model_extents[1] = ((model_extents[1] / full_shape[0]) - 0.5) * 2
                model_extents[2] = ((model_extents[2] / full_shape[1]) - 0.5) * 2
                model_extents[3] = ((model_extents[3] / full_shape[1]) - 0.5) * 2
                model_extents[4] = ((model_extents[4] / full_shape[2]) - 0.5) * 2
                model_extents[5] = ((model_extents[5] / full_shape[2]) - 0.5) * 2
                local_extents.append(model_extents)
        local_extents = torch.tensor(local_extents)

        ensemble_grid = [eval(i) for i in opt['ensemble_grid'].split(',')]
        print(f"Loaded {len(models)} models in ensemble model")

        self.register_buffer("model_grid_shape",
            torch.tensor(ensemble_grid, dtype=torch.long),
            persistent=False)
        self.register_buffer("full_data_shape",
            torch.tensor(full_shape, dtype=torch.long),
            persistent=False)
        self.register_buffer("local_min_extents",
            torch.tensor(local_extents[:,0::2], dtype=torch.float32),
            persistent=False)
        self.register_buffer("local_max_extents",
            torch.tensor(local_extents[:,1::2], dtype=torch.float32),
            persistent=False)
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):    

        indices = (x+1)/2
        indices = indices*self.model_grid_shape
        indices = indices.type(torch.int)
        indices = indices.flip(-1)
        indices = indices[:,0] + indices[:,1]*self.model_grid_shape[0] + \
            indices[:,2]*(self.model_grid_shape[0]*self.model_grid_shape[1])

        y = torch.empty([x.shape[0], 1], 
            device=x.device, dtype=x.dtype)

        for i in range(len(self.models)):
            mask = (indices == i)
            x_i = x[mask]
            y[mask] = self.models[i](x_i)#(-1+2*((x_i-self.local_min_extents[i])/\
                #(self.local_max_extents[i]-self.local_min_extents[i])))

        return y

        