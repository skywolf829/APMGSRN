from __future__ import absolute_import, division, print_function
from Models.options import *
from Models.models import load_model
import argparse
import torch
import math
from Other.utility_functions import create_folder

def next_highest_multiple(num:int, base:int):
    return base*int(math.ceil(max(1, num/base)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Loads a model trained using TCNN and converts it to a model that is pure PyTorch.')

    parser.add_argument('--model_name',default=None,type=str,
        help='Saved model to load and convert to torchscript')
    parser.add_argument('--padded_size',default=16,type=int,
        help='When TCNN trains, it padds the input and output dimensions to the nearest multiple of 16 or 8, ' + \
        "depending on the GPU. TensorCore natively pad to 16, while CutlassMLP padds to 8. " + \
        "See https://github.com/NVlabs/tiny-cuda-nn/issues/6 for more details.")

    args = vars(parser.parse_args())

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..")
    data_folder = os.path.join(project_folder_path, "Data")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")
    
    opt = load_options(os.path.join(save_folder, args["model_name"]))
    opt["device"] = "cpu"
    opt['use_tcnn_if_available'] = False
    
    ckpt = torch.load(os.path.join(save_folder, opt['save_name'], 'model.ckpt.tar'), 
        map_location = "cpu")
    print(ckpt['state_dict']['decoder.params'].shape)    
    
    base = args['padded_size']
    requires_padding = False
    if(opt['model'] == "fVSRN"):
        input_dim = opt['n_features']+opt['num_positional_encoding_terms']*opt['n_dims']*2
        input_dim_padded = next_highest_multiple(input_dim, base)
        if(input_dim != input_dim_padded):
            requires_padding = True
        
        output_dim = opt['n_outputs']
        output_dim_padded = next_highest_multiple(output_dim, base)
        
    elif(opt['model'] == "AMGSRN"):
        input_dim = opt['n_features']*opt['n_grids']
        input_dim_padded = next_highest_multiple(input_dim, base)
        if(input_dim != input_dim_padded):
            requires_padding = True
        
        output_dim = opt['n_outputs']
        output_dim_padded = next_highest_multiple(output_dim, base)
    else:
        print(f"Currently we do not support converting model type {opt['model']} to pure PyTorch.")
        quit()
        
        
    layer_weight_shape = []
    
    first_layer_shape = [opt['nodes_per_layer'], input_dim_padded]
    layer_weight_shape.append(first_layer_shape)
    
    for i in range(opt['n_layers']-1):
        layer_shape = [opt['nodes_per_layer'], opt['nodes_per_layer']]
        layer_weight_shape.append(layer_shape)
        
    last_layer_shape = [output_dim_padded,opt['nodes_per_layer']]
    layer_weight_shape.append(last_layer_shape)
    
    weights = ckpt['state_dict']['decoder.params']
    new_weights = []
    current_weight_index = 0
    for i in range(len(layer_weight_shape)):
        print(f"Layer {i}: {layer_weight_shape[i]}")
        
        weight_shape_this_layer = layer_weight_shape[i]        
        num_weights_this_layer = weight_shape_this_layer[0]*weight_shape_this_layer[1]
        
        if(current_weight_index+num_weights_this_layer > weights.shape[0]):
            print(f"Number of expected weights {current_weight_index+num_weights_this_layer} is larger than the number of weights saved {weights.shape}.")
            quit()
        this_layer_weights = weights[current_weight_index:current_weight_index+num_weights_this_layer].clone()
        this_layer_weights = this_layer_weights.reshape(weight_shape_this_layer)
        new_weights.append(this_layer_weights)
        current_weight_index += num_weights_this_layer
        
    if(current_weight_index != weights.shape[0]):
        print(f"The total number of weights used in conversion {current_weight_index} is less than the number of weights saved in the TCNN model {weights.shape[0]}.")
    else:
        print(f"Layer sizes are as expected. Populating new stat_dict.")
        
    del ckpt['state_dict']['decoder.params']
    for i in range(len(new_weights)):
        if(i == len(new_weights) - 1):
            # In the last layer, we can actually prune the unused weights when
            # moving back to PyTorch
            name = f"decoder.{i}.weight"
            ckpt['state_dict'][name] = new_weights[i][0:1]
        else:
            # The first layer is not pruned, even if it was padded, as TCNN learned
            # to use those padded weights as a bias essentially
            # All other layers are not changed            
            name = f"decoder.{i}.linear.weight"
            ckpt['state_dict'][name] = new_weights[i]
        
    opt['requires_padded_feats'] = requires_padding
    opt["save_name"] = args["model_name"]+"_convert_to_pytorch"
    
    folder = create_folder(save_folder, opt["save_name"])
    path_to_save = os.path.join(save_folder, folder)
    print(f"Saving converted model to {path_to_save}")
    
    torch.save({'state_dict': ckpt['state_dict']}, 
        os.path.join(path_to_save, "model.ckpt.tar"),
        pickle_protocol=4
    )
    save_options(opt, path_to_save)
    
    model = load_model(opt, opt['device'])
    
    #model_jit = torch.jit.script(model)
    #torch.jit.save(model_jit,
    #    os.path.join(save_folder, args["model_name"], "traced_model.pt"))

