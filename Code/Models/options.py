import os
import json

class Options():
    def get_default():
        opt = {}

        # For descriptions of all variables, see train.py
        opt['model']                                = 'AMRSRN'
        opt['n_dims']                               = 3       
        opt['n_outputs']                            = 1
        opt['feature_grid_shape']                   = "8,8,8"   
        opt['n_features']                           = 2      
        opt['n_grids']                              = 64
        opt['num_positional_encoding_terms']        = 6
        opt['extents']                              = None
        opt['use_bias']                             = False
        opt['use_global_position']                  = False
        opt['hash_log2_size']                       = 19           # hash grid: table size
        opt['hash_base_resolution']                 = 16           # hash grid: min resolution per dim
        opt['hash_max_resolution']                  = 2048         # hash grid: max resolution per dim
        
        opt['data']                                 = 'tornado.nc'
        opt['grid_initialization']                  = "default"
        opt['ensemble']                             = False
        opt['ensemble_grid']                        = "1,1,1"
        opt['ensemble_ghost_cells']                 = 0
        opt['grid_index']                           = "1,1,1"
        opt['save_name']                            = 'tornado'
        opt['full_shape']                           = None
        opt['align_corners']                        = True
        opt['precondition']                         = False
        opt['use_tcnn_if_available']                = True
        opt['n_layers']                             = 2       
        opt['nodes_per_layer']                      = 64
        opt['interpolate']                          = True
        opt['requires_padded_feats']                = False
        
        opt['iters_to_train_new_layer']             = 1000
        opt['iters_since_new_layer']                = 0

        opt['device']                               = 'cuda:0'
        opt['data_device']                          = 'cuda:0'

        opt['iterations']                           = 10000
        opt['points_per_iteration']                 = 100000   
        opt['lr']                                   = 0.01
        opt['beta_1']                               = 0.9
        opt['beta_2']                               = 0.99

        opt['iteration_number']                     = 0
        opt['save_every']                           = 0
        opt['log_every']                            = 0
        opt['log_features_every']                   = 0
        opt['log_image_every']                      = 0
        opt['log_image']                            = False
        opt['profile']                              = False

        return opt

def save_options(opt, save_location):
    with open(os.path.join(save_location, "options.json"), 'w') as fp:
        json.dump(opt, fp, sort_keys=True, indent=4)
    
def load_options(load_location):
    opt = Options.get_default()
    #print(load_location)
    if not os.path.exists(load_location):
        print("%s doesn't exist, load failed" % load_location)
        return
        
    if os.path.exists(os.path.join(load_location, "options.json")):
        with open(os.path.join(load_location, "options.json"), 'r') as fp:
            opt2 = json.load(fp)
    else:
        print("%s doesn't exist, load failed" % "options.json")
        return
    
    # For forward compatibility with new attributes in the options file
    for attr in opt2.keys():
        opt[attr] = opt2[attr]

    return opt
