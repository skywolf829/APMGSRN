import os
import json

class Options():
    def get_default():
        opt = {}

        # For descriptions of all variables, see train.py
        opt['n_dims']                               = 3       
        opt['n_outputs']                            = 2
        opt['feature_grid_shape']                   = "32,32,32"   
        opt['n_features']                           = 6      
        opt['num_positional_encoding_terms']        = 6
        opt['extents']                              = None
        
        opt['data']                                 = 'tornado.nc'
        opt['save_name']                            = 'tornado'
        opt['align_corners']                        = True
        opt['n_layers']                             = 4       
        opt['nodes_per_layer']                      = 128
        opt['interpolate']                          = False
        
        opt['iters_to_train_new_layer']             = 1000
        opt['iters_since_new_layer']                = 0

        opt['device']                               = 'cuda:0'
        opt['data_device']                          = 'cuda:0'

        opt['iterations']                           = 20000
        opt['points_per_iteration']                 = 200000   
        opt['lr']                                   = 0.01
        opt['beta_1']                               = 0.9
        opt['beta_2']                               = 0.999

        opt['iteration_number']                     = 0
        opt['save_every']                           = 100
        opt['log_every']                            = 5
        opt['log_image_every']                      = 100
        opt['log_image']                            = False
        opt['profile']                              = False

        return opt

def save_options(opt, save_location):
    with open(os.path.join(save_location, "options.json"), 'w') as fp:
        json.dump(opt, fp, sort_keys=True, indent=4)
    
def load_options(load_location):
    opt = Options.get_default()
    print(load_location)
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
