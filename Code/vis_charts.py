import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from math import log2
from Other.utility_functions import create_path

#plt.style.use('Solarize_Light2')
plt.style.use('fivethirtyeight')
#plt.style.use('ggplot')
#plt.style.use('seaborn')
#plt.style.use('seaborn-paper')
font = {#'font.family' : 'normal',
    #'font.weight' : 'bold',
    'font.size'   : 16,
    'lines.linewidth' : 2}
plt.rcParams.update(font)

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..",)
save_folder = os.path.join(project_folder_path, "Output", "Charts")
output_folder = os.path.join(project_folder_path, "Output")
   

def data_to_figure(data, name):
    
    create_path(os.path.join(output_folder, "Charts"))
    markers = ['o', 'v', 's']
    idx = 0
    for method in data.keys():
        method_data = data[method]        
        label = method
        x = []
        x_labels = []
        y = []
        for model_size in method_data.keys():
            x.append(float(model_size))
            x_label = f"2^{{{int(log2(model_size))}}}"
            x_label = rf'${x_label}$'
            x_labels.append(x_label)
            y.append(float(method_data[model_size]))
        plt.plot(x, y, label=label, marker=markers[idx])
        idx += 1
    plt.xscale('log')
    plt.minorticks_off()
    plt.xticks(x, labels=x_labels)
    plt.legend()
    plt.xlabel("# parameters")
    plt.ylabel("PSNR")
    plt.title(name + " Model Performance")
    plt.savefig(os.path.join(save_folder, name + ".png"),
                bbox_inches='tight',
                dpi=200)
    plt.clf()

def rotation_performance_chart():
    supernova_results = {
        "No rotation": {
            2**16 : 41.360,
            2**20 : 46.155, 
            2**24 : 49.295 
        },
        "Rotation": {
            2**16 : 41.907,
            2**20 : 46.610,
            2**24 : 49.710 
        }
    }

    plume_results = {
        "No rotation": {
            2**16 : 49.592,
            2**20 : 55.973,
            2**24 : 58.178 
        },
        "Rotation": {
            2**16 : 50.064,
            2**20 : 57.372,
            2**24 : 59.668
        }
    }

    isotropic_results = {
        "No rotation": {
            2**16 : 27.724,
            2**20 : 32.128,
            2**24 : 38.027 
        },
        "Rotation": {
            2**16 : 27.700,
            2**20 : 32.222,
            2**24 : 38.111
        }
    }
    
    nyx_results = {
        "No rotation": {
            2**16 : 29.633, 
            2**20 : 38.601, 
            2**24 : 45.186 
        },
        "Rotation": {
            2**16 : 29.628,
            2**20 : 37.728,
            2**24 : 43.926
        }
    }

    asteroid_results = {
        "No rotation": {
            2**16 : 35.348, 
            2**20 : 39.583, 
            2**24 : 42.767 
        },
        "Rotation": {
            2**16 : 35.325,
            2**20 : 39.775,
            2**24 : 43.135
        }
    }

    data_to_figure(supernova_results, "Supernova_rotation")    
    data_to_figure(asteroid_results, "Asteroid_rotation")    
    data_to_figure(plume_results, "Plume_rotation")    
    data_to_figure(isotropic_results, "Isotropic_rotation")    
    data_to_figure(nyx_results, "Nyx_rotation")
    
def model_size_performance_chart():
    supernova_results = {
        "Ours": {
            2**16 : 41.360, # with rotation 41.907,
            2**20 : 46.155, # with rotation 46.610,
            2**24 : 49.295  # with rotation 49.710 
        },
        "NGP": {
            2**16 : 39.420,
            2**20 : 41.197,
            2**24 : 45.870
        },
        "fVSRN": {
            2**16 : 39.461,
            2**20 : 43.437,
            2**24 : 47.453
        }
    }

    plume_results = {
        "Ours": {
            2**16 : 49.592, # with rotation 50.064,
            2**20 : 55.973, # with rotation 57.372,
            2**24 : 58.178 # with rotation 59.668
        },
        "NGP": {
            2**16 : 44.321,
            2**20 : 45.340,
            2**24 : 49.079
        },
        "fVSRN": {
            2**16 : 44.844,
            2**20 : 52.550,
            2**24 : 55.369
        }
    }

    isotropic_results = {
        "Ours": {
            2**16 : 27.724, # with rotation 27.700,
            2**20 : 32.128, # with rotation 32.222,
            2**24 : 38.027 # with rotation 38.111
        },
        "NGP": {
            2**16 : 25.283,
            2**20 : 27.568,
            2**24 : 34.557
        },
        "fVSRN": {
            2**16 : 27.416,
            2**20 : 31.784,
            2**24 : 38.470
        }
    }
    
    nyx_results = {
        "Ours": {
            2**16 : 29.633, # with rotation 29.628,
            2**20 : 38.601, # with rotaiton 37.728,
            2**24 : 45.186 # with rotation 43.926
        },
        "NGP": {
            2**16 : 26.730,
            2**20 : 30.230,
            2**24 : 38.677
        },
        "fVSRN": {
            2**16 : 29.181,
            2**20 : 37.754,
            2**24 : 43.912
        }
    }

    asteroid_results = {
        "Ours": {
            2**16 : 35.348, # with rotation 35.325,
            2**20 : 39.583, # with rotation 39.775,
            2**24 : 42.767 # with rotation 43.135
        },
        "NGP": {
            2**16 : 35.554,
            2**20 : 40.985,
            2**24 : 47.925
        },
        "fVSRN": {
            2**16 : 33.040,
            2**20 : 35.427,
            2**24 : 40.397
        }
    }

    data_to_figure(supernova_results, "Supernova")    
    data_to_figure(asteroid_results, "Asteroid")    
    data_to_figure(plume_results, "Plume")    
    data_to_figure(isotropic_results, "Isotropic")    
    data_to_figure(nyx_results, "Nyx")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained SSR model')
    
    parser.add_argument('--save_folder',default="Vorts_vis_results",
        type=str,help='Folder to save images to')
    
    args = vars(parser.parse_args())

    model_size_performance_chart()
    rotation_performance_chart()
    quit()