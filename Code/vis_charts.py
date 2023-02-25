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
    
    asteroid_results = {
        "Ours": {
            2**16 : 35.319,
            2**20 : 39.645,
            2**24 : 42.825
        },
        "NGP": {
            2**16 : 35.806,
            2**20 : 41.626,
            2**24 : 45.248
        },
        "fVSRN": {
            2**16 : 33.058,
            2**20 : 35.407,
            2**24 : 40.413
        }
    }
    
    isotropic_results = {
        "Ours": {
            2**16 : 27.703, 
            2**20 : 32.206, 
            2**24 : 38.604
        },
        "NGP": {
            2**16 : 27.006,
            2**20 : 30.776,
            2**24 : 37.199
        },
        "fVSRN": {
            2**16 : 27.417,
            2**20 : 31.780,
            2**24 : 38.468
        }
    }
    
    nyx_results = {
        "Ours": {
            2**16 : 29.616,
            2**20 : 38.294,
            2**24 : 44.302 
        },
        "NGP": {
            2**16 : 28.381,
            2**20 : 35.147,
            2**24 : 42.696
        },
        "fVSRN": {
            2**16 : 29.204,
            2**20 : 37.741,
            2**24 : 43.923
        }
    }
    
    plume_results = {
        "Ours": {
            2**16 : 50.195, 
            2**20 : 57.344,
            2**24 : 59.154
        },
        "NGP": {
            2**16 : 46.895,
            2**20 : 50.031,
            2**24 : 53.283
        },
        "fVSRN": {
            2**16 : 44.667,
            2**20 : 52.520,
            2**24 : 55.297
        }
    }
    
    supernova_results = {
        "Ours": {
            2**16 : 41.990,
            2**20 : 46.594,
            2**24 : 48.260
        },
        "NGP": {
            2**16 : 41.544,
            2**20 : 44.828,
            2**24 : 47.831
        },
        "fVSRN": {
            2**16 : 39.467,
            2**20 : 43.507,
            2**24 : 47.494
        }
    }

    ensemble_results = {
        "asteroid_2**24": 45.164,
        "isotropic_2**24": 41.646,
        "nyx_2**24": 47.054,
        "plume_2**24": 59.358,
        "supernova_2**24": 50.414
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