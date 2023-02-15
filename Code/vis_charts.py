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
            2**16 : 35.253,
            2**20 : 39.456,
            2**24 : 42.700
        },
        "NGP": {
            2**16 : 35.793,
            2**20 : 41.620,
            2**24 : 45.202
        },
        "fVSRN": {
            2**16 : 33.014,
            2**20 : 35.390,
            2**24 : 40.449
        }
    }
    
    isotropic_results = {
        "Ours": {
            2**16 : 27.698, 
            2**20 : 32.224, 
            2**24 : 38.358
        },
        "NGP": {
            2**16 : 27.001,
            2**20 : 30.789,
            2**24 : 37.208
        },
        "fVSRN": {
            2**16 : 27.415,
            2**20 : 31.781,
            2**24 : 38.458
        }
    }
    
    nyx_results = {
        "Ours": {
            2**16 : 29.650,
            2**20 : 38.195,
            2**24 : 44.031 
        },
        "NGP": {
            2**16 : 28.405,
            2**20 : 35.121,
            2**24 : 42.757
        },
        "fVSRN": {
            2**16 : 29.203,
            2**20 : 37.758,
            2**24 : 43.916
        }
    }
    
    plume_results = {
        "Ours": {
            2**16 : 50.222, 
            2**20 : 57.421,
            2**24 : 57.250
        },
        "NGP": {
            2**16 : 46.671,
            2**20 : 50.841,
            2**24 : 53.300
        },
        "fVSRN": {
            2**16 : 44.698,
            2**20 : 52.496,
            2**24 : 55.071
        }
    }
    
    supernova_results = {
        "Ours": {
            2**16 : 41.897,
            2**20 : 46.609,
            2**24 : 48.203
        },
        "NGP": {
            2**16 : 41.549,
            2**20 : 44.781,
            2**24 : 47.815
        },
        "fVSRN": {
            2**16 : 39.543,
            2**20 : 43.456,
            2**24 : 47.499
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