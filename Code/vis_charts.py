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
#plt.style.use('fivethirtyeight')
#plt.style.use('ggplot')
#plt.style.use('seaborn')
#plt.style.use('seaborn-paper')
plt.style.use('seaborn-v0_8-whitegrid')
font = {#'font.family' : 'normal',
    #'font.weight' : 'bold',
    'font.size'   : 16,
    'lines.linewidth' : 2}
plt.rcParams.update(font)

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..",)
save_folder = os.path.join(project_folder_path, "Output", "Charts")
output_folder = os.path.join(project_folder_path, "Output")
   
asteroid_results = {
    "AMGSRN": {
        2**16 : 35.350,
        2**20 : 39.970,
        2**24 : 45.004
    },
    "fVSRN": {
        2**16 : 32.992,
        2**20 : 35.408,
        2**24 : 40.442
    },
    "NGP": {
        2**16 : 35.848,
        2**20 : 41.645,
        2**24 : 45.269
    }
}

isotropic_results = {
    "AMGSRN": {
        2**16 : 27.679, 
        2**20 : 32.028, 
        2**24 : 38.268
    },
    "fVSRN": {
        2**16 : 27.427,
        2**20 : 31.781,
        2**24 : 38.464
    },
    "NGP": {
        2**16 : 27.007,
        2**20 : 30.786,
        2**24 : 37.198
    }
}

nyx_results = {
    "AMGSRN": {
        2**16 : 29.499,
        2**20 : 37.864,
        2**24 : 44.608 
    },
    "fVSRN": {
        2**16 : 29.203,
        2**20 : 37.761,
        2**24 : 43.889
    },
    "NGP": {
        2**16 : 28.391,
        2**20 : 35.142,
        2**24 : 42.750
    }
}

plume_results = {
    "AMGSRN": {
        2**16 : 49.172, 
        2**20 : 56.459,
        2**24 : 57.702
    },
    "fVSRN": {
        2**16 : 44.775,
        2**20 : 52.580,
        2**24 : 55.064
    },
    "NGP": {
        2**16 : 46.713,
        2**20 : 50.847,
        2**24 : 53.282
    }
}

supernova_results = {
    "AMGSRN": {
        2**16 : 41.914,
        2**20 : 46.787,
        2**24 : 49.695
    },
    "fVSRN": {
        2**16 : 39.497,
        2**20 : 43.513,
        2**24 : 47.480
    },
    "NGP": {
        2**16 : 41.523,
        2**20 : 44.811,
        2**24 : 47.806
    },
}

ensemble_results = {
    "Asteroid": {
        "Ensemble": 44.795,
        "Single": 45.004
        },
    "Isotropic": {
        "Ensemble": 41.067,
        "Single": 38.268
        },
    "Nyx": {
        "Ensemble": 46.230,
        "Single": 44.608 
        },
    "Plume": {
        "Ensemble": 58.829,
        "Single": 57.702
        },
    "Supernova": {
        "Ensemble": 50.991,
        "Single": 49.695
        }
}
    
ghostcell_results = {
    "1 ghost cell": 41.604,
    "4 ghost cell": 41.325,
    "8 ghost cell": 41.046,
    "16 ghost cell": 40.558
}

# Sizes in KB:
data_sizes = {
    "Asteroid" : 3906250,
    "Channel": 471859200,
    "Isotropic": 4194305,
    "Nyx": 65537,
    "Plume": 65219,
    "Rotstrat": 286435456,
    "Supernova": 314929
}

# in lists of KB, PSNR
compression_results = {
    
    "Nyx": {
        "TTHRESH": [
            [4202.0, 46.230],
            [3486.0, 44.608],
            [1137.0, 37.864],
            [211.0, 29.499] 
        ],
        "AMGSRN": [
            [65858.0, 46.230],
            [65574.0, 44.608],
            [4127.0, 37.864],
            [283.0, 29.499]            
        ]
    },
    "Plume": {
        "TTHRESH": [
            [99.7237, 58.829],  
            [114.242, 57.702],          
            [124.611, 56.459],       
            [304.084, 49.172]
        ],
        "AMGSRN": [
            [65644.0, 57.702],
            [65574.0, 58.829],
            [4127.0, 56.459],
            [283.0, 49.172]            
        ]
    },
    "Rotstrat": {
        "TTHRESH": [
            [4202.0, 49.160]
        ],
        "AMGSRN": [
            [885600.0, 49.160]        
        ]
    },
    "Supernova": {
        "TTHRESH": [
            [8928.0, 50.991],
            [7922.0, 49.695],
            [5099.0, 46.787],
            [2065.0, 41.914] 
        ],
        "AMGSRN": [
            [65744.0, 50.991],
            [65574.0, 49.695],
            [4127.0, 46.787],
            [283.0, 41.914]            
        ]
    }
}


largescale_results = {
    "Channel": {
        "97 MB": 40.337
    },
    "Rotstrat": {
        "864 MB": 49.160
    }
}     

def architecture_comparison(data, name):
    
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

def ensemble_comparison(data):
    
    create_path(os.path.join(output_folder, "Charts"))
    markers = ['o', 'v']
    x_labels = []
    y_ensemble = []
    y_single = []
    for dataset in data.keys():
        method_data = data[dataset]        
        x_labels.append(dataset)
        y_ensemble.append(method_data["Ensemble"])
        y_single.append(method_data["Single"])
    plt.plot(np.arange(len(y_ensemble)), y_ensemble, 
             label="Ensemble", marker=markers[0])
    plt.plot(np.arange(len(y_ensemble)), y_single, 
             label="Single network", marker=markers[1])
    
    plt.minorticks_off()
    plt.xticks(np.arange(len(y_ensemble)), labels=x_labels)
    plt.legend()
    plt.xlabel("Dataset")
    plt.ylabel("PSNR")
    plt.title("Ensemble Model Comparison")
    plt.savefig(os.path.join(save_folder, "Ensemble.png"),
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
    
def flat_top_chart():
    create_path(os.path.join(output_folder, "Charts"))

    x = np.arange(-2.0, 2.0, 0.01)
    
    y_gaussian = np.exp(-np.power(np.power(x, 2), 1))
    y_flat_top_5 = np.exp(-np.power(np.power(x, 2), 5))
    y_flat_top_10 = np.exp(-np.power(np.power(x, 2), 10))
    y_box = x**2 < 1.0

    plt.plot(x, y_box, label="Box")
    plt.plot(x, y_gaussian, label="Gaussian")
    plt.plot(x, y_flat_top_5, label="Flat-top, p=5")
    plt.plot(x, y_flat_top_10, label="Flat-top, p=10")
    plt.legend()
    plt.savefig(os.path.join(save_folder,"Flattop.png"),
                bbox_inches='tight',
                dpi=200)
    plt.clf()

def model_size_performance_chart():
    
    architecture_comparison(supernova_results, "Supernova")    
    architecture_comparison(asteroid_results, "Asteroid")    
    architecture_comparison(plume_results, "Plume")    
    architecture_comparison(isotropic_results, "Isotropic")    
    architecture_comparison(nyx_results, "Nyx")

def ensemble_performance_chart():
    ensemble_comparison(ensemble_results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained SSR model')
    
    parser.add_argument('--save_folder',default="Vorts_vis_results",
        type=str,help='Folder to save images to')
    
    args = vars(parser.parse_args())

    model_size_performance_chart()
    ensemble_comparison(ensemble_results)
    flat_top_chart()
    
    quit()