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
plt.style.use('seaborn')
#plt.style.use('seaborn-paper')
#plt.style.use('seaborn-v0_8-whitegrid')

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
        2**16 : 34.776,
        2**20 : 40.040,
        2**24 : 44.652
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
        2**16 : 27.622, 
        2**20 : 31.825, 
        2**24 : 38.120
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
        2**16 : 29.424,
        2**20 : 37.807,
        2**24 : 44.075 
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
        2**16 : 48.954, 
        2**20 : 56.327,
        2**24 : 58.134
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
        2**16 : 41.891,
        2**20 : 46.831,
        2**24 : 49.880
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
        "Ensemble": 44.688,
        "Single": 44.652
        },
    "Isotropic": {
        "Ensemble": 41.086,
        "Single": 38.268
        },
    "Nyx": {
        "Ensemble": 46.118,
        "Single": 44.075 
        },
    "Plume": {
        "Ensemble": 58.363,
        "Single": 58.134
        },
    "Supernova": {
        "Ensemble": 50.644,
        "Single": 49.880
        }
}
    
ghostcell_results = {
    "Isotropic":
        {
            "1 ghost cell": 40.490,
            "16 ghost cells": 39.915
        },
    "Supernova":
        {
            "1 ghost cell": 50.152,
            "16 ghost cells": 49.449
        }
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
# in lists of KB, PSNR, compression time (s), decomp time (s)
compression_results = {
    "Isotropic":{
        "AMGSRN": [
            [64248, 40.762, 106, 2.8],
            [4127, 31.860, 66, 1.7],
            [283, 27.629, 56, 1.37]
        ],
        "TTHRESH": [ # tthresh -i Isotropic.raw -c Isotropic.tthresh -o Isotropic.tthresh.out -t float -s 1024 1024 1024 -r VAL -v -d
            #[176350, 55.04, 566, 87], # -r 0.001
            [81224, 48.83, 476.18, 76.39], # -r 0.002
            [48954, 45.17, 445.52, 59.1], # -r 0.003
            #[21921, 40.73, 392.49, 50.47], # -r 0.005
            [4308, 32.80, 244.14, 38.45], # -r 0.012
            [1282, 28.16, 435.74, 35.00], # -r 0.02
            [313, 24.19, 582.12, 29.50], # -r 0.03
        ],
        "SZ3": [ # sz3 -f -i Isotropic.raw -z Isotropic.raw.sz -o Isotropic.raw.sz.out -3 1024 1024 1024 -M REL -R VAL -a
            [98784, 52.53, 41.52, 50.03], #REL error 0.005
            [53734, 47.46, 37.68, 48.23], #REL error 0.01
            [6607, 37.12, 29.50, 28.89], #REL error 0.05
            [1919, 33.16, 35.34, 18.84], #REL error 0.1
            [641, 28.28, 33.80, 18.78], #REL error 0.2
        ]
    },
    "Rotstrat": {
        "TTHRESH": [
            ["OOM", 49.160]
        ],
        "AMGSRN": [
            [885600.0, 49.160]        
        ]
    }
}

largescale_results = {
    "Channel": {
        "99072": 40.201,            
        "393864": 40.011,
        "787200": 40.147
    },
    "Rotstrat": {
        "55 MB": 41.322,        
        "183 MB": 44.294,
        "864 MB": 49.074
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

def compression_charts():
    create_path(os.path.join(output_folder, "Charts"))
    markers = ['o', 'v', 's']
    idx = 0
    isotropic_comp_results = compression_results['Isotropic']
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    ax3.set_box_aspect(1)
    #ax1.set_title("Reconstruction")
    ax1.set_ylabel("Compression ratio")
    ax1.set_xlabel("PSNR (dB)")
    #ax2.set_title("Compression time")
    ax2.set_xlabel("Compression time (sec.)")
    #ax3.set_title("Decompression time")
    ax3.set_xlabel("Decompression time (sec.)")
    idx = 0
    for method in isotropic_comp_results.keys():
        method_data = isotropic_comp_results[method]        
        label = method

        data = np.array(method_data)
        crs = data_sizes["Isotropic"]/data[:,0] 
        psnrs = data[:,1]
        comp_times = data[:,2]
        decomp_times = data[:,3]

        ax1.plot(psnrs, crs, label=label, marker = markers[idx])
        ax2.plot(comp_times, crs, label=label, marker = markers[idx])
        ax3.plot(decomp_times, crs, label=label, marker = markers[idx])
        idx += 1
    plt.legend()
    plt.yscale("log") 
    plt.yticks([10**i for i in range(1, 5)], 
               labels=[f"1:{10**i}" for i in range(1,5)])
    plt.savefig(os.path.join(save_folder, "Compression.png"),
                bbox_inches='tight',
                dpi=200)

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

def render_sequence_to_mp4(folder, save_name,
        text_data = None, fps = 20):
    import imageio.v3 as imageio
    import cv2
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1.5
    fontColor              = (0,0,0)
    thickness              = 3
    lineType               = cv2.LINE_AA

    imgs = [None]*len(os.listdir(folder))
    for im in os.listdir(folder):
        if ".png" in im:
            im_data = imageio.imread(os.path.join(folder, im))
            
            im_ind = int(im.split('.')[1])
            if(text_data is not None):
                for i in range(len(text_data)):
                    words = text_data[i][0]
                    position = text_data[i][1]
                    if("iter" in words):                        
                        cv2.putText(im_data, f"Iteration {im_ind*50}/{len(imgs)*50}",
                            position, font, fontScale,
                            fontColor, thickness, lineType)            
                    else:
                        cv2.putText(im_data, words,
                            position, font, fontScale,
                            fontColor, thickness, lineType)    
            
            if(im_ind >= len(imgs)):
                imgs.append(im_data)  
            else:      
                imgs[im_ind] = im_data      
            print(f"{im_ind}/{len(os.listdir(folder))}")
            
    if imgs[0] is None:
        imgs.pop(0)
        
    save_location = os.path.join(folder, "..", save_name)
    stacked_imgs = np.stack(imgs, axis=0)
    #print(stacked_imgs.shape)
    imageio.imwrite(save_location, stacked_imgs,
                    extension=".mp4", fps=fps)

def do_all_renders():
    '''
    # Do all Asteroid renders
    render_sequence_to_mp4(
        os.path.join(output_folder, 
                     "RenderSequences", 
                     "Asteroid_grids_training"),
        "Asteroid_grids_training.mp4", 
        text_data=[["iter", (1300, 150)]],
        fps=60)
    render_sequence_to_mp4(
        os.path.join(output_folder, 
                     "RenderSequences", 
                     "Asteroid_gt"),
        "Asteroid_gt.mp4", 
        #text_data=[["Asteroid", (50, 100)],["Raw data", (50, 150)]],
        fps=60)
    render_sequence_to_mp4(
        os.path.join(output_folder, 
                     "RenderSequences", 
                     "Asteroid_grids"),
        "Asteroid_grids.mp4", 
        #text_data=[["Asteroid", (1500, 100)], ["Model - large", (1500, 150)],["39.455 dB", (1500, 200)]],
        fps=60)
    render_sequence_to_mp4(
        os.path.join(output_folder, 
                     "RenderSequences", 
                     "Asteroid_model"),
        "Asteroid_model.mp4", 
        #text_data=[["Asteroid", (1500, 100)],["Model - large", (1500, 150)],["39.455 dB", (1500, 200)]],
        fps=60)
    
    # Do all Isotropic renders
    render_sequence_to_mp4(
        os.path.join(output_folder, 
                     "RenderSequences", 
                     "Isotropic_grids_training"),
        "Isotropic_grids_training.mp4", 
        text_data=[["iter", (1300, 150)]],
        fps=60
        )
    render_sequence_to_mp4(
        os.path.join(output_folder, 
                     "RenderSequences", 
                     "Isotropic_gt"),
        "Isotropic_gt.mp4", 
        fps=60)
    render_sequence_to_mp4(
        os.path.join(output_folder, 
                     "RenderSequences", 
                     "Isotropic_grids"),
        "Isotropic_grids.mp4", 
        fps=60)
    render_sequence_to_mp4(
        os.path.join(output_folder, 
                     "RenderSequences", 
                     "Isotropic_model"),
        "Isotropic_model.mp4", 
        fps=60)
    
    # Do all Nyx renders
    render_sequence_to_mp4(
        os.path.join(output_folder, 
                     "RenderSequences", 
                     "Nyx_grids_training"),
        "Nyx_grids_training.mp4", 
        text_data=[["iter", (1300, 150)]],
        fps=60
        )
    render_sequence_to_mp4(
        os.path.join(output_folder, 
                     "RenderSequences", 
                     "Nyx_gt"),
        "Nyx_gt.mp4", 
        fps=60)
    render_sequence_to_mp4(
        os.path.join(output_folder, 
                     "RenderSequences", 
                     "Nyx_grids"),
        "Nyx_grids.mp4", 
        fps=60)
    render_sequence_to_mp4(
        os.path.join(output_folder, 
                     "RenderSequences", 
                     "Nyx_model"),
        "Nyx_model.mp4", 
        fps=60)
    
    # Do all Plume renders
    render_sequence_to_mp4(
        os.path.join(output_folder, 
                     "RenderSequences", 
                     "Plume_grids_training"),
        "Plume_grids_training.mp4", 
        text_data=[["iter", (1300, 150)]],
        fps=60
        )
    render_sequence_to_mp4(
        os.path.join(output_folder, 
                     "RenderSequences", 
                     "Plume_gt"),
        "Plume_gt.mp4", 
        fps=60)
    render_sequence_to_mp4(
        os.path.join(output_folder, 
                     "RenderSequences", 
                     "Plume_grids"),
        "Plume_grids.mp4", 
        fps=60)
    render_sequence_to_mp4(
        os.path.join(output_folder, 
                     "RenderSequences", 
                     "Plume_model"),
        "Plume_model.mp4", 
        fps=60)
    
    # Do all Supernova renders
    render_sequence_to_mp4(
        os.path.join(output_folder, 
                     "RenderSequences", 
                     "Supernova_grids_training"),
        "Supernova_grids_training.mp4", 
        text_data=[["iter", (1300, 150)]],
        fps=60
        )
    render_sequence_to_mp4(
        os.path.join(output_folder, 
                     "RenderSequences", 
                     "Supernova_gt"),
        "Supernova_gt.mp4", 
        fps=60)
    '''
    render_sequence_to_mp4(
        os.path.join(output_folder, 
                     "RenderSequences", 
                     "Supernova_grids"),
        "Supernova_grids.mp4", 
        fps=60)
    render_sequence_to_mp4(
        os.path.join(output_folder, 
                     "RenderSequences", 
                     "Supernova_model"),
        "Supernova_model.mp4", 
        fps=60)
   


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained SSR model')
    
    parser.add_argument('--save_folder',default="Vorts_vis_results",
        type=str,help='Folder to save images to')
    
    args = vars(parser.parse_args())

    #model_size_performance_chart()
    #ensemble_comparison(ensemble_results)
    #flat_top_chart()
    #do_all_renders()
    compression_charts()
    
    quit()