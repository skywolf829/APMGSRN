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
    
def model_size_performance_chart():
    supernova_results = {
        "Ours": {
            2**16 : 41.907,
            2**20 : 46.610,
            2**24 : 49.710 
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
            2**16 : 50.064,
            2**20 : 57.372,
            2**24 : 59.668
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
            2**16 : 27.700,
            2**20 : 32.222,
            2**24 : 38.111
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
            2**16 : 29.628,
            2**20 : 37.728,
            2**24 : 43.926
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
            2**16 : 35.325,
            2**20 : 39.775,
            2**24 : 43.135
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
    quit()
    
    for scale_factor in results.keys():
        print(scale_factor)

        interp_results = results[scale_factor][interp]


        for metric in interp_results.keys():
            fig = plt.figure()
            y_label = metric

            for SR_type in results[scale_factor].keys():
                # model results plotting
                x = np.arange(args['start_ts'], 
                    args['start_ts'] + args['ts_skip']*len(results[scale_factor][SR_type][metric]),
                    args['ts_skip'])
                y = results[scale_factor][SR_type][metric]
                l = SR_type
                plt.plot(x, y, label=l)

            #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            #    fancybox=True, ncol=4, shadow=True)
            plt.legend()
            plt.xlabel("Timestep")
            plt.ylabel(y_label)

            plt.title(scale_factor + " SR - " + metric)
            plt.savefig(os.path.join(save_folder, scale_factor, metric+".png"),
                        bbox_inches='tight',
                        dpi=200)
            plt.clf()

    # Overall graphs

    averaged_results = {}

    scale_factors = []

    for scale_factor in results.keys():

        scale_factor_int = int(scale_factor.split('x')[0])
        scale_factors.append(scale_factor_int)

        for metric in results[scale_factor][interp].keys():
            for SR_type in results[scale_factor].keys():
                if SR_type not in averaged_results.keys():
                    averaged_results[SR_type] = {}

                if(metric not in averaged_results[SR_type].keys()):
                    averaged_results[SR_type][metric] = []

                averaged_results[SR_type][metric].append(np.median(
                    np.array(results[scale_factor][SR_type][metric])))

    
    for metric in averaged_results[interp].keys():
        fig = plt.figure()
        y_label = metric
        for SR_type in averaged_results.keys():
            # model results plotting
            x = scale_factors
            y = averaged_results[SR_type][metric]
            l = SR_type
            
            plt.plot(x, y, label=l)

        #plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
        #        mode="expand", borderaxespad=0, ncol=4)
        if(metric == "SSIM"):
            plt.xlabel("Scale factor")
        if(args['output_file_name'] == "Isomag2D.results"):
            plt.ylabel(y_label)
        plt.xscale('log')
        plt.minorticks_off()
        plt.xticks(scale_factors, labels=scale_factors)
        #plt.title("Median " + metric + " over SR factors")
        if(metric == "PSNR (dB)"):
            plt.ylim(bottom=20, top=55)
            t = args['output_file_name'].split(".")[0]
            if(t == "Nyx256"):
                t = "Nyx"
            elif(t == "Boussinesq"):
                t = "Heated flow"
            plt.title(t)
        elif(metric == "SSIM"):
            plt.ylim(bottom=0.45, top=1.0)
        plt.savefig(os.path.join(save_folder, "MedianValues", metric+".png"),
            bbox_inches='tight',
            dpi=200)
        #plt.show()
        plt.clf()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    left_y_label = "PSNR (dB)"
    right_y_label = "SSIM"

    for SR_type in averaged_results.keys():
        # model results plotting
        x = scale_factors
        left_y = averaged_results[SR_type][left_y_label]
        right_y = averaged_results[SR_type][right_y_label]
        l = SR_type
        ax1.plot(x, left_y, label=l, marker="s")
        ax2.plot(x, right_y, label=l, marker="^", linestyle='dashed')

    ax1.legend()
    #ax2.legend()
    ax1.set_xlabel("Scale factor")
    ax1.set_ylabel(left_y_label)
    ax2.set_ylabel(right_y_label)

    ax1.set_xscale('log')
    ax1.minorticks_off()
    
    ax1.set_xticks(scale_factors)
    ax1.set_xticklabels(scale_factors)
    ax1.set_title("Median PSNR/SSIM over SR factors")
    plt.savefig(os.path.join(save_folder, "MedianValues", "Combined.png"),
                bbox_inches='tight',
                dpi=200)
    plt.clf()