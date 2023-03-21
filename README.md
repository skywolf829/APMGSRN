# AMGSRN

![Image unavailable](/Figures/AMGSRN_teaser.jpg "Adaptive grids fitting to the data")

This repo contains code for adaptive multi-grid scene representation network (AMGSRN), an ensemble training routine for large-scale data, and a neural volume renderer.
Materials are prepared for submission to VIS2023 for our paper titled "Adaptive Multi-Grid Scene Representation Networks for Large-Scale Data Visualization", submission ID 1036, submitted on March 31, 2023.
Included is all code used to train networks giving performance metrics shown in our submitted manuscript. A CUDA accelerated device, preferably a NVidia 2060 or newer, is required.

![Image unavailable](/Figures/AMGSRN_architecture.jpg "Network architecture")

## Architecture

Our model encodes 3D input coordinates to a high-dimensional feature space using a set of adaptive feature grids.
During training, the adaptive grids learn to scale, translate, share, and rotate to overlap the features of the grids with regions of high complexity, helping the model perform better in areas that need more network parameters.
After finding a good spot, the grids freeze, allowing the decoder to fine-tune the feature grids to their spatial position.

## Videos

Click to watch videos:

[![AMGSRN grids during training](https://img.youtube.com/vi/utYqmFmyRaE/0.jpg)](https://www.youtube.com/watch?v=utYqmFmyRaE)
[![AMGSRN grids during training](https://img.youtube.com/vi/mFsk2LYAJ-E/0.jpg)](https://www.youtube.com/watch?v=mFsk2LYAJ-E)

Timestamps are available on YouTube for jumping to points of interest.

## Installation

We recommend the use of conda for management of Python version and packages. To install, run the following:
```
conda env create --file environment.yml
conda activate AMGSRN
```

Once thats finished (could take a minute depending on system) and the environment has been activated, navigate to https://pytorch.org/get-started/locally/ and follow instructions to install pytorch on your machine.

For instance:
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

was the command we used to install pytorch on our Windows machine for testing.

Next, install a few pip packages (mostly needed for the renderer) with 
```
pip install -r requirements.txt
```

We also highly recommend using TinyCUDA-NN (TCNN) for improved decoder performance if you have TensorCores available.
In our paper, all models used TCNN and see a significant speedup and lower memory requirment due to half precision training and the fully-fused MLP.
See installation guide on their github: https://github.com/NVlabs/tiny-cuda-nn.
Installation on Linux (or WSL) is straightforward, but Windows requires more effort.
For this reason, we highly recommend Windows users use WSL, as there are no performance decreases, but the OS is more suited for the existing packages and enviroments.
With or without TCNN, our code should automatically detects if you have it installed  and uses it if available.
For linux/WSL, the following will install TCNN:
```
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
Regardless of if a model was trained with/without TCNN, the machine it is loaded on will convert it to PyTorch if necessary.

Lastly, install nerfacc: https://github.com/KAIR-BAIR/nerfacc.
Preferably use the pre-built wheels for your torch+cuda version, as compiling from the latest release occasionally has some bugs (which we find will still be okay if you just run the same code again).
We use torch 1.13 and cuda 11.7, so we install with:
```
pip install nerfacc -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-1.13.0_cu117.html
```

The training/testing code has been tested on:

- Windows 11 using WSL with Python 3.9.12, Pytorch 1.13.1 (with CUDA 11.7)
- Ubuntu 20.04.4 LTS with Python 3.9.13, Pytorch 1.12.1 (with CUDA 11.4)

### Verify installation

After all the packages are downloaded, try to do a small training run. 
First, see our Data section to download three of our scalar fields used in our paper.
Then, run:

```
python Code/start_jobs.py --settings train.json
```

This will train a model on the plume dataset for 10000 iterations.
A training/testing log for every trained model can be found in ```/SavedModels/modelname/train_log.txt``` and ```/SavedModels/modelname/test_log.txt```.
Our 2080Ti machine with and without TCNN trains in about 1 minute.

Next, test the model with:
```
python Code/start_jobs.py --settings test.json
```

In the test log, you should see the throughput (154/302 million point per second without/with TCNN on our 2080Ti), and the trained PSNR (about 53 dB for us).
Performance may vary based on computer load, feel free to run multiple times to see outputs.
On smaller graphics cards, you may have to reduce the batch size used for tests
This test will also save a reconstructed scalar field sampled from the network at the same resolution of the original data in ```Output/Reconstruction/temp.nc```, which may be readily visualized in Paraview.

To check that the offline renderer works:

```
python Code/renderer.py --load_from temp
```

This will render a 512x512 image with 256 samples per ray and save it to ```/Output/render.png```.
It will also render 10 more frames and give you an average framerate and print out frame times (24/31 fps on our machine without/with TCNN).

The renderer with GUI is the last check:

```
python Code/UI/renderer_app.py
```

which should automatically load the first model in the SavedModels folder and begin rendering.

---

## Data

A few of our datasets are too large to be hosted for download, but can independently be downloaded from Johns Hopkins Turbulence Database: https://turbulence.pha.jhu.edu/.
However, we do host 3 smaller-scale datasets and provide pretrained models for all models tested in the paper in an anonymous Google Drive folder: https://drive.google.com/file/d/1FXRxMdcJ53cdeZ6mlAyDI254IvZ2OFoo/view?usp=sharing (~3GB).
Extract the folder and make sure the ```Data``` and ```SavedModels``` folders are at the same directory level as the ```Code``` folder. ```Data``` hosts the volume data as NetCDF files, which can readily be visualized in ParaView, and ```SavedModels``` is where all the models are saved and loaded from.

---

## Training and Testing Use

### ```start_jobs.py```
This script is where jobs, both training and testing, get started from.
In all situations, it is preferred to use this rather than directly launch ```train.py``` or ```test.py```.
This script will start a set of jobs hosted in a JSON file in ```/Code/BatchRunSettings```, and issuing each job to available GPUs (or cpu/mps) on the system. 
The jobs in the JSON file can be training (```train.py```) or testing (```test.py```), and one job will be addressed to each device available. 
When a job completes on a device, the device is released and becomes available for other jobs to be designated that device. 
The jobs are not run in sequential order unless you only have 1 device, so do not expect this script to train+test a model sequentially unless you use only one device. 
Please see ```/Code/BatchRunSettings``` for examples of settings files - each job to run is either ```train.py``` or ```test.py``` with the command line arguments for those files. 
When an ensemble model is trained, ```start_jobs.py``` is responsible for splitting the training of the volume into a grid of models belonging to an ensemble, including domain partitioning with ghost cells.

Command line arguments are:

```--settings```: the .json file (located in ```/Code/Batch_run_settings/```) with the training/testing setups to run. See the examples in the folder for how to create a new settings file. Required argument.

```--devices```: the list of devices (comma separated) to issue jobs to. By default, "all" available CUDA devices are used. If no CUDA devices are detected, the CPU is used. Can also be "mps" for metal performance shaders with PyTorch on an M1 or M2 Mac.

```--data_devices```: the device to host the data on. In some cases, the data may be too large to host on the same GPU that training is happening on, using system RAM instead of GPU VRAM may be preferred. Options are "same", implying using the same device for the data as is used for the model, and "cpu", which puts the data on system RAM. Default is "same".

#### Example training/testing:

The following will run the jobs defined in example_file.json on all available CUDA devices (if available) or the CPU if no CUDA devices are detected by PyTorch. The vector field data will be hosted on the same device that the models train on.

```python Code/start_jobs.py --settings example_file.json```

The following will run the jobs defined in example_file.json on cuda devices 1, 2, 4, and 7, with the data hosted on the same device.

```python Code/start_jobs.py --settings example_file.json --devices cuda:1,cuda:2,cuda:4,cuda:7```

The following will run the jobs defined in example_file.json on cuda devices 1 and 2, with the data hosted on the system memory.

```python Code/start_jobs.py --settings example_file.json --devices cuda:1,cuda:2 --data_devices cpu```

The following will run the jobs defined in example_file.json on cuda devices 3 and cpu, with the data hosted on the same devices as the model.

```python Code/start_jobs.py --settings example_file.json --devices cuda:3,cpu```

(For M1 Macs with MPS) - The following will run the jobs defined in example_file.json on MPS (metal performance shaders), which is Apple's hardware for acceleration. 
Many PyTorch functions are not yet implemented for MPS as of Torch version 1.13.1, and as such our code cannot natively run on MPS at this time, but as more releases of PyTorch come out, we expect this to run without issue in the future.

```python Code/start_jobs.py --settings example_file.json --devices mps```

### ```train.py```
This script will begin training a defined model with chosen hyperparameters and selected volume with extents. 
Default hyperparemeter values are what is shown in ```/Code/Models/options.py```, and will only be changed if added as a command line argument when running ```train.py```. 
A description of each argument can be seen by running ```python Code/train.py --help```. 
The only data format currently supported is NetCDF, although you could add your own dataloader in ```Code/Datasets/datasets.py``` as well.

Example of training an AMGSRN model on the plume data:

```python Code/train.py --model AMGSRN --data Plume.nc```

More examples of usage of this code are encoded into the settings JSON files which are used by ```start_jobs.py```, and we recommend not launching ```train.py``` from the command line yourself, and instead letting ```start_jobs.py``` do it.

### ```test.py```
This script is responsible for the testing of trained models for PSNR, error, and reconstruction. 
Output is saved to the correct folder in```/Output/``` depending on task chosen. 
Similar to ```train.py```, it is usually ran from our ```start_jobs.py``` script, but can also be ran on its own. 
A description of each command line argument can be seen by running ```python Code/test.py --help```. 
Just as with ```train.py```, only NetCDF file formats are supported for loading data to evaluate against.

Example of testing a trained model named plume:

```python Code/Tests/test.py --load_from plume --tests_to_run psnr,reconstruction```

---

## Renderer Use

We have two ways to access our renderer. 
An offline command-line version is available at at ```Code/renderer.py```, which we will explain in this document.
The online renderer with a GUI is in ```Code/UI```. 
Please see that folder's README if you'd like to use the realtime renderer with interactivity.

For offline single-frame rendering of a network or NetCDF file, see the descriptions of command line arguments with

```python Code/renderer.py --help```

Noteable options are:
1. ```--hw```: the height and width for the output image, separated by comma
2. ```--load_from```: the model name to load from for the render
3. ```--spp```: samples per ray on a pixel
4. ```--azi, --polar, --dist```: viewing parameters for the arcball camera
5. ```colormap```: pick the colormap json file from ```Colormaps``` folder

Examples:

```python Code/renderer.py --hw 1024,1024 --spp 512 --azi 45 --polar 45 --load_from Supernova_AMGSRN_small --colormap Viridis.json```
```python Code/renderer.py --azi 45 --polar 45 --raw_data true --load_from Supernova.nc```

