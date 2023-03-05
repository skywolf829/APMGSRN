# AMGSRN

This repo contains code for adaptive multi-grid scene representation network (AMGSRN), an ensemble training routine for large-scale data, and a neural volume renderer.
Materials are prepared for submission to VIS2023 for our paper titled "Adaptive Multi-Grid Scene Representation Networks for Large-Scale Data Visualization", submission ID 1036, submitted on March 31, 2023.
Included is all code used to train networks giving performance metrics shown in our submitted manuscript.

## Installation

We recommend the use of conda for management of Python version and packages. To install, run the following:
```
conda env create --file environment.yml
conda activate AMGSRN
```

Creating the environment will take a while. If the above fails, use this as a backup:
```
conda create --name NeuralStreamFunction python=3.9
conda activate NeuralStreamFunction
```
and then do ```conda install packageName1 packageName2 ...``` for each package name in env.yml.

Once thats finished and the environment has been activated, navigate to https://pytorch.org/get-started/locally/ and follow instructions to install pytorch on your machine.

For instance:
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

was the command we used to install pytorch on our Windows machine for testing.

This code has been tested on:

Windows 11/WSL with Python 3.9.12, Pytorch 1.13.1 (with CUDA 11.6)

Ubuntu 20.04.4 LTS with Python 3.9.13, Pytorch 1.12.1 (with CUDA 11.4)

A number of additional packaged should be installed with pip, including TODO, TODO, TODO.
Install these with

```
pip install -r requirements.txt
```

### Data

A few of our datasets are too large to be hosted for download, but can independently be downloaded from Johns Hopkins Turbulence Database: https://turbulence.pha.jhu.edu/.
However, we do host 3 smaller-scale datasets and provide pretrained models for all models tested in the paper in an anonymous Google Drive folder: https://drive.google.com/file/d/1FXRxMdcJ53cdeZ6mlAyDI254IvZ2OFoo/view?usp=sharing.
Extract the folder and make sure the ```Data``` and ```SavedModels``` folders are at the same directory as ```Code```. ```/Data``` hosts the volume data as NetCDF files, which can readily be visualized in ParaView, and ```/SavedModels``` is where all the models are saved and loaded from.

## Training and Testing Use

### ```start_jobs.py```
This script is responsible for starting a set of jobs hosted in a JSON file in ```/Code/BatchRunSettings```, and issuing each job to available GPUs on the system. The jobs in the JSON file can be training (```train.py```) or testing (```test.py```), and one job will be addressed to each device available for training/testing. When a job completes on a device, the device is released and becomes available for other jobs to be designated that device. The jobs are not run in sequential order unless you only have 1 device, so do not expect this script to train+test a model sequentially unless you use only one device. Please see ```/Code/BatchRunSettings``` for examples of settings files - each job to run is either ```train.py``` or ```test.py``` with the command line arguments for those files. When an ensemble model is trained, ```start_jobs.py``` is responsible for splitting the training of the volume into a grid of models belonging to an ensemble, including domain partitioning with ghost cells.

Command line arguments are:

```--settings```: the .json file (located in /Code/Batch_run_settings/) with the training/testing setups to run. See the examples in the folder for how to create a new settings file. Required argument.

```--devices```: the list of devices (comma separated) to issue jobs to. By default, "all" available CUDA devices are used. If no CUDA devices are detected, the CPU is used. 

```--data_devices```: the device to host the data on. In some cases, the data may be too large to host on the same GPU that training is happening on, using system RAM instead of GPU VRAM may be preferred. Options are "same", implying using the same device for the data as is used for the model, and "cpu", which puts the data on system RAM. Default is "same".

#### Example training/testing:

The following will run the jobs defined in example_file.json on all available CUDA devices (if available) or the CPU if no CUDA devices are detected by PyTorch. The vector field data will be hosted on the same device that the models train on.

```python Code/start_jobs.py --settings example_file.json```

The following will run the jobs defined in example_file.json on cuda devices 1, 2, 4, and 7, with the data hosted on the same device.

```python Code/start_jobs.py --settings example_file.json --devices cuda:1,cuda:2,cuda:4,cuda:7```

The following will run the jobs defined in example_file.json on cuda devices 1 and 2, with the data hosted on the system memory.

```python Code/start_jobs.py --settings example_file.json --devices cuda:1,cuda:2 --data_devices cpu```

The following will run the jobs defined in example_file.json on cuda devices 1 and cpu, with the data hosted on the same devices as the model.

```python Code/start_jobs.py --settings example_file.json --devices cuda:1,cpu```

(For M1 Macs with MPS) - The following will run the jobs defined in example_file.json on MPS (metal performance shaders), which is Apple's hardware for acceleration. Many PyTorch functions are not yet implemented for MPS as of Torch version 1.13.1, and as such our code cannot natively run on MPS at this time, but as more releases of PyTorch come out, we expect this to run without issue in the future.

```python Code/start_jobs.py --settings example_file.json --devices mps```

### ```train.py```
This script will begin training a defined model with chosen hyperparameters and selected volume with extents. Default hyperparemeter values are what is shown in ```/Code/Models/options.py```, and will only be changed if added as a command line argument when running ```train.py```. A description of each argument can be seen by running ```python Code/train.py --help```. The only data format currently supported is NetCDF, although you could add your own dataloader as well.

Example of training an AMGSRN model on the plume data:

```python Code/train.py --model fSRN --data Plume.nc```

More examples of usage of this code are encoded into the settings JSON files which are used by ```start_jobs.py```, and we recommend not launching ```train.py``` from the command line yourself, and instead letting ```start_jobs.py``` do it.

### ```test.py```
This script is responsible for the testing of trained models for PSNR, error, and reconstruction. Output is saved to the correct folder in```/Output/``` depending on task chosen. Similar to ```train.py```, it is usually ran from our ```start_jobs.py``` script, but can also be ran on its own. A description of each command line argument can be seen by running ```python Code/test.py --help```. Just as with ```train.py```, only NetCDF file formats are supported for loading data to evaluate against.

Example of testing a trained model named plume:

```python Code/Tests/test.py --load_from plume --tests_to_run psnr,reconstruction```

## Renderer Use

Our renderer uses a Python backend and a HTML/JS frontend. To start the Python server, run ```flask Code/Renderer/app.py```, which will host serve to localhost at endpoint 5000. In a web browser, navigate to localhost:5000 to view the renderer. If the port is open, you can connect to it from other machines as well, such as a laptop that is not CUDA-accelerated.

Inside the renderer, you can choose which model to load to render with as well as transfer function. Rotate the camera by clicking and dragging, pan the camera with middle mouse clicking and dragging, and zoom with mousewheel. Our progressive renderer will update the image in a checkerboard pattern with mip maps to give a resonable view of the render while pixels are still being evaluated on the backend.

