# Compiling the renderer

First, we need to download and install code for the VTK build with neural rendering available, the C++ libtorch code to run the neural network in C++, and a driver program.

## Downloading dependencies

To install libtorhc, navigate to PyTorch's website and download the latest libtorch library available for your system. https://pytorch.org/get-started/locally/.

To install the VTK fork with neural rendering available and the driver program for the renderer, clone the following two repositories:

```
git clone https://github.com/skywolf829/VTK_meshless-DVR
git clone https://github.com/skywolf829/meshless-DVR.git
```

## Installing dependencies

With the 3 dependencies downloaded, your folder structure should look like this:
```
DistributedINR
|--Code
   |--BatchRunSettings
   |--Datasets
   |--Models
   |--Other
   |--Renderer
      |--libtorch
      |--meshless-DVR
      |--NeuralSRN
      |--VTK_meshless-DVR
|--Data
|--Output
|--SavedModels
|--tensorboard
```

We only need to compile the VTK build and then the meshless-DVR, in that order.

### Installing VTK_meshless-DVR
 
The original instructions are listed in a related repo https://github.com/sunjianxin/VTK_MFA-DVR, which uses a spline-based data fitting model called MFA to reconstruct point queries, and does rendering based on that. We follow the original build instructions with the following commands.

From ```DistributedINR/Code/Renderer/VTK_meshless-DVR```,

```
mkdir build
cd build
ccmake ..
```

This opens the CMake configuration. From here, press 'c' to run the configuration. Afterward, press 'e' to exit (if the configuration went well).

Now, you should see an options screen. Be sure that:
```
BUILD_SHARED_LIBS       ON
CMAKE_BUILD_TYPE        Release
```

With these settings, do configuration two more times with 'c' and then 'e' when its done (twice), then press 'g' to generate the final make files. After generating, ccmake should spit you back out to the command line if no errors occured. Finally, build the library with

```
make install
```

Depending on where you are running this code, you may need sudo privelages. In addition, you can speed up the install with ```-j num_cores``` to multithread the build. For instance

```
sudo make install -j 8
```

### Installing meshless-DVR

This last step is to install the C++ code that creates a GUI and performs rendering using a trained neural network.

# Running the renderer

The C++ volume renderer (using VTK) relies on a TorchScript model. This model type is one step away from what PyTorch natively saves, so a conversion is necessary.
Then, the model can be loaded and evaluated in C++ in our renderer.

## Converting PyTorch model to TorchScript

Select a trained model to convert to torchscript (only AMGSRN and fVSRN models supported currently).

```
python Code/model_to_torchscript.py --model_name trained_model_name_goes_here 
```

Since our models use TinyCUDANN to accelerate training, and because that model is not supported by torchscript, we also add an option that converts TCNN to native PyTorch layers for conversion to torchscript. Currently, only their fullyfused decoder network is supported for this transfer. Use the following argument to enable this conversion:

```
python Code/model_to_torchscript.py --model_name trained_model_name_goes_here --convert_from_tcnn true 
```

Finally, the converted model is saved in the original model's folder as "traced_model.pt", which is the file that will be loaded by the renderer.


# Fixes to build VTK_meshless-DVR

Fix MPI include locations in files for your MPI installation location:
DistributedINR/Code/Renderer/mfa/include/mfa/encode.hpp
DistributedINR/Code/Renderer/mfa/include/diy/include/diy/mpi/config.hpp

Fix MPI angled bracket include in /opt/homebrew/Cellar/mpich/4.1/include/mpi.h:
#include <mpi_proto.h> 
to
#include "mpi_proto.h"

Fix includes in:
DistributedINR/Code/Renderer/VTK_meshless-DVR/external/block.hpp
to be relative paths
#include    "../../mfa/include/mfa/mfa.hpp"
#include    "../../mfa/include/mfa/block_base.hpp"

# Fixes to build example_MFA-DVR

```
cmake .. \
-DCMAKE_CXX_COMPILER=mpicxx \
-DMFA_INCLUDE_DIR=/Users/sky/Documents/Github/DistributedINR/Code/Renderer/mfa/include/ \
-DVTK_DIR;PATH=/Users/sky/Documents/GitHub/DistributedINR/Code/Renderer/VTK_meshless-DVR/build/
```