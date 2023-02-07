# Compiling the renderer

Point to torch.h from the install of the python you have.

$ mkdir build 
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
$ cmake --build . --config Release

# Running the network test

First make sure your model has been converted to Torchscript.

$ python Code/model_to_torchscript.py --model_name Supernova_AMGSRN_small --convert_from_tcnn true

Then run the test
$ ./model "../../../../SavedModels/Supernova_AMGSRN_small_convert_to_pytorch/traced_model.py"


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

Comment out the Threads::Threads stuff in Common/Systems/Core/CMakeLists.txt

```
# vtk_module_link(VTK::CommonSystem
#  PRIVATE
#    $<$<PLATFORM_ID:WIN32>:wsock32>
#    Threads::Threads)
```

# Fixes to build example_MFA-DVR

```
cmake .. \
-DCMAKE_CXX_COMPILER=mpicxx \
-DMFA_INCLUDE_DIR=/Users/sky/Documents/Github/DistributedINR/Code/Renderer/mfa/include/ \
-DVTK_DIR;PATH=/Users/sky/Documents/GitHub/DistributedINR/Code/Renderer/VTK_meshless-DVR/build/
```