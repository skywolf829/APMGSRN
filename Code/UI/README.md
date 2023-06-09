# Python Neural Volume Renderer

This is a python-based neural volume renderer that supports arbitrary PyTorch models.
In our examples and provided models, NGP, fVSRN, and our proposed AMGSRN (as well as ensemble models) can be used for rendering.
In this readme, we'll explain how to use the renderer and how to plug in your own volume SRN.

Click to watch videos:

[![Renderer use](https://img.youtube.com/vi/slepZiP0Ojo/0.jpg)](https://www.youtube.com/watch?v=slepZiP0Ojo)
[![Image-hierarchy checkerboard rendering](https://img.youtube.com/vi/aCPtuteGfSE/0.jpg)](https://www.youtube.com/watch?v=aCPtuteGfSE)

## Installation

Please see the top-level readme for installation instructions. 
We note that this renderer does require CUDA support, as our dependencies use CUDA.
It also requires that you have a ```/Data``` and ```/SavedModels``` folder at the top level directory, with at least one model already saved in the folder.

---

## Data

See our top-level readme for data/pre-trained model downloading. 
The renderer supports neural network models as well as NetCDF volume data.


---

## Renderer application

To run the renderer, run
```
python Code/UI/renderer.py
```

This will start the GUI and load the first available saved neural network from the ```SavedModels``` folder.
The renderer can be resized, moved, fullscreened, etc.
There will be a performance drop with larger windows, as there are more pixels to render.

![Figure unavailable](/Figures/AMGSRN_renderer.jpg "Renderer application")

1. The rendering view.
Inside this area, left click controls rotation, middle click controls panning, and scrollwheel zooms.
The image displayed is a masking of the evaluated pixels and the finest available image from the image hierarchy.
2. Model/Data choice.
Depending on what is chosen, different options are available to load from in the next dropdown. 
Switching this option will automatically load the first model/data available and begin rendering.
3. Model/Data dropdown.
A dropdown listing all available saved models and data you have available to render with.
4. A choice of colormap to use.
We provide a number of colormaps from ParaView with our repository, but you can also export your own custom one from ParaView as well.
This includes opacity, which will be loaded into the opacity transfer function editor.
5. Batch size slider.
When inferring from the model, a number of pixels, their rays, and samples along those rays have to be generated ahead of time for efficient inference.
We call the desired number of samples per network inference the batch.
A smaller batch uses less GPU memory, but will take longer for a full render.
The batch size can be decreased for improved interactivity, or can be increased for shorter total frame times.
We initialize the batch size somewhat low ($2^{20}$), but suggest increasing this if your GPU can handle it. 
Our 2080Ti works best at $2^{23}$ or $2^{24}$ batch size.
Update fps should remain constant for the same batch size, regardless of window resolution.
6. Samples per pixel.
This controls how many samples are taken along each ray. 
We use the maximum corner-to-corner distance and divide that by this samples per pixel setting to get a step size for the renderer.
7. Reset view button.
When dataset/model extents change, you may want to reset your view. 
This button does exactly that, since we don't change your view between models by default, in case you are trying to compare some regions between two models.
8. Opacity transfer function editor.
For exploring features of the data, we use a ParaView-like transfer function editor for opacity specifically.
Click and drag on existing points to change the opacity.
Click (and release) anywhere between two points to create a new control point.
Click a point and then press the delete key to remove a control point.
Endpoints cannot be deleted or moved along the x-axis, and control points cannot move past their neighbors.
Selecting a new color map will reset the opacity transfer function.
9. Data range slider for colormapping. Choose which range of the data to apply the colormap to. (useful if you know the data distribution)
10. Renderer statistics.
Statistics about frame time (the time it takes to fully render the image at full resolution), update framerate (the fps the renderer is updating with progressive rendering) and maximum GPU memory use during rendering.
11. Save image button.
Opens a file browser fiew to pick where to save the current renderer image in PNG format at the resolution of the render view.

---

## Using your own model

To prepare your model to work in the renderer, there are 3 steps:
1. Implement a loading function for your model in ```Code/Models/models.py``` functions ```create_model(opt)``` and ```load_model(opt,device)```.
   - We use a dictionary object ```opt``` to hold various hyperparameters and settings that exist for the network saved to disk separate from the network.
   - If this is not specific enough for a new model, you can also adjust how the options/models are loaded in the ```RendererThread``` object in ```Code/UI/renderer_app.py```, in methods ```initialize_model``` and ```do_change_model```.
   - The device is the string representing the device for PyTorch to load the model to. This is important since PyTorch will default try to load the model to the same place it was last, which may not be possible if it was trained on "cuda:4" previously, and you are trying to open it on a machine with one GPU.
2. Your model must have accessible ```min()```, ```max()```, and ```get_volume_extents()``` functions. These are used by the renderer and the ```Scene``` objects.
   - ```min()``` should return the minimum value in the volume it represents. We set this value before training and store it in the options file. Used mostly for color mapping.
   - ```max()``` should return the maximum value in the volume the model represents. Used for color mapping.
   - ```get_volume_extents()``` should return the [z,y,x] shape of the volume in that order. This is not the normalized $[-1, 1]$ input query range, but the dimensions of the volume before training. Used for the AABB in the renderer.
3. Your model's ```forward(x)``` function must take a batch of input coordinates in the shape $[B, 3]$, where inputs are coordinates in the normalized volume range $[-1, 1]$, and return output values in the shape $[B,1]$. Other output shapes are not supported, as a transfer function is applied to the output $[B,1]$ to generate colors/opacities per output.

With these three things checked, your model should natively work with our renderer.
Add your saved model into our ```SavedModels``` folder (matching our format of having it inside its own folder) and happy visualizing!


---

## Caveats

1. It is required to have a CUDA enabled GPU to do any rendering.
2. The renderer uses only "cuda:0", and does not currently support using any other slots unless that is overwritten in the code. (see init for ```RendererThread``` where it is defined).
3. A 3-channel-coordinate is expected as input, and only single channel output is expected from the network for use in a transfer function, which makes this incompatible with radiance fields networks that may expect 3D coordinate + viewing angle input, and (r,g,b,a) output. 
4. Renderer was only tested with a GTX 1060 3GB, RTX 2080Ti 11GB and an RTX 3070 8GB, so performance on cards older (or newer!) than these is unknown. However, even the GTX 1060 was performant during renders, but smaller batch sizes were necessary to fit in the VRAM.


---
## Known issues/bugs

1. Resizing the window such that the render view has 0 or less pixels instantly crashes the program.
2. Resizing to small heights can cause the right side editor look a bit strange.
3. The image saving feature will save the current frame the renderer is on, even if it is not complete with rendering all pixels.
4. The app may silently crash if you exceed GPU memory, which is easiest to do if you increase the batch size too high with a viewpoint with dense rays.
5. Requires a SavedModels and Data folder at the top level to run, and at least 1 model saved.
6. We notice that the same code and the same models on 2 different systems with different versions of PyTorch (1.13.1 vs 2.0.1) may have the xyz dimensions reversed in our renderer, creating incorrect visualizations for non-cube shaped data, likely due to some change in order returned from a meshgrid function. We add a boolean to our scene in ```renderer.py``` line 418 that reverses the dimensions. Please enable this boolean if you experience this issue.