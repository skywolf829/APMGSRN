import os
import imageio
import argparse
from typing import Union, Tuple
import numpy as np
import zeep
import struct
import base64
import time
import h5py
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed


client = zeep.Client('http://turbulence.pha.jhu.edu/service/turbulence.asmx?WSDL')
ArrayOfFloat = client.get_type('ns0:ArrayOfFloat')
ArrayOfArrayOfFloat = client.get_type('ns0:ArrayOfArrayOfFloat')
SpatialInterpolation = client.get_type('ns0:SpatialInterpolation')
TemporalInterpolation = client.get_type('ns0:TemporalInterpolation')
token="edu.osu.buckeyemail.wurster.18-92fb557b" #replace with your own token

def tensor_to_cdf(t, location, channel_names=None):
    # Assumes t is a tensor with shape (1, c, d, h[, w])
    from netCDF4 import Dataset
    d = Dataset(location, 'w')

    # Setup dimensions
    d.createDimension('x')
    d.createDimension('y')
    dims = ['x', 'y']

    if(len(t.shape) == 5):
        d.createDimension('z')
        dims.append('z')

    # ['u', 'v', 'w']
    if(channel_names is None):
        ch_default = 'a'

    for i in range(t.shape[1]):
        if(channel_names is None):
            ch = ch_default
            ch_default = chr(ord(ch)+1)
        else:
            ch = channel_names[i]
        d.createVariable(ch, np.float32, dims)
        d[ch][:] = t[0,i].clone().detach().cpu().numpy()
    d.close()

def get_frame(x_start, x_end, x_step, 
y_start, y_end, y_step, 
z_start, z_end, z_step, 
sim_name, timestep, field, num_components):
    #print(x_start)
    #print(x_end)
    #print(y_start)
    #print(y_end)
    #print(z_start)
    #print(z_end)
    result=client.service.GetAnyCutoutWeb(token,sim_name, field, timestep,
                                            x_start+1, 
                                            y_start+1, 
                                            z_start+1, 
                                            x_end, y_end, z_end,
                                            x_step, y_step, z_step, 0, "")  # put empty string for the last parameter
    # transfer base64 format to numpy
    nx=int((x_end-x_start)/x_step)
    ny=int((y_end-y_start)/y_step)
    nz=int((z_end-z_start)/z_step)
    base64_len=int(nx*ny*nz*num_components)
    base64_format='<'+str(base64_len)+'f'

    result=struct.unpack(base64_format, result)
    result=np.array(result).reshape((nz, ny, nx, num_components)).swapaxes(0,2)
    return result, x_start, x_end, y_start, y_end, z_start, z_end

def get_full_frame_parallel(x_start, x_end, x_step,
y_start, y_end, y_step, 
z_start, z_end, z_step,
sim_name, timestep, field, num_components, num_workers):
    threads= []
    full = np.zeros((int((x_end-x_start)/x_step), 
    int((y_end-y_start)/y_step), 
    int((z_end-z_start)/z_step), num_components), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        done = 0
        x_len = 128
        y_len = 128
        z_len = 64
        for k in range(x_start, x_end, x_len):
            for i in range(y_start, y_end, y_len):
                for j in range(z_start, z_end, z_len):
                    x_stop = min(k+x_len, x_end)
                    y_stop = min(i+y_len, y_end)
                    z_stop = min(j+z_len, z_end)
                    threads.append(executor.submit(get_frame, 
                    k, x_stop, x_step,
                    i, y_stop, y_step,
                    j, z_stop, z_step,
                    sim_name, timestep, field, num_components))
        for task in as_completed(threads):
           r, x1, x2, y1, y2, z1, z2 = task.result()
           x1 -= x_start
           x2 -= x_start
           y1 -= y_start
           y2 -= y_start
           z1 -= z_start
           z2 -= z_start
           x1 = int(x1 / x_step)
           x2 = int(x2 / x_step)
           y1 = int(y1 / y_step)
           y2 = int(y2 / y_step)
           z1 = int(z1 / z_step)
           z2 = int(z2 / z_step)
           full[x1:x2,y1:y2,z1:z2,:] = r.astype(np.float32)
           del r
           done += 1
           #print("Done: %i/%i" % (done, len(threads)))
    return full

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..", "..")
data_folder = os.path.join(project_folder_path, "Data")
save_folder = os.path.join(data_folder)

name = "isotropic1024coarse"
t0 = time.time()
count = 0
startts = 1
endts = 2
ts_skip = 10
frames = []
for i in range(startts, endts, ts_skip):
    print("TS %i/%i" % (i, endts))
    f = get_full_frame_parallel(0,1024, 1,#x
    0, 1024, 1, #y
    0, 1024, 1, #z
    name, i, 
    "u", 3, 
    16)    
    print(f.shape)
    mags = np.linalg.norm(f, axis=3)
    f *= (1/mags.max())
    #print(f.shape)
    # If 2D do next 2 lines
    # f = f[...,0]
    # print(f.shape)
    f = np.expand_dims(f, 0)
    #f -= f.min()
    #f *= 1/(f.max() + 1e-6)
    f = np.transpose(f, (0, 4, 1, 2, 3))
    print(f.shape)
    #frames.append(f)
    tensor_to_cdf(torch.tensor(f), "vf")
    print("Finished " + str(i))
    count += 1
print("finished")
print(time.time() - t0)

#frames = np.array(frames)
#frames = frames[:,0,:,:]
#imageio.mimwrite("frames.gif", frames)
