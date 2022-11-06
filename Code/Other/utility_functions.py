import numpy as np
import torch
from torch.nn import functional as F
from matplotlib.pyplot import cm
from math import exp
from typing import Optional
import argparse
import os
from netCDF4 import Dataset
import pickle
import h5py
import numba as nb
  
def reset_grads(model,require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        if(m.weight is not None):
            m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        if(m.weight is not None):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

def gaussian(window_size : int, sigma : float) -> torch.Tensor:
    gauss : torch.Tensor = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x \
        in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size : torch.Tensor, channel : int) -> torch.Tensor:
    _1D_window : torch.Tensor = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window : torch.Tensor = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window : torch.Tensor = torch.Tensor(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = torch.Tensor(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def PSNR(x, y, range = torch.tensor(1.0, dtype=torch.float32)):
    range = range.to(x.device)
    return 20*torch.log10(range) - \
        10*torch.log10(((y-x)**2).mean())
        
def _ssim(img1 : torch.Tensor, img2 : torch.Tensor, window : torch.Tensor, 
window_size : torch.Tensor, channel : int, size_average : Optional[bool] = True):
    mu1 : torch.Tensor = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 : torch.Tensor = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq : torch.Tensor = mu1.pow(2)
    mu2_sq : torch.Tensor = mu2.pow(2)
    mu1_mu2 : torch.Tensor = mu1*mu2

    sigma1_sq : torch.Tensor = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq : torch.Tensor = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 : torch.Tensor = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 : float = 0.01**2
    C2 : float= 0.03**2

    ssim_map : torch.Tensor = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    ans : torch.Tensor = torch.Tensor([0])
    if size_average:
        ans = ssim_map.mean()
    else:
        ans = ssim_map.mean(1).mean(1).mean(1)
    return ans

def _ssim_3D_distributed(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1.to("cuda:2"), window.to("cuda:2"), padding = window_size//2, groups = channel).to("cuda:2")
    mu2 = F.conv3d(img2.to("cuda:3"), window.to("cuda:3"), padding = window_size//2, groups = channel).to("cuda:3")

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1.to("cuda:4")*mu2.to("cuda:4")

    sigma1_sq = F.conv3d(img1.to("cuda:4")*img1.to("cuda:4"), window.to("cuda:4"), padding = window_size//2, groups = channel).to("cuda:2") - mu1_sq.to("cuda:2")
    sigma2_sq = F.conv3d(img2.to("cuda:5")*img2.to("cuda:5"), window.to("cuda:5"), padding = window_size//2, groups = channel).to("cuda:3") - mu2_sq.to("cuda:3")
    sigma12 = F.conv3d(img1.to("cuda:6")*img2.to("cuda:6"), window.to("cuda:6"), padding = window_size//2, groups = channel) - mu1_mu2.to("cuda:6")

    C1 = 0.01**2
    C2 = 0.03**2

    #ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
    mu1_sq += mu2_sq.to("cuda:2")
    mu1_sq += C1

    sigma1_sq += sigma2_sq.to("cuda:2")
    sigma1_sq += C2

    mu1_sq *= sigma1_sq

    mu1_mu2 *= 2
    mu1_mu2 += C1

    sigma12 *= 2
    sigma12 += C2

    mu1_mu2 *= sigma12.to("cuda:4")

    mu1_mu2 /= mu1_sq.to("cuda:4")

    if size_average:
        return mu1_mu2.mean().to('cuda:0')
    else:
        return mu1_mu2.mean(1).mean(1).mean(1).to('cuda:0')

def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq : torch.Tensor = mu1.pow(2)
    mu2_sq : torch.Tensor = mu2.pow(2)
    mu1_mu2 : torch.Tensor = mu1*mu2

    sigma1_sq : torch.Tensor = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq : torch.Tensor = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 : torch.Tensor = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 : float = 0.01**2
    C2 : float = 0.03**2

    ssim_map : torch.Tensor = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
    ans : torch.Tensor = torch.Tensor([0])
    if size_average:
        ans = ssim_map.mean()
    else:
        ans = ssim_map.mean(1).mean(1).mean(1)
    return ans

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def ssim3D(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)

def ssim3D_distributed(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim_3D_distributed(img1, img2, window, window_size, channel, size_average)

def toImg(data, renorm_channels = True):
    #print("In to toImg: " + str(data.shape))
    if(renorm_channels):
        for c in range(data.shape[0]):
            data[c] -= data[c].min()
            data[c] *= 1 / data[c].max()
    if(len(data.shape) == 3):
        im =  cm.coolwarm(data[0])
        im *= 255
        im = im.astype(np.uint8)
    elif(len(data.shape) == 4):
        im = toImg(data[:,:,:,int(data.shape[3]/2)], renorm_channels)
    #print("Out of toImg: " + str(im.shape))
    
    return im

def bilinear_interpolate(im, x, y):
    dtype = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
    
    x0 = torch.floor(x).type(dtype_long)
    x1 = x0 + 1
    
    y0 = torch.floor(y).type(dtype_long)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[2]-1)
    x1 = torch.clamp(x1, 0, im.shape[2]-1)
    y0 = torch.clamp(y0, 0, im.shape[3]-1)
    y1 = torch.clamp(y1, 0, im.shape[3]-1)
    
    Ia = im[0, :, x0, y0 ]
    Ib = im[0, :, x1, y0 ]
    Ic = im[0, :, x0, y1 ]
    Id = im[0, :, x1, y1 ]
    wa = (x1.type(dtype)-x) * (y1.type(dtype)-y)
    wb = (x1.type(dtype)-x) * (y-y0.type(dtype))
    wc = (x-x0.type(dtype)) * (y1.type(dtype)-y)
    wd = (x-x0.type(dtype)) * (y-y0.type(dtype))
    return Ia*wa + Ib*wb + Ic*wc + Id*wd


def trilinear_interpolate(im, x, y, z, device, periodic=False):

    if(device == "cpu"):
        dtype = torch.float
        dtype_long = torch.long
    else:
        dtype = torch.cuda.FloatTensor
        dtype_long = torch.cuda.LongTensor

    x0 = torch.floor(x).type(dtype_long)
    x1 = x0 + 1
    
    y0 = torch.floor(y).type(dtype_long)
    y1 = y0 + 1

    z0 = torch.floor(z).type(dtype_long)
    z1 = z0 + 1
    
    if(periodic):
        x1_diff = x1-x
        x0_diff = 1-x1_diff  
        y1_diff = y1-y
        y0_diff = 1-y1_diff
        z1_diff = z1-z
        z0_diff = 1-z1_diff

        x0 %= im.shape[2]
        y0 %= im.shape[3]
        z0 %= im.shape[4]

        x1 %= im.shape[2]
        y1 %= im.shape[3]
        z1 %= im.shape[4]
        
    else:
        x0 = torch.clamp(x0, 0, im.shape[2]-1)
        x1 = torch.clamp(x1, 0, im.shape[2]-1)
        y0 = torch.clamp(y0, 0, im.shape[3]-1)
        y1 = torch.clamp(y1, 0, im.shape[3]-1)
        z0 = torch.clamp(z0, 0, im.shape[4]-1)
        z1 = torch.clamp(z1, 0, im.shape[4]-1)
        x1_diff = x1-x
        x0_diff = x-x0    
        y1_diff = y1-y
        y0_diff = y-y0
        z1_diff = z1-z
        z0_diff = z-z0
    
    c00 = im[0,:,x0,y0,z0] * x1_diff + im[0,:,x1,y0,z0]*x0_diff
    c01 = im[0,:,x0,y0,z1] * x1_diff + im[0,:,x1,y0,z1]*x0_diff
    c10 = im[0,:,x0,y1,z0] * x1_diff + im[0,:,x1,y1,z0]*x0_diff
    c11 = im[0,:,x0,y1,z1] * x1_diff + im[0,:,x1,y1,z1]*x0_diff

    c0 = c00 * y1_diff + c10 * y0_diff
    c1 = c01 * y1_diff + c11 * y0_diff

    c = c0 * z1_diff + c1 * z0_diff
    return c   

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def make_coord_grid(shape, device, flatten=True, align_corners=False):
    """ 
    Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        left = -1.0
        right = 1.0
        if(align_corners):
            r = (right - left) / (n-1)
            seq = left + r * \
            torch.arange(0, n, 
            device=device, 
            dtype=torch.float32).float()

        else:
            r = (right - left) / (n+1)
            seq = left + r + r * \
            torch.arange(0, n, 
            device=device, 
            dtype=torch.float32).float()

        coord_seqs.append(seq)

    ret = torch.meshgrid(*coord_seqs, indexing="ij")
    ret = torch.stack(ret, dim=-1)
    if(flatten):
        ret = ret.view(-1, ret.shape[-1])
    return ret.flip(-1)

def save_obj(obj,location):
    with open(location, 'wb') as f:
        pickle.dump(obj, f, pickle.DEFAULT_PROTOCOL)

def load_obj(location):
    with open(location, 'rb') as f:
        return pickle.load(f)

def create_path(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            print(f"Creation of the directory {path} failed")
    
def create_folder(start_path, folder_name):
    f_name = folder_name
    full_path = os.path.join(start_path, f_name)
    if not os.path.exists(full_path):
        try:
            os.makedirs(full_path)
        except OSError:
            print("Creation of the directory %s failed" % full_path)
    else:
        #print_to_log_and_console("%s already exists, overwriting save " % (f_name))
        full_path = os.path.join(start_path, f_name)
    return f_name

def tensor_to_cdf(t, location, channel_names=None):
    # Assumes t is a tensor with shape (1, c, d, h[, w])

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

def nc_to_tensor(location):
    import netCDF4 as nc
    f = nc.Dataset(location)
    channels = []
    for a in f.variables:
        d = np.array(f[a])
        channels.append(d)
    d = np.stack(channels)
    d = torch.tensor(d).unsqueeze(0)
    return d
        
def cdf_to_tensor(location, channel_names):
    # Assumes t is a tensor with shape (1, c, d, h[, w])

    d = Dataset(location, 'r')

    chans = []
    for name in channel_names:
        chans.append(torch.tensor(d[name][:]))
    chans = torch.stack(chans)
    d.close()
    return chans.unsqueeze(0)

def tensor_to_h5(t, location):
    h = h5py.File(location, mode='w')
    h['data'] = t[0].clone().detach().cpu().numpy()
    h.close()

# x,y,z coordiantes either numpy / vtk array 
def get_vtr(dims, xCoords, yCoords, zCoords, 
            scalar_fields={}, vector_fields={}):
    
    import vtk
    from vtkmodules.util import numpy_support  
    assert type(xCoords) == type(yCoords) and type(yCoords) == type(zCoords)
    assert isinstance(xCoords, np.ndarray) or isinstance(xCoords, vtk.vtkDataArray)


    grid = vtk.vtkRectilinearGrid()
    grid.SetDimensions(dims)

    if isinstance(xCoords, np.ndarray):
        xCoords = numpy_support.numpy_to_vtk(xCoords)
        yCoords = numpy_support.numpy_to_vtk(yCoords)
        zCoords = numpy_support.numpy_to_vtk(zCoords)

    grid.SetXCoordinates(xCoords)
    grid.SetYCoordinates(yCoords)
    grid.SetZCoordinates(zCoords)

    pd = grid.GetPointData()
    
    for i, (k, v) in enumerate(scalar_fields.items()):
        vtk_array = numpy_support.numpy_to_vtk(v)
        vtk_array.SetName(k)
        if i == 0:
            pd.SetScalars(vtk_array)
        else:
            pd.AddArray(vtk_array)

    for i, (k, v) in enumerate(vector_fields.items()):
        vtk_array = numpy_support.numpy_to_vtk(v)
        vtk_array.SetName(k)
        if i == 0:
            pd.SetVectors(vtk_array)
        else:
            pd.AddArray(vtk_array)
    return grid

def solution_to_cdf(data, location, channel_names = ['data']):
    '''
    Saves a 3D grid of data as a NetCDF file.

    data: a numpy array of shape [channels, depth, height, width]
        to conform to axis ordering c, z, y, x. 
    location: the location on disc to save the file to
    channel_names: a list of channel names for each
        channel in data
    '''
    assert data.shape[0] == len(channel_names), \
        "data.shape[0] should equal len(channel_names)"
    assert len(data.shape) == 4, \
        "len(data.shape) should equal 4, for [c, d, h, w]"

    d = Dataset(location, 'w')

    # Setup dimensions
    d.createDimension('z')
    d.createDimension('y')
    d.createDimension('x')

    # Put data into the NetCDF file
    for i in range(data.shape[1]):
        d.createVariable(channel_names[i], 
            data.dtype, ('z', 'y', 'x'))
        d[channel_names[i]][:] = data[i]
    d.close()

def normal(vf, b=None, normalize=True):
    # vf: [1, 3, d, h, w]
    # jac: [1, 3, 3, d, h, w]
    
    if b is None:
        b = binormal(vf)
    b = b.squeeze().flatten(1).permute(1,0).unsqueeze(2)
    n = torch.cross(b,
        vf[0].permute(1, 2, 3, 0).flatten(0, 2).unsqueeze(2))
    n = n.squeeze().permute(1,0).reshape(
        vf.shape[1], vf.shape[2],
        vf.shape[3], vf.shape[4]).unsqueeze(0)
    if(normalize):
        n /= (n.norm(dim=1) + 1e-8)
    return n

    
def binormal(vf, jac=None, normalize=True):
    # vf: [1, 3, d, h, w]
    # jac: [1, 3, 3, d, h, w]
    if jac is None:
        jac = jacobian(vf, normalize=normalize)
    Jt = torch.bmm(jac[0].permute(2, 3, 4, 0, 1).flatten(0, 2), 
        vf[0].permute(1, 2, 3, 0).flatten(0, 2).unsqueeze(2))
    b = torch.cross(Jt,
        vf[0].permute(1, 2, 3, 0).flatten(0, 2).unsqueeze(2))
    #b = torch.cross(vf[0].permute(1, 2, 3, 0).flatten(0, 2).unsqueeze(2),
    #    Jt)
    b = b.squeeze().permute(1,0).reshape(
        vf.shape[1], vf.shape[2],
        vf.shape[3], vf.shape[4]).unsqueeze(0)
    if(normalize):
        b /= (b.norm(dim=1) + 1e-8)
    return b

def jacobian(data, normalize=True):
    # Takes [b, c, d, h, w]
    jac = []
    for i in range(data.shape[1]):
        grads = []
        for j in range(len(data.shape)-2):
            g = spatial_gradient(data, i, j)
            grads.append(g)
        jac.append(torch.cat(grads, dim=1))
    jac = torch.cat(jac, dim=0).unsqueeze(0)
    if(normalize):
        jac /= (data.norm(dim=1) + 1e-8)
    return jac

def curl(data):
    dwdy = spatial_gradient(data,2,1)
    dvdz = spatial_gradient(data,1,0)
    
    dudz = spatial_gradient(data,0,0)
    dwdx = spatial_gradient(data,2,2)
    
    dvdx = spatial_gradient(data,1,2)
    dudy = spatial_gradient(data,0,1)
    
    x = dwdy - dvdz
    y = dudz - dwdx
    z = dvdx - dudy
    return torch.stack([x,y,z], dim=1)

def spatial_gradient(data, channel, dimension):
    # takes the gradient along dimension in channel
    # expects data to be [b, c, d, h, w]
    data_padded = F.pad(data[:,channel:channel+1], 
        [1, 1, 1, 1, 1, 1],
        mode = "replicate")
    
    data_padded[:,:,0] += data_padded[:,:,1] - \
        data_padded[:,:,2]
    data_padded[:,:,:,0] += data_padded[:,:,:,1] - \
        data_padded[:,:,:,2]
    data_padded[:,:,:,:,0] += data_padded[:,:,:,:,1] - \
        data_padded[:,:,:,:,2]

    data_padded[:,:,-1] += -data_padded[:,:,-2] + \
        data_padded[:,:,-3]
    data_padded[:,:,:,-1] += -data_padded[:,:,:,-2] + \
        data_padded[:,:,:,-3]
    data_padded[:,:,:,:,-1] += -data_padded[:,:,:,:,-2] + \
        data_padded[:,:,:,:,-3]

    if(dimension == 2):
        weights = torch.tensor(
            [[[0, 0, 0], 
            [0, -0.5, 0],
            [0, 0, 0]],

            [[0, 0, 0], 
            [0, 0, 0], 
            [0, 0, 0]],

            [[0, 0, 0], 
            [0, 0.5, 0], 
            [0, 0, 0]]]
            ).to(data.device).type(torch.float32)
    elif(dimension == 1):        
        # the second (b) axis in [a, b, c]
        weights = torch.tensor([
            [[0, 0, 0], 
            [0, 0, 0], 
            [0, 0, 0]],

            [[0, -0.5, 0], 
            [0, 0, 0], 
            [0, 0.5, 0]],

            [[0, 0, 0], 
            [0, 0, 0], 
            [0, 0, 0]]]
            ).to(data.device).type(torch.float32)
    elif(dimension == 0):
        # the third (c) axis in [a, b, c]
        weights = torch.tensor([
            [[0, 0, 0], 
            [0, 0, 0], 
            [0, 0, 0]],

            [[0, 0, 0], 
            [-0.5, 0, 0.5], 
            [0, 0, 0]],

            [[0, 0, 0], 
            [0, 0,  0], 
            [ 0, 0, 0]]]
            ).to(data.device).type(torch.float32)
    weights = weights.view(1, 1, 3, 3, 3)
    output = F.conv3d(data_padded, weights)

    return output

#Modified Code from Scipy-source
#https://github.com/scipy/scipy/blob/master/scipy/spatial/_hausdorff.pyx
#Copyright (C)  Tyler Reddy, Richard Gowers, and Max Linke, 2016
#Copyright © 2001, 2002 Enthought, Inc.
#All rights reserved.

#Copyright © 2003-2013 SciPy Developers.
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following 
#disclaimer in the documentation and/or other materials provided with the distribution.
#Neither the name of Enthought nor the names of the SciPy Developers may be used to endorse or promote products derived 
#from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, 
#BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
#IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, 
#OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
#OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
@nb.njit()
def directed_hausdorff_nb(ar1, ar2):
    N1 = ar1.shape[0]
    N2 = ar2.shape[0]

    # Shuffling for very small arrays disbabled
    # Enable it for larger arrays
    #resort1 = np.arange(N1)
    #resort2 = np.arange(N2)
    #np.random.shuffle(resort1)
    #np.random.shuffle(resort2)

    #ar1 = ar1[resort1]
    #ar2 = ar2[resort2]

    d_max = 0
    for i in range(N1):
        d_min = np.inf
        for j in range(N2):
            # faster performance with square of distance
            # avoid sqrt until very end
            # Simplificaten (loop unrolling) for (n,2) arrays
            if(ar1.shape[1] == 2):
                d = (ar1[i, 0] - ar2[j, 0])**2+\
                    (ar1[i, 1] - ar2[j, 1])**2
            elif(ar1.shape[1] == 3):
                d = (ar1[i, 0] - ar2[j, 0])**2+\
                    (ar1[i, 1] - ar2[j, 1])**2+\
                    (ar1[i, 2] - ar2[j, 2])**2
            if d < d_min: # always true on first iteration of for-j loop
                d_min = d
                
            if(d_min < d_max):
                break

        # always true on first iteration of for-j loop, after that only
        # if d >= d_max
        if d_min > d_max:
            d_max = d_min

    return np.sqrt(d_max)

@torch.jit.script
def RK4_advection(vf : torch.Tensor, seeds : torch.Tensor, 
        h : float = 0.1, align_corners : bool = True):
    k1 = F.grid_sample(vf, seeds.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                       mode="bilinear", align_corners=align_corners).squeeze().permute(1,0)
    k2_spot = seeds + 0.5 * k1 * h
    k2 = F.grid_sample(vf, k2_spot.unsqueeze(0).unsqueeze(0).unsqueeze(0), 
                       mode="bilinear", align_corners=align_corners).squeeze().permute(1,0)
    k3_spot = seeds + 0.5 * k2 * h
    k3 = F.grid_sample(vf, k3_spot.unsqueeze(0).unsqueeze(0).unsqueeze(0), 
                       mode="bilinear", align_corners=align_corners).squeeze().permute(1,0)
    k4_spot = seeds + k3 * h
    k4 = F.grid_sample(vf, k4_spot.unsqueeze(0).unsqueeze(0).unsqueeze(0), 
                       mode="bilinear", align_corners=align_corners).squeeze().permute(1,0)
    return seeds + (1/6) * (k1+  2*k2 + 2*k3 + k4) * h

@torch.jit.script
def particle_tracing(vf : torch.Tensor, seeds : torch.Tensor, 
                     steps : int = 100, h : float = 0.1,
                     align_corners : bool = True):
    p = seeds.clone()
    positions = torch.empty([steps+1, p.shape[0], p.shape[1]], 
                            device=vf.device)
    positions[0] = p
    
    for i in range(steps):
        p = RK4_advection(vf, p, h, align_corners)
        positions[1+i] = p.clone()
    
    return positions

def visualize_traces(traces):
    '''
    Uses matplotlib to visualize 3D streamline traces
    Expectes traces to be of shape [s, n, 3], where
    s is the number of steps, n is the number of particles,
    and 3 is the [z,y,x] position.
    '''
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    print(traces.shape)
    for i in range(traces.shape[1]):
        ax.plot3D(traces[:,i,0].cpu().numpy(),
                  traces[:,i,1].cpu().numpy(),
                  traces[:,i,2].cpu().numpy())
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.title("Streamline traces")
    plt.show()