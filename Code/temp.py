import torch
import os
import argparse
import json
import time
import subprocess
import shlex
from Other.utility_functions import create_path, nc_to_tensor, tensor_to_cdf, make_coord_grid, npy_to_cdf
import h5py
import numpy as np


project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

def gaussian(x, u, sig, p=1):
    x = x.unsqueeze(1).repeat(1, u.shape[0], 1)
    coeffs = 1 / \
        (torch.prod(sig, dim=-1).unsqueeze(0) * \
            (2*torch.pi)**(x.shape[1]/2))
        
    exps = torch.exp(-1 * \
        torch.sum(
            (((x - u.unsqueeze(0))) / \
            (2**0.5 * sig.unsqueeze(0)))**(2*p), 
        dim=-1))
    
    return torch.sum(coeffs * exps, dim=-1, keepdim=True)

def create_random_sum_of_gaussians(num_gaussians,dims=3):
    torch.manual_seed(123456789)

    means = torch.rand([num_gaussians, dims])*2 - 1
    covs = 0.05+torch.rand([num_gaussians, dims])*0.1

    grid = make_coord_grid([100]*dims, "cpu",
        flatten=True, align_corners=True)

    resulting_sum = torch.zeros([grid.shape[0], 1])
    resulting_sum = gaussian(grid, means, covs).reshape([100]*dims)
    return resulting_sum, means, covs
    
def create_set_of_gaussians(num_gaussians,dims=3):
    means = torch.rand([num_gaussians, dims])*2 - 1
    covs = torch.rand([num_gaussians, dims])

    return means, covs
    
def generate_image(current_density, target_density):
    import matplotlib.pyplot as plt
    x = make_coord_grid([100], "cpu", 
        flatten=True, align_corners=True)
    
    fig = plt.figure()

    plt.plot(x, current_density, color='blue', label='current')
    plt.plot(x, target_density, color='red', label='target')
    plt.legend()

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h)*2, int(w)*2, -1))[:,:,0:3]
    plt.close()
    return im[::2,::2,:]

def training(target_density, dims=1):
    means, covs = create_set_of_gaussians(10,dims=dims)

    means = torch.nn.Parameter(means, requires_grad=True)
    covs = torch.nn.Parameter(covs, requires_grad=True)
    optim = torch.optim.Adam([means, covs], lr = 0.01, betas=[0.9, 0.99])
    #optim = torch.optim.SGD([means, covs], lr = 1, momentum=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 200, 0.1)
    imgs = []

    x = make_coord_grid([100]*dims, "cpu",
            flatten=True, align_corners=True)
    for iteration in range(250):
        optim.zero_grad()

        current_density = gaussian(x, means, covs, p=1)
        current_density /= (current_density.detach().sum()+1e-14)
        
        #error = current_density * torch.log((current_density/target_density)+1e-14)
        #error = torch.abs(current_density-target_density)
        #error = (current_density - target_density)**2
        error = -torch.log((((current_density+1e-24)*(target_density+1e-14))**0.5).sum()+1e-14)
        
        error = error.mean()
        #error = error.mean()**0.5
        error.backward()

        optim.step()
        scheduler.step()
        #with torch.no_grad():
        #    covs.clamp_(0.01, 2)
        print(f"Step {iteration} error: {error.item() : 0.08f}")

        flat_top = gaussian(x, means, covs, p=10).detach().numpy()
        flat_top /= flat_top.sum()
        imgs.append(
            generate_image(
                flat_top,
                #current_density.detach().numpy(), 
                target_density.detach().numpy()
                )
            )

    result = gaussian(x, means, covs)
    result /= result.sum()

    import imageio

    imageio.mimwrite("save.gif", imgs)

def alg(target_density, n_gaussians=1, dims=3):
    target_density /= target_density.sum()
    #target_density *= (4**dims)/(2**dims)
    #target_density *= n_gaussians
    #x = make_coord_grid([100]*dims, "cpu",
    #        flatten=True, align_corners=True)
    lefts = torch.zeros([n_gaussians, dims])-1
    rights = torch.zeros([n_gaussians, dims])+1
    
    current_density = target_density.clone()
    
    tensor_to_cdf(current_density.unsqueeze(0).unsqueeze(0), "targetdensity.nc")

    for i in range(1):
        current_cumsum = current_density.clone()
        for j in range(dims):
            current_cumsum = torch.cumsum(current_cumsum, dim=j)
        tensor_to_cdf(current_cumsum.unsqueeze(0).unsqueeze(0), "cumsum.nc")

def vti_to_nc():
    import numpy as np
    import vtk

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName("asteroid.vti")
    reader.Update()
    image = reader.GetOutput()

    print(image)
    
    dims = image.GetDimensions()
    print(f"Dimensions: {dims}")
    
    point_data = image.GetPointData()
    
    print(f"Point data:")
    print(point_data)
    
    '''
    data = point_data.GetArray('v02')
    print(f"V02")
    print(data)
    
    data_npy = np.array(data)
    print(data_npy.shape)
    data_npy = data_npy.reshape(dims)
    print(data_npy.shape)
    data_npy = np.expand_dims(data_npy, 0)
    data_npy = np.expand_dims(data_npy, 0)
    
    data_npy -= data_npy.min()
    data_npy /= data_npy.max()
    data[-1,:,:] = 0
    data[:,:,0] = 0
    npy_to_cdf(data_npy, "asteroid_v02.nc")
    
    data = point_data.GetArray('v03')
    print(f"V03")
    print(data)
    
    data_npy = np.array(data)
    print(data_npy.shape)
    data_npy = data_npy.reshape(dims)
    print(data_npy.shape)
    data_npy = np.expand_dims(data_npy, 0)
    data_npy = np.expand_dims(data_npy, 0)
    
    data_npy -= data_npy.min()
    data_npy /= data_npy.max()
    data[-1,:,:] = 0
    data[:,:,0] = 0
    npy_to_cdf(data_npy, "asteroid_v03.nc")
    '''
    
    data = point_data.GetArray('vtkValidPointMask')
    print(f"vtkValidPointMask")
    print(data)
    
    data_npy = np.array(data)
    print(data_npy.shape)
    data_npy = data_npy.reshape(dims)
    print(data_npy.shape)
    data_npy = np.expand_dims(data_npy, 0)
    data_npy = np.expand_dims(data_npy, 0)
    
    npy_to_cdf(data_npy, "asteroid_validpointmask.nc")
    
    data = point_data.GetArray('prs')
    print(f"prs")
    print(data)
    
    data_npy = np.array(data)
    print(data_npy.shape)
    data_npy = data_npy.reshape(dims)
    print(data_npy.shape)
    data_npy = np.expand_dims(data_npy, 0)
    data_npy = np.expand_dims(data_npy, 0)
    
    data_npy -= data_npy.min()
    data_npy /= data_npy.max()
    data[-1,:,:] = 0
    data[:,:,0] = 0
    npy_to_cdf(data_npy, "asteroid_prs.nc")
    
    data = point_data.GetArray('rho')
    print(f"rho")
    print(data)
    
    data_npy = np.array(data)
    print(data_npy.shape)
    data_npy = data_npy.reshape(dims)
    print(data_npy.shape)
    data_npy = np.expand_dims(data_npy, 0)
    data_npy = np.expand_dims(data_npy, 0)
    
    data_npy -= data_npy.min()
    data_npy /= data_npy.max()
    data[-1,:,:] = 0
    data[:,:,0] = 0
    npy_to_cdf(data_npy, "asteroid_rho.nc")
    
    data = point_data.GetArray('snd')
    print(f"snd")
    print(data)
    
    data_npy = np.array(data)
    print(data_npy.shape)
    data_npy = data_npy.reshape(dims)
    print(data_npy.shape)
    data_npy = np.expand_dims(data_npy, 0)
    data_npy = np.expand_dims(data_npy, 0)
    
    data_npy -= data_npy.min()
    data_npy /= data_npy.max()
    data[-1,:,:] = 0
    data[:,:,0] = 0
    npy_to_cdf(data_npy, "asteroid_snd.nc")
    
    data = point_data.GetArray('tev')
    print(f"tev")
    print(data)
    
    data_npy = np.array(data)
    print(data_npy.shape)
    data_npy = data_npy.reshape(dims)
    print(data_npy.shape)
    data_npy = np.expand_dims(data_npy, 0)
    data_npy = np.expand_dims(data_npy, 0)
    
    data_npy -= data_npy.min()
    data_npy /= data_npy.max()
    data[-1,:,:] = 0
    data[:,:,0] = 0
    npy_to_cdf(data_npy, "asteroid_tev.nc")
    
    quit()

    u = VN.vtk_to_numpy(data.GetCellData().GetArray('velocity'))
    b = VN.vtk_to_numpy(data.GetCellData().GetArray('cell_centered_B'))

    u = u.reshape(vec,order='F')
    b = b.reshape(vec,order='F')

    x = zeros(data.GetNumberOfPoints())
    y = zeros(data.GetNumberOfPoints())
    z = zeros(data.GetNumberOfPoints())

    for i in range(data.GetNumberOfPoints()):
            x[i],y[i],z[i] = data.GetPoint(i)

    x = x.reshape(dim,order='F')
    y = y.reshape(dim,order='F')
    z = z.reshape(dim,order='F')

def np_to_nc(data, name):
    import netCDF4 as nc
    d = nc.Dataset(os.path.join(data_folder, name), 'w')
    d.createDimension('x')
    d.createDimension('y')
    d.createDimension('z')
    dims = ['x', 'y', 'z']
    d.createVariable("data", np.float32, dims)
    d["data"][:] = data
    d.close()

def nc_to_raw(name):
    data, _ = nc_to_tensor(os.path.join(data_folder, name))
    data : np.ndarray = data[0,0].cpu().float().numpy().flatten()
    with open(os.path.join(data_folder, f"{name.split('.')[0]}.raw"), 'wb') as f:
        data.tofile(f)  
    

class Rect():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        if w > 1 and h > 1:
            self.queue = [
                #(x,y),
                (x+w//2,y+h//2),
                (x+w//2, y),
                (x,y+h//2)
            ]
        elif w > 1:
            self.queue = [
                #(x,y),
                (x+w//2,y)
            ]
        elif h > 1:
            self.queue = [
                #(x,y),
                (x,y+h//2)
            ]
        else:
            self.queue = [
                #(x,y)
            ]

    def subdivide(self):
        if(self.w > 1 and self.h > 1):
            return [
                Rect(self.x, self.y, self.w//2, self.h//2),
                Rect(self.x+self.w//2, self.y+self.h//2, self.w-self.w//2, self.h-self.h//2),
                Rect(self.x+self.w//2, self.y, self.w-self.w//2, self.h//2),
                Rect(self.x, self.y+self.h//2, self.w//2, self.h-self.h//2)
            ]
        elif(self.w > 1):
            return [
                Rect(self.x, self.y, self.w//2, self.h),
                Rect(self.x+self.w//2, self.y, self.w-self.w//2, self.h),
            ]
        elif(self.h > 1):
            return [
                Rect(self.x, self.y, self.w, self.h//2),
                Rect(self.x, self.y+self.h//2, self.w, self.h-self.h//2),
            ]
        return []
       
    def get_next(self):
        if(len(self.queue) > 0):
            return self.queue.pop(0)
        return None
    
    def needs_subdivide(self):
        return len(self.queue) == 0

def checkerboard_render(w,h):
    rects = [Rect(0,0,w,h)]

    offset_order = [(0,0)]
    rects_to_add = []
    # continue until all rects are done
    while len(rects) > 0:
        # loop through current rects 1 at a time
        indices_to_remove = []
        # Get the next spot for each rect
        for i in range(len(rects)):
            spot = rects[i].get_next()
            # make sure it is valid
            if spot is not None:
                offset_order.append(spot)
            # see if the rect need subdivision
            if rects[i].needs_subdivide():
                indices_to_remove.append(i)
                rects_to_add += rects[i].subdivide()
        # remove finished rects
        for i in range(len(indices_to_remove)):
            rects.pop(indices_to_remove[len(indices_to_remove)-i-1])
        
        # put new rects into queue
        if(len(rects) == 0):
            while(len(rects_to_add) > 0):
                rects.append(rects_to_add.pop(0))

    return offset_order

def checkerboard_code():
    import imageio.v3 as imageio
    a = checkerboard_render(4,4)
    print(len(a))
    d = np.zeros((16,16,1), np.uint8)
    seq = []
    for _ in range(15):
        seq.append(d.copy())
    for i in range(len(a)):
        b = a[i]
        d[b[0]*4:b[0]*4+4,b[1]*4:b[1]*4+4] = 255
        for _ in range(15):
            seq.append(d.copy())
    imageio.imwrite("checkerboard.mp4", seq, fps=15)
    
def psnr_test(x, y):
    from Other.utility_functions import PSNR
    p = PSNR(torch.tensor(x).permute(2, 0, 1).unsqueeze(0), torch.tensor(y).permute(2, 0, 1).unsqueeze(0))
    print(f"PSNR: {p:0.04f} dB")
    
def ssim_test(x, y):
    from Other.utility_functions import ssim
    s = ssim(torch.tensor(x).permute(2, 0, 1).unsqueeze(0), torch.tensor(y).permute(2, 0, 1).unsqueeze(0))
    print(f"SSIM: {s:0.04f}")
  
def table2_test():
    import imageio.v3 as imageio
    gt_img = imageio.imread(os.path.join(output_folder, "Table2_GT.png")) / 255.0
    amgsrn_img = imageio.imread(os.path.join(output_folder, "Table2_AMGSRN.png")) / 255.0
    amgsrn_ensemble_img = imageio.imread(os.path.join(output_folder, "Table2_AMGSRN_ensemble.png")) / 255.0
    fvsrn_img = imageio.imread(os.path.join(output_folder, "Table2_fVSRN.png")) / 255.0
    ngp_img = imageio.imread(os.path.join(output_folder, "Table2_NGP.png")) / 255.0
    
    print(f"===================")
    print(f"AMGSRN")
    psnr_test(gt_img, amgsrn_img)
    ssim_test(gt_img, amgsrn_img)
    print(f"===================")
    print()
    print()
    print(f"AMGSRN ensemble")
    psnr_test(gt_img, amgsrn_ensemble_img)
    ssim_test(gt_img, amgsrn_ensemble_img)
    print(f"===================")
    print()
    print()
    print(f"fVSRN")
    psnr_test(gt_img, fvsrn_img)
    ssim_test(gt_img, fvsrn_img)
    print(f"===================")
    print()
    print()
    print(f"NGP")
    psnr_test(gt_img, ngp_img)
    ssim_test(gt_img, ngp_img)
    print(f"===================")
    print()
    print()
    
if __name__ == '__main__':
    
    
    quit()