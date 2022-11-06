import torch
import torch.nn.functional as F

def l1(x, y):
    return F.l1_loss(x, y)

def mse(x, y):
    return F.mse_loss(x, y)

def l1_loss(network_output, target):
    return l1(network_output, target['data'])

def l1_occupancy(gt, y):
    # Expects x to be [..., 3] or [..., 4] for (u, v, o) or (u, v, w, o)
    # Where o is occupancy
    is_nan_mask = torch.isnan(gt)[...,0].detach()
    
    o_loss = l1((~is_nan_mask).to(torch.float32).detach(), y[..., -1])
    vf_loss = l1(gt[~is_nan_mask, :].detach(), y[~is_nan_mask, 0:-1])
    return o_loss + vf_loss

def angle_same_loss(x, y):
    angles = (1 - F.cosine_similarity(x, y))
    mask = (y.norm(dim=1) != 0).type(torch.float32).detach()
    weighted_angles = angles * mask
    return weighted_angles.mean()

def angle_parallel_loss(x, y):
    angles = (1 - F.cosine_similarity(x, y)**2)
    mask = (y.norm(dim=1) != 0).type(torch.float32).detach()
    weighted_angles = angles * mask
    return weighted_angles.mean()

def angle_orthogonal_loss(x, y):
    angles = torch.abs(F.cosine_similarity(x, y))
    mask = (y.norm(dim=1) != 0).type(torch.float32).detach()
    weighted_angles = angles * mask
    return weighted_angles.mean()

def magangle_orthogonal_loss(x, y):
    mags = F.mse_loss(torch.norm(x,dim=1), torch.norm(y,dim=1))
    angles = (F.cosine_similarity(x, y)**2)
    mask = (y.norm(dim=1) != 0).type(torch.float32).detach()
    weighted_angles = angles * mask
    return 0.9*mags + 0.1*weighted_angles.mean()

def magangle_parallel_loss(x, y):
    x_norm = torch.norm(x,dim=1)
    y_norm = torch.norm(y,dim=1)
    mags = F.mse_loss(x_norm, y_norm)
    angles = (1 - F.cosine_similarity(x, y)**2)
    mask = (y.norm(dim=1) != 0).type(torch.float32).detach()
    weighted_angles = angles * mask
    return 0.9*mags + 0.1*weighted_angles.mean()

def magangle_same_loss(x, y):
    x_norm = torch.norm(x,dim=1)
    y_norm = torch.norm(y,dim=1)
    mags = F.mse_loss(x_norm, y_norm)
    angles = (1 - F.cosine_similarity(x, y))
    mask = (y.norm(dim=1) != 0).type(torch.float32).detach()
    weighted_angles = angles * mask
    return 0.9*mags + 0.1*weighted_angles.mean()

def uvwf_any_loss(network_output, target):
    l1_err = l1(network_output[:,0:3], target['data'])
    grads_f = torch.autograd.grad(network_output[:,3], target['inputs'], 
        grad_outputs=torch.ones_like(network_output[:,3]),
        create_graph=True)[0]
    f_err = angle_orthogonal_loss(grads_f, target['data'])
    return l1_err + f_err

def uvwf_parallel_loss(network_output, target):
    l1_err = l1(network_output[:,0:3], target['data'])
    grads_f = torch.autograd.grad(network_output[:,3], target['inputs'], 
        grad_outputs=torch.ones_like(network_output[:,3]),
        create_graph=True)[0]
    f_err = angle_parallel_loss(grads_f, target['normal'])
    return l1_err + f_err

def uvwf_direction_loss(network_output, target):
    l1_err = l1(network_output[:,0:3], target['data'])
    grads_f = torch.autograd.grad(network_output[:,3], target['inputs'], 
        grad_outputs=torch.ones_like(network_output[:,3]),
        create_graph=True)[0]
    f_err = angle_same_loss(grads_f, target['normal'])
    return l1_err + f_err

def dsf_any_loss(network_output, target):
    grads_f = torch.autograd.grad(network_output[:,0], target['inputs'], 
        grad_outputs=torch.ones_like(network_output[:,0]),
        create_graph=True)[0]
    grads_g = torch.autograd.grad(network_output[:,1], target['inputs'], 
        grad_outputs=torch.ones_like(network_output[:,1]),
        create_graph=True)[0]
    dsf = torch.cross(grads_f, grads_g, dim=1)
    angle_err = angle_same_loss(dsf, target['data'])
    return angle_err

def dsf_parallel_loss(network_output, target):
    grads_f = torch.autograd.grad(network_output[:,0], target['inputs'], 
        grad_outputs=torch.ones_like(network_output[:,0]),
        create_graph=True)[0]
    grads_g = torch.autograd.grad(network_output[:,1], target['inputs'], 
        grad_outputs=torch.ones_like(network_output[:,1]),
        create_graph=True)[0]
    normal_err = angle_parallel_loss(grads_f, target['normal'])
    dsf = torch.cross(grads_f.detach(), grads_g, dim=1)
    angle_err = angle_same_loss(dsf, target['data'])
    return normal_err + angle_err

def dsf_direction_loss(network_output, target):
    grads_f = torch.autograd.grad(network_output[:,0], target['inputs'], 
        grad_outputs=torch.ones_like(network_output[:,0]),
        create_graph=True)[0]
    grads_g = torch.autograd.grad(network_output[:,1], target['inputs'], 
        grad_outputs=torch.ones_like(network_output[:,1]),
        create_graph=True)[0]
    normal_err = angle_same_loss(grads_f, target['normal'])
    dsf = torch.cross(grads_f.detach(), grads_g, dim=1)
    angle_err = angle_same_loss(dsf, target['data'])
    return normal_err + angle_err

def dsfm_any_loss(network_output, target):
    grads_f = torch.autograd.grad(network_output[:,0], target['inputs'], 
        grad_outputs=torch.ones_like(network_output[:,0]),
        create_graph=True)[0]
    grads_g = torch.autograd.grad(network_output[:,1], target['inputs'], 
        grad_outputs=torch.ones_like(network_output[:,1]),
        create_graph=True)[0]
    dsf = torch.cross(grads_f, grads_g, dim=1)
    angle_err = angle_same_loss(dsf, target['data'])
    l1_err = l1(network_output[:,-1], torch.norm(target['data'], dim=-1))
    return angle_err + l1_err

def dsfm_parallel_loss(network_output, target):
    grads_f = torch.autograd.grad(network_output[:,0], target['inputs'], 
        grad_outputs=torch.ones_like(network_output[:,0]),
        create_graph=True)[0]
    grads_g = torch.autograd.grad(network_output[:,1], target['inputs'], 
        grad_outputs=torch.ones_like(network_output[:,1]),
        create_graph=True)[0]
    normal_err = angle_parallel_loss(grads_f, target['normal'])
    dsf = torch.cross(grads_f.detach(), grads_g, dim=1)
    angle_err = angle_same_loss(dsf, target['data'])    
    l1_err = l1(network_output[:,-1], torch.norm(target['data'], dim=-1))
    return normal_err + angle_err + l1_err

def dsfm_direction_loss(network_output, target):
    grads_f = torch.autograd.grad(network_output[:,0], target['inputs'], 
        grad_outputs=torch.ones_like(network_output[:,0]),
        create_graph=True)[0]
    grads_g = torch.autograd.grad(network_output[:,1], target['inputs'], 
        grad_outputs=torch.ones_like(network_output[:,1]),
        create_graph=True)[0]
    normal_err = angle_same_loss(grads_f, target['normal'])
    dsf = torch.cross(grads_f.detach(), grads_g, dim=1)
    angle_err = angle_same_loss(dsf, target['data'])
    l1_err = l1(network_output[:,-1], torch.norm(target['data'], dim=-1))
    return normal_err + angle_err + l1_err

def f_any_loss(network_output, target):
    grads_f = torch.autograd.grad(network_output[:,0], target['inputs'], 
        grad_outputs=torch.ones_like(network_output[:,0]),
        create_graph=True)[0]
    normal_err = angle_orthogonal_loss(grads_f, target['data'])
    return normal_err

def f_parallel_loss(network_output, target):
    grads_f = torch.autograd.grad(network_output[:,0], target['inputs'], 
        grad_outputs=torch.ones_like(network_output[:,0]),
        create_graph=True)[0]
    normal_err = angle_parallel_loss(grads_f, target['normal'])
    return normal_err

def f_direction_loss(network_output, target):
    grads_f = torch.autograd.grad(network_output[:,0], target['inputs'], 
        grad_outputs=torch.ones_like(network_output[:,0]),
        create_graph=True)[0]
    normal_err = angle_same_loss(grads_f, target['normal'])
    return normal_err

def hhd_loss(network_output, target):
    print("Not yet implemented")
    return 0

def seeding_loss(network_seeds_output):
    return torch.abs(network_seeds_output).mean()

def get_loss_func(opt):    
    return l1_loss