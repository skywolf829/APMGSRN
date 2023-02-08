import torch
from torch import Tensor
from nerfacc import OccupancyGrid, ray_marching, unpack_info, rendering
import argparse
import os
from Models.models import load_model
from Models.options import load_options


def rgb_sigma_fn(t_starts, t_ends, ray_indices):
    # This is a dummy function that returns random values.
    rgbs = torch.rand((t_starts.shape[0], 3), device="cuda:0")
    sigmas = torch.rand((t_starts.shape[0], 1), device="cuda:0")
    return rgbs, sigmas

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model on some tests')

    parser.add_argument('--load_from',default=None,type=str,help="Model name to load")
    parser.add_argument('--device',default=None,type=str,
                        help="Device to load model to")
    args = vars(parser.parse_args())

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..")
    data_folder = os.path.join(project_folder_path, "Data")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")

    # Load the model
    opt = load_options(os.path.join(save_folder, args['load_from']))
    opt['device'] = args['device']
    model = load_model(opt, args['device']).to(opt['device'])
    print(f"Moved model to {opt['device']}.")
    model.eval()
    if("cuda" in args['device']):        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    

    device = args['device']
    batch_size = 64*64
    rays_o = torch.rand((batch_size, 3), device=device)
    rays_d = torch.randn((batch_size, 3), device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    # Ray marching with near far plane.
    ray_indices, t_starts, t_ends = ray_marching(
        rays_o, rays_d, near_plane=0.1, far_plane=1.0, render_step_size=1e-3
    )

    # Ray marching with aabb.
    scene_aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
    ray_indices, t_starts, t_ends = ray_marching(
        rays_o, rays_d, scene_aabb=scene_aabb, render_step_size=1e-3
    )

    # Ray marching with per-ray t_min and t_max.
    t_min = torch.zeros((batch_size,), device=device)-1
    t_max = torch.ones((batch_size,), device=device)
    ray_indices, t_starts, t_ends = ray_marching(
        rays_o, rays_d, t_min=t_min, t_max=t_max, render_step_size=1e-3
    )

    # Convert t_starts and t_ends to sample locations.
    t_mid = (t_starts + t_ends) / 2.0
    sample_locs = rays_o[ray_indices] + t_mid * rays_d[ray_indices]

    colors, opacities, depths = rendering(
        t_starts, t_ends, ray_indices, n_rays=128, rgb_sigma_fn=rgb_sigma_fn)
    print(colors.shape, opacities.shape, depths.shape)