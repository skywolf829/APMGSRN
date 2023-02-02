from __future__ import absolute_import, division, print_function
import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import os
from torch.utils.tensorboard import SummaryWriter

class benchmark_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.transformation_matrices = torch.nn.Parameter(
            torch.zeros(
                [16, 4, 4],
                device = "cuda:0",
                dtype=torch.float32
            ),
            requires_grad=True
        )
    
        self.transformation_matrices[:] = torch.eye(4, device="cuda:0", dtype=torch.float32)
        self.transformation_matrices[:,0:3,:] += torch.rand_like(
            self.transformation_matrices[:,0:3,:],
            device="cuda:0", dtype=torch.float32) * 0.1
        self.transformation_matrices = torch.nn.Parameter(
            self.transformation_matrices @ \
            self.transformation_matrices.transpose(-1, -2),
            requires_grad=True)
        self.transformation_matrices[:,3,0:3] = 0  

    def transform(self, x):
        torch.cuda.synchronize()
        transformation_matrices = self.transformation_matrices
        
        torch.cuda.synchronize()            
        transformed_points = torch.cat(
            [x, torch.ones([x.shape[0], 1], 
            device=x.device,
            dtype=torch.float32)], 
            dim=1).unsqueeze(0).repeat(
                transformation_matrices.shape[0], 1, 1
            )
        
        torch.cuda.synchronize()        
        transformed_points = torch.bmm(transformation_matrices, 
                            transformed_points.transpose(-1, -2)).transpose(-1, -2)
        
        
        torch.cuda.synchronize()        
        transformed_points = transformed_points[...,0:3]
            
        torch.cuda.synchronize()        
        return transformed_points 
    
    def forward(self, x):
        torch.cuda.synchronize()
        transformed_points = self.transform(x)
        
        torch.cuda.synchronize()
        coeffs = torch.linalg.det(self.transformation_matrices[:,0:3,0:3]).unsqueeze(0) / ((2.0*torch.pi)**(3/2))
        
        torch.cuda.synchronize()
        exps = torch.exp(-0.5 * \
            torch.sum(
                transformed_points.transpose(0,1)**20, 
            dim=-1))
        
        torch.cuda.synchronize()
        result = torch.sum(coeffs * exps, dim=-1, keepdim=True)
        
        torch.cuda.synchronize()
        return result
    
if __name__ == '__main__':

    start_time = time.time()
    writer = SummaryWriter(os.path.join('tensorboard','profiletest'))
    
    target_density = torch.randn([10000, 1], device="cuda:0", dtype=torch.float32)
    target_density /= target_density.sum()
    
    model = benchmark_model()
    
    optimizer = optim.Adam([{"params": model.transformation_matrices}], lr=0.0001, 
            betas=[0.99, 0.99]) 
    
    x = torch.rand([10000, 3], device="cuda:0", dtype=torch.float32)*2 - 1 
    '''
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=8,
            active=1,
            repeat=1),
        profile_memory=True,
        with_stack=True,
        with_modules=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            os.path.join('tensorboard',"profiletest"))) as profiler:
    '''
    for iteration in range(10000):
        torch.cuda.synchronize()
        optimizer.zero_grad()
        
        torch.cuda.synchronize() 
        
        torch.cuda.synchronize()          
        density = model(x)
        
        torch.cuda.synchronize()
        density /= density.sum().detach()  
        
        torch.cuda.synchronize()
        density_loss = F.kl_div(
            torch.log(density+1e-16), 
                torch.log(target_density.detach()+1e-16), 
                reduction='none', 
                log_target=True)
        
        torch.cuda.synchronize()
        density_loss.mean().backward()
        
        torch.cuda.synchronize()
        optimizer.step()   
        
        torch.cuda.synchronize()
        #profiler.step()    

    end_time = time.time()
    print(f"Took {(end_time-start_time)/60 : 0.02f} min")    

    writer.close()