import os
import numpy as np
import zeep
import struct
import time
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

client = zeep.Client('http://turbulence.pha.jhu.edu/service/turbulence.asmx?WSDL')
ArrayOfFloat = client.get_type('ns0:ArrayOfFloat')
ArrayOfArrayOfFloat = client.get_type('ns0:ArrayOfArrayOfFloat')
SpatialInterpolation = client.get_type('ns0:SpatialInterpolation')
TemporalInterpolation = client.get_type('ns0:TemporalInterpolation')
token="edu.osu.buckeyemail.wurster.18-92fb557b" #replace with your own token

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..", "..")
data_folder = os.path.join(project_folder_path, "Data")
save_folder = os.path.join(data_folder)


    
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

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 4, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

'''
def get_frame_using_library(x_start, x_end, x_step, 
y_start, y_end, y_step, 
z_start, z_end, z_step, 
sim_name, timestep, field, num_components, job_id=None):
    lJHTDB = libJHTDB()
    lJHTDB.initialize()
    lJHTDB.add_token(token)
    result = lJHTDB.getbigCutout(
        data_set=sim_name, fields=field, 
        t_start=timestep, t_end=timestep+1, t_step=2,
        start=np.array([x_start, y_start, z_start], dtype = int),
        end=np.array([x_end, y_end, z_end], dtype = int),
        step=np.array([x_step, y_step, z_step], dtype = int),
        filter_width=1,
        filename="na")
    
    lJHTDB.finalize()
    return result, job_id
'''

def get_frame(x_start, x_end, x_step, 
y_start, y_end, y_step, 
z_start, z_end, z_step, 
sim_name, timestep, field, num_components, job_id=None):
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
    result=np.array(result).reshape((nz, ny, nx, num_components)).astype(np.float32)
    return result, job_id

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
        x_len = 256
        y_len = 256
        z_len = 256
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
           print("Done: %i/%i" % (done, len(threads)))
    return full

def download_with_buffer(save_location, 
    x_start, x_end, x_step,
    y_start, y_end, y_step, 
    z_start, z_end, z_step,
    sim_name, timestep, field, num_components, num_workers):
    
    start_at = 0
    with open(save_location,"wb" if start_at == 1 else "ab") as f:
        print(f"Opened file {save_location}")
        
        x_len = 256
        y_len = 256
        z_len = 128
        request_no = 1
        threads = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for k in range(x_start, x_end, x_len):
                for i in range(y_start, y_end, y_len):
                    for j in range(z_start, z_end, z_len):
                        x_stop = min(k+x_len, x_end)
                        y_stop = min(i+y_len, y_end)
                        z_stop = min(j+z_len, z_end)
                        if(request_no>=start_at):
                            threads.append(executor.submit(get_frame,
                                            k, x_stop, x_step,
                                            i, y_stop, y_step,
                                            j, z_stop, z_step,
                                            sim_name, timestep, field, 
                                            num_components, request_no))
                        request_no += 1
            print(f"Submitted {request_no-1} jobs. Beggining fetch in parallel on {num_workers} workers.")
            
            total_requests = request_no-1
            waiting_tasks = {}
            current_request_no = start_at  
            
            num_downloaded = start_at-1
            num_written = start_at-1          
            
            printProgressBar(num_written, total_requests, prefix=f"{num_downloaded-num_written} waiting to write. {num_written}/{num_downloaded}/{total_requests} written/downloaded")
            for future in as_completed(threads):
                result, job_id = future.result()
                threads.remove(future)
                #result=np.linalg.norm(result, axis=3)
                result=result.flatten().tobytes()
                num_downloaded += 1
                #print(f"Job {job_id}/{request_no-1} completed and processed. Adding to waitlist")
                printProgressBar(num_written, total_requests, prefix=f"{num_downloaded-num_written} waiting to write. {num_written}/{num_downloaded}/{total_requests} written/downloaded")
                waiting_tasks[job_id] = result
                
                while(current_request_no in waiting_tasks.keys()):
                    f.write(waiting_tasks[current_request_no])
                    f.flush()    
                    os.fsync(f)                
                    #print(f"Request {current_request_no} written to disk")
                    waiting_tasks.pop(current_request_no)
                    num_written += 1
                    current_request_no += 1
                        
                gc.collect()
    printProgressBar(num_written, total_requests, prefix=f"{num_downloaded-num_written} waiting to write. {num_written}/{num_downloaded}/{total_requests} written/downloaded")
            
if __name__ == '__main__':
    name = "channel5200"
    #name = "isotropic1024coarse"
    #name = "rotstrat4096"
    save_location = "channel5200.raw"
    t0 = time.time()
    count = 0
    startts = 10
    endts = 11
    ts_skip = 10
    frames = []
    for i in range(startts, endts, ts_skip):
        print("TS %i/%i" % (i, endts))
        
        download_with_buffer(save_location,
            0, 10240, 1, #x
            0, 1536, 1, #y
            0, 7680, 1, #z
            name, i, 
            "p", 1, 64)
        
    print(f"Finished in {time.time() - t0} sec.")

    
    