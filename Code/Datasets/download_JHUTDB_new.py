import numpy as np
import pyJHTDB
from pyJHTDB import libJHTDB


token="edu.osu.buckeyemail.wurster.18-92fb557b" #replace with your own token
params = {
    "token":token,
    "dataset":"rotstrat4096",
    "fields":"u",
    "tstart":1,"tend":1,"tstep":1,
    "xstart":1,"ystart":1,"zstart":1,
    "xend":128,"yend":128,"zend":128,
    "xstep":1,"ystep":1,"zstep":1,
    "Filter_Width":1,
    "output_filename":"na"
}

auth_token=params["token"]
tstart=int(params.get("tstart"))
tend=int(params.get("tend"))
tstep=int(params.get("tstep"))
xstart=int(params.get("xstart"))
ystart=int(params.get("ystart"))
zstart=int(params.get("zstart"))
xend=int(params.get("xend"))
yend=int(params.get("yend"))
zend=int(params.get("zend"))
xstep=int(params.get("xstep",1))
ystep=int(params.get("ystep",1))
zstep=int(params.get("zstep",1))
Filter_Width=int(params.get("Filter_Width",1))
time_step=int(params.get("time_step",0))
fields=params.get("fields","u")
data_set=params.get("dataset","isotropic1024coarse")
output_filename=params.get("output_filename",data_set)

lJHTDB = libJHTDB()
lJHTDB.initialize()
lJHTDB.add_token(auth_token)

## "filename" parameter is the file names of output files, if filename='N/A', no files will be written. 
##             For example, if filename='results', the function will write "results.h5" and "results.xmf".
## The function only returns the data at the last time step within [t_start:t_step:t_end]
## The function only returns the data in the last field. For example, result=p if field=[up].
import time
t0 = time.time()
result = lJHTDB.getbigCutout(
        data_set=data_set, fields=fields, t_start=tstart, t_end=tend, t_step=tstep,
        start=np.array([xstart, ystart, zstart], dtype = int),
        end=np.array([xend, yend, zend], dtype = int),
        step=np.array([xstep, ystep, zstep], dtype = int),
        filter_width=Filter_Width,
        filename=output_filename)
t1 = time.time()
print(f"Time passed: {t1-t0: 0.02f}")
lJHTDB.finalize()
print(result.shape)