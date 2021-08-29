#!/usr/bin/env python

# Load modules
# ------------
import numpy as np
import scipy.io.netcdf as nc
import scipy.io as sio
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import glob
plt.ion()

# Set parameters
# --------------
# Inputs
main_dir = '/Volumes/Long/q-gcm/gyres_ocean_SST/'  
subdir_ref = 'REF5/yrs*'
res_mod = 120
yrs_mod = 'yrs000-200'
subdir_mod = [r'LR%d/%s/'%(res_mod,yrs_mod), \
              r'POD%d/%s/'%(res_mod,yrs_mod), \
              r'POD%d-P/%s/'%(res_mod,yrs_mod)]
t1 = 180. # last time (years)
# Outputs
outdir = '../manuscript/ocemod-v1/figures/sst/'
nam_ref = 'REF (5 km)'
nam_mod = [r'DLR (%d km)'%(res_mod), \
           r'MLU (%d km)'%(res_mod), \
           r'MLU-P (%d km)'%(res_mod)]

# List subfolders of segment run
list_yrs = glob.glob(main_dir + subdir_ref)[::-1]

# Read monit.nc of ref. model 
# ---------------------------
f = nc.netcdf_file(list_yrs[0] + '/monit.nc', 'r')
z = f.variables['zo'][:].copy() # layer axis
nz = len(z)
n = 1 # counter of files
tyrs = np.empty(0) # time axis (years)
jval = np.empty((0,nz)) # max. zonal jet velocity (m/s)
for s in list_yrs:
    print 'Read file: ', s
    f = nc.netcdf_file(s + '/monit.nc', 'r')
    nn = np.minimum(n,2) - 1
    tmp = f.variables['time'][nn:].copy()
    tyrs = np.append(tyrs, tmp, axis=0) # append rows
    tmp = f.variables['ocjval'][nn:,:].copy()
    jval = np.append(jval, tmp, axis=0)
    n += 1
f.close()

# Non-uniform weight of vertical integration for 'p'
pwt = []
tmp = 0.
for k in range(nz):
    hoc = 2.0*(z[k] - tmp)
    tmp += hoc
    pwt.append(hoc)
pwt = pwt/sum(pwt)

# Low-pass filtering of 'jval'
# ----------------------------
fs = 1/(tyrs[1] - tyrs[0]) # original freq. (1/yrs)
fc = 0.5  # cut-off freq. (1/yrs) of filter
w = fc / (fs/2) # normalize freq.
b, a = signal.butter(5, w, 'low') # 5th order Butterworth low-pass filter
#jval_lowf = signal.filtfilt(b, a, jval, axis=0) # filter by layers
# first int. then filter 
jvalm = np.matmul(jval, pwt) # vertical integration
jvalm_ref = signal.filtfilt(b, a, jvalm)

# Filtering of coarse model 'jval'
# --------------------------------
f = nc.netcdf_file(main_dir + subdir_mod[0] + 'monit.nc', 'r')
tmp = f.variables['time'][:].copy()
f.close()
id1 = np.flatnonzero(tmp>=t1)[0]
tyrs1 = tmp[:id1] 
fs1 = 1/(tyrs1[1] - tyrs1[0]) 
w1 = fc / (fs1/2)
b1, a1 = signal.butter(5, w1, 'low') 
jvalm_mod = np.zeros((len(tyrs1),len(subdir_mod)))
j = 0
for s in subdir_mod:
    f = nc.netcdf_file(main_dir + s + 'monit.nc', 'r')
    tmp = f.variables['ocjval'][:id1,:].copy()
    tmp = np.matmul(tmp, pwt) 
    jvalm_mod[:,j] = signal.filtfilt(b1, a1, tmp)
    j += 1
f.close()


# Plot time series of 'jval'
# --------------------------
fig = plt.figure(figsize=(5,3.5))
#plt.style.use("fivethirtyeight") # "fivethirtyeight", "ggplot", "seaborn"
plt.plot(tyrs, jvalm_ref, 'k', label=nam_ref)
for j in range(len(subdir_mod)):
    plt.plot(tyrs1, jvalm_mod[:,j], label=nam_mod[j])
plt.yscale('log')
plt.ylim([1.0e-3, 1.25*jvalm_ref.max()])
plt.grid(which='both', axis='both')
plt.xlabel(r'Times (years)')
plt.ylabel(r'Magnitude (m/s)')
plt.legend(fontsize=9)
#plt.title(r'REF (5 km)')
#plt.tick_params(axis='both', which='major', labelsize=10)
fig.tight_layout()
#plt.show()
plt.savefig(outdir + r'jval%d.eps'%(res_mod), dpi=200, \
            bbox_inches='tight', pad_inches=0)

