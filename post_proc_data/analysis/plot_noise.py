#!/usr/bin/env python

# Load modules
# ------------
import numpy as np
import scipy.io.netcdf as nc
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
plt.ion()

# Set parameters
# --------------
# Inputs
file1 = '/Volumes/Long/q-gcm/gyres_ocean_SST/REF5/POD/ocludat80_v1.nc'
file2 = '/Volumes/Long/q-gcm/gyres_ocean_SST/POD80/yrs000-200/oclu.nc'
k = 1 # layer
t1 = 120.0 # time (year)
# Outputs
output1 = r'../manuscript/ocemod-v1/figures/sst/u%s-MLU80.eps'%(k)
output2 = r'../manuscript/ocemod-v1/figures/sst/v%s-MLU80.eps'%(k)
fig_tit1 = 'MLU zonal (80 km)'
fig_tit2 = 'MLU meridional (80 km)'
cmax = 0.1
cb_nam = '(m/s)'

# Read noise mean
# ---------------
f = nc.netcdf_file(file1,'r')
xpo = f.variables['xp'][:].copy()
ypo = f.variables['yp'][:].copy()
xto = f.variables['xt'][:].copy()
yto = f.variables['yt'][:].copy()
umo = f.variables['uco'][k-1,:,:].copy() 
vmo = f.variables['vco'][k-1,:,:].copy() 
f.close()

# Read noise fluctuations
# -----------------------
f = nc.netcdf_file(file2,'r')
tyrs = f.variables['time'][:].copy()
id1 = np.flatnonzero(tyrs>=t1)[0]
uro = f.variables['ur'][id1,k-1,:,:].copy() 
vro = f.variables['vr'][id1,k-1,:,:].copy() 
f.close()
# Total noise
uro += umo 
vro += vmo 
print uro.max(), vro.max()

# Plot zonal noise
# ----------------
fig = plt.figure(figsize=(5,6.5))
im  = plt.imshow(uro, interpolation='bicubic', origin='lower', 
        cmap='RdBu_r', vmin=-cmax, vmax=cmax, 
        extent=[xpo.min(), xpo.max(), yto.min(), yto.max()])
plt.xlabel(r'$x$ (km)')
plt.ylabel(r'$y$ (km)')
plt.title(r'%s'%(fig_tit1))
cb = plt.colorbar(im, orientation='horizontal', extend='both', 
        format='%3.2f', fraction=0.035, pad=0.1)
cb.set_label(cb_nam)
plt.savefig(output1, ndpi=200, bbox_inches='tight', pad_inches=0)

# Plot meridional noise
# ---------------------
fig = plt.figure(figsize=(5,6.5))
im  = plt.imshow(vro, interpolation='bicubic', origin='lower', 
        cmap='RdBu_r', vmin=-cmax, vmax=cmax, 
        extent=[xto.min(), xto.max(), ypo.min(), ypo.max()])
plt.xlabel(r'$x$ (km)')
plt.ylabel(r'$y$ (km)')
plt.title(r'%s'%(fig_tit2))
cb = plt.colorbar(im, orientation='horizontal', extend='both', 
        format='%3.2f', fraction=0.035, pad=0.1)
cb.set_label(cb_nam)
plt.savefig(output2, ndpi=200, bbox_inches='tight', pad_inches=0)

