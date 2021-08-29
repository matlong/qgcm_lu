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
#file1 = '/Volumes/Long/q-gcm/gyres_ocean_SST/REF5/yrs105-120/ocpo.nc'
file1 = '/Volumes/Long/q-gcm/gyres_ocean_SST/POD80-BC/yrs000-200/ocpo.nc'
beta = 1.7536e-11
k = 1 # layer
t1 = 120.0 # time (year)
# Outputs
output = r'../manuscript/ocemod-v1/figures/sst/w%s-MLUBC80.eps'%(k)
#fig_tit = r'Layer %s'%(k)
fig_tit = 'MLU-BC (80 km)'
cunit = 1.0e-5
cmax = 5.0
cb_nam = '($10^{-5}$ s$^{-1}$)'

# Read PV
# -------
f = nc.netcdf_file(file1,'r')
xpo = f.variables['xp'][:].copy() 
ypo = f.variables['yp'][:].copy()
tyrs = f.variables['time'][:].copy()
id1 = np.flatnonzero(tyrs>=t1)[0]
#print id1, tyrs[id1]
qo = f.variables['q'][id1,k-1,:,:].copy() 
f.close()

# Substract beta effect
# ---------------------
betay = 0.*qo
y0 = 0.5e3*ypo[-1]
for i in range(betay.shape[1]):
    betay[:,i] = beta*( 1.0e3*ypo - y0 )
wo = qo - betay
print wo.max()

# Plot
# ----
fig = plt.figure(figsize=(5,6.5))
im  = plt.imshow(wo/cunit, interpolation='bilinear', origin='lower', 
        cmap='RdBu_r', vmin=-cmax, vmax=cmax, 
        extent=[xpo.min(), xpo.max(), ypo.min(), ypo.max()])
plt.xlabel(r'$x$ (km)')
plt.ylabel(r'$y$ (km)')
plt.title(r'%s'%(fig_tit))
cb = plt.colorbar(im, orientation='horizontal', extend='both', 
        format='%3.1f', fraction=0.035, pad=0.1)
cb.set_label(cb_nam)
plt.savefig(output, ndpi=200, bbox_inches='tight', pad_inches=0)





