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
#file1 = '/Volumes/Long/q-gcm/gyres_ocean_SST/REF5/diag/stats.nc'  
#file1 = '/Volumes/Long/q-gcm/gyres_ocean/LR120/yrs000-200/stats.nc'  
file1 = '/Volumes/Long/q-gcm/gyres-couple/stress-at-oc/REF5/diag/stats.nc'
var_nam = 'sstst' # 'pst' or 'hst'
ns = 1 # subsampling size
#fig_tit = 'DLR (120 km)'
fig_tit = 'REF'
k = 1 # layer
#output = '/Users/loli/Desktop/q-gcm/manuscript/ocemod-v1/figures/gyres/pom1-DLR120.eps' 
output = '/Users/loli/Desktop/STUOD/Reports/annual-2021/figures/sstm-REF40.eps'
#cmax = 8.0 # max. color
cmax = 12.5
cb_nam = '(K)' # '(Sv)' or '(m)'

print 'Layer = ', k
print 'Variable = ', var_nam
print 'Model = ', fig_tit

# Read time-mean
# --------------
f = nc.netcdf_file(file1,'r')
xpo = f.variables['xto'][::ns].copy() 
ypo = f.variables['yto'][::ns].copy()
#pom = f.variables[var_nam][0,k-1,::ns,::ns].copy() 
pom = f.variables[var_nam][0,::ns,::ns].copy() 
f.close()
	
# Plot contour
# ------------
fig = plt.figure(figsize=(5,6.5))
nstep = 10  # nb. of contour intervals
step = cmax/nstep
clin = np.arange(-cmax,cmax+step,step)
#clin = clin[clin!=0.]
print 'Contour levels = ', clin
im  = plt.imshow(pom, interpolation='bilinear', origin='lower', 
        cmap='RdYlBu_r', vmin=-cmax, vmax=cmax, 
        extent=[xpo.min(), xpo.max(), ypo.min(), ypo.max()])
ct = plt.contour(xpo, ypo, pom, levels=clin, linewidths=1.0, colors='k')
#zc = ct.collections[nstep]
#plt.setp(zc, linewidth=1.5)
plt.xlabel(r'$x$ (km)')
plt.ylabel(r'$y$ (km)')
plt.title(r'%s'%(fig_tit))
cb = plt.colorbar(im, orientation='horizontal', extend='both', 
        format='%3.1f', fraction=0.035, pad=0.1)
cb.set_label(cb_nam)
plt.savefig(output, ndpi=200, bbox_inches='tight', pad_inches=0)

