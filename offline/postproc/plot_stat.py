#!/usr/bin/env python

# Load modules
# ------------
import numpy as np
import scipy.io.netcdf as nc
import scipy.io as sio
from scipy import signal
import matplotlib.pyplot as plt
import os
#from netCDF4 import Dataset
plt.ion()

# Set parameters
with_sst = True
nam = 'EOF40'
tit = 'STO-EOF (40 km)'
l = 1 # 0 or 1
if with_sst:
    datdir = r'/Volumes/Long/q-gcm/double_gyre_ocean_only_sst/%s/yrs090-210/'%(nam)
    #datdir = r'/Volumes/Long/q-gcm/double_gyre_ocean_only_sst/%s/yrs195-210/'%(nam)
    datdir = r'/Volumes/Long/q-gcm/double_gyre_ocean_only_sst/%s/salt/'%(nam)
    cmax = 25. if l==0 else 50.
else:
    datdir = r'/Volumes/Long/q-gcm/double_gyre_ocean_only/%s/yrs090-210/'%(nam)
    #datdir = r'/Volumes/Long/q-gcm/double_gyre_ocean_only/%s/yrs195-210/'%(nam)
    datdir = r'/Volumes/Long/q-gcm/double_gyre_ocean_only/%s/salt/'%(nam)
    cmax = 10. if l==0 else 20.
outdir = '/Users/loli/Desktop/JAMES/salt/'

file1 = datdir + 'ocdiag.nc' 
print 'Read data from ',file1
fin = nc.netcdf_file(file1,'r')
nz = fin.dimensions['z']
x = fin.variables['xp'][:].copy() # (km)
y = fin.variables['yp'][:].copy()
pm = fin.variables['pstat_vm'][0,:,:,:].copy()
fin.close()
ax = [x.min(), x.max(), y.min(), y.max()]

nstep = 10  # nb. of contour intervals
step = cmax/nstep
clin = np.arange(-cmax,cmax+step,step)
plt.figure(figsize=(5,6.5))
im = plt.imshow(1.e3*pm[nz-l-1,...], interpolation='bilinear', origin='lower', cmap='RdBu_r', 
        vmin=-cmax, vmax=cmax, extent=ax)
plt.contour(x, y, 1.e3*pm[nz-l-1,...], levels=clin, linewidths=1.0, colors='k')
plt.yticks(rotation = 90) 
plt.xlabel(r'$x$ (km)')
plt.ylabel(r'$y$ (km)')
plt.title(tit)
tk = np.linspace(-cmax,cmax,5,endpoint=True)
cb = plt.colorbar(im, ticks=tk, orientation='horizontal', extend='both', fraction=0.035, pad=0.1)
cb.set_label(r'(Sv/km)')
if with_sst:
    plt.savefig(outdir + r'p%d-mean-%s-sst.pdf'%(l,nam), ndpi=200, bbox_inches='tight', pad_inches=0)
else:
    plt.savefig(outdir + r'p%d-mean-%s.pdf'%(l,nam), ndpi=200, bbox_inches='tight', pad_inches=0)
