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
with_sst = True
res = 80
if with_sst:
    main_dir = '/Volumes/Long/q-gcm/double_gyre_ocean_only_sst/' 
else:
    main_dir = '/Volumes/Long/q-gcm/double_gyre_ocean_only/'     
subdir_ref = 'REF5/yrs195-210/'
subdir_mod = [r'DET%d/yrs090-210/'%(res), r'EOF%d/yrs090-210/'%(res), r'EOFP%d/yrs090-210/'%(res)]
subdir_mod = [r'DET%d/yrs090-210/'%(res), r'EOF%d/salt/'%(res), r'EOFP%d/salt/'%(res)]
nam_mod = [r'DET (%d km)'%(res), r'STO-EOF (%d km)'%(res), r'STO-EOF-P (%d km)'%(res)]
outdir = '/Users/loli/Desktop/JAMES/salt/'

# Read data
file1 = main_dir + subdir_ref + 'ocdiag.nc'
fin = nc.netcdf_file(file1,'r')
nz = fin.dimensions['z']
h = fin.variables['h'][:].copy()
wnh = fin.variables['k'][:].copy() # isotropic wavenumbers
keh = fin.variables['psdke'][:,:].copy()
fin.close()

file1 = main_dir + subdir_mod[0] + 'ocdiag.nc'
fin = nc.netcdf_file(file1,'r')
wnl = fin.variables['k'][:].copy()
fin.close()
nmod = len(subdir_mod)
kel = np.zeros((nmod,nz,len(wnl)), dtype='float64')
for m in range(nmod):
    file1 = main_dir + subdir_mod[m] + 'ocdiag.nc'
    fin = nc.netcdf_file(file1,'r')
    kel[m,...] = fin.variables['psdke'][...].copy()
    fin.close()

dwn = wnl[1]-wnl[0]
wnh /= np.pi
wnl /= np.pi
keh *= np.pi/np.sum(h) 
kel *= np.pi/np.sum(h)
dwh = wnh[1]-wnh[0]

# Plot energy spectrum
plt.figure(figsize=(5,3))
plt.loglog(wnh, keh[0,:], 'k', label='REF (5 km)')
for m in range(nmod):
    plt.loglog(wnl, kel[m,0,:], label=nam_mod[m])
plt.xlim([5e-7,wnh[-1]+dwh])
if with_sst:
    plt.ylim([1e1,1e6])
else:
    plt.ylim([1.5e1,1e5])
plt.grid(which='both', axis='both')
plt.legend(loc='lower left')
plt.xlabel(r'Isotropic wavenumbers (m$^{-1}$)') 
plt.title(r'Power spectral density (J/m$^2$)') 
if with_sst:
    plt.savefig(outdir+r'spec%d-up-sst.pdf'%(res), ndpi=200, bbox_inches='tight', pad_inches=0)
else:
    plt.savefig(outdir+r'spec%d-up.pdf'%(res), ndpi=200, bbox_inches='tight', pad_inches=0)

plt.figure(figsize=(5,3))
plt.loglog(wnh, np.sum(keh,axis=0), 'k', label='REF (5 km)')
for m in range(nmod):
    plt.loglog(wnl, np.sum(kel[m,:,:],axis=0), label=nam_mod[m])
plt.xlim([5e-7,wnh[-1]+dwh])
if with_sst:
    plt.ylim([1e2,1e7])
else:
    plt.ylim([1.5e1,3e5])
plt.grid(which='both', axis='both')
plt.legend(loc='upper right')
plt.xlabel(r'Isotropic wavenumbers (m$^{-1}$)') 
plt.title(r'Power spectral density (J/m$^2$)') 
if with_sst:
    plt.savefig(outdir+r'spec%d-mean-sst.pdf'%(res), ndpi=200, bbox_inches='tight', pad_inches=0)
else:
    plt.savefig(outdir+r'spec%d-mean.pdf'%(res), ndpi=200, bbox_inches='tight', pad_inches=0)

