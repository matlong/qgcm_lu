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
#base_dir = '/Volumes/Long/q-gcm/ocean-only/REF5/'
base_dir = '/Volumes/Long/q-gcm/ocean-only/SPOD80/LU-full/'
#subs_dir = ['yrs120-135/','yrs135-150/','yrs150-165/', \
#           'yrs165-180/','yrs180-195/','yrs195-210/']
subs_dir = ['yrs110-210/']

t0 = 120.0 # (year)
fnot = 9.37456e-5 # Coriolis parameter (s^-1)
hoc = [350.0,750.0,2900.0] # background thickness (m)
unit_sv = 1.0e6 # (1 Sv = 10^6 m^3/s) 

# Read 1st file
# -------------
file1 = base_dir + subs_dir[0] + 'ocpo.nc'
print 'Reading file: ', file1
f = nc.netcdf_file(file1,'r')
# Get p-grid axis (km)
xp = f.variables['xp'][:].copy() 
yp = f.variables['yp'][:].copy()
z  = f.variables['z'][:].copy()
#zi = f.variables['zi'][:].copy()
nxp, = xp.shape
nyp, = yp.shape
nz,  = z.shape
# Get time axis (years)
ntot = 0
tyrs = f.variables['time'][:].copy()
ids = np.flatnonzero(tyrs>t0) # truncate from t0
nt,  = ids.shape
# Recursive summing 
pm = np.zeros((nz,nyp,nxp)) # (m^2/s^2)
ps = pm.copy()
#hm = np.zeros((nz-1,nyp,nxp)) # (m)
#hs = hm.copy()
for i in range(0,nt):
    print 'Time (years) = ', tyrs[ids[i]]
    p = f.variables['p'][ids[i],:,:,:].copy()
    pm += p
    ps += p**2
#    h = f.variables['h'][ids[i],:,:,:].copy()
#    hm += h
#    hs += h**2
f.close() # close NetCDF file
ntot += nt

# Read other files
# ----------------
for fi in subs_dir[1:]:
    file1 = base_dir + fi + 'ocpo.nc'
    print 'Reading file: ', file1
    f = nc.netcdf_file(file1,'r')
    tyrs = f.variables['time'][:].copy()
    nt,  = tyrs.shape
    for j in range(1,nt):
        print 'Time (years) = ', tyrs[j]
        p = f.variables['p'][j,:,:,:].copy()
        pm += p
        ps += p**2
#        h = f.variables['h'][j,:,:,:].copy()
#        hm += h
#        hs += h**2
    f.close()
    ntot += nt-1

# Derive mean and std
# -------------------
pm = pm/ntot
ps = np.sqrt(ps/ntot - pm**2)
#h_mean = hm/ntot
#h_std = np.sqrt(hs/ntot - hm**2)

# From pressure to streamfunction (Sv)
# ------------------------------------
psi_mean = np.zeros((nz,nyp,nxp))
psi_std = psi_mean.copy()
for i in range(0,nz):
    sfac = hoc[i]/fnot/unit_sv
    psi_mean[i,:,:] = sfac*pm[i,:,:]
    psi_std[i,:,:]  = sfac*ps[i,:,:]

# Write output
# ------------
# Create file
outdir = base_dir + 'diag/'
if not os.path.exists(outdir):
    os.makedirs(outdir)
file1 = outdir + 'stat.nc'
f = nc.netcdf_file(file1,'w')
# Create dimensions
#f.createDimension('zoi',nz-1)
f.createDimension('zo',nz)
f.createDimension('ypo',nyp)
f.createDimension('xpo',nxp)
# Create variables
#zoi = f.createVariable('zoi', 'f4', ('zoi',))
zo = f.createVariable('zo', 'f4', ('zo',))
ypo = f.createVariable('ypo', 'f4', ('ypo',))
xpo = f.createVariable('xpo', 'f4', ('xpo',))
pmean = f.createVariable('pmean', 'f4', ('zo','ypo','xpo',))
pstd = f.createVariable('pstd', 'f4', ('zo','ypo','xpo',))
#hmean = f.createVariable('hmean', 'f4', ('zoi','ypo','xpo',))
#hstd = f.createVariable('hstd', 'f4', ('zoi','ypo','xpo',))
# Add attributes
#zoi.long_name = 'Ocean interface depth axis'
#zoi.units = 'km'
zo.long_name = 'Ocean mid-layer depth axis'
zo.units = 'km'
ypo.long_name = 'Ocean Y axis (p-grid)'
ypo.units = 'km'
xpo.long_name = 'Ocean X axis (p-grid)'
xpo.units = 'km'
pmean.long_name = 'Time-mean of streamfunction'
pmean.units = 'Sv'
pstd.long_name = 'Time-std of streamfunction'
pstd.units = 'Sv'
#hmean.long_name = 'Time-mean of interface displacement'
#hmean.units = 'm'
#hstd.long_name = 'Time-std of interface displacement'
#hstd.units = 'm'
# Write data
#zoi[:]  = zi
zo[:]  = z
ypo[:] = yp
xpo[:] = xp
pmean[:,:,:] = psi_mean
pstd[:,:,:]  = psi_std
#hmean[:,:,:] = h_mean
#hstd[:,:,:]  = h_std
# Close file
f.close()
print 'Output file written in ', outdir

'''
# Plot mean and std
# -----------------
cmax1 = 20.0 #25.0
lmax = 20.0 #30.0
step = 2.0 #3.0
clm = np.arange(-lmax,lmax+step,step)
#clm = clm[clm!=0.]
cmax2 = 10.0 #120.0
lmax = 10.0 #120.0
step = 1.0 #12.0
cls = np.arange(step,lmax+step,step)
for k in range(0,nz):
    # Mean
    fig = plt.figure(figsize=(5,6.5))
    im  = plt.imshow(psi_mean[k,:,:], \
            interpolation='none', origin='lower', \
            cmap='RdBu_r', vmin=-cmax1, vmax=cmax1, \
            extent=[xp.min(), xp.max(), yp.min(), yp.max()])
    ct  = plt.contour(xp, yp, psi_mean[k,:,:], \
            levels=clm, linewidths=1.0, colors='k')
    zc  = ct.collections[10]
    plt.setp(zc, linewidth=2)
    plt.xlabel(r'$x$ (km)')
    plt.ylabel(r'$y$ (km)')
    plt.title(r'%s'%(fig_tit))
    cb   = plt.colorbar(im, orientation='horizontal', extend='both', \
            format='%3.1f', pad=0.11)
    cb.set_label('(Sv)')
    output = outdir + 'pm%s-%s.eps'%(str(k+1),fig_nam)
    plt.savefig(output, ndpi=200, bbox_inches='tight', pad_inches=0)
    # Std
    fig  = plt.figure(figsize=(5,6.5))
    im   = plt.imshow(psi_std[k,:,:], \
            interpolation='none', origin='lower', \
            cmap='OrRd', vmin=0.0, vmax=cmax2, \
            extent=[xp.min(), xp.max(), yp.min(), yp.max()])
    ct   = plt.contour(xp, yp, psi_std[k,:,:], \
            levels=cls, linewidths=1.0, colors='k')
    plt.xlabel(r'$x$ (km)')
    plt.ylabel(r'$y$ (km)')
    plt.title(r'%s'%(fig_tit))
    cb = plt.colorbar(im, orientation='horizontal', extend='both', \
            format='%3.1f', pad=0.11)
    cb.set_label('(Sv)')
    output = outdir + 'ps%s-%s.eps'%(str(k+1),fig_nam)
    plt.savefig(output, ndpi=200, bbox_inches='tight', pad_inches=0)
print 'Outout figure printed in ', outdir
'''
