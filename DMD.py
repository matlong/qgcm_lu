#!/usr/bin/env python

# Load modules
# ------------
import numpy as np
from numpy import linalg as LA
import scipy.io.netcdf as nc
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import glob
import snapshot_POD as pod 
from netCDF4 import Dataset

# Set parammeters
# ---------------
base_dir = '/Volumes/Long/q-gcm/double_gyre/oc_only/sym_wind/no_sst/REF5/POD5/'
subs_dir = ['data/yrs090-095/','data/yrs095-100/','data/yrs100-105/','data/yrs105-110/', \
            'data/yrs110-115/','data/yrs115-120/','data/yrs120-125/','data/yrs125-130/']
infile = 'ocref120.nc'
outfile = 'ocdmd120.nc'
enpp = 0.99

# Read axis
# ---------
fin = Dataset(base_dir + subs_dir[0] + infile,'r')
xp = fin.variables['xp'][:].copy() # P-grid axis (km)
yp = fin.variables['yp'][:].copy()
xt = fin.variables['xt'][:].copy() # T-grid axis (km)
yt = fin.variables['yt'][:].copy()
z  = fin.variables['z'][:].copy() # vertical axis (km)
fin.close()
nxp = len(xp)
nyp = len(yp)
nxt = len(xt)
nyt = len(yt)
nl  = len(z)

# Collect data segments
# ---------------------
n = 1
tyrs = np.empty(0) 
u = np.empty((0,nl,nyt,nxp))
v = np.empty((0,nl,nyp,nxt))
for s in subs_dir:
    file1 = base_dir + s + infile
    print 'Opening file: ', file1
    fin = Dataset(file1,'r')
    nn = np.minimum(n,2) - 1
    tmp = fin.variables['time'][nn:].copy()
    tyrs = np.append(tyrs, tmp, axis=0) 
    tmp = fin.variables['ur'][nn:,...].copy()
    #tmp = fin.variables['ued'][nn:,...].copy()
    u = np.append(u, tmp, axis=0)
    tmp = fin.variables['vr'][nn:,...].copy()
    #tmp = fin.variables['ved'][nn:,...].copy()
    v = np.append(v, tmp, axis=0)
    del tmp
    n += 1
    fin.close() 
nt = len(tyrs)
dt = (tyrs[1]-tyrs[0])*365.*86400.
print 'Total number of snapshots = ',nt
print 'Sampling step = ',dt

# Build data matrix
um = np.mean(u, axis=0, dtype=np.float64)
vm = np.mean(v, axis=0, dtype=np.float64)
u -= um[np.newaxis] 
v -= vm[np.newaxis] 
nu = nl*nyt*nxp
nv = nl*nyp*nxt
X1 = np.zeros((nu+nv,nt), dtype=np.float64)
X1[:nu,:] = u.reshape((nt,nu)).transpose() 
X1[nu:,:] = v.reshape((nt,nv)).transpose() 

# POD
U1, S1, V1 = LA.svd(X1[:,:-1], full_matrices=False)
'''
ric = np.cumsum(S1, dtype=np.float64)/np.sum(S1, dtype=np.float64)
plt.figure()
plt.plot(ric)
plt.show()
idm = np.abs(ric - enpp).argmin()
print r'%d EOFs is used to capture %4.3f of total energy'%(idm,enpp)
# Low-rank
U1 = U1[:,:idm]
S1 = S1[:idm]
V1 = V1[:,:idm]
nm = len(S1)
'''

# DMD
A1 = np.matmul(np.matmul(np.matmul(U1.transpose(), X1[:,1:]), V1), np.diag(1./S1))
lamb, W1 = LA.eig(A1) # 'lamb' is discrete-time eigenvalues
Phi = np.matmul(np.matmul(np.matmul(X1[:,1:], V1), np.diag(1./S1)), W1) # DMD modes
omega = np.log(lamb)/dt # continuous-time eigenvalues
b = np.matmul(LA.pinv(Phi), X1[:,0]) # DMD amptitude

# Create ncfile
fout = Dataset(base_dir + outfile, 'w', format='NETCDF4')
nm = nt-1
# Create dimensions
fout.createDimension('time', nt)
fout.createDimension('mode', nm)
fout.createDimension('yt', nyt)
fout.createDimension('xt', nxt)
fout.createDimension('yp', nyp)
fout.createDimension('xp', nxp)
fout.createDimension('z', nl)
# Create variables
tid = fout.createVariable('time', 'f8', ('time',))
mid = fout.createVariable('mode', 'i4', ('mode',))
ytid = fout.createVariable('yt', 'f8', ('yt',))
xtid = fout.createVariable('xt', 'f8', ('xt',))
ypid = fout.createVariable('yp', 'f8', ('yp',))
xpid = fout.createVariable('xp', 'f8', ('xp',))
zid = fout.createVariable('z', 'f8', ('z',))
lreid = fout.createVariable('lambda_real', 'f8', ('mode',))
limid = fout.createVariable('lambda_imag', 'f8', ('mode',))
oreid = fout.createVariable('omega_real', 'f8', ('mode',))
oimid = fout.createVariable('omega_imag', 'f8', ('mode',))
breid = fout.createVariable('b_real', 'f8', ('mode',))
bimid = fout.createVariable('b_imag', 'f8', ('mode',))
umid = fout.createVariable('umean', 'f8', ('z','yt','xp',))
vmid = fout.createVariable('vmean', 'f8', ('z','yp','xt',))
ureid = fout.createVariable('umode_real', 'f8', ('mode','z','yt','xp',))
uimid = fout.createVariable('umode_imag', 'f8', ('mode','z','yt','xp',))
vreid = fout.createVariable('vmode_real', 'f8', ('mode','z','yp','xt',))
vimid = fout.createVariable('vmode_imag', 'f8', ('mode','z','yp','xt',))
# Add attributes
tid.long_name = 'Time axis'
tid.units = 'years'
mid.long_name = 'Mode index'
ytid.long_name = 'Ocean Y axis (T-grid)'
ytid.units = 'km'
xtid.long_name = 'Ocean X axis (T-grid)'
xtid.units = 'km'
ypid.long_name = 'Ocean Y axis (p-grid)'
ypid.units = 'km'
xpid.long_name = 'Ocean X axis (p-grid)'
xpid.units = 'km'
zid.long_name = 'Ocean mid-layer axis'
zid.units = 'km'
umid.long_name = 'Mean of zonal velocity'
umid.units = 'm/s'
vmid.long_name = 'Mean of meridional velocity'
vmid.units = 'm/s'
lreid.long_name = 'DMD eigenvalues (real part)'
limid.long_name = 'DMD eigenvalues (imag part)'
oreid.long_name = 'Continuous eigenvalues (real part)'
oimid.long_name = 'Continuous eigenvalues (imag part)'
breid.long_name = 'DMD amplitudes (real part)'
bimid.long_name = 'DMD amplitudes (imag part)'
ureid.long_name = 'Zonal DMD modes (real part)'
vreid.long_name = 'Meridional DMD modes (real part)'
uimid.long_name = 'Zonal DMD modes (imag part)'
vimid.long_name = 'Meridional DMD modes (imag part)'
# Write data
tid[:] = tyrs
mid[:] = range(nm)
ytid[:] = yt
xtid[:] = xt
ypid[:] = yp
xpid[:] = xp
zid[:] = z
lreid[:] = lamb.real
limid[:] = lamb.imag
oreid[:] = omega.real
oimid[:] = omega.imag
breid[:] = b.real
bimid[:] = b.imag
umid[:,:,:] = um
vmid[:,:,:] = vm
ureid[:,:,:,:] = (Phi[:nu,:].real.transpose()).reshape((nm,nl,nyt,nxp))
vreid[:,:,:,:] = (Phi[nu:,:].real.transpose()).reshape((nm,nl,nyp,nxt))
uimid[:,:,:,:] = (Phi[:nu,:].imag.transpose()).reshape((nm,nl,nyt,nxp))
vimid[:,:,:,:] = (Phi[nu:,:].imag.transpose()).reshape((nm,nl,nyp,nxt))
# Close file
fout.close()

print
print 'Program terminates'

