#!/usr/bin/env python

# Load modules
# ------------
import numpy as np
from numpy import linalg as LA
import scipy.io.netcdf as nc
from netCDF4 import Dataset
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
plt.ion()

# Set parameters
iodir = '/Volumes/Long/q-gcm/double_gyre/oc_only/sym_wind/no_sst/REF5/POD5/'
infile = iodir + 'ocdmd120.nc'
outfile = iodir + 'ocludat_dmd120.nc'
init_file = iodir + 'data/yrs090-095/ocref120.nc'
freq_cut = np.pi/(30.*86400.) # cutoff correlated/uncorrelated (s) 
tau = 1800. # (s)

# Read DMD data
f = Dataset(infile,'r')
xp = f.variables['xp'][:].copy() 
yp = f.variables['yp'][:].copy()
xt = f.variables['xt'][:].copy() 
yt = f.variables['yt'][:].copy()
z = f.variables['z'][:].copy()
tyrs = f.variables['time'][:].copy()
re = f.variables['lambda_real'][:].copy() 
im = f.variables['lambda_imag'][:].copy()
lam = re + 1j*im
re = f.variables['omega_real'][:].copy() 
im = f.variables['omega_imag'][:].copy() 
omg = re + 1j*im
re = f.variables['b_real'][:].copy() 
im = f.variables['b_imag'][:].copy() 
amp = re + 1j*im
uc = f.variables['umean'][...].copy()
vc = f.variables['vmean'][...].copy()
re = f.variables['umode_real'][...].copy() 
im = f.variables['umode_imag'][...].copy() 
um = re + 1j*im
re = f.variables['vmode_real'][...].copy() 
im = f.variables['vmode_imag'][...].copy() 
vm = re + 1j*im
del re, im
f.close()
dt = (tyrs[1]-tyrs[0])*365.*86400.
nxp = len(xp)
nyp = len(yp)
nxt = len(xt)
nyt = len(yt)
nl = len(z)

# Plot (unfiltered) eigenvalues and amptitudes 
plt.figure(figsize=(5,5))
plt.plot(lam.real, lam.imag, '.')
plt.grid(which='both', axis='both')
plt.xlabel(r'Re ($\lambda$)')
plt.ylabel(r'Im ($\lambda$)')
#plt.show()
plt.savefig('eval.eps', ndpi=200, bbox_inches='tight', pad_inches=0)

amp_mod = np.abs(amp)
plt.figure(figsize=(5,5))
plt.plot(1.0e5*omg.imag, amp_mod, '+')
plt.grid(which='both', axis='both')
plt.xlabel(r'Im ($\omega$) $\times 10^{-5}$')
plt.ylabel(r'amplitude')
#plt.show()
plt.savefig('amp.eps', ndpi=200, bbox_inches='tight', pad_inches=0)

# Filtering eigenvalues (kill those not on unit cercle)
lam_mod = np.abs(lam)
idm = np.where((lam_mod > 1.0 - 1.0e-2) & (lam_mod <= 1.0 + 1.0e-3))
mask = np.full(len(lam_mod), False, dtype=bool)
mask[idm] = True
lam = lam[mask]
omg = omg[mask]
amp = amp[mask]
amp_mod = amp_mod[mask]
um = um[mask,...]
vm = vm[mask,...]
del lam_mod

plt.figure(figsize=(5,5))
plt.plot(lam.real, lam.imag, '.')
plt.grid(which='both', axis='both')
plt.xlabel(r'Re ($\lambda$)')
plt.ylabel(r'Im ($\lambda$)')
#plt.show()
plt.savefig('eval_filt.eps', ndpi=200, bbox_inches='tight', pad_inches=0)

plt.figure(figsize=(5,5))
plt.plot(1.0e5*omg.imag, amp_mod, '+')
plt.grid(which='both', axis='both')
plt.xlabel(r'Im ($\omega$) $\times 10^{-5}$')
plt.ylabel(r'amplitude')
#plt.show()
plt.savefig('amp_filt.eps', ndpi=200, bbox_inches='tight', pad_inches=0)

# Seperate temporal scales
idm = np.where(np.abs(omg.imag) <= freq_cut)
mask = np.full(len(omg.imag), False, dtype=bool)
mask[idm] = True

um_c = um[mask,...]
vm_c = vm[mask,...]
um_nc = um[~mask,...]
vm_nc = vm[~mask,...]
del um, vm

lam_c = lam[mask]
lam_nc = lam[~mask]
del lam

omg_c = omg[mask]
omg_nc = omg[~mask]
del omg

amp_c = amp[mask]
amp_mod_c = amp_mod[mask]
amp_nc = amp[~mask]
amp_mod_nc = amp_mod[~mask]
del amp, amp_mod

plt.figure(figsize=(5,5))
plt.plot(lam_c.real, lam_c.imag, '.', label=r'correlated')
plt.plot(lam_nc.real, lam_nc.imag, '.', label='uncorrelated')
plt.grid(which='both', axis='both')
plt.xlabel(r'Re ($\lambda$)')
plt.ylabel(r'Im ($\lambda$)')
plt.legend(loc='best')
#plt.show()
plt.savefig('eval_filt_sep.eps', ndpi=200, bbox_inches='tight', pad_inches=0)

plt.figure(figsize=(5,5))
plt.plot(1.0e5*omg_c.imag, amp_mod_c, '+', label=r'correlated')
plt.plot(1.0e5*omg_nc.imag, amp_mod_nc, '+', label=r'uncorrelated')
plt.grid(which='both', axis='both')
plt.xlabel(r'Im ($\omega$) $\times 10^{-5}$')
plt.ylabel(r'amplitude')
plt.legend(loc='best')
#plt.show()
plt.savefig('amp_filt_sep.eps', ndpi=200, bbox_inches='tight', pad_inches=0)

# Filtering modes by energy
amp_max = float(input("Set trucation amplitude:\n"))
#idm = np.where(amp_mod_c >= amp_max)
idm = np.where( (amp_mod_c >= amp_max) & (np.abs(omg_c.imag) > 2.5e-7) )
mask = np.full(len(amp_mod_c), False, dtype=bool)
mask[idm] = True
lam_c = lam_c[mask]
omg_c = omg_c[mask]
amp_c = amp_c[mask]
amp_mod_c = amp_mod_c[mask]
um_c = um_c[mask]
vm_c = vm_c[mask]
nm_c = len(amp_mod_c)
print 'Number of modes used for correlated drift = ',nm_c

idm = np.where( (amp_mod_nc >= amp_max) & (np.abs(omg_nc.imag) < 7.e-6) )
mask = np.full(len(amp_mod_nc), False, dtype=bool)
mask[idm] = True
lam_nc = lam_nc[mask]
omg_nc = omg_nc[mask]
amp_nc = amp_nc[mask]
amp_mod_nc = amp_mod_nc[mask]
um_nc = um_nc[mask]
vm_nc = vm_nc[mask]
nm_nc = len(amp_mod_nc)
print 'Number of modes used for uncorrelated noise = ',nm_nc

plt.figure(figsize=(5,5))
plt.plot(lam_c.real, lam_c.imag, '.', label=r'correlated')
plt.plot(lam_nc.real, lam_nc.imag, '.', label='uncorrelated')
plt.xlim([-1.1,1.1])
plt.ylim([-1.1,1.1])
plt.grid(which='both', axis='both')
plt.xlabel(r'Re ($\lambda$)')
plt.ylabel(r'Im ($\lambda$)')
plt.legend(loc='best')
#plt.show()
plt.savefig('eval_filt_sep_cut.eps', ndpi=200, bbox_inches='tight', pad_inches=0)

plt.figure(figsize=(5,5))
plt.plot(1.0e5*omg_c.imag, amp_mod_c, '+', label=r'correlated')
plt.plot(1.0e5*omg_nc.imag, amp_mod_nc, '+', label=r'uncorrelated')
plt.xlim([-0.8,0.8])
plt.grid(which='both', axis='both')
plt.xlabel(r'Im ($\omega$) $\times 10^{-5}$')
plt.ylabel(r'amplitude')
plt.legend(loc='best')
#plt.show()
plt.savefig('amp_filt_sep_cut.eps', ndpi=200, bbox_inches='tight', pad_inches=0)

# Rescale truncated amplitudes
fin = Dataset(init_file,'r')
u = fin.variables['ur'][0,...].copy() 
v = fin.variables['vr'][0,...].copy() 
fin.close()
u -= uc
v -= vc
X0 = u.flatten() 
X0 = np.append(X0, v.flatten())
del u, v
nu = nl*nxp*nyt
nv = nl*nxt*nyp
re = np.zeros((nu+nv,nm_c), dtype=np.float64)
re[:nu,:] = um_c.real.reshape((nm_c,nu)).transpose() 
re[nu:,:] = vm_c.real.reshape((nm_c,nv)).transpose() 
im = np.zeros((nu+nv,nm_c), dtype=np.float64)
im[:nu,:] = um_c.imag.reshape((nm_c,nu)).transpose() 
im[nu:,:] = vm_c.imag.reshape((nm_c,nv)).transpose() 
Phi = re + 1j*im
Psi = np.matmul( Phi, LA.pinv(np.matmul(Phi.conj().transpose(), Phi)) )
amp_c = np.matmul(Psi.conj().transpose(), X0)
amp_mod_c = np.abs(amp_c)

re = np.zeros((nu+nv,nm_nc), dtype=np.float64)
re[:nu,:] = um_nc.real.reshape((nm_nc,nu)).transpose() 
re[nu:,:] = vm_nc.real.reshape((nm_nc,nv)).transpose() 
im = np.zeros((nu+nv,nm_nc), dtype=np.float64)
im[:nu,:] = um_nc.imag.reshape((nm_nc,nu)).transpose() 
im[nu:,:] = vm_nc.imag.reshape((nm_nc,nv)).transpose() 
Phi = re + 1j*im
del re, im
Psi = np.matmul( Phi, LA.pinv(np.matmul(Phi.conj().transpose(), Phi)) )
amp_nc = np.matmul(Psi.conj().transpose(), X0)
amp_mod_nc = np.abs(amp_nc)
del Phi, Psi, X0

plt.figure(figsize=(5,5))
plt.plot(1.0e5*omg_c.imag, amp_mod_c, '+', label=r'correlated')
plt.plot(1.0e5*omg_nc.imag, amp_mod_nc, '+', label=r'uncorrelated')
plt.xlim([-0.8,0.8])
plt.grid(which='both', axis='both')
plt.xlabel(r'Im ($\omega$) $\times 10^{-5}$')
plt.ylabel(r'amplitude')
plt.legend(loc='best')
#plt.show()
plt.savefig('amp_filt_sep_cut_res.eps', ndpi=200, bbox_inches='tight', pad_inches=0)

# Rescale DMD modes
for i in range(nm_c):
    um_c[i,...] *= amp_c[i]
    vm_c[i,...] *= amp_c[i]

for i in range(nm_nc):
    um_nc[i,...] *= amp_nc[i]
    vm_nc[i,...] *= amp_nc[i]

# Derive stationary variance
axx = tau*np.sum((um_nc.conj()*um_nc).real.astype('float64'), axis=0)
ayy = tau*np.sum((vm_nc.conj()*vm_nc).real.astype('float64'), axis=0)
axy = np.zeros((nl,nyp,nxp), dtype=np.float64)
ctmp = (um_nc[...,:-1,1:-1] + um_nc[...,1:,1:-1]).conj() \
      *(vm_nc[...,1:-1,:-1] + vm_nc[...,1:-1,1:])
axy[:,1:-1,1:-1] = 0.25*tau*np.sum( ctmp.real.astype('float64'), axis=0 )

# Save outputs
fout = Dataset(outfile, 'w', format='NETCDF4')
# Create dimensions
fout.createDimension('mode_c', nm_c)
fout.createDimension('mode_r', nm_nc)
fout.createDimension('yt', nyt)
fout.createDimension('xt', nxt)
fout.createDimension('yp', nyp)
fout.createDimension('xp', nxp)
fout.createDimension('z', nl)
# Create variables
mcid = fout.createVariable('mode_c', 'i4', ('mode_c',))
mrid = fout.createVariable('mode_r', 'i4', ('mode_r',))
ytid = fout.createVariable('yt', 'f8', ('yt',))
xtid = fout.createVariable('xt', 'f8', ('xt',))
ypid = fout.createVariable('yp', 'f8', ('yp',))
xpid = fout.createVariable('xp', 'f8', ('xp',))
zid = fout.createVariable('z', 'f8', ('z',))
wcid = fout.createVariable('omega_c', 'f8', ('mode_c',))
umid = fout.createVariable('uco', 'f8', ('z','yt','xp',))
vmid = fout.createVariable('vco', 'f8', ('z','yp','xt',))
ucreid = fout.createVariable('umode_real_c', 'f8', ('mode_c','z','yt','xp',))
ucimid = fout.createVariable('umode_imag_c', 'f8', ('mode_c','z','yt','xp',))
vcreid = fout.createVariable('vmode_real_c', 'f8', ('mode_c','z','yp','xt',))
vcimid = fout.createVariable('vmode_imag_c', 'f8', ('mode_c','z','yp','xt',))
wrid = fout.createVariable('omega_r', 'f8', ('mode_r',))
urreid = fout.createVariable('umode_real_r', 'f8', ('mode_r','z','yt','xp',))
urimid = fout.createVariable('umode_imag_r', 'f8', ('mode_r','z','yt','xp',))
vrreid = fout.createVariable('vmode_real_r', 'f8', ('mode_r','z','yp','xt',))
vrimid = fout.createVariable('vmode_imag_r', 'f8', ('mode_r','z','yp','xt',))
axxid = fout.createVariable('axx', 'f8', ('z','yt','xp',))
ayyid = fout.createVariable('ayy', 'f8', ('z','yp','xt',))
axyid = fout.createVariable('axy', 'f8', ('z','yp','xp',))
# Add attributes
mcid.long_name = 'Mode index (for correction drift)'
mrid.long_name = 'Mode index (for random noise)'
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
wcid.long_name = 'Continuous eigenvalues for correction drift'
wcid.units = 's^-1'
umid.long_name = 'Mean of zonal correction drift'
umid.units = 'm/s'
vmid.long_name = 'Mean of meridional correction drift'
vmid.units = 'm/s'
ucreid.long_name = 'Zonal DMD modes for correction drift (real part)'
ucreid.units = 'm/s'
vcreid.long_name = 'Meridional DMD modes for correction drift (real part)'
vcreid.units = 'm/s'
ucimid.long_name = 'Zonal DMD modes for correction drift (imag part)'
ucimid.units = 'm/s'
vcimid.long_name = 'Meridional DMD modes for correction drift (imag part)'
vcimid.units = 'm/s'
wrid.long_name = 'Continuous eigenvalues for random noise'
wrid.units = 's^-1'
urreid.long_name = 'Zonal DMD modes for random noise (real part)'
urreid.units = 'm/s'
vrreid.long_name = 'Meridional DMD modes for random noise (real part)'
vrreid.units = 'm/s'
urimid.long_name = 'Zonal DMD modes for random noise (imag part)'
urimid.units = 'm/s'
vrimid.long_name = 'Meridional DMD modes for random noise (imag part)'
vrimid.units = 'm/s'
axxid.long_name = 'Variance of zonal noise'
axxid.units = 'm^2/s'
ayyid.long_name = 'Variance of meridional noise'
ayyid.units = 'm^2/s'
axyid.long_name = 'Cross variance of noise'
axyid.units = 'm^2/s'
# Write data
mcid[:] = range(nm_c)
mrid[:] = range(nm_nc)
ytid[:] = yt
xtid[:] = xt
ypid[:] = yp
xpid[:] = xp
zid[:] = z
wcid[:] = omg_c.imag
umid[...] = -uc 
vmid[...] = -vc 
ucreid[...] = um_c.real 
vcreid[...] = vm_c.real
ucimid[...] = um_c.imag
vcimid[...] = vm_c.imag
wrid[:] = omg_nc.imag
urreid[...] = um_nc.real 
vrreid[...] = vm_nc.real
urimid[...] = um_nc.imag
vrimid[...] = vm_nc.imag
axxid[...] = axx
ayyid[...] = ayy
axyid[...] = axy
# Close file
fout.close()

print
print 'Program terminates'
