#!/usr/bin/env python3

import numpy as np
import scipy.io.netcdf as nc
import matplotlib.pyplot as plt

def iterative_mean(it, psi_me, psi):
  '''
  Parameters
  ----------
  it : iteration number (starts at 1)
  psi_me : mean
  psi: field
  Returns
  -------
  psi_me : mean
  '''

  return psi_me + (psi - psi_me)/it


def iterative_variance(it, psi_me, psi_var, psi):
  '''
  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
  Welford's online algorithm
  Parameters
  ----------
  it : iteration number (starts at 1)
  psi_me : mean
  psi_var : variance
  psi: field
  Returns
  -------
  psi_me : mean
  psi_var : variance
  '''

  delta = psi - psi_me
  psi_me = psi_me + delta/it
  delta2 = psi - psi_me
  psi_var = (psi_var*(it-1) + delta*delta2)/it

  return psi_me, psi_var

# Parameters
#mdir = '/media/long/Long/q-gcm/double_gyre_coupled/DMD80/'
mdir = '/media/long/Long/q-gcm/double_gyre_coupled/REF5/'
#sdir = ['yrs000-180/']
sdir = ['yrs090-105/','yrs105-120/','yrs120-135/','yrs135-150/', \
        'yrs150-165/','yrs165-180/']
t0 = 90.
f0 = 9.37456e-5 # Coriolis param. (s^-1)
h = [350.0,750.0,2900.0] # thickness (m)
gp = [0.025, 0.0125] # reduced gravity (m/s^2)
rho = 1.e3 # density (kg/m^3)
plot_res = False

# Read 1st file
file1 = mdir + sdir[0] + 'ocpo.nc'
fin = nc.netcdf_file(file1, 'r')
zi = fin.variables['zi'][:].copy().astype('float64')
z = fin.variables['z'][:].copy().astype('float64')
y = fin.variables['yp'][:].copy().astype('float64')
x = fin.variables['xp'][:].copy().astype('float64') # P-grid axis (km)
nz, ny, nx = len(z), len(y), len(x) 
h, gp = np.asarray(h), np.asarray(gp)
dx = 1.e3*(x[1]-x[0]) # (m)
rdxf0, rdz = 1./(f0*dx), 2./(h[:-1] + h[1:])
N2 = rdz*gp
t = fin.variables['time'][:].copy().astype('float64')
id0 = np.abs(t - t0).argmin()
t = t[id0:]
n = 1
u_mean = np.zeros((nz,ny-1,nx), dtype=np.float64)
v_mean = np.zeros((nz,ny,nx-1), dtype=np.float64)
b_mean = np.zeros((nz-1,ny,nx), dtype=np.float64)
u_var, v_var, b_var = u_mean.copy(), v_mean.copy(), b_mean.copy()
for i in range(len(t)):
    print(f'iter = {n}')
    # Read pressure
    p = fin.variables['p'][i+id0].copy().astype('float64')
    # Derive velocities
    u, v = -rdxf0*(p[:,1:,:] - p[:,:-1,:]), rdxf0*(p[...,1:] - p[...,:-1])
    # Derive buoyancy
    b = rdz[:,np.newaxis,np.newaxis]*(p[:-1] - p[1:])
    # Update stattistics
    u_mean, u_var = iterative_variance(n, u_mean, u_var, u)
    v_mean, v_var = iterative_variance(n, v_mean, v_var, v)
    b_mean, b_var = iterative_variance(n, b_mean, b_var, b)
    n += 1
fin.close()

# Read subsequent files
for s in sdir[1:]:
    file1 = mdir + s + 'ocpo.nc'
    print(f'Open file: {file1}')
    fin = nc.netcdf_file(file1, 'r')
    nt = fin.dimensions['time']
    for i in range(nt-1):
        print(f'iter = {n}')
        # Read pressure
        p = fin.variables['p'][i].copy().astype('float64')
        # Derive velocities
        u, v = -rdxf0*(p[:,1:,:] - p[:,:-1,:]), rdxf0*(p[...,1:] - p[...,:-1])
        # Derive buoyancy
        b = rdz[:,np.newaxis,np.newaxis]*(p[:-1] - p[1:])
        # Update stattistics
        u_mean, u_var = iterative_variance(n, u_mean, u_var, u)
        v_mean, v_var = iterative_variance(n, v_mean, v_var, v)
        b_mean, b_var = iterative_variance(n, b_mean, b_var, b)
        n += 1
    fin.close()

# Energy (interpolated onto T-grid)
um2, vm2 = u_mean**2, v_mean**2
mke = 0.25*(um2[...,:-1] + um2[...,1:] + vm2[:,:-1,:] + vm2[:,1:,:])
eke = 0.25*(u_var[...,:-1] + u_var[...,1:] + v_var[:,:-1,:] + v_var[:,1:,:])
mpe, epe = 0.5*b_mean**2/N2[:,np.newaxis,np.newaxis], 0.5*b_var/N2[:,np.newaxis,np.newaxis]

file1 = mdir + sdir[-1] + 'ocdiag.nc'
file1 = mdir + sdir[-1] + 'ocdiag1.nc'
print(f'Append outputs into: {file1}')
fout = nc.netcdf_file(file1, 'w')
fout.createDimension('zi', nz-1)
fout.createDimension('z', nz)
fout.createDimension('yp', ny)
fout.createDimension('xp', nx)
fout.createDimension('yt', ny-1)
fout.createDimension('xt', nx-1)
fout.createVariable('mke', 'f8', ('z','yt','xt',))
fout.createVariable('eke', 'f8', ('z','yt','xt',))
fout.createVariable('mpe', 'f8', ('zi','yp','xp',))
fout.createVariable('epe', 'f8', ('zi','yp','xp',))
fout.variables['mke'].long_name = 'Mean kinetic energy'
fout.variables['mke'].units = 'm^2/s^2'
fout.variables['eke'].long_name = 'Eddy kinetic energy'
fout.variables['eke'].units = 'm^2/s^2'
fout.variables['mpe'].long_name = 'Mean potential energy'
fout.variables['mpe'].units = 'm^2/s^2'
fout.variables['epe'].long_name = 'Eddy potential energy'
fout.variables['epe'].units = 'm^2/s^2'
fout.variables['mke'][...] = mke
fout.variables['eke'][...] = eke
fout.variables['mpe'][...] = mpe
fout.variables['epe'][...] = epe
fout.close()

if plot_res:
    # Vertical int
    hm, ht = 0.5*(h[:-1] + h[1:]), np.sum(h)
    mke_int, eke_int = rho*np.einsum('kji,k->ji',mke,h)/ht, rho*np.einsum('kji,k->ji',eke,h)/ht
    mpe_int, epe_int = rho*np.einsum('kji,k->ji',mpe,hm)/ht, rho*np.einsum('kji,k->ji',epe,hm)/ht
    # Plot KE
    plt.figure()
    plt.subplot(121)
    plt.imshow(mke_int, origin='lower', interpolation='bilinear', cmap='Spectral_r', 
               vmin=0., vmax=0.75*mke_int.max())
    plt.title('MKE')
    plt.subplot(122)
    plt.imshow(eke_int, origin='lower', interpolation='bilinear', cmap='Spectral_r', 
               vmin=0., vmax=0.75*eke_int.max())
    plt.title('EKE')
    plt.show()
    print(f'Max of MKE = {mke_int.max()}') 
    print(f'Max of EKE = {eke_int.max()}') 
    # Plot PE
    plt.figure()
    plt.subplot(121)
    plt.imshow(mpe_int, origin='lower', interpolation='bilinear', cmap='Spectral_r', 
               vmin=0., vmax=0.75*mpe_int.max())
    plt.title('MPE')
    plt.subplot(122)
    plt.imshow(epe_int, origin='lower', interpolation='bilinear', cmap='Spectral_r', 
               vmin=0., vmax=0.75*epe_int.max())
    plt.title('EPE')
    plt.show()
    print(f'Max of MPE = {mpe_int.max()}') 
    print(f'Max of EPE = {epe_int.max()}') 

