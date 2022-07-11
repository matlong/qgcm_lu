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

def p2uv(p, rdxf0, bcco):
    "Derive geostrophic velocities (on P-grid) from pressure"
    zbfc = rdxf0/(0.5*bcco+1.0)
    hxfc = 0.5*rdxf0
    u = np.zeros_like(p)
    v = np.zeros_like(p)
    # Inner points
    u[1:-1,1:-1] = -hxfc*( p[2:,1:-1] - p[0:-2,1:-1] )
    v[1:-1,1:-1] =  hxfc*( p[1:-1,2:] - p[1:-1,0:-2] )
    # Boundary values
    u[0,:] = -zbfc*( p[1,:] - p[0,:] )
    u[-1,:] = -zbfc*( p[-1,:] - p[-2,:] )
    v[:,0] = zbfc*( p[:,1] - p[:,0] )
    v[:,-1] = zbfc*( p[:,-1] - p[:,-2] )
    return u, v

def p2w(p, rdxm2, bcco):
    """Compute Laplacian of `p` with mixed boundary conditions"""
    bcfc = bcco*rdxm2/( 0.5*bcco + 1. )
    w = np.zeros_like(p, dtype=np.float64)
    # Inner points
    w[...,1:-1,1:-1] = rdxm2*( p[...,:-2,1:-1] + p[...,1:-1,:-2] + p[...,1:-1,2:] \
                             + p[...,2:,1:-1] - 4.*p[...,1:-1,1:-1] )
    # N & S boundaries (including corners) 
    w[..., 0,:] = bcfc*( p[..., 1,:] - p[..., 0,:] )
    w[...,-1,:] = bcfc*( p[...,-2,:] - p[...,-1,:] )
    # W & E boundaries
    w[...,1:-1, 0] = bcfc*( p[...,1:-1, 1] - p[...,1:-1, 0] )
    w[...,1:-1,-1] = bcfc*( p[...,1:-1,-2] - p[...,1:-1,-1] )
    return w

def coarse_grain(ph, mask, ratio):
    """Coarse-graining of HR data on LR mesh."""
    # Low-pass filtering in Fourier
    fh = np.fft.fft2(ph[:-1,:-1], axes=(-2,-1))
    ptmp = np.zeros_like(ph, dtype=np.float64)
    ptmp[1:-1,1:-1] = (np.fft.ifft2(fh*mask, axes=(-2,-1)).real)[1:,1:]
    pbc = ph[0,0]#*np.mean(ptmp)/np.mean(ph[1:-1,1:-1])
    # Subsampling
    ptmp = ptmp[ratio-1:-ratio+1:ratio,ratio-1:-ratio+1:ratio]
    pl = np.zeros((ptmp.shape[0]+2,ptmp.shape[1]+2), dtype=np.float64)
    pl[0,:] = pbc
    pl[-1,:] = pbc
    pl[:,0] = pbc
    pl[:,-1] = pbc
    pl[1:-1,1:-1] = ptmp # inner points    
    return pl

# Set parameters
# --------------
with_sst = True
modnam = 'EOF40'
titnam = 'STO-EOF (40 km)'
ratio = 1
k = 1 # layer
if with_sst:
    file1 = r'/Volumes/Long/q-gcm/double_gyre_ocean_only_sst/%s/yrs090-210/ocpo.nc'%(modnam)
    #file1 = r'/Volumes/Long/q-gcm/double_gyre_ocean_only_sst/%s/yrs150-165/ocpo.nc'%(modnam)
    file1 = r'/Volumes/Long/q-gcm/double_gyre_ocean_only_sst/%s/salt/ocpo.nc'%(modnam)
    wmax = 0.1 if k==0 else 0.05
else:
    file1 = r'/Volumes/Long/q-gcm/double_gyre_ocean_only/%s/yrs090-210/ocpo.nc'%(modnam)
    #file1 = r'/Volumes/Long/q-gcm/double_gyre_ocean_only/%s/yrs150-165/ocpo.nc'%(modnam)
    file1 = r'/Volumes/Long/q-gcm/double_gyre_ocean_only/%s/salt/ocpo.nc'%(modnam)
    wmax = 0.04 if k==0 else 0.02
outdir = '/Users/loli/Desktop/JAMES/salt/' 
t1 = 150.0 # time (year)
f0 = 9.37456e-5 # Coriolis param. (s^-1)
bcco = 0.2
qmax = 0.9 # 0.5, 0.9

# Read PV
# -------
f = nc.netcdf_file(file1,'r')
xpo = f.variables['xp'][:].copy() 
ypo = f.variables['yp'][:].copy()
tyrs = f.variables['time'][:].copy()
id1 = np.flatnonzero(tyrs>=t1)[0]
po = f.variables['p'][id1,k,:,:].copy() 
qo = f.variables['q'][id1,k,:,:].copy() 
f.close()

# Vorticity
dxo = (xpo[1]-xpo[0])*1.0e3
rdxom2 = 1.0/(f0*dxo**2)
wo = p2w(po, rdxom2, bcco)

if ratio>1:
    xpo, ypo = xpo[::ratio], ypo[::ratio]
    # Set mask for low-pass filtering
    [nypo_HR, nxpo_HR] = po.shape
    mask = np.ones((nypo_HR-1,nxpo_HR-1), dtype=np.float64) 
    mask[(len(ypo)-1)/2:(nypo_HR-1)/2,:] = 0.
    mask[:,(len(xpo)-1)/2:(nxpo_HR-1)/2] = 0.
    # Downsamping
    po1 = coarse_grain(po, mask, ratio)
else:
    po1 = po.copy()

# Velocity
xo, yo = np.meshgrid(xpo, ypo)
dxo = (xpo[1]-xpo[0])*1.0e3
rdxof0 = 1.0/(f0*dxo)
uo, vo = p2uv(po1, rdxof0, bcco)

# Plot
# ----
'''
fig = plt.figure(figsize=(5,6.5))
im = plt.quiver(xo, yo, uo, vo, width=0.0015)
im  = plt.imshow(qo/f0, interpolation='bicubic', origin='lower', 
        cmap='RdBu_r', vmin=-qmax, vmax=qmax, 
        extent=[xpo.min(), xpo.max(), ypo.min(), ypo.max()])
plt.xlabel(r'$x$ (km)')
plt.ylabel(r'$y$ (km)')
plt.title(r'%s'%(titnam))
cb = plt.colorbar(im, orientation='horizontal', extend='both', 
        format='%3.1f', fraction=0.035, pad=0.1)
cb.set_label('(unit of $f_0$)')
plt.savefig(outdir + r'q%d-%s-sst.pdf'%(k+1,modnam), ndpi=200, bbox_inches='tight', pad_inches=0)
'''

fig = plt.figure(figsize=(5,6.5))
im = plt.quiver(xo, yo, uo, vo, width=0.0015)
im  = plt.imshow(wo/f0, interpolation='bicubic', origin='lower', 
        cmap='RdBu_r', vmin=-wmax, vmax=wmax, 
        extent=[xpo.min(), xpo.max(), ypo.min(), ypo.max()])
plt.yticks(rotation = 90) 
plt.xlabel(r'$x$ (km)')
plt.ylabel(r'$y$ (km)')
plt.title(r'%s'%(titnam))
tk = np.linspace(-wmax, wmax, 5, endpoint=True) 
cb = plt.colorbar(im, ticks=tk, format='%3.2f', 
        orientation='horizontal', extend='both', fraction=0.035, pad=0.1)
cb.set_label('(unit of $f_0$)')
if with_sst:
    plt.savefig(outdir + r'w%d-%s-sst.pdf'%(k+1,modnam), ndpi=200, bbox_inches='tight', pad_inches=0)
else:
    plt.savefig(outdir + r'w%d-%s.pdf'%(k+1,modnam), ndpi=200, bbox_inches='tight', pad_inches=0)
