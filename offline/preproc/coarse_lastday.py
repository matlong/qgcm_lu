#!/usr/bin/env python3

import numpy as np
import scipy.ndimage
from scipy import interpolate
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def coarse_grain_P(ph, ratio):
    """Coarse-graining of HR P-grid data on LR mesh."""
    ph_m = area_int(ph, 0.5, 0.5)/(ratio**2.)
    pf = scipy.ndimage.gaussian_filter(ph, sigma=0.25*ratio, mode='constant')[ratio-1:-ratio+1:ratio,ratio-1:-ratio+1:ratio]
    pf_m = area_int(pf, 1.0, 1.0)
    pbc = (ph_m - pf_m)/(np.sum(pf.shape)-1.)
    pl = np.zeros((pf.shape[0]+2,pf.shape[1]+2), dtype=np.float64)
    pl[1:-1,1:-1] = pf.copy()
    pl[0,:] = pbc
    pl[-1,:] = pbc
    pl[1:-1,0] = pbc
    pl[1:-1,-1] = pbc
    return pl

def coarse_grain_T(th, xh, yh, xl, yl, ratio):
    """Coarse-graining of HR T-grid data on LR mesh."""
    tf = scipy.ndimage.gaussian_filter(th, sigma=0.5*ratio, mode='mirror')
    tf_int = interpolate.interp2d(xh, yh, tf)
    return tf_int(xl, yl)

def area_int(p, wefc, snfc):
    "Area-integration of 2D array with specific W-E and S-N boundary factors"    
    res = 0.
    # Inner points
    res += np.sum(p[1:-1,1:-1])
    # S-N boundary
    res += snfc*( np.sum(p[0,1:-1]) + np.sum(p[-1,1:-1]) ) 
    # W-E boundary
    res += wefc*( np.sum(p[1:-1,0]) + np.sum(p[1:-1,-1]) ) 
    # Corners
    res += wefc*snfc*( p[0,0] + p[-1,0] + p[0,-1] + p[-1,-1] )
    return res

def init_wvnb(nx, ny, dx, dy):
    """Initialize horizaontal and isotropic wavenumbers `wv` and `kr`"""
    assert nx % 2 == 0, 'Wavenumbers length in x-axis is not even'
    assert ny % 2 == 0, 'Wavenumbers length in y-axis is not even'
    # Create horizontal wavenumbers
    dkx = 2.*np.pi/(nx*dx) 
    dky = 2.*np.pi/(ny*dy)
    kx = dkx * np.append( np.arange(0.,nx/2), np.arange(-nx/2,0.) )
    ky = dky * np.append( np.arange(0.,ny/2), np.arange(-ny/2,0.) )
    kxx, kyy = np.meshgrid(kx.astype('float64'), ky.astype('float64'))
    wv = np.sqrt(kxx**2 + kyy**2).flatten()
    # Create isotropic wavenumbers
    nmax = np.ceil( np.sqrt(2.)*np.maximum(nx,ny)/2. )
    dkr = np.maximum(dkx,dky)
    kr = dkr * np.arange(1.,nmax+1, dtype=np.float64)
    # Scaling factor for spectrum due to DFT and integration
    efac = dx*dy*np.minimum(dkx,dky)/( 4.*np.pi**2*nx*ny )
    return wv, kr, efac

def iso_spec(fh, wv, kr, efc):
    """Compute isotropic spectrum `psdr` of 2D Fourier coef. `fh`"""
    nk = kr.shape[0] # number of wavenumbers
    fh = fh.reshape((fh.shape[:len(fh.shape)-2] + (-1,)))
    psdr = np.zeros((fh.shape[:len(fh.shape)-1] + (nk,)), dtype=np.float64)
    # Summing over annular rings
    dkr = kr[1] - kr[0]
    for p in range(nk):
        # Integration between lower and upper bounds of annular rings
        idk = (wv >= kr[p]-dkr/2) & (wv <= kr[p]+dkr/2)
        psdr[...,p] = efc * np.sum(fh[...,idk], axis=-1, dtype=np.float64) 
    return psdr

# Set parameters
with_sst = True
dxl = 120.0 # (km)
if with_sst:
    indir = '/Volumes/Long/q-gcm/double_gyre_ocean_only_sst/REF5/yrs075-090/'
    outdir = '/Volumes/Long/q-gcm/double_gyre_ocean_only_sst/REF5/data1d/'
else:
    indir = '/Volumes/Long/q-gcm/double_gyre_ocean_only/REF5/yrs075-090/'
    outdir = '/Volumes/Long/q-gcm/double_gyre_ocean_only/REF5/data1d/'
check_opt = True
f0 = 9.37456e-5 # Coriolis param. (s^-1)

# Read HR data
file1 = indir + 'lastday.nc'
print(f'Read file: {file1}')
fin = Dataset(file1, 'r') 
x = fin.variables['xpo'][:].data.astype('float64') # P-grid axis (km)
y = fin.variables['ypo'][:].data.astype('float64')
dxh = x[1] - x[0] # (km)
nxh, nyh = len(x), len(y)
nz = fin.variables['zo'][:].size
xt, yt = 0.5*(x[1:] + x[:-1]), 0.5*(y[1:] + y[:-1])

# Create LR axis
ratio = int(dxl//dxh)
nxl, nyl = (nxh-1)//ratio+1, (nyh-1)//ratio+1
xl, yl = dxl*np.arange(nxl), dxl*np.arange(nyl) # (km)
xlt, ylt = 0.5*(xl[1:] + xl[:-1]), 0.5*(yl[1:] + yl[:-1])

# Initialize output ncfile
file1 = outdir + r'lastday%d.nc'%(int(dxl))
print(f'Output file: {file1}')
fout = Dataset(file1, 'w')
# Create dimensions
fout.createDimension('time', 1)
fout.createDimension('zo', nz)
fout.createDimension('ypo', nyl)
fout.createDimension('xpo', nxl)
fout.createDimension('yto', nyl-1)
fout.createDimension('xto', nxl-1)
fout.createDimension('za', nz)
fout.createDimension('ypa', nyl)
fout.createDimension('xpa', nxl)
fout.createDimension('yta', nyl-1)
fout.createDimension('xta', nxl-1)
# Create variables
fout.createVariable('time', 'f8', ('time',))
fout.createVariable('zo', 'f8', ('zo',))
fout.createVariable('ypo', 'f8', ('ypo',))
fout.createVariable('xpo', 'f8', ('xpo',))
fout.createVariable('yto', 'f8', ('yto',))
fout.createVariable('xto', 'f8', ('xto',))
fout.createVariable('po', 'f8', ('zo','ypo','xpo',))
fout.createVariable('pom', 'f8', ('zo','ypo','xpo',))
fout.createVariable('sst', 'f8', ('yto','xto',))
fout.createVariable('sstm', 'f8', ('yto','xto',))
fout.createVariable('za', 'f8', ('za',))
fout.createVariable('ypa', 'f8', ('ypa',))
fout.createVariable('xpa', 'f8', ('xpa',))
fout.createVariable('yta', 'f8', ('yta',))
fout.createVariable('xta', 'f8', ('xta',))
fout.createVariable('pa', 'f8', ('za','ypa','xpa',))
fout.createVariable('pam', 'f8', ('za','ypa','xpa',))
fout.createVariable('ast', 'f8', ('yta','xta',))
fout.createVariable('astm', 'f8', ('yta','xta',))
fout.createVariable('hmixa', 'f8', ('yta','xta',))
fout.createVariable('hmixam', 'f8', ('yta','xta',))
# Add atttributes
fout.variables['time'].units = 'years'
fout.variables['zo'].units = 'km'
fout.variables['ypo'].units = 'km'
fout.variables['xpo'].units = 'km'
fout.variables['yto'].units = 'km'
fout.variables['xto'].units = 'km'
fout.variables['po'].units = 'm^2/s^2'
fout.variables['pom'].units = 'm^2/s^2'
fout.variables['sst'].units = 'K'
fout.variables['sstm'].units = 'K'
fout.variables['za'].units = 'km'
fout.variables['ypa'].units = 'km'
fout.variables['xpa'].units = 'km'
fout.variables['yta'].units = 'km'
fout.variables['xta'].units = 'km'
fout.variables['pa'].units = 'm^2/s^2'
fout.variables['pam'].units = 'm^2/s^2'
fout.variables['ast'].units = 'K'
fout.variables['astm'].units = 'K'
fout.variables['hmixa'].units = 'm'
fout.variables['hmixam'].units = 'm'
# Write axis data
fout.variables['time'][:] = fin.variables['time'][:].data.astype('float64')
fout.variables['zo'][:] = fin.variables['zo'][:].data.astype('float64') 
fout.variables['ypo'][:] = yl
fout.variables['xpo'][:] = xl
fout.variables['yto'][:] = ylt 
fout.variables['xto'][:] = xlt
fout.variables['za'][:] = fin.variables['za'][:].data.astype('float64')
fout.variables['ypa'][:] = fout.variables['ypo'][:].data
fout.variables['xpa'][:] = fout.variables['xpo'][:].data
fout.variables['yta'][:] = fout.variables['yto'][:].data
fout.variables['xta'][:] = fout.variables['xto'][:].data
fout.variables['pa'][...] = np.zeros((nz,nyl,nxl), dtype=np.float64)
fout.variables['pam'][...] = fout.variables['pa'][...].data
fout.variables['ast'][...] = np.zeros((nyl-1,nxl-1), dtype=np.float64)
fout.variables['astm'][...] = fout.variables['ast'][...].data
fout.variables['hmixa'][...] = fout.variables['ast'][...].data
fout.variables['hmixam'][...] = fout.variables['ast'][...].data
# Coarse-grain pressure
for k in range(nz):
    fout.variables['po'][k] = coarse_grain_P(fin.variables['po'][k].data.astype('float64'), ratio)
    fout.variables['pom'][k] = coarse_grain_P(fin.variables['pom'][k].data.astype('float64'), ratio)
# Coarse-grain SST
if with_sst:
    fout.variables['sst'][...] = coarse_grain_T(fin.variables['sst'][...].data.astype('float64'), 
            xt, yt, xlt, ylt, ratio)
    fout.variables['sstm'][...] = coarse_grain_T(fin.variables['sstm'][...].data.astype('float64'), 
            xt, yt, xlt, ylt, ratio)
else:
    fout.variables['sst'][...] = np.zeros((nyl-1,nxl-1), dtype=np.float64)
    fout.variables['sstm'][...] = fout.variables['sst'][...].data

if check_opt:
    ph = fin.variables['po'][0].data
    pl = fout.variables['po'][0].data
    # Snapshot of pressure
    plt.figure()
    plt.subplot(121)
    plt.imshow(ph, origin='lower', cmap='RdBu_r')
    plt.title('HR')
    plt.subplot(122)
    plt.imshow(pl, origin='lower', cmap='RdBu_r')
    plt.title('LR')
    plt.show()
    # KE spectrum
    k2h, k1h, eh = init_wvnb(nxh-1, nyh-1, dxh, dxh)
    k2l, k1l, el = init_wvnb(nxl-1, nyl-1, dxl, dxl)
    rdxhf0, rdxlf0 = 1.0e-3/(f0*dxh), 1.0e-3/(f0*dxl)
    uh = -rdxhf0*( ph[1:,:] - ph[:-1,:] )
    vh =  rdxhf0*( ph[:,1:] - ph[:,:-1] )
    ul = -rdxlf0*( pl[1:,:] - pl[:-1,:] )
    vl =  rdxlf0*( pl[:,1:] - pl[:,:-1] )
    fh = 0.5*( np.abs(np.fft.fft2(uh[:,:-1], axes=(-2,-1)))**2 
             + np.abs(np.fft.fft2(vh[:-1,:], axes=(-2,-1)))**2 )
    fl = 0.5*( np.abs(np.fft.fft2(ul[:,:-1], axes=(-2,-1)))**2 
             + np.abs(np.fft.fft2(vl[:-1,:], axes=(-2,-1)))**2 )
    plt.figure()
    plt.loglog(k1h, iso_spec(fh, k2h, k1h, eh), label=r'HR')
    plt.loglog(k1l, iso_spec(fl, k2l, k1l, el), label=r'LR')
    plt.grid(which='both', axis='both')
    plt.legend(loc='best')
    plt.xlabel(r'Isotropic wavenumbers (rad/m)') 
    plt.ylabel(r'KE spectral density') 
    plt.show()
    if with_sst:
        th = fin.variables['sst'][...].data
        tl = fout.variables['sst'][...].data
        # Snapshot of SST
        plt.figure()
        plt.subplot(121)
        plt.imshow(th, origin='lower', cmap='RdBu_r')
        plt.title('HR')
        plt.subplot(122)
        plt.imshow(tl, origin='lower', cmap='RdBu_r')
        plt.title('LR')
        plt.show()
        # SST spectrum
        fh = np.abs(np.fft.fft2(th, axes=(-2,-1)))**2 
        fl = np.abs(np.fft.fft2(tl, axes=(-2,-1)))**2 
        plt.figure()
        plt.loglog(k1h, iso_spec(fh, k2h, k1h, eh), label=r'HR')
        plt.loglog(k1l, iso_spec(fl, k2l, k1l, el), label=r'LR')
        plt.grid(which='both', axis='both')
        plt.legend(loc='best')
        plt.xlabel(r'Isotropic wavenumbers (rad/m)') 
        plt.ylabel(r'SST spectral density') 
        plt.show()

# Close files
fin.close()
fout.close()

print('Program terminates')
