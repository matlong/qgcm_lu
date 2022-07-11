#!/usr/bin/env python
"""Module of QG diagnostic tools"""

import numpy as np
import scipy.io.netcdf as nc
from scipy.stats import skew
from scipy.stats import kurtosis


def init_outnc(fout, dims, rand=False):
    """Initialize output netcdf file"""
 
    # Create dimensions
    fout.createDimension('trun', dims['nt'])
    fout.createDimension('tdiag', dims['nt1'])
    fout.createDimension('z', dims['nz'])
    fout.createDimension('zi', dims['nz']-1)
    fout.createDimension('vmode', dims['nz'])
    fout.createDimension('bcmode', dims['nz']-1)
    fout.createDimension('yp', dims['ny'])
    fout.createDimension('xp', dims['nx'])
    fout.createDimension('k', dims['nk'])
    fout.createDimension('order', 4)

    # Create variables with attributes
    var = fout.createVariable('time_simu', 'f8', ('trun',))
    var.long_name = 'Time axis of simulation'
    var.units = 'years'
    
    var = fout.createVariable('time_diag', 'f8', ('tdiag',))
    var.long_name = 'Time axis of diagnostics'
    var.units = 'years'
    
    var = fout.createVariable('z', 'f8', ('z',))
    var.long_name = 'Ocean mid-layer axis'
    var.units = 'km'
    
    var = fout.createVariable('zi', 'f8', ('zi',))
    var.long_name = 'Ocean interface depth axis'
    var.units = 'km'
    
    var = fout.createVariable('vmode', 'i4', ('vmode',))
    var.long_name = 'Vertical modes index'
    
    var = fout.createVariable('bcmode', 'i4', ('bcmode',))
    var.long_name = 'Baroclinic modes index'
    
    var = fout.createVariable('yp', 'f8', ('yp',))
    var.long_name = 'Ocean Y axis (p-grid)'
    var.units = 'km'
    
    var = fout.createVariable('xp', 'f8', ('xp',))
    var.long_name = 'Ocean X axis (p-grid)'
    var.units = 'km'
    
    var = fout.createVariable('order', 'i4', ('order',))
    var.long_name = 'Statistics order'
    
    var = fout.createVariable('k', 'f8', ('k',))
    var.long_name = 'Ocean isotropic wavenumbers'
    var.units = 'rad/m'
    
    var = fout.createVariable('h', 'f8', ('z',)) 
    var.long_name = 'Ocean layer thickness'
    var.units = 'm'
    
    var = fout.createVariable('tsseke', 'f8', ('trun','z',))
    var.long_name = 'Time series of standing EKE'
    var.units = 'J/m^2'
    
    var = fout.createVariable('tsteke', 'f8', ('trun','z',))
    var.long_name = 'Time series of transient EKE'
    var.units = 'J/m^2'
    
    var = fout.createVariable('tmseke', 'f8', ('z','yp','xp',))
    var.long_name = 'Temporal mean of standing EKE'
    var.units = 'J/m^2'
    
    var = fout.createVariable('tmteke', 'f8', ('z','yp','xp',))
    var.long_name = 'Temporal mean of transient EKE'
    var.units = 'J/m^2'
    
    var = fout.createVariable('pstat', 'f8', ('order','z','yp','xp',))
    var.long_name = 'Temporal statistics of streamfunction'
    var.units = 'Sv (only for mean and std)'
    
    var = fout.createVariable('psdke', 'f8', ('z','k',)) 
    var.long_name = 'Power spectral density of kinetic energy'
    var.units = 'rad J/m^3'
    
    var = fout.createVariable('psdpe', 'f8', ('zi','k',)) 
    var.long_name = 'Power spectral density of potential energy'
    var.units = 'rad J/m^3'

    var = fout.createVariable('psdfke', 'f8', ('z','k',)) 
    var.long_name = 'Power spectral density of KE flux'
    var.units = 'rad W/m^3'
      
    var = fout.createVariable('psdfpe', 'f8', ('z','k',)) 
    var.long_name = 'Power spectral density of PE flux'
    var.units = 'rad W/m^3'
    
    var = fout.createVariable('psdby', 'f8', ('z','k',)) 
    var.long_name = 'Power spectral density of transfert from beta effect'
    var.units = 'rad W/m^3'
    
    var = fout.createVariable('psdwf', 'f8', ('k',)) 
    var.long_name = 'Power spectral density of transfert from wind forcing'
    var.units = 'rad W/m^3'
    
    var = fout.createVariable('psdek', 'f8', ('k',)) 
    var.long_name = 'Power spectral density of transfert from Ekman drag'
    var.units = 'rad W/m^3'
     
    var = fout.createVariable('psdbf', 'f8', ('zi','k',)) 
    var.long_name = 'Power spectral density of transfert from buoyancy forcing'
    var.units = 'rad W/m^3'
     
    var = fout.createVariable('psdvs', 'f8', ('z','k',)) 
    var.long_name = 'Power spectral density of transfert from viscosity'
    var.units = 'rad W/m^3'
    
    var = fout.createVariable('tsseke_vm', 'f8', ('trun','vmode',))
    var.long_name = 'Time series of standing EKE (in vertical modes)'
    var.units = 'J/m^3'
    
    var = fout.createVariable('tsteke_vm', 'f8', ('trun','vmode',))
    var.long_name = 'Time series of transient EKE (in vertical modes)'
    var.units = 'J/m^3'
     
    var = fout.createVariable('tmseke_vm', 'f8', ('vmode','yp','xp',))
    var.long_name = 'Temporal mean of standing EKE (in vertical modes)'
    var.units = 'J/m^3'
        
    var = fout.createVariable('tmteke_vm', 'f8', ('vmode','yp','xp',))
    var.long_name = 'Temporal mean of transient EKE (in vertical modes)'
    var.units = 'J/m^3'
    
    var = fout.createVariable('pstat_vm', 'f8', ('order','vmode','yp','xp',))
    var.long_name = 'Temporal statistics of streamfunction (in vertical modes)'
    var.units = 'Sv/m (only for mean and std)'
    
    var = fout.createVariable('psdke_vm', 'f8', ('vmode','k',)) 
    var.long_name = 'Power spectral density of kinetic energy (in vertical modes)'
    var.units = 'rad J/m^4'
    
    var = fout.createVariable('psdpe_vm', 'f8', ('bcmode','k',)) 
    var.long_name = 'Power spectral density of potential energy (in vertical modes)'
    var.units = 'rad J/m^4'

    if rand:
        var = fout.createVariable('psdadnoi', 'f8', ('z','k',)) 
        var.long_name = 'Power spectral density of transfert from noise advection'
        var.units = 'rad W/m^3'

        var = fout.createVariable('psdsonoi', 'f8', ('z','k',)) 
        var.long_name = 'Power spectral density of transfert from noise source'
        var.units = 'rad W/m^3'

        var = fout.createVariable('psddivar', 'f8', ('z','k',)) 
        var.long_name = 'Power spectral density of transfert from variance diffusion'
        var.units = 'rad W/m^3'
    
        var = fout.createVariable('psdsivar', 'f8', ('z','k',)) 
        var.long_name = 'Power spectral density of transfert from variance sink'
        var.units = 'rad W/m^3'

        var = fout.createVariable('psdadsta', 'f8', ('z','k',)) 
        var.long_name = 'Power spectral density of transfert from statistical-induced advection'
        var.units = 'rad W/m^3'

        var = fout.createVariable('psdsosta', 'f8', ('z','k',)) 
        var.long_name = 'Power spectral density of transfert from statistical-induced source'
        var.units = 'rad W/m^3'

    # Write data
    fout.variables['order'][:] = range(1,5)
    fout.variables['vmode'][:] = range(dims['nz'])[::-1]
    fout.variables['bcmode'][:] = range(1,dims['nz'])[::-1]
    
    return fout


def pstat(p):
    """Temporal statistics `ps` of `p`"""
     
    ps = np.zeros(((4,) + p.shape[1:]), dtype=np.float64)
    ps[0,...] = np.mean(p, axis=0, dtype=np.float64) # mean
    ps[1,...] = np.std(p, axis=0, dtype=np.float64, ddof=1) # std (unbias estimator)
    ps[2,...] = skew(p, axis=0, bias=False, nan_policy='propagate') # skewness
    ps[3,...] = kurtosis(p, axis=0, fisher=True, bias=True, nan_policy='propagate') # kurtosis
    ps[np.isnan(ps)] = 0. # impose 'nan' values to 0

    return ps


def area_int(u, facwe, facsn):
    "Area-integration of `u` with specific W-E and S-N boundary factors `facwe` and `facsn`"
    
    res = np.zeros_like(u[...,0,0], dtype=np.float64)

    # Inner points
    res += np.sum( np.sum(u[...,1:-1,1:-1], axis=-1, dtype=np.float64), \
                   axis=-1, dtype=np.float64 )
    
    # S-N boundary
    res += facsn*( np.sum(u[...,0,1:-1], axis=-1, dtype=np.float64) + \
                   np.sum(u[...,-1,1:-1], axis=-1, dtype=np.float64) ) 
    
    # W-E boundary
    res += facwe*( np.sum(u[...,1:-1,0], axis=-1, dtype=np.float64) + \
                   np.sum(u[...,1:-1,-1], axis=-1, dtype=np.float64) ) 
    
    # Corners
    res += facwe*facsn*( u[...,0,0].astype('float64') + u[...,-1,0].astype('float64') + \
                         u[...,0,-1].astype('float64') + u[...,-1,-1].astype('float64') )
    
    return res


def vert_prod(weight, u): 
    """Vertical product of `u` and `weight`"""

    u1 = u.reshape((u.shape[:len(u.shape)-2] + (-1,)))
    uw = np.matmul(weight, u1).reshape(u.shape)

    return uw


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
    #nmax = np.minimum(nx,ny)/2.
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
