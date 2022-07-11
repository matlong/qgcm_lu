#!/usr/bin/env python
"""Module of QG kernel functions"""

import numpy as np


def stretch_matrix(h, g):
    """Build stretching matrix `A` from layer thickness `h`and reduced grativity `g`"""

    nl = len(h)
    assert nl == len(g)+1, 'Wrong number of heights and reduced gravities'
    
    A = np.zeros((nl,nl), dtype='float64')
    A[0,0] =  1./(h[0]*g[0])
    A[0,1] = -1./(h[0]*g[0])
    for i in range(1, nl-1):
        A[i,i-1] = -1./(h[i]*g[i-1])
        A[i,i]   =  1./h[i]*(1./g[i] + 1./g[i-1])
        A[i,i+1] = -1./(h[i]*g[i])
    A[-1,-1] =  1./(h[nl-1]*g[nl-2])
    A[-1,-2] = -1./(h[nl-1]*g[nl-2])
    
    return A


def layers_modes_matrices(A):
    """Derive layers-to-modes matrix `Cl2m` and modes-to-layers matrix `Cm2l`""" 
    """from the stretching matrix `A`"""

    lambda_r, R = np.linalg.eig(A)
    lambda_l, L = np.linalg.eig(A.transpose())
    
    assert np.allclose(lambda_r, lambda_l), \
            'Left and right eigenvalues of A are not close'
    assert np.isclose(np.abs(lambda_r).min(), 0.), \
            'Lowest eigenvalue must be close to zero'
    eigval = lambda_r.real

    assert np.allclose(R.imag, 0.), \
            'Right eigenvectors of A are not real numbers'
    assert np.allclose(L.imag, 0.), \
            'Left eigenvectors of A are not real numbers'
    R, L = R.real, L.real

    assert np.allclose(np.matmul(L.transpose(),R) - \
           np.diag(np.diag(np.matmul(L.transpose(),R))), 0.), \
            'Left and right eigenvectors of A are not orthogonals'
    Cl2m = np.matmul(np.diag(1./np.diag(np.matmul(L.transpose(),R))), L.transpose())
    Cm2l = R

    assert np.allclose(np.matmul(Cl2m,Cm2l), np.eye(Cl2m.shape[0], dtype=np.float64)), \
            'Cl2m and Cm2l are not inverse of each other'

    return Cl2m, Cm2l, eigval


def jacobi(p, q, dxm2):
    """Compute Jacobi of `p` and `q`"""

    jpq = np.zeros_like(p, dtype=np.float64)
    jpq[...,1:-1,1:-1] = (dxm2/12.)*( \
            ( p[...,1:-1,2:] - p[...,1:-1,:-2] )*( q[...,2:,1:-1] - q[...,:-2,1:-1] ) \
          + ( p[...,:-2,1:-1] - p[...,2:,1:-1] )*( q[...,1:-1,2:] - q[...,1:-1,:-2] ) \
          + p[...,1:-1,2:] * ( q[...,2:,2:] - q[...,:-2,2:] ) \
          - p[...,1:-1,:-2] * ( q[...,2:,:-2] - q[...,:-2,:-2] ) \
          - p[...,2:,1:-1] * ( q[...,2:,2:] - q[...,2:,:-2] ) \
          + p[...,:-2,1:-1] * ( q[...,:-2,2:] - q[...,:-2,:-2] ) \
          + q[...,2:,1:-1] * ( p[...,2:,2:] - p[...,2:,:-2] ) \
          - q[...,:-2,1:-1] * ( p[...,:-2,2:] - p[...,:-2,:-2] ) \
          - q[...,1:-1,2:] * ( p[...,2:,2:] - p[...,:-2,2:] ) \
          + q[...,1:-1,:-2] * ( p[...,2:,:-2] - p[...,:-2,:-2] ) )

    return jpq


def laplac(p, dxm2, bc=False, fc=None):
    """Compute Laplacian of `p` with/without mixed boundary conditions"""
    
    d2p = np.zeros_like(p, dtype=np.float64)
    # Inner points
    d2p[...,1:-1,1:-1] = dxm2*( p[...,:-2,1:-1] + p[...,1:-1,:-2] + p[...,1:-1,2:] \
            + p[...,2:,1:-1] - 4.*p[...,1:-1,1:-1] )
 
    if bc:
        bcfc = fc*dxm2/( 0.5*fc + 1. )
        # N & S boundaries (including corners) 
        d2p[..., 0,:] = bcfc*( p[..., 1,:] - p[..., 0,:] )
        d2p[...,-1,:] = bcfc*( p[...,-2,:] - p[...,-1,:] )
        # W & E boundaries
        d2p[...,1:-1, 0] = bcfc*( p[...,1:-1, 1] - p[...,1:-1, 0] )
        d2p[...,1:-1,-1] = bcfc*( p[...,1:-1,-2] - p[...,1:-1,-1] )

    return d2p


def p_to_uv(p, df, pint=False, fc=None):
    """Derive geostrophic velocities `u` and `v` from `p``"""

    if pint:
        # Interpolated velocities on p-grid
        u = np.zeros_like(p, dtype=np.float64) # zonal
        v = np.zeros_like(p, dtype=np.float64) # meridional
        bcfc = 1./( df*(0.5*fc + 1.) )
        # Inner points
        u[...,1:-1,1:-1] = -( p[...,2:,1:-1] - p[...,:-2,1:-1] )/(2.*df)
        v[...,1:-1,1:-1] =  ( p[...,1:-1,2:] - p[...,1:-1,:-2] )/(2.*df) 
        # Southern boundary
        u[...,0,:]  = -bcfc*( p[...,1,:] - p[...,0,:] )
        # Northern boundary
        u[...,-1,:] = -bcfc*( p[...,-1,:] - p[...,-2,:] )
        # Western boundary
        v[...,0] = bcfc*( p[...,1] - p[...,0] )
        # Eastern boundary
        v[...,-1] = bcfc*( p[...,-1] - p[...,-2] )
    else:
        # Non-interpolated velocities (on stargered grid)
        u = -( p[...,1:,:] - p[...,:-1,:] )/df
        v =  ( p[...,1:] - p[...,:-1] )/df

    return u, v


def curl_wind(tau, rdxf0):
    """Compute Ekman pumping `wek` by curl of wind stress `tau`"""
    
    wekp = np.zeros_like(tau[...,0], dtype=np.float64) # p-grid 

    # Ekman velocity on T-grid
    wekt = 0.5*rdxf0*( (tau[:-1,1:,1] + tau[1,:1:,1]) \
                     - (tau[:-1,:-1,1] + tau[1:,:-1,1]) \
                     - (tau[1:,:-1,0] + tau[1:,1:,0]) \
                     + (tau[:-1,:-1,0] + tau[:-1,1:,0]) )

    # Inner p-grid points
    wekp[1:-1,1:-1] = 0.25*( wekt[:-1,:-1] + wekt[:-1,1:] + wekt[1:,:-1] + wekt[1:,1:] )
    # Western buondary
    wekp[1:-1,0] = 0.5*( wekt[:-1,0] + wekt[1:,0] )
    # Eastern buondary
    wekp[1:-1,-1] = 0.5*( wekt[:-1,-1] + wekt[1:,-1] )
    # Southern boundary
    wekp[0,1:-1] = 0.5*( wekt[0,:-1] + wekt[0,1:] )
    # Northern boundary
    wekp[-1,1:-1] = 0.5*( wekt[-1,:-1] + wekt[-1,1:] )
    # Corner points
    wekp[0,0] = wekt[0,0]
    wekp[0,-1] = wekt[0,-1]
    wekp[-1,0] = wekt[-1,0]
    wekp[-1,-1] = wekt[-1,-1]

    return wekp
