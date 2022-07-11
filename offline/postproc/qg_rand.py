"""Contains subroutines for Model under Location Uncertainty (MLU)"""

import numpy as np

''' 
    Grid interpretation
    ===================

                    |   |   |
                  - P - V - P -
                    |   |   |
                  - U - T - U -
      y ^           |   |   |
        |         - P - V - P -
        0 -> x      |   |   |
    
    P-grid dimension: N_y * N_x 
    P-grid prognostic variables: p, q, axy
    
    U-grid dimension: N_y * (N_x-1) 
    U-grid prognostic variables: u, unoi, usta, axx
    
    V-grid dimension: (N_y-1) * N_x
    V-grid prognostic variables: v, vnoi, vsta, ayy
    
    T-grid dimension: (N_y-1) * (N_x-1) 
    T-grid prognostic variables: sst (not used yet)
'''

def sto_advection(q, du, dv, dx, dy):
    """Conservative form of PV advection by the noise or the correction drift"""
   
    dq = np.zeros_like(q, dtype=np.float64)
    
    # Advection of PV by noise plus correction drift
    '''
    $Adv = - \partial_x (dU q) - \partial_y(dV q)$
    '''
    # (A conservative form is used for adv. which corresponds to the 
    #  discrete Arakawa jacobian operator in diff_op/jacobi_h)
    # Outflux in x-axis
    fxp = (du[...,1:,1:-1] + du[...,1:,2:] + du[...,:-1,1:-1] + du[...,:-1,2:]) \
         *(q[...,1:-1,1:-1] + q[...,1:-1,2:]) + 0.5*( \
          (du[...,1:,1:-1] + du[...,1:,2:])*(q[...,1:-1,1:-1] + q[...,2:,2:]) \
         +(du[...,:-1,1:-1] + du[...,:-1,2:])*(q[...,1:-1,1:-1] + q[...,:-2,2:]) )
    # Influx in x-axis
    fxm = (du[...,1:,:-2] + du[...,:-1,:-2] + du[...,:-1,1:-1] + du[...,1:,1:-1]) \
         *(q[...,1:-1,1:-1] + q[...,1:-1,:-2]) + 0.5*( \
         (du[...,1:,1:-1] + du[...,1:,:-2])*(q[...,1:-1,1:-1] + q[...,2:,:-2]) \
         +(du[...,:-1,:-2] + du[...,:-1,1:-1])*(q[...,1:-1,1:-1] + q[...,:-2,:-2]) )
    # Outflux in y-axis
    fyp = (dv[...,2:,:-1] + dv[...,1:-1,:-1] + dv[...,1:-1,1:] + dv[...,2:,1:]) \
         *(q[...,1:-1,1:-1] + q[...,2:,1:-1]) + 0.5*( \
          (dv[...,1:-1,1:] + dv[...,2:,1:])*(q[...,1:-1,1:-1] + q[...,2:,2:]) \
         +(dv[...,2:,:-1] + dv[...,1:-1,:-1])*(q[...,1:-1,1:-1] + q[...,2:,:-2]) )
    # Influx in y-axis
    fym = (dv[...,1:-1,:-1] + dv[...,:-2,:-1] + dv[...,:-2,1:] + dv[...,1:-1,1:]) \
         *(q[...,1:-1,1:-1] + q[...,:-2,1:-1]) + 0.5*( \
         (dv[...,1:-1,1:] + dv[...,:-2,1:])*(q[...,1:-1,1:-1] + q[...,:-2,2:]) \
         +(dv[...,1:-1,:-1] + dv[...,:-2,:-1])*(q[...,1:-1,1:-1] + q[...,:-2,:-2]) )

    dq[...,1:-1,1:-1] = - (fxp - fxm)/(12.*dx) - (fyp - fym)/(12.*dy)

    return dq

def sto_source(p, du, dv, dx, dy):
    """Conservative form of PV source by the noise or the correction drift"""
    """(Note that 'p' is streamfunction [m^2/s] not pressure [m^2/s^2])"""
    
    dq = np.zeros_like(p, dtype=np.float64)

    # Geostrophic velocities from streamfunction
    u = -(p[...,1:,:] - p[...,:-1,:])/dy # zonal
    v =  (p[...,1:] - p[...,:-1])/dx # meridional
    
    # Sources of (relative) vorticity by rotation of noise and correction drift
    '''
    $Src = \partial_x (u \partial_y dU - v \partial_x dU) 
         + \partial_y (u \partial_y dV - v \partial_x dV)$
    '''
    # (A simple centered flux form is used)    
    # Outflux in x-axis
    fxp = ( (du[...,1:,1:-1] - du[...,:-1,1:-1])*(u[...,1:,1:-1] + u[...,:-1,1:-1]) \
           +(du[...,1:,2:] - du[...,:-1,2:])*(u[...,:-1,2:] + u[...,1:,2:]) )/(2.*dy) \
         -(du[...,1:,2:] - du[...,1:,1:-1] + du[...,:-1,2:] - du[...,:-1,1:-1]) \
         *(v[...,1:-1,1:]/dx)
    # Influx in x-axis
    fxm = ( (du[...,1:,:-2] - du[...,:-1,:-2])*(u[...,1:,:-2] + u[...,:-1,:-2]) \
           +(du[...,1:,1:-1] - du[...,:-1,1:-1])*(u[...,:-1,1:-1] + u[...,1:,1:-1]) )/(2.*dy) \
         -(du[...,:-1,1:-1] - du[...,:-1,:-2] + du[...,1:,1:-1] - du[...,1:,:-2]) \
         *(v[...,1:-1,:-1]/dx)
    # Outflux in y-axis
    fyp =-( (dv[...,2:,1:] - dv[...,2:,:-1])*(v[...,2:,1:] + v[...,2:,:-1]) \
           +(dv[...,1:-1,1:] - dv[...,1:-1,:-1])*(v[...,1:-1,:-1] + v[...,1:-1,1:]) )/(2.*dx) \
         +(dv[...,2:,:-1] - dv[...,1:-1,:-1] + dv[...,2:,1:] - dv[...,1:-1,1:]) \
         *(u[...,1:,1:-1]/dy)
    # Influx in y-axis
    fym =-( (dv[...,1:-1,1:] - dv[...,1:-1,:-1])*(v[...,1:-1,1:] + v[...,1:-1,:-1])\
           +(dv[...,:-2,1:] - dv[...,:-2,:-1])*(v[...,:-2,:-1] +  v[...,:-2,1:]) )/(2.*dx) \
         +(dv[...,1:-1,:-1] - dv[...,:-2,:-1] + dv[...,1:-1,1:] - dv[...,:-2,1:]) \
         *(u[...,:-1,1:-1]/dy)

    dq[...,1:-1,1:-1] = (fxp - fxm)/(2.*dx) + (fyp - fym)/(2.*dy)

    return dq 

def sto_diffusion(q, axx, ayy, axy, dx, dy):
    """Compute PV diffusion due to the variance tensor of noise"""

    dq = np.zeros_like(q, dtype=np.float64)
    
    # Diffusion of PV by variance tensor of noise
    '''
    $Dif = 1/2 \partial_x (a_{xx} \partial_x q + a_{xy} \partial_y q) +
           1/2 \partial_y (a_{xy} \partial_x q + a_{yy} \partial_y q)$
    '''
    # (A simple centered flux form is used)
    # Outflux in x-axis
    fxp = (axx[...,1:,1:-1] + axx[...,:-1,1:-1] + axx[...,:-1,2:] + axx[...,1:,2:]) \
         *(q[...,1:-1,2:] - q[...,1:-1,1:-1])/dx \
         + axy[...,1:-1,1:-1]*(q[...,2:,1:-1] - q[...,:-2,1:-1])/dy \
         + axy[...,1:-1,2:]*(q[...,2:,2:] - q[...,:-2,2:])/dy
    # Influx in x-axis
    fxm = (axx[...,1:,:-2] + axx[...,:-1,:-2] + axx[...,:-1,1:-1] + axx[...,1:,1:-1]) \
         *(q[...,1:-1,1:-1] - q[...,1:-1,:-2])/dx \
         + axy[...,1:-1,:-2]*(q[...,2:,:-2] - q[...,:-2,:-2])/dy \
         + axy[...,1:-1,1:-1]*(q[...,2:,1:-1] - q[...,:-2,1:-1])/dy
    # Outflux in y-axis
    fyp = (ayy[...,2:,1:] + ayy[...,1:-1,:-1] + ayy[...,1:-1,1:] + ayy[...,2:,:-1]) \
         *(q[...,2:,1:-1] - q[...,1:-1,1:-1])/dy \
         + axy[...,2:,1:-1]*(q[...,2:,2:] - q[...,2:,:-2])/dx \
         + axy[...,1:-1,1:-1]*(q[...,1:-1,2:] - q[...,1:-1,:-2])/dx  
    # Influx in y-axis
    fym = (ayy[...,1:-1,:-1] + ayy[...,:-2,:-1] + ayy[...,:-2,1:] + ayy[...,1:-1,1:]) \
         *(q[...,1:-1,1:-1] - q[...,:-2,1:-1])/dy \
         + axy[...,:-2,1:-1]*(q[...,:-2,2:] - q[...,:-2,:-2])/dx \
         + axy[...,1:-1,1:-1]*(q[...,1:-1,2:] - q[...,1:-1,:-2])/dx 

    dq[...,1:-1,1:-1] = (fxp - fxm)/(8.*dx) + (fyp - fym)/(8.*dy)
    
    return dq 

def sto_sink(p, axx, ayy, axy, dx, dy, bcco):
    """Compute PV sink due to the variance tensor of noise"""
    """(Note that 'p' is streamfunction [m^2/s] not pressure [m^2/s^2])"""

    dq = np.zeros_like(p, dtype=np.float64)
    bc = bcco/(1.0 + 0.5*bcco)

    # Hessian of 'p' 
    # Diagonal compoents defined on P-grid
    pxx = np.zeros_like(p, dtype=np.float64) 
    pxx[...,1:-1,1:-1] = (p[...,1:-1,2:] + p[...,1:-1,:-2] - 2.*p[...,1:-1,1:-1])/(dx**2)
    pxx[...,1:-1,0]    = bc*(p[...,1:-1,1] - p[...,1:-1,0])/(dx**2)
    pxx[...,1:-1,-1]   = bc*(p[...,1:-1,-2] - p[...,1:-1,-1])/(dx**2) 
    pyy = np.zeros_like(p, dtype=np.float64)
    pyy[...,1:-1,1:-1] = (p[...,2:,1:-1] + p[...,:-2,1:-1] - 2.*p[...,1:-1,1:-1])/(dy**2)
    pyy[...,0,1:-1]    = bc*(p[...,1,1:-1] - p[...,0,1:-1])/(dy**2)
    pyy[...,-1,1:-1]   = bc*(p[...,-2,1:-1] - p[...,-1,1:-1])/(dy**2)
    # Non-diagonal component defined on T-grid
    pxy = (p[...,1:,1:] - p[...,1:,:-1] - p[...,:-1,1:] + p[...,:-1,:-1])/(dx*dy)
    
    # Sinks of (relative) vorticity by rotation of variance tensor
    '''
    $Sink = 1/2 \partial_x \Big( \partial_x a_{xx} \partial_{xx}^2 p +
                                (\partial_y a_{xx} + \partial_x a_{xy}) \partial_{xy}^2 p
                                +\partial_y a_{xy} \partial_{yy}^2 p \Big)
          + 1/2 \partial_y \Big( \partial_x a_{xy} \partial_{xx}^2 p +
                                (\partial_y a_{xy} + \partial_x a_{yy}) \partial_{xy}^2 p
                                +\partial_y a_{yy} \partial_{yy}^2 p \Big)$
    '''
    # (A simple centered flux form is used)
    # Outflux in x-axis
    fxp = (axx[...,1:,2:] - axx[...,1:,1:-1] + axx[...,:-1,2:] - axx[...,:-1,1:-1]) \
         *(pxx[...,1:-1,1:-1] + pxx[...,1:-1,2:])/dx \
        + (axy[...,2:,1:-1] - axy[...,:-2,1:-1])*pyy[...,1:-1,1:-1]/dy \
        + (axy[...,2:,2:] - axy[...,:-2,2:])*pyy[...,1:-1,2:]/dy \
        + ( (axx[...,1:,1:-1] - axx[...,:-1,1:-1] + axx[...,1:,2:] - axx[...,:-1,2:])/dy \
        + 2.*(axy[...,1:-1,2:] - axy[...,1:-1,1:-1])/dx )*(pxy[...,1:,1:] + pxy[...,:-1,1:])
    # Influx in x-axis
    fxm = (axx[...,1:,1:-1] - axx[...,1:,:-2] + axx[...,:-1,1:-1] - axx[...,:-1,:-2]) \
         *(pxx[...,1:-1,1:-1] + pxx[...,1:-1,:-2])/dx \
        + (axy[...,2:,1:-1] - axy[...,:-2,1:-1])*pyy[...,1:-1,1:-1]/dy \
        + (axy[...,2:,:-2] - axy[...,:-2,:-2])*pyy[...,1:-1,:-2]/dy \
        + ( (axx[...,1:,1:-1] - axx[...,:-1,1:-1] + axx[...,1:,:-2] - axx[...,:-1,:-2])/dy \
        + 2.*(axy[...,1:-1,1:-1] - axy[...,1:-1,:-2])/dx )*(pxy[...,:-1,:-1] + pxy[...,1:,:-1])
    # Outflux in y-axis
    fyp = (ayy[...,2:,:-1] - ayy[...,1:-1,:-1] + ayy[...,2:,1:] - ayy[...,1:-1,1:]) \
         *(pyy[...,1:-1,1:-1] + pyy[...,2:,1:-1])/dy \
        + (axy[...,1:-1,2:] - axy[...,1:-1,:-2])*pxx[...,1:-1,1:-1]/dx \
        + (axy[...,2:,2:] - axy[...,2:,:-2])*pxx[...,2:,1:-1]/dx \
        + ( (ayy[...,2:,1:] - ayy[...,2:,:-1] + ayy[...,1:-1,1:] - ayy[...,1:-1,:-1])/dx \
        + 2.*(axy[...,2:,1:-1] - axy[...,1:-1,1:-1])/dy )*(pxy[...,1:,:-1] + pxy[...,1:,1:])
    # Influx in y-axis
    fym = (ayy[...,1:-1,:-1] - ayy[...,:-2,:-1] + ayy[...,1:-1,1:] - ayy[...,:-2,1:]) \
         *(pyy[...,1:-1,1:-1] + pyy[...,:-2,1:-1])/dy \
        + (axy[...,1:-1,2:] - axy[...,1:-1,:-2])*pxx[...,1:-1,1:-1]/dx \
        + (axy[...,:-2,2:] - axy[...,:-2,:-2])*pxx[...,:-2,1:-1]/dx \
        + ( (ayy[...,1:-1,1:] - ayy[...,1:-1,:-1] + ayy[...,:-2,1:] - ayy[...,:-2,:-1])/dx \
        + 2.*(axy[...,1:-1,1:-1] - axy[...,:-2,1:-1])/dy )*(pxy[...,:-1,:-1] + pxy[...,:-1,1:])
    
    dq[...,1:-1,1:-1] = (fxp - fxm)/(8.*dx) + (fyp - fym)/(8.*dy)

    return dq 

def init_eof_data(param, nc_file):
    """Intitialize (stationary) EOFs and variance tensor for noise
       and derive statistical-induced drift."""

    nm = param['n_modes']
    dx = param['dx']
    dy = param['dy']

    # Read and truncate pattern correlations 
    # (eigenvalues of covariance)
    pc = nc_file.variables['lambda'][:,:nm].copy() # (m^2/s^2)
    pc = param['tco']*pc.T # (m^2/s)

    # Read and truncate EOFs
    wrk4 = nc_file.variables['umode'][:nm,:,:,:].copy().transpose((0,1,3,2))
    ueof = wrk4*np.sqrt(pc[:,:,np.newaxis,np.newaxis]) # (m/s^{1/2})
    wrk4 = nc_file.variables['vmode'][:nm,:,:,:].copy().transpose((0,1,3,2))
    veof = wrk4*np.sqrt(pc[...,np.newaxis,np.newaxis])
    del wrk4

    # Build stationary variance as diffusion tensor (m^2/s)
    '''
    $a_{xx} = \sum_{n=1}^{N_m} u_{eof}^n u_{eof}^n$
    $a_{yy} = \sum_{n=1}^{N_m} v_{eof}^n v_{eof}^n$
    $a_{xy} = \sum_{n=1}^{N_m} u_{eof}^n v_{eof}^n$
    '''
    # Diagonal components
    axx = np.sum(ueof*ueof, axis=0, dtype='float64') # defined on U-grid
    ayy = np.sum(veof*veof, axis=0, dtype='float64') # defined on V-grid
    # Non-diagonal component (interpolated on P-grid)
    axy = np.zeros((param['nl'],param['nx'],param['ny']), dtype='float64')
    axy[:,1:-1,1:-1] = 0.25*np.sum( (ueof[...,1:-1,1:] + ueof[...,1:-1,:-1]) * \
                                    (veof[...,1:,1:-1] + veof[...,:-1,1:-1]), \
                                    axis=0, dtype='float64' )

    # Derive statistical-induced drift
    '''
    $u_s = \overline{u} + 1/2 (\partial_x a_{xx} + \partial_y a_{xy})$
    $v_s = \overline{v} + 1/2 (\partial_x a_{xy} + \partial_y a_{yy})$
    '''
    # Mean of eddies effect (derived from changement of probability measure)
    usta = nc_file.variables['umean'][:,:,:].copy().transpose((0,2,1)) # Zonal (m/s)
    vsta = nc_file.variables['vmean'][:,:,:].copy().transpose((0,2,1)) # Meridional (m/s)
    # Add Ito-Stokes drift (derived from stochastic transport)
    usta[:,1:-1,:] += (axx[...,2:,:] - axx[...,:-2,:])/(4.*dx) \
                    + (axy[...,1:-1,1:] - axy[...,1:-1,:-1])/(2.*dy)                  
    wrk3 = (axy[...,1:,1:-1] - axy[...,:-1,1:-1])/(2.*dx) \
         + (ayy[...,2:] - ayy[...,:-2])/(4.*dy)
    vsta[...,1:-1] += wrk3
    
    # Derive additive beta-forcing of PV due to Ito-Stokes drift
    bfvs = -0.5*param['beta']*(wrk3[...,:-1,:] + wrk3[...,1:,:]) 
    # (return only inner points without boundaries)

    return ueof, veof, axx, ayy, axy, usta, vsta, bfvs

def draw_eof_noise(param, ueof, veof):
    """Draw noise by Karhunen-Loeve decomposition"""

    nm = param['n_modes']
    ne = param['n_ens']
    
    '''
    $(\sigma dB_t)_x = \sum_{n=1}^{N_m} u_{eof}^n dW_t^n$
    $(\sigma dB_t)_y = \sum_{n=1}^{N_m} v_{eof}^n dW_t^n$
    where $W_t^n$ are i.i.d. stanard Wiener processes.
    '''
    xi = np.sqrt(param['dt'])*np.random.randn(ne,nm) # Standard Wiener increments
    unoi = np.matmul(xi, ueof.reshape((nm,-1))).reshape((ne,)+ueof.shape[1:])
    vnoi = np.matmul(xi, veof.reshape((nm,-1))).reshape((ne,)+veof.shape[1:])

    return unoi, vnoi

#TODO: permute dimensions
def sto_advection_v0(param, q, p, du, dv):
    """Non-conservative form of stochastic advections and sources."""
    """(Note that 'p' is streamfunction [m^2/s] not pressure [m^2/s^2])"""
    """(Return only inner points without boundary values)"""

    dx = param['dx']
    dy = param['dy']

    # Advection of PV
    '''
    $Adv = - dU \partial_x q - dV \partial_y q$
    '''
    adv = -(du[...,1:-1,1:] + du[...,1:-1,:-1])*(q[...,2:,1:-1] - q[...,:-2,1:-1])/dx \
          -(dv[...,1:,1:-1] + dv[...,:-1,1:-1])*(q[...,1:-1,2:] - q[...,1:-1,:-2])/dy
    
    # Geostrophic velocities from streamfunction
    u = -(p[...,1:] - p[...,:-1])/dy # zonal
    v =  (p[...,1:,:] - p[...,:-1,:])/dx # meridional
    
    # Sources of (relative) vorticity by rotation of noise 
    '''
    $Src = \partial_x u \partial_y dU - \partial_x v \partial_x dU
         + \partial_y u \partial_y dV - \partial_y v \partial_x dV$
    '''
    src = (u[...,2:,1:] - u[...,:-2,1:] + u[...,2:,:-1] - u[...,:-2,:-1]) \
         *(du[...,1:-1,1:] - du[...,1:-1,:-1])/(dx*dy) \
         -(du[...,2:,1:] - du[...,:-2,1:] + du[...,2:,:-1] + du[...,:-2,:-1]) \
         *(v[...,1:,1:-1] - v[...,:-1,1:-1])/(dx*dx) \
         +(dv[...,:-1,2:] - dv[...,:-1,:-2] + dv[...,1:,2:] - dv[...,1:,:-2]) \
         *(u[...,1:-1,1:] - u[...,1:-1,:-1])/(dy*dy) \
         -(v[...,:-1,2:] - v[...,:-1,:-2] + v[...,1:,2:] - v[...,1:,:-2]) \
         *(dv[...,1:,1:-1] - dv[...,:-1,1:-1])/(dx*dy)
    
    return 0.25*(adv + src)
