#!/usr/bin/env python3

# Load modules
# ------------
import numpy as np
import scipy.io.netcdf as nc
import scipy.io as sio
from scipy import signal
from qg_kernel import curl_wind, p_to_uv, jacobi, laplac, stretch_matrix, layers_modes_matrices
from qg_diag import init_wvnb, init_outnc, pstat, area_int, iso_spec, vert_prod
from qg_rand import sto_advection, sto_diffusion, sto_source, sto_sink

# Set param.
stomod = False
stationary_mode = False
with_sst = False
#eofile = '/Volumes/Long/q-gcm/double_gyre_ocean_only/REF5/EOF/oceof80_cut.nc'
base_dir = '/media/long/Long/q-gcm/double_gyre_coupled/DET80/'
subs_dir = ['yrs000-180/']
outfile = base_dir + subs_dir[-1] + 'ocdiag.nc'
tstat = [90.0,180.0] # starting and ending time (years) for stats.
freq_cut = 0.5 # per 2 years

param = {
        'fnot': 9.37456e-5, # Coriolis parameter (s^-1)
        'beta': 1.7536e-11, # gradient of Coriolis (m^-1 s^-1)
        'bcco': 0.2, # boundary coef. (nondim.)
        'rhooc': 1.0e3, # background density of water (kg/m^3)
        'hoc': [350.0,750.0,2900.0], # background layer thickness (m)
        'gpoc': [0.025, 0.0125], # reduced gravity numbers (m/s^2)
        'rhooc': 1.0e3, # density of sea (kg m^3)
        'delek': 2.0, # Ekman layer thickness (m)
        'tau0': 2.0e-5, # wind stress magnitude
        'ah4oc': 5.0e12, # bi-laplacian diffusion coef. (m^4/s)
        }

#print 'Read spatial axis'
fin = nc.netcdf_file(base_dir+subs_dir[0]+'ocpo.nc', 'r')
x = fin.variables['xp'][:].copy() # (km)
y = fin.variables['yp'][:].copy() # (km)
z = fin.variables['z'][:].copy() # (km)
zi = fin.variables['zi'][:].copy() # (km)
fin.close()
x = x.astype('float64') 
y = y.astype('float64') 
z = z.astype('float64') 
zi = zi.astype('float64') 
dx = 1.0e3*(x[1] - x[0]) # (m)

# Create wavenumbers
dims = dict()
dims['nx'] = len(x) # p-grid size
dims['ny'] = len(y)
dims['nz'] = len(z)
wv, kr, efac = init_wvnb(dims['nx']-1, dims['ny']-1, dx, dx)
dims['nk'] = len(kr)

#print 'Read temporal axis'
n = 1
t = np.empty(0) # (years)
for s in subs_dir:
    fin = nc.netcdf_file(base_dir+s+'ocpo.nc', 'r')
    nn = np.minimum(n,2) - 1
    tmp = fin.variables['time'][nn:].copy()
    t = np.append(t, tmp.astype('float64'), axis=0) 
    del tmp
    n += 1
    fin.close() 
dims['nt'] = len(t)

# Find time indices for statistics 
id0 = np.abs(t - tstat[0]).argmin()
id1 = np.abs(t - tstat[1]).argmin()
t1 = t[id0:id1]
dims['nt1'] = len(t1)

# Define low-pass filter 
freq_samp = 1./(t[1] - t[0]) 
wid = freq_cut / (freq_samp/2)
b, a = signal.butter(5, wid, 'low') # 5th order Butterworth

#print 'Build vertical matrices'
A = stretch_matrix(param['hoc'], param['gpoc'])
Cl2m, Cm2l, eigval = layers_modes_matrices(A) 
# Baroclinic deformation radiis
Ld = 1./(np.sqrt(eigval[:-1])*param['fnot']) # barotropic mode (infinite) not include
#print 'Baroclinic deformation radiis (km) = ',1e-3*Ld

#print 'Create forcings'
tau = np.zeros((dims['ny'],dims['nx'],2), dtype=np.float64) # Wind stress
tau[:,:,0] = -param['tau0']*np.cos(2.*np.pi*(np.arange(dims['ny'])+0.5)
        /dims['ny']).reshape((dims['ny'],1))
wek = curl_wind(tau, 1./dx/param['fnot']) # Ekman velocity
betay = 1.e3*param['beta']*np.tile(y - 0.5*y[-1], (dims['nx'],1)).transpose()

#print 'Create output file: ', outfile
fout = nc.netcdf_file(outfile, 'w')
fout = init_outnc(fout, dims, stomod)
fout.variables['time_simu'][:] = t
fout.variables['time_diag'][:] = t1
fout.variables['z'][:] = z
fout.variables['zi'][:] = zi
fout.variables['yp'][:] = y
fout.variables['xp'][:] = x
fout.variables['k'][:] = kr
fout.variables['h'][:] = param['hoc']

#print 'Collect p-data'
n = 1
p = np.empty((0,dims['nz'],dims['ny'],dims['nx']), dtype=np.float64)
for s in subs_dir:
    nn = np.minimum(n,2) - 1
    file1 = base_dir + s + 'ocpo.nc'
    #print 'Opening file: ', file1
    fin = nc.netcdf_file(file1,'r')
    tmp = fin.variables['p'][nn:,:,:,:].copy()
    p = np.append(p, tmp.astype('float64'), axis=0)
    fin.close() 
    del tmp
    n += 1
p /= param['fnot'] # from pressure (m^2/s^2) to streamfunction (m^2/s)  

#print 'Filter mean and derive eddy'
pm = signal.filtfilt(b, a, p, axis=0) # mean
p -= pm # eddy

#print 'Compute time series of EKEs'
afac = 0.5/(dims['nx']-1)/(dims['ny']-1) # area factor for integration
rhoh = param['rhooc']*np.diag(param['hoc']) # (kg/m^2)
# Standing EKE
u, v = p_to_uv(pm, dx) # mean geostrophic velocities
fout.variables['tsseke'][:,:] = afac*np.matmul(area_int(u**2, 0.5, 1.0) + 
        area_int(v**2, 1.0, 0.5), rhoh) # (J/m^2)

# Transient EKE
u, v = p_to_uv(p, dx) # eddy geostrophic velocities
fout.variables['tsteke'][:,:] = afac*np.matmul(area_int(u**2, 0.5, 1.0) 
        + area_int(v**2, 1.0, 0.5), rhoh)

# Standing modal EKE
rhoa = param['rhooc']*afac # (kg/m^3)
pm1 = vert_prod(Cl2m[np.newaxis], pm) # Convert layers to modes
u, v = p_to_uv(pm1, dx) 
fout.variables['tsseke_vm'][:,:] = rhoa*(area_int(u**2, 0.5, 1.0) + area_int(v**2, 1.0, 0.5)) # (J/m^3)

# Transient modal EKE
p1 = vert_prod(Cl2m[np.newaxis], p) # Convert layers to modes
u, v = p_to_uv(p1, dx) 
fout.variables['tsteke_vm'][:,:] = rhoa*(area_int(u**2, 0.5, 1.0) + area_int(v**2, 1.0, 0.5)) 
del u, v

# Reduce temporal size
pm = pm[id0:id1,...] 
p = p[id0:id1,...]
pm1 = pm1[id0:id1,...] 
p1 = p1[id0:id1,...]

#print 'Compute temporal mean of EKEs' 
# Standing EKE
u, v = p_to_uv(pm, dx, pint=True, fc=param['bcco']) # on p-grid
fout.variables['tmseke'][:,:,:] = 0.5*vert_prod(rhoh, np.mean(u**2 + v**2, axis=0, dtype=np.float64))

# Transient EKE 
u, v = p_to_uv(p, dx, pint=True, fc=param['bcco']) 
p += pm # recover total
del pm
fout.variables['tmteke'][:,:,:] = 0.5*vert_prod(rhoh, np.mean(u**2 + v**2, axis=0, dtype=np.float64))

# Standing modal EKE
u, v = p_to_uv(pm1, dx, pint=True, fc=param['bcco'])
fout.variables['tmseke_vm'][:,:,:] = 0.5*param['rhooc']*np.mean(u**2 + v**2, axis=0, dtype=np.float64)

# Transient modal EKE 
u, v = p_to_uv(p1, dx, pint=True, fc=param['bcco'])
p1 += pm1
del pm1
fout.variables['tmteke_vm'][:,:,:] = 0.5*param['rhooc']*np.mean(u**2 + v**2, axis=0, dtype=np.float64)
del u, v

#print 'Compute statistics of streamfunction' # (1 Sv = 1e6 m^3/s)
fout.variables['pstat'][:,:,:,:] = 1.e-6*vert_prod(np.diag(param['hoc'])[np.newaxis], pstat(p)) 
fout.variables['pstat_vm'][:,:,:,:] = 1.e-6*pstat(p1)

#print 'Compute KE spectrum'
u, v = p_to_uv(p, dx)
fh = 0.5*np.mean( np.abs( np.fft.fft2(u[...,:-1], axes=(-2,-1)) )**2, \
                  axis=0, dtype=np.float64 )
fh += 0.5*np.mean( np.abs( np.fft.fft2(v[...,:-1,:], axes=(-2,-1)) )**2, \
                  axis=0, dtype=np.float64 ) 
fout.variables['psdke'][:,:] = iso_spec(vert_prod(rhoh, fh), wv, kr, efac)
del u, v

#print 'Compute PE spectrum'
rf2g = param['rhooc']*param['fnot']**2*np.diag(
        np.ones_like(param['gpoc'], dtype=np.float64)/param['gpoc']) # (kg/m^4)
fh = np.fft.fft2(p[...,1:,:-1,:-1] - p[...,:-1,:-1,:-1], axes=(-2,-1))
fh = 0.5*np.mean( np.abs(fh)**2, axis=0, dtype=np.float64 ) # (m^4/s^2)
fout.variables['psdpe'][:,:] = iso_spec(vert_prod(rf2g, fh), wv, kr, efac) 

#print 'Compute modal KE spectrum'
u, v = p_to_uv(p1, dx) 
fh = 0.5*np.mean( np.abs( np.fft.fft2(u[...,:-1], axes=(-2,-1)) )**2, \
                  axis=0, dtype=np.float64 )
fh += 0.5*np.mean( np.abs( np.fft.fft2(v[...,:-1,:], axes=(-2,-1)) )**2, \
                  axis=0, dtype=np.float64 ) 
fout.variables['psdke_vm'][:,:] = iso_spec(0.5*param['rhooc']*fh, wv, kr, efac)
del u, v

#print 'Compute modal PE spectrum'
rf2e = param['rhooc']*param['fnot']**2*np.diag(eigval[:-1]) # (kg/m^5)
fh = 0.5*np.mean( np.abs(np.fft.fft2(p1[...,:-1,:-1,:-1]))**2, \
                  axis=0, dtype=np.float64 )
fout.variables['psdpe_vm'][:,:] = iso_spec(vert_prod(rf2e, fh), wv, kr, efac)
del p1

#print 'Compute spectral flux of KE'
dxm2 = 1./(dx**2)
phc = np.conj( np.fft.fft2(p[...,:-1,:-1], axes=(-2,-1)) )
#d2p = laplac(p, dxm2, bc=True, fc=param['bcco'])
d2p = laplac(p, dxm2)
adv = jacobi(p, d2p, dxm2)
fh = np.mean( (phc * np.fft.fft2(adv[...,:-1,:-1], axes=(-2,-1))).real, \
              axis=0, dtype=np.float64 )
fout.variables['psdfke'][:,:] = iso_spec(vert_prod(rhoh, fh), wv, kr, efac)

#print 'Compute spectral flux of PE'
Sp = -param['fnot']**2 * vert_prod(A[np.newaxis], p)
adv = jacobi(p, Sp, dxm2)
if stomod:
    q = d2p + Sp + np.tile(betay, (dims['nt1'],dims['nz'],1,1)) 
del Sp
fh = np.mean( (phc * np.fft.fft2(adv[...,:-1,:-1], axes=(-2,-1))).real, \
              axis=0, dtype=np.float64 )
fout.variables['psdfpe'][:,:] = iso_spec(vert_prod(rhoh, fh), wv, kr, efac)

#print 'Compute spectral budget of beta effect'
adv = jacobi(p, np.tile(betay, (dims['nt1'],dims['nz'],1,1)), dxm2) 
fh = np.mean( (phc * np.fft.fft2(adv[...,:-1,:-1], axes=(-2,-1))).real, \
              axis=0, dtype=np.float64 )
del adv
fout.variables['psdby'][:,:] = iso_spec(vert_prod(rhoh, fh), wv, kr, efac)

#print 'Compute spectral budget of wind forcing'
frho = param['fnot']*param['rhooc']
fh = - (np.mean(phc[:,0,:,:], axis=0)*np.fft.fft2(wek[:-1,:-1], axes=(-2,-1))).real
fout.variables['psdwf'][:] = iso_spec(frho*fh, wv, kr, efac)

#print 'Compute spectral budget of bottom drag'
fh = np.mean( (phc[:,-1,:,:] * np.fft.fft2(d2p[...,-1,:-1,:-1], axes=(-2,-1))).real,\
              axis=0, dtype=np.float64 )
fout.variables['psdek'][:] = iso_spec(0.5*frho*param['delek']*fh, wv, kr, efac)

#print 'Compute spectral budget of dissipation'
#d6p = laplac( laplac(d2p, dxm2, bc=True, fc=param['bcco']), dxm2 )
d6p = laplac( laplac(d2p, dxm2), dxm2 )
del d2p
fh = np.mean( (phc * np.fft.fft2(d6p[...,:-1,:-1], axes=(-2,-1))).real, \
              axis=0, dtype=np.float64 )
del d6p
fout.variables['psdvs'][:,:] = iso_spec(param['ah4oc']*vert_prod(rhoh, fh), wv, kr, efac)

if with_sst:

    #print 'Collect e1-data'
    e1 = np.empty((0,dims['ny'],dims['nx']), dtype=np.float64)
    n = 1
    for s in subs_dir:
        nn = np.minimum(n,2) - 1
        file1 = base_dir + s + 'qocdiag.nc'
        #print 'Opening file: ', file1
        fin = nc.netcdf_file(file1,'r')
        tmp = fin.variables['qotent'][nn:,1,:,:].copy()
        e1 = np.append(e1, tmp.astype('float64'), axis=0)
        fin.close() 
        del tmp
        n += 1
    e1 = e1[id0:id1,np.newaxis,...]*param['hoc'][1]

    #print 'Compute spectral budget of buoyancy forcing'
    fh = np.mean( (phc[:,:-1,...] * np.fft.fft2(e1[...,:-1,:-1], axes=(-2,-1))).real, \
                  axis=0, dtype=np.float64 )
    del e1
    fout.variables['psdbf'][:,:] = iso_spec(param['rhooc']*vert_prod(np.diag([1.,-1.]),fh), wv, kr, efac)
else:
    fout.variables['psdbf'][:,:] = 0.

if stomod:
 
    #print 'Read noise'
    u = np.empty((0,dims['nz'],dims['ny']-1,dims['nx']), dtype=np.float64)
    v = np.empty((0,dims['nz'],dims['ny'],dims['nx']-1), dtype=np.float64)
    n = 1
    for s in subs_dir:
        nn = np.minimum(n,2) - 1
        file1 = base_dir + s + 'oclu.nc'
        #print 'Opening file: ', file1
        fin = nc.netcdf_file(file1,'r')
        tmp = fin.variables['ur'][nn:,:,:,:].copy()
        u = np.append(u, tmp.astype('float64'), axis=0)
        tmp = fin.variables['vr'][nn:,:,:,:].copy()
        v = np.append(v, tmp.astype('float64'), axis=0)
        fin.close() 
        del tmp
        n += 1
    u = u[id0:id1,...]
    v = v[id0:id1,...]

    #print 'Compute spectral budget of PV advection by noise'
    dq = sto_advection(q, u, v, dx, dx)
    fh = -np.mean((phc*np.fft.fft2(dq[...,:-1,:-1], axes=(-2,-1))).real, axis=0, dtype=np.float64)
    fout.variables['psdadnoi'][:,:] = iso_spec(vert_prod(rhoh, fh), wv, kr, efac)

    #print 'Compute spectral budget of PV source by noise'
    dq = sto_source(p, u, v, dx, dx)
    del u, v
    fh = -np.mean((phc*np.fft.fft2(dq[...,:-1,:-1], axes=(-2,-1))).real, axis=0, dtype=np.float64)
    fout.variables['psdsonoi'][:,:] = iso_spec(vert_prod(rhoh, fh), wv, kr, efac)
    
    #print 'Read variance'
    if stationary_mode:
        fin = nc.netcdf_file(base_dir+subs_dir[0]+'oclu.nc','r')
        axx = fin.variables['axx'][:,:,:].copy()
        axx = axx.astype('float64')[np.newaxis]
        ayy = fin.variables['ayy'][:,:,:].copy()
        ayy = ayy.astype('float64')[np.newaxis]
        axy = fin.variables['axy'][:,:,:].copy()
        axy = axy.astype('float64')[np.newaxis]
        fin.close()
    else:
        axx = np.empty((0,dims['nz'],dims['ny']-1,dims['nx']), dtype=np.float64)
        ayy = np.empty((0,dims['nz'],dims['ny'],dims['nx']-1), dtype=np.float64)
        axy = np.empty((0,dims['nz'],dims['ny'],dims['nx']), dtype=np.float64)
        n = 1
        for s in subs_dir:
            nn = np.minimum(n,2) - 1
            file1 = base_dir + s + 'oclu.nc'
            #print 'Opening file: ', file1
            fin = nc.netcdf_file(file1,'r')
            tmp = fin.variables['axx'][nn:,:,:,:].copy()
            axx = np.append(axx, tmp.astype('float64'), axis=0)
            tmp = fin.variables['ayy'][nn:,:,:,:].copy()
            ayy = np.append(ayy, tmp.astype('float64'), axis=0)
            tmp = fin.variables['axy'][nn:,:,:,:].copy()
            axy = np.append(axy, tmp.astype('float64'), axis=0)
            fin.close() 
            del tmp
            n += 1
        axx = axx[id0:id1,...]
        ayy = ayy[id0:id1,...]
        axy = axy[id0:id1,...] 

    #print 'Compute spectral budget of PV diffusion by variance'
    dq = sto_diffusion(q, axx, ayy, axy, dx, dx)
    fh = np.mean((-phc*np.fft.fft2(dq[...,:-1,:-1], axes=(-2,-1))).real, axis=0, dtype=np.float64)
    fout.variables['psddivar'][:,:] = iso_spec(vert_prod(rhoh, fh), wv, kr, efac)
    
    #print 'Compute spectral budget of PV sink by variance'
    dq = sto_sink(p, axx, ayy, axy, dx, dx, param['bcco'])
    fh = np.mean((-phc*np.fft.fft2(dq[...,:-1,:-1], axes=(-2,-1))).real, axis=0, dtype=np.float64)
    fout.variables['psdsivar'][:,:] = iso_spec(vert_prod(rhoh, fh), wv, kr, efac)

    #print 'Derive Ito-Stokes drift'
    u = np.zeros_like(axx, dtype=np.float64)
    v = np.zeros_like(ayy, dtype=np.float64)
    u[...,1:-1] = -(axx[...,2:] - axx[...,:-2])/(4.*dx) \
                  -(axy[...,1:,1:-1] - axy[...,:-1,1:-1])/(2.*dx)
    del axx               
    tmp = -(axy[...,1:-1,1:] - axy[...,1:-1,:-1])/(2.*dx) \
          -(ayy[...,2:,:] - ayy[...,:-2,:])/(4.*dx)
    del ayy    
    v[...,1:-1,:] = tmp
    bf = np.zeros_like(axy, dtype=np.float64)
    del axy
    bf[...,1:-1,1:-1] = param['beta']*(tmp[...,:-1] + tmp[...,1:]) 
    del tmp

    #print 'Read correction drift from ',eofile
    fin = nc.netcdf_file(eofile,'r')
    tmp = fin.variables['umean'][:,:,:].copy()
    u -= tmp.astype('float64')[np.newaxis]
    tmp = fin.variables['vmean'][:,:,:].copy()
    v -= tmp.astype('float64')[np.newaxis]
    fin.close()
    del tmp

    #print 'Compute spectral budget of PV advection by correction drift'
    dq = sto_advection(q, u, v, dx, dx)
    del q
    fh = np.mean((-phc*np.fft.fft2(dq[...,:-1,:-1], axes=(-2,-1))).real, axis=0, dtype=np.float64)
    fout.variables['psdadsta'][:,:] = iso_spec(vert_prod(rhoh, fh), wv, kr, efac)
    
    #print 'Compute spectral budget of PV source by correction drift'
    dq = sto_source(p, u, v, dx, dx) + bf
    del p, u, v, bf
    fh = np.mean((-phc*np.fft.fft2(dq[...,:-1,:-1], axes=(-2,-1))).real, axis=0, dtype=np.float64)
    fout.variables['psdsosta'][:,:] = iso_spec(vert_prod(rhoh, fh), wv, kr, efac)

del phc, fh

# Close NetCDF file
fout.close()
#print 'Program terminates'
