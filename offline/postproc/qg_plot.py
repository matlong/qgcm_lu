#!/usr/bin/env python

# Load modules
# ------------
import numpy as np
import scipy.io.netcdf as nc
import scipy.io as sio
from scipy import signal
import matplotlib.pyplot as plt
import os
#from netCDF4 import Dataset

plt.ion()

# Set parameters
stomod = False
datdir = '/Volumes/Long/q-gcm/double_gyre_ocean_only/EOF80/paper/yrs090-210/'
Ld = 39.e3 # max. Baroclinic deformation radiis (m)
freq_cut = 0.5
#ymin = 0.
#ymax = 10.
#tit = 'LU-DMD (80 km)'
#cmax = 20. # [40,50,20]
#l = 2

file1 = datdir + 'ocdiag.nc' 
print 'Read data from ',file1
fin = nc.netcdf_file(file1,'r')
nz = fin.dimensions['z']
wn = fin.variables['k'][:].copy() # isotropic wavenumbers
h = fin.variables['h'][:].copy()
x = fin.variables['xp'][:].copy() # (km)
y = fin.variables['yp'][:].copy()
t = fin.variables['time_simu'][:].copy() # (yrs)
ke = fin.variables['psdke'][:,:].copy()
pe = fin.variables['psdpe'][:,:].copy()
kem = fin.variables['psdke_vm'][:,:].copy()
pem = fin.variables['psdpe_vm'][:,:].copy()
dke = fin.variables['psdfke'][:,:].copy()
dpe = fin.variables['psdfpe'][:,:].copy()
by = fin.variables['psdby'][:,:].copy()
ek = fin.variables['psdek'][:].copy()
vs = fin.variables['psdvs'][:,:].copy()
wf = fin.variables['psdwf'][:].copy()
bf = fin.variables['psdbf'][:,:].copy()
ts_mke = fin.variables['tsseke'][:,:].copy()
ts_eke = fin.variables['tsteke'][:,:].copy()
ts_mkem = fin.variables['tsseke_vm'][:,:].copy()
ts_ekem = fin.variables['tsteke_vm'][:,:].copy()
p = fin.variables['pstat'][0,:,:,:].copy()
pm = fin.variables['pstat_vm'][0,:,:,:].copy()
tm_mke = fin.variables['tmseke'][:,:,:].copy()
tm_eke = fin.variables['tmteke'][:,:,:].copy()
tm_mkem = fin.variables['tmseke_vm'][:,:,:].copy()
tm_ekem = fin.variables['tmteke_vm'][:,:,:].copy()
if stomod:
    adnoi = fin.variables['psdadnoi'][:,:].copy()
    sonoi = fin.variables['psdsonoi'][:,:].copy()
    adsta = fin.variables['psdadsta'][:,:].copy()
    sosta = fin.variables['psdsosta'][:,:].copy()
    divar = fin.variables['psddivar'][:,:].copy()
    sivar = fin.variables['psdsivar'][:,:].copy()
fin.close()

dw = wn[1] - wn[0]
etfc = 1./np.sum(h.astype('float64'))
ekfc = etfc*wn/dw 
dx = 1.e3*(x[1]-x[0]) # (m)
dt = t[1] - t[0]
freq_samp = 1./dt  
b, a = signal.butter(5, freq_cut/(freq_samp/2.), 'low') # low-pass filter
ax = [x.min(), x.max(), y.min(), y.max()]
leg = ['Barotropic','1st Baroclinic','2nd Baroclinic']

outdir = datdir + 'fig/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

'''
plt.figure(figsize=(5,3))
plt.semilogx(wn, ekfc*np.sum(dke, axis=0, dtype=np.float64), label=r'KE flux')
plt.semilogx(wn, ekfc*np.sum(dpe, axis=0, dtype=np.float64), label=r'PE flux')
plt.vlines(1./Ld, -0.3, 0.3, colors='k', label=r'$1 / L_d$')
plt.xlim([wn[0],np.pi/dx])
plt.ylim([-0.3,0.3])
plt.grid(which='both', axis='both')
plt.legend(fontsize=7, loc='best')
plt.xlabel(r'Wavenumbers (rad/m)') 
plt.ylabel(r'Spectral density (W/m$^2$)') 
plt.title(tit)
plt.savefig(datdir+'psd_trans1.eps', ndpi=200, bbox_inches='tight', pad_inches=0)

nstep = 10  # nb. of contour intervals
step = cmax/nstep
clin = np.arange(-cmax,cmax+step,step)
plt.figure(figsize=(5,6.5))
im = plt.imshow(1.e3*pm[nz-l-1,...], interpolation='bilinear', origin='lower', cmap='RdBu_r', 
        vmin=-cmax, vmax=cmax, extent=ax)
plt.contour(x, y, 1.e3*pm[nz-l-1,...], levels=clin, linewidths=1.0, colors='k')
plt.xlabel(r'$x$ (km)')
plt.ylabel(r'$y$ (km)')
plt.title(tit)
cb = plt.colorbar(im, orientation='horizontal', extend='both', fraction=0.035, pad=0.1)
cb.set_label(r'(Sv/km)')
plt.savefig(datdir+r'pm_mod%d.eps'%(l), ndpi=200, bbox_inches='tight', pad_inches=0)
'''

print 'Plot time-mean of KE'
plt.figure(figsize=(5,6.5))
im = plt.imshow(etfc*np.sum(tm_mke, axis=0, dtype=np.float64), interpolation='bilinear', 
            origin='lower', cmap='Reds', extent=ax)
plt.xlabel(r'$x$ (km)')
plt.ylabel(r'$y$ (km)')
plt.title(r'Total MKE')
cb = plt.colorbar(im, orientation='horizontal', extend='both', fraction=0.035, pad=0.1)
cb.set_label(r'(J/m$^3$)')
plt.savefig(outdir+'mke_tot.eps', ndpi=200, bbox_inches='tight', pad_inches=0)
#plt.show()

plt.figure(figsize=(5,6.5))
im = plt.imshow(etfc*np.sum(tm_eke, axis=0, dtype=np.float64), interpolation='bilinear', 
            origin='lower', cmap='Reds', extent=ax)
plt.xlabel(r'$x$ (km)')
plt.ylabel(r'$y$ (km)')
plt.title(r'Total EKE')
cb = plt.colorbar(im, orientation='horizontal', extend='both', fraction=0.035, pad=0.1)
cb.set_label(r'(J/m$^3$)')
plt.savefig(outdir+'eke_tot.eps', ndpi=200, bbox_inches='tight', pad_inches=0)
#plt.show()

for l in range(3):
    plt.figure(figsize=(5,6.5))
    im = plt.imshow(tm_mkem[nz-l-1,...], interpolation='bilinear', 
            origin='lower', cmap='Reds', extent=ax)
    plt.xlabel(r'$x$ (km)')
    plt.ylabel(r'$y$ (km)')
    plt.title(r'%s MKE'%(leg[l]))
    cb = plt.colorbar(im, orientation='horizontal', extend='both', fraction=0.035, pad=0.1)
    cb.set_label(r'(J/m$^3$)')
    plt.savefig(outdir+r'mke_mod%d.eps'%(l), ndpi=200, bbox_inches='tight', pad_inches=0)
    #plt.show()

for l in range(3):
    plt.figure(figsize=(5,6.5))
    im = plt.imshow(tm_ekem[nz-l-1,...], interpolation='bilinear', 
            origin='lower', cmap='Reds', extent=ax)
    plt.xlabel(r'$x$ (km)')
    plt.ylabel(r'$y$ (km)')
    plt.title(r'%s EKE'%(leg[l]))
    cb = plt.colorbar(im, orientation='horizontal', extend='both', fraction=0.035, pad=0.1)
    cb.set_label(r'(J/m$^3$)')
    plt.savefig(outdir+r'eke_mod%d.eps'%(l), ndpi=200, bbox_inches='tight', pad_inches=0)
    #plt.show()

print 'Plot time-mean of streamfunction'
for l in range(3):
    plt.figure(figsize=(5,6.5))
    im = plt.imshow(1.e3*pm[nz-l-1,...], interpolation='bilinear', 
            origin='lower', cmap='RdBu_r', extent=ax)
    plt.contour(x, y, 1.e3*pm[nz-l-1,...], linewidths=1.0, colors='k')
    plt.xlabel(r'$x$ (km)')
    plt.ylabel(r'$y$ (km)')
    plt.title(r'%s streamfunction'%(leg[l]))
    cb = plt.colorbar(im, orientation='horizontal', extend='both', fraction=0.035, pad=0.1)
    cb.set_label(r'(Sv/km)')
    plt.savefig(outdir+r'pm_mod%d.eps'%(l), ndpi=200, bbox_inches='tight', pad_inches=0)
    #plt.show()

print 'Plot energy spectrum'
plt.figure(figsize=(5,3.5))
for l in range(nz):
    plt.loglog(wn, ke[l,:], label=r'Layer %d'%(l+1))
plt.ylim([1e0,1e9])
plt.grid(which='both', axis='both')
plt.legend(loc='lower left')
plt.xlabel(r'Isotropic wavenumbers (rad/m)') 
plt.ylabel(r'Power spectral density (J/m/rad)') 
plt.title(r'Kinetic Energy Spectra')
plt.savefig(outdir+'psdke.eps', ndpi=200, bbox_inches='tight', pad_inches=0)
#plt.show()

plt.figure(figsize=(5,3.5))
for l in range(nz-1):
    plt.loglog(wn, pe[l,:], label=r'Interface %d'%(l+1))
plt.ylim([1e-2,1e11])
plt.grid(which='both', axis='both')
plt.legend(loc='lower left')
plt.xlabel(r'Isotropic wavenumbers (rad/m)') 
plt.ylabel(r'Power spectral density (J/m/rad)') 
plt.title(r'Potential Energy Spectra')
plt.savefig(outdir+'psdpe.eps', ndpi=200, bbox_inches='tight', pad_inches=0)
#plt.show()

print 'Plot modal energy spectrum'
plt.figure(figsize=(5,3.5))
for l in range(3):
    plt.loglog(wn, kem[nz-l-1,:])
plt.ylim([1e-3,1e6])
plt.grid(which='both', axis='both')
plt.legend(leg, loc='lower left')
plt.xlabel(r'Isotropic wavenumbers (rad/m)') 
plt.ylabel(r'Power spectral density (J/m$^2$/rad)') 
plt.title(r'Kinetic Energy Spectra')
plt.savefig(outdir+'psdke_mod.eps', ndpi=200, bbox_inches='tight', pad_inches=0)
#plt.show()

plt.figure(figsize=(5,3.5))
for l in range(2):
    plt.loglog(wn, pem[nz-l-2,:])
plt.ylim([1e-3,1e10])
plt.grid(which='both', axis='both')
plt.legend(leg[1:], loc='lower left')
plt.xlabel(r'Isotropic wavenumbers (rad/m)') 
plt.ylabel(r'Power spectral density (J/m$^2$/rad)') 
plt.title(r'Potential Energy Spectra')
plt.savefig(outdir+'psdpe_mod.eps', ndpi=200, bbox_inches='tight', pad_inches=0)
#plt.show()

print 'Plot spectral transfert'
plt.figure(figsize=(5,3.5))
plt.semilogx(wn, ekfc*np.sum(dke, axis=0, dtype=np.float64), label=r'KE flux')
plt.semilogx(wn, ekfc*np.sum(dpe, axis=0, dtype=np.float64), label=r'PE flux')
#plt.semilogx(wn, ekfc*np.sum(by, axis=0, dtype=np.float64), label=r'Beta effect')
plt.semilogx(wn, ekfc*ek, label=r'Ekman drag')
plt.semilogx(wn, ekfc*np.sum(vs, axis=0, dtype=np.float64), label=r'Dissipation')
plt.semilogx(wn, ekfc*wf, label=r'Wind forcing')
plt.semilogx(wn, ekfc*np.sum(bf, axis=0, dtype=np.float64), label=r'Buoyancy forcing')
axes = plt.gca()
ymin, ymax = axes.get_ylim()
plt.vlines(1./Ld, ymin, ymax, colors='k', label=r'$1 / L_d$')
plt.xlim([wn[0],np.pi/dx])
plt.grid(which='both', axis='both')
plt.legend(fontsize=7, loc='upper right')
plt.xlabel(r'Isotropic wavenumbers (rad/m)') 
plt.ylabel(r'Power spectral density (W/m$^2$)') 
plt.title(r'Spectral Energy Transfers')
plt.savefig(outdir+'psd_trans.eps', ndpi=200, bbox_inches='tight', pad_inches=0)
#plt.show()

if stomod:
    print 'Plot spectral transfert by stochastic terms'
    plt.figure(figsize=(5,3.5))
    plt.semilogx(wn, ekfc*np.sum(adnoi, axis=0, dtype=np.float64), label=r'Advection by noise')
    plt.semilogx(wn, ekfc*np.sum(adsta, axis=0, dtype=np.float64), label=r'Advection by correction')
    plt.semilogx(wn, ekfc*np.sum(sonoi, axis=0, dtype=np.float64), label=r'Source by noise')
    plt.semilogx(wn, ekfc*np.sum(sosta, axis=0, dtype=np.float64), label=r'Source by correction')
    plt.semilogx(wn, ekfc*np.sum(divar, axis=0, dtype=np.float64), label=r'Diffusion by variance')
    plt.semilogx(wn, ekfc*np.sum(sivar, axis=0, dtype=np.float64), label=r'Sink by variance')
    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    plt.vlines(1./Ld, ymin, ymax, colors='k', label=r'$1 / L_d$')
    plt.xlim([wn[0],np.pi/dx])
    plt.grid(which='both', axis='both')
    plt.legend(fontsize=7, loc='lower left')
    plt.xlabel(r'Isotropic wavenumbers (rad/m)') 
    plt.ylabel(r'Power spectral density (W/m$^2$)') 
    plt.title(r'Spectral Energy Transfers')
    plt.savefig(outdir+'psd_trans_sto.eps', ndpi=200, bbox_inches='tight', pad_inches=0)
    #plt.show()

print 'Plot time series of KE'
plt.figure(figsize=(5,3.5))
plt.plot(t, etfc*np.sum(ts_mke, axis=-1, dtype=np.float64), label=r'MKE')
plt.plot(t, signal.filtfilt(b, a, etfc*np.sum(ts_eke,axis=-1,dtype=np.float64), axis=0), label=r'EKE')
plt.grid(which='both', axis='both')
plt.legend(loc='lower right')
plt.xlabel(r'Times (years)')
plt.ylabel(r'Energy (J/m$^3$)')
plt.title(r'Time-series of Kinetic Energy')
plt.savefig(outdir+'tske.eps', ndpi=200, bbox_inches='tight', pad_inches=0)
#plt.show()

plt.figure(figsize=(5,3.5))
for l in range(nz):
    plt.plot(t, ts_mkem[:,nz-l-1])
plt.grid(which='both', axis='both')
plt.legend(leg, loc='lower right')
plt.xlabel(r'Times (years)')
plt.ylabel(r'Energy (J/m$^3$)')
plt.title(r'Time-series of modal MKE')
plt.savefig(outdir+'tsmke_mod.eps', ndpi=200, bbox_inches='tight', pad_inches=0)
#plt.show()

plt.figure(figsize=(5,3.5))
for l in range(nz):
    plt.plot(t, signal.filtfilt(b, a, ts_ekem[:,nz-l-1], axis=0))
plt.grid(which='both', axis='both')
plt.legend(leg, loc='lower right')
plt.xlabel(r'Times (years)')
plt.ylabel(r'Energy (J/m$^3$)')
plt.title(r'Time-series of modal EKE')
plt.savefig(outdir+'tseke_mod.eps', ndpi=200, bbox_inches='tight', pad_inches=0)
#plt.show()
