#!/usr/bin/env python3

# Load modules
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.ion()

def draw_energy (eng, ax, cmax, tit, out):
    fig = plt.figure(figsize=(5,6.5))
    im = plt.imshow(eng, interpolation='bilinear', origin='lower', 
            cmap='gist_heat_r', vmin=0., vmax=cmax, extent=ax)
    plt.yticks(rotation = 90) 
    plt.xlabel(r'$x$ (km)')
    plt.ylabel(r'$y$ (km)')
    plt.title(tit)
    cb = plt.colorbar(im, format='%3.1f', orientation='horizontal', extend='both', fraction=0.035, pad=0.1)
    cb.set_label('(J/m$^3$)')
    plt.savefig(out, bbox_inches='tight', pad_inches=0)
    
# Set param
with_sst = True
#model = 'REF5'
#model = 'DET40'
model = 'EOFP40'
#tit = 'REF (5 km)'
#tit = 'DET (40 km)'
tit = 'STO-EOF-P (40 km)'
if with_sst:
    fc = 0.4
    #indir = r'/media/long/Long/q-gcm/double_gyre_ocean_only_sst/%s/yrs195-210/'%(model)
    #indir = r'/media/long/Long/q-gcm/double_gyre_ocean_only_sst/%s/yrs090-210/'%(model)
    indir = r'/media/long/Long/q-gcm/double_gyre_ocean_only_sst/%s/salt/'%(model)
    outdir = r'/home/long/Desktop/fig/oc_sst/%s/'%(model)
else:
    fc = 0.1
    #indir = r'/media/long/Long/q-gcm/double_gyre_ocean_only/%s/yrs195-210/'%(model)
    #indir = r'/media/long/Long/q-gcm/double_gyre_ocean_only/%s/yrs090-210/'%(model)
    indir = r'/media/long/Long/q-gcm/double_gyre_ocean_only/%s/salt/'%(model)
    outdir = r'/home/long/Desktop/fig/oc_only/%s/'%(model) 
h = np.asarray([350.0,750.0,2900.0]) # thickness (m)
rho = 1.e3 # density (kg/m^3)
#fc = 1.

# Read data
hm, rhfc = 0.5*(h[:-1]+h[1:]), rho/np.sum(h)
f = Dataset(indir+'ocdiag.nc','r')
z = f.variables['z'][:].data 
y = f.variables['yp'][:].data 
x = f.variables['xp'][:].data 
mke = rhfc*np.einsum('kji,k->ji',f.variables['mke'][...].data,h) 
eke = rhfc*np.einsum('kji,k->ji',f.variables['eke'][...].data,h) 
mpe = rhfc*np.einsum('kji,k->ji',f.variables['mpe'][...].data,hm) 
epe = rhfc*np.einsum('kji,k->ji',f.variables['epe'][...].data,hm) 
f.close()
axp = [x[0], x[-1], y[0], y[-1]]
axt = [0.5*(x[0]+x[1]), 0.5*(x[-2]+x[-1]), 0.5*(y[0]+y[1]), 0.5*(y[-2]+y[-1])]

# Plot MKE
if with_sst:
    output = outdir + r'mke-%s-sst.pdf'%(model)
    cm = 20.
else:
    output = outdir + r'mke-%s.pdf'%(model)
    cm = 10.
draw_energy(mke, axt, cm, tit, output)

# Plot EKE
if with_sst:
    output = outdir + r'eke-%s-sst.pdf'%(model)
    cm = fc*50.
else:
    output = outdir + r'eke-%s.pdf'%(model)
    cm = fc*10.
draw_energy(eke, axt, cm, tit, output)

# Plot MPE
if with_sst:
    output = outdir + r'mpe-%s-sst.pdf'%(model)
    cm = 150.
else:
    output = outdir + r'mpe-%s.pdf'%(model)
    cm = 30.
draw_energy(mpe, axp, cm, tit, output)

# Plot EPE
if with_sst:
    output = outdir + r'epe-%s-sst.pdf'%(model)
    cm = fc*50.
else:
    output = outdir + r'epe-%s.pdf'%(model)
    cm = fc*10.    
draw_energy(epe, axp, cm, tit, output)
