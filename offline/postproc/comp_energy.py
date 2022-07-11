#!/usr/bin/env python3

# Load modules
# ------------
import numpy as np
import scipy.io.netcdf as nc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import FormatStrFormatter

# Define functions
# ----------------
def bar_plot(param, u):
    "Bar-plot of data by group"
    [dim_data, dim_group] = np.shape(u)
    x = np.arange(dim_data)  # label locations
    width = 0.25  # width of bars
    plt.style.use("fivethirtyeight") # "fivethirtyeight", "ggplot", "seaborn"
    fig, ax = plt.subplots(figsize=(3,3))
    x1 = x - width
    for j in range(dim_group):
        ax.bar(x1, u[:,j], width, label=param['list_group'][j])
        x1 += width
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_yscale(param['yscale'])
    ax.set_xlabel(param['xlabel'], fontsize=10)
    ax.set_title(param['title'], fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(param['list_data'])
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    plt.savefig(param['output'], bbox_inches='tight', pad_inches=0)

def area_int(u, facwe, facsn):
    "Area-integration of `u` with specific W-E and S-N boundary factors `facwe` and `facsn`"    
    # Inner points
    res = np.sum(u[1:-1,1:-1])
    # S-N boundary
    res += facsn*( np.sum(u[0,1:-1]) + np.sum(u[-1,1:-1]) )  
    # W-E boundary
    res += facwe*( np.sum(u[1:-1,0]) + np.sum(u[1:-1,-1]) ) 
    # Corners
    return res + facwe*facsn*( u[0,0]+ u[-1,0] + u[0,-1] + u[-1,-1] )

# Set parameters
# --------------
# Inputs
with_sst = True
if with_sst:
    main_dir = '/media/long/Long/q-gcm/double_gyre_ocean_only_sst/'  
    outdir = '/home/long/Desktop/fig/oc_sst/' 
else:
    main_dir = '/media/long/Long/q-gcm/double_gyre_ocean_only/'  
    outdir = '/home/long/Desktop/fig/oc_only/' 
main_dir = '/media/long/Long/q-gcm/double_gyre_coupled/'  
outdir = '/home/long/Desktop/fig/couple/' 
subdir_ref = 'REF5/yrs165-180/'    
#subdir_mod = [ ['DET40/yrs090-210/',  'EOF40/salt/',  'EOFP40/salt/' ],
#               ['DET80/yrs090-210/',  'EOF80/salt/',  'EOFP80/salt/' ], 
#               ['DET120/yrs090-210/', 'EOF120/salt/', 'EOFP120/salt/'] ] 
subdir_mod = [ ['DET80/yrs000-180/',  'EOF80/yrs000-180/', 'DMD80/yrs000-180/' ] ] 
#res_mod = ['40','80','120']
res_mod = ['80']
#nam_mod = ['DET','STO-EOF','STO-EOF-P']
nam_mod = ['DET','STO-EOF','STO-DMD']
h = np.asarray([350.0,750.0,2900.0]) # thickness (m)
rho = 1.e3 # density (kg/m^3)
hm = 0.5*(h[:-1] + h[1:])

f = nc.netcdf_file(main_dir + subdir_ref + 'ocdiag1.nc', 'r')
efc = rho/(f.dimensions['yt']*f.dimensions['xt']*np.sum(h))
mke_ref = efc * area_int(np.einsum('kji,k->ji',f.variables['mke'][...].data,h), 1.0, 1.0)
eke_ref = efc * area_int(np.einsum('kji,k->ji',f.variables['eke'][...].data,h), 1.0, 1.0) 
mpe_ref = efc * area_int(np.einsum('kji,k->ji',f.variables['mpe'][...].data,hm), 0.5, 0.5)
epe_ref = efc * area_int(np.einsum('kji,k->ji',f.variables['epe'][...].data,hm), 0.5, 0.5) 
f.close()

print(f'Ref. MKE = {mke_ref} (J/m^3)')
print(f'Ref. MPE = {mpe_ref} (J/m^3)')
print(f'Ref. EKE = {eke_ref} (J/m^3)')
print(f'Ref. EPE = {epe_ref} (J/m^3)')

nmod, nres = len(subdir_mod[0]), len(subdir_mod[:][0])
mke, mpe = np.zeros((nres,nmod)), np.zeros((nres,nmod))
eke, epe = mke.copy(), mpe.copy()
for r in range(nres):
    f = nc.netcdf_file(main_dir + subdir_mod[r][0] + 'ocdiag.nc', 'r')
    efc = rho/(f.dimensions['yt']*f.dimensions['xt']*np.sum(h))
    f.close()
    for m in range(nmod):
        f = nc.netcdf_file(main_dir + subdir_mod[r][m] + 'ocdiag.nc', 'r')
        mke[r,m] = efc * area_int(np.einsum('kji,k->ji',f.variables['mke'][...].data,h), 1.0, 1.0)
        eke[r,m] = efc * area_int(np.einsum('kji,k->ji',f.variables['eke'][...].data,h), 1.0, 1.0) 
        mpe[r,m] = efc * area_int(np.einsum('kji,k->ji',f.variables['mpe'][...].data,hm), 0.5, 0.5)
        epe[r,m] = efc * area_int(np.einsum('kji,k->ji',f.variables['epe'][...].data,hm), 0.5, 0.5) 
        f.close()

# Bar-plot of metrics
# -------------------
plt.ion()
param = dict()
param['list_data'] = res_mod 
param['list_group'] = nam_mod 
param['xlabel'] = 'Resolution (km)'
param['yscale'] = 'linear'
param['title'] = 'MKE (J/m$^3$)'
if with_sst:
    param['output'] = outdir + 'bar-mke-sst.pdf'
else:
    param['output'] = outdir + 'bar-mke.pdf'
bar_plot(param, mke)

param['title'] = 'MPE (J/m$^3$)'
if with_sst:
    param['output'] = outdir + 'bar-mpe-sst.pdf'
else:
    param['output'] = outdir + 'bar-mpe.pdf'
bar_plot(param, mpe)

#param['yscale'] = 'log'
param['title'] = 'EKE (J/m$^3$)'
if with_sst:
    param['output'] = outdir + 'bar-eke-sst.pdf'
else:
    param['output'] = outdir + 'bar-eke.pdf'
bar_plot(param, eke)

param['title'] = 'EPE (J/m$^3$)'
if with_sst:
    param['output'] = outdir + 'bar-epe-sst.pdf'
else:
    param['output'] = outdir + 'bar-epe.pdf'
bar_plot(param, epe)
