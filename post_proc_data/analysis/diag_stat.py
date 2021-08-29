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

# Set parameters
# --------------
# Inputs
main_dir = '/Volumes/Long/q-gcm/gyres_ocean_SST/'  
subdir_ref = 'REF5/diag/'
subdir_mod = [ ['LR40/yrs000-200/',  'POD40/yrs000-200/',  'POD40-P/yrs000-200/' ],
               ['LR80/yrs000-200/',  'POD80/yrs000-200/',  'POD80-P/yrs000-200/' ], 
               ['LR120/yrs000-200/', 'POD120/yrs000-200/', 'POD120-P/yrs000-200/'] ] 
ns = [8,16,24] # subsampling sizes
order_stat = 2 # order of stats: 1-mean; 2-std; 3-skew; 4-kurt.
res_mod = ['40','80','120']
# Outputs
outdir = '../manuscript/ocemod-v1/figures/sst/'
nam_mod = ['DLR','MLU','MLU-P']

# Define functions
# ----------------
def bar_plot(param, u):
    "Bar-plot of data by group"
    [dim_data, dim_group] = np.shape(u)
    x = np.arange(dim_data)  # label locations
    width = 0.25  # width of bars
    plt.style.use("fivethirtyeight") # "fivethirtyeight", "ggplot", "seaborn"
    fig, ax = plt.subplots(figsize=(4,3))
    x1 = x - width
    for j in range(dim_group):
        ax.bar(x1, u[:,j], width, label=param['list_group'][j])
        x1 += width
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_yscale(param['yscale'])
    ax.set_xlabel(param['xlabel'], fontsize=10)
    if param['yscale'] == 'log':
        ax.set_ylabel(r'Log-scale', fontsize=10)
    else:
        ax.set_ylim([0.,1.25*u.max()])
    ax.set_title(param['title'], fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(param['list_data'])
    ax.tick_params(axis='both', which='major', labelsize=10)
    fig.tight_layout()
    plt.savefig(param['output'], dpi=200, bbox_inches='tight', pad_inches=0)
    if param['add_legend']:
        handles, labels = ax.get_legend_handles_labels()
        fig1, ax1 = plt.subplots(figsize=(4,3))
        ax1.legend(handles=handles, loc='center', borderaxespad=0., fontsize=15)
        plt.gca().set_axis_off()
        fig1.tight_layout()
        plt.savefig('leg-mod.eps', dpi=200, bbox_inches='tight', pad_inches=0)
    return    

def metric_pstat(pmod, pref):
    "Compute some metrics of statistics "
    
    [nm,nz,ny,nx] = np.shape(pmod)
    afac = 1./(ny-1)/(nx-1)
    
    res = np.zeros((nm+3,nz)) 
    for k in range(nz):
        
        # Root-mean-squared-error (RMSE) of moments
        for m in range(nm):
            tmp = (pmod[m,k,:,:] - pref[m,k,:,:])**2.
            res[m,k] = np.sqrt(afac * area_int(tmp, 0.5, 0.5))
        
        # Pattern correlation (PC)
        smod = pmod[1,k,:,:] # std of k-th pmod
        sref = pref[1,k,:,:] # std of k-th pref
        tmp_num = area_int((smod*sref)**2., 0.5, 0.5)
        tmp_denom = np.sqrt( area_int(sref**4., 0.5, 0.5) * \
                             area_int(smod**4., 0.5, 0.5) )
        res[nm,k] = tmp_num/tmp_denom
        
        # Dispersion
        tmp = (sref/smod)**2.
        tmp1 = tmp - np.ones((ny,nx)) - np.log(tmp)
        res[nm+1,k] = afac*area_int(tmp1, 0.5, 0.5)

        # Gaussian relative entropy (GRE)
        tmp1 += (pmod[0,k,:,:] - pref[0,k,:,:])**2. / smod
        res[nm+2,k] = 0.5*afac*area_int(tmp1, 0.5, 0.5)

    return res

def area_int(u, facwe, facsn):
    "Area-integration of 2D array with specific W-E and S-N boundary factors"
    
    res = 0.
    # Inner points
    res += np.sum(u[1:-1,1:-1])
    
    # S-N boundary
    res += facsn*( np.sum(u[0,1:-1]) + np.sum(u[-1,1:-1]) ) 
    
    # W-E boundary
    res += facwe*( np.sum(u[1:-1,0]) + np.sum(u[1:-1,-1]) ) 
    
    # Corners
    res += facwe*facsn*( u[0,0] + u[-1,0] + u[0,-1] + u[-1,-1] )
    
    return res


# Read ref. model stats
# ---------------------
f = nc.netcdf_file(main_dir + subdir_ref + 'stats.nc', 'r')
#xp = f.variables['xp'][::ns].copy() 
#yp = f.variables['yp'][::ns].copy()
z  = f.variables['z'][:].copy()
zi = f.variables['zi'][:].copy()
pref = f.variables['pst'][0:order_stat,:,:,:].copy() # (Sv) 
href = f.variables['hst'][0:order_stat,:,:,:].copy() # (m)
f.close()

# Compute metrics of coarse model stats
# -------------------------------------
nmod = len(subdir_mod[0])
nres = len(subdir_mod[:][0])
pmet = np.zeros((nres,nmod,order_stat+3))
hmet = pmet.copy()
# Non-uniform weight of vertical integration for 'p'
pwt = []
tmp = 0.
for k in range(len(z)):
    hoc = 2.0*(z[k] - tmp)
    tmp += hoc
    pwt.append(hoc)
pwt = pwt/sum(pwt)
# Uniform weight of vertical integration for 'h'
hwt = zi.copy()
hwt[:] = 0.5
for r in range(nres):
    pref1 = pref[:,:,::ns[r],::ns[r]]
    href1 = href[:,:,::ns[r],::ns[r]]
    for m in range(nmod):
        # Read stats
        f = nc.netcdf_file(main_dir + subdir_mod[r][m] + 'stats.nc', 'r')
        pmod = f.variables['pst'][0:2,:,:,:].copy() 
        hmod = f.variables['hst'][0:2,:,:,:].copy() 
        # Compute metrics
        ptmp = metric_pstat(pmod, pref1)
        htmp = metric_pstat(hmod, href1)
        # Vertical integration
        pmet[r,m,:] = np.matmul(ptmp, pwt)
        hmet[r,m,:] = np.matmul(htmp, hwt)
f.close()


# Bar-plot of metrics
# -------------------
param = dict()
param['list_data'] = res_mod 
param['list_group'] = nam_mod 
param['xlabel'] = r'Resolution (km)'

param['yscale'] = 'linear'
param['add_legend'] = True

param['title'] = r'RMSE of $\psi$-mean (Sv)'
param['output'] = outdir + 'RMSE-pm.eps'
bar_plot(param, pmet[:,:,0])

param['add_legend'] = False

param['title'] = r'RMSE of $\eta$-mean (m)'
param['output'] = outdir + 'RMSE-hm.eps'
bar_plot(param, hmet[:,:,0])

param['title'] = r'RMSE of $\psi$-std (Sv)'
param['output'] = outdir + 'RMSE-ps.eps'
bar_plot(param, pmet[:,:,1])

param['title'] = r'RMSE of $\eta$-std (m)'
param['output'] = outdir + 'RMSE-hs.eps'
bar_plot(param, hmet[:,:,1])

param['title'] = r'PC of $\psi$'
param['output'] = outdir + 'PC-p.eps'
bar_plot(param, pmet[:,:,2])

param['title'] = r'PC of $\eta$'
param['output'] = outdir + 'PC-h.eps'
bar_plot(param, hmet[:,:,2])

param['yscale'] = 'log'

param['title'] = r'Dispersion of $\psi$'
param['output'] = outdir + 'Disp-p.eps'
bar_plot(param, pmet[:,:,3])

param['title'] = r'Dispersion of $\eta$'
param['output'] = outdir + 'Disp-h.eps'
bar_plot(param, hmet[:,:,3])

param['title'] = r'GRE of $\psi$'
param['output'] = outdir + 'GRE-p.eps'
bar_plot(param, pmet[:,:,4])

param['title'] = r'GRE of $\eta$'
param['output'] = outdir + 'GRE-h.eps'
bar_plot(param, hmet[:,:,4])

