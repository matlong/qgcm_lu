import matplotlib.pyplot as plt
import numpy as np
from scipy.io import netcdf

fileout = './avges.nc'

# physical constants
tau0 = 2e-5

# 5km
#si_x = 768+1
#si_y = 960+1 

# 10km
#si_x = 384+1 
#si_y = 480+1 

# 20km
#si_x = 192+1
#si_y = 240+1 

# 40km
#si_x = 96+1
#si_y = 120+1 

# 80km
si_x = 48+1
si_y = 60+1

# 120km
#si_x = 32+1
#si_y = 40+1 

taux = np.zeros((si_y,si_x))
tauy = np.zeros((si_y,si_x))
fnet = np.zeros((si_y-1,si_x-1))

# Store 
f = netcdf.netcdf_file(fileout,'w')

f.createDimension('ypo',si_y)
f.createDimension('xpo',si_x)
f.createDimension('yto',si_y-1)
f.createDimension('xto',si_x-1)

ypo = f.createVariable('ypo', 'd', ('ypo',))
xpo = f.createVariable('xpo', 'd', ('xpo',))
yto = f.createVariable('yto', 'd', ('yto',))
xto = f.createVariable('xto', 'd', ('xto',))

tauxo  = f.createVariable('tauxo' , 'd', ('ypo','xpo',))
tauyo  = f.createVariable('tauyo' , 'd', ('ypo','xpo',))
fnetoc = f.createVariable('fnetoc', 'd', ('yto','xto',))

ypo[:] = np.arange(si_y)
xpo[:] = np.arange(si_x)
yto[:] = np.arange(si_y-1)
xto[:] = np.arange(si_x-1)

for ny in range(0,si_y):
  taux[ny,:] = tau0*(-np.cos((ny+0.5)/si_y*2*np.pi))


tauxo [:,:] = taux
tauyo [:,:] = np.zeros((si_y,si_x))
fnetoc[:,:] = np.zeros((si_y-1,si_x-1))

f.close()
