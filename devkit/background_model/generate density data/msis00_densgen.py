#!/usr/bin/env python
import argparse
from pymsis import msis
import datetime as datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.io import loadmat
import pandas as pd
import os
from astropy.time import Time
from astropy.coordinates import get_sun

# Create the parser
parser = argparse.ArgumentParser()

# Add an argument
parser.add_argument('--year', type=int, default=1999)

# Parse the argument
args = parser.parse_args()

# Print "Hello" + the user input argument
print('Currently running sample ', args.year)
year = args.year

# Density grid 
localSolarTimes = np.linspace(0,24,24)
latitudes = np.linspace(-87.5,87.5,20)
altitudes = np.linspace(100,800,36)
nofAlt = altitudes.shape[0]
nofLst = localSolarTimes.shape[0]
nofLat = latitudes.shape[0]

## Generate hourly density data using density grid resolution
# For year 1999 to 2022 
month = 1
day = 1
date0 = datetime.datetime(year, month, day)

if year%4:
    no_days = 365
else:
    no_days = 366

msis00_rho_mat = np.zeros((no_days*24,17280))
msis00_sw_mat = np.zeros((no_days*24,11))

# Calculate hourly density data
for ik in range(no_days*24): #range(no_days*24)
    date1 = date0+datetime.timedelta(hours=ik)
    time1 = date1.strftime("%Y-%m-%d %H:%M:%S")
    print(time1, ik%24)
    
    lon = 15*(localSolarTimes-(ik%24))
    
    msis00_sw = msis.create_input(date1, lon[0], latitudes[0], altitudes[0])[1]
    msis00_sw_mat[ik,:] = msis00_sw[0,[0,1,5,6,7,8,9,10,11,12,13]]
    
    if msis00_sw[0,7] < 50:
        msis00_rho = msis.run(date1, lon, latitudes, altitudes, version=0)[:,:,:,:,0]
    else: # Use storm-time Ap mode during high geomagnetic condition
        msis00_rho = msis.run(date1, lon, latitudes, altitudes, geomagnetic_activity=-1, version=0)[:,:,:,:,0]
    msis00_rho_mat[ik,:] = msis00_rho.flatten()

filename_rho = 'MSIS00_density/'+str(year)+'_MSIS00_density.npy'
np.save(filename_rho,msis00_rho_mat)

filename_sw = 'MSIS00_density/'+str(year)+'_MSIS00_sw.npy'
np.save(filename_sw,msis00_sw_mat)

