from readsnap import readsnap
from dust_plots import *
from astropy.table import Table
import os
import subprocess

halo_dir = '/oasis/tscc/scratch/cchoban/FIRE_2_0_or_h553_criden1000_noaddm_sggs_dust/Species/AHF_data/halos/'
snap_dir = '/oasis/tscc/scratch/cchoban/FIRE_2_0_or_h553_criden1000_noaddm_sggs_dust/Species/output/'
image_dir = './images/'
halo_name = 'halo_0000000.dat'

# First create ouput directory if needed
try:
    # Create target Directory
    os.mkdir(image_dir)
    print "Directory " + image_dir +  " Created " 
except:
    print "Directory " + image_dir +  " already exists"

# Load in halohistory data for main halo. All values should be in code units
halo_data = Table.read(halo_dir + halo_name,format='ascii')

# First and last snapshot numbers
startnum = 10
endnum = 598

# Maximum radius used for getting data
r_max_phys = 5 # kpc


# Now preload the time evolution data
compile_dust_data(snap_dir, foutname='data_5_kpc.pickle', mask=True, overwrite=True, halo_dir=halo_dir+halo_name, r_max=r_max_phys, startnum=startnum, endnum=endnum, implementation='species')

# Plot precompiled data
DZ_vs_time(dataname='data_5_kpc.pickle', data_dir='data/', time=True)

all_data_vs_time(dataname='data_5_kpc.pickle', data_dir='data/', time=True)