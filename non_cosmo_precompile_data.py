from readsnap import readsnap
from dust_plots import *
from astropy.table import Table
import os
import subprocess


main_dir = '/oasis/tscc/scratch/cchoban/non_cosmological/non_cosmological_runs/Elemental/'
snap_dirs = [main_dir+'fiducial_model/', main_dir+'species_creation_eff', main_dir+'enhanced_dest', main_dir+'decreased_stellar']
names = ['fiducial_model','species_creation_eff','enhanced_dest','decreased_stellar']
image_dir = './images/'

main_dir = '/oasis/tscc/scratch/cchoban/non_cosmological/non_cosmological_runs/Elemental/'
snap_dirs = ['./']
names = 'fiducial_model'
image_dir = './images/'

implementation = 'elemental'


dataname = implementation+'_'+names+'_data_20_kpc.pickle'

# First create ouput directory if needed
try:
    # Create target Directory
    os.mkdir(image_dir)
    print "Directory " + image_dir +  " Created " 
except:
    print "Directory " + image_dir +  " already exists"

# First and last snapshot numbers
startnum = 400
endnum = 401

# Maximum radius used for getting data
r_max_phys = 20 # kpc

#dataname = implementation+'_'+names[i]+'_data_20_kpc.pickle'

# Now preload the time evolution data
for i,snap_dir in enumerate(snap_dirs):
	compile_dust_data(snap_dir, foutname=dataname, mask=True, overwrite=True, cosmological=False, r_max=r_max_phys, startnum=startnum, endnum=endnum, implementation=implementation)

# Plot precompiled data
DZ_vs_time(dataname=dataname, data_dir='data/', time=True, cosmological=False, foutname=image_dir+'DZ_vs_time.png')

all_data_vs_time(dataname=dataname, data_dir='data/', time=True, cosmological=False, foutname=image_dir+'all_data_vs_time.png')