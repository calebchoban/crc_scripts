#from dust_plots import *
from astropy.table import Table
import os
import subprocess
from gizmo import *

import sys
sys.path.insert(1, "./")



main_dir = '/oasis/tscc/scratch/cchoban/non_cosmological_runs/Species/'
names = ['fiducial_model']
snap_dirs = [main_dir + i + '/output/' for i in names] 
labels = ['Fiducial']
image_dir = './non_cosmo_species_images/'
sub_dir = 'compare_snapshots/' # subdirectory 

implementation = 'species'

cosmological = False

Tcut = 300

# First create ouput directory if needed
try:
    # Create target Directory
    os.mkdir(image_dir)
    print "Directory " + image_dir +  " Created " 
except:
    print "Directory " + image_dir +  " already exists"
try:
    # Create target Directory
    os.mkdir(image_dir + sub_dir)
    print "Directory " + image_dir + sub_dir + " Created " 
except:
    print "Directory " + image_dir + sub_dir + " already exists"


# List of snapshots to compare
snaps = [300]

# Maximum radius, disk, height, and disk orientation used for getting data
r_max_phys = 20 # kpc
disk_height = 2 # kpc
Lz_hat = [0.,0.,1.] # direction of disk

mask = 'halo'

for i, num in enumerate(snaps):
	print(num)
	galaxies = []; r_maxes = []; Lz_hats = []; disk_heights = [];
	for j,snap_dir in enumerate(snap_dirs):
		name = names[j]
		print(name)

		sp = loadsnap(snap_dir, num, cosmological=cosmological, periodic_bound_fix=True)
		halo = sp.loadhalo()
		print vars(halo)


	exit()
	#DZ_vs_params(['r','nH'], [[0,20],[1E-2,1E3]], snaps, mask='disk', bin_nums=50, time=False, depletion=False, \
	#            labels=None, foutname='DZ_vs_param.png', std_bars=True, style='color', log=True, include_obs=True)