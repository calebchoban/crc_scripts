from readsnap import readsnap
from dust_plots import *
from astropy.table import Table
import os
import subprocess


main_dir = '/oasis/tscc/scratch/cchoban/non_cosmological_runs/Elemental/'
names = ['fiducial_model','species_creation_eff','enhanced_dest','decreased_acc']
snap_dirs = [main_dir + i + '/output/' for i in names] 
labels = ['Fiducial','Spec. Creation Eff.','Enhanced Dest.','Decreased Acc.']
image_dir = './non_cosmo_images/'

implementation = 'elemental'

cosmological = False

# First create ouput directory if needed
try:
    # Create target Directory
    os.mkdir(image_dir)
    print("Directory ", image_dir, " Created")
except:
    print("Directory ", image_dir, " already exists")


# First and last snapshot numbers
startnum = 0
endnum = 380

# Maximum radius used for getting data
r_max = 20 # kpc


data_names = []

# Now preload the time evolution data

for i,snap_dir in enumerate(snap_dirs):
	name = names[i]
	print name
	dataname = implementation+'_'+name+'_data_'+str(r_max)+'_kpc.pickle'
	data_names += [dataname]
	compile_dust_data(snap_dir, foutname=dataname, mask=True, overwrite=True, cosmological=cosmological, r_max=r_max, startnum=startnum, endnum=endnum, implementation=implementation)

	# Plot precompiled data
	DZ_vs_time(dataname=dataname, data_dir='data/', time=True, cosmological=cosmological, foutname=image_dir+name+'_DZ_vs_time.png')

	all_data_vs_time(dataname=dataname, data_dir='data/', time=True, cosmological=cosmological, foutname=image_dir+name+'_all_data_vs_time.png')

# Now plot a comparison of each of the runs
compare_runs_vs_time(datanames=data_names, data_dir='data/', foutname=image_dir+'compare_runs_vs_time.png', labels=labels, cosmological=cosmological)