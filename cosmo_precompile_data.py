from readsnap import readsnap
from dust_plots import *
from astropy.table import Table
import os
import subprocess

main_dir = '/oasis/tscc/scratch/cchoban/FIRE_2_0_or_h553_criden1000_noaddm_sggs_dust/'
names = ['Species', 'Elemental', 'Species_no_cutoff']
snap_dirs = [main_dir + i + '/output/' for i in names] 
halo_dirs = [main_dir + i + '/AHF_data/halos/' for i in names] 
image_dir = './cosmo_images/'
halo_name = 'halo_0000000.dat'
set_name = 'routine' # name of set of runs to be compared

cosmological = True
implementations = ['species','elemental','species']


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
r_max = 5 # kpc

data_names = []

for i,snap_dir in enumerate(snap_dirs):
	implementation = implementations[i]
	name = names[i]
	print(name)
	halo_dir = halo_dirs[i]

	# Load in halohistory data for main halo. All values should be in code units
	halo_data = Table.read(halo_dir + halo_name,format='ascii')

	dataname = implementation+'_'+name+'_data_'+str(r_max)+'_kpc.pickle'
	data_names += [dataname]
	compile_dust_data(snap_dir, foutname=dataname, mask=True, overwrite=True, halo_dir=halo_dir+halo_name, cosmological=cosmological, r_max=r_max, startnum=startnum, endnum=endnum, implementation=implementation)

	# Plot precompiled data
	DZ_vs_time(dataname=dataname, data_dir='data/', time=True, cosmological=cosmological, foutname=image_dir+implementation+'_'+name+'_DZ_vs_time.png')

	all_data_vs_time(dataname=dataname, data_dir='data/', time=True, cosmological=cosmological, foutname=image_dir+implementation+'_'+name+'_all_data_vs_time.png')

# Now plot a comparison of each of the runs
compare_runs_vs_time(datanames=data_names, data_dir='data/', foutname=image_dir+set_name+'_compare_runs_vs_time.png', labels=labels, cosmological=cosmological)