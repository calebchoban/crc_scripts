from readsnap import readsnap
from dust_plots import *
from astropy.table import Table
import os
import subprocess


main_dir = '/oasis/tscc/scratch/cchoban/non_cosmological_runs/Species/'
names_list = [['fiducial_model','no_temp_cutoff','enhanced_acc','extra_O'],
			  ['fiducial_model','elem_creation_eff','enhanced_SNe','enhanced_AGB'],
			  ['fiducial_model','no_CO','no_AGB_creation','no_SNe_creation'],
			  ['fiducial_model','enhanced_dest','no_sputtering']]
labels_list = [['Fiducial','No Temp. Cutoff','Enhanced Acc.','Enhanced O'],
			   ['Fiducial','Elem. Creation Eff.','Enhanced SNe','Enhanced AGB'],
			   ['Fiducial','No CO','No SNe Dust','No AGB Dust'],
			   ['Fiducial','Enhanced Dest.','No Sputtering']]
extra_names = ['largest_change','creation','turnoff','dest']

image_dir = './non_cosmo_images/'
sub_dir = 'time_evolution/' # subdirectory 

implementation = 'species'

cosmological = False

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


# First and last snapshot numbers
startnum = 0
endnum = 10

# Maximum radius used for getting data
r_max = 20 # kpc
disk_height = 2 # kpc
Lz_hat = [0.,0.,1.] # direction of disk

# Now preload the time evolution data
for k in range(len(extra_names)):
	extra_name = extra_names[k]
	names = names_list[k]
	labels = labels_list[k]
	snap_dirs = [main_dir + i + '/output/' for i in names] 

	data_names = []
	for i,snap_dir in enumerate(snap_dirs):
		name = names[i]
		print(name)
		dataname = implementation+'_'+name+'_data_'+str(r_max)+'_kpc_2_height.pickle'
		data_names += [dataname]
		compile_dust_data(snap_dir, foutname=dataname, mask=True, overwrite=True, cosmological=cosmological, r_max=r_max, Lz_hat=Lz_hat, disk_height=disk_height, startnum=startnum, endnum=endnum, implementation=implementation)

		# Plot precompiled data
		DZ_vs_time(dataname=dataname, data_dir='data/', time=True, cosmological=cosmological, foutname=image_dir+implementation+'_'+name+'_DZ_vs_time.png', log=False)

		all_data_vs_time(dataname=dataname, data_dir='data/', time=True, cosmological=cosmological, foutname=image_dir+implementation+'_'+name+'_all_data_vs_time.png', log=False)

	# Now plot a comparison of each of the runs
	compare_runs_vs_time(datanames=data_names, data_dir='data/', foutname=image_dir+implementation+'_'+extra_name+'_compare_runs_vs_time.png', labels=labels, cosmological=cosmological, log=False)


main_dir = '/oasis/tscc/scratch/cchoban/non_cosmological_runs/Elemental/'
names_list = [['fiducial_model','species_creation_eff','enhanced_dest','decreased_acc'],
			  ['fiducial_model','no_AGB_creation','no_SNe_creation','no_CO'],
			  ['fiducial_model','no_sputtering','decreased_stellar',]]
labels_list = [['Fiducial','Spec. Creation Eff.','Enhanced Dest.','Decreased Acc.'],
			   ['Fiducial','No AGB Dust','No SNe Dust','No CO'],
			   ['Fiducial','No Sputtering','Decreased Stardust']]
extra_names = ['largest_change','creation','misc']


implementation = 'elemental'


# Now preload the time evolution data
for k in range(len(extra_names)):
	extra_name = extra_names[k]
	names = names_list[k]
	labels = labels_list[k]
	snap_dirs = [main_dir + i + '/output/' for i in names] 

	data_names = []
	for i,snap_dir in enumerate(snap_dirs):
		name = names[i]
		print(name)
		dataname = implementation+'_'+name+'_data_'+str(r_max)+'_kpc_2_height.pickle'
		data_names += [dataname]
		compile_dust_data(snap_dir, foutname=dataname, mask=True, overwrite=True, cosmological=cosmological, r_max=r_max, Lz_hat=Lz_hat, disk_height=disk_height, startnum=startnum, endnum=endnum, implementation=implementation)

		# Plot precompiled data
		DZ_vs_time(dataname=dataname, data_dir='data/', time=True, cosmological=cosmological, foutname=image_dir+implementation+'_'+name+'_DZ_vs_time.png', log=False)

		all_data_vs_time(dataname=dataname, data_dir='data/', time=True, cosmological=cosmological, foutname=image_dir+implementation+'_'+name+'_all_data_vs_time.png', log=False)

	# Now plot a comparison of each of the runs
	compare_runs_vs_time(datanames=data_names, data_dir='data/', foutname=image_dir+implementation+'_'+extra_name+'_compare_runs_vs_time.png', labels=labels, cosmological=cosmological, log=False)