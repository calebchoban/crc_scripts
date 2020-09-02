from readsnap import readsnap
from dust_plots import *
from astropy.table import Table
import os
import subprocess




###############################################################################
# Plot analytical dust creation for given stellar population
###############################################################################

Z_list = [1,0.008/0.02,0.001/0.02]
data_dirc = './dust_yields'
dust_species = ['carbon','silicates+']
compare_dust_creation(Z_list, dust_species, data_dirc, FIRE_ver=2, transition_age = 0.03753)
exit()

###############################################################################
# Plot D/Z evolution over time
###############################################################################

image_dir = './non_cosmo_images/'
sub_dir = 'time_evolution/' # subdirectory 

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
endnum = 380

# Maximum radius used for getting data
r_max = 20 # kpc
disk_height = 2 # kpc
Lz_hat = [0.,0.,1.] # direction of disk


###############################################################################
# Species Implementation w/ creation efficienc variations
###############################################################################


main_dir = '/oasis/tscc/scratch/cchoban/non_cosmological_runs/Species/'
names = ['fiducial_model','elem_creation_eff','enhanced_SNe','enhanced_AGB']
labels = ['Fiducial','Elem. Creation Eff.','Enhanced SNe','Enhanced AGB']
implementation = 'species'

cosmological = False

# Now preload the time evolution data
snap_dirs = [main_dir + i + '/output/' for i in names] 

data_names = []
for i,snap_dir in enumerate(snap_dirs):
	name = names[i]
	print(name)
	dataname = implementation+'_'+name+'_data_'+str(r_max)+'_kpc_2_height.pickle'
	data_names += [dataname]
	compile_dust_data(snap_dir, foutname=dataname, mask=True, overwrite=False, cosmological=cosmological, r_max=r_max, Lz_hat=Lz_hat, disk_height=disk_height, startnum=startnum, endnum=endnum, implementation=implementation)

# Now plot a comparison of each of the runs
# dust_data_vs_time(['DZ','source'], [[0,1.],[1E-2,1.1]], implementation=implementation, datanames=data_names, data_dir='data/', foutname='creation_spec_dust_data_vs_time.pdf', \
# 	                     labels=labels, time=True, cosmological=cosmological, log=True, std_bars=False)
# dust_data_vs_time(['DZ','species'], [[0,1.],[0,1]], implementation=implementation, datanames=data_names, data_dir='data/', foutname='creation_spec_dust_comp_data_vs_time.pdf', \
# 	                     labels=labels, time=True, cosmological=cosmological, log=True, std_bars=False)
dust_data_vs_time(['DZ','source', 'species'], [[0,1.],[1E-2,1.1],[0,1.]], implementation=implementation, datanames=data_names, data_dir='data/', foutname='creation_spec_all_data_vs_time.pdf', \
	                     labels=labels, time=True, cosmological=cosmological, log=True, std_bars=False)


###############################################################################
# Elemental Implementation w/ creation efficienc variations
###############################################################################


main_dir = '/oasis/tscc/scratch/cchoban/non_cosmological_runs/Elemental/'
names = ['fiducial_model','species_creation_eff','decreased_stellar']
labels = ['Fiducial','Spec. Creation Eff.','Decreased Stardust']
implementation = 'elemental'


# Now preload the time evolution data
snap_dirs = [main_dir + i + '/output/' for i in names] 

data_names = []
for i,snap_dir in enumerate(snap_dirs):
	name = names[i]
	print(name)
	dataname = implementation+'_'+name+'_data_'+str(r_max)+'_kpc_2_height.pickle'
	data_names += [dataname]
	compile_dust_data(snap_dir, foutname=dataname, mask=True, overwrite=False, cosmological=cosmological, r_max=r_max, Lz_hat=Lz_hat, disk_height=disk_height, startnum=startnum, endnum=endnum, implementation=implementation)

# Now plot a comparison of each of the runs
# dust_data_vs_time(['DZ','source'], [[0,1.],[1E-2,1.1]], implementation=implementation, datanames=data_names, data_dir='data/', foutname='creation_elem_dust_data_vs_time.pdf', \
#                      labels=labels, time=True, cosmological=cosmological, log=True, std_bars=False)
# dust_data_vs_time(['DZ','species'], [[0,1.],[0,1]], implementation=implementation, datanames=data_names, data_dir='data/', foutname='creation_elem_dust_comp_data_vs_time.pdf', \
# 	                     labels=labels, time=True, cosmological=cosmological, log=True, std_bars=False)
dust_data_vs_time(['DZ','source', 'species'], [[0,1.],[1E-2,1.1],[0,1.]], implementation=implementation, datanames=data_names, data_dir='data/', foutname='creation_elem_all_data_vs_time.pdf', \
	                     labels=labels, time=True, cosmological=cosmological, log=True, std_bars=False)


###############################################################################
# Species Implementation w/ modifications that produced the largest changes
###############################################################################


main_dir = '/oasis/tscc/scratch/cchoban/non_cosmological_runs/Species/'
names = ['fiducial_model','no_temp_cutoff','enhanced_acc','enhanced_dest']
labels = ['Fiducial','No Temp. Cutoff','Enhanced Acc.', 'Enhanced Dest.']
implementation = 'species'

cosmological = False

# Now preload the time evolution data
snap_dirs = [main_dir + i + '/output/' for i in names] 

data_names = []
for i,snap_dir in enumerate(snap_dirs):
	name = names[i]
	print(name)
	dataname = implementation+'_'+name+'_data_'+str(r_max)+'_kpc_2_height.pickle'
	data_names += [dataname]
	compile_dust_data(snap_dir, foutname=dataname, mask=True, overwrite=False, cosmological=cosmological, r_max=r_max, Lz_hat=Lz_hat, disk_height=disk_height, startnum=startnum, endnum=endnum, implementation=implementation)

# Now plot a comparison of each of the runs
# dust_data_vs_time(['DZ','source'], [[0,1.],[1E-2,1.1]], implementation=implementation, datanames=data_names, data_dir='data/', foutname='acc_spec_dust_data_vs_time.pdf', \
# 	                     labels=labels, time=True, cosmological=cosmological, log=True, std_bars=False)
# dust_data_vs_time(['DZ','species'], [[0,1.],[0,1]], implementation=implementation, datanames=data_names, data_dir='data/', foutname='acc_spec_dust_comp_data_vs_time.pdf', \
# 	                     labels=labels, time=True, cosmological=cosmological, log=True, std_bars=False)
dust_data_vs_time(['DZ','source', 'species'], [[0,1.],[1E-2,1.1],[0,1.]], implementation=implementation, datanames=data_names, data_dir='data/', foutname='acc_spec_all_data_vs_time.pdf', \
	                     labels=labels, time=True, cosmological=cosmological, log=True, std_bars=False)


###############################################################################
# Elemental Implementation w/ modifications that produced the largest changes
###############################################################################


main_dir = '/oasis/tscc/scratch/cchoban/non_cosmological_runs/Elemental/'
names = ['fiducial_model','decreased_acc','enhanced_dest']
labels = ['Fiducial','Decreased Acc.','Enhanced Dest.']
implementation = 'elemental'


# Now preload the time evolution data
snap_dirs = [main_dir + i + '/output/' for i in names] 

data_names = []
for i,snap_dir in enumerate(snap_dirs):
	name = names[i]
	print(name)
	dataname = implementation+'_'+name+'_data_'+str(r_max)+'_kpc_2_height.pickle'
	data_names += [dataname]
	compile_dust_data(snap_dir, foutname=dataname, mask=True, overwrite=False, cosmological=cosmological, r_max=r_max, Lz_hat=Lz_hat, disk_height=disk_height, startnum=startnum, endnum=endnum, implementation=implementation)

# Now plot a comparison of each of the runs
# dust_data_vs_time(['DZ','source'], [[0,1.],[1E-2,1.1]], implementation=implementation, datanames=data_names, data_dir='data/', foutname='acc_elem_dust_data_vs_time.pdf', \
#                      labels=labels, time=True, cosmological=cosmological, log=True, std_bars=False)
# dust_data_vs_time(['DZ','species'], [[0,1.],[0,1]], implementation=implementation, datanames=data_names, data_dir='data/', foutname='acc_elem_dust_comp_data_vs_time.pdf', \
# 	                     labels=labels, time=True, cosmological=cosmological, log=True, std_bars=False)
dust_data_vs_time(['DZ','source', 'species'], [[0,1.],[1E-2,1.1],[0,1.]], implementation=implementation, datanames=data_names, data_dir='data/', foutname='acc_elem_all_data_vs_time.pdf', \
	                     labels=labels, time=True, cosmological=cosmological, log=True, std_bars=False)



###############################################################################
# Plot last snapshot D/Z values vs observations
###############################################################################



snap_dirs = []
main_dir = '/oasis/tscc/scratch/cchoban/non_cosmological_runs/Species/'
names = ['fiducial_model','extra_O']
snap_dirs += [main_dir + i + '/output/' for i in names] 
labels = ['Species','Species w/ O']
main_dir = '/oasis/tscc/scratch/cchoban/non_cosmological_runs/Elemental/'
names = ['fiducial_model','decreased_acc']
snap_dirs += [main_dir + i + '/output/' for i in names] 
labels += ['Elemental', 'Elemental Low Acc.']

implementations = ['species','elemental']
t_ref_factors = [1,1]

image_dir = './non_cosmo_species_images/'
sub_dir = 'compare_snapshots/' # subdirectory 


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
r_max_phys = 40 # kpc
disk_height = 4 # kpc
Lz_hat = [0.,0.,1.] # direction of disk

for i, num in enumerate(snaps):
	print(num)
	Gas_snaps = []; Star_snaps = []; Headers = []; masks = []; centers = []; r_maxes = []; Lz_hats = []; disk_heights = []; Rds = [];
	for j,snap_dir in enumerate(snap_dirs):
		print snap_dir

		H = readsnap(snap_dir, num, 0, header_only=1, cosmological=cosmological)
		Headers += [H]
		G = readsnap(snap_dir, num, 0, cosmological=cosmological)
		Gas_snaps += [G]
		S = readsnap(snap_dir, num, 4, cosmological=cosmological)
		# Need to remember the stars in the inital conditions
		S1 = readsnap(snap_dir, num, 2, cosmological=cosmological)
		S2 = readsnap(snap_dir, num, 3, cosmological=cosmological)
		for key in ['m','p']:
			S[key] = np.append(S[key],S1[key],axis=0)
			S[key] = np.append(S[key],S2[key],axis=0)
		Star_snaps += [S]



		# Since this is a shallow copy, this fixes G['p'] as well
		coords = G['p']
		# Recenter coords at center of periodic box
		boxsize = H['boxsize']
		mask1 = coords > boxsize/2; mask2 = coords <= boxsize/2
		# This also changes G['p'] as well
		coords[mask1] -= boxsize/2; coords[mask2] += boxsize/2; 
		center = np.average(coords, weights = G['m'], axis = 0)
		centers += [center]


		coords = S['p']
		# Recenter coords at center of periodic box
		mask1 = coords > boxsize/2; mask2 = coords <= boxsize/2
		# This also changes G['p'] as well
		coords[mask1] -= boxsize/2; coords[mask2] += boxsize/2; 


		Rds += [calc_stellar_Rd(S, center, r_max_phys, Lz_hat=Lz_hat, disk_height=disk_height, bin_nums=30)]

		r_maxes += [r_max_phys]
		disk_heights += [disk_height]
		Lz_hats += [Lz_hat]

	DZ_vs_params(['nH'], [[1E-2,1000]], Gas_snaps, Headers, centers, r_maxes, Lz_list = Lz_hats, height_list = disk_heights, bin_nums=40, time=False, depletion=False, \
				cosmological=False, labels=labels, foutname='DZ_vs_nH.pdf', std_bars=True, style='color', log=False, include_obs=True)

	DZ_vs_params(['r'], [[0,20]], Gas_snaps, Headers, centers, r_maxes, Lz_list = Lz_hats, height_list = disk_heights, bin_nums=40, time=False, depletion=False, \
				cosmological=False, labels=labels, foutname='S12_DZ_vs_radius.pdf', std_bars=True, style='color', log=False, include_obs=True, Rd=Rds, CO_opt='S12')

	observed_DZ_vs_param(['gas'], [[1,100]], Gas_snaps, Headers, centers, r_maxes, Lz_list = Lz_hats, height_list = disk_heights, bin_nums=40, time=False, depletion=False, \
				cosmological=False, labels=labels, foutname='S12_obs_DZ_vs_surf.pdf', std_bars=True, style='color', log=False, include_obs=True, CO_opt='S12')

	DZ_vs_params(['r'], [[0,20]], Gas_snaps, Headers, centers, r_maxes, Lz_list = Lz_hats, height_list = disk_heights, bin_nums=40, time=False, depletion=False, \
				cosmological=False, labels=labels, foutname='B13_DZ_vs_radius.pdf', std_bars=True, style='color', log=False, include_obs=True, Rd=Rds, CO_opt='B13')
	
	observed_DZ_vs_param(['gas'], [[1,100]], Gas_snaps, Headers, centers, r_maxes, Lz_list = Lz_hats, height_list = disk_heights, bin_nums=40, time=False, depletion=False, \
				cosmological=False, labels=labels, foutname='B13_obs_DZ_vs_surf.pdf', std_bars=True, style='color', log=False, include_obs=True, CO_opt='B13')

	elems = ['Mg','Si','Fe','O','C']
	elem_depletion_vs_dens(elems, Gas_snaps, Headers, centers, r_maxes, Lz_list = Lz_hats, height_list = disk_heights, \
			bin_nums=50, time=False, depletion=False, cosmological=False, labels=labels, \
			foutname='obs_elemental_dep_vs_dens.pdf', std_bars=True, style='color', log=True, include_obs=True)