from readsnap import readsnap
from dust_plots import *
from astropy.table import Table
import os
import subprocess



main_dir = '/oasis/tscc/scratch/cchoban/non_cosmological_runs/Species/'
names = ['fiducial_model','elem_creation_eff','enhanced_acc','extra_O']
snap_dirs = [main_dir + i + '/output/' for i in names] 
labels = ['Fiducial','Elem. Creation Eff.','Enhanced Acc.','Enhanced O']
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

for i, num in enumerate(snaps):
	print(num)
	Gas_snaps = []; Headers = []; masks = []; centers = []; r_maxes = []; Lz_hats = []; disk_heights = [];
	for j,snap_dir in enumerate(snap_dirs):
		name = names[j]
		print(name)

		H = readsnap(snap_dir, num, 0, header_only=1, cosmological=cosmological)
		Headers += [H]
		G = readsnap(snap_dir, num, 0, cosmological=cosmological)
		Gas_snaps += [G]

		# Since this is a shallow copy, this fixes G['p'] as well
		coords = G['p']
		# Recenter coords at center of periodic box
		boxsize = H['boxsize']
		mask1 = coords > boxsize/2; mask2 = coords <= boxsize/2
		# This also changes G['p'] as well
		coords[mask1] -= boxsize/2; coords[mask2] += boxsize/2; 
		center = np.average(coords, weights = G['m'], axis = 0)
		centers += [center]

		r_maxes += [r_max_phys]
		disk_heights += [disk_height]
		Lz_hats += [Lz_hat]

	DZ_vs_params(['nH','r','fH2'], [[1E-3,1E3],[0,r_max_phys],[0.01,1]], Gas_snaps, Headers, centers, r_maxes, Lz_list = Lz_hats, height_list = disk_heights, bin_nums=50, time=False, depletion=False, \
          cosmological=False, labels=labels, foutname='DZ_vs_param.png', std_bars=True, style='color', log=False, include_obs=True)


	# Make D/Z vs r plot
	#DZ_vs_r(Gas_snaps, Headers, centers, r_maxes, Lz_list = Lz_hats, height_list = disk_heights, bin_nums=50, time=False, \
	#    `  foutname=image_dir+sub_dir+'disk_'+implementation+'_DZ_vs_r_snapshot_%03d.png' % num,labels=labels, \
	#	    cosmological=cosmological, log=False, observation=True)
	#DZ_vs_fH2(Gas_snaps, Headers, centers, r_maxes,  Lz_list=Lz_hats, height_list=disk_heights, bin_nums=50, time=False, \
	#        depletion=False, cosmological=True, labels=labels, foutname=image_dir+sub_dir+'disk_'+implementation+'_DZ_vs_fH2_snapshot_%03d.png' % num, \
	#        std_bars=True, style='color', log=False, observation=True, fH2_min=1E-2, fH2_max=1)
	#nH_vs_fH2(Gas_snaps, Headers, centers, r_maxes,  Lz_list=Lz_hats, height_list=disk_heights, foutname='nH_vs_fH2.png', labels=labels)
	#DZ_pixel_bin('fH2', Gas_snaps, Headers, centers, r_maxes,  Lz_list=Lz_hats, height_list=disk_heights, num_bins=100, observation=True, \
	#			 labels=labels, foutname='DZ_pixel_bin_vs_fH2.png')
	"""
	# Make D/Z vs density plot
	DZ_vs_dens(Gas_snaps,Headers, centers, r_maxes, Lz_list = Lz_hats, height_list = disk_heights, time=True, \
		       foutname=image_dir+sub_dir+'disk_'+implementation+'_compare_DZ_vs_dens_snapshot_%03d.png' % num,labels=labels, \
		       cosmological=cosmological, log=False)
	# Make D/Z vs Z plot
	DZ_vs_Z(Gas_snaps,Headers, centers, r_maxes, Lz_list = Lz_hats, height_list = disk_heights, time=True, Zmin=1E0, Zmax=1e1, \
		    foutname=image_dir+sub_dir+'disk_'+implementation+'_compare_DZ_vs_Z_snapshot_%03d.png' % num,labels=labels, \
		    cosmological=cosmological, log=False)
	"""
	#DZ_vs_all(Gas_snaps,Headers, centers, r_maxes, Lz_list = Lz_hats, height_list = disk_heights, bin_nums=50, time=True, depletion=False, \
	#	      cosmological=cosmological, labels=labels, foutname=image_dir+sub_dir+implementation+'_compare_DZ_vs_all_snapshot_%03d.png' % num, \
	#	      std_bars=True, style='color', nHmin=1E-3, nHmax=1E3, Zmin=1E0, Zmax=1E1, log=False)
	#temp_dist(Gas_snaps, Headers, centers, r_maxes, Lz_list = Lz_hats, height_list = disk_heights, bin_nums=100, time=False, \
	#           cosmological=cosmological, Tmin=1, Tmax=1E6, labels=labels, foutname='compare_temp_dist.png',  style='color')

	#accretion_analysis_plot(Gas_snaps, Headers, centers, r_maxes, Lz_list = Lz_hats, height_list = disk_heights, bin_nums=100, time=False, \
    #       cosmological=cosmological, Tmin=1, Tmax=1E5, Tcut=Tcut, labels=labels, implementation=implementation)
	#binned_phase_plot('DZ', Gas_snaps, Headers, centers, r_maxes, Lz_list = Lz_hats, height_list = disk_heights, bin_nums=100, time=True, \
    #       cosmological=cosmological, Tmin=1, Tmax=1E5, labels=labels, vmin=0, vmax=0.45)

	#surface_dens_vs_radius(Gas_snaps, Headers, centers, r_maxes, Lz_hats, disk_heights, bin_nums=100, time=False, \
    #      cosmological=cosmological, labels=labels, foutname=implementation+'_dust_surface_dens_vs_r_snapshot_%03d.png' % num)

