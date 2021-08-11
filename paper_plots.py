import os
from gizmo_library.time_evolution import Dust_Evo
from gizmo import *
from dust_plots import *


# First setup directory for all the plots
plot_dir = './dust_plots/'

# First create ouput directory if needed
try:
	# Create target Directory
	os.mkdir(plot_dir)
	print("Directory " + plot_dir +  " Created ")
except:
	print("Directory " + plot_dir +  " already exists")


###############################################################################
# Plot analytical dust creation for given stellar population
###############################################################################

Z_list = [1,0.008/0.02,0.001/0.02]
data_dirc = './analytic_dust_yields/'
dust_species = ['carbon','silicates+']
compare_dust_creation(Z_list, dust_species, data_dirc, FIRE_ver=2, foutname=plot_dir+'FIRE2_creation_routine_compare.pdf')
compare_dust_creation(Z_list, dust_species, data_dirc, FIRE_ver=3, foutname=plot_dir+'FIRE3_creation_routine_compare.pdf')

###############################################################################
# FIRE-2 vs FIRE-3 Metal Returns
###############################################################################

Z=1
elems =['C','O','Mg','Si','Fe']
compare_FIRE_metal_yields(Z, elems, foutname=plot_dir+'FIRE_yields_comparison.pdf')

###############################################################################
# Plot D/Z evolution over time
###############################################################################

# Here is the all the main parameters for the D/Z evolution plots

# First and last snapshot numbers
startnum = 0
endnum = 300
snap_lims = [startnum,endnum]

# Maximum radius used for getting data
r_max = 20 # kpc
disk_height = 2 # kpc

pb_fix=True
dust_depl=False

config.FIG_XRATIO=1.2 # Make the aspect ratio a little wider

###############################################################################
# Species Implementation w/ creation efficiency variations
###############################################################################


main_dir = '/oasis/tscc/scratch/cchoban/non_cosmological_runs/Species/'
names = ['fiducial_model','elem_creation_eff','enhanced_SNe','enhanced_AGB']
labels = ['Fiducial','Elem. Creation Eff.','Enhanced SNe','Enhanced AGB']
implementation = 'species'

cosmological = False

# Now preload the time evolution data
snap_dirs = [main_dir + i + '/output/' for i in names] 

dust_evo_data = []
for i,snap_dir in enumerate(snap_dirs):
	name = names[i]
	print(name)
	dust_avg = Dust_Evo(snap_dir, snap_lims, cosmological=cosmological, periodic_bound_fix=pb_fix, dust_depl=dust_depl, statistic = 'average', dirc='./time_evo_data/')
	dust_avg.set_disk(id=-1, mode='AHF', hdir=None, rmax=r_max, height=disk_height)
	dust_avg.load()
	dust_avg.save()
	dust_evo_data += [dust_avg]
	

# Now plot a comparison of each of the runs
dust_data_vs_time(['D/Z','source_frac', 'spec_frac'], dust_evo_data, foutname=plot_dir+'creation_spec_all_data_vs_time.pdf',labels=labels, style='color')


###############################################################################
# Elemental Implementation w/ creation efficiency variations
###############################################################################


main_dir = '/oasis/tscc/scratch/cchoban/non_cosmological_runs/Elemental/'
names = ['fiducial_model','species_creation_eff','decreased_stellar']
labels = ['Fiducial','Spec. Creation Eff.','Decreased Stardust']
implementation = 'elemental'


# Now preload the time evolution data
snap_dirs = [main_dir + i + '/output/' for i in names] 

dust_evo_data = []
for i,snap_dir in enumerate(snap_dirs):
	name = names[i]
	print(name)
	dust_avg = Dust_Evo(snap_dir, snap_lims, cosmological=cosmological, periodic_bound_fix=pb_fix, dust_depl=dust_depl, statistic = 'average', dirc='./time_evo_data/')
	dust_avg.set_disk(id=-1, mode='AHF', hdir=None, rmax=r_max, height=disk_height)
	dust_avg.load()
	dust_avg.save()
	dust_evo_data += [dust_avg]

# Now plot a comparison of each of the runs
dust_data_vs_time(['D/Z','source_frac', 'spec_frac'], dust_evo_data, foutname=plot_dir+'creation_elem_all_data_vs_time.pdf',labels=labels, style='color')


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

dust_evo_data = []
for i,snap_dir in enumerate(snap_dirs):
	name = names[i]
	print(name)
	dust_avg = Dust_Evo(snap_dir, snap_lims, cosmological=cosmological, periodic_bound_fix=pb_fix, dust_depl=dust_depl, statistic = 'average', dirc='./time_evo_data/')
	dust_avg.set_disk(id=-1, hdir=None, rmax=r_max, height=disk_height)
	dust_avg.load()
	dust_avg.save()
	dust_evo_data += [dust_avg]

# Now plot a comparison of each of the runs
dust_data_vs_time(['D/Z','source_frac', 'spec_frac'], dust_evo_data, foutname=plot_dir+'acc_spec_all_data_vs_time.pdf',labels=labels, style='color')


###############################################################################
# Elemental Implementation w/ modifications that produced the largest changes
###############################################################################


main_dir = '/oasis/tscc/scratch/cchoban/non_cosmological_runs/Elemental/'
names = ['fiducial_model','decreased_acc','enhanced_dest']
labels = ['Fiducial','Decreased Acc.','Enhanced Dest.']
implementation = 'elemental'


# Now preload the time evolution data
snap_dirs = [main_dir + i + '/output/' for i in names] 

dust_evo_data = []
for i,snap_dir in enumerate(snap_dirs):
	name = names[i]
	print(name)
	dust_avg = Dust_Evo(snap_dir, snap_lims, cosmological=cosmological, periodic_bound_fix=pb_fix, dust_depl=dust_depl, statistic = 'average', dirc='./time_evo_data/')
	dust_avg.set_disk(id=-1, hdir=None, rmax=r_max, height=disk_height)
	dust_avg.load()
	dust_avg.save()
	dust_evo_data += [dust_avg]

# Now plot a comparison of each of the runs
dust_data_vs_time(['D/Z','source_frac', 'spec_frac'], dust_evo_data, foutname=plot_dir+'acc_elem_all_data_vs_time.pdf',labels=labels, style='color')


config.FIG_XRATIO=1. # Reset to normal

###############################################################################
# Plot last snapshot D/Z values vs observations for optional dust species physics 
###############################################################################


snap_dirs = []
main_dir ='/oasis/tscc/scratch/cchoban/non_cosmo/Species/'
names = ['fiducial','O_reservoir', 'nano_Fe']
snap_dirs += [main_dir + i + '/output/' for i in names] 
labels = ['Species','Species w/ O', 'Species w/ O & Fe']
main_dir = '/oasis/tscc/scratch/cchoban/non_cosmo/Elemental/'
names = ['reduced_acc']
snap_dirs += [main_dir + i + '/output/' for i in names] 
labels += ['Elemental Low Acc.']

implementations = ['species','species','elemental']

cosmological = False

# List of snapshots to compare
snaps = [300]

# Maximum radius, disk, height, and disk orientation used for getting data
r_max = 20 # kpc
disk_height = 2 # kpc

for i, num in enumerate(snaps):
	print(num)
	galaxies = []
	for j,snap_dir in enumerate(snap_dirs):
		print(snap_dir)
		galaxy = load_disk(snap_dir, num, cosmological=cosmological, id=-1, mode='AHF', hdir=None, periodic_bound_fix=pb_fix, rmax=r_max, height=disk_height)
		galaxies += [galaxy]

	plot_prop_vs_prop(['nH'], ['D/Z'], galaxies, bin_nums=40, labels=labels, foutname=plot_dir+'DZ_vs_nH.pdf', std_bars=True, style='color-linestyle', include_obs=True)

	plot_obs_prop_vs_prop(['sigma_gas','r'], ['D/Z','D/Z'], galaxies, pixel_res=2, bin_nums=40, labels=labels, foutname=plot_dir+'B13_obs_DZ_vs_surf.pdf', \
						std_bars=True, style='color-linestyle', include_obs=True)

	# Need larger font size for this plot
	config.LARGE_FONT       = 40
	config.EXTRA_LARGE_FONT = 56
	elems = ['Mg','Si','Fe','O','C']
	plot_elem_depletion_vs_prop(elems, 'nH', galaxies, bin_nums=50, labels=labels, foutname=plot_dir+'obs_elemental_dep_vs_dens.pdf', \
						std_bars=True, style='color-linestyle', include_obs=True)
	config.LARGE_FONT       = 30
	config.EXTRA_LARGE_FONT = 36


###############################################################################
# Plot last snapshot comparisons for FIRE-2/3
###############################################################################

snap_dirs = []
main_dir ='/oasis/tscc/scratch/cchoban/new_gizmo/'
names = ['FIRE-2', 'FIRE-3_cool']
snap_dirs += [main_dir + i + '/output/' for i in names] 
labels = ['FIRE-2','FIRE-3']

implementations = ['species','species']

cosmological = False

# List of snapshots to compare
snaps = [300]

# Maximum radius, disk, height, and disk orientation used for getting data
r_max = 20 # kpc
disk_height = 2 # kpc

for i, num in enumerate(snaps):
	print(num)
	galaxies = []
	for j,snap_dir in enumerate(snap_dirs):
		print(snap_dir)
		galaxy = load_disk(snap_dir, num, cosmological=cosmological, id=-1, mode='AHF', hdir=None, periodic_bound_fix=pb_fix, rmax=r_max, height=disk_height)
		galaxies += [galaxy]


	config.FIG_XRATIO=.85 # Make the aspect ratio more 1:1
	config.PROP_INFO['nH'][1]=[1.1E-3, 0.9E3] # Increase the density range
	config.PROP_INFO['T'][1]=[1.1E1, 2E6] # Increase the temp range
	binned_phase_plot('M_gas', galaxies, bin_nums=250, labels=labels, color_map='plasma', foutname=plot_dir+"FIRE2-3_phase.pdf")
	binned_phase_plot('D/Z', galaxies, bin_nums=250, labels=labels, color_map='magma', foutname=plot_dir+"FIRE2-3_DZ_phase.pdf")
	config.FIG_XRATIO=1.
	config.PROP_INFO['nH'][1]=[1.1E-2, 0.9E3]
	config.PROP_INFO['T'][1]=[1.1E1, 0.9E5]

	plot_prop_vs_prop(['nH'], ['D/Z'], galaxies, bin_nums=40, labels=labels, foutname=plot_dir+'FIRE2-3_DZ_vs_nH.pdf', std_bars=True, style='color-linestyle', include_obs=True)

	plot_obs_prop_vs_prop(['sigma_gas','r'], ['D/Z','D/Z'], galaxies, pixel_res=2, bin_nums=40, labels=labels, foutname=plot_dir+'FIRE2-3_B13_obs_DZ_vs_surf.pdf', \
						std_bars=True, style='color-linestyle', include_obs=True)

	# Need larger font size for this plot
	config.LARGE_FONT       = 40
	config.EXTRA_LARGE_FONT = 56
	elems = ['Mg','Si','Fe','O','C']
	plot_elem_depletion_vs_prop(elems, 'nH', galaxies, bin_nums=50, labels=labels, foutname=plot_dir+'FIRE2-3_obs_elemental_dep_vs_dens.pdf', \
						std_bars=True, style='color-linestyle', include_obs=True)
	config.LARGE_FONT       = 30
	config.EXTRA_LARGE_FONT = 36

	dmol_vs_props(['fH2','fMC'], ['nH', 'T'], galaxies, bin_nums=50, labels=labels, foutname=plot_dir+'FIRE2-3_fMC.pdf', std_bars=True)
	dmol_vs_props(['fMC','CinCO'], ['nH', 'T'], galaxies, bin_nums=50, labels=labels, foutname=plot_dir+'FIRE2-3_CinCO.pdf', std_bars=True)


###############################################################################
# Plot comparisons for sub-resolved fMC routine
###############################################################################

# Directory of snap file
snap_dirs = ['/oasis/tscc/scratch/cchoban/non_cosmo/fMC_test/NH2_0.5/output/','/oasis/tscc/scratch/cchoban/non_cosmo/fMC_test/NH2_1.0/output/',
				'/oasis/tscc/scratch/cchoban/non_cosmo/fMC_test/NH2_2.0/output/']

# Label for test plots
labels = [r'$N_{\rm H_2}^{\rm crit}=0.5\times10^{21}$ cm$^{-3}$',r'$N_{\rm H_2}^{\rm crit}=1.0\times10^{21}$ cm$^{-3}$',
			r'$N_{\rm H_2}^{\rm crit}=2.0\times10^{21}$ cm$^{-3}$']

cosmological = False

# Snapshot to check
snap_num = 19

galaxies = []
for j,snap_dir in enumerate(snap_dirs):
	print(snap_dir)
	galaxy = load_disk(snap_dir, snap_num, cosmological=cosmological, periodic_bound_fix=pb_fix, rmax=r_max, height=disk_height)
	galaxies += [galaxy]

dmol_vs_props(['fH2','fMC'], ['nH', 'T'], galaxies, bin_nums=50, labels=labels, foutname=plot_dir+'NH2_crit_variation.pdf', std_bars=True)