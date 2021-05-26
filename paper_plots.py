
import os
import subprocess
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
elems =['C','Mg','Si','Fe']
compare_FIRE_metal_yields(Z, elems, foutname=plot_dir+'FIRE_yields_comparison.png')

###############################################################################
# Plot D/Z evolution over time
###############################################################################

# Here is the all the main parameteres for the D/Z evolution plots

# First and last snapshot numbers
startnum = 0
endnum = 300
snap_lims = [startnum,endnum]

# Maximum radius used for getting data
r_max = 20 # kpc
disk_height = 2 # kpc

pb_fix=True
dust_depl=False


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
dust_data_vs_time(['DZ','source_frac', 'spec_frac'], dust_evo_data, foutname=plot_dir+'creation_spec_all_data_vs_time.pdf',labels=labels, style='color')


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
dust_data_vs_time(['DZ','source_frac', 'spec_frac'], dust_evo_data, foutname=plot_dir+'creation_elem_all_data_vs_time.pdf',labels=labels, style='color')


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
	dust_avg.set_disk(id=-1, mode='AHF', hdir=None, rmax=r_max, height=disk_height)
	dust_avg.load()
	dust_avg.save()
	dust_evo_data += [dust_avg]

# Now plot a comparison of each of the runs
dust_data_vs_time(['DZ','source_frac', 'spec_frac'], dust_evo_data, foutname=plot_dir+'acc_spec_all_data_vs_time.pdf',labels=labels, style='color')


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
	dust_avg.set_disk(id=-1, mode='AHF', hdir=None, rmax=r_max, height=disk_height)
	dust_avg.load()
	dust_avg.save()
	dust_evo_data += [dust_avg]

# Now plot a comparison of each of the runs
dust_data_vs_time(['DZ','source_frac', 'spec_frac'], dust_evo_data, foutname=plot_dir+'acc_elem_all_data_vs_time.pdf',labels=labels, style='color')

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

	DZ_vs_params(['nH'], galaxies, bin_nums=40, time=None, labels=labels, foutname=plot_dir+'DZ_vs_nH.pdf', std_bars=True, style='color', include_obs=True)

	observed_DZ_vs_param(['sigma_gas','r'], galaxies, pixel_res=2, bin_nums=40, time=None, labels=labels, foutname=plot_dir+'S12_obs_DZ_vs_surf.pdf', \
						std_bars=True, style='color', include_obs=True, CO_opt='S12')
	
	observed_DZ_vs_param(['sigma_gas','r'], galaxies, pixel_res=2, bin_nums=40, time=None, labels=labels, foutname=plot_dir+'B13_obs_DZ_vs_surf.pdf', \
						std_bars=True, style='color', include_obs=True, CO_opt='B13')

	elems = ['Mg','Si','Fe','O','C']
	elem_depletion_vs_param(elems, 'nH', galaxies, bin_nums=50, time=None, labels=labels, foutname=plot_dir+'obs_elemental_dep_vs_dens.pdf', \
						std_bars=True, style='color', include_obs=True)


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
snaps = [275]

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

		
	binned_phase_plot('m', galaxies, bin_nums=250, labels=labels, color_map='plasma', foutname=plot_dir+"FIRE2-3_phase.png")
	binned_phase_plot('D/Z', galaxies, bin_nums=250, labels=labels, color_map='magma', foutname=plot_dir+"FRE2-3_DZ_phase.png")


	DZ_vs_params(['nH'], galaxies, bin_nums=40, time=None, labels=labels, foutname=plot_dir+'FIRE2-3_DZ_vs_nH.pdf', std_bars=True, style='color', include_obs=True)

	observed_DZ_vs_param(['sigma_gas','r'], galaxies, pixel_res=2, bin_nums=40, time=None, labels=labels, foutname=plot_dir+'FIRE2-3_S12_obs_DZ_vs_surf.pdf', \
						std_bars=True, style='color', include_obs=True, CO_opt='S12')
	
	observed_DZ_vs_param(['sigma_gas','r'], galaxies, pixel_res=2, bin_nums=40, time=None, labels=labels, foutname=plot_dir+'FIRE2-3_B13_obs_DZ_vs_surf.pdf', \
						std_bars=True, style='color', include_obs=True, CO_opt='B13')

	elems = ['Mg','Si','Fe','O','C']
	elem_depletion_vs_param(elems, 'nH', galaxies, bin_nums=50, time=None, labels=labels, foutname=plot_dir+'FIRE2-3_obs_elemental_dep_vs_dens.pdf', \
						std_bars=True, style='color', include_obs=True)