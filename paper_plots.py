import os

import matplotlib.pyplot as plt

from gizmo_library.time_evolution import Dust_Evo
from gizmo_library.sightline import Sight_Lines
from gizmo import *
from dust_plots import *
import analytical_models.dust_accretion as dust_acc


# First setup directory for all the plots
plot_dir = './paper_plots/'

# First create output directory if needed
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


main_dir = '/work/06185/tg854841/frontera/non_cosmo/Species/'
names = ['fiducial','elem_creation_eff','enhanced_SNe','enhanced_AGB']
labels = ['Fiducial','Elem. Creation Eff.','Enhanced SNe','Enhanced AGB']

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

main_dir = '/work/06185/tg854841/frontera/non_cosmo/Elemental/'
names = ['fiducial','spec_creation_eff','decreased_AGB','decreased_SNe']
labels = ['Fiducial','Spec. Creation Eff.','Decreased AGB','Decreased SNe']

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


main_dir = '/work/06185/tg854841/frontera/non_cosmo/Species/'
names = ['fiducial','enhanced_acc','no_temp_cutoff','enhanced_dest']
labels = ['Fiducial','Enhanced Acc.','No Temp. Cutoff','Enhanced Dest.']


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

main_dir = '/work/06185/tg854841/frontera/non_cosmo/Elemental/'
names = ['fiducial','enhanced_acc','enhanced_dest']
labels = ['Fiducial','Enhanced Acc.','Enhanced Dest.']



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

main_dir = '/work/06185/tg854841/frontera/non_cosmo/Species/'

names = ['fiducial','O_reservoir', 'nano_Fe']
snap_dirs += [main_dir + i + '/output/' for i in names] 
labels = ['Species','Species w/ O', 'Species w/ O & Fe']
main_dir = '/oasis/tscc/scratch/cchoban/non_cosmo/Elemental/'

main_dir = '/work/06185/tg854841/frontera/non_cosmo/Elemental/'

names = ['fiducial']
snap_dirs += [main_dir + i + '/output/' for i in names] 
labels += ['Elemental']

implementations = ['species','species','species','elemental']

cosmological = False

# List of snapshots to compare
snaps = [300]

# Number of sightlines from simulation
N_sightlines = 10000

# Maximum radius, disk, height, and disk orientation used for getting data
r_max = 20 # kpc
disk_height = 2 # kpc

for i, num in enumerate(snaps):
	print(num)
	galaxies = []
	for j,snap_dir in enumerate(snap_dirs):
		print(snap_dir)
		galaxy = load_disk(snap_dir, num, cosmological=cosmological, mode=None, hdir=None, periodic_bound_fix=pb_fix, rmax=r_max, height=disk_height)
		galaxy.set_disk()
		galaxies += [galaxy]

	plot_prop_vs_prop(['nH'], ['D/Z'], galaxies, bin_nums=40, labels=labels, foutname=plot_dir+'DZ_vs_nH.pdf', std_bars=True, style='color-linestyle', include_obs=True)
	plot_prop_vs_prop(['nH_neutral'], ['D/Z'], galaxies, bin_nums=40, labels=labels, foutname=plot_dir+'DZ_vs_nH_neutral.pdf', std_bars=True, style='color-linestyle', include_obs=True)

	config.SMALL_FONT=20
	plot_obs_prop_vs_prop(['sigma_gas','r'], ['D/Z','D/Z'], galaxies, pixel_res=2, bin_nums=40, labels=labels, foutname=plot_dir+'B13_obs_DZ_vs_surf.pdf', \
						std_bars=True, style='color-linestyle', include_obs=True)
	plot_obs_prop_vs_prop(['sigma_gas_neutral','r'], ['D/Z','D/Z'], galaxies, pixel_res=2, bin_nums=40, labels=labels, foutname=plot_dir+'B13_obs_DZ_vs_surf_neutral.pdf', \
						std_bars=True, style='color-linestyle', include_obs=True)
	config.SMALL_FONT=22

	# Need larger font size for this plot
	config.LARGE_FONT       = 40
	config.EXTRA_LARGE_FONT = 56
	elems = ['Mg','Si','Fe','O','C']
	plot_elem_depletion_vs_prop(elems, 'nH', galaxies, bin_nums=50, labels=labels, foutname=plot_dir+'obs_elemental_dep_vs_nH.pdf', \
						std_bars=True, style='color-linestyle', include_obs=True)
	plot_elem_depletion_vs_prop(elems, 'nH_neutral', galaxies, bin_nums=50, labels=labels, foutname=plot_dir+'obs_elemental_dep_vs_nH_neutral.pdf', \
						std_bars=True, style='color-linestyle', include_obs=True)

	sightline_data_files = []
	for j,snap_dir in enumerate(snap_dirs):
		sight_lines = Sight_Lines(snap_dir, num, cosmological=0, dust_impl=implementations[j], periodic_bound_fix=True, dirc='./', name=None)
		sight_lines.create_sightlines(N=N_sightlines, radius=config.SOLAR_GAL_RADIUS, dist_lims=[0.1,1.9])
		sightline_data_files += [sight_lines.name]

	config.FIG_XRATIO = 1.1

	plot_sightline_depletion_vs_prop(elems, 'NH_neutral', sightline_data_files, bin_data=True, bin_nums=20, labels=labels, foutname=plot_dir+'binned_sightline_depl_vs_NH.pdf', \
							 std_bars=True, style='color-linestyle', include_obs=True)
	plot_sightline_depletion_vs_prop(elems, 'NH_neutral', sightline_data_files, bin_data=False, labels=labels, foutname=plot_dir+'raw_sightline_depl_vs_NH.pdf', \
							 std_bars=True, style='color-linestyle', include_obs=True)
	config.LARGE_FONT       = 30
	config.EXTRA_LARGE_FONT = 36
	config.FIG_XRATIO = 1.


###############################################################################
# Plot last snapshot D/Z values vs observations for Species with and without Coulomb enhancing.
###############################################################################


snap_dirs = []
main_dir ='/oasis/tscc/scratch/cchoban/non_cosmo/Species/'

main_dir = '/work/06185/tg854841/frontera/non_cosmo/Species/'

names = ['nano_Fe','full_no_Coulomb_enh']
snap_dirs += [main_dir + i + '/output/' for i in names]
labels = ['With Coulomb/CO','Without Coulomb/CO']

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
		galaxy = load_disk(snap_dir, num, cosmological=cosmological, mode=None, hdir=None, periodic_bound_fix=pb_fix, rmax=r_max, height=disk_height)
		galaxy.set_disk()
		galaxies += [galaxy]

	plot_prop_vs_prop(['nH'], ['D/Z'], galaxies, bin_nums=40, labels=labels, foutname=plot_dir+'Coulomb_DZ_vs_nH.pdf', std_bars=True, style='color-linestyle', include_obs=True)
	plot_prop_vs_prop(['nH_neutral'], ['D/Z'], galaxies, bin_nums=40, labels=labels, foutname=plot_dir+'Coulomb_DZ_vs_nH_neutral.pdf', std_bars=True, style='color-linestyle', include_obs=True)

	config.SMALL_FONT=20
	plot_obs_prop_vs_prop(['sigma_gas_neutral','r'], ['D/Z','D/Z'], galaxies, pixel_res=2, bin_nums=40, labels=labels, foutname=plot_dir+'Coulomb_B13_obs_DZ_vs_surf.pdf', \
						std_bars=True, style='color-linestyle', include_obs=True)
	config.SMALL_FONT=22
	# Need larger font size for this plot
	config.LARGE_FONT       = 40
	config.EXTRA_LARGE_FONT = 56
	elems = ['Mg','Si','Fe','O','C']
	plot_elem_depletion_vs_prop(elems, 'nH', galaxies, bin_nums=50, labels=labels, foutname=plot_dir+'Coulomb_elemental_dep_vs_nH.pdf', \
						std_bars=True, style='color-linestyle', include_obs=True)
	plot_elem_depletion_vs_prop(elems, 'nH_neutral', galaxies, bin_nums=50, labels=labels, foutname=plot_dir+'Coulomb_elemental_dep_vs_nH_neutral.pdf', \
						std_bars=True, style='color-linestyle', include_obs=True)

	sightline_data_files = []
	for j,snap_dir in enumerate(snap_dirs):
		sight_lines = Sight_Lines(snap_dir, num, cosmological=0, dust_impl=implementations[j], periodic_bound_fix=True, dirc='./', name=None)
		sight_lines.create_sightlines(N=N_sightlines, radius=config.SOLAR_GAL_RADIUS, dist_lims=[0.1,1.9])
		sightline_data_files += [sight_lines.name]

	config.FIG_XRATIO = 1.1

	plot_sightline_depletion_vs_prop(elems, 'NH_neutral', sightline_data_files, bin_data=True, bin_nums=20, labels=labels, foutname=plot_dir+'Coulomb_binned_sightline_depl_vs_NH.pdf', \
							 std_bars=True, style='color-linestyle', include_obs=True)
	plot_sightline_depletion_vs_prop(elems, 'NH_neutral', sightline_data_files, bin_data=False, labels=labels, foutname=plot_dir+'Coulomb_raw_sightline_depl_vs_NH.pdf', \
							 std_bars=True, style='color-linestyle', include_obs=True)
	config.LARGE_FONT       = 30
	config.EXTRA_LARGE_FONT = 36
	config.FIG_XRATIO = 1.0



###############################################################################
# Plot last snapshot comparisons for FIRE-2/3
###############################################################################

snap_dirs = []
main_dir ='/oasis/tscc/scratch/cchoban/new_gizmo/'

main_dir = '/work/06185/tg854841/frontera/non_cosmo/'
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
		galaxy = load_disk(snap_dir, num, cosmological=cosmological, mode=None, hdir=None, periodic_bound_fix=pb_fix, rmax=r_max, height=disk_height)
		galaxy.set_disk()
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
	# Need to cutoff weird low density portion due to odd FIRE-3 results (most likely due to using development version)
	config.PROP_INFO['nH_neutral'][1]=[0.5E-1, 0.9E3]
	plot_prop_vs_prop(['nH_neutral'], ['D/Z'], galaxies, bin_nums=40, labels=labels, foutname=plot_dir+'FIRE2-3_DZ_vs_nH_neutral.pdf', std_bars=True, style='color-linestyle', include_obs=True)
	config.PROP_INFO['nH_neutral'][1]=[1.1E-2, 0.9E3]
	config.SMALL_FONT=20
	plot_obs_prop_vs_prop(['sigma_gas_neutral','r'], ['D/Z','D/Z'], galaxies, pixel_res=2, bin_nums=40, labels=labels, foutname=plot_dir+'FIRE2-3_B13_obs_DZ_vs_surf.pdf', \
						std_bars=True, style='color-linestyle', include_obs=True)
	config.SMALL_FONT=22
	# Need larger font size for this plot
	config.LARGE_FONT       = 40
	config.EXTRA_LARGE_FONT = 56
	plot_elem_depletion_vs_prop(elems, 'nH', galaxies, bin_nums=50, labels=labels, foutname=plot_dir+'FIRE2-3_elemental_dep_vs_nH.pdf', \
						std_bars=True, style='color-linestyle', include_obs=True)
	# Need to cutoff weird low density portion due to odd FIRE-3 results (most likely due to using development version)
	config.PROP_INFO['nH_neutral'][1]=[0.5E-1, 0.9E3]
	plot_elem_depletion_vs_prop(elems, 'nH_neutral', galaxies, bin_nums=50, labels=labels, foutname=plot_dir+'FIRE2-3_elemental_dep_vs_nH_neutral.pdf', \
						std_bars=True, style='color-linestyle', include_obs=True)
	config.PROP_INFO['nH_neutral'][1]=[1.1E-2, 0.9E3]

	sightline_data_files = []
	for j,snap_dir in enumerate(snap_dirs):
		sight_lines = Sight_Lines(snap_dir, num, cosmological=0, dust_impl=implementations[j], periodic_bound_fix=True, dirc='./', name=None)
		sight_lines.create_sightlines(N=N_sightlines, radius=config.SOLAR_GAL_RADIUS, dist_lims=[0.1,1.9])
		sightline_data_files += [sight_lines.name]

	config.FIG_XRATIO = 1.1
	plot_sightline_depletion_vs_prop(elems, 'NH_neutral', sightline_data_files, bin_data=True, bin_nums=20, labels=labels, foutname=plot_dir+'FIRE2-3_binned_sightline_depl_vs_NH.pdf', \
							 std_bars=True, style='color-linestyle', include_obs=True)
	plot_sightline_depletion_vs_prop(elems, 'NH_neutral', sightline_data_files, bin_data=False, labels=labels, foutname=plot_dir+'FIRE2-3_raw_sightline_depl_vs_NH.pdf', \
							 std_bars=True, style='color-linestyle', include_obs=True)
	config.LARGE_FONT       = 30
	config.EXTRA_LARGE_FONT = 36
	config.FIG_XRATIO = 1.0


	dmol_vs_props(['fH2','fdense'], ['nH', 'T'], galaxies, bin_nums=50, labels=labels, foutname=plot_dir+'FIRE2-3_fdense.pdf', std_bars=True)
	dmol_vs_props(['fdense','CinCO'], ['nH', 'T'], galaxies, bin_nums=50, labels=labels, foutname=plot_dir+'FIRE2-3_CinCO.pdf', std_bars=True)


###############################################################################
# Plot comparisons for sub-resolved fdense routine
###############################################################################

# Directory of snap file
main_dir = '/oasis/tscc/scratch/cchoban/non_cosmo/fMC_test/'
main_dir = '/scratch1/06185/tg854841/NH2_test'
snap_dirs = [main_dir+'/NH2_0.5/output/',main_dir+'/NH2_1.0/output/',
				main_dir+'/NH2_1.5/output/',main_dir+'/NH2_2.0/output/']

# Label for test plots
labels = [r'$N_{\rm H_2}^{\rm crit}=0.5\times10^{21}$ cm$^{-3}$',r'$N_{\rm H_2}^{\rm crit}=1.0\times10^{21}$ cm$^{-3}$',
			r'$N_{\rm H_2}^{\rm crit}=1.5\times10^{21}$ cm$^{-3}$',r'$N_{\rm H_2}^{\rm crit}=2.0\times10^{21}$ cm$^{-3}$']
cosmological = False

# Snapshot to check
snap_num = 19

galaxies = []
for j,snap_dir in enumerate(snap_dirs):
	print(snap_dir)
	galaxy = load_disk(snap_dir, snap_num, cosmological=cosmological, mode=None, hdir=None, periodic_bound_fix=pb_fix, rmax=r_max, height=disk_height)
	galaxy.set_disk()
	galaxies += [galaxy]

dmol_vs_props(['fH2','fdense'], ['nH', 'T'], galaxies, bin_nums=50, labels=labels, foutname=plot_dir+'NH2_crit_variation.pdf', std_bars=True)



###############################################################################
# Get median gas-dust accretion timescales for analytical model
###############################################################################

snap_dir = '/work/06185/tg854841/frontera/non_cosmo/Elemental/fiducial/output/'
snap_num = 300

snap = load_snap(snap_dir, snap_num, cosmological=cosmological, periodic_bound_fix=pb_fix)
t, sfr = snap.get_SFH()

plt.plot(t,sfr)
plt.xlabel('Time (Gyr)')
plt.ylabel(r'SFR ($M_{\odot}$/Gyr)')
plt.savefig('elem_last_snap_sfh.png')
plt.close()


snap_dir = '/work/06185/tg854841/frontera/non_cosmo/Species/fiducial/output/'
snap_num = 300

snap = load_snap(snap_dir, snap_num, cosmological=cosmological, periodic_bound_fix=pb_fix)
t, sfr = snap.get_SFH()

plt.plot(t,sfr)
plt.xlabel('Time (Gyr)')
plt.ylabel(r'SFR ($M_{\odot}$/Gyr)')
plt.savefig('spec_last_snap_sfh.png')
plt.close()



galaxy = load_disk(snap_dir, snap_num, cosmological=cosmological, mode=None, hdir=None, periodic_bound_fix=pb_fix, rmax=r_max, height=disk_height)
galaxy.set_disk()
G = galaxy.loadpart(0)

print("Species Accretion Timescales")
gtimes = dust_acc.calc_spec_acc_timescale(G,nano_iron=False)
bins = np.logspace(np.log10(1E5),np.log10(1E9),100)
for key in gtimes.keys():
	mask = ~np.isinf(gtimes[key])
	print(key)
	print(np.median(gtimes[key][mask])/1E6)
	plt.hist(gtimes[key][mask], bins=bins, weights=G.m[mask], histtype='step', label=key, cumulative=True)
plt.legend()
plt.xscale('log')
plt.savefig("spec_t_grow_hist.png")
plt.close()

print("Elemental Accretion Timescales")
gtimes = dust_acc.calc_elem_acc_timescale(G)
bins = np.logspace(np.log10(1E5),np.log10(1E11),1000)
for key in gtimes.keys():
	mask = ~np.isinf(gtimes[key])
	print(key)
	print(np.median(gtimes[key][mask])/1E6)
	plt.hist(gtimes[key][mask], bins=bins, weights=G.m[mask], histtype='step', label=key, cumulative=True)
plt.legend()
plt.xscale('log')
plt.savefig("elem_t_grow_hist.png")
plt.close()

S = galaxy.loadpart(4)

total_SNe = 0.
for k in range(len(S.sft)):
	star_age=1.5-S.sft[k]
	agemin=0.003401; agebrk=0.01037; agemax=0.03753; # in Gyr
	RSNe = 0
	if star_age > agemin:
		if star_age<=agebrk:
			RSNe = 5.408e-4 # NSNe/Myr *if* each SNe had exactly 10^51 ergs; really from the energy curve
		elif star_age<=agemax:
			RSNe = 2.516e-4 # this is for a 1 Msun population
		# Ia (prompt Gaussian+delay, Manucci+06)
		if star_age>agemax:
			RSNe = 5.3e-8 + 1.6e-5*np.exp(-0.5*((star_age-0.05)/0.01)*((star_age-0.05)/0.01));
	total_SNe+=RSNe*S.m[k]*config.UnitMass_in_Msolar
print("Total SNe per Year = ",total_SNe/1E6)
print("M_ISM =",np.sum(G.m*config.UnitMass_in_Msolar)/1E9, ' 10^9 M_solar')

