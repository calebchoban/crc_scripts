from gizmo_library.time_evolution import Dust_Evo
from gizmo_library.sightline import Sight_Lines
from gizmo import *
from dust_plots import *
import analytical_models.dust_accretion as dust_acc




###############################################################################
#
# This routine creates the plots for "The Galactic Dust-Up: Modelling Dust Evolution in FIRE" paper.
# Plots are created in the order they appear in the text with all Appendix plots at the end.
#
###############################################################################




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

config.FIG_XRATIO=1.2 # Make the aspect ratio a little wider
config.SMALL_FONT = 28
config.LARGE_FONT = 36

config.PROP_INFO['time'][1]=[0.8E-2,1.5] # simulations only run for 1.5 Gyr

disk_load_args = {'id':-1, 'hdir':None, 'rmax':r_max, 'height':disk_height}
disk_set_args = {'ptype':0, 'age_limits':None}





snap_dirs = ['/work/06185/tg854841/frontera/non_cosmo/Species/nano_Fe/output/',
			 '/scratch1/06185/tg854841/FIRE2_AGB_test/output/',
			 '/scratch1/06185/tg854841/FIRE2_AGB_increase/output/']
names = ['nano_Fe','orig_FIRE2','increase_FIRE2']
labels = ['Before','Orig. FIRE-2','Decreased AGB FIRE-2']

cosmological = False

# Now preload the time evolution data

implementation = 'species'

dust_evo_datas = []
for i,snap_dir in enumerate(snap_dirs):
	name = names[i]
	print(name)
	dust_evo_data = Dust_Evo(snap_dir, snap_lims, cosmological=cosmological, periodic_bound_fix=pb_fix,
							 dirc='./time_evo_data/', name_prefix=implementation)
	dust_evo_data.set_disk(load_kwargs=disk_load_args, set_kwargs=disk_set_args)
	dust_evo_data.load(increment=10)
	dust_evo_data.save()
	dust_evo_datas += [dust_evo_data]

# Now plot a comparison of each of the runs
dust_data_vs_time(['D/Z','source_frac', 'spec_frac'], dust_evo_datas, stat='median',
				  foutname=plot_dir+'AGB_compare_median_spec_all_data_vs_time.pdf',labels=labels, style='color-linestyle')

dust_data_vs_time(['D/Z','source_frac', 'spec_frac'], dust_evo_datas, stat='total',
				  foutname=plot_dir+'AGB_compare_total_spec_all_data_vs_time.pdf',labels=labels, style='color-linestyle')




###############################################################################
# Species Implementation w/ modifications that produced the largest changes
###############################################################################


main_dir = '/work/06185/tg854841/frontera/non_cosmo/Species/'
names = ['fiducial','enhanced_acc','no_temp_cutoff','enhanced_dest']
labels = ['Fiducial','Enhanced Acc.','No Temp. Cutoff','Enhanced Dest.']


cosmological = False

# Now preload the time evolution data
snap_dirs = [main_dir + i + '/output/' for i in names]

implementation = 'species'

dust_evo_datas = []
for i,snap_dir in enumerate(snap_dirs):
	name = names[i]
	print(name)
	dust_evo_data = Dust_Evo(snap_dir, snap_lims, cosmological=cosmological, periodic_bound_fix=pb_fix,
							 dirc='./time_evo_data/', name_prefix=implementation)
	dust_evo_data.set_disk(load_kwargs=disk_load_args, set_kwargs=disk_set_args)
	dust_evo_data.load(increment=10)
	dust_evo_data.save()
	dust_evo_datas += [dust_evo_data]

# Now plot a comparison of each of the runs
dust_data_vs_time(['D/Z','source_frac', 'spec_frac'], dust_evo_datas, stat='median',
				  foutname=plot_dir+'acc_median_spec_all_data_vs_time.pdf',labels=labels, style='color-linestyle')

dust_data_vs_time(['D/Z','source_frac', 'spec_frac'], dust_evo_datas, stat='total',
				  foutname=plot_dir+'acc_total_spec_all_data_vs_time.pdf',labels=labels, style='color-linestyle')


###############################################################################
# Elemental Implementation w/ modifications that produced the largest changes
###############################################################################


main_dir = '/work/06185/tg854841/frontera/non_cosmo/Elemental/'
names = ['fiducial','enhanced_acc','enhanced_dest']
labels = ['Fiducial','Enhanced Acc.','Enhanced Dest.']



# Now preload the time evolution data
snap_dirs = [main_dir + i + '/output/' for i in names]

implementation = 'elemental'

dust_evo_datas = []
for i,snap_dir in enumerate(snap_dirs):
	name = names[i]
	print(name)
	dust_evo_data = Dust_Evo(snap_dir, snap_lims, cosmological=cosmological, periodic_bound_fix=pb_fix,
							 dirc='./time_evo_data/' ,name_prefix=implementation)
	dust_evo_data.set_disk(load_kwargs=disk_load_args, set_kwargs=disk_set_args)
	dust_evo_data.load(increment=10)
	dust_evo_data.save()
	dust_evo_datas += [dust_evo_data]

# Now plot a comparison of each of the runs
dust_data_vs_time(['D/Z','source_frac', 'spec_frac'], dust_evo_datas, stat='median',
				  foutname=plot_dir+'acc_median_elem_all_data_vs_time.pdf',labels=labels, style='color-linestyle')

dust_data_vs_time(['D/Z','source_frac', 'spec_frac'], dust_evo_datas, stat='total',
				  foutname=plot_dir+'acc_total_elem_all_data_vs_time.pdf',labels=labels, style='color-linestyle')


config.FIG_XRATIO=1. # Reset to normal
config.SMALL_FONT = 22
config.LARGE_FONT = 30



###############################################################################
# Plot last snapshot D/Z values vs observations for optional dust species physics 
###############################################################################


snap_dirs = []
main_dir = '/work/06185/tg854841/frontera/non_cosmo/Species/'

names = ['fiducial','O_reservoir', 'nano_Fe']
snap_dirs += [main_dir + i + '/output/' for i in names] 
labels = ['Species','Species w/ O', 'Species w/ O & Fe']

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
	config.SMALL_FONT 		= 30
	config.LARGE_FONT       = 44
	config.EXTRA_LARGE_FONT = 56
	config.FIG_XRATIO = 1.25
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

	config.FIG_XRATIO = 1.25

	plot_sightline_depletion_vs_prop(elems, 'NH_neutral', sightline_data_files, bin_data=True, bin_nums=20, labels=labels, foutname=plot_dir+'binned_sightline_depl_vs_NH.pdf', \
							 std_bars=True, style='color-linestyle', include_obs=True)
	plot_sightline_depletion_vs_prop(elems, 'NH_neutral', sightline_data_files, bin_data=False, labels=labels, foutname=plot_dir+'raw_sightline_depl_vs_NH.pdf', \
							 std_bars=True, style='color-linestyle', include_obs=True)

	config.SMALL_FONT 		= 22
	config.LARGE_FONT       = 30
	config.EXTRA_LARGE_FONT = 36
	config.FIG_XRATIO = 1.

###############################################################################
# Get median values for certain properties for last snapshots.
# Mostly used for IC's listed in paper.
###############################################################################



main_dir = '/work/06185/tg854841/frontera/non_cosmo/Species/'
names = ['fiducial','O_reservoir', 'nano_Fe']
snap_dirs = [main_dir + i + '/output/' for i in names]
labels = ['Species','Species w/ O', 'Species w/ O & Fe']

main_dir = '/work/06185/tg854841/frontera/non_cosmo/Elemental/'
names = ['fiducial']
snap_dirs += [main_dir + i + '/output/' for i in names]
labels += ['Elemental']

implementations = ['species','species','species','elemental']

cosmological = False

# List of snapshots to compare
snaps = [300]


for i, num in enumerate(snaps):
	print(num)
	galaxies = []
	for j,snap_dir in enumerate(snap_dirs):
		print(snap_dir)
		print(labels[j])
		galaxy = load_disk(snap_dir, num, cosmological=cosmological, mode=None, hdir=None, periodic_bound_fix=pb_fix, rmax=r_max, height=disk_height)
		if num == 0:
			galaxy.set_disk(ptype=1,age_limits=None)
		else:
			galaxy.set_disk()
		gas = galaxy.loadpart(0)
		median = utils.weighted_percentile(gas.get_property("D/Z"), percentiles=[50], weights=gas.get_property("M"), ignore_invalid=True)
		print("Median D/Z:", median)

		Z = gas.get_property("Z")
		median = utils.weighted_percentile(Z, percentiles=[50], weights=gas.get_property("M"), ignore_invalid=True)
		print("Median Z:", median)

		h = gas.get_property("h")
		median = utils.weighted_percentile(h, percentiles=[50], weights=gas.get_property("M"), ignore_invalid=True)
		print("Median gas h:", median)
		min_ind = np.argmin(h)
		print("Min gas h:", h[min_ind])
		print("Mass: ", gas.get_property("M")[min_ind]," nH:",gas.get_property("nH")[min_ind])

		softening = 16*np.power(gas.get_property("M")/1000,1./3.)*np.power(gas.get_property("nH")/10,-1./3.)
		print("Min gas softening:", np.min(softening))


		m = gas.get_property("m")
		median = utils.weighted_percentile(m, percentiles=[50], ignore_invalid=True)
		print("Median gas mass:", median)

		dm = galaxy.loadpart(1)
		m = dm.get_property("m")
		median = utils.weighted_percentile(m, percentiles=[50], ignore_invalid=True)
		print("Median DM mass:", median)

		dm = galaxy.loadpart(2)
		m = dm.get_property("m")
		median = utils.weighted_percentile(m, percentiles=[50], ignore_invalid=True)
		print("Median type 2 mass:", median)

		dm = galaxy.loadpart(3)
		m = dm.get_property("m")
		median = utils.weighted_percentile(m, percentiles=[50], ignore_invalid=True)
		print("Median type 3 mass:", median)


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
plt.savefig(plot_dir+'elem_last_snap_sfh.png')
plt.close()


snap_dir = '/work/06185/tg854841/frontera/non_cosmo/Species/fiducial/output/'
snap_num = 300

snap = load_snap(snap_dir, snap_num, cosmological=cosmological, periodic_bound_fix=pb_fix)
t, sfr = snap.get_SFH()

plt.plot(t,sfr)
plt.xlabel('Time (Gyr)')
plt.ylabel(r'SFR ($M_{\odot}$/yr)')
plt.savefig(plot_dir+'spec_last_snap_sfh.png')
plt.close()



galaxy = load_disk(snap_dir, snap_num, cosmological=cosmological, mode=None, hdir=None, periodic_bound_fix=pb_fix, rmax=r_max, height=disk_height)
galaxy.set_disk()
G = galaxy.loadpart(0)
gas_Z = G.get_property('Z')

print("Median Z:",np.median(gas_Z))
vals = utils.weighted_percentile(gas_Z, weights=G.get_property("M"), ignore_invalid=True)
print("Percentile Z:",vals)

plt.hist(gas_Z, bins = 20)
plt.xlabel(r'Z ($Z_{\odot}$)')
plt.ylabel("Count")
plt.yscale('log')
plt.savefig(plot_dir+'Z_hist.png')
plt.close()


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
plt.savefig(plot_dir+"spec_t_grow_hist.png")
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
plt.savefig(plot_dir+"elem_t_grow_hist.png")
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






###############################################################################
###############################################################################
#
# Now all the Appendix plots
#
###############################################################################
###############################################################################


###############################################################################
# Plot time evolution for different initial conditions
###############################################################################

config.FIG_XRATIO=1.2 # Make the aspect ratio a little wider
config.SMALL_FONT = 28
config.LARGE_FONT = 36


cosmological = False

startnum = 0
endnum = 300
snap_lims = [startnum,endnum]

# Maximum radius used for getting data
r_max = 20 # kpc
disk_height = 2 # kpc

pb_fix=True

config.FIG_XRATIO=1.2 # Make the aspect ratio a little wider
config.SMALL_FONT = 28
config.LARGE_FONT = 36

config.PROP_INFO['time'][1]=[0.8E-2,1.5] # simulations only run for 1.5 Gyr

# Now preload the time evolution data
snap_dirs = ['/work/06185/tg854841/frontera/non_cosmo/Species/nano_Fe/output', '/scratch1/06185/tg854841/init_Z_dust_tests/init_0.5Z/output',
			 '/scratch1/06185/tg854841/init_Z_dust_tests/fiducial_init_dust/output']
labels = [r'$Z_{\rm initial}=Z_{\odot}$',r'$Z_{\rm initial}=\frac{1}{2} Z_{\odot}$',
		  r'$Z_{\rm initial}=Z_{\odot}$ w/ dust']

implementation = 'species'

dust_evo_datas = []
for i,snap_dir in enumerate(snap_dirs):
	print(snap_dir)
	print(snap_lims)
	dust_evo_data = Dust_Evo(snap_dir, snap_lims, cosmological=cosmological, periodic_bound_fix=pb_fix,
							 dirc='./time_evo_data/', name_prefix=implementation)
	dust_evo_data.set_disk(load_kwargs=disk_load_args, set_kwargs=disk_set_args)
	dust_evo_data.load(increment=10)
	dust_evo_data.save()
	dust_evo_datas += [dust_evo_data]

# Now plot a comparison of each of the runs
dust_data_vs_time(['D/Z','source_frac', 'spec_frac'], dust_evo_datas, stat='median',
				  foutname=plot_dir+'ICs_spec_all_data_vs_time.pdf',labels=labels, style='color-linestyle')

config.FIG_XRATIO = 1. # Reset to normal
config.SMALL_FONT = 22
config.LARGE_FONT = 30



###############################################################################
# Plot last snapshot D/Z values vs observations for different initial conditions
###############################################################################

names = ['nano_Fe','full_no_Coulomb_enh']
snap_dirs = ['/work/06185/tg854841/frontera/non_cosmo/Species/nano_Fe/output', '/scratch1/06185/tg854841/init_Z_dust_tests/init_0.5Z/output',
			 '/scratch1/06185/tg854841/init_Z_dust_tests/fiducial_init_dust/output']
labels = ['Fiducial', r'$Z_{\rm init}=\frac{1}{2} Z_{\odot}$', 'Init. Dust']

implementations = ['species','species','species']

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
		if j == 2:
			num = 141
		else:
			num = 300
		print(num)
		galaxy = load_disk(snap_dir, num, cosmological=cosmological, mode=None, hdir=None, periodic_bound_fix=pb_fix, rmax=r_max, height=disk_height)
		galaxy.set_disk()
		galaxies += [galaxy]

	plot_prop_vs_prop(['nH_neutral'], ['D/Z'], galaxies, bin_nums=40, labels=labels, foutname=plot_dir+'ICs_DZ_vs_nH_neutral.pdf', \
					  std_bars=True, style='color-linestyle', include_obs=True)
	config.SMALL_FONT=20
	plot_obs_prop_vs_prop(['sigma_gas_neutral','r'], ['D/Z','D/Z'], galaxies, pixel_res=2, bin_nums=40, labels=labels, foutname=plot_dir+'ICs_B13_obs_DZ_vs_surf.pdf', \
						std_bars=True, style='color-linestyle', include_obs=True)
	config.SMALL_FONT=22
	# Need larger font size for this plot
	config.SMALL_FONT 		= 30
	config.LARGE_FONT       = 44
	config.EXTRA_LARGE_FONT = 56
	config.FIG_XRATIO = 1.25
	elems = ['Mg','Si','Fe','O','C']
	plot_elem_depletion_vs_prop(elems, 'nH_neutral', galaxies, bin_nums=50, labels=labels, foutname=plot_dir+'ICs_elemental_dep_vs_nH_neutral.pdf', \
						std_bars=True, style='color-linestyle', include_obs=True)

	config.FIG_XRATIO = 1.25


	config.SMALL_FONT 		= 22
	config.LARGE_FONT       = 30
	config.EXTRA_LARGE_FONT = 36
	config.FIG_XRATIO = 1.0




snap_dir = '/scratch1/06185/tg854841/init_Z_dust_tests/init_0.5Z/output/'
snap_num = 300

snap = load_snap(snap_dir, snap_num, cosmological=cosmological, periodic_bound_fix=pb_fix)
t, sfr = snap.get_SFH()

plt.plot(t,sfr)
plt.xlabel('Time (Gyr)')
plt.ylabel(r'SFR ($M_{\odot}$/yr)')
plt.savefig(plot_dir+'initZ_spec_last_snap_sfh.png')
plt.close()



galaxy = load_disk(snap_dir, snap_num, cosmological=cosmological, mode=None, hdir=None, periodic_bound_fix=pb_fix, rmax=r_max, height=disk_height)
galaxy.set_disk()
G = galaxy.loadpart(0)
gas_Z = G.get_property('Z')

print("Median Z:",np.median(gas_Z))
vals = utils.weighted_percentile(gas_Z, weights=G.get_property("M"), ignore_invalid=True)
print("Percentile Z:",vals)

snap_dir = '/scratch1/06185/tg854841/init_Z_dust_tests/fiducial_init_dust/output/'
snap_nums = [140,300]


for snap_num in snap_nums:
	snap = load_snap(snap_dir, snap_num, cosmological=cosmological, periodic_bound_fix=pb_fix)
	t, sfr = snap.get_SFH()
	print(snap_num)

	plt.plot(t,sfr)
	plt.xlabel('Time (Gyr)')
	plt.ylabel(r'SFR ($M_{\odot}$/yr)')
	plt.savefig(plot_dir+str(snap_num)+'_initdust_spec_last_snap_sfh.png')
	plt.close()



	galaxy = load_disk(snap_dir, snap_num, cosmological=cosmological, mode=None, hdir=None, periodic_bound_fix=pb_fix, rmax=r_max, height=disk_height)
	galaxy.set_disk()
	G = galaxy.loadpart(0)
	gas_Z = G.get_property('Z')

	print("Median Z:",np.median(gas_Z))
	vals = utils.weighted_percentile(gas_Z, weights=G.get_property("M"), ignore_invalid=True)
	print("Percentile Z:",vals)





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

config.PROP_INFO['T'][1]=[1.1E1, 0.9E4] # Clip off high T since fH2 is sometimes non-zero here for FIRE-2
dmol_vs_props(['fH2','fdense'], ['nH', 'T'], galaxies, bin_nums=50, labels=labels, foutname=plot_dir+'NH2_crit_variation.pdf', std_bars=True)
config.PROP_INFO['T'][1]=[1.1E1, 0.9E5]

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

implementation = 'species'

dust_evo_datas = []
for i,snap_dir in enumerate(snap_dirs):
	name = names[i]
	print(name)
	dust_evo_data = Dust_Evo(snap_dir, snap_lims, cosmological=cosmological, periodic_bound_fix=pb_fix,
							 dirc='./time_evo_data/', name_prefix=implementation)
	dust_evo_data.set_disk(load_kwargs=disk_load_args, set_kwargs=disk_set_args)
	dust_evo_data.load(increment=10)
	dust_evo_data.save()
	dust_evo_datas += [dust_evo_data]



# Now plot a comparison of each of the runs
dust_data_vs_time(['D/Z','source_frac', 'spec_frac'], dust_evo_datas, stat='median',
				  foutname=plot_dir+'creation_median_spec_all_data_vs_time.pdf',labels=labels, style='color-linestyle')

dust_data_vs_time(['D/Z','source_frac', 'spec_frac'], dust_evo_datas, stat='total',
				  foutname=plot_dir+'creation_total_spec_all_data_vs_time.pdf',labels=labels, style='color-linestyle')

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

implementation = 'elemental'

dust_evo_datas = []
for i,snap_dir in enumerate(snap_dirs):
	name = names[i]
	print(name)
	dust_evo_data = Dust_Evo(snap_dir, snap_lims, cosmological=cosmological, periodic_bound_fix=pb_fix,
							 dirc='./time_evo_data/', name_prefix=implementation)
	dust_evo_data.set_disk(load_kwargs=disk_load_args, set_kwargs=disk_set_args)
	dust_evo_data.load(increment=10)
	dust_evo_data.save()
	dust_evo_datas += [dust_evo_data]


# Now plot a comparison of each of the runs
dust_data_vs_time(['D/Z','source_frac', 'spec_frac'], dust_evo_datas, stat='median',
				  foutname=plot_dir+'creation_median_elem_all_data_vs_time.pdf',labels=labels, style='color-linestyle')

dust_data_vs_time(['D/Z','source_frac', 'spec_frac'], dust_evo_datas, stat='total',
				  foutname=plot_dir+'creation_total_elem_all_data_vs_time.pdf',labels=labels, style='color-linestyle')


###############################################################################
# FIRE-2 vs FIRE-3 Metal Returns
###############################################################################

Z=1
elems =['C','O','Mg','Si','Fe']
compare_FIRE_metal_yields(Z, elems, foutname=plot_dir+'FIRE_yields_comparison.pdf')


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
	config.PROP_INFO['nH'][1]=[3E-2, 0.9E3]
	config.PROP_INFO['T'][1]=[1.1E1, 0.9E5]


	plot_prop_vs_prop(['nH'], ['D/Z'], galaxies, bin_nums=40, labels=labels, foutname=plot_dir+'FIRE2-3_DZ_vs_nH.pdf', std_bars=True, style='color-linestyle', include_obs=True)
	# Need to cutoff weird low density portion due to odd FIRE-3 results (most likely due to using development version)
	plot_prop_vs_prop(['nH_neutral'], ['D/Z'], galaxies, bin_nums=40, labels=labels, foutname=plot_dir+'FIRE2-3_DZ_vs_nH_neutral.pdf', std_bars=True, style='color-linestyle', include_obs=True)
	config.SMALL_FONT=20
	plot_obs_prop_vs_prop(['sigma_gas_neutral','r'], ['D/Z','D/Z'], galaxies, pixel_res=2, bin_nums=40, labels=labels, foutname=plot_dir+'FIRE2-3_B13_obs_DZ_vs_surf.pdf', \
						std_bars=True, style='color-linestyle', include_obs=True)
	config.SMALL_FONT=22
	# Need larger font size for this plot
	config.SMALL_FONT 		= 30
	config.LARGE_FONT       = 44
	config.EXTRA_LARGE_FONT = 56
	config.FIG_XRATIO = 1.25
	elems = ['Mg','Si','Fe','O','C']
	plot_elem_depletion_vs_prop(elems, 'nH', galaxies, bin_nums=50, labels=labels, foutname=plot_dir+'FIRE2-3_elemental_dep_vs_nH.pdf', \
						std_bars=True, style='color-linestyle', include_obs=True)
	# Need to cutoff weird low density portion due to odd FIRE-3 results (most likely due to using development version)
	plot_elem_depletion_vs_prop(elems, 'nH_neutral', galaxies, bin_nums=50, labels=labels, foutname=plot_dir+'FIRE2-3_elemental_dep_vs_nH_neutral.pdf', \
						std_bars=True, style='color-linestyle', include_obs=True)

	sightline_data_files = []
	for j,snap_dir in enumerate(snap_dirs):
		sight_lines = Sight_Lines(snap_dir, num, cosmological=0, dust_impl=implementations[j], periodic_bound_fix=True, dirc='./', name=None)
		sight_lines.create_sightlines(N=N_sightlines, radius=config.SOLAR_GAL_RADIUS, dist_lims=[0.1,1.9])
		sightline_data_files += [sight_lines.name]

	config.FIG_XRATIO = 1.25
	plot_sightline_depletion_vs_prop(elems, 'NH_neutral', sightline_data_files, bin_data=True, bin_nums=20, labels=labels, foutname=plot_dir+'FIRE2-3_binned_sightline_depl_vs_NH.pdf', \
							 std_bars=True, style='color-linestyle', include_obs=True)
	plot_sightline_depletion_vs_prop(elems, 'NH_neutral', sightline_data_files, bin_data=False, labels=labels, foutname=plot_dir+'FIRE2-3_raw_sightline_depl_vs_NH.pdf', \
							 std_bars=True, style='color-linestyle', include_obs=True)


	config.SMALL_FONT 		= 22
	config.LARGE_FONT       = 30
	config.EXTRA_LARGE_FONT = 36
	config.FIG_XRATIO = 1.0


	dmol_vs_props(['fH2','fdense'], ['nH', 'T'], galaxies, bin_nums=50, labels=labels, foutname=plot_dir+'FIRE2-3_fdense.pdf', std_bars=True)
	dmol_vs_props(['fdense','CinCO'], ['nH', 'T'], galaxies, bin_nums=50, labels=labels, foutname=plot_dir+'FIRE2-3_CinCO.pdf', std_bars=True)



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
	config.SMALL_FONT 		= 30
	config.LARGE_FONT       = 44
	config.EXTRA_LARGE_FONT = 56
	config.FIG_XRATIO = 1.25
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

	config.FIG_XRATIO = 1.25

	plot_sightline_depletion_vs_prop(elems, 'NH_neutral', sightline_data_files, bin_data=True, bin_nums=20, labels=labels, foutname=plot_dir+'Coulomb_binned_sightline_depl_vs_NH.pdf', \
							 std_bars=True, style='color-linestyle', include_obs=True)
	plot_sightline_depletion_vs_prop(elems, 'NH_neutral', sightline_data_files, bin_data=False, labels=labels, foutname=plot_dir+'Coulomb_raw_sightline_depl_vs_NH.pdf', \
							 std_bars=True, style='color-linestyle', include_obs=True)


	config.SMALL_FONT 		= 22
	config.LARGE_FONT       = 30
	config.EXTRA_LARGE_FONT = 36
	config.FIG_XRATIO = 1.0