import os
import subprocess
from gizmo_library.time_evolution import Dust_Evo
from gizmo import *
from dust_plots import *

# Directory of snap file
snap_dirs = ['/oasis/tscc/scratch/cchoban/non_cosmo/Species/output/']
# Snapshots to check
snap_num = 260

cosmological = False
pb_fix=True
dust_depl=False

# Label for test plots
labels = ['Species Fidcuial']


# Maximum radius, disk, height, and disk orientation used for getting data
r_max = 20 # kpc
disk_height = 2 # kpc


atomic_mass = [1.01, 2.0, 12.01, 14, 15.99, 20.2, 24.305, 28.086, 32.065, 40.078, 55.845]



print("Checking snapshot ",snap_num)
galaxies = []
for j,snap_dir in enumerate(snap_dirs):
	print("Snap Dirc: ",snap_dir)
	galaxy = load_disk(snap_dir, snap_num, cosmological=cosmological, id=-1, mode='AHF', hdir=None, periodic_bound_fix=pb_fix, rmax=r_max, height=disk_height)
	galaxies += [galaxy]


	print("Dust Implementation:", galaxy.sp.dust_impl)
	print("Number of Dust Elements:",galaxy.sp.Flag_DustMetals)
	print("Number of Dust Species:",galaxy.sp.Flag_DustSpecies)

	flag_species = 0
	if galaxy.sp.dust_impl=='species':
		flag_species = 1


	G = galaxy.loadpart(0)
	nH = G.rho*config.UnitDensity_in_cgs * ( 1. - (G.z[:,0]+G.z[:,1])) / config.H_MASS

	print("\n########################################\n")

	print("Checking for Nans...")
	nan_ind = np.argwhere(np.isnan(G.dz[:,0]))
	if len(nan_ind) > 0:
		print("%i particles with NaNs detected"%len(nan_ind))
		nan_ind = np.argwhere(np.isnan(G.dz[:,0]))
		print("Dust Metals:",G.dz[nan_ind])
		if flag_species:
			print("Species:",G.spec[nan_ind])
		print("Sources:",G.dzs[nan_ind])
		print("nH:", nH[nan_ind])
		print("T:", G.T[nan_ind])

	print("########################################\n")

	print("Checking for negative numbers...\n")
	neg_ind = np.argwhere(np.logical_and(np.any(G.dz<0,axis=1),np.any(G.dzs<0,axis=1),np.any(G.spec<0,axis=1)))
	if len(neg_ind) > 0:
		print("%i particles with negative numbers detected"%len(neg_ind))
		print("Dust Metals:",G.dz[neg_ind])
		if flag_species:
			print("Species:",G.spec[neg_ind])
		print("Sources:",G.dzs[neg_ind])
		print("nH:", nH[neg_ind])
		print("T:", G.T[neg_ind])


	print("\n########################################\n")

	print("Checking for too much dust...\n")
	over_ind = np.argwhere(np.any(G.dz>G.z[:,:11],axis=1)).flatten()
	if len(over_ind) > 0:
		print("%i particles with too much dust detected"%len(over_ind))
		print("Metals:",G.z[over_ind])
		print("D/Z:",G.dz[over_ind]/G.z[over_ind,:11])
		if flag_species:
			print("Species:",G.spec[over_ind])
		print("Sources:",G.dzs[over_ind])
		print("nH:", nH[over_ind])
		print("T:", G.T[over_ind])
		print("\t fH2:",G.dust_mol[over_ind,0])
		print("\t fMC:",G.dust_mol[over_ind,1])
		print("\t CinCO:",G.dust_mol[over_ind,2]/G.z[over_ind,2],"\n")


	if flag_species:

		print("########################################\n")

		print("Checking dust metals and dust species add up...\n")
		# Maximum allowed error between species and dust metals
		abs_error = 1E-2
		# Add up the elements from each dust species
		dust_metals = np.zeros(np.shape(G.dz))
		sil_num_atoms = [3.631,1.06,1.,0.571] # O, Mg, Si, Fe
		sil_elems_index = [4,6,7,10] # O,Mg,Si,Fe 
		dust_formula_mass = 0

		if galaxy.sp.Flag_DustSpecies==4:
			# Silicates
			for k in range(len(sil_num_atoms)):
				dust_formula_mass += sil_num_atoms[k] * atomic_mass[sil_elems_index[k]]
			for k in range(len(sil_num_atoms)):
				dust_metals[:,sil_elems_index[k]] += G.spec[:,0] * sil_num_atoms[k] * atomic_mass[sil_elems_index[k]] / dust_formula_mass
			
			# Carbon
			dust_metals[:,2] += G.spec[:,1]

			# Silicon Carbide
			dust_formula_mass = atomic_mass[2] + atomic_mass[7]
			dust_metals[:,2] += G.spec[:,2] * atomic_mass[2] / dust_formula_mass
			dust_metals[:,7] += G.spec[:,2] * atomic_mass[7] / dust_formula_mass

			# Iron
			dust_metals[:,10] += G.spec[:,3]
		
		elif galaxy.sp.Flag_DustSpecies==5:
			# Silicates
			for k in range(len(sil_num_atoms)):
				dust_formula_mass += sil_num_atoms[k] * atomic_mass[sil_elems_index[k]]
			for k in range(len(sil_num_atoms)):
				dust_metals[:,sil_elems_index[k]] += G.spec[:,0] * sil_num_atoms[k] * atomic_mass[sil_elems_index[k]] / dust_formula_mass
			
			# Carbon
			dust_metals[:,2] += G.spec[:,1]

			# Silicon Carbide
			dust_formula_mass = atomic_mass[2] + atomic_mass[7]
			dust_metals[:,2] += G.spec[:,2] * atomic_mass[2] / dust_formula_mass
			dust_metals[:,7] += G.spec[:,2] * atomic_mass[7] / dust_formula_mass

			# Iron
			dust_metals[:,10] += G.spec[:,3]

			# Oxygen Reservoir
			dust_metals[:,4] += G.spec[:,4]

		elif galaxy.sp.Flag_DustSpecies==6:
			# Iron in silicates comes in the form of a seperate dust species 'iron inclusions'
			sil_num_atoms = [3.631,1.06,1.] # O, Mg, Si, Fe
			sil_elems_index = [4,6,7] # O,Mg,Si,Fe 

			# Silicates
			for k in range(len(sil_num_atoms)):
				dust_formula_mass += sil_num_atoms[k] * atomic_mass[sil_elems_index[k]]
			for k in range(len(sil_num_atoms)):
				dust_metals[:,sil_elems_index[k]] += G.spec[:,0] * sil_num_atoms[k] * atomic_mass[sil_elems_index[k]] / dust_formula_mass
			
			# Carbon
			dust_metals[:,2] += G.spec[:,1]

			# Silicon Carbide
			dust_formula_mass = atomic_mass[2] + atomic_mass[7]
			dust_metals[:,2] += G.spec[:,2] * atomic_mass[2] / dust_formula_mass
			dust_metals[:,7] += G.spec[:,2] * atomic_mass[7] / dust_formula_mass

			# Free-Flying Iron and Iron Inclusions
			dust_metals[:,10] += G.spec[:,3] + G.spec[:,5]

			# Oxygen Reservoir
			dust_metals[:,4] += G.spec[:,4]
		
		else:
			print("\t Number of dust species not supported for this check:",galaxy.sp.Flag_DustSpecies)

		dust_metals[:,0]=np.sum(dust_metals[:,2:],axis=1)
		bad_ind = np.argwhere(np.logical_and(np.any(~np.isclose(G.dz, dust_metals, rtol=abs_error, atol=0,equal_nan=True),axis=1),G.dz[:,0]/G.z[:,0]>0.01)).flatten()

		if len(bad_ind) > 0:
			bad_ind = bad_ind[:5]
			print("%i particles with D/Z>0.01 and element and species not matching by %f%% "%(len(bad_ind),abs_error*100))
			print("Dust Metals:",G.dz[bad_ind])
			print("Dust Metals from Species:",dust_metals[bad_ind])
			print("D/Z:",G.dz[bad_ind]/G.z[bad_ind,:11])
			print("Species:",G.spec[bad_ind])
			print("Sources:",G.dzs[bad_ind])
			print("nH:", nH[bad_ind])
			print("T:", G.T[bad_ind])
			print("\t fH2:",G.dust_mol[bad_ind,0])
			print("\t fMC:",G.dust_mol[bad_ind,1])
			print("\t CinCO:",G.dust_mol[bad_ind,2]/G.z[bad_ind,2],"\n")
			print("\t Sum of Species::",np.sum(G.spec[bad_ind],axis=1))
			print("\t Sum of Elements:",np.sum(G.dz[bad_ind,2:],axis=1))
			print("\t Total Dust:",G.dz[bad_ind,0])


	print("########################################\n")

	print("Sanity Checks...\n")

	print("Particle with Max Dust Mass...")
	max_ind = np.nanargmax(G.dz[:,0])
	print("\t D/Z:",G.dz[max_ind]/G.z[max_ind,:11])
	print("\t Dust Metals:",G.dz[max_ind])
	print("\t Metals:",G.z[max_ind])
	if flag_species:
			print("\t Species:",G.spec[max_ind])
	print("\t Sources:",G.dzs[max_ind])
	print("\t nH:", nH[max_ind])
	print("\t T:", G.T[max_ind])
	if flag_species:
		print("\t Sum of Species: %e \t Sum of Elements: %e Total Dust: %e\n"%(np.sum(G.spec[max_ind]),np.sum(G.dz[max_ind,2:]),G.dz[max_ind,0]))
	else:
		print("\t Sum of Elements: %e Total Dust: %e\n"%(np.sum(G.dz[max_ind,2:]),G.dz[max_ind,0]))
	print("\t fH2: %e \t fMC: %e \t CinCO: %e \n"%(G.dust_mol[max_ind,0],G.dust_mol[max_ind,1],G.dust_mol[max_ind,2]/G.z[max_ind,2]))


	if flag_species and galaxy.sp.Flag_DustSpecies>4:
		print("Particle with Max O Reservoir...")
		max_ind = np.argmax(G.spec[:,4])
		print("\t D/Z:",G.dz[max_ind]/G.z[max_ind,:11])
		print("\t Dust Metals:",G.dz[max_ind])
		print("\t Metals:",G.z[max_ind])
		print("\t Species:",G.spec[max_ind])
		print("\t Sources:",G.dzs[max_ind])
		print("\t nH:", nH[max_ind])
		print("\t T:", G.T[max_ind])
		print("\t Sum of Species: %e \t Sum of Elements: %e Total Dust: %e\n"%(np.sum(G.spec[max_ind]),np.sum(G.dz[max_ind,2:]),G.dz[max_ind,0]))
		print("\t fH2: %e \t fMC: %e \t CinCO: %e \n"%(G.dust_mol[max_ind,0],G.dust_mol[max_ind,1],G.dust_mol[max_ind,2]/G.z[max_ind,2]))




	print("Creating dust plots to check by eye...")

	dmol_vs_params('fH2', ['nH', 'T'], galaxies, bin_nums=50, time=None, labels=None, foutname='check_snap_'+str(snap_num)+'_fH2_vs_nH_T.png', std_bars=True, style='color')
	dmol_vs_params('fMC', ['nH', 'T'], galaxies, bin_nums=50, time=None, labels=None, foutname='check_snap_'+str(snap_num)+'_fMC_vs_nH_T.png', std_bars=True, style='color')
	dmol_vs_params('CinCO', ['nH', 'T'], galaxies, bin_nums=50, time=None, labels=None, foutname='check_snap_'+str(snap_num)+'_CinCO_vs_nH_T.png', std_bars=True, style='color')

	DZ_vs_params(['nH'], galaxies, bin_nums=40, time=None, labels=labels, foutname='check_snap_'+str(snap_num)+'_DZ_vs_nH.png', std_bars=True, style='color', include_obs=True)

	elems = ['Mg','Si','Fe','O','C']
	elem_depletion_vs_param(elems, 'nH', galaxies, bin_nums=50, time=None, labels=labels, foutname='check_snap_'+str(snap_num)+'_obs_elemental_dep_vs_dens.png', \
						std_bars=True, style='color', include_obs=True)
