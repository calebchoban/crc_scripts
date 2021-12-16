import os
from gizmo_library.time_evolution import Dust_Evo
from gizmo import *
from dust_plots import *

# Directory of snap file
snap_dirs = ['/scratch1/06185/tg854841/cosmo/m12i_res7100_new/output/']
# Snapshots to check
snaps = [201]

cosmological = True
pb_fix=False

# Label for test plots
labels = ['cosmo']

hdir='/scratch1/06185/tg854841/cosmo/m12i_res7100_new/AHF_data/AHF_output/'


# First setup directory for all the plots
plot_dir = './snap_check_plots/'

# First create output directory if needed
try:
	# Create target Directory
	os.mkdir(plot_dir)
	print("Directory " + plot_dir +  " Created ")
except:
	print("Directory " + plot_dir +  " already exists")


# Maximum radius, disk, height, and disk orientation used for getting data in the galactic disk
rmax = 20 # kpc
disk_height = 5 # kpc

for snap_num in snaps:
	galaxies = []
	for j,snap_dir in enumerate(snap_dirs):
		print("Snap Dirc: ",snap_dir)
		print("Snap Num:",snap_num)
		label = labels[j]

		create_visualization(snap_dir, snap_num, image_key='star', foutprefix='test_image', fov=50, pixels=2048)

		sp = load_snap(snap_dir, snap_num, cosmological=cosmological)
		galaxies += [sp]


		print("NumParts:", sp.npart)
		print("Dust Implementation:", sp.dust_impl)
		print("Number of Metal Elements:",sp.Flag_Metals)
		print("Number of Dust Elements:",sp.Flag_DustMetals)
		print("Number of Dust Species:",sp.Flag_DustSpecies)

		flag_species = 0
		if sp.dust_impl=='species':
			flag_species = 1



		G=sp.loadpart(0)
		nH = G.get_property('nH')
		T = G.get_property('T')

		print("*****************************************************")
		print("First check all particles in snapshot for any issues")
		print("*****************************************************")

		print("\n########################################\n")

		print("Checking for Nans...\n")
		nan_ind = np.argwhere(np.isnan(G.dz).any(axis=1) | np.isnan(G.dzs).any(axis=1) | np.isnan(G.spec).any(axis=1))

		if len(nan_ind) > 0:
			print("%i particles with NaNs detected"%len(nan_ind))
			print("Dust Metals:",G.dz[nan_ind])
			if flag_species:
				print("Species:",G.spec[nan_ind])
			print("Sources:",G.dzs[nan_ind])
			print("nH:", nH[nan_ind])
			print("T:", T[nan_ind])

		print("########################################\n")

		print("Checking for negative numbers...\n")
		neg_ind = np.argwhere(np.logical_or(np.any(G.dz<0,axis=1),np.any(G.dzs<0,axis=1),np.any(G.spec<0,axis=1)))
		if len(neg_ind) > 0:
			print("%i particles with negative numbers detected"%len(neg_ind))
			print("Dust Metals:",G.dz[neg_ind])
			if flag_species:
				print("Species:",G.spec[neg_ind])
			print("Sources:",G.dzs[neg_ind])
			print("nH:", nH[neg_ind])
			print("T:", T[neg_ind])


		print("########################################\n")

		print("Checking for too much dust compared to metals...\n")
		over_ind = np.argwhere(np.any(G.dz>G.z[:,:11],axis=1)).flatten()
		if len(over_ind) > 0:
			print("%i particles with too much dust detected"%len(over_ind))
			print("Metals:",G.z[over_ind])
			print("D/Z:",G.dz[over_ind]/G.z[over_ind,:11])
			if flag_species:
				print("Species:",G.spec[over_ind])
			print("Sources:",G.dzs[over_ind])
			print("nH:", nH[over_ind])
			print("T:", T[over_ind])
			print("\t fH2:",G.fH2[over_ind])
			print("\t fMC:",G.fMC[over_ind])
			print("\t CinCO:",G.CinCO[over_ind]/G.z[over_ind,2],"\n")


		if flag_species:

			print("########################################\n")

			print("Checking dust metals and dust species add up...\n")
			# Maximum allowed error between species and dust metals
			abs_error = 1E-2
			# Min DZ to look for error. Helps avoid vanishingly small amounts of dust which are prone
			# to precision errors
			min_DZ = 1E-2
			# Add up the elements from each dust species
			dust_metals = np.zeros(np.shape(G.dz))
			sil_num_atoms = [3.631,1.06,1.,0.571] # O, Mg, Si, Fe
			sil_elems_index = [4,6,7,10] # O,Mg,Si,Fe
			dust_formula_mass = 0

			print("%i dust species detected."%sp.Flag_DustSpecies)
			if sp.Flag_DustSpecies==4:
				print("Assuming silicates, carbonaceous, SiC, and free-flying metallic iron...\n")
				# Silicates
				for k in range(len(sil_num_atoms)):
					dust_formula_mass += sil_num_atoms[k] * config.ATOMIC_MASS[sil_elems_index[k]]
				for k in range(len(sil_num_atoms)):
					dust_metals[:,sil_elems_index[k]] += G.spec[:,0] * sil_num_atoms[k] * config.ATOMIC_MASS[sil_elems_index[k]] / dust_formula_mass

				# Carbon
				dust_metals[:,2] += G.spec[:,1]

				# Silicon Carbide
				dust_formula_mass = config.ATOMIC_MASS[2] + config.ATOMIC_MASS[7]
				dust_metals[:,2] += G.spec[:,2] * config.ATOMIC_MASS[2] / dust_formula_mass
				dust_metals[:,7] += G.spec[:,2] * config.ATOMIC_MASS[7] / dust_formula_mass

				# Iron
				dust_metals[:,10] += G.spec[:,3]

			elif sp.Flag_DustSpecies==5:
				print("Assuming silicates, carbonaceous, SiC, free-flying metallic iron, and oxygen reservoir...\n")
				# Silicates
				for k in range(len(sil_num_atoms)):
					dust_formula_mass += sil_num_atoms[k] * config.ATOMIC_MASS[sil_elems_index[k]]
				for k in range(len(sil_num_atoms)):
					dust_metals[:,sil_elems_index[k]] += G.spec[:,0] * sil_num_atoms[k] * config.ATOMIC_MASS[sil_elems_index[k]] / dust_formula_mass

				# Carbon
				dust_metals[:,2] += G.spec[:,1]

				# Silicon Carbide
				dust_formula_mass = config.ATOMIC_MASS[2] + config.ATOMIC_MASS[7]
				dust_metals[:,2] += G.spec[:,2] * config.ATOMIC_MASS[2] / dust_formula_mass
				dust_metals[:,7] += G.spec[:,2] * config.ATOMIC_MASS[7] / dust_formula_mass

				# Iron
				dust_metals[:,10] += G.spec[:,3]

				# Oxygen Reservoir
				dust_metals[:,4] += G.spec[:,4]

			elif sp.Flag_DustSpecies==6:
				print("Assuming silicates, carbonaceous, SiC, free-flying metallic iron, oxygen reservoir, and metallic iron inclusions...\n")
				# Iron in silicates comes in the form of a separate dust species 'iron inclusions'
				sil_num_atoms = [3.631,1.06,1.] # O, Mg, Si, Fe
				sil_elems_index = [4,6,7] # O,Mg,Si,Fe

				# Silicates
				for k in range(len(sil_num_atoms)):
					dust_formula_mass += sil_num_atoms[k] * config.ATOMIC_MASS[sil_elems_index[k]]
				for k in range(len(sil_num_atoms)):
					dust_metals[:,sil_elems_index[k]] += G.spec[:,0] * sil_num_atoms[k] * config.ATOMIC_MASS[sil_elems_index[k]] / dust_formula_mass

				# Carbon
				dust_metals[:,2] += G.spec[:,1]

				# Silicon Carbide
				dust_formula_mass = config.ATOMIC_MASS[2] + config.ATOMIC_MASS[7]
				dust_metals[:,2] += G.spec[:,2] * config.ATOMIC_MASS[2] / dust_formula_mass
				dust_metals[:,7] += G.spec[:,2] * config.ATOMIC_MASS[7] / dust_formula_mass

				# Free-Flying Iron and Iron Inclusions
				dust_metals[:,10] += G.spec[:,3] + G.spec[:,5]

				# Oxygen Reservoir
				dust_metals[:,4] += G.spec[:,4]

			else:
				print("\t Number of dust species not supported for this check:",sp.Flag_DustSpecies)

			dust_metals[:,0]=np.sum(dust_metals[:,2:],axis=1)
			bad_ind = np.argwhere(np.logical_and(np.any(~np.isclose(G.dz, dust_metals, rtol=abs_error, atol=0,equal_nan=True),axis=1),G.dz[:,0]/G.z[:,0]>min_DZ)).flatten()

			if len(bad_ind) > 0:
				bad_ind = bad_ind[:5]
				print("%i particles with D/Z>%f and element and species not matching by %f%% "%(len(bad_ind),min_DZ,abs_error*100))
				print("Dust Metals:",G.dz[bad_ind])
				print("Dust Metals from Species:",dust_metals[bad_ind])
				print("D/Z:",G.dz[bad_ind]/G.z[bad_ind,:11])
				print("Species:",G.spec[bad_ind])
				print("Sources:",G.dzs[bad_ind])
				print("nH:", nH[bad_ind])
				print("T:", T[bad_ind])
				print("\t fH2:",G.fH2[bad_ind])
				print("\t fMC:",G.fMC[bad_ind])
				print("\t CinCO:",G.CinCO[bad_ind]/G.z[bad_ind,2],"\n")
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
		print("\t fH2: %e \t fMC: %e \t CinCO: %e \n"%(G.fH2[max_ind],G.fMC[max_ind],G.CinCO[max_ind]/G.z[max_ind,2]))


		if flag_species and sp.Flag_DustSpecies>4:
			print("Particle with Max O Reservoir...")
			max_ind = np.nanargmax(G.spec[:,4])
			print("\t D/Z:",G.dz[max_ind]/G.z[max_ind,:11])
			print("\t Dust Metals:",G.dz[max_ind])
			print("\t Metals:",G.z[max_ind])
			print("\t Species:",G.spec[max_ind])
			print("\t Sources:",G.dzs[max_ind])
			print("\t nH:", nH[max_ind])
			print("\t T:", G.T[max_ind])
			print("\t Sum of Species: %e \t Sum of Elements: %e Total Dust: %e\n"%(np.sum(G.spec[max_ind]),np.sum(G.dz[max_ind,2:]),G.dz[max_ind,0]))
			print("\t fH2: %e \t fMC: %e \t CinCO: %e \n"%(G.fH2[max_ind],G.fMC[max_ind],G.CinCO[max_ind]/G.z[max_ind,2]))


		del(G)

		if cosmological:
			print("*****************************************************")
			print("Since this snapshot is cosmological let's look at \n gas in the halo and make some plots!")
			print("*****************************************************")

			hl=sp.loadhalo(id=-1,mode='AHF',hdir=hdir)
			print(hl.id,hl.npart)
			print(hl.mvir/1E10,hl.mgas/1E10)
			print(hl.xc,hl.yc,hl.zc,hl.rvir)


			halo = load_halo(snap_dir, snap_num, cosmological=cosmological, id=-1, mode='AHF',hdir=hdir)
			halo.set_zoom(rout=0.5)

			G = halo.loadpart(0)
			L=0.8*0.5*hl.rvir
			Lz=0.5*L

			config.PROP_INFO['sigma_gas'][1]=[1E0,1E3] # Increase the density range
			snap_projection(['sigma_gas','sigma_metals','sigma_dust', 'D/Z'], halo, L=L, Lz=Lz, pixel_res=0.1,
							labels=['gas','metals','dust','D/Z'], color_map=['inferno','viridis','cividis','cividis'],
							foutname=label+'_halo_0.5rvir_snap_projection.png')
			snap_projection(['sigma_sil','sigma_carb','sigma_iron', 'sigma_ORes'], halo, L=L, Lz=Lz, pixel_res=0.1,
					labels=['silicates','carbon','iron','O res'], color_map='inferno',
							foutname=label+'_halo_0.5rvir_dust_snap_projection.png')
			config.PROP_INFO['sigma_gas'][1]=[1E0,1E2] # Increase the density range

			config.PROP_INFO['nH'][1]=[1.1E-3, 0.9E4] # Increase the density range
			config.PROP_INFO['T'][1]=[1.1E1, 2E6] # Increase the temp range
			binned_phase_plot('M_gas', [halo], bin_nums=250, labels=None, color_map='plasma',
							  foutname=label+'_halo_0.5rvir_phase_plot.png')
			binned_phase_plot('D/Z', [halo], bin_nums=250, labels=None, color_map='magma',
							  foutname=label+'_halo_0.5rvir_DZ_phase_plot.png')
			config.PROP_INFO['nH'][1]=[1.1E-2, 0.9E4] # Increase the density range


			halo.set_zoom(rout=0.1)
			L=0.8*0.1*hl.rvir
			Lz=0.5*L

			config.PROP_INFO['sigma_gas'][1]=[1E0,1E3] # Increase the density range
			snap_projection(['sigma_gas','sigma_metals','sigma_dust', 'D/Z'], halo, L=L, Lz=Lz, pixel_res=0.05,
							labels=['gas','metals','dust','D/Z'], color_map=['inferno','viridis','cividis','cividis'],
							foutname=label+'_halo_0.1rvir_snap_projection.png')
			snap_projection(['sigma_sil','sigma_carb','sigma_iron', 'sigma_ORes'], halo, L=L, Lz=Lz, pixel_res=0.05,
					labels=['silicates','carbon','iron','O res'], color_map='inferno',
							foutname=label+'_halo_0.1rvir_dust_snap_projection.png')
			config.PROP_INFO['sigma_gas'][1]=[1E0,1E2] # Increase the density range

			config.PROP_INFO['nH'][1]=[1.1E-3, 2E3] # Increase the density range
			config.PROP_INFO['T'][1]=[1.1E1, 2E6] # Increase the temp range
			binned_phase_plot('M_gas', [halo], bin_nums=250, labels=None, color_map='plasma',
							  foutname=label+'_halo_0.1rvir_phase_plot.png')
			binned_phase_plot('D/Z', [halo], bin_nums=250, labels=None, color_map='magma',
							  foutname=label+'_halo_0.1rvir_DZ_phase_plot.png')
			config.PROP_INFO['nH'][1]=[1.1E-2, 0.9E4] # Increase the density range

			dmol_vs_props(['fH2','fMC','CinCO'], ['nH','T'], [halo], labels=None, bin_nums=100, std_bars=True,
				foutname=label+'_halo_0.1rvir_dmol_vs_props.png')

			galaxy_int_DZ_vs_prop(['Z','O/H'],[halo],labels=labels,
								  foutname=label+'_halo_0.1rvir_galaxy_int_DZ_vs_Z.png')

			del(halo)


		print("*****************************************************")
		print("Now let's look at gas in the disc and make some more plots!")
		print("*****************************************************")

		disk = sp.loaddisk(id=-1,mode='AHF',hdir=hdir,rmax=rmax,height=disk_height)
		disk.set_disk()

		L=0.8*rmax
		Lz=disk_height



		config.PROP_INFO['sigma_gas'][1]=[1E0,1E3] # Increase the density range
		snap_projection(['sigma_gas','sigma_metals','sigma_dust', 'D/Z'], disk, L=L, Lz=Lz, pixel_res=0.05,
						labels=['gas','metals','dust','D/Z'], color_map=['inferno','viridis','cividis','cividis'],
						foutname=label+'_disk_gas_snap_projection.png')
		snap_projection(['sigma_sil','sigma_carb','sigma_iron', 'sigma_ORes'], disk, L=L, Lz=Lz, pixel_res=0.05,
				labels=['silicates','carbon','iron','O res'], color_map='inferno', foutname=
						label+'_disk_dust_snap_projection.png')
		config.PROP_INFO['sigma_gas'][1]=[1E0,1E2] # Increase the density range


		config.PROP_INFO['nH'][1]=[1.1E-3, 2E3] # Increase the density range
		config.PROP_INFO['T'][1]=[1.1E1, 2E6] # Increase the temp range
		binned_phase_plot('M_gas', [disk], bin_nums=250, labels=None, color_map='plasma',
						  foutname=label+'_disk_phase_plot.png')
		binned_phase_plot('D/Z', [disk], bin_nums=250, labels=None, color_map='magma',
						  foutname=label+'_disk_DZ_phase_plot.png')
		config.PROP_INFO['nH'][1]=[1.1E-2, 0.9E4] # Increase the density range

		dmol_vs_props(['fH2','fMC','CinCO'], ['nH','T'], [disk], labels=None, bin_nums=100, std_bars=True,
			foutname=label+'_disk_dmol_vs_props.png')

		galaxy_int_DZ_vs_prop(['Z','O/H'],[disk],labels=None,foutname=label+'_halo_DZ_disk.png')


		plot_prop_vs_prop(['nH','T'],['D/Z','D/Z'],[disk],labels=None,std_bars=True, style='color-linestyle',
						foutname=label+'_disk_DZ_vs_nH_T.png')
		elems = ['Mg','Si','Fe','O','C']
		plot_elem_depletion_vs_prop(elems, 'nH', [disk], bin_nums=100, labels=None,
						foutname=label+'_disk_'+str(snap_num)+'_obs_elemental_dep_vs_dens.png',
						std_bars=True, style='color-linestyle', include_obs=True)

		plot_obs_prop_vs_prop(['sigma_gas_neutral','r','Z'], ['D/Z','D/Z','D/Z'], [disk], pixel_res=2, bin_nums=40,
						labels=None, foutname=label+'_disk_obs_DZ_vs_surf_'+str(snap_num)+'.png',
						std_bars=True, style='color-linestyle', include_obs=True)
		plot_obs_prop_vs_prop(['r','r25'], ['D/Z','D/Z'], [disk], pixel_res=2, bin_nums=40, labels=None,
						foutname=label+'_disk_obs_DZ_vs_r25_'+str(snap_num)+'.png',
						std_bars=True, style='color-linestyle', include_obs=True)

		del(disk)

		# TODO: Add ability to plot mock Hubble images using Phil's visualization routine
