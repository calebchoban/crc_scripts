from gizmo import *
from dust_plots import *

# Name prefix for all saved images. Useful if you run this for multiple sims.
prefix=''

# Directory of snap file
snap_dirs = ['/scratch1/06185/tg854841/m12i_FIRE3_dust_test/output/',
			 '/scratch1/06185/tg854841/cosmo/Species/m12i_res7100_FIRE3/output/']
snap_dirs = ['/scratch1/06185/tg854841/development_test/output',
			 '/scratch1/06185/tg854841/init_Z_dust_tests/fiducial_init_dust/output']
snap_dirs = ['/work/06185/tg854841/frontera/non_cosmo/Species/nano_Fe/output/',
			 '/scratch1/06185/tg854841/updated_AGB_test/output/']
# Snapshots to check
snaps = [1,2,3,10]

cosmological = False
pb_fix=True

# Label for test plots
names  =  ['old','new']
labels = ['old','new']

# Directory for AHF halo files for snaps. Only used for cosmological simulations
hdir=None

# First setup directory for all the plots
plot_dir = './snap_check_plots/'

# Maximum radius, disk, height, and disk orientation used for getting data in the galactic disk
rmax = 10 # kpc
disk_height = 4 # kpc

# These flags determine what check are made for each snapshot
create_Hubble_image 	= False # Creates mock hubble images
check_values 			= False  # Check dust values in snapshots for unphysical values and inconsistencies
create_individual_plots = False  # Creates various individual plots for each snapshot
create_halo_plots		= False	# Creates plots of the galactic halo
zooms = [2.0]					# Sets the zoom parameter for the halo plots in case you want multiple zooms
create_disk_plots		= False	# Creates plots of the galactic disk
create_comparison_plots = False # Create various plots comparing simulations



# First create output directory if needed
try:
	# Create target Directory
	os.mkdir(plot_dir)
	print("Directory " + plot_dir +  " Created ")
except:
	print("Directory " + plot_dir +  " already exists")

for snap_num in snaps:
	galaxies = []
	for j,snap_dir in enumerate(snap_dirs):
		print("Snap Dirc: ",snap_dir)
		print("Snap Num:",snap_num)
		label = labels[j]
		name = names[j]

		name_prefix = plot_dir+prefix+'_'+name+'_'+str(snap_num)

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

		print("Dust Source Masses")
		print(np.sum(G.dzs*G.dz[:,0,np.newaxis]*G.m[:,np.newaxis],axis=0))
		print("Total Dust Mass")
		print(np.sum(G.dz[:,0]*G.m,axis=0))
		print("Dust Species Mass")
		print(np.sum(G.spec*G.m[:,np.newaxis],axis=0))

		if check_values:
			print("*****************************************************")
			print("First check all particles in snapshot for any issues")
			print("*****************************************************")

			print("\n########################################\n")

			print("Checking for Nans...\n")
			nan_ind = np.argwhere(np.isnan(G.dz).any(axis=1) | np.isnan(G.dzs).any(axis=1) | np.isnan(G.spec).any(axis=1)).flatten()

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
				print("\t fdense:",G.fdense[over_ind])
				print("\t CinCO:",G.CinCO[over_ind]/G.z[over_ind,2],"\n")


			if flag_species:

				print("########################################\n")

				print("Checking dust metals and dust species add up...\n")
				# Maximum allowed error between species and dust metals
				abs_error = 1E-2
				# Min DZ and Z to look for error. Helps avoid vanishingly small amounts of dust which are prone
				# to precision errors
				min_DZ = 1E-2
				min_Z = 1E-4
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
				bad_ind = np.logical_and(np.any(~np.isclose(G.dz, dust_metals, rtol=abs_error, atol=0,equal_nan=True),axis=1),G.dz[:,0]/G.z[:,0]>min_DZ)
				bad_ind = np.argwhere(np.logical_and(bad_ind,G.z[:,0]>min_Z*config.SOLAR_Z)).flatten()
				if len(bad_ind) > 0:
					print("%i particles with D/Z>%f and Z>%f Z_solar and element and species not matching by %f%% "%(len(bad_ind),min_DZ,min_Z,abs_error*100))
					print("Dust Metals:",G.dz[bad_ind])
					print("Dust Metals from Species:",dust_metals[bad_ind])
					print("D/Z:",G.dz[bad_ind]/G.z[bad_ind,:11])
					print("Species:",G.spec[bad_ind])
					print("Sources:",G.dzs[bad_ind])
					print("nH:", nH[bad_ind])
					print("T:", T[bad_ind])
					print("\t fH2:",G.fH2[bad_ind])
					print("\t fdense:",G.fdense[bad_ind])
					print("\t CinCO:",G.CinCO[bad_ind]/G.z[bad_ind,2],"\n")
					print("\t Sum of Species::",np.sum(G.spec[bad_ind],axis=1))
					print("\t Sum of Elements:",np.sum(G.dz[bad_ind,2:],axis=1))
					print("\t Total Dust:",G.dz[bad_ind,0])
				else:
					print("No particles with D/Z>%f, Z>%f Z_solar, and element and species not matching by %f%% \n"%(min_DZ,min_Z,abs_error*100))


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
			print("\t fH2: %e \t fdense: %e \t CinCO: %e \n"%(G.fH2[max_ind],G.fdense[max_ind],G.CinCO[max_ind]/G.z[max_ind,2]))


			print("Particle with Max fdense...")
			max_ind = np.nanargmax(G.fdense)
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
			print("\t fH2: %e \t fdense: %e \t CinCO: %e \n"%(G.fH2[max_ind],G.fdense[max_ind],G.CinCO[max_ind]/G.z[max_ind,2]))


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
				print("\t fH2: %e \t fdense: %e \t CinCO: %e \n"%(G.fH2[max_ind],G.fdense[max_ind],G.CinCO[max_ind]/G.z[max_ind,2]))

				del(G)



		if create_Hubble_image:
			print("*****************************************************")
			print("Creating mock Hubble images....")
			print("*****************************************************")

			#create_visualization(snap_dir, snap_num, image_key='star', foutprefix=label+'_mock_Hubble_image', fov=10, pixels=2048)
			from visualization.image_maker import image_maker, edgeon_faceon_projection
			faceon_image = edgeon_faceon_projection(snap_dir, snap_num, centering='', field_of_view=10, image_key='star',
												pixels=2048, faceon=True, load_dust=False)



		if create_individual_plots:
			if cosmological and create_halo_plots:
				print("*****************************************************")
				print("Since this snapshot is cosmological let's look at \n gas in the halo and make some plots!")
				print("*****************************************************")

				hl=sp.loadhalo(id=-1,mode='AHF',hdir=hdir)
				print(hl.id,hl.npart)
				print(hl.mvir/1E10,hl.mgas/1E10)
				print(hl.xc,hl.yc,hl.zc,hl.rvir)

				halo = load_halo(snap_dir, snap_num, cosmological=cosmological, id=-1, mode='AHF',hdir=hdir)


				for zoom in zooms:
					halo.set_zoom(rout=zoom)
					L=0.8*zoom*halo.rvir
					Lz=0.5*L
					res = L/200

					G = halo.loadpart(0)
					S = halo.loadpart(4)
					print("Zoom =",str(zoom),'Rvir')
					print("Halo Gas Mass:",np.sum(G.get_property('M'))/1E10,"10^10 M_Solar")
					print("Halo Star Mass:",np.sum(S.get_property('M'))/1E10,"10^10 M_Solar")

					z = halo.sp.redshift

					config.PROP_INFO['sigma_gas'][1]=[1E0,1E3] # Increase the density range
					config.PROP_INFO['sigma_H2'][1]=[1E0,1E3] # Increase the density range
					config.PROP_INFO['T'][1]=[1E3,1E7] # Increase the T range
					snap_projection(['sigma_gas','T'], halo, L=L, Lz=Lz,
									pixel_res=res,labels=['gas \n z=%.3g'%z,'T \n z=%.3g'%z], color_map=['inferno','viridis'], show_rvir=True,
									foutname=name_prefix+'_halo_'+str(zoom)+'rvir_gasT_snap_projection.png')
					config.PROP_INFO['sigma_gas'][1]=[1E0,1E2] # Increase the density range
					config.PROP_INFO['sigma_H2'][1]=[1E0,1E2] # Increase the density range
					config.PROP_INFO['T'][1]=[1.1E1,0.9E5] # Increase the T range

					continue

					config.PROP_INFO['sigma_gas'][1]=[1E0,1E3] # Increase the density range
					config.PROP_INFO['sigma_H2'][1]=[1E0,1E3] # Increase the density range
					config.PROP_INFO['sigma_star'][1]=[1E0,1E4] # Increase the density range
					snap_projection(['sigma_gas','sigma_H2','sigma_metals','sigma_dust'], halo, L=L, Lz=Lz,
									pixel_res=res,labels=['gas','H2','metals','dust'], color_map='inferno',
									foutname=name_prefix+'_halo_'+str(zoom)+'rvir_gas_snap_projection.png')
					snap_projection(['sigma_sil','sigma_carb','sigma_iron', 'sigma_ORes'], halo, L=L, Lz=Lz, pixel_res=res,
							labels=['silicates','carbon','iron','O res'], color_map='inferno',
									foutname=name_prefix+'_halo_'+str(zoom)+'rvir_dust_snap_projection.png')
					config.PROP_INFO['sigma_gas'][1]=[1E0,1E2] # Increase the density range
					config.PROP_INFO['sigma_H2'][1]=[1E0,1E2] # Increase the density range
					config.PROP_INFO['sigma_star'][1]=[1E0,1E2] # Increase the density range

					config.PROP_INFO['nH'][1]=[1.1E-3, 0.9E4] # Increase the density range
					config.PROP_INFO['T'][1]=[1.1E1, 2E6] # Increase the temp range
					config.PROP_INFO['M_gas'][1]=[1E4,1E8] # Increase the density range
					binned_phase_plot('M_gas', [halo], bin_nums=250, labels=None, color_map='plasma',
									  foutname=name_prefix+'_halo_'+str(zoom)+'rvir_phase_plot.png')
					binned_phase_plot('D/Z', [halo], bin_nums=250, labels=None, color_map='magma',
									  foutname=name_prefix+'_halo_'+str(zoom)+'rvir_DZ_phase_plot.png')
					config.PROP_INFO['nH'][1]=[1.1E-2, 0.9E4] # Increase the density range
					config.PROP_INFO['M_gas'][1]=[1E8,1E11] # Increase the density range

					dmol_vs_props(['fH2','fdense','CinCO'], ['nH','T'], [halo], labels=None, bin_nums=100, std_bars=True,
					foutname=name_prefix+'_halo_'+str(zoom)+'rvir_dmol_vs_props.png')

					galaxy_int_DZ_vs_prop(['Z','O/H'],[halo],labels=labels,
									  foutname=name_prefix+'_halo_'+str(zoom)+'rvir_galaxy_int_DZ_vs_Z.png')

				del(halo)

			if create_disk_plots:
				print("*****************************************************")
				print("Now let's look at gas in the disc and make some more plots!")
				print("*****************************************************")

				if cosmological:
					disk = sp.loaddisk(id=-1,mode='AHF',hdir=hdir,rmax=rmax,height=disk_height)
				else:
					disk = sp.loaddisk(id=-1,mode='AHF',rmax=rmax,height=disk_height)
				disk.set_disk()

				L=0.8*rmax
				Lz=disk_height

				res = L/200

				G = disk.loadpart(0)
				S = disk.loadpart(4)
				print("Disk Gas Mass:",np.sum(G.get_property('M'))/1E10,"10^10 M_Solar")
				print("Disk Star Mass:",np.sum(S.get_property('M'))/1E10,"10^10 M_Solar")

				z = disk.sp.redshift

				config.PROP_INFO['sigma_gas'][1]=[1E0,1E3] # Increase the density range
				config.PROP_INFO['sigma_H2'][1]=[1E0,1E3] # Increase the density range
				config.PROP_INFO['T'][1]=[1E3,1E7] # Increase the T range
				snap_projection(['sigma_gas','T'], disk, L=L, Lz=Lz,
								pixel_res=res,labels=['gas \n z=%.3g'%z,'T \n z=%.3g'%z], color_map=['inferno','viridis'], show_rvir=True,
								foutname=name_prefix+'_disk_gasT_snap_projection.png')
				config.PROP_INFO['sigma_gas'][1]=[1E0,1E2] # Increase the density range
				config.PROP_INFO['sigma_H2'][1]=[1E0,1E2] # Increase the density range
				config.PROP_INFO['T'][1]=[1.1E1,0.9E5] # Increase the T range


				config.PROP_INFO['sigma_gas'][1]=[1E0,1E3] # Increase the density range
				config.PROP_INFO['sigma_H2'][1]=[1E0,1E3] # Increase the density range
				config.PROP_INFO['sigma_star'][1]=[1E0,1E4] # Increase the density range
				snap_projection(['sigma_gas','sigma_H2','sigma_metals','sigma_dust'], disk, L=L, Lz=Lz,
								pixel_res=res,labels=['gas','H2','metals','dust'], color_map='inferno',
								foutname=name_prefix+'_disk_gas_snap_projection.png')
				snap_projection(['sigma_sil','sigma_carb','sigma_iron', 'sigma_ORes'], disk, L=L, Lz=Lz, pixel_res=res,
						labels=['silicates','carbon','iron','O res'], color_map='inferno',
								foutname=name_prefix+'_disk_dust_snap_projection.png')
				config.PROP_INFO['sigma_gas'][1]=[1E0,1E2] # Increase the density range
				config.PROP_INFO['sigma_H2'][1]=[1E0,1E2] # Increase the density range
				config.PROP_INFO['sigma_star'][1]=[1E0,1E2] # Increase the density range

				config.PROP_INFO['nH'][1]=[1.1E-3, 2E3] # Increase the density range
				config.PROP_INFO['T'][1]=[1.1E1, 2E6] # Increase the temp range
				config.PROP_INFO['M_gas'][1]=[1E4,1E8] # Increase the density range
				binned_phase_plot('M_gas', [disk], bin_nums=250, labels=None, color_map='plasma',
								  foutname=name_prefix+'_disk_phase_plot.png')
				binned_phase_plot('D/Z', [disk], bin_nums=250, labels=None, color_map='magma',
								  foutname=name_prefix+'_disk_DZ_phase_plot.png')
				config.PROP_INFO['nH'][1]=[1.1E-2, 0.9E4] # Increase the density range
				config.PROP_INFO['M_gas'][1]=[1E8,1E11] # Increase the density range


				dmol_vs_props(['fH2','fdense','CinCO'], ['nH','T'], [disk], labels=None, bin_nums=100, std_bars=True,
					foutname=name_prefix+'_disk_dmol_vs_props.png')

				galaxy_int_DZ_vs_prop(['Z','O/H'],[disk],labels=None,foutname=name_prefix+'_halo_DZ_disk.png')


				plot_prop_vs_prop(['nH','T'],['D/Z','D/Z'],[disk],labels=None,std_bars=True, style='color-linestyle',
								foutname=name_prefix+'_disk_DZ_vs_nH_T.png')
				elems = ['Mg','Si','Fe','O','C']
				plot_elem_depletion_vs_prop(elems, 'nH', [disk], bin_nums=100, labels=None,
								foutname=name_prefix+'_disk_obs_elemental_dep_vs_dens.png',
								std_bars=True, style='color-linestyle', include_obs=True)

				plot_obs_prop_vs_prop(['sigma_gas_neutral','r','Z'], ['D/Z','D/Z','D/Z'], [disk], pixel_res=2, bin_nums=40,
								labels=None, foutname=name_prefix+'_disk_obs_DZ_vs_surf.png',
								std_bars=True, style='color-linestyle', include_obs=True)
				plot_obs_prop_vs_prop(['r','r25'], ['D/Z','D/Z'], [disk], pixel_res=2, bin_nums=40, labels=None,
								foutname=name_prefix+'_disk_obs_DZ_vs_r25.png',
								std_bars=True, style='color-linestyle', include_obs=True)

				del(disk)

	if create_comparison_plots and len(snap_dirs) > 1:
		config.PROP_INFO['nH'][1]=[1.1E-3, 2E3] # Increase the density range
		config.PROP_INFO['T'][1]=[1.1E1, 2E6] # Increase the temp range
		config.PROP_INFO['M_gas'][1]=[1E4,1E8] # Increase the density range
		binned_phase_plot('M_gas', galaxies, bin_nums=250, labels=names, color_map='plasma',
					  foutname=name_prefix+'_disk_phase_plot.png')

		for k,sp in enumerate(galaxies):
			t, sfr = sp.get_SFH()
			plt.plot(t,sfr, label=names[k])
		plt.xlabel('Time (Gyr)')
		plt.ylabel(r'SFR ($M_{\odot}$/yr)')
		plt.yscale('log')
		plt.ylim([1E-1,1E2])
		plt.legend()
		plt.savefig(name_prefix+'_sfh.png')
		plt.close()

		for k,sp in enumerate(galaxies):
			t, sfr = sp.get_SFH(cum=1)
			plt.plot(t,sfr, label=names[k])
		plt.xlabel('Time (Gyr)')
		plt.ylabel(r'Cumulative SFR ($M_{\odot}$)')
		plt.yscale('log')
		plt.ylim([1E7,1E11])
		plt.legend()
		plt.savefig(name_prefix+'_cum_sfh.png')
		plt.close()