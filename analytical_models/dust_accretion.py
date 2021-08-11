import numpy as np
from gizmo_library import config
from gizmo_library import utils

# Theoretical dust yields for all sources of creation


# Solar Abundace values used in FIRE-2 (Anders+) and FIRE-3 (Asplund+)
def solarMetallicity(elem, FIRE_v=2):
	SolarAbundances = np.zeros(11)
	if FIRE_v<=2:
		SolarAbundances[0]=0.02;	 # Z
		SolarAbundances[1]=0.28;	# He  (10.93 in units where log[H]=12, so photospheric mass fraction -> Y=0.2485 [Hydrogen X=0.7381]; Anders+Grevesse Y=0.2485, X=0.7314)
		SolarAbundances[2]=3.26e-3; # C   (8.43 -> 2.38e-3, AG=3.18e-3)
		SolarAbundances[3]=1.32e-3; # N   (7.83 -> 0.70e-3, AG=1.15e-3)
		SolarAbundances[4]=8.65e-3; # O   (8.69 -> 5.79e-3, AG=9.97e-3)
		SolarAbundances[5]=2.22e-3; # Ne  (7.93 -> 1.26e-3, AG=1.72e-3)
		SolarAbundances[6]=9.31e-4; # Mg  (7.60 -> 7.14e-4, AG=6.75e-4)
		SolarAbundances[7]=1.08e-3; # Si  (7.51 -> 6.71e-4, AG=7.30e-4)
		SolarAbundances[8]=6.44e-4; # S   (7.12 -> 3.12e-4, AG=3.80e-4)
		SolarAbundances[9]=1.01e-4; # Ca  (6.34 -> 0.65e-4, AG=0.67e-4)
		SolarAbundances[10]=1.73e-3; # Fe (7.50 -> 1.31e-3, AG=1.92e-3)
	elif FIRE_v>2:
		SolarAbundances[0]=0.0142;	# Z
		SolarAbundances[1]=0.2703;	# He  
		SolarAbundances[2]=2.53e-3; # C   
		SolarAbundances[3]=7.41e-4; # N   
		SolarAbundances[4]=6.13e-3; # O   
		SolarAbundances[5]=1.34e-3; # Ne  
		SolarAbundances[6]=7.57e-4; # Mg  
		SolarAbundances[7]=7.12e-4; # Si  
		SolarAbundances[8]=3.31e-4; # S   
		SolarAbundances[9]=6.87e-5; # Ca  
		SolarAbundances[10]=1.38e-3; # Fe 


	return SolarAbundances[elem]



# The "Elemental" implementation dust accrection growth timescale
def elementalGrowthTime(temp, dens):
	ref_dens = H_MASS # g cm^-3
	T_ref = 20. # K
	time_ref = 0.2 # Gyr
	return time_ref * (ref_dens / dens) * np.power((T_ref / temp),0.5)


# The "Species"	implementation dust accretion growth timescale assuming growth species is gas form of key element and solar metallicities
def speciesGrowthTime(temp, dens, metallicity, species, fMC=0.3, nanoFe=True):
	# Check if detailed gas metallicity has been given or just total Z
	if hasattr(metallicity, "__len__"):
		assumed_Z = False
		metallicity = metallicity[:,:11] # Don't want r-process elements
		key_elem = np.zeros(np.shape(metallicity[:,0]))
	else:
		assumed_Z = True
		ZZ = metallicity / solarMetallicity(0)
		key_elem = 0
	
	time_ref = 0.00153 # Gyr
	key_in_dust = 0 # number of atoms of key element in one formula unit of the dust species
	dust_formula_mass = 0

	# Check if list has been given
	if hasattr(temp, "__len__") and hasattr(dens, "__len__"):
		sticking_eff = np.zeros(np.shape(temp))
		sticking_eff[temp <= 300] = 1.
		sticking_eff[temp > 300] = 0.
	elif hasattr(temp, "__len__") or hasattr(dens, "__len__"):
		print("Both temp and dens must be either arrays or scalars!")
		return 0
	else:
		if (temp < 300):
			sticking_eff = 1.
		else:
			sticking_eff = 0.
			return None,None

	t_ref_CNM = 0.   # Gyr
	t_ref_MC = 0.     # Gyr
	t_ref = (t_ref_CNM * t_ref_MC) / (fMC * t_ref_CNM + (1.-fMC) * t_ref_MC);

	if species == 'silicates':
		for k in range(4): dust_formula_mass += config.SIL_NUM_ATOMS[k] * config.ATOMIC_MASS[config.SIL_ELEM_INDEX[k]];
		t_ref_CNM = 0.252E-3   # Gyr
		t_ref_MC = 1.38E-3     # Gyr
		# Calculate effective timescale assuming produced dust is redistributed throughout the gas
		t_ref = (t_ref_CNM * t_ref_MC) / (fMC * t_ref_CNM + (1.-fMC) * t_ref_MC)
		if assumed_Z:

			# Since we assume solar metallicities we can just say Si is the key element
			key = 2
			key_elem = config.SIL_ELEM_INDEX[key]
			num_dens = dens * ZZ * solarMetallicity(key_elem)/ (config.ATOMIC_MASS[key_elem]*config.H_MASS)
			key_in_dust = config.SIL_NUM_ATOMS[key]
			cond_dens = 3.13
		else:
			num_dens = np.multiply(metallicity, dens[:, np.newaxis]) / (config.ATOMIC_MASS*config.H_MASS)
			# If snapshot was run with nano-iron dust species, Fe from nano-iron inclusions are assumed to contribute to
			# the Fe needed in silicates
			if nanoFe:
				sil_num_dens = num_dens[:,config.SIL_ELEM_INDEX[:-1]]
				# Find element with lowest number density factoring in number of atoms needed to make one formula unit of the dust species
				key = np.argmin(sil_num_dens / config.SIL_NUM_ATOMS[:-1], axis = 1)
			else:
				sil_num_dens = num_dens[:,config.SIL_ELEM_INDEX]
				# Find element with lowest number density factoring in number of atoms needed to make one formula unit of the dust species
				key = np.argmin(sil_num_dens / config.SIL_NUM_ATOMS, axis = 1)

			num_dens = sil_num_dens[range(sil_num_dens.shape[0]),key]
			key_elem = config.SIL_ELEM_INDEX[key]
			key_in_dust = config.SIL_NUM_ATOMS[key]
			cond_dens = 3.13


	elif species == 'carbon':
		t_ref_CNM = 1.54E-3 # Gyr
		t_ref = t_ref_CNM / (1.-fMC)
		if not assumed_Z:
			key_elem = np.full(np.shape(metallicity[:,0]),2)
		else:
			key_elem = 2

		key_in_dust = 1
		dust_formula_mass = config.ATOMIC_MASS[key_elem]
		cond_dens = 2.25
		if assumed_Z:
			num_dens = dens * ZZ * solarMetallicity(key_elem) / (config.ATOMIC_MASS[key_elem]*config.H_MASS)
		else:
			num_dens = np.multiply(metallicity[:,key_elem[0]], dens) / (config.ATOMIC_MASS[key_elem]*config.H_MASS)

	elif 'iron' in species:
		if 'nano' in species:
			t_ref_CNM = 1.66E-6    # Gyr
			t_ref_MC = 0.139E-3    # Gyr
		else:
			t_ref_CNM = 0.252E-3   # Gyr
			t_ref_MC = 1.38E-3     # Gyr
		# Calculate effective timescale assuming produced dust is redistributed throughout the gas
		t_ref = (t_ref_CNM * t_ref_MC) / (fMC * t_ref_CNM + (1.-fMC) * t_ref_MC)

		if not assumed_Z:
			key_elem = np.full(np.shape(metallicity[:,0]),10)
		else:
			key_elem = 10

		key_in_dust = 1
		dust_formula_mass = config.ATOMIC_MASS[key_elem]
		cond_dens = 7.86
		if assumed_Z:
			num_dens = dens * ZZ * solarMetallicity(key_elem) / (config.ATOMIC_MASS[key_elem]*config.H_MASS)
		else:
			num_dens = np.multiply(metallicity[:,key_elem[0]], dens) / (config.ATOMIC_MASS[key_elem]*config.H_MASS)
	else:
		print(str(species), " is not a valid species")
		return None,None

	print("For " + species + " the key element was found to be ", config.ELEMENTS[key_elem])
	print("Its number density is ", str(num_dens), " g/cm^3")

	acc_time = t_ref * (key_in_dust * np.power(config.ATOMIC_MASS[key_elem],0.5) / (sticking_eff * dust_formula_mass)) * (1. / num_dens) * cond_dens * np.power((1. / temp),0.5)

	if not assumed_Z:
		# Keep track of no accretion due to no key elements being present
		no_acc = np.where(metallicity[range(metallicity.shape[0]),key_elem] <= 0.)
		acc_time[no_acc] = np.inf
		key_elem[no_acc] = -1

	return acc_time, key_elem



# Calculates the Elemental gas-dust accretion timescale in years for all gas particles in the given snapshot gas particle structure
def calc_elem_acc_timescale(G, t_ref_factor=1.):

	t_ref = 0.2E9*t_ref_factor 	# yr
	T_ref = 20					# K
	dens_ref = config.H_MASS	# g cm^-3
	T = G.T
	dens = G.rho*config.UnitDensity_in_cgs
	growth_time = t_ref * (dens_ref/dens) * np.power(T_ref/T,0.5)

	timescales = dict.fromkeys(['Silicates', 'Carbon', 'Iron'], None) 
	timescales['Silicates'] = np.copy(growth_time)
	timescales['Carbon'] = np.copy(growth_time)
	timescales['Iron'] = np.zeros(len(growth_time))

	return timescales

# Calculates the key element for silicates for the Species implementation
def calc_spec_key_elem(G, nano_iron=False):
	elem_num_dens = np.multiply(G.z[:,:len(config.ATOMIC_MASS)], G.rho[:, np.newaxis]*config.UnitDensity_in_cgs) / (config.ATOMIC_MASS*config.H_MASS)
	# If snapshot was run with nano-iron dust species, Fe from nano-iron inclusions are assumed to contribute to
	# the Fe needed in silicates
	if nano_iron:
		sil_num_dens = elem_num_dens[:,config.SIL_ELEM_INDEX[:-1]]
		# Find element with lowest number density factoring in number of atoms needed to make one formula unit of the dust species
		key = np.argmin(sil_num_dens / config.SIL_NUM_ATOMS[:-1], axis = 1)
	else:
		sil_num_dens = elem_num_dens[:,config.SIL_ELEM_INDEX]
		# Find element with lowest number density factoring in number of atoms needed to make one formula unit of the dust species
		key = np.argmin(sil_num_dens / config.SIL_NUM_ATOMS, axis = 1)
	key_num_dens = sil_num_dens[range(sil_num_dens.shape[0]),key]
	key_elem = config.SIL_ELEM_INDEX[key]
	key_in_dust = config.SIL_NUM_ATOMS[key]

	return key_elem, key_num_dens, key_in_dust



# Calculates the Species gas-dust accretion timescale in years for all gas particles in the given snapshot gas particle structure
def calc_spec_acc_timescale(G, nano_iron=False,set_fMC=None):

	T_ref = 300 		# K
	nM_ref = 1E-2   	# reference number density for metals in 1 H cm^-3
	ref_cond_dens = 3	# reference condensed dust species density g cm^-3
	T_cut = 300 		# K cutoff temperature for step func. sticking efficiency

	T = G.T
	if set_fMC != None:
		fMC = np.full(len(T),set_fMC)
	else:
		fMC = G.fMC

	timescales = dict.fromkeys(['Silicates', 'Carbon', 'Iron'], None) 

	###############
    ## SILICATES 
    ###############
	t_ref_CNM = 4.4E6 # years
	t_ref_MC = 23.9E6 # years
	t_ref = (t_ref_CNM * t_ref_MC) / (fMC * t_ref_CNM + (1.-fMC) * t_ref_MC)

	dust_formula_mass = 0.0
	elem_num_dens = np.multiply(G.z[:,:len(config.ATOMIC_MASS)], G.rho[:, np.newaxis]*config.UnitDensity_in_cgs) / (config.ATOMIC_MASS*config.H_MASS)


	for k in range(4): dust_formula_mass += config.SIL_NUM_ATOMS[k] * config.ATOMIC_MASS[config.SIL_ELEM_INDEX[k]];

	key_elem, key_num_dens,key_in_dust = calc_spec_key_elem(G, nano_iron=nano_iron)
	key_mass = config.ATOMIC_MASS[key_elem]
	cond_dens = 3.13
	growth_time = t_ref * key_in_dust * np.sqrt(key_mass) / dust_formula_mass * (cond_dens/ref_cond_dens) * (nM_ref/key_num_dens) * np.power(T_ref/T,0.5)

	growth_time[T>T_cut] = np.inf

	timescales['Silicates'] = np.copy(growth_time)

	###############
    ## CARBONACEOUS
    ###############
	t_ref_CNM = 26.7E6 # years
	t_ref = t_ref_CNM / (1.-fMC)

	key_elem = 2
	key_in_dust = 1
	key_mass = config.ATOMIC_MASS[key_elem]
	dust_formula_mass = key_mass
	cond_dens = 2.25
	key_num_dens = elem_num_dens[:,key_elem]
	growth_time = t_ref * key_in_dust * np.sqrt(key_mass) / dust_formula_mass * (cond_dens/ref_cond_dens) * (nM_ref/key_num_dens) * np.power(T_ref/T,0.5)
	growth_time[T>T_cut] = np.inf
	timescales['Carbon'] = np.copy(growth_time)

	###############
    ## IRON 
    ###############
	if nano_iron:
		t_ref_CNM = 0.029E6 # years
		t_ref_MC = 2.42E6 # years
	else:
		t_ref_CNM = 4.4E6 # years
		t_ref_MC = 23.9E6 # years
	t_ref = (t_ref_CNM * t_ref_MC) / (fMC * t_ref_CNM + (1.-fMC) * t_ref_MC)

	key_elem = 10
	key_in_dust = 1
	dust_formula_mass = config.ATOMIC_MASS[key_elem]
	cond_dens = 7.86
	key_num_dens = elem_num_dens[:,key_elem]
	key_mass = config.ATOMIC_MASS[key_elem]
	growth_time = t_ref * key_in_dust * np.sqrt(key_mass) / dust_formula_mass * (cond_dens/ref_cond_dens) * (nM_ref/key_num_dens) * np.power(T_ref/T,0.5)
	growth_time[T>T_cut] = np.inf
	timescales['Iron'] = np.copy(growth_time)

	return timescales



# Calculates the instantaneous dust production from accertion for the given snapshot gas particle structure
def calc_dust_acc(G, implementation='species', nano_iron=False, O_res=False, set_fMC=None):


	iron_incl = 0.7

	M = G.m*1E10
	fH2 = G.fH2
	if set_fMC == None:
		fMC = G.fMC
	else:
		fMC = np.fill(set_fMC, len(G.m))

	C_in_CO = G.CinCO
	O_in_CO = C_in_CO * G.z[:,2] * config.ATOMIC_MASS[4] / config.ATOMIC_MASS[2] / G.z[:,4]

	# Needed to select arbitrary elements from each row for 2D numpy arrays
	farg = np.arange(len(M))

	if implementation == 'elemental':
		timescales = calc_elem_acc_timescale(G)
		growth_timescale = timescales['Silicates']
		sil_DZ = G.dz[:,[4,6,7,10]]/G.z[:,[4,6,7,10]]
		# Account for O locked in CO which reduces the max amount of O in dust
		sil_DZ[:,0] = np.multiply(sil_DZ[:,0], 1./(1.-O_in_CO))
		sil_DZ[np.logical_or(sil_DZ <= 0,sil_DZ >= 1)] = 1.
		sil_dust_mass = np.multiply(G.dz[:,[4,6,7,10]],M[:,np.newaxis])
		sil_dust_prod = np.sum((1.-sil_DZ)*sil_dust_mass/growth_timescale[:,np.newaxis],axis=1)
		

		growth_timescale = timescales['Carbon']
		C_DZ = G.dz[:,2]/((1-C_in_CO)*G.z[:,2])
		C_dust_mass = G.dz[:,2]*M
		carbon_dust_prod = (1.-C_DZ)*C_dust_mass/growth_timescale
		carbon_dust_prod[np.logical_or(C_DZ <= 0,C_DZ >= 1)] = 0.

		iron_dust_prod = np.zeros(len(sil_dust_prod))

		O_dust_prod = np.zeros(len(sil_dust_prod))

	else:
		timescales = calc_spec_acc_timescale(G, nano_iron=nano_iron)
		####################
		## SILICATES 
		####################
		growth_timescale = timescales['Silicates']
		key_elem, key_num_dens, key_in_dust = calc_spec_key_elem(G, nano_iron=nano_iron)

		sil_dust_formula_mass = 0.0
		for k in range(4): sil_dust_formula_mass += config.SIL_NUM_ATOMS[k] * config.ATOMIC_MASS[config.SIL_ELEM_INDEX[k]];
		key_DZ = G.dz[farg,key_elem]/G.z[farg,key_elem]
		# Deal with nan data
		key_DZ[np.isnan(key_DZ)] = 0.

		key_M_dust = G.spec[farg,0]*M
		sil_dust_prod = (1.-key_DZ)*key_M_dust/growth_timescale
		sil_dust_prod[np.logical_or(key_DZ <= 0,key_DZ >= 1)] = 0.
		sil_dust_prod /= key_in_dust*config.ATOMIC_MASS[key_elem]/sil_dust_formula_mass

		####################
		## CARBONACEOUS
		####################
		growth_timescale = timescales['Carbon']
		key_elem = 2
		key_DZ = G.dz[:,key_elem]/((1-C_in_CO)*G.z[:,key_elem])
		# Deal with nan data
		key_DZ[np.isnan(key_DZ)] = 0.

		key_M_dust = G.spec[:,1]*M
		carbon_dust_prod = (1.-key_DZ)*key_M_dust/growth_timescale
		carbon_dust_prod[np.logical_or(key_DZ <= 0,key_DZ >= 1)] = 0.


		####################
		## IRON 
		####################
		growth_timescale = timescales['Iron']
		key_elem = 10
		key_DZ = G.dz[:,key_elem]/G.z[:,key_elem]
		# Deal with nan data
		key_DZ[np.isnan(key_DZ)] = 0.

		# Need to specifically use only the free-flying iron since there rest is locked in silicates and cannot accrete
		if nano_iron:
			key_M_dust = G.spec[:,4]*M
		else:
			key_M_dust = G.dz[:,key_elem]*M
		iron_dust_prod = (1.-key_DZ)*key_M_dust/growth_timescale
		iron_dust_prod[np.logical_or(key_DZ <= 0,key_DZ >= 1)] = 0.


		####################
		## OXYGEN RESERVOIR 
		####################
		# Check if sim was run with optional O reservoir
		if O_res:
			nH = G.rho*config.UnitDensity_in_cgs * ( 1. - (G.z[:,0]+G.z[:,1])) / config.H_MASS
			# expected fractional O depletion
			D_O = 1. - 0.65441 / np.power(nH,0.103725)
			D_O[D_O<0] = 0
			key_elem, key_num_dens, key_in_dust = calc_spec_key_elem(G, nano_iron=nano_iron)
			key_mass = config.ATOMIC_MASS[key_elem]

			# fraction of maximum possible silicate dust present
			frac_of_sil = (G.spec[:,0]) / ((G.z[farg,key_elem]) * sil_dust_formula_mass/(key_in_dust * key_mass))
			max_O_in_sil = G.z[farg,key_elem] * ((config.SIL_NUM_ATOMS[0] * config.ATOMIC_MASS[4])/(key_in_dust * key_mass))
			extra_O = frac_of_sil * D_O * G.z[:,4] - max_O_in_sil - G.spec[:,4] - O_in_CO*G.z[:,4]

			print(frac_of_sil)
			print(D_O)
			print(np.max(G.spec[:,4]))

			mask = extra_O > 0.
			print('n_H = ', nH[mask])
			print('Z_O = ',G.z[:,4][mask])
			print('f_sil = ',frac_of_sil[mask])
			print('f_(max O in sil) =' ,max_O_in_sil[mask]/G.z[:,4][mask])
			print('D_O = ',D_O[mask])
			print('f_(O res) = ', G.spec[:,4][mask]/G.z[:,4][mask])
			print('O_in_CO = ',O_in_CO[mask])
			print(frac_of_sil[mask] * D_O[mask])
			print(extra_O[mask])

			print(1.*len(extra_O[extra_O > 0])/len(extra_O))

			extra_O[extra_O < 0] = 0
			extra_O[sil_dust_prod <= 0] = 0
			# If needed O depletion can't be attributed to silicate dust and what's already in the oxygen reservoir throw more oxygen into the reservoir
			O_dust_prod = extra_O*G.z[:,4]

		else:
			O_dust_prod = np.zeros(len(sil_dust_prod))


	dust_prod = {'Silicates':sil_dust_prod,'Carbon':carbon_dust_prod,'Iron':iron_dust_prod,'O Reservoir':O_dust_prod}


	#dust_prod = {'Silicates':sil_dust_prod,'Carbon':carbon_dust_prod,'Iron':iron_dust_prod}
	return dust_prod