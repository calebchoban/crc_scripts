import numpy as np
import config
import gizmo_library.utils as utils

# Theoretical dust yields for all sources of creation


# Solar Abundace values used in FIRE-2 and FIRE-3
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



def calculate_relative_light_to_mass_ratio_from_imf(i):
	return 1.

def stellarRates(star_age, Z, time_step):
	D_RETURN_FRAC = 1E-7
	p = 0.0
	mass_return = 0.0
	GasReturnFraction = 1.
	if(Z>3): Z=3;
	if(Z<0.01): Z=0.01;
	if(star_age<=0.001):
		p=11.6846 
	else:
		if(star_age<=0.0035):
			p=11.6846*Z*np.power(10.,1.838*(0.79+np.log10(Z))*(np.log10(star_age)-(-3.00)));
		else:
			p= 72.1215* np.power(star_age / 0.0035,-1.3)

	if(star_age < 0.1): p *= calculate_relative_light_to_mass_ratio_from_imf(0) # late-time independent of massive stars
	p *= GasReturnFraction * (time_step) # fraction of particle mass expected to return in the timestep 
	p = 1.0 - np.exp(-p); # need to account for p>1 cases 
	p *= 1.4 * 0.291175; # to give expected return fraction from stellar winds alone (~17%)

	n_wind_0=np.floor(p/D_RETURN_FRAC); 
	p-=n_wind_0*D_RETURN_FRAC; # if p >> return frac, should have > 1 event, so we inject the correct wind mass
	mass_return += n_wind_0*D_RETURN_FRAC; # add this in, then determine if there is a 'remainder' to be added as well
	if np.random.random() < p/D_RETURN_FRAC: 
		mass_return += D_RETURN_FRAC; # add the 'remainder' stochastically

	return mass_return





# Rate of SNe Ia and II
def SNeRates(star_age, Z, time_step, FIRE_v=2):
	# basic variables we will use 
	agemin=0.003401; agebrk=0.01037; agemax=0.03753; # in Gyr 
	RSNe = 0
	p = 0.
	if FIRE_v==2:
		if star_age > agemin:
			if star_age<=agebrk:
				RSNe = 5.408e-4; # NSNe/Myr *if* each SNe had exactly 10^51 ergs; really from the energy curve 
			if star_age<=agemax:
				RSNe = 2.516e-4; # this is for a 1 Msun population 
			# Ia (prompt Gaussian+delay, Manucci+06)
			if star_age>agemax:
				RSNe = 5.3e-8 + 1.6e-5*np.exp(-0.5*((star_age-0.05)/0.01)*((star_age-0.05)/0.01));
	else:
		agemin=0.0037; agebrk=0.7e-2; agemax=0.044; f1=3.9e-4; f2=5.1e-4; f3=1.8e-4;
		# core-collapse; updated with same stellar evolution models for wind mass loss [see there for references]. simple 2-part power-law provides extremely-accurate fit. models predict a totally negligible metallicity-dependence.
		if star_age<agemin: 
			RSNe=0;
		elif star_age<=agebrk:
			RSNe=f1*np.power(star_age/agemin,np.log(f2/f1)/np.log(agebrk/agemin));
		elif star_age<=agemax:
			RSNe=f2*np.power(star_age/agebrk,np.log(f3/f2)/np.log(agemax/agebrk)); 
		else:
			RSNe=0; # core-collapse; updated with same stellar evolution models for wind mass loss [see there for references]. simple 2-part power-law provides extremely-accurate fit. models predict a totally negligible metallicity-dependence.
		
		# t_Ia_min = delay time to first Ia, in Gyr; norm_Ia = Hubble-time integrated number of Ia's per solar mass
		t_Ia_min=agemax; norm_Ia=1.6e-3; 
		if star_age>t_Ia_min:
			RSNe += norm_Ia * 7.94e-5 * np.power(star_age,-1.1) / np.abs(np.power(t_Ia_min/0.1,-0.1) - 0.61); # Ia DTD following Maoz & Graur 2017, ApJ, 848, 25
	

	p = time_step * 1000 *  RSNe # Total mass returned for this time step per unit star mass
	return p




# Metal and dust yields for stellar winds
def stellarYields(star_age, Z, time_step, routine = 'species', age_cutoff = 0.03753):
	yields = np.zeros(11)
	dust_yields = np.zeros(11)
	species_yields = np.zeros(4)
	min_age = age_cutoff
	atomic_mass = [1.01, 2.0, 12.01, 14, 15.99, 20.2, 24.305, 28.086, 32.065, 40.078, 55.845]
	if routine == 'species':
		sil_num_atoms = [3.631,1.06,1.,0.571] # O, Mg, Si, Fe
		sil_elems_index = [4,6,7,10] # O,Mg,Si,Fe 
		dust_formula_mass = 0
	else:
		condens_eff = 0.8;
	
	for k in range(11):
		yields[k] = solarMetallicity(k)*Z

	# All, then He,C,N,O,Ne,Mg,Si,S,Ca,Fe ;; follow AGB/O star yields in more detail for the light elements 
	#   the interesting species are He & CNO: below is based on a compilation of van den Hoek & Groenewegen 1997, Marigo 2001, Izzard 2004 
	yields[1]=0.36; # He 
	yields[2]=0.016; # C
	yields[3]=0.0041; # N
	yields[4]=0.0118; # O
	# metal-dependent yields: O scaling is strongly dependent on initial metallicity of the star //
	if solarMetallicity(0)*Z<0.033:
		yields[4] *= Z
	else: 
		yields[4] *= 1.65
	for k in range(1,5):
		yields[k]=yields[k]*(1.- Z*solarMetallicity(0)) + (solarMetallicity(k)*Z-solarMetallicity(k)) 
		if yields[k]<0:
			yields[k]=0.0 
		if yields[k]>1:
			yields[k]=1.
	yields[0]=0.0
	for k in range(2,11):
		yields[0]+=yields[k]


	if routine == 'species':
		# Now check whether the yields are from AGB or O/B since dust only forms for AGB
		if star_age >= min_age:
			# convert star age to mass of stars
			mass = 2.51  * pow(star_age, -0.4);
			dM = mass - 2.51 * pow(star_age+time_step, -0.4);
			if mass >= 1.0:
				IMF = 0.2724 * pow(mass, -2.7);
			else:
				IMF = 0.2724 * pow(mass, -2.3);


			species_yields = dM * IMF * AGBDustYields(mass, Z)

			# Convert species to elemental yields
			# Silicates
			dust_formula_mass = 0
			for k in range(4):
				dust_formula_mass += sil_num_atoms[k] * atomic_mass[sil_elems_index[k]]
			for k in range(4):
				dust_yields[sil_elems_index[k]] += species_yields[0] * sil_num_atoms[k] * atomic_mass[sil_elems_index[k]] / dust_formula_mass

			# Carbon
			dust_yields[2] += species_yields[1]

			# Silicon Carbide
			dust_formula_mass = atomic_mass[2] + atomic_mass[7]
			dust_yields[2] += species_yields[2] * atomic_mass[2] / dust_formula_mass
			dust_yields[7] += species_yields[2] * atomic_mass[7] / dust_formula_mass

			# Iron
			dust_yields[10] += species_yields[3]

			for k in range(1,11):
				dust_yields[0] += dust_yields[k]

	else:
		if star_age >= min_age:
			# AGB stars with C/O number density > 1 
			if (yields[4] <= 0. or (yields[2]/atomic_mass[2])/(yields[4]/atomic_mass[4]) > 1.0):
				dust_yields[2] = yields[2] - 0.75*yields[4]; # C 
				dust_yields[0] = dust_yields[2]; 
				species_yields[1] = dust_yields[2];
			# AGB stars with C/O < 1 
			else:
				dust_yields[6] = condens_eff * yields[6]; # Mg
				dust_yields[7] = condens_eff * yields[7]; # Si
				dust_yields[10] = condens_eff * yields[10]; # Fe
				dust_yields[4] = 16 * (dust_yields[6]/atomic_mass[6] + dust_yields[7]/atomic_mass[7] + dust_yields[10]/atomic_mass[10]); # O
				for k in range(2,11): dust_yields[0]+=dust_yields[k];
				species_yields[0] = dust_yields[0];


	return yields, dust_yields, species_yields


def SNeYields(star_age, Z, routine="species"):
	agemax=0.03753

	yields = np.zeros(11)
	dust_yields = np.zeros(11)
	species_yields = np.zeros(4)
	atomic_mass = [1.01, 2.0, 12.01, 14, 15.99, 20.2, 24.305, 28.086, 32.065, 40.078, 55.845]
	if routine == 'species':
		sil_num_atoms = [3.631,1.06,1.,0.571]  # O, Mg, Si, Fe
		sil_elems_index = [4,6,7,10] # O,Mg,Si,Fe 
		SNeII_sil_cond = 0.00035; SNeII_C_cond = 0.15; SNeII_SiC_cond = 0.0003; SNeII_Fe_cond = 0.001; SNeI_Fe_cond = 0.005;
		dust_formula_mass = 0
		sil_elem_abund = np.zeros(4)
		missing_element = 0
		key_elem = 0
	else:
		C_condens_eff = 0.5;
		other_condens_eff = 0.8;

def SNeYields(star_age, Z, FIRE_v=2):
	yields = np.zeros(11)
	if star_age == 0.:
		return yields

	if FIRE_v>2:
		# match to rates tabulation above to determine if Ia or CC 
		# default to a mean mass for Ia vs CC SNe; for updated table of SNe rates and energetics, this is the updated mean mass per explosion to give the correct total SNe mass
		if star_age > 0.044: 
			SNeIaFlag=1
			Msne=1.4 
		else:
			SNeIaFlag=0 
			Msne=8.72 
		for k in range(10):
			yields[k]= Z*solarMetallicity(k) # initialize to surface abundances
		if SNeIaFlag: 
			# SNIa :: from Iwamoto et al. 1999; 'W7' models: total ejecta mass = Msne = 1.4. yields below are -fractional-
			yields[0]=1; # total metal mass (species below, + residuals primarily in Ar, Cr, Mn, Ni) */ 
			yields[1]=0; # He
			# adopted yield: mean of W7 and WDD2 in Mori+2018. other models included below for reference in comments 
			yields[2]=1.76e-2; yields[3]=2.10e-06; yields[4]=7.36e-2; yields[5]=2.02e-3; yields[6]=6.21e-3; yields[7]=1.46e-1; yields[8]=7.62e-2; yields[9]=1.29e-2; yields[10]=5.58e-1; # arguably better obs calibration vs LN/NL papers
			# Iwamoto et al. 1999, 'W7' model  //yields[2]=3.50e-2; yields[3]=8.57e-07; yields[4]=1.02e-1; yields[5]=3.21e-3; yields[6]=6.14e-3; yields[7]=1.11e-1; yields[8]=6.21e-2; yields[9]=8.57e-3; yields[10]=5.31e-1; // old, modestly disfavored albeit for species like Mn not here
			# updated W7 in Nomoto+Leung 18 review  //yields[2]=3.71e-2; yields[3]=7.79e-10; yields[4]=1.32e-1; yields[5]=3.11e-3; yields[6]=3.07e-3; yields[7]=1.19e-1; yields[8]=5.76e-2; yields[9]=8.21e-3; yields[10]=5.73e-1; // not significantly different from updated W7 below, bit more of an outlier and review tables seem a bit unreliable (typos, etc)
			# mean of new yields for W7 + WDD2 in Leung+Nomoto+18  //yields[2]=1.54e-2; yields[3]=1.24e-08; yields[4]=8.93e-2; yields[5]=2.41e-3; yields[6]=3.86e-3; yields[7]=1.34e-1; yields[8]=7.39e-2; yields[9]=1.19e-2; yields[10]=5.54e-1; // not significantly different from updated W7 below, bit more of an outlier and review tables seem a bit unreliable (typos, etc)
			# W7   [Mori+18] [3.42428571e-02, 4.16428571e-06, 9.68571429e-02, 2.67928571e-03, 7.32857143e-03, 1.25296429e-01, 5.65937143e-02, 8.09285714e-03, 5.68700000e-01] -- absolute yield in solar // WDD2 [Mori+18] [9.70714286e-04, 2.36285714e-08, 5.04357143e-02, 1.35621429e-03, 5.10112857e-03, 1.65785714e-01, 9.57078571e-02, 1.76928571e-02, 5.47890000e-01] -- absolute yield in solar
			# updated W7 in Leung+Nomoto+18  //yields[2]=1.31e-2; yields[3]=7.59e-10; yields[4]=9.29e-2; yields[5]=1.79e-3; yields[6]=2.82e-3; yields[7]=1.06e-1; yields[8]=5.30e-2; yields[9]=6.27e-3; yields[10]=5.77e-1; // seems bit low in Ca/Fe, less plausible if those dominated by Ia's
			# Seitenzahl et al. 2013, model N100 [favored]  //yields[2]=2.17e-3; yields[3]=2.29e-06; yields[4]=7.21e-2; yields[5]=2.55e-3; yields[6]=1.10e-2; yields[7]=2.05e-1; yields[8]=8.22e-2; yields[9]=1.05e-2; yields[10]=5.29e-1; // very high Si, seems bit less plausible vs other models here
			# new benchmark model in Leung+Nomoto+18 [closer to WDD2 in lighter elements, to W7 in heavier elements] */ //yields[2]=1.21e-3; yields[3]=1.40e-10; yields[4]=4.06e-2; yields[5]=1.29e-4; yields[6]=7.86e-4; yields[7]=1.68e-1; yields[8]=8.79e-2; yields[9]=1.28e-2; yields[10]=6.14e-1; # arguably better theory motivation vs Mori+ combination
		
		 # Core collapse :: temporary new time-dependent fits
		else:
			t=star_age; tmin=0.0037; tbrk=0.0065; tmax=0.044; Mmax=35.; Mbrk=10.; Mmin=6.; # numbers for interpolation of ejecta masses [must be careful here that this integrates to the correct -total- ejecta mass]
			# note these break times: tmin=3.7 Myr corresponds to the first explosions (Eddington-limited lifetime of the most massive stars),
			# tbrk=6.5 Myr to the end of this early phase, stars with ZAMS mass ~30+ Msun here. curve flattens both from IMF but also b/c mass-loss less efficient. tmax=44 Myr to the last explosion determined by lifetime of 8 Msun stars
			if t<=tbrk :
				Msne=Mmax*np.power(t/tmin, np.log(Mbrk/Mmax)/np.log(tbrk/tmin))

			else:
				   Msne=Mbrk*np.power(t/tbrk, np.log(Mmin/Mbrk)/np.log(tmax/tbrk)) # power-law interpolation of ejecta mass from initial to final value over duration of CC phase
			i_tvec = 5; # number of entries 
			tvec = [3.7, 8., 18., 30., 44.]; # time in Myr
			fvec = np.array([
				[4.61e-01, 3.30e-01, 3.58e-01, 3.65e-01, 3.59e-01], # He [IMF-mean y=3.67e-01]  [note have to remove normal solar correction and take care with winds]
				[2.37e-01, 8.57e-03, 1.69e-02, 9.33e-03, 4.47e-03], # C  [IMF-mean y=3.08e-02]  [note care needed in fitting out winds: wind=6.5e-3, ejecta_only=1.0e-3]
				[1.07e-02, 3.48e-03, 3.44e-03, 3.72e-03, 3.50e-03], # N  [IMF-mean y=4.47e-03]  [some care needed with winds, but not as essential]
				[9.53e-02, 1.02e-01, 9.85e-02, 1.73e-02, 8.20e-03], # O  [IMF-mean y=7.26e-02]  [reasonable - generally IMF-integrated alpha-element total mass-yields lower vs fire-2 by factor ~0.7 or so]
				[2.60e-02, 2.20e-02, 1.93e-02, 2.70e-03, 2.75e-03], # Ne [IMF-mean y=1.58e-02]  [roughly a hybrid of fit direct to ejecta and fit to all mass as above, truncating at highest masses]
				[2.89e-02, 1.25e-02, 5.77e-03, 1.03e-03, 1.03e-03], # Mg [IMF-mean y=9.48e-03]  [fit directly on ejecta and ignore mass-fraction rescaling since that's not reliable at early times: this gives a reasonable number. important to note that early SNe dominate Mg here, quite strongly]
				[4.12e-04, 7.69e-03, 8.73e-03, 2.23e-03, 1.18e-03], # Si [IMF-mean y=4.53e-03]  [lots comes from 1a's, so low here isn't an issue]
				[3.63e-04, 5.61e-03, 5.49e-03, 1.26e-03, 5.75e-04], # S  [IMF-mean y=3.01e-03]  [more from Ia's]
				[4.28e-05, 3.21e-04, 6.00e-04, 1.84e-04, 9.64e-05], # Ca [IMF-mean y=2.77e-04]  [Ia]
				[5.46e-04, 2.18e-03, 1.08e-02, 4.57e-03, 1.83e-03]  # Fe [IMF-mean y=4.11e-03]  [Ia]
			]) # compare nomoto '06: y = [He: 3.69e-1, C: 1.27e-2, N: 4.56e-3, O: 1.11e-1, Ne: 3.81e-2, Mg: 9.40e-3, Si: 8.89e-3, S: 3.78e-3, Ca: 4.36e-4, Fe: 7.06e-3]
			# ok now use the fit parameters above for the piecewise power-law components to define the yields at each time 
			t_myr=star_age*1000.; i_t=-1; 
			for k in range(i_tvec):
				if t_myr>tvec[k]:
					i_t=k
			for k in range(10): 
				i_y = k + 1; 
				if i_t<0: 
					yields[i_y]=fvec[k,0]
				elif i_t>=i_tvec-1:
					yields[i_y]=fvec[k,i_tvec-1] 
				else: 
					yields[i_y] = fvec[k][i_t] * np.power(t_myr/tvec[i_t] , np.log(fvec[k,i_t+1]/fvec[k,i_t]) / np.log(tvec[i_t+1]/tvec[i_t]))
			# sum heavy element yields to get the 'total Z' yield here, multiplying by a small correction term to account for trace species not explicitly followed above [mean for CC] */
			yields[0]=0 
			for k in range(2,10):
				 yields[0] += 1.0144 * yields[k] # assume here that there is some trace species proportional to each species, not really correct but since it's such a tiny correction this is pretty negligible //

	if FIRE_v == 2:
		if star_age > 0.03753: 
			SNeIaFlag=1
			Msne=1.4
		else:
			SNeIaFlag=0 
			Msne=10.5 
		 # SNIa  from Iwamoto et al. 1999; 'W7' models 
		if SNeIaFlag: 
			yields[0]=1; # total metal mass
			yields[1]=0.0; # He
			yields[2]=0.035; # C
			yields[3]=8.57e-7; # N
			yields[4]=0.102; #O
			yields[5]=0.00321; # Ne
			yields[6]=0.00614; # Mg
			yields[7]=0.111; # Si
			yields[8]=0.0621; # S
			yields[9]=0.00857; # Ca
			yields[10]=0.531; # Fe
		# SNII (IMF-averaged... may not be the best approx on short timescales..., Nomoto 2006 (arXiv:0605725). here divided by ejecta mass total
		else: 
			yields[0]=0.19; # Z [total metal mass]
			yields[1]=0.369; # He 
			yields[2]=0.0127; # C
			yields[3]=0.00456; # N
			yields[4]=0.111; # O
			yields[5]=0.0381; # Ne
			yields[6]=0.00940; # Mg
			yields[7]=0.00889; # Si
			yields[8]=0.00378; # S
			yields[9]=0.000436; # Ca
			yields[10]=0.00706; # Fe
			if(Z*solarMetallicity(0)<0.033):
				yields[3]*=Z
			else:
				yields[3]*=1.65 # metal-dependent yields: N scaling is strongly dependent on initial metallicity of the star 
			  
			yields[0] += yields[3]-0.00456; # augment total metal mass for this correction 

 
	if SNeIaFlag:
		yields[1]=0.0 # no He yield for Ia SNe 

	if routine == "species":
		if star_age < agemax:
				# silicates
				# first check that there are non-zero amounts of all elements required to make dust species
				missing_element = 0;
				for k in range(4):
					if (yields[sil_elems_index[k]] <= 0): missing_element = 1;
				if not missing_element:
					# used to find the key element for silicate dust
					for k in range(4): sil_elem_abund[k] = yields[sil_elems_index[k]] / (atomic_mass[sil_elems_index[k]] * sil_num_atoms[k]);
					dust_formula_mass=0;
					for k in range(4): dust_formula_mass += sil_num_atoms[k] * atomic_mass[sil_elems_index[k]];
					key_elem=0;
					for k in range(4):
						if (sil_elem_abund[key_elem]>sil_elem_abund[k]):
							key_elem = k;
					if (sil_elem_abund[key_elem] > 0):
						species_yields[0] = SNeII_sil_cond * yields[sil_elems_index[key_elem]] * dust_formula_mass / (sil_num_atoms[key_elem] * atomic_mass[sil_elems_index[key_elem]]);
						for k in range(4):
							dust_yields[sil_elems_index[k]] += species_yields[0] * sil_num_atoms[k] * atomic_mass[sil_elems_index[k]] / dust_formula_mass;
				# carbon
				species_yields[1] = SNeII_C_cond * yields[2];
				dust_yields[2] += species_yields[1];
				# silicon carbide
				if (yields[2]>0 and yields[7]>0):
					dust_formula_mass = atomic_mass[2] + atomic_mass[7];
					if (yields[7]/atomic_mass[7] < yields[2]/atomic_mass[2]): key_elem = 7;
					else: key_elem = 2;
					species_yields[2] = SNeII_SiC_cond * yields[key_elem] * (dust_formula_mass / atomic_mass[key_elem]);
					dust_yields[2] += species_yields[2] * atomic_mass[2] / dust_formula_mass;
					dust_yields[7] += species_yields[2] * atomic_mass[7] / dust_formula_mass;
				# iron
				species_yields[3] = SNeII_Fe_cond * yields[10];
				dust_yields[10] += species_yields[3];
		else:
				# only a little bit of iron dust from SNIa
				species_yields[3] = SNeI_Fe_cond * yields[10];
				dust_yields[10] = species_yields[3]; 

		for k in range(11):
			dust_yields[0] += dust_yields[k];

	else:
		dust_yields[2] = C_condens_eff * yields[2]; # C
		dust_yields[6] = other_condens_eff * yields[6]; # Mg
		dust_yields[7] = other_condens_eff * yields[7]; # Si
		dust_yields[10] = other_condens_eff * yields[10]; # Fe
		dust_yields[4] = 16 * (dust_yields[6]/atomic_mass[6] + dust_yields[7]/atomic_mass[7] + dust_yields[10]/atomic_mass[10]); # O
		for k in range(2,11):
			dust_yields[0] += dust_yields[k]; # Fraction of yields that is dust
		species_yields[0] = dust_yields[4] + dust_yields[6] + dust_yields[7] + dust_yields[10];
		species_yields[1] = dust_yields[2];
	
	return yields, dust_yields, species_yields

# Fitted AGB dust yields from Zhukovska et al. (2008)
def AGBDustYields(m, z):
	z *= solarMetallicity(0)
	returns = np.zeros(4)
	max_mass = 8
# Silicates
###############################################################################
	if z >= 0.02: 
		if m <= 2:
			returns[0] = 0.00190032 + m*(-0.00331876 + 0.20186*z) - 0.111861*z
		elif m <= 3.:
			returns[0] = -0.0208116 + m*(0.0069372 - 0.34686*z) + 1.04058*z
		elif m>= 4 and m <=4.5:
			returns[0] = 0.053802 + m*(-0.0134505 + 0.953023*z) - 3.81209*z
		elif m >= 4.5:
			returns[0] = -0.00255896 + m*(-0.000925843 + 0.0720291*z) + 0.152381*z
	elif z >= 0.008:
		if m <= 2:
			returns[0] = 0.000224593 + m*(-0.000478963 + 0.0598704*z) - 0.0280742*z		   
		elif m>= 4 and m <=4.5:
			returns[0] = 0.0110913 + m*(-0.00277281 + 0.419138*z) - 1.67655*z			
		elif m >= 4.5:
			returns[0] = -0.00052737 + m*(-0.000190897 + 0.0352818*z) + 0.050801*z
	elif z >= 0.004:
		if m>= 4 and m <=4.5:
			returns[0] = 0.145073*(-4. + m)*(-0.004 + z)	 
		elif m >= 4.5:
			returns[0] = (-0.0302405 + 0.0228393*m)*(-0.004 + z)	
# Carbon
###############################################################################
	if z >= 0.02:
		if m > 4.5 and m <= max_mass:
			returns[1] = .00001405*(2.55544 + z)
		elif m > 4.0 and m <= 4.5:
			returns[1] = -0.824501*(-0.0163695 + 0.00365816*m - 4.50722*z + m*z)
		elif m >= 1. and m <= 4:
			returns[1] = -1.63903e-14*np.exp((17.0613 - 2.33094*m)*m)*(np.exp( \
						  1.11698*(-3.44843 + m)**2)*(0.0281428 - 1.40714*z) +   \
						  np.exp(1.21396*(-3.85419 + m)**2)*(-0.04 + z))
	elif z >= 0.008:
		if m > 4.5 and m <= max_mass:
			returns[1] = -0.0163371*(-0.0222149 + z)
		elif m > 4.0 and m <= 4.5:
			returns[1] = -0.315198*(-0.189505 + 0.0418854*m - 4.4558*z + m*z)
		elif m >= 1. and m <= 4:
			returns[1] = -6.83526e-10*np.exp((12.662 - 1.90399*m)*m)*(np.exp( \
						  0.787013*(-3.15012 + m)**2)*(0.00880479 - 1.1006*z) + \
						  np.exp(1.11698*(-3.44843 + m)**2)*(-0.02 + z))
	elif z >= 0.004:
		if m > 4.5 and m <= max_mass:
			returns[1] = -0.0451125*(-0.0131478 + z)
		elif m > 4.0 and m <= 4.5:
			returns[1] = 1.17905*(0.0965292 - 0.021336*m - 4.54239*z + m*z)
		elif m >= 1. and m <= 4:
			returns[1] = 1.11697e-6*np.exp((8.54724 - 1.33322*m)*m)*(np.exp( \
						 0.787013*(-3.15012 + m)**2)*(0.0250515 - 3.13144*z) + \
						 np.exp(0.546211*(-3.28524 + m)**2)*(-0.0118657 + 2.96642*z))
	elif z > 0.001:
		if m > 4.5 and m <= max_mass:
			returns[1] = -0.0618633*(-0.0106708 + z)
		elif m > 4.0 and m <= 4.5:
			returns[1] = 1.35407*(0.0863978 - 0.0190953*m - 4.54158*z + m*z)
		elif m >= 1. and m <= 4:
			returns[1] = 0.0000443843*np.exp((5.43239 - 0.752061*m)*m)*(np.exp( \
						 0.546211*(-3.28524 + m)**2)*(0.017395 - 4.34876*z) +	\
						 np.exp(0.205849*(-4.47784 + m)**2)*(-0.00417525 + 4.17525*z))
	else:	
		if m > 4.5 and m <= max_mass:
			returns[1] = 0.00059827;
		elif m > 4.0 and m <= 4.5:
			returns[1] = 0.110839 - 0.0245022*m
		elif m >= 1. and m <= 4:
			returns[1] = 0.0130463*np.exp(-0.205849*(-4.47784 + m)**2)
# SiC
###############################################################################
	if z >= 0.02:
		if m >= 2 and m <= 4:
			returns[2] = 0.00303088 + m*(-0.00118122 + 0.0811555*z) - 0.179022*z
		elif m >= 4 and m <= 4.5:
			returns[2] = -0.015246 + m*(0.003388 - 0.2912*z) + 1.3104*z
	elif z >= 0.008:
		if m >= 2 and m <= 4:
			returns[2] = 0.000319773 + m*(-0.000260343 + 0.0351117*z) - 0.0434667*z
		elif m >= 4 and m <= 4.5:
			returns[2] = -0.0064944 + m*(0.0014432 - 0.19396*z) + 0.87282*z
	elif z >= 0.004:
		if m >= 2 and m <= 4:
			returns[2] = (-0.00699 + 0.0051375*m)*(-0.004 + z)
		elif m >= 4 and m <= 4.5:
			returns[2] = -0.02712*(-4.5 + m)*(-0.004 + z)  
# Iron
###############################################################################
	if z >= 0.02:
		if m >= 4 and m <= max_mass:
			returns[3] = 0.000662464*(-0.271934 + 0.0157263*m + 14.6246*z + m*z)
		elif m >=3 and m <= 4:
			returns[3] = -0.0976492*(-4.1818 + m)*(-0.02 + z)
		elif m >= 2 and m <=3:
			returns[3] = 0.100999*(0.0461624 - 0.0229906*m - 1.85846*z + m*z)
		elif m >= 1 and m <=2:
			returns[3] = 0.0032885*(-0.108566 + 0.0570655*m + 2.34697*z + m*z)
	elif z >= 0.008:
		if m >= 4 and m <= max_mass:
			returns[3] = 0.000324989*(-0.259803 + 0.0528252*m + 15.0854*z + m*z)
		elif m >= 2 and m <=3:
			returns[3] = -0.0251706*(-3.00713 + m)*(-0.008 + z)
		elif m >= 1 and m <=2:
			returns[3] = 0.0211192*(-0.799669 + m)*(-0.008 + z)
	elif z >= 0.004:
		if m >= 4 and m <= max_mass:
			returns[3] = (-0.0113032 + 0.00494188*m)*(-0.004 + z)

	return returns


# Create plot of stellar feedback for elements and dust for a stellar population over a given time in Gyr
def totalStellarYields(max_time, N, Z, routine = 'species'):
	time_step = max_time/N
	time = np.arange(0,max_time,time_step)
	cum_yields = np.zeros((len(time),11))
	cum_dust_yields = np.zeros((len(time),11))
	cum_species_yields = np.zeros((len(time),4))
	for i,age in enumerate(time):

		p = stellarRates(age, Z, time_step)
		stellar_yields, stellar_dust_yields, stellar_species_yields = stellarYields(age,Z,time_step,routine=routine)
		if routine == 'species':
			stellar_yields *= p
		else:
			stellar_yields *= p
			stellar_dust_yields *= p
			stellar_species_yields *= p

		p = SNeRates(age, Z, time_step)
		SNe_yields,SNe_dust_yields,SNe_species_yields = SNeYields(age,Z,routine=routine)
		SNe_yields *= p
		SNe_dust_yields *= p
		SNe_species_yields *= p

		cum_yields[i] = cum_yields[i-1] + stellar_yields + SNe_yields
		cum_dust_yields[i] = cum_dust_yields[i-1] + stellar_dust_yields + SNe_dust_yields
		cum_species_yields[i] = cum_species_yields[i-1] + stellar_species_yields + SNe_species_yields

	return cum_yields, cum_dust_yields, cum_species_yields



def totalFeedbackRates(max_time, N, Z):
	time_step = max_time/N
	time = np.arange(0,max_time,time_step)
	windRate = np.zeros(len(time))
	SNeRate = np.zeros(len(time))
	for i,age in enumerate(time):
		windRate[i] = stellarRates(age, Z, time_step)
		SNeRate[i] = SNeRates(age, Z, time_step)

	return windRate, SNeRate

def onlyAGBYields(max_time, N, Z, routine = 'species'):
	time_step = max_time/N
	time = np.arange(0,max_time,time_step)
	cum_yields = np.zeros((len(time),11))
	cum_dust_yields = np.zeros((len(time),11))
	cum_species_yields = np.zeros((len(time),4))
	for i,age in enumerate(time):
		p = stellarRates(age, Z, time_step)
		stellar_yields, stellar_dust_yields, stellar_species_yields = stellarYields(age,Z,time_step,routine=routine)
		if routine == 'species':
			stellar_yields *= p
		else:
			stellar_yields *= p
			stellar_dust_yields *= p

		cum_yields[i] = cum_yields[i-1] + stellar_yields
		cum_dust_yields[i] = cum_dust_yields[i-1] + stellar_dust_yields
		cum_species_yields[i] = cum_species_yields[i-1] + stellar_species_yields

	return cum_yields, cum_dust_yields, cum_species_yields

def onlySNeYields(max_time, N, Z, routine = 'species'):
	time_step = max_time/N
	time = np.arange(0,max_time,time_step)
	cum_yields = np.zeros((len(time),11))
	cum_dust_yields = np.zeros((len(time),11))
	cum_species_yields = np.zeros((len(time),4))
	for i,age in enumerate(time):
		p = SNeRates(age, Z, time_step)
		SNe_yields,SNe_dust_yields,SNe_species_yields = SNeYields(age,Z,routine=routine)
		SNe_yields *= p
		SNe_dust_yields *= p
		SNe_species_yields *= p

		cum_yields[i] = cum_yields[i-1] + SNe_yields
		cum_dust_yields[i] = cum_dust_yields[i-1] + SNe_dust_yields
		cum_species_yields[i] = cum_species_yields[i-1] + SNe_species_yields

	return cum_yields, cum_dust_yields, cum_species_yields


# The "Elemental" implementation dust accrection growth timescale
def elementalGrowthTime(temp, dens):
	ref_dens = H_MASS # g cm^-3
	T_ref = 20. # K
	time_ref = 0.2 # Gyr
	return time_ref * (ref_dens / dens) * np.power((T_ref / temp),0.5)


# The "Species"	implementation dust accrection growth timescale assuming growth species is gas form of key elemen and solar metallicities
def speciesGrowthTime(temp, dens, metallicity, species, verbose = False):
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
	all_species = ["silicates","carbon","iron"]
	elem_names = np.array(['Z','He','C','N','O','Ne','Mg','Si','S','Ca','Fe'])
	atomic_mass = np.array([1.01, 2.0, 12.01, 14, 15.99, 20.2, 24.305, 28.086, 32.065, 40.078, 55.845])
	sil_num_atoms = np.array([3.631,1.06,1.,0.571]) # O, Mg, Si, Fe
	sil_elems_index = np.array([4,6,7,10]) # O,Mg,Si,Fe 
	key_in_dust = 0 # number of atoms of key element in one formula unit of the dust species
	dust_formula_mass = 0

	# Check if list has been given
	if hasattr(temp, "__len__") and hasattr(dens, "__len__"):
		sticking_eff = np.zeros(np.shape(temp))
		sticking_eff[temp <= 300] = 1.
		sticking_eff[temp > 300] = 0.
	elif hasattr(temp, "__len__") or hasattr(dens, "__len__"):
		print "Both temp and dens must be either arrays or scalars!"
		exit()
	else:
		if (temp < 300):
			sticking_eff = 1.
		else:
			sticking_eff = 0.
			return 0.

	if species == all_species[0]:
		for k in range(4): dust_formula_mass += sil_num_atoms[k] * atomic_mass[sil_elems_index[k]];
		if assumed_Z:
			# Since we assume solar metallicities we can just say Si is the key element
			key = 2
			key_elem = sil_elems_index[key]
			num_dens = dens * ZZ * solarMetallicity(key_elem)/ (atomic_mass[key_elem]*H_MASS)
			key_in_dust = sil_num_atoms[key]
			cond_dens = 3.13
		else:
			num_dens = np.multiply(metallicity, dens[:, np.newaxis]) / (atomic_mass*H_MASS)
			sil_num_dens = num_dens[:,sil_elems_index] 
			# Find element with lowest number density factoring in number of atoms needed to make one formula unit of the dust species
			key = np.argmin(sil_num_dens / sil_num_atoms, axis = 1)
			num_dens = sil_num_dens[range(sil_num_dens.shape[0]),key]
			key_elem = sil_elems_index[key]
			key_in_dust = sil_num_atoms[key]
			cond_dens = 3.13


	elif species == all_species[1]:
		if not assumed_Z:
			key_elem = np.full(np.shape(metallicity[:,0]),2)
		else:
			key_elem = 2

		key_in_dust = 1
		dust_formula_mass = atomic_mass[key_elem]
		cond_dens = 2.25
		if assumed_Z:
			num_dens = dens * ZZ * solarMetallicity(key_elem) / (atomic_mass[key_elem]*H_MASS)
		else:
			num_dens = np.multiply(metallicity[:,key_elem[0]], dens) / (atomic_mass[key_elem]*H_MASS)

	elif species == all_species[2]:
		if not assumed_Z:
			key_elem = np.full(np.shape(metallicity[:,0]),10)
		else:
			key_elem = 10

		key_in_dust = 1
		dust_formula_mass = atomic_mass[key_elem]
		cond_dens = 7.86
		if assumed_Z:
			num_dens = dens * ZZ * solarMetallicity(key_elem) / (atomic_mass[key_elem]*H_MASS)
		else:
			num_dens = np.multiply(metallicity[:,key_elem[0]], dens) / (atomic_mass[key_elem]*H_MASS)
	else:
		if verbose:
			print str(species), " is not a valid species"
		return 0

	if verbose:
		print "For "  + species + " the key element was found to be ",elem_names[key_elem]
		print "Its number density is ", str(num_dens), " g/cm^3"

	acc_time = time_ref * (key_in_dust * np.power(atomic_mass[key_elem],0.5) / (sticking_eff * dust_formula_mass)) * (1. / num_dens) * cond_dens * np.power((1. / temp),0.5)
	# Convert from element timescale to species timescale (only matters for silicates)
	acc_time /= dust_formula_mass / (key_in_dust * atomic_mass[key_elem])

	if not assumed_Z:
		# Keep track of no accretion due to no key elements being present
		no_acc = np.where(metallicity[range(metallicity.shape[0]),key_elem] <= 0.)
		acc_time[no_acc] = np.inf
		key_elem[no_acc] = -1

	return acc_time, key_elem


# Calculates the dust accretion for a gas of a given density, temperature, and inital dust fraction
def elemDustAccretionEvolution(temp, dens, initial_frac, metallicity, time_step, N):
	time = np.zeros(N)
	dust_frac = np.zeros(N)
	dust_frac[0] = initial_frac
	for i in range(N-1):
		growth_timescale,_ = elementalGrowthTime(temp,dens)
		dust_frac[i+1] = dust_frac[i] + time_step * (1. - dust_frac[i])*dust_frac[i]/growth_timescale
		time[i+1] += time_step + time[i]
		
	return dust_frac, time

# Calculates the dust accretion for a gas of a given density, temperature, and inital dust fraction
def specDustAccretionEvolution(temp, dens, initial_frac, metallicity, time_step, N):
	time = np.zeros(N)
	dust_frac = np.zeros([4,N])
	dust_frac[:,0] = initial_frac
	species = ['silicates', 'carbon', 'SiC', 'iron']
	for i in range(N-1):
		for j,s in enumerate(species):
			growth_timescale,_ = speciesGrowthTime(temp,dens,metallicity,s)
			if growth_timescale == 0. or np.isinf(growth_timescale):
				dust_frac[j,i+1] = dust_frac[j,i]
			else:
				dust_frac[j,i+1] = dust_frac[j,i] + time_step * (1. - dust_frac[j,i])*dust_frac[j,i]/growth_timescale

		time[i+1] += time_step + time[i]

	return dust_frac, time




# Calculates the  Elemental gas-dust accretion timescale in years for all gas particles in the given snapshot gas particle structure
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
def calc_spec_key_elem(G):
	atomic_mass = np.array([1.01, 2.0, 12.01, 14, 15.99, 20.2, 24.305, 28.086, 32.065, 40.078, 55.845])
	elem_num_dens = np.multiply(G.z[:,:len(atomic_mass)], G.rho[:, np.newaxis]*config.UnitDensity_in_cgs) / (atomic_mass*config.H_MASS)
	sil_elems_index = np.array([4,6,7,10]) # O,Mg,Si,Fe
	# number of atoms that make up one formula unit of silicate dust assuming an olivine, pyroxene mixture
	# with olivine fraction of 0.32 and Mg fraction of 0.8
	sil_num_atoms = np.array([3.63077,1.06,1.,0.570769]) # O, Mg, Si, Fe

	sil_num_dens = elem_num_dens[:,sil_elems_index] 
	# Find element with lowest number density factoring in number of atoms needed to make one formula unit of the dust species
	key = np.argmin(sil_num_dens / sil_num_atoms, axis = 1)
	key_num_dens = sil_num_dens[range(sil_num_dens.shape[0]),key]
	key_elem = sil_elems_index[key]
	key_in_dust = sil_num_atoms[key]

	return key_elem, key_num_dens, key_in_dust



# Calculates the Species gas-dust accretion timescale in years for all gas particles in the given snapshot gas particle structure
def calc_spec_acc_timescale(G, CNM_thresh=0.95, nano_iron=False):

	T_ref = 300 		# K
	nM_ref = 1E-2   	# reference number density for metals in 1 H cm^-3
	ref_cond_dens = 3	# reference condensed dust species density g cm^-3
	T_cut = 300 		# K cutoff temperature for step func. sticking efficiency
	iron_incl = 0.7		# when using nan_iron, fraction of iron hidden in silicate dust and not available for acc.

	T = G.T
	fH2 = utils.calc_fH2(G)

	timescales = dict.fromkeys(['Silicates', 'Carbon', 'Iron'], None) 

	###############
    ## SILICATES 
    ###############
	t_ref = np.zeros(len(T))
	t_ref[fH2<CNM_thresh] = 4.4E6
	t_ref[fH2>=CNM_thresh] = 23.9E6

	dust_formula_mass = 0.0
	atomic_mass = np.array([1.01, 2.0, 12.01, 14, 15.99, 20.2, 24.305, 28.086, 32.065, 40.078, 55.845])
	elem_num_dens = np.multiply(G.z[:,:len(atomic_mass)], G.rho[:, np.newaxis]*config.UnitDensity_in_cgs) / (atomic_mass*config.H_MASS)
	sil_elems_index = np.array([4,6,7,10]) # O,Mg,Si,Fe
	# number of atoms that make up one formula unit of silicate dust assuming an olivine, pyroxene mixture
	# with olivine fraction of 0.32 and Mg fraction of 0.8
	sil_num_atoms = np.array([3.63077,1.06,1.,0.570769]) # O, Mg, Si, Fe


	for k in range(4): dust_formula_mass += sil_num_atoms[k] * atomic_mass[sil_elems_index[k]];

	key_elem, key_num_dens,key_in_dust = calc_spec_key_elem(G)
	key_mass = atomic_mass[key_elem]
	cond_dens = 3.13
	growth_time = t_ref * key_in_dust * np.sqrt(key_mass) / dust_formula_mass * (cond_dens/ref_cond_dens) * (nM_ref/key_num_dens) * np.power(T_ref/T,0.5) 
	# Now get silicate dust species timescale from key element timescale
	#growth_time *= key_in_dust*key_mass/dust_formula_mass

	growth_time[T>T_cut] = np.inf
	timescales['Silicates'] = np.copy(growth_time)

	###############
    ## CARBONACOUS 
    ###############
	t_ref = np.zeros(len(T))
	t_ref[fH2<CNM_thresh] = 26.7E6
	t_ref[fH2>=CNM_thresh] = 23.9E6

	key_elem = 2
	key_in_dust = 1
	key_mass = atomic_mass[key_elem]
	dust_formula_mass = key_mass
	cond_dens = 2.25
	key_num_dens = elem_num_dens[:,key_elem]
	growth_time = t_ref * key_in_dust * np.sqrt(key_mass) / dust_formula_mass * (cond_dens/ref_cond_dens) * (nM_ref/key_num_dens) * np.power(T_ref/T,0.5)
	growth_time[T>T_cut] = np.inf
	timescales['Carbon'] = np.copy(growth_time)

	###############
    ## IRON 
    ###############
	t_ref = np.zeros(len(T))
	if nano_iron:
		t_ref[fH2<CNM_thresh] = 0.029E6
		t_ref[fH2>=CNM_thresh] = 2.42E6
	else:
		t_ref[fH2<CNM_thresh] = 4.4E6
		t_ref[fH2>=CNM_thresh] = 23.9E6

	key_elem = 10
	key_in_dust = 1
	dust_formula_mass = atomic_mass[key_elem]
	cond_dens = 7.86
	key_num_dens = elem_num_dens[:,key_elem]
	key_mass = atomic_mass[key_elem]
	growth_time = t_ref * key_in_dust * np.sqrt(key_mass) / dust_formula_mass * (cond_dens/ref_cond_dens) * (nM_ref/key_num_dens) * np.power(T_ref/T,0.5)
	growth_time[T>T_cut] = np.inf
	timescales['Iron'] = np.copy(growth_time)

	return timescales



# Calculates the instantaneous dust production from accertion for the given snapshot gas particle structure
def calc_dust_acc(G, implementation='species', CNM_thresh=0.95, CO_frac=0.2, nano_iron=False):

	iron_incl = 0.7

	atomic_mass = np.array([1.01, 2.0, 12.01, 14, 15.99, 20.2, 24.305, 28.086, 32.065, 40.078, 55.845])
	sil_elems_index = np.array([4,6,7,10])
	# number of atoms that make up one formula unit of silicate dust assuming an olivine, pyroxene mixture
	# with olivine fraction of 0.32 and Mg fraction of 0.8
	sil_num_atoms = np.array([3.63077,1.06,1.,0.570769]) # O, Mg, Si, Fe

	M = G.m
	fH2 = utils.calc_fH2(G)
	C_in_CO = np.zeros(len(M))
	C_in_CO[fH2>=CNM_thresh] = CO_frac

	O_in_CO = np.zeros(len(M))
	# Special case where you put the rest of C into CO
	if CO_frac == 1.:
		O_in_CO[fH2>=CNM_thresh] = (G.z[fH2>=CNM_thresh,2]-G.dz[:,2]) * atomic_mass[4] / atomic_mass[2] / G.z[fH2>=CNM_thresh,4]
	else:
		O_in_CO[fH2>=CNM_thresh] = CO_frac * G.z[fH2>=CNM_thresh,2] * atomic_mass[4] / atomic_mass[2] / G.z[fH2>=CNM_thresh,4]

	# Needed to select arbitrary elements from each row for 2D numpy arrays
	farg = np.arange(len(M))

	if implementation == 'elemental':
		timescales = calc_elem_acc_timescale(G)
		growth_timescale = timescales['Silicates']
		sil_DZ = G.dz[:,[4,6,7,10]]/G.z[:,[4,6,7,10]]
		# Account for O locked in CO which reduced the max amount of O in dust
		sil_DZ[:,0] = np.multiply(sil_DZ[:,0], 1./(1.-O_in_CO))
		sil_DZ[np.logical_or(sil_DZ <= 0,sil_DZ >= 1)] = 1.
		sil_dust_mass = np.multiply(G.dz[:,[4,6,7,10]],M[:,np.newaxis]*1E10)
		sil_dust_prod = np.sum((1.-sil_DZ)*sil_dust_mass/growth_timescale[:,np.newaxis],axis=1)
		

		growth_timescale = timescales['Carbon']
		C_DZ = G.dz[:,2]/((1-C_in_CO)*G.z[:,2])
		C_dust_mass = G.dz[:,2]*M*1E10
		carbon_dust_prod = (1.-C_DZ)*C_dust_mass/growth_timescale
		carbon_dust_prod[np.logical_or(C_DZ <= 0,C_DZ >= 1)] = 0.

		iron_dust_prod = np.zeros(len(sil_dust_prod))

		O_dust_prod = np.zeros(len(sil_dust_prod))

	else:
		timescales = calc_spec_acc_timescale(G, CNM_thresh=CNM_thresh, nano_iron=nano_iron)
		####################
		## SILICATES 
		####################
		growth_timescale = timescales['Silicates']
		key_elem, key_num_dens, key_in_dust = calc_spec_key_elem(G)

		sil_dust_formula_mass = 0.0
		for k in range(4): sil_dust_formula_mass += sil_num_atoms[k] * atomic_mass[sil_elems_index[k]];
		key_DZ = G.dz[farg,key_elem]/G.z[farg,key_elem]
		# Deal with nan data
		key_DZ[np.isnan(key_DZ)] = 0.

		key_M_dust = G.dz[farg,key_elem]*M*1E10
		sil_dust_prod = (1.-key_DZ)*key_M_dust/growth_timescale
		sil_dust_prod[np.logical_or(key_DZ <= 0,key_DZ >= 1)] = 0.
		sil_dust_prod /= key_in_dust*atomic_mass[key_elem]/sil_dust_formula_mass

		####################
		## CARBONACOUS 
		####################
		growth_timescale = timescales['Carbon']
		key_elem = 2
		key_DZ = G.dz[:,key_elem]/((1-C_in_CO)*G.z[:,key_elem])
		# Deal with nan data
		key_DZ[np.isnan(key_DZ)] = 0.

		key_M_dust = G.dz[:,key_elem]*M*1E10
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

		key_M_dust = G.dz[:,key_elem]*M*1E10
		# If nanoparticle iron dust need to account for amount of iron locked in silicate grains
		# and unavailable for accretion
		if nano_iron:
			key_M_dust *= 1.- iron_incl
		iron_dust_prod = (1.-key_DZ)*key_M_dust/growth_timescale
		iron_dust_prod[np.logical_or(key_DZ <= 0,key_DZ >= 1)] = 0.


		####################
		## OXYGEN RESERVOIR 
		####################
		# Check if sim was run with optional O reservoir
		# TODO : Try and implement a meaningful O reservoir dust production rate
		"""
		if np.shape(G['spec'])[1] > 4:
			nH = G['rho']*UnitDensity_in_cgs * ( 1. - (G['z'][:,0]+G['z'][:,1])) / H_MASS
			nH = G['rho']*UnitDensity_in_cgs * 0.76 / H_MASS
			# expected fractional O depletion
			D_O = 1. - 0.65441 / np.power(nH,0.103725)
			D_O[D_O<0] = 0
			key_elem = timescales['silicates'][1].astype(int)
			key_mass = atomic_mass[key_elem]
			atomic_mass = np.array([1.01, 2.0, 12.01, 14, 15.99, 20.2, 24.305, 28.086, 32.065, 40.078, 55.845])
			num_atoms = np.array([0,0,0,0,3.63077,0,1.06,1.,0,0,0.570769]) # num atoms in sil
			key_num_atoms = num_atoms[key_elem]

			# fraction of maximum possible silicate dust present
			if depletion:
				frac_of_sil = G['spec'][:,0] / ((G['z'][farg,key_elem]+G['dz'][farg,key_elem]) * sil_dust_formula_mass/(key_num_atoms * key_mass))
				max_O_in_sil = (G['z'][farg,key_elem]+G['dz'][farg,key_elem]) * ((sil_num_atoms[0] * atomic_mass[4])/(key_num_atoms * key_mass));
				extra_O = frac_of_sil * D_O * (G['z'][:,4]+G['dz'][:,4]) - max_O_in_sil - G['spec'][:,4];
			else:
				frac_of_sil = (G['spec'][:,0]) / ((G['z'][farg,key_elem]) * sil_dust_formula_mass/(key_num_atoms * key_mass))
				max_O_in_sil = G['z'][farg,key_elem] * ((sil_num_atoms[0] * atomic_mass[4])/(key_num_atoms * key_mass));
				extra_O = frac_of_sil * D_O * G['z'][:,4] - max_O_in_sil - G['spec'][:,4];
				#extra_O *= M*1E10

			print(np.unique(key_elem))

			mask = extra_O > 0.
			mask = nH > 10
			print('n_H = ', nH[mask])
			print('Z_O = ',G['z'][:,4][mask])
			print('f_sil = ',frac_of_sil[mask])
			print('f_(max O in sil) =' ,max_O_in_sil[mask]/G['z'][:,4][mask])
			print('D_O = ',D_O[mask])
			print('f_(O res) = ', G['spec'][:,4][mask]/G['z'][:,4][mask])
			print(frac_of_sil[mask] * D_O[mask])
			print(extra_O[mask])

			print 1.*len(extra_O[extra_O > 0])/len(extra_O)

			extra_O[extra_O < 0] = 0
			extra_O[sil_dust_prod <= 0] = 0
			# If needed O depletion can't be attributed to silicate dust and what's already in the oxygen reservoir throw more oxygen into the reservoir
			O_dust_prod = extra_O

			print(np.sum(extra_O))

		else:
			O_dust_prod = np.zeros(len(sil_dust_prod))


	dust_prod = {'Silicates':sil_dust_prod,'Carbon':carbon_dust_prod,'Iron':iron_dust_prod,'O Reservoir':O_dust_prod}
		"""

	dust_prod = {'Silicates':sil_dust_prod,'Carbon':carbon_dust_prod,'Iron':iron_dust_prod}
	return dust_prod