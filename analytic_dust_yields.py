import numpy as np


# Theoretical dust yields for all sources of creation

H_MASS = 1.67E-24 #grams


# Solar Abundace values use in FIRE-2
def solarMetallicity(elem):
	SolarAbundances = np.zeros(11)
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

def SNeRates(star_age, Z, time_step):
	D_RETURN_FRAC = 0.01 # fraction of particle mass to return on a recycling step 
	GasReturnFraction = 1.
	# basic variables we will use 
	agemin=0.003401; agebrk=0.01037; agemax=0.03753; # in Gyr 
	RSNe = 0
	p = 0.
	if(star_age > agemin):
		if((star_age>=agemin) and (star_age<=agebrk)):
			RSNe = 5.408e-4; # NSNe/Myr *if* each SNe had exactly 10^51 ergs; really from the energy curve 
		if((star_age>=agebrk) and (star_age<=agemax)):
			RSNe = 2.516e-4; # this is for a 1 Msun population 
			# add contribution from Type-Ia 
		if(star_age>agemax):
			RSNe = 5.3e-8 + 1.6e-5*np.exp(-0.5*((star_age-0.05)/0.01)*((star_age-0.05)/0.01));
			# delayed population (constant rate)  +  prompt population (gaussian) 
	p = time_step * 1000 *  RSNe

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
	# Type Ia or II
	if star_age > agemax: 
		Msne = 1.4;
	else:
		Msne = 10.4

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

	if star_age > agemax:
			yields[0]=1.4;# total metal mass 
			yields[1]=0.0;#He 
			yields[2]=0.049;#C 
			yields[3]=1.2e-6;#N 
			yields[4]=0.143;#O
			yields[5]=0.0045;#Ne 
			yields[6]=0.0086;#Mg 
			yields[7]=0.156;#Si
			yields[8]=0.087;#S 
			yields[9]=0.012;#Ca 
			yields[10]=0.743;#Fe
	else:
			# SNII (IMF-averaged... may not be the best approx on short timescales..., Nomoto 2006 (arXiv:0605725) 
			yields[0]=2.0;#Z [total metal mass]
			yields[1]=3.87;#He 
			yields[2]=0.133;#C 
			yields[3]=0.0479;#N 
			yields[4]=1.17;#O
			yields[5]=0.30;#Ne 
			yields[6]=0.0987;#Mg 
			yields[7]=0.0933;#Si
			yields[8]=0.0397;#S 
			yields[9]=0.00458;#Ca 
			yields[10]=0.0741;#Fe
			# metal-dependent yields:
			if Z*solarMetallicity(0)<0.033: 
				yields[3]*=Z
			else: 
				yields[3]*=1.65 # N scaling is strongly dependent on initial metallicity of the star 
			yields[0] += yields[3]-0.0479; # correct total metal mass for this correction 

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