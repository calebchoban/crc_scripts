import numpy as np
from .. import config

# Theoretical metal dust yields for all sources of stellar creation


# Solar Abundace values used in FIRE-2 (Anders+) and FIRE-3 (Asplund+)
def solarMetallicity(elem, FIRE_ver=2):
	SolarAbundances = np.zeros(11)
	if FIRE_ver<=2:
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
	elif FIRE_ver>2:
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



# Rate of stellar winds
def stellarRates(star_age, Z, time_step, FIRE_ver=2, AGB_change='base'):
	p=0;
	# This is far smaller than usually used to get better fits to AGB returns needed for AGB dust returns
	D_RETURN_FRAC = 1E-7; #fraction of particle mass to return on a recycling step 
	if(Z>3): Z=3;
	if(Z<0.01): Z=0.01;
	# normalized  to give expected return fraction from stellar winds alone (~17%)
	if FIRE_ver <= 2:
		if star_age<=0.001:
			p=4.76317*Z
		else:
			if star_age<=0.0035:
				p=4.76317*Z*np.power(10.,1.838*(0.79+np.log10(Z))*(np.log10(star_age)-(-3.00)))
			elif AGB_change == 'base':
				if star_age<=0.1:
					p=29.4*np.power(star_age/0.0035,-3.25)+0.0041987
				else:
					p=0.41987*np.power(star_age,-1.1)/(12.9-np.log(star_age)) # normalized  to give expected return fraction from stellar winds alone (~17%)
			elif AGB_change == 'SB99fix': # This has been changed from usual FIRE-2 values to better match AGB mass returns needed for dust
				p= 29.4 * np.power(star_age / 0.0035,-1.3)
			elif AGB_change == 'AGBshift':
				if star_age<=0.03753:
					p=29.4*np.power(star_age/0.0035,-3.25)+0.0041987
				else:
					p=0.41987*np.power(star_age,-1.1)/(12.9-np.log(star_age)) # normalized  to give expected return fraction from stellar winds alone (~17%)

	# updated fit. separates the more robust line-driven winds [massive-star-dominated] component, and -very- uncertain AGB. extremely good fits to updated STARBURST99 result for a 3-part Kroupa IMF (0.3,1.3,2.3 slope, 0.01-0.08-0.5-100 Msun, 8-120 SNe/BH cutoff, wind model evolution, Geneva v40 [rotating, Geneva 2013 updated tracks, at all metallicities available, ~0.1-1 solar], sampling times 1e4-2e10 yr at high resolution 
	elif FIRE_ver > 2:
		f1=3.*np.power(Z,0.87)
		f2=20.*np.power(Z,0.45)
		f3=0.6*Z
		t1=0.0017; t2=0.004; t3=0.02; t=star_age; # fit parameters for 'massive star' mass-loss */
		# piecewise continuous function linking constant early and rapid late decay
		if t<=t1:
			p=f1
		elif t<=t2: 
			p=f1*np.power(t/t1,np.log(f2/f1)/np.log(t2/t1))
		elif t<=t3: 
			p=f2*np.power(t/t2,np.log(f3/f2)/np.log(t3/t2))
		else:
			p=f3*np.power(t/t3,-3.1) 
		# add AGB component. note that essentially no models [any of the SB99 geneva or padova tracks, or NuGrid, or recent other MESA models] predict a significant dependence on metallicity (that shifts slightly when the 'bump' occurs, but not the overall loss rate), so this term is effectively metallicity-independent */
		f_agb=0.01; t_agb=1.
		p += f_agb/((1. + np.power(t/t_agb,1.1)) * (1. + 0.01/(t/t_agb)))

	p *= time_step # fraction of particle mass expected to return in the timestep

	n_wind_0 = np.floor(p/D_RETURN_FRAC)
	p -= n_wind_0*D_RETURN_FRAC # if p >> return frac, should have > 1 event, so we inject the correct wind mass
	mass_return = n_wind_0*D_RETURN_FRAC; # add this in, then determine if there is a 'remainder' to be added as well
	if np.random.random() < p/D_RETURN_FRAC:
		mass_return += D_RETURN_FRAC # add the 'remainder' stochastically

	return mass_return


# Rate of stellar winds
def new_stellarRates(star_age, Z, time_step, FIRE_ver=2, AGB_change='base'):
	p=0;
	# This is far smaller than usually used to get better fits to AGB returns needed for AGB dust returns
	D_RETURN_FRAC = 0.01 #fraction of particle mass to return on a recycling step
	if star_age > 0.03753: D_RETURN_FRAC = 1E-3; #fraction of particle mass to return on a recycling step

	if(Z>3): Z=3;
	if(Z<0.01): Z=0.01;
	# normalized  to give expected return fraction from stellar winds alone (~17%)
	if FIRE_ver <= 2:
		if star_age<=0.001:
			p=4.76317*Z
		else:
			if star_age<=0.0035:
				p=4.76317*Z*np.power(10.,1.838*(0.79+np.log10(Z))*(np.log10(star_age)-(-3.00)))
			elif AGB_change == 'base':
				if star_age<=0.1:
					p=29.4*np.power(star_age/0.0035,-3.25)+0.0041987
				else:
					p=0.41987*np.power(star_age,-1.1)/(12.9-np.log(star_age)) # normalized  to give expected return fraction from stellar winds alone (~17%)
			elif AGB_change == 'SB99fix': # This has been changed from usual FIRE-2 values to better match AGB mass returns needed for dust
				p= 29.4 * np.power(star_age / 0.0035,-1.3)
			elif AGB_change == 'AGBshift':
				if star_age<=0.03753:
					p=29.4*np.power(star_age/0.0035,-3.25)+0.0041987
				else:
					p=0.41987*np.power(star_age,-1.1)/(12.9-np.log(star_age)) # normalized  to give expected return fraction from stellar winds alone (~17%)

	# updated fit. separates the more robust line-driven winds [massive-star-dominated] component, and -very- uncertain AGB. extremely good fits to updated STARBURST99 result for a 3-part Kroupa IMF (0.3,1.3,2.3 slope, 0.01-0.08-0.5-100 Msun, 8-120 SNe/BH cutoff, wind model evolution, Geneva v40 [rotating, Geneva 2013 updated tracks, at all metallicities available, ~0.1-1 solar], sampling times 1e4-2e10 yr at high resolution
	elif FIRE_ver > 2:
		f1=3.*np.power(Z,0.87)
		f2=20.*np.power(Z,0.45)
		f3=0.6*Z
		t1=0.0017; t2=0.004; t3=0.02; t=star_age; # fit parameters for 'massive star' mass-loss */
		# piecewise continuous function linking constant early and rapid late decay
		if t<=t1:
			p=f1
		elif t<=t2:
			p=f1*np.power(t/t1,np.log(f2/f1)/np.log(t2/t1))
		elif t<=t3:
			p=f2*np.power(t/t2,np.log(f3/f2)/np.log(t3/t2))
		else:
			p=f3*np.power(t/t3,-3.1)
		# add AGB component. note that essentially no models [any of the SB99 geneva or padova tracks, or NuGrid, or recent other MESA models] predict a significant dependence on metallicity (that shifts slightly when the 'bump' occurs, but not the overall loss rate), so this term is effectively metallicity-independent */
		#f_agb=0.01; t_agb=1.
		# p += f_agb/((1. + np.power(t/t_agb,1.1)) * (1. + 0.01/(t/t_agb)))
		# Updated to refir AGB wind rates since previous FIRE-3 AGB rates were too low to better match for stars 1.5-4 Msol sub-Chandrasekhar WDs and the correct initial-final mass relation.
		f_agb=0.1; t_agb=0.8;
		x_agb=t_agb/np.max([t,1E-4]);
		x_agb*=x_agb;
		p = f_agb * pow(x_agb,0.8) * (np.exp(-np.min([50.,x_agb*x_agb*x_agb])) + 1./(100. + x_agb))

	p *= time_step # fraction of particle mass expected to return in the timestep

	n_wind_0 = np.floor(p/D_RETURN_FRAC)
	p -= n_wind_0*D_RETURN_FRAC # if p >> return frac, should have > 1 event, so we inject the correct wind mass
	mass_return = n_wind_0*D_RETURN_FRAC; # add this in, then determine if there is a 'remainder' to be added as well
	if np.random.random() < p/D_RETURN_FRAC:
		mass_return += D_RETURN_FRAC # add the 'remainder' stochastically

	return mass_return


def wind_velocity(star_age, Z, FIRE_ver=2, AGB_change='base'):
	if FIRE_ver == 2:
		E_wind_tscaling = 0.0013
		if AGB_change == 'base':
			AGB_age = 0.1
		else:
			AGB_age = 0.03753
		if star_age <= AGB_age:
			# stellar population age dependence of specific wind energy, in units of an effective internal energy/temperature
			E_wind_tscaling = 0.0013 + 16.0/(1+np.power((star_age)/0.0025,1.4)+np.power(star_age/0.01,5.0))
		# get the actual wind velocity (multiply specific energy by units, user-set normalization, and convert)
		v_ejecta = np.sqrt(2.0 * (E_wind_tscaling * (3.0E7/((5./3.-1.)*config.U_to_temp))))
	elif FIRE_ver > 2:
		# setup: updated fit here uses the same stellar evolution models/tracks as used to compute mass-loss rates. see those for references here.
		f0=np.power(Z,0.12)
		#interpolates smoothly from OB winds through AGB, also versus Z
		v_ejecta = f0 * (3000./(1.+np.power(star_age/0.003,2.5)) + 600./(1.+np.power(np.sqrt(Z)*star_age/0.05,6.) +
					np.power(Z/0.2,1.5)) + 30.) / (config.UnitVelocity_in_cm_per_s/1E5)


	return v_ejecta


def stellar_winds(min_time, max_time, N, Z, FIRE_ver=2, AGB_change='base'):

	time = np.logspace(np.log10(min_time),np.log10(max_time),N)
	windRate = np.zeros(len(time))
	windVel = np.zeros(len(time))
	windE = np.zeros(len(time))
	cum_windE = np.zeros(len(time))
	for i,age in enumerate(time):
		if i == 0:
			time_step = time[i]
		else:
			time_step = time[i]-time[i-1]
		windRate[i] = stellarRates(age, Z, time_step,FIRE_ver=FIRE_ver,AGB_change=AGB_change)/time_step
		windVel[i] = wind_velocity(age, Z, FIRE_ver=FIRE_ver,AGB_change=AGB_change)
		windE[i] = 0.5 * (windRate[i]/config.sec_per_Gyr) * (windVel[i]*1E3)**2 * config.Ergs_per_joule
		if i == 0:
			cum_windE[i] = windE[i]*time_step*config.sec_per_Gyr
		else:
			cum_windE[i] = cum_windE[i-1]+windE[i]*time_step*config.sec_per_Gyr


	return time, windRate, windVel, windE, cum_windE




# Rate of SNe Ia and II
def SNeRates(star_age, Z, time_step, FIRE_ver=2):
	# basic variables we will use 
	agemin=0.003401; agebrk=0.01037; agemax=0.03753; # in Gyr 
	RSNe = 0
	p = 0.
	if FIRE_ver<=2:
		if star_age > agemin:
			if star_age<=agebrk:
				RSNe = 5.408e-4; # NSNe/Myr *if* each SNe had exactly 10^51 ergs; really from the energy curve 
			elif star_age<=agemax:
				RSNe = 2.516e-4; # this is for a 1 Msun population 
			# Ia (prompt Gaussian+delay, Manucci+06)
			if star_age>agemax:
				RSNe = 5.3e-8 + 1.6e-5*np.exp(-0.5*((star_age-0.05)/0.01)*((star_age-0.05)/0.01));
	elif FIRE_ver>2:
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




# Total metal and dust yields for stellar winds
def stellarYields(star_age, Z, time_step, FIRE_ver=2, routine='species_nano', trim_excess=True, AGB_change='base'):
	yields = np.zeros(11)
	dust_yields = np.zeros(11)
	species_yields = np.zeros(4)
	atomic_mass = [1.01, 2.0, 12.01, 14, 15.99, 20.2, 24.305, 28.086, 32.065, 40.078, 55.845]
	if 'species' in routine:
		M_wind = stellarRates(star_age,Z,time_step,FIRE_ver=FIRE_ver, AGB_change=AGB_change)
		sil_num_atoms = [3.631,1.06,1.,0.571] # O, Mg, Si, Fe
		sil_elems_index = [4,6,7,10] # O,Mg,Si,Fe 
		sil_formula_mass = 0
		for k in range(4):
			sil_formula_mass += sil_num_atoms[k] * atomic_mass[sil_elems_index[k]]
	else:
		condens_eff = 0.8

	for k in range(11):
		yields[k]=Z*solarMetallicity(k,FIRE_ver=FIRE_ver) # return surface abundances, to leading order

	# All, then He,C,N,O,Ne,Mg,Si,S,Ca,Fe follow AGB/O star yields in more detail for the light elements
	if FIRE_ver > 2:
		# everything except He and CNO and S-process is well-approximated by surface abundances. and CNO is conserved to high accuracy in sum for secondary production
		# define initial H, He, CNO fraction
		f_H_0=1.-(yields[0]+yields[1])
		f_He_0=yields[1]
		f_C_0=yields[2]
		f_N_0=yields[3]
		f_O_0=yields[4]
		f_CNO_0=f_C_0+f_N_0+f_O_0
		t = star_age
		z_sol = f_CNO_0 / (solarMetallicity(2,FIRE_ver=FIRE_ver)+solarMetallicity(3,FIRE_ver=FIRE_ver)+solarMetallicity(4,FIRE_ver=FIRE_ver)) # stellar population age in Gyr, and solar-scaled CNO abundance
		# model He production : this scales off of the fraction of H in IC: y here represents the yield of He produced by burning H, scales off availability
		t1=0.0028; t2=0.01; t3=2.3; t4=3.0; y1=0.4*min(np.power(z_sol+1.E-3,0.6),2.); y2=0.08; y3=0.07; y4=0.042
		if t<t1: y=y1*np.power(t/t1,3)
		elif t<t2: y=y1*np.power(t/t1,np.log(y2/y1)/np.log(t2/t1))
		elif t<t3: y=y2*np.power(t/t2,np.log(y3/y2)/np.log(t3/t2))
		elif t<t4: y=y3*np.power(t/t3,np.log(y4/y3)/np.log(t4/t3))
		else: y=y4
		yields[1] = f_He_0 + y * f_H_0 # y above
		# model secondary N production in CNO cycle: scales off of initial fraction of CNO: y here represents fraction of CO mass converted to -additional- N
		t1=0.001; t2=0.0028; t3=0.05; t4=1.9; t5=14.0; y1=0.2*max(1.E-4,min(z_sol*z_sol,0.9)); y2=0.68*min(np.power(z_sol+1.E-3,0.1),0.9); y3=0.4; y4=0.23; y5=0.065
		if t<t1:
			y=y1*np.power(t/t1,3.5)
		elif t<t2:
			y=y1*np.power(t/t1,np.log(y2/y1)/np.log(t2/t1))
		elif t<t3:
			y=y2*np.power(t/t2,np.log(y3/y2)/np.log(t3/t2))
		elif t<t4:
			y=y3*np.power(t/t3,np.log(y4/y3)/np.log(t4/t3))
		elif t<t5:
			y=y4*np.power(t/t4,np.log(y5/y4)/np.log(t5/t4))
		else:
			y=y5
		y=max(0.,min(1.,y)); frac_loss_from_C = 0.5; floss_CO = y * (f_C_0 + f_O_0); floss_C = min(frac_loss_from_C * floss_CO, 0.99*f_C_0); floss_O = floss_CO - floss_C;
		yields[3] = f_N_0 + floss_CO; yields[2] = f_C_0 - floss_C; yields[4] = f_O_0 - floss_O; # convert mass from CO to N, conserving exactly total CNO mass
		# model primary C production: scales off initial H+He, generally small compared to loss fraction above in SB99, large in some other models, very small for early OB winds
		t1=0.005; t2=0.04; t3=10.; y1=1.e-6; y2=0.001; y3=0.005
		if t<t1:
			y=y1*np.power(t/t1,3)
		elif t<t2:
			y=y1*np.power(t/t1,np.log(y2/y1)/np.log(t2/t1));
		elif t<t3:
			y=y2*np.power(t/t2,np.log(y3/y2)/np.log(t3/t2))
		else:
			y=y3
		y_H_to_C = (1.-(yields[0]+yields[1])) * y; y_He_to_C = f_He_0 * y # simply multiple initial He by this factor to get final production
		yields[1] -= y_He_to_C; yields[2] += y_H_to_C + y_He_to_C # transfer this mass fraction from H+He to C; gives stable results if 0 < f_He_0_to_C < 1
		# model S-process production: currently no S-process tracers -explicitly- followed, so we skip this step
		yields[0]=0.0
		for k in range(2,11):
			yields[0]+=yields[k] #finally, add up metals [not He!] to get actual metal yield

	elif FIRE_ver <= 2:
		# the interesting species are He & CNO: below is based on a compilation of van den Hoek & Groenewegen 1997, Marigo 2001, Izzard 2004 */
		yields[1]=0.36; # He 
		yields[2]=0.016; # C
		yields[3]=0.0041; # N
		yields[4]=0.0118; # O
		if solarMetallicity(0,FIRE_ver=FIRE_ver)*Z<0.033:
			yields[4] *= Z
		else:
			yields[4] *= 1.65 # metal-dependent yields: O scaling is strongly dependent on initial metallicity of the star //
		yields[0]=0.0 
		for k in range(2,11):
			yields[0]+=yields[k]

	AGB_age = 0.03753 if FIRE_ver<=2 else 0.044
	# Now create the dust
	if 'species' in routine:
		# Now check whether the yields are from AGB or O/B since dust only forms for AGB
		if star_age >= AGB_age:
			# convert star age to mass of stars
			mass = 2.51  * pow(star_age, -0.4);
			dM = mass - 2.51 * pow(star_age+time_step, -0.4);
			if mass >= 1.0:
				IMF = 0.2724 * pow(mass, -2.7);
			else:
				IMF = 0.2724 * pow(mass, -2.3);


			species_yields = dM * IMF * AGBDustYields(mass, Z)/M_wind

			if trim_excess:
				# Check to make sure we don't produce too much dust compared to metal yields
				# Renorm dust species if too much is produced
				# Check C
				elem_yield = species_yields[1] + species_yields[2] * atomic_mass[2] / (atomic_mass[2] + atomic_mass[7])
				if elem_yield > yields[2]:
					species_yields[1] *= yields[2]/elem_yield
					species_yields[2] *= yields[2]/elem_yield
				# Check O
				elem_yield = species_yields[0] * atomic_mass[4] * sil_num_atoms[0] / sil_formula_mass
				if elem_yield > yields[4]:
					species_yields[0] *= yields[4]/elem_yield
				# Check Mg
				elem_yield = species_yields[0] * atomic_mass[6] * sil_num_atoms[1] / sil_formula_mass
				if elem_yield > yields[6]:
					species_yields[0] *= yields[6]/elem_yield
				# Check Si
				elem_yield = species_yields[0] * atomic_mass[7] * sil_num_atoms[2] / sil_formula_mass + species_yields[2] * atomic_mass[7] / (atomic_mass[2] + atomic_mass[7])
				if elem_yield > yields[7]:
					species_yields[0] *= yields[7]/elem_yield
					species_yields[2] *= yields[7]/elem_yield
				# Check Fe
				if 'nano' in routine:
					# Fe is only in free-flying iron, assume no iron inclusions in stellar dust
					if species_yields[3] > yields[10]:
						species_yields[3] = yields[10]
				else:
					# Fe is in free-flying iron and silicates
					elem_yield = species_yields[0] * atomic_mass[10] * sil_num_atoms[3] / sil_formula_mass + species_yields[3];
					if elem_yield > yields[10]:
						species_yields[0] *= yields[10]/elem_yield;
						species_yields[3] *= yields[10]/elem_yield;

			# Convert species to elemental yields
			# Silicates
			for k in range(4):
				dust_yields[sil_elems_index[k]] += species_yields[0] * sil_num_atoms[k] * atomic_mass[sil_elems_index[k]] / sil_formula_mass
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
		if star_age >= AGB_age:
			# AGB stars with C/O number density > 1 
			if (yields[4] <= 0. or (yields[2]/atomic_mass[2])/(yields[4]/atomic_mass[4]) > 1.0):
				dust_yields[2] = yields[2] - 0.75*yields[4]; # C 
				dust_yields[0] = dust_yields[2]; 
				species_yields[1] = dust_yields[2];
			# AGB stars with C/O < 1 
			else:
				dust_yields[6] = condens_eff * yields[6] # Mg
				dust_yields[7] = condens_eff * yields[7] # Si
				dust_yields[10] = condens_eff * yields[10] # Fe
				dust_yields[4] = 16 * (dust_yields[6]/atomic_mass[6] + dust_yields[7]/atomic_mass[7] + dust_yields[10]/atomic_mass[10]); # O
				for k in range(2,11): dust_yields[0]+=dust_yields[k];
				species_yields[0] = dust_yields[4]+dust_yields[6]+dust_yields[7]+dust_yields[10]

	return yields, dust_yields, species_yields


# Total metal and dust yields for stellar winds
def new_stellarYields(star_age, Z, time_step, FIRE_ver=2, routine='species_nano', trim_excess=True, AGB_change = 'base'):
	yields = np.zeros(11)
	dust_yields = np.zeros(11)
	species_yields = np.zeros(4)
	atomic_mass = [1.01, 2.0, 12.01, 14, 15.99, 20.2, 24.305, 28.086, 32.065, 40.078, 55.845]
	if 'species' in routine:
		M_wind = new_stellarRates(star_age,Z,time_step,FIRE_ver=FIRE_ver)
		sil_num_atoms = [3.631,1.06,1.,0.571] # O, Mg, Si, Fe
		sil_elems_index = [4,6,7,10] # O,Mg,Si,Fe
		sil_formula_mass = 0
		for k in range(4):
			sil_formula_mass += sil_num_atoms[k] * atomic_mass[sil_elems_index[k]]
	else:
		condens_eff = 0.8

	for k in range(11):
		yields[k]=Z*solarMetallicity(k,FIRE_ver=FIRE_ver) # return surface abundances, to leading order

	# All, then He,C,N,O,Ne,Mg,Si,S,Ca,Fe follow AGB/O star yields in more detail for the light elements
	if FIRE_ver > 2:
		# everything except He and CNO and S-process is well-approximated by surface abundances. and CNO is conserved to high accuracy in sum for secondary production
		# define initial H, He, CNO fraction
		f_H_0=1.-(yields[0]+yields[1])
		f_He_0=yields[1]
		f_C_0=yields[2]
		f_N_0=yields[3]
		f_O_0=yields[4]
		f_CNO_0=f_C_0+f_N_0+f_O_0
		t = star_age
		z_sol = f_CNO_0 / (solarMetallicity(2,FIRE_ver=FIRE_ver)+solarMetallicity(3,FIRE_ver=FIRE_ver)+solarMetallicity(4,FIRE_ver=FIRE_ver)) # stellar population age in Gyr, and solar-scaled CNO abundance
		# model He production : this scales off of the fraction of H in IC: y here represents the yield of He produced by burning H, scales off availability
		t1=0.0028; t2=0.01; t3=2.3; t4=3.0; y1=0.4*min(np.power(z_sol+1.E-3,0.6),2.); y2=0.08; y3=0.07; y4=0.042
		if t<t1: y=y1*np.power(t/t1,3)
		elif t<t2: y=y1*np.power(t/t1,np.log(y2/y1)/np.log(t2/t1))
		elif t<t3: y=y2*np.power(t/t2,np.log(y3/y2)/np.log(t3/t2))
		elif t<t4: y=y3*np.power(t/t3,np.log(y4/y3)/np.log(t4/t3))
		else: y=y4
		yields[1] = f_He_0 + y * f_H_0 # y above
		# model secondary N production in CNO cycle: scales off of initial fraction of CNO: y here represents fraction of CO mass converted to -additional- N
		t1=0.001; t2=0.0028; t3=0.05; t4=1.9; t5=14.0; y1=0.2*max(1.E-4,min(z_sol*z_sol,0.9)); y2=0.68*min(np.power(z_sol+1.E-3,0.1),0.9); y3=0.4; y4=0.23; y5=0.065
		if t<t1: y=y1*np.power(t/t1,3.5)
		elif t<t2: y=y1*np.power(t/t1,np.log(y2/y1)/np.log(t2/t1))
		elif t<t3: y=y2*np.power(t/t2,np.log(y3/y2)/np.log(t3/t2))
		elif t<t4: y=y3*np.power(t/t3,np.log(y4/y3)/np.log(t4/t3))
		elif t<t5: y=y4*np.power(t/t4,np.log(y5/y4)/np.log(t5/t4))
		else: y=y5
		y=max(0.,min(1.,y)); frac_loss_from_C = 0.5; floss_CO = y * (f_C_0 + f_O_0); floss_C = min(frac_loss_from_C * floss_CO, 0.99*f_C_0); floss_O = floss_CO - floss_C;
		yields[3] = f_N_0 + floss_CO; yields[2] = f_C_0 - floss_C; yields[4] = f_O_0 - floss_O; # convert mass from CO to N, conserving exactly total CNO mass
		# model primary C production: scales off initial H+He, generally small compared to loss fraction above in SB99, large in some other models, very small for early OB winds
		t1=0.005; t2=0.04; t3=10.; y1=1.e-6; y2=0.001; y3=0.005
		if t<t1: y=y1*np.power(t/t1,3)
		elif t<t2: y=y1*np.power(t/t1,np.log(y2/y1)/np.log(t2/t1));
		elif t<t3: y=y2*np.power(t/t2,np.log(y3/y2)/np.log(t3/t2))
		else: y=y3
		y_H_to_C = (1.-(yields[0]+yields[1])) * y; y_He_to_C = f_He_0 * y # simply multiple initial He by this factor to get final production
		yields[1] -= y_He_to_C; yields[2] += y_H_to_C + y_He_to_C # transfer this mass fraction from H+He to C; gives stable results if 0 < f_He_0_to_C < 1
		# model S-process production: currently no S-process tracers -explicitly- followed, so we skip this step
		yields[0]=0.0
		for k in range(2,11):
			yields[0]+=yields[k] #finally, add up metals [not He!] to get actual metal yield

	elif FIRE_ver <= 2:
		# the interesting species are He & CNO: below is based on a compilation of van den Hoek & Groenewegen 1997, Marigo 2001, Izzard 2004 */
		yields[1]=0.36; # He
		yields[2]=0.016; # C
		yields[3]=0.0041; # N
		yields[4]=0.0118; # O
		if solarMetallicity(0,FIRE_ver=FIRE_ver)*Z<0.033:
			yields[4] *= Z
		else:
			yields[4] *= 1.65 # metal-dependent yields: O scaling is strongly dependent on initial metallicity of the star //
		yields[0]=0.0
		for k in range(2,11):
			yields[0]+=yields[k]

	AGB_age = 0.03753 if FIRE_ver<=2 else 0.044
	# Now create the dust
	if 'species' in routine:
		# Now check whether the yields are from AGB or O/B since dust only forms for AGB
		if star_age+time_step >= AGB_age:
			dust_rates = (CumAGBDustYields(star_age+time_step,Z)-CumAGBDustYields(star_age,Z))/time_step
			# Now recalculate wind rate
			if FIRE_ver <= 2:
				if star_age<=0.001:
					p=4.76317*Z
				else:
					if star_age<=0.0035:
						p=4.76317*Z*np.power(10.,1.838*(0.79+np.log10(Z))*(np.log10(star_age)-(-3.00)))
					elif AGB_change == 'base':
						if star_age<=0.1:
							p=29.4*np.power(star_age/0.0035,-3.25)+0.0041987
						else:
							p=0.41987*np.power(star_age,-1.1)/(12.9-np.log(star_age)) # normalized  to give expected return fraction from stellar winds alone (~17%)
					elif AGB_change == 'SB99fix': # This has been changed from usual FIRE-2 values to better match AGB mass returns needed for dust
						p= 29.4 * np.power(star_age / 0.0035,-1.3)
					elif AGB_change == 'AGBshift':
						if star_age<=0.03753:
							p=29.4*np.power(star_age/0.0035,-3.25)+0.0041987
						else:
							p=0.41987*np.power(star_age,-1.1)/(12.9-np.log(star_age)) # normalized  to give expected return fraction from stellar winds alone (~17%)
			# updated fit. separates the more robust line-driven winds [massive-star-dominated] component, and -very- uncertain AGB. extremely good fits to updated STARBURST99 result for a 3-part Kroupa IMF (0.3,1.3,2.3 slope, 0.01-0.08-0.5-100 Msun, 8-120 SNe/BH cutoff, wind model evolution, Geneva v40 [rotating, Geneva 2013 updated tracks, at all metallicities available, ~0.1-1 solar], sampling times 1e4-2e10 yr at high resolution
			elif FIRE_ver > 2:
				f1=3.*np.power(Z,0.87)
				f2=20.*np.power(Z,0.45)
				f3=0.6*Z
				t1=0.0017; t2=0.004; t3=0.02; t=star_age; # fit parameters for 'massive star' mass-loss */
				# piecewise continuous function linking constant early and rapid late decay
				if t<=t1:
					p=f1
				elif t<=t2:
					p=f1*np.power(t/t1,np.log(f2/f1)/np.log(t2/t1))
				elif t<=t3:
					p=f2*np.power(t/t2,np.log(f3/f2)/np.log(t3/t2))
				else:
					p=f3*np.power(t/t3,-3.1)
				# Only need AGB component
				f_agb=0.1; t_agb=0.8;
				x_agb=t_agb/np.max([t,1E-4]);
				x_agb*=x_agb;
				p = f_agb * np.pow(x_agb,0.8) * (np.exp(-np.min([50.,x_agb*x_agb*x_agb])) + 1./(100. + x_agb))

			wind_rate = p
			species_yields = dust_rates/wind_rate

			species_yields[species_yields<0] = 0


			if trim_excess:
				# Check to make sure we don't produce too much dust compared to metal yields
				# Renorm dust species if too much is produced
				# Check C
				elem_yield = species_yields[1] + species_yields[2] * atomic_mass[2] / (atomic_mass[2] + atomic_mass[7])
				if elem_yield > yields[2]:
					species_yields[1] *= yields[2]/elem_yield
					species_yields[2] *= yields[2]/elem_yield
				# Check O
				elem_yield = species_yields[0] * atomic_mass[4] * sil_num_atoms[0] / sil_formula_mass
				if elem_yield > yields[4]:
					species_yields[0] *= yields[4]/elem_yield
				# Check Mg
				elem_yield = species_yields[0] * atomic_mass[6] * sil_num_atoms[1] / sil_formula_mass
				if elem_yield > yields[6]:
					species_yields[0] *= yields[6]/elem_yield
				# Check Si
				elem_yield = species_yields[0] * atomic_mass[7] * sil_num_atoms[2] / sil_formula_mass + species_yields[2] * atomic_mass[7] / (atomic_mass[2] + atomic_mass[7])
				if elem_yield > yields[7]:
					species_yields[0] *= yields[7]/elem_yield
					species_yields[2] *= yields[7]/elem_yield
				# Check Fe
				if 'nano' in routine:
					# Fe is only in free-flying iron, assume no iron inclusions in stellar dust
					if species_yields[3] > yields[10]:
						species_yields[3] = yields[10]
				else:
					# Fe is in free-flying iron and silicates
					elem_yield = species_yields[0] * atomic_mass[10] * sil_num_atoms[3] / sil_formula_mass + species_yields[3];
					if elem_yield > yields[10]:
						species_yields[0] *= yields[10]/elem_yield;
						species_yields[3] *= yields[10]/elem_yield;

			# Convert species to elemental yields
			# Silicates
			for k in range(4):
				dust_yields[sil_elems_index[k]] += species_yields[0] * sil_num_atoms[k] * atomic_mass[sil_elems_index[k]] / sil_formula_mass
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
		if star_age >= AGB_age:
			# AGB stars with C/O number density > 1
			if (yields[4] <= 0. or (yields[2]/atomic_mass[2])/(yields[4]/atomic_mass[4]) > 1.0):
				dust_yields[2] = yields[2] - 0.75*yields[4]; # C
				dust_yields[0] = dust_yields[2];
				species_yields[1] = dust_yields[2];
			# AGB stars with C/O < 1
			else:
				dust_yields[6] = condens_eff * yields[6] # Mg
				dust_yields[7] = condens_eff * yields[7] # Si
				dust_yields[10] = condens_eff * yields[10] # Fe
				dust_yields[4] = 16 * (dust_yields[6]/atomic_mass[6] + dust_yields[7]/atomic_mass[7] + dust_yields[10]/atomic_mass[10]); # O
				for k in range(2,11): dust_yields[0]+=dust_yields[k];
				species_yields[0] = dust_yields[4]+dust_yields[6]+dust_yields[7]+dust_yields[10]


	return yields, dust_yields, species_yields



# SNe total metal and dust yields 
def SNeYields(star_age, Z, FIRE_ver=2, routine='species'):
	yields = np.zeros(11)
	dust_yields = np.zeros(11)
	species_yields = np.zeros(4)
	atomic_mass = [1.01, 2.0, 12.01, 14, 15.99, 20.2, 24.305, 28.086, 32.065, 40.078, 55.845]
	if 'species' in routine:
		sil_num_atoms = [3.631,1.06,1.,0.571]  # O, Mg, Si, Fe
		sil_elems_index = [4,6,7,10] # O,Mg,Si,Fe 
		SNeII_sil_cond = 0.00035; SNeII_C_cond = 0.15; SNeII_SiC_cond = 0.0003; SNeII_Fe_cond = 0.001; SNeI_Fe_cond = 0.005;
		dust_formula_mass = 0
		sil_elem_abund = np.zeros(4)
		missing_element = 0
		key_elem = 0
	else:
		C_condens_eff = 0.5
		other_condens_eff = 0.8

	# Return if first timestep
	if star_age == 0.:
		return yields, dust_yields, species_yields

	if FIRE_ver>2:
		agemax = 0.044
		# match to rates tabulation above to determine if Ia or CC 
		# default to a mean mass for Ia vs CC SNe; for updated table of SNe rates and energetics, this is the updated mean mass per explosion to give the correct total SNe mass
		if star_age > agemax: 
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

	if FIRE_ver <= 2:
		agemax = 0.03753
		if star_age > agemax: 
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

	# Convert to total mass yields
	yields = yields*Msne

	# Now create the dust
	if "species" in routine:
		# Create all the dust species w for SNe II
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
			# only a little bit of iron dust from SNe Ia
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
		species_yields[0] = dust_yields[4] + dust_yields[6] + dust_yields[7] + dust_yields[10]
		species_yields[1] = dust_yields[2];
	
	return yields, dust_yields, species_yields


def FittedCumAGBDustYields(time, z):
	returns = np.zeros(4)
	logt = np.log10(time)
	# Silicates
	###############################################################################
	if z == 2.:
		if time < 284:
			returns[0] = 1.77E-4*logt - 2.87E-4
		elif time < 1244:
			returns[0] = 3.03E-5*(logt - 2.45) + 1.47E-4
		else:
			returns[0] = 5.93E-4*(logt - 3.09) + 1.67E-4
	elif z == 1.:
		if time < 295:
			returns[0] = 4.18E-5  * logt - 6.78E-5
		elif time < 1808:
			returns[0] = 1.72E-6*(logt - 2.47) + 3.55E-5
		else:
			returns[0] = 1.05E-4*(logt - 3.26) + 3.68E-5
	elif z == 0.4:
		if time < 286:
			returns[0] = 5.00E-6*logt - 8.01E-6
		elif time < 3948:
			returns[0] = 1.85E-9*(logt - 2.46) + 4.25E-6
		else:
			returns[0] = 3.35E-7*(logt - 3.60) + 4.26E-6
	elif z == 0.2:
		if time < 269:
			returns[0] = 1.04E-8*logt - 1.64E-08
		elif time < 1560:
			returns[0] = 2.69E-10*(logt - 2.43) + 8.82E-9
		else:
			returns[0] = 3.05E-19*(logt - 3.19) + 9.03E-9
	elif z == 0.05:
		if time < 147:
			returns[0] = 5.80E-11*logt - 9.10E-11
		elif time < 252:
			returns[0] = 4.10E-11*(logt - 2.17) + 3.47E-11
		else:
			returns[0] = 6.05E-14*(logt - 2.40) + 4.43E-11
	# Carbon
	###############################################################################
	if z == 2.:
		if time < 262:
			returns[1] = 8.10E-7*logt - 1.54E-6
		elif time < 840:
			returns[1] = 4.00E-4*(logt - 2.42) + 4.23E-7
		else:
			returns[1] = 7.37E-6*(logt - 2.92) + 2.02E-4
	elif z == 1.:
		if time < 305:
			returns[1] = 3.66E-6*logt - 6.75E-6
		elif time < 1250:
			returns[1] = 3.71E-4*(logt - 2.48) +  2.34E-6
		else:
			returns[1] = 8.67E-6*(logt - 3.10) + 2.29E-4
	elif z == 0.4:
		if time < 367:
			returns[1] = 1.27E-5*logt - 2.34E-5
		elif time < 2329:
			returns[1] = 4.24E-4*(logt - 2.56) + 9.27E-6
		else:
			returns[1] = 7.55E-5*(logt - 3.37) + 3.49E-4
	elif z == 0.2:
		if time < 344:
			returns[1] = 1.40E-5*logt - 2.53E-5
		elif time < 3105:
			returns[1] = 4.44E-4*(logt - 2.54) + 1.03e-5
		else:
			returns[1] = 9.59e-5*(logt - 3.49) + 4.35E-4
	elif z == 0.05:
		if time < 280:
			returns[1] = 8.48E-6*logt - 1.43E-5
		elif time < 4504:
			returns[1] = 3.10E-4*(logt - 2.45) + 6.47E-6
		else:
			returns[1] = 5.73E-5*(logt - 3.65) + 3.80E-4
	# SiC
	###############################################################################
	if z == 2.:
		if time < 272:
			returns[2] = 3.55E-7*logt - 6.64E-7
		elif time < 890:
			returns[2] = 1.03E-4*(logt - 2.43) + 2.00E-7
		else:
			returns[2] = 8.64E-7*(logt - 2.95) + 5.31E-5
	elif z == 1.:
		if time < 272:
			returns[2] = 3.73E-7*logt - 6.33E-7
		elif time < 1544:
			returns[2] = 2.93E-5*(logt - 2.43) +  2.75E-7
		else:
			returns[2] = 5.82E-7*(logt - 3.19) + 2.24E-5
	elif z == 0.4:
		if time < 235:
			returns[2] = 1.06E-7*logt - 1.74E-7
		elif time < 4812:
			returns[2] = 1.04E-6*(logt - 2.37) + 7.84E-8
		else:
			returns[2] = 2.47E-7*(logt - 3.68) + 1.44E-6
	elif z == 0.2:
		if time < 202:
			returns[2] = 6.38E-9*logt - 1.02E-8
		elif time < 3394:
			returns[2] = 2.48E-8*(logt - 2.31) + 4.52E-9
		else:
			returns[2] = 9.51E-9*(logt - 3.53) + 3.48E-8
	elif z == 0.05:
		if time < 245:
			returns[2] = 2.97E-11*logt - 4.81E-11
		elif time < 2392:
			returns[2] = 1.11E-10*(logt - 2.39) + 2.28E-11
		else:
			returns[2] = 3.73E-13*(logt - 3.38) + 1.33E-10
	# Iron
	###############################################################################
	if z == 2.:
		if time < 525:
			returns[3] = 5.98E-6*logt - 9.97E-6
		elif time < 1108:
			returns[3] = 1.02E-4*(logt - 2.72) + 6.30E-6
		else:
			returns[3] = 5.98E-5*(logt - 3.04) + 3.94E-5
	elif z == 1.:
		if time < 339:
			returns[3] = 2.16E-6*logt - 3.60E-6
		elif time < 1074:
			returns[3] = 4.09E-7*(logt - 2.53) + 1.87E-6
		else:
			returns[3] = 1.50E-5*(logt - 3.03) + 2.08E-6
	elif z == 0.4:
		if time < 307:
			returns[3] = 9.86E-7*logt - 1.62E-6
		elif time < 4120:
			returns[3] = 1.13e-9*(logt - 2.49) + 8.34E-7
		else:
			returns[3] = 2.23E-7*(logt - 3.61) + 8.36E-07
	elif z == 0.2:
		if time < 253:
			returns[3] = 5.53E-9*logt - 8.71E-9
		elif time < 297:
			returns[3] = 2.50E-9*(logt - 2.40) +  4.57E-9
		else:
			returns[3] = 9.50E-12 *(logt - 2.47) + 4.75E-9
	elif z == 0.05:
		if time < 128:
			returns[3] = 3.18E-11*logt - 5.00E-11
		elif time < 269:
			returns[3] = 2.69E-11*(logt - 2.11) + 1.70E-11
		else:
			returns[3] = 2.14E-14*(logt - 2.43) + 2.57E-11

	returns[returns<0] = 0.
	return returns


def CumAGBDustYields(time,z):
	time *= 1E3 # Convert to Myr

	if z <= 0.05:  returns = FittedCumAGBDustYields(time,0.05)
	elif z <= 0.2: returns = FittedCumAGBDustYields(time,0.05) + (z-0.05)/(0.2-0.05)*(FittedCumAGBDustYields(time,0.2) - FittedCumAGBDustYields(time,0.05))
	elif z <= 0.4: returns = FittedCumAGBDustYields(time,0.2) + (z-0.2)/(0.4-0.2)*(FittedCumAGBDustYields(time,0.4) - FittedCumAGBDustYields(time,0.2))
	elif z <= 1.:  returns = FittedCumAGBDustYields(time,0.4) + (z-0.4)/(1.-0.4)*(FittedCumAGBDustYields(time,1.) - FittedCumAGBDustYields(time,0.4))
	elif z <= 2.:  returns = FittedCumAGBDustYields(time,1.) + (z-1.)/(2.-1.)*(FittedCumAGBDustYields(time,2.) - FittedCumAGBDustYields(time,1.))
	else: returns = FittedCumAGBDustYields(time,2.)

	return returns




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



def totalFeedbackRates(max_time, N, Z, FIRE_ver=2):
	time_step = max_time/N
	time = np.arange(0,max_time,time_step)
	windRate = np.zeros(len(time))
	SNeRate = np.zeros(len(time))
	for i,age in enumerate(time):
		windRate[i] = stellarRates(age, Z, time_step,FIRE_ver=FIRE_ver)
		SNeRate[i] = SNeRates(age, Z, time_step,FIRE_ver=FIRE_ver)

	return windRate, SNeRate


# Create plot of stellar feedback for elements and dust for a stellar population over a given time in Gyr
def totalStellarYields(max_time, N, Z, FIRE_ver=2, routine='species'):
	time_step = max_time/N
	time = np.arange(0,max_time,time_step)
	cum_yields = np.zeros((len(time),11))
	cum_dust_yields = np.zeros((len(time),11))
	cum_species_yields = np.zeros((len(time),4))
	for i,age in enumerate(time):

		p = stellarRates(age, Z, time_step,FIRE_ver=FIRE_ver)
		stellar_yields, stellar_dust_yields, stellar_species_yields = stellarYields(age,Z,time_step,FIRE_ver=FIRE_ver,routine=routine)
		stellar_yields *= p
		stellar_dust_yields *= p
		stellar_species_yields *= p

		p = SNeRates(age, Z, time_step,FIRE_ver=FIRE_ver)
		SNe_yields,SNe_dust_yields,SNe_species_yields = SNeYields(age,Z,FIRE_ver=FIRE_ver,routine=routine)
		SNe_yields *= p
		SNe_dust_yields *= p
		SNe_species_yields *= p

		cum_yields[i] = cum_yields[i-1] + stellar_yields + SNe_yields
		cum_dust_yields[i] = cum_dust_yields[i-1] + stellar_dust_yields + SNe_dust_yields
		cum_species_yields[i] = cum_species_yields[i-1] + stellar_species_yields + SNe_species_yields

	return cum_yields, cum_dust_yields, cum_species_yields



def onlyAGBYields(max_time, N, Z, FIRE_ver=2, routine='species', AGB_change='base'):
	time_step = max_time/N
	time = np.arange(0,max_time,time_step)
	cum_yields = np.zeros((len(time),11))
	cum_dust_yields = np.zeros((len(time),11))
	cum_species_yields = np.zeros((len(time),4))
	for i,age in enumerate(time):
		p = stellarRates(age, Z, time_step,FIRE_ver=FIRE_ver, AGB_change=AGB_change)
		stellar_yields, stellar_dust_yields, stellar_species_yields = stellarYields(age,Z,time_step,FIRE_ver=FIRE_ver,routine=routine, AGB_change=AGB_change)
		stellar_yields *= p
		stellar_dust_yields *= p
		stellar_species_yields *= p

		cum_yields[i] = cum_yields[i-1] + stellar_yields
		cum_dust_yields[i] = cum_dust_yields[i-1] + stellar_dust_yields
		cum_species_yields[i] = cum_species_yields[i-1] + stellar_species_yields

	return time, cum_yields, cum_dust_yields, cum_species_yields


def new_onlyAGBYields(max_time, N, Z, FIRE_ver=2, routine='species', trim_excess=True,AGB_change='base'):
	time_step = max_time/N
	time = np.arange(0,max_time,time_step)
	time = np.array([2E-4])
	star_mass = 7000
	while time[-1] < 10:
		dt_stellar_evol = np.max([2.0e-4, time[-1]/250.]) # restrict to small steps for young stars //
		mcorr = 1.e-5 * star_mass
		if mcorr < 1 and mcorr > 0: dt_stellar_evol /= mcorr
		time = np.append(time, time[-1]+dt_stellar_evol)


	cum_yields = np.zeros((len(time),11))
	cum_dust_yields = np.zeros((len(time),11))
	cum_species_yields = np.zeros((len(time),4))
	p_total = 0
	ps = np.zeros(len(time))
	for i,age in enumerate(time):
		time_step = time[i]-time[i-1]
		if i == 0: time_step = time[i]
		p = new_stellarRates(age, Z, time_step,FIRE_ver=FIRE_ver, AGB_change = AGB_change)
		ps[i] = p
		p_total+=p
		stellar_yields, stellar_dust_yields, stellar_species_yields = new_stellarYields(age,Z,time_step,FIRE_ver=FIRE_ver,routine=routine, trim_excess=trim_excess,AGB_change = AGB_change)
		stellar_yields *= p
		stellar_dust_yields *= p
		stellar_species_yields *= p

		cum_yields[i] = cum_yields[i-1] + stellar_yields
		cum_dust_yields[i] = cum_dust_yields[i-1] + stellar_dust_yields
		cum_species_yields[i] = cum_species_yields[i-1] + stellar_species_yields

	print(time,ps)
	print(p_total)
	print(len(time))
	print(time[ps>0],ps[ps>0])

	return time, cum_yields, cum_dust_yields, cum_species_yields


def onlySNeYields(max_time, N, Z, FIRE_ver=2, routine = 'species'):
	time_step = max_time/N
	time = np.arange(0,max_time,time_step)
	cum_yields = np.zeros((len(time),11))
	cum_dust_yields = np.zeros((len(time),11))
	cum_species_yields = np.zeros((len(time),4))
	for i,age in enumerate(time):
		p = SNeRates(age, Z, time_step,FIRE_ver=FIRE_ver)
		SNe_yields,SNe_dust_yields,SNe_species_yields = SNeYields(age,Z,FIRE_ver=FIRE_ver,routine=routine)
		SNe_yields *= p
		SNe_dust_yields *= p
		SNe_species_yields *= p

		cum_yields[i] = cum_yields[i-1] + SNe_yields
		cum_dust_yields[i] = cum_dust_yields[i-1] + SNe_dust_yields
		cum_species_yields[i] = cum_species_yields[i-1] + SNe_species_yields

	return cum_yields, cum_dust_yields, cum_species_yields