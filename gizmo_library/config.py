import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Allow plotting when run on command line
plt.switch_backend('agg')
# Set style of plots
plt.style.use('seaborn-talk')
# Set base colors, linewidths, and styles for plotting
BASE_COLOR = 'xkcd:black'
BASE_CMAP = 'plasma'
BASE_DIVERGING_CMAP = 'Spectral'
BASE_LINEWIDTH = 4.0
BASE_LINESTYLE = '-'
BASE_MARKERSTYLE = 'o'
BASE_MARKERSIZE = 10

# Set personal color, linewidths, and styles cycle
LINE_COLORS = ["xkcd:azure","xkcd:tomato","xkcd:green","xkcd:orchid","xkcd:teal","xkcd:sienna","xkcd:magenta","xkcd:orange","xkcd:gold"]
MARKER_COLORS = ["xkcd:orange","xkcd:teal","xkcd:sienna","xkcd:gold","xkcd:magenta","xkcd:azure","xkcd:tomato","xkcd:green","xkcd:orchid"]
LINE_STYLES = ['-','--',':','-.',(0, (3, 5, 1, 5, 1, 5)),'-','--',':','-.',(0, (3, 5, 1, 5, 1, 5))]
MARKER_STYLES = ['o','^','X','s','v']
LINE_WIDTHS = np.array([0.25,0.5,0.75,1.0,1.25,1.5])*BASE_LINEWIDTH
AXIS_BORDER_WIDTH = 3
BASE_ELINEWIDTH = 3

# Font sizes and fig sizes for plots
XSMALL_FONT					= 16
SMALL_FONT					= 22
LARGE_FONT					= 30
EXTRA_LARGE_FONT			= 40
BASE_FIG_XSIZE 				= 10
BASE_FIG_YSIZE 				= 7.5
BASE_FIG_SIZE  				= 10
# Change these on the fly if you want to increase or decrease image size
FIG_XRATIO 					= 1.
FIG_YRATIO 					= 1.



plt.rcParams['figure.dpi'] = 200
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=LINE_COLORS)
mpl.rcParams["legend.labelspacing"] = 0.35
mpl.rcParams["legend.columnspacing"] = 0.75
# This looks to be the only way to set hatch line widths!
mpl.rcParams['hatch.linewidth'] = BASE_ELINEWIDTH

# Conversion factors for code to cgs units
UnitLength_in_cm            = 3.085678e21   # 1.0 kpc/h
UnitMass_in_g               = 1.989e43  	# 1.0e10 solar masses/h
UnitMass_in_Msolar			= UnitMass_in_g / 1.989E33
Grams_to_Msolar 			= 5.02785e-34
UnitVelocity_in_cm_per_s    = 1.0e5   	    # 1 km/sec
UnitTime_in_s 				= UnitLength_in_cm / UnitVelocity_in_cm_per_s # 0.978 Gyr/h
UnitTime_in_Gyr 			= UnitTime_in_s /1e9/365./24./3600.
UnitEnergy_per_Mass 		= np.power(UnitLength_in_cm, 2) / np.power(UnitTime_in_s, 2)
UnitDensity_in_cgs 			= UnitMass_in_g / np.power(UnitLength_in_cm, 3)
H_MASS 						= 1.674E-24 # grams
PROTONMASS					= H_MASS
BoltzMann_ergs              = 1.3806e-16
EPSILON						= 1E-7 # small number to avoid zeros
U_to_temp					=  ((PROTONMASS/BoltzMann_ergs)*(UnitVelocity_in_cm_per_s**2))
Cm_to_pc					= 3.24078e-19
Kpc_to_cm					= 3.086E21
km_per_kpc					= 3.086E16
sec_per_Gyr					= 3.16E16
Ergs_per_joule				= 1E7
SOLAR_GAL_RADIUS			= 8 # kpc
HUBBLE 						= 0.702
OMEGA_MATTER 				= 0.272

FIRE_VERSION 				= 2
if FIRE_VERSION == 2:
	# FIRE-2 uses Anders & Grevesse 1989 for Solar
	SOLAR_Z					= 0.02
	SOLAR_MASSFRAC			= np.array([0.02,0.28,3.26e-3,1.32e-3,8.65e-3,2.22e-3,9.31e-4,1.08e-3,6.44e-4,1.01e-4,1.73e-3])
else:
	# FIRE-3 uses Asplund+ 2009 for Solar
	SOLAR_Z					= 0.0142
	SOLAR_MASSFRAC 			= np.array([0.0142,0.2703,2.53e-3,7.41e-4,6.13e-3,1.34e-3,7.57e-4,7.12e-4,3.31e-4,6.87e-5,1.38e-3])


ELEMENTS					= ['Z','He','C','N','O','Ne','Mg','Si','S','Ca','Fe']
ELEMENT_NAMES				= ['Metals','Helium','Carbon','Nitrogen','Oxygen','Neon','Magnesium','Silicon','Sulfur','Calcium','Iron']
ATOMIC_MASS					= np.array([1.01, 4.0, 12.01, 14, 15.99, 20.2, 24.305, 28.086, 32.065, 40.078, 55.845])
SIL_ELEM_INDEX				= np.array([4,6,7,10]) # O,Mg,Si,Fe
# number of atoms that make up one formula unit of silicate dust assuming an olivine, pyroxene mixture
# with olivine fraction of 0.32 and Mg fraction of 0.8
SIL_NUM_ATOMS				= np.array([3.631,1.06,1.,0.571]) # O,Mg,Si,Fe

DUST_SPECIES				= ['Silicates','Carbon','SiC','Iron','O Reservoir','Iron Inclusions']
DUST_SOURCES				= ['Accretion','SNe Ia', 'SNe II', 'AGB']

# Houses labels, limits, and if they should be plotted in log space for possible properties
PROP_INFO  				= {'fH2': [r'$f_{\rm H_2}$', 													[0., 1.], 		False],
						'fdense': [r'$f_{\rm dense}$', 													[0,1.05], 		False],
						 'CinCO': [r'$f_{\rm C\;in\;CO}$', 												[0,1.05], 		False],
							 'r': ['Radius [kpc]', 														[0.1,20], 		False],
						   'r25': [r'Radius [R$_{25}$]', 												[0.,1], 		False],
					 'sigma_gas': [r'$\Sigma_{\rm gas}$ [M$_{\odot}$ pc$^{-2}$]', 						[1E0,1E2], 		True],
					'sigma_star': [r'$\Sigma_{\rm star}$ [M$_{\odot}$ pc$^{-2}$]', 						[1E0,1E2], 		True],
			  'sigma_young_star': [r'$\Sigma_{\rm star}$ (<10 Myr) [M$_{\odot}$ pc$^{-2}$]',			[1E0,1E2], 		True],
				 'sigma_stellar': [r'$\Sigma_{\rm star}$ [M$_{\odot}$ pc$^{-2}$]', 						[1E0,1E2], 		True],
					'sigma_sfr': [r'$\Sigma_{\rm SFR}$ [M$_{\odot}$ pc$^{-2}$]', 						[1E-3,1E-1], 	True],
						  'sfr': [r'SFR [M$_{\odot}/$yr]', 												[1E-3,5E1], 	True],
			 'sigma_gas_neutral': [r'$\Sigma_{\rm gas,neutral}$ [M$_{\odot}$ pc$^{-2}$]', 				[2E0,1E2], 		True],
				  'sigma_metals': [r'$\Sigma_{\rm metals}$ [M$_{\odot}$ pc$^{-2}$]', 					[1E-2,1E1], 	True],
					'sigma_dust': [r'$\Sigma_{\rm dust}$ [M$_{\odot}$ pc$^{-2}$]', 						[1E-3,1E0], 	True],
					 'sigma_sil': [r'$\Sigma_{\rm sil}$ [M$_{\odot}$ pc$^{-2}$]', 						[1E-3,1E0], 	True],
					'sigma_carb': [r'$\Sigma_{\rm carb}$ [M$_{\odot}$ pc$^{-2}$]', 						[1E-3,1E0], 	True],
					 'sigma_SiC': [r'$\Sigma_{\rm SiC}$ [M$_{\odot}$ pc$^{-2}$]', 						[1E-7,1E-3], 	True],
					'sigma_iron': [r'$\Sigma_{\rm iron}$ [M$_{\odot}$ pc$^{-2}$]', 						[1E-3,1E0], 	True],
					'sigma_ORes': [r'$\Sigma_{\rm O\;Res}$ [M$_{\odot}$ pc$^{-2}$]', 					[1E-3,1E0], 	True],
					'sigma_sil+': [r'$\Sigma_{\rm sil+}$ [M$_{\odot}$ pc$^{-2}$]', 						[1E-3,1E0], 	True],
					  'sigma_H2': [r'$\Sigma_{H_2}$ [M$_{\odot}$ pc$^{-2}$]', 							[1E-1,1E2], 	True],
					'NH_neutral': [r'$N_{\rm H,neutral}$ [cm$^{-2}$]',									[1.1E18,0.9E22],True],
							'NX': [r'$N_{\rm X}$ [cm$^{-2}$]',											[1E16,1E19],	True],
						  'time': ['Time [Gyr]',														[1E-2,1E1],		True],
				 'time_lookback': ['Lookback Time [Gyr]',												[1E-1,1E1],		True],
					  'star_age': ['Stellar Population Age [Gyr]',										[3E-4,1E1],		True],
						   'age': ['Stellar Population Age [Gyr]',										[3E-4,1E1],		True],
					  'redshift': [r'$z$',															    [6,0],			False],
			   'redshift_plus_1': [r'1+$z$',															[7,1],			True],
						 'M_gas': [r'$M_{\rm gas}\;[M_{\odot}]$',										[1E8,1E11],		True],
				 'M_gas_neutral': [r'$M_{\rm gas,neutral}\;[M_{\odot}]$',								[1E8,1E11],		True],
						'M_star': [r'$M_{\rm star}\;[M_{\odot}]$',										[1E8,1E11],		True],
				  'M_young_star': [r'$M_{\rm star}\;[M_{\odot} (<10 Myr)]$',							[1E8,1E11],		True],
						  'M_H2': [r'$M_{\rm H_2}\;[M_{\odot}]$',										[1E7,1E11],		True],
					  'M_metals': [r'$M_{\rm metals}\;[M_{\odot}]$',									[1E6,1E10],		True],
						'M_dust': [r'$M_{\rm dust}\;[M_{\odot}]$',										[1E4,1E9],		True],
							'nH': [r'$n_{\rm H}$ [cm$^{-3}$]', 											[3E-2, 0.9E3],  True],
					'nH_neutral': [r'$n_{\rm H,neutral}$ [cm$^{-3}$]', 									[3E-2, 0.9E3],  True],
							 'T': [r'T [K]', 															[1.1E1,0.9E5],  True],
							 'Z': [r'Z [Z$_{\odot}$]', 													[1.1E-3,5E0], 	True],
					 'stellar_Z': [r'Z_{\rm star} [Z$_{\odot}$]', 										[1.1E-3,5E0], 	True],
						   'Z_O': ['[O/H]', 															[1.1E-3,5E0], 	True],
				  	   'Z_O_gas': [r'[O/H]$_{\rm gas}$', 												[1.1E-3,5E0],	True],
						   'Z_C': ['[C/H]', 															[1.1E-3,5E0], 	True],
				  	   'Z_C_gas': [r'[C/H]$_{\rm gas}$', 												[1.1E-3,5E0],	True],
						  'Z_Mg': ['[Mg/H]', 															[1.1E-3,5E0], 	True],
				  	  'Z_Mg_gas': [r'[Mg/H]$_{\rm gas}$', 												[1.1E-3,5E0], 	True],
						 ' Z_Si': ['[Si/H]', 															[1.1E-3,5E0], 	True],
				  	  'Z_Si_gas': [r'[Si/H]$_{\rm gas}$', 												[1.1E-3,5E0], 	True],
						  'Z_Fe': ['[Fe/H]', 															[1.1E-3,5E0], 	True],
				  	  'Z_Fe_gas': [r'[Fe/H]$_{\rm gas}$', 												[1.1E-3,5E0], 	True],
						   'O/H': ['12+log(O/H)', 														[8,9], 	    	False],
				  	   'O/H_gas': [r'12+log(O/H)$_{\rm gas}$', 											[8,9],	 	    False],
						   'C/H': ['12+log(C/H)', 														[8,9], 	    	False],
				  	   'C/H_gas': [r'12+log(C/H)$_{\rm gas}$', 											[8,9],	 	    False],
						  'Mg/H': ['12+log(Mg/H)', 														[6.5,8.5], 	    False],
				  	  'Mg/H_gas': [r'12+log(Mg/H)$_{\rm gas}$', 										[6.5,8.5], 	    False],
						  'Si/H': ['12+log(Si/H)', 														[6.5,8.5], 	    False],
				  	  'Si/H_gas': [r'12+log(Si/H)$_{\rm gas}$', 										[6.5,8.5], 	    False],
						  'Fe/H': ['12+log(Fe/H)', 														[6.5,8.5], 	    False],
				  	  'Fe/H_gas': [r'12+log(Fe/H)$_{\rm gas}$', 										[6.5,8.5], 	    False],
						   'D/Z': ['D/Z', 																[0,1.05],   	False],
					 'depletion': [r'$\delta_{\rm X}$', 												[1E-3,1.1E0],   True],
				   'C_depletion': [r'$\delta_{\rm C}$', 												[1E-1,1.1E0],   True],
				   'O_depletion': [r'$\delta_{\rm O}$', 												[1E-1,1.1E0],   True],
				  'Mg_depletion': [r'$\delta_{\rm Mg}$', 												[1E-3,1.1E0],   True],
				  'Si_depletion': [r'$\delta_{\rm Si}$', 												[1E-3,1.1E0],   True],
				  'Fe_depletion': [r'$\delta_{\rm Fe}$', 												[1E-3,1.1E0],   True],
				 'cum_dust_prod': [r'Cum. Dust Mass $[M_{\rm dust}/M_{\star}]$', 						[1E-6,1E-2], 	True],
			   'cum_metal_yield': [r'Cum. Metal Mass $[M_{\rm metal}/M_{\star}]$',						[1E-4,0.7E-1], 	True],
				'inst_dust_prod': [r'Cum. Inst. Dust Prod. [$M_{\odot}/$yr]', 							[1E-2,1E0], 	True],
				   'g_timescale': [r'$\tau_{\rm g}$ [Gyr]',												[1E-4,1E0],		True],
			  'g_timescale_frac': [r'Fraction of Gas < $\tau_{\rm g}$',									[0,1.05],		False],
				   'source_frac': ['Source Mass Fraction', 												[1E-2,1.05], 	True],
					 'spec_frac': ['Species Mass Fraction', 											[0,1.05], 		False],
						  'Si/C': ['Sil-to-C Ratio', 													[0,10], 		False],
					 'mass_frac': ['Mass Fraction',														[0,1.05],		False],
					 'wind_rate': [r'Cont. Mass-Loss $\dot{M}_{\rm W}/M_{\star}$ [Gyr$^{-1}$]',			[3E-4,2E2],		True],
					  'wind_vel': [r'Mass-Loss Velocity $v_{\rm w,inj}$ [km s$^{-1}$]',					[2E1,5E3],		True],
					    'wind_E': [r'Inst. Energy Inj. $E_{\rm inj}}/M_{\star}$ [erg $s^{-1}\;M_{\star}^{-1}$]',[1E-5,1E6],	True],
					'cum_wind_E': [r'Cum. Energy $E_{\rm inj,cum}}/M_{\star}$ [erg $M_{\star}^{-1}$]',	[6E17,5E18],	True]
							}


def get_prop_label(property):
	return PROP_INFO[property][0]

def get_prop_limits(property):
	return np.array(PROP_INFO[property][1])

def get_prop_if_log(property):
	return PROP_INFO[property][2]

def set_prop_limits(property, lims):
	PROP_INFO[property][1] = lims

def set_prop_if_log(property, value):
	PROP_INFO[property][2] = value