import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Allow plotting when run on command line
plt.switch_backend('agg')
# Set style of plots
plt.style.use('seaborn-talk')
# Set personal color cycle
BLACK = 'xkcd:black'
LINE_COLORS = ["xkcd:azure", "xkcd:tomato", "xkcd:green", "xkcd:orchid", "xkcd:teal", "xkcd:sienna"]
MARKER_COLORS = ["xkcd:orange", "xkcd:teal", "xkcd:sienna", "xkcd:gold", "xkcd:magenta"]
LINE_STYLES = ['-','--',':','-.']
MARKER_STYLES = ['o','^','X','s','v']
LINE_WIDTHS = [0.5,1.0,1.5,2.0,2.5,3.0]

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=LINE_COLORS)

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
H_MASS 						= 1.67E-24 # grams
PROTONMASS					= H_MASS
SOLAR_Z						= 0.02
BoltzMann_ergs              = 1.3806e-16
EPSILON						= 1E-7 # small number to avoid zeros
Cm_to_pc					= 3.24078e-19

# Small and large fonts for plots
SMALL_FONT					= 18
LARGE_FONT					= 26

ELEMENTS					= ['Z','He','C','N','O','Ne','Mg','Si','S','Ca','Fe']

DUST_SPECIES				= ['Silicates','Carbon','SiC','Iron','O Reservoir']
DUST_SOURCES				= ['Accretion','SNe Ia', 'SNe II', 'AGB']

# Houses labels, limits, and if they should be plotted in log space for possible parameters
PARAM_INFO  				= {'fH2': [r'$f_{H2}$', 									[0.,1.], 		False],
								 'r': ['Radius (kpc)', 									[0,20,], 		False],
							   'r25': [r'Radius (R$_{25}$)', 							[0,2], 			False],
					     'sigma_gas': [r'$\Sigma_{gas}$ (M$_{\odot}$ pc$^{-2}$)', 		[1E0,1E2], 		True],
						   'sigma_Z': [r'$\Sigma_{metals}$ (M$_{\odot}$ pc$^{-2}$)', 	[1E-3,1E0], 	True],
						'sigma_dust': [r'$\Sigma_{dust}$ (M$_{\odot}$ pc$^{-2}$)', 		[1E-3,1E0], 	True],
						  'sigma_H2': [r'$\Sigma_{H2}$ (M$_{\odot}$ pc$^{-2}$)', 		[1E-3,1E0], 	True],
						  	  'time': ['Time (Gyr)',									[1E-2,1E1],		True],
						  'redshift': ['z',												[1E-1,100],		True],
						 		 'm': [r'$M_{gas}$',									[1E1,1E7],		True],
						 	   'mH2': [r'$M_{H_2}$',									[1E1,1E7],		True],
						        'nH': [r'$n_{H}$ (cm$^{-3}$)', 							[1E-2,1E4], 	True],
						         'T': [r'T (K)', 										[0.9*1E1,1E5], 	True],
						         'Z': [r'Z (Z$_{\odot}$)', 								[1E-3,5E0], 	True],
						        'DZ': ['D/Z Ratio', 									[0,1], 			False],
						 'depletion': [r'[X/H]$_{gas}$', 								[1E-3,1E0], 	True],
				     'cum_dust_prod': [r'Cumulative Dust Ratio $(M_{dust}/M_{\star})$', [1E-6,1E-2], 	True],
					'inst_dust_prod': [r'Cumulative Inst. Dust Prod. $(M_{\odot}/yr)$', [0,2], 			False],
					   'g_timescale': [r'$\tau_{g}$ (Gyr)',								[1E-4,1E0],		True],
				  'g_timescale_frac': [r'Fraction of Gas < $\tau_{g}$',					[0,1.05],		False],
					   'source_frac': ['Source Mass Fraction', 							[1E-2,1.05], 	True],
					     'spec_frac': ['Species Mass Fraction', 						[0,1.05], 		False],
					          'Si/C': ['Sil-to-C Ratio', 								[0,10], 		False]
					     }