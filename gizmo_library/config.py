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
BASE_LINEWIDTH = 4.0
BASE_LINESTYLE = '-'
BASE_MARKERSTYLE = 'o'
BASE_MARKERSIZE = 10

# Set personal color, linewidths, and styles cycle
LINE_COLORS = ["xkcd:azure", "xkcd:tomato", "xkcd:green", "xkcd:orchid", "xkcd:teal", "xkcd:sienna"]
MARKER_COLORS = ["xkcd:orange", "xkcd:teal", "xkcd:sienna", "xkcd:gold", "xkcd:magenta"]
LINE_STYLES = ['-','--',':','-.',(0, (3, 5, 1, 5, 1, 5))]
MARKER_STYLES = ['o','^','X','s','v']
LINE_WIDTHS = np.array([0.25,0.5,0.75,1.0,1.25,1.5])*BASE_LINEWIDTH
AXIS_BORDER_WIDTH = 3
BASE_ELINEWIDTH = 3

# Font sizes and fig sizes for plots
XSMALL_FONT					= 16
SMALL_FONT					= 22
LARGE_FONT					= 30
EXTRA_LARGE_FONT			= 40
BASE_FIG_XSIZE = 10
BASE_FIG_YSIZE = 7.5
# Change these on the fly if you want to increase or decrease image size
FIG_XRATIO = 1.
FIG_YRATIO = 1.



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
Cm_to_pc					= 3.24078e-19
Kpc_to_cm					= 3.086E21
SOLAR_GAL_RADIUS			= 8 # kpc

FIRE_VERSION 				= 2
if FIRE_VERSION == 2:
	SOLAR_Z					= 0.02
else:
	SOLAR_Z					= 0.014


ELEMENTS					= ['Z','He','C','N','O','Ne','Mg','Si','S','Ca','Fe']
ELEMENT_NAMES				= ['Metals','Helium','Carbon','Nitrogen','Oxygen','Neon','Magnesium','Silicon','Sulfur','Calcium','Iron']
ATOMIC_MASS					= np.array([1.01, 2.0, 12.01, 14, 15.99, 20.2, 24.305, 28.086, 32.065, 40.078, 55.845])
SIL_ELEM_INDEX				= np.array([4,6,7,10]) # O,Mg,Si,Fe
# number of atoms that make up one formula unit of silicate dust assuming an olivine, pyroxene mixture
# with olivine fraction of 0.32 and Mg fraction of 0.8
SIL_NUM_ATOMS				= np.array([3.631,1.06,1.,0.571]) # O,Mg,Si,Fe

DUST_SPECIES				= ['Silicates','Carbon','SiC','Iron','O Reservoir','Iron Inclusions']
DUST_SOURCES				= ['Accretion','SNe Ia', 'SNe II', 'AGB']

# Houses labels, limits, and if they should be plotted in log space for possible properties
PROP_INFO  				= {'fH2': [r'$f_{\rm H_2}$', [0., 1.], False],
						   'fMC': [r'$f_{\rm MC}$', 										[0,1.05], 		False],
						 'CinCO': [r'$f_{\rm C\;in\;CO}$', 									[0,1.05], 		False],
							 'r': ['Radius [kpc]', 											[0.1,20], 		False],
						   'r25': [r'Radius [R$_{25}$]', 									[0.1,2], 			False],
					 'sigma_gas': [r'$\Sigma_{\rm gas}$ [M$_{\odot}$ pc$^{-2}$]', 			[1E0,1E2], 		True],
				  'sigma_metals': [r'$\Sigma_{\rm metals}$ [M$_{\odot}$ pc$^{-2}$]', 		[1E-2,1E1], 	True],
					'sigma_dust': [r'$\Sigma_{\rm dust}$ [M$_{\odot}$ pc$^{-2}$]', 			[1E-3,1E0], 	True],
					 'sigma_sil': [r'$\Sigma_{\rm sil}$ [M$_{\odot}$ pc$^{-2}$]', 			[1E-3,1E0], 	True],
					'sigma_carb': [r'$\Sigma_{\rm carb}$ [M$_{\odot}$ pc$^{-2}$]', 			[1E-3,1E0], 	True],
					 'sigma_SiC': [r'$\Sigma_{\rm SiC}$ [M$_{\odot}$ pc$^{-2}$]', 			[1E-7,1E-3], 	True],
					'sigma_iron': [r'$\Sigma_{\rm iron}$ [M$_{\odot}$ pc$^{-2}$]', 			[1E-3,1E0], 	True],
					'sigma_ORes': [r'$\Sigma_{\rm O\;Res}$ [M$_{\odot}$ pc$^{-2}$]', 		[1E-3,1E0], 	True],
					'sigma_sil+': [r'$\Sigma_{\rm sil+}$ [M$_{\odot}$ pc$^{-2}$]', 			[1E-3,1E0], 	True],
					  'sigma_H2': [r'$\Sigma_{H_2}$ [M$_{\odot}$ pc$^{-2}$]', 				[1E-3,1E0], 	True],
					'NH_neutral': [r'$N_{\rm H,neutral}$ [cm$^{-2}$]',						[1.1E18,0.9E22],True],
							'NX': [r'$N_{\rm X}$ [cm$^{-2}$]',								[1E16,1E19],	True],
						  'time': ['Time [Gyr]',											[1E-2,1E1],		True],
					  'star_age': ['Stellar Population Age [Gyr]',							[2E-3,1E1],		True],
					  'redshift': ['z',														[1E-1,100],		True],
						 'M_gas': [r'$M_{\rm gas}\;[M_{\odot}]$',							[1E4,1E8],		True],
						  'M_H2': [r'$M_{\rm H_2}\;[M_{\odot}]$',							[1E1,1E7],		True],
							'nH': [r'$n_{\rm H}$ [cm$^{-3}$]', 								[1.1E-2,0.9E3], True],
							 'T': [r'T [K]', 												[1.1E1,0.9E5],  True],
							 'Z': [r'Z [Z$_{\odot}$]', 										[1.1E-3,5E0], 	True],
						   'O/H': ['12+log(O/H)', 											[6.5,9.5], 	    False],
						   'D/Z': ['D/Z', 													[0,1.05],   	False],
					 'depletion': [r'$\delta_{\rm X}$', 									[1E-3,1.1E0],   True],
				   'C_depletion': [r'$\delta_{\rm C}$', 									[1E-1,1.1E0],   True],
				   'O_depletion': [r'$\delta_{\rm O}$', 									[1E-1,1.1E0],   True],
				  'Mg_depletion': [r'$\delta_{\rm Mg}$', 									[1E-3,1.1E0],   True],
				  'Si_depletion': [r'$\delta_{\rm Si}$', 									[1E-3,1.1E0],   True],
				  'Fe_depletion': [r'$\delta_{\rm Fe}$', 									[1E-3,1.1E0],   True],
				 'cum_dust_prod': [r'Cum. Dust Ratio $[M_{\rm dust}/M_{\star}]$', 			[1E-6,1E-2], 	True],
			   'cum_metal_yield': [r'Cum. Metal Ratio $[M_{\rm metal}/M_{\star}]$',			[1E-4,0.7E-1], 	True],
				'inst_dust_prod': [r'Cum. Inst. Dust Prod. [$M_{\odot}/$yr]', 				[1E-2,1E0], 	True],
				   'g_timescale': [r'$\tau_{\rm g}$ [Gyr]',									[1E-4,1E0],		True],
			  'g_timescale_frac': [r'Fraction of Gas < $\tau_{\rm g}$',						[0,1.05],		False],
				   'source_frac': ['Source Mass Fraction', 									[1E-2,1.05], 	True],
					 'spec_frac': ['Species Mass Fraction', 								[0,1.05], 		False],
						  'Si/C': ['Sil-to-C Ratio', 										[0,10], 		False],
					 'mass_frac': ['Mass Fraction',											[0,1.05],		False]
							}