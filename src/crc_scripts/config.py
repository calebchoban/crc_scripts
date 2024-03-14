import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(__file__)
OBS_DIR = os.path.join(BASE_DIR, 'observations/data/')

# # Allow plotting when run on command line
# plt.switch_backend('agg')
# Set style of plots
#plt.style.use('seaborn-v0_8-talk')
# Set base colors, linewidths, and styles for plotting
BASE_COLOR = 'xkcd:black'
BASE_CMAP = 'plasma'
BASE_DIVERGING_CMAP = 'Spectral'
BASE_LINEWIDTH = 3.0
BASE_LINESTYLE = '-'
BASE_MARKERSTYLE = 'o'
BASE_MARKERSIZE = 10**2
LARGE_MARKERSIZE = 15**2
SMALL_MARKERSIZE = 7.5**2

# Set personal color, linewidths, and styles cycle
LINE_COLORS = ["xkcd:azure","xkcd:tomato","xkcd:green","xkcd:orchid","xkcd:orange","xkcd:teal","xkcd:magenta","xkcd:gold","xkcd:sienna","xkcd:dark royal blue","xkcd:indian red","xkcd:dark grass green","xkcd:light eggplant","xkcd:apricot","xkcd:goldenrod","xkcd:peach"]
SECOND_LINE_COLORS = ["xkcd:dark royal blue","xkcd:indian red","xkcd:dark grass green","xkcd:light eggplant","xkcd:apricot","xkcd:goldenrod","xkcd:peach"]
MARKER_COLORS = ["xkcd:orange","xkcd:gold","xkcd:magenta","xkcd:teal","xkcd:sienna","xkcd:azure","xkcd:tomato","xkcd:green","xkcd:orchid",
				 "xkcd:apricot","xkcd:pale lime","xkcd:dark royal blue","xkcd:indian red","xkcd:cinnamon","xkcd:light eggplant",
				 "xkcd:peach","xkcd:olive green"]*10
LINE_STYLES = ['-','--',':','-.','-','--',':','-.','-','--',':','-.']
MARKER_STYLES = ['o','^','X','s','d','>','P','<','v','*','D','p']*10
LINE_WIDTHS = np.array([1.5,1.25,1.0,0.75,0.5,0.25])*BASE_LINEWIDTH
AXIS_BORDER_WIDTH = 3
BASE_ELINEWIDTH = 3

LATEX_PAGEWIDTH=6.9738480697 ## in
LATEX_COLUMNWIDTH=3.32 ## in
# Font sizes and fig sizes for plots
XSMALL_FONT					= 16
SMALL_FONT					= 22
LARGE_FONT					= 30
EXTRA_LARGE_FONT			= 40
BASE_FIG_XSIZE 				= LATEX_PAGEWIDTH
BASE_FIG_YSIZE 				= LATEX_PAGEWIDTH
BASE_AXES_RATIO				= 0.7 # Y to X axes length ratio
BASE_FIG_SIZE  				= 10
BASE_FIG_SIZE  				= LATEX_PAGEWIDTH
# Change these on the fly if you want to increase or decrease image size
FIG_XRATIO 					= 1.
FIG_YRATIO 					= 1.

DEFAULT_PLOT_ORIENTATION = 'horizontal'



mpl.rcParams['legend.frameon'] = False

mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'

mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['axes.labelcolor'] = 'black'
mpl.rcParams['text.color'] = 'black'

mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'

# Set all axes to have ticks facing inwards and include minor ticks
mpl.rcParams["xtick.direction"] = 'in'
mpl.rcParams["ytick.direction"] = 'in'
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["ytick.minor.visible"] = True
mpl.rcParams["xtick.top"] = True
mpl.rcParams["ytick.right"] = True

# Make the x and y ticks bigger                                                    
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = .5
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = .5

# Make the axes linewidths bigger                                                  
mpl.rcParams['axes.linewidth'] = AXIS_BORDER_WIDTH
																					
# Make the x and y ticks bigger                                                    
mpl.rcParams['xtick.labelsize'] = SMALL_FONT
mpl.rcParams['xtick.major.size'] = 4*AXIS_BORDER_WIDTH
mpl.rcParams['xtick.major.width'] = AXIS_BORDER_WIDTH
mpl.rcParams['xtick.minor.size'] = 2*AXIS_BORDER_WIDTH
mpl.rcParams['xtick.minor.width'] = AXIS_BORDER_WIDTH/2
mpl.rcParams['ytick.labelsize'] = SMALL_FONT
mpl.rcParams['ytick.major.size'] = 4*AXIS_BORDER_WIDTH
mpl.rcParams['ytick.major.width'] = AXIS_BORDER_WIDTH
mpl.rcParams['ytick.minor.size'] = 2*AXIS_BORDER_WIDTH
mpl.rcParams['ytick.minor.width'] = AXIS_BORDER_WIDTH/2


mpl.rcParams['font.size'] = LARGE_FONT
mpl.rcParams['axes.labelsize'] = LARGE_FONT
mpl.rcParams['legend.fontsize'] = SMALL_FONT

mpl.rcParams['figure.figsize'] = [5*LATEX_PAGEWIDTH/2,5*LATEX_PAGEWIDTH/2]
mpl.rcParams['figure.dpi'] = 120

mpl.rcParams['figure.subplot.bottom'] = 0
mpl.rcParams['figure.subplot.top'] = 1
mpl.rcParams['figure.subplot.left'] = 0
mpl.rcParams['figure.subplot.right'] = 1

mpl.rcParams['figure.subplot.hspace'] = 0.3
mpl.rcParams['figure.subplot.wspace'] = 0.3

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=LINE_COLORS)
mpl.rcParams["legend.labelspacing"] = 0.35
mpl.rcParams["legend.columnspacing"] = 0.75
# This looks to be the only way to set hatch line widths!
mpl.rcParams['hatch.linewidth'] = BASE_ELINEWIDTH

mpl.rcParams['lines.markersize'] = BASE_MARKERSIZE


# Conversion factors for code to cgs units
UnitLength_in_cm            = 3.085678e21   # 1.0 kpc/h
UnitMass_in_g               = 1.989e43  	# 1.0e10 solar masses/h
UnitMass_in_Msolar			= UnitMass_in_g / 1.989E33
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
pc_to_m						= 3.086E16
Kpc_to_cm					= 3.086E21
km_per_kpc					= 3.086E16
sec_per_Gyr					= 3.16E16
cm_to_um					= 1E4
Ergs_per_joule				= 1E7
grams_to_Msolar				= 5.02785e-34
SOLAR_GAL_RADIUS			= 8 # kpc
HUBBLE 						= 0.702
OMEGA_MATTER 				= 0.272
L_solar 					= 3.828E26 # Watts

AG89_SOLAR_Z = 0.02
AG89_ABUNDANCES = np.array([0.02,0.28,3.26e-3,1.32e-3,8.65e-3,2.22e-3,9.31e-4,1.08e-3,6.44e-4,1.01e-4,1.73e-3])
A09_SOLAR_Z = 0.0142
A09_ABUNDANCES = np.array([0.0142,0.2703,2.53e-3,7.41e-4,6.13e-3,1.34e-3,7.57e-4,7.12e-4,3.31e-4,6.87e-5,1.38e-3])

FIRE_VERSION 				= 2
if FIRE_VERSION == 2:
	# FIRE-2 uses Anders & Grevesse 1989 for Solar
	SOLAR_Z					= AG89_SOLAR_Z
	SOLAR_MASSFRAC			= AG89_ABUNDANCES
else:
	# FIRE-3 uses Asplund+ 2009 for proto-solar
	SOLAR_Z					= A09_SOLAR_Z
	SOLAR_MASSFRAC 			= AG89_ABUNDANCES



ELEMENTS					= ['Z','He','C','N','O','Ne','Mg','Si','S','Ca','Fe']
ELEMENT_NAMES				= ['Metals','Helium','Carbon','Nitrogen','Oxygen','Neon','Magnesium','Silicon','Sulfur','Calcium','Iron']
ATOMIC_MASS					= np.array([1.01, 4.0, 12.01, 14, 15.99, 20.2, 24.305, 28.086, 32.065, 40.078, 55.845])
SIL_ELEM_INDEX				= np.array([4,6,7,10]) # O,Mg,Si,Fe
# number of atoms that make up one formula unit of silicate dust assuming an olivine, pyroxene mixture
# with olivine fraction of 0.32 and Mg fraction of 0.8
SIL_NUM_ATOMS				= np.array([3.631,1.06,1.,0.571]) # O,Mg,Si,Fe
DUST_BULK_DENS				= np.array([3.13,2.25,3.21,7.86]) # silicates, carbonaceous, SiC, metallic iron (g/cm^-3)

DUST_SPECIES				= ['Silicates','Carbon','Iron','O Reservoir','SiC','Iron Inclusions']
DUST_SPECIES_SIL_CARB		= ['Silicates+','Carbon']
DUST_SOURCES				= ['Accretion', 'SNe II', 'AGB', 'SNe Ia']

# Houses labels, limits, and if they should be plotted in log space for possible properties
PROP_INFO  				= {
**dict.fromkeys(['fH2','f_H2'], [r'$f_{\rm H_2}$', 														[0., 1.], 		False]),
**dict.fromkeys(['fHn','f_neutral'], [r'$f_{\rm neutral}$', 											[0., 1.], 		False]),
						'f_cold': [r'$f_{\rm cold}$', 													[0., 1.], 		False],
						'f_warm': [r'$f_{\rm warm}$', 													[0., 1.], 		False],
						 'f_hot': [r'$f_{\rm hot}$', 													[0., 1.], 		False],
					 'f_coronal': [r'$f_{\rm coronal}$', 												[0., 1.], 		False],
					 'f_ionized': [r'$f_{\rm ionized}$', 												[0., 1.], 		False],					 
					  'f_conden': [r'$f_{\rm condensation}$', 											[0., 1.], 		False],
						'fdense': [r'$f_{\rm dense}$', 													[0,1.05], 		False],
						 'CinCO': [r'$f_{\rm C\;in\;CO}$', 												[0,1.05], 		False],
							 'r': ['Radius [kpc]', 														[0.1,20], 		False],
						   'r25': [r'Radius [R$_{25}$]', 												[0.,1.1], 		False],
						 'r_1/2': [r'R$_{1/2}$ [kpc]', 												[0.1,5], 		False],
					 'sigma_gas': [r'$\Sigma_{\rm gas}$ [M$_{\odot}$ pc$^{-2}$]', 						[1E0,1E2], 		True],
					'sigma_star': [r'$\Sigma_{\rm star}$ [M$_{\odot}$ pc$^{-2}$]', 						[1E0,1E2], 		True],
			  'sigma_young_star': [r'$\Sigma_{\rm star}$ (<10 Myr) [M$_{\odot}$ pc$^{-2}$]',			[1E0,1E2], 		True],
				 'sigma_stellar': [r'$\Sigma_{\rm star}$ [M$_{\odot}$ pc$^{-2}$]', 						[1E0,1E2], 		True],
					 'sigma_sfr': [r'$\Sigma_{\rm SFR}$ [M$_{\odot}$ pc$^{-2}$]', 						[1E-3,1E-1], 	True],
						   'sfr': [r'SFR [M$_{\odot}/$yr]', 											[1E-3,5E1], 	True],
					 'sfr_10Myr': [r'SFR$_{\rm 10\;Myr}$ [M$_{\odot}/$yr]', 							[1E-3,5E1], 	True],
					'sfr_100Myr': [r'SFR$_{\rm 100\;Myr}$ [M$_{\odot}/$yr]', 							[1E-3,5E1], 	True],						   						   
			 'sigma_gas_neutral': [r'$\Sigma_{\rm gas,neutral}$ [M$_{\odot}$ pc$^{-2}$]', 				[2E0,1E2], 		True],
			 'sigma_gas_ionized': [r'$\Sigma_{\rm gas,ionized}$ [M$_{\odot}$ pc$^{-2}$]', 				[2E0,1E2], 		True],
				  'sigma_metals': [r'$\Sigma_{\rm metals}$ [M$_{\odot}$ pc$^{-2}$]', 					[1E-2,1E1], 	True],
					'sigma_dust': [r'$\Sigma_{\rm dust}$ [M$_{\odot}$ pc$^{-2}$]', 						[1E-3,1E0], 	True],
					 'sigma_sil': [r'$\Sigma_{\rm sil}$ [M$_{\odot}$ pc$^{-2}$]', 						[1E-3,1E0], 	True],
					'sigma_carb': [r'$\Sigma_{\rm carb}$ [M$_{\odot}$ pc$^{-2}$]', 						[1E-3,1E0], 	True],
					 'sigma_SiC': [r'$\Sigma_{\rm SiC}$ [M$_{\odot}$ pc$^{-2}$]', 						[1E-7,1E-3], 	True],
					'sigma_iron': [r'$\Sigma_{\rm iron}$ [M$_{\odot}$ pc$^{-2}$]', 						[1E-3,1E0], 	True],
					'sigma_ORes': [r'$\Sigma_{\rm O\;Res}$ [M$_{\odot}$ pc$^{-2}$]', 					[1E-3,1E0], 	True],
					'sigma_sil+': [r'$\Sigma_{\rm sil+}$ [M$_{\odot}$ pc$^{-2}$]', 						[1E-3,1E0], 	True],
					  'sigma_H2': [r'$\Sigma_{\rm H_2}$ [M$_{\odot}$ pc$^{-2}$]', 						[1E-1,1E2], 	True],
					'NH_neutral': [r'$N_{\rm H,neutral}$ [cm$^{-2}$]',									[1.1E18,0.9E22],True],
							'NX': [r'$N_{\rm X}$ [cm$^{-2}$]',											[1E16,1E19],	True],
						  'time': ['Time [Gyr]',														[1E-2,1E1],		False],
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
							 'T': [r'T [K]', 															[1.1E1,0.9E7],  True],
							 'Z': [r'Z [Z$_{\odot}$]', 													[1.1E-3,5E0], 	True],
					 'stellar_Z': [r'Z_{\rm star} [Z$_{\odot}$]', 										[1.1E-3,5E0], 	True],
						   'Z_O': ['[O/H]', 															[1.1E-3,5E0], 	True],
				  	   'Z_O_gas': [r'[O/H]$_{\rm gas}$', 												[1.1E-3,5E0],	True],
						   'Z_C': ['[C/H]', 															[1.1E-3,5E0], 	True],
				  	   'Z_C_gas': [r'[C/H]$_{\rm gas}$', 												[1.1E-3,5E0],	True],
						  'Z_Mg': ['[Mg/H]', 															[1.1E-3,5E0], 	True],
				  	  'Z_Mg_gas': [r'[Mg/H]$_{\rm gas}$', 												[1.1E-3,5E0], 	True],
						  'Z_Si': ['[Si/H]', 															[1.1E-3,5E0], 	True],
				  	  'Z_Si_gas': [r'[Si/H]$_{\rm gas}$', 												[1.1E-3,5E0], 	True],
						  'Z_Fe': ['[Fe/H]', 															[1.1E-3,5E0], 	True],
				  	  'Z_Fe_gas': [r'[Fe/H]$_{\rm gas}$', 												[1.1E-3,5E0], 	True],
**dict.fromkeys(['O/H', 'O/H_all','O/H_offset','O/H_gas_ionized_offset'], ['12+log(O/H)', 				[8,9], 	    	False]),
**dict.fromkeys(['O/H_gas','O/H_gas_offset'], [r'12+log(O/H)$_{\rm gas}$', 								[8,9],	 	    False]),
				  	  'O/H_dust': [r'12+log(O/H)$_{\rm dust}$', 										[8,9],	 	    False],
				   'O/H_ionized': [r'12+log(O/H)$_{\rm ionized}$', 										[8,9],	 	    False],
			   'O/H_gas_ionized': [r'12+log(O/H)$_{\rm ionized,gas}$', 									[8,9],	 	    False],
**dict.fromkeys(['C/H', 'C/H_all'], ['12+log(C/H)', 													[8,9], 	    	False]),
				  	   'C/H_gas': [r'12+log(C/H)$_{\rm gas}$', 											[8,9],	 	    False],
				  	  'C/H_dust': [r'12+log(C/H)$_{\rm dust}$', 										[8,9],	 	    False],
**dict.fromkeys(['Mg/H', 'Mg/H_all'], ['12+log(Mg/H)', 													[6.5,8.5], 	    False]),
				  	  'Mg/H_gas': [r'12+log(Mg/H)$_{\rm gas}$', 										[6.5,8.5], 	    False],
					 'Mg/H_dust': [r'12+log(Mg/H)$_{\rm dust}$', 										[6.5,8.5], 	    False],
**dict.fromkeys(['Si/H', 'Si/H_all'], ['12+log(Si/H)', 													[6.5,8.5], 	    False]),
				  	  'Si/H_gas': [r'12+log(Si/H)$_{\rm gas}$', 										[6.5,8.5], 	    False],
					 'Si/H_dust': [r'12+log(Si/H)$_{\rm dust}$', 										[6.5,8.5], 	    False],
**dict.fromkeys(['Fe/H', 'Fe/H_all'], ['12+log(Fe/H)', 													[6.5,8.5], 	    False]),
				  	  'Fe/H_gas': [r'12+log(Fe/H)$_{\rm gas}$', 										[6.5,8.5], 	    False],
					 'Fe/H_dust': [r'12+log(Fe/H)$_{\rm dust}$', 										[6.5,8.5], 	    False],
						   'D/Z': ['D/Z', 																[0,1.05],   	False],
						   'D/H': ['D/H',																[7E-5,2E-2],	True],
				   'D/H_neutral': [r'D/H$_{\rm neutral}$',												[7E-5,2E-2],	True],
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
				   'source_frac': ['Source Mass\nFraction', 												[1E-2,1.05], 	True],
**dict.fromkeys(['spec_frac','spec_frac_Si/C'], ['Species Mass\nFraction', 								[0,1.05], 		False]),
						  'Si/C': ['Sil-to-C Ratio', 													[0,10], 		False],
					 'mass_frac': ['Mass Fraction',														[0,1.05],		False],
					 'wind_rate': [r'Cont. Mass-Loss $\dot{M}_{\rm W}/M_{\star}$ [Gyr$^{-1}$]',			[3E-4,2E2],		True],
					  'wind_vel': [r'Mass-Loss Velocity $v_{\rm w,inj}$ [km s$^{-1}$]',					[2E1,5E3],		True],
					    'wind_E': [r'Inst. Energy Inj. $E_{\rm inj}}/M_{\star}$ [erg $s^{-1}\;M_{\star}^{-1}$]',[1E-5,1E6],	True],
					'cum_wind_E': [r'Cum. Energy $E_{\rm inj,cum}}/M_{\star}$ [erg $M_{\star}^{-1}$]',	[6E17,5E18],	True],
**dict.fromkeys(['lambda','wavelength'], [r'$\lambda \, [\mu m]$', 										[6E-2,1E3], 	True]),
			   			   'SED': [r'$\lambda L_{\lambda} \,[L_{\odot}]$',								[1E8,2E11],		True],
					'grain_size': [r'a ($\mu m$)',														[7E-4,10],		True],
**dict.fromkeys(['dn/da','sil_dn/da','carb_dn/da','SiC_dn/da','iron_dn/da'],
								  [r'$\frac{\partial n}{\partial a}$',									[1E-10,1E4],	True]),
**dict.fromkeys(['dm/da','sil_dm/da','carb_dm/da','SiC_dm/da','iron_dm/da'],
								  [r'$a^4\frac{\partial n}{\partial a}$',								[1E-10,1E-4],	True])
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