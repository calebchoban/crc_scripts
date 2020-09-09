import matplotlib as mpl
import matplotlib.pyplot as plt

# Allow plotting when run on command line
plt.switch_backend('agg')
# Set style of plots
plt.style.use('seaborn-talk')
# Set personal color cycle
Line_Colors = ["xkcd:blue", "xkcd:red", "xkcd:green", "xkcd:orange", "xkcd:violet", "xkcd:teal", "xkcd:brown"]
Line_Styles = ['-','--',':','-.']
Line_Widths = [0.5,1.0,1.5,2.0,2.5,3.0]

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=Line_Colors)


# Conversion factors for code to cgs units
UnitLength_in_cm            = 3.085678e21   # 1.0 kpc/h
UnitMass_in_g               = 1.989e43  	# 1.0e10 solar masses/h
UnitMass_in_Msolar			= UnitMass_in_g / 1.989E33
UnitVelocity_in_cm_per_s    = 1.0e5   	    # 1 km/sec
UnitTime_in_s 				= UnitLength_in_cm / UnitVelocity_in_cm_per_s # 0.978 Gyr/h
UnitTime_in_Gyr 			= UnitTime_in_s /1e9/365./24./3600.
UnitEnergy_per_Mass 		= np.power(UnitLength_in_cm, 2) / np.power(UnitTime_in_s, 2)
UnitDensity_in_cgs 			= UnitMass_in_g / np.power(UnitLength_in_cm, 3)
BoltzMann_ergs              = 1.3806e-16
H_MASS 						= 1.67E-24 # grams
SOLAR_Z						= 0.02
EPSILON						= 1E-7 # small number to avoid zeros


# Small and large fonts for plots
SMALL_FONT					= 20
LARGE_FONT					= 26


