import numpy as np
import pandas as pd
from astropy.io import ascii
from gizmo_library import config
import os

OBS_DIR = os.path.join(os.path.dirname(__file__), 'data/')

def galaxy_integrated_DZ(paper):
	"""
	Read literature data into DataFrame

	Parameters
	----------
	paper : str
		R14, DV19, or PH20

	Returns
	-------
	df : pandas.DataFrame
		Loaded data
	"""
	if paper == 'R14':
		# load Remy-Ruyer+14
		df_r14 = pd.read_csv(OBS_DIR+"remyruyer/Remy-Ruyer_2014.csv")
		df_r14['metal'] = df_r14['12+log(O/H)']
		# Convert from O/H to solar using Eq 9 from Chiang+21 which uses Loddars+03 solar abundances
		df_r14['metal_z'] = 10**(df_r14['metal'] - 12.0) * \
			16.0 / 1.008 / 0.51 / 1.36
		df_r14['metal_z_solar'] = df_r14['metal_z'] / config.A09_SOLAR_Z
		df_r14['gas'] = df_r14['MU_GAL'] * (df_r14['MHI_MSUN'] + df_r14['MH2_Z_MSUN'])
		df_r14['fh2'] = df_r14['MH2_Z_MSUN'] / (df_r14['MHI_MSUN'] + df_r14['MH2_Z_MSUN'])
		df_r14['dtm'] = df_r14['MDUST_MSUN'] / df_r14['metal_z'] / df_r14['gas']
		return df_r14

	elif paper == 'DV19':
		# load De Vis+19
		path_dp = OBS_DIR+'dustpedia/'
		df_d19 = pd.read_csv(path_dp + 'dustpedia_cigale_results_final_version.csv')
		df_d19 = df_d19.rename(columns={'id': 'name'})
		# dp metal
		df_d19_temp = pd.read_csv(path_dp + 'DP_metallicities_global.csv')
		df_d19 = df_d19.merge(df_d19_temp, on='name')
		# dp hi
		df_d19_temp = pd.read_csv(path_dp + 'DP_HI.csv')
		df_d19 = df_d19.merge(df_d19_temp, on='name')
		# dp h2
		df_d19_temp = pd.read_excel(path_dp + 'DP_H2.xlsx')
		df_d19_temp = df_d19_temp.rename(columns={'Name': 'name'})
		df_d19 = df_d19.merge(df_d19_temp, on='name')
		# renames
		del df_d19_temp
		df_d19 = \
			df_d19.rename(columns={'SFR__Msol_per_yr': 'sfr',
								   'Mstar__Msol': 'star',
								   'MHI': 'hi',
								   '12+logOH_PG16_S': 'metal'})
		df_d19['h2'] = df_d19['MH2-r25'] * 1.36
		df_d19['gas'] = df_d19['hi'] * 1.36 + df_d19['h2']
		df_d19['metal_z'] = 10**(df_d19['metal'] - 12.0) * \
			16.0 / 1.008 / 0.51 / 1.36
		df_d19['dtm'] = df_d19['Mdust__Msol'] / (df_d19['metal_z'] * df_d19['gas'])
		df_d19['metal_z_solar'] = df_d19['metal_z'] / config.A09_SOLAR_Z

		return df_d19

	elif paper == 'PH20':
		# load Howk's review (Peroux+20)
		path_dp = OBS_DIR+'howk/tableSupplement_howk.csv'
		df_p20 = pd.read_csv(path_dp, comment='#')
		df_p20['metal'] = 8.69 + df_p20['[M/H]']
		df_p20['metal_z'] = 10**(df_p20['metal'] - 12.0) * \
			16.0 / 1.008 / 0.51 / 1.36
		df_p20['metal_z_solar'] = df_p20['metal_z'] / config.A09_SOLAR_Z
		df_p20['dtm'] = 10**df_p20['log_DTM']
		df_p20['limit'] = df_p20['log_DTM'] <= -1.469
		return df_p20




def Dwek_2014_M31_dust_dens_vs_radius():
	"""
	Gives the dust surface density (M_sun pc^-2) vs radius (kpc) from galactic center for M31 (Andromeda) determined by Dwek et al. (2014)
	"""
	radius = np.array([3.2487e-1,1.0161e+0,1.7033e+0,2.3469e+0,3.0348e+0,3.6811e+0,4.3691e+0,5.0136e+0,5.7446e+0,6.3579e+0,7.0488e+0, \
			 7.7346e+0,8.4225e+0,9.1109e+0,9.7594e+0,1.0408e+1,1.1098e+1,1.1791e+1,1.2446e+1,1.3140e+1,1.3832e+1,1.4484e+1,1.5176e+1, \
			 1.5873e+1,1.6527e+1,1.7222e+1,1.7873e+1,1.8568e+1,1.9262e+1,1.9917e+1,2.0610e+1,2.1304e+1,2.1959e+1,2.2656e+1,2.3349e+1, \
			 2.4003e+1,2.4697e+1])

	surface_dens = np.array([9.1852e+3,9.1114e+3,1.1216e+4,1.6233e+4,1.9227e+4,2.4036e+4,2.8250e+4,3.9040e+4,4.0559e+4,3.3438e+4,3.3685e+4, \
				   4.4789e+4,5.3049e+4,6.1396e+4,6.7845e+4,7.6727e+4,7.7891e+4,7.2086e+4,5.5874e+4,4.8244e+4,4.4648e+4,4.1964e+4,3.8538e+4, \
				   2.8963e+4,2.3695e+4,1.9087e+4,1.8078e+4,1.5136e+4,1.2770e+4,1.0130e+4,9.0899e+3,7.7883e+3,5.9904e+3,4.3654e+3,3.8873e+3, \
				   3.1557e+3,2.7038e+3])

	# Covert to correct units a
	kpc_to_pc = 1E3
	surface_dens /= (kpc_to_pc * kpc_to_pc)

	return radius, surface_dens



def Menard_2010_dust_dens_vs_radius(sigma_dust_scale, r_scale):
	"""
	Gives data for observed Menard (2010) dust density (M_sun pc^-2) vs radius (kpc) relation (Sigma_dust ~ r^-0.8) which is observed for r > 25 kpc.

	Parameters
	----------
	sigma_dust_scale : double
		Dust surface density value to scale relation to
	r_scale : double
		Radius value for given sigma_dust

	Returns
	------
	radius : array
		Array of radius values-
	sigma_dust : array
		Array of dust surface density values for given radii
	"""	

	r_vals = np.linspace(1, 500, num=200) # kpc
	r_indx = np.argmin(np.abs(r_vals-r_scale))
	sigma_dust = sigma_dust_scale * np.power(r_vals/ r_vals[r_indx],-0.8)

	return r_vals, sigma_dust



def Jenkins_Savage_2009_WNM_Depl(elem):
	"""
	Gives the depletion for Mg, Si, or Fe in the WNM based on comparison of Jenkins (2009)
	with Savage & Sembach (1996a). Note C depletions are reduced by a factor of 2 in line with
	Sofia+2011 and Parvathi+2012.
	"""

	# Typical nH for WNM
	nH_dens = 0.5 # cm^-3
	# F_star value for "warm disk" environment from Savage & Sembach (1996)
	F_star = 0.12

	amu_H = 1.008
	elems = ['C','N','O','Mg','Si','P','Cl','Ti','Cr','Mn','Fe','Ni','Cu','Zn','Ge','Kr']
	# All values are for C,N,O,Mg,Si,P,Cl,Ti,Cr,Mn,Fe,Ni,Cu,Zn,Ge,Kr respectively
	amu = np.array([12.01,14.01,16,24.3,28.085,30.973762,35.45,47.867,51.9961,54.938044,55.845,58.6934,63.546,65.38,72.63,83.798])
	# 12+log(X/H)_solar values
	solar = np.array([8.46,7.9,8.76,7.62,7.61,5.54,5.33,5,5.72,5.58,7.54,6.29,4.34,4.7,3.7,3.36])
	solar_Mx_MH = np.power(10,solar-12)*amu/amu_H
	total_Z = np.sum(solar_Mx_MH)
	# Fit parameters for depletions
	corr_factor = np.log10(2)
	Ax = np.array([-0.101,0,-0.225,-0.997,-1.136,-0.945,-1.242,-2.048,-1.447,-0.857,-1.285,-1.49,-0.71,-0.61,-0.615,-0.166])
	Bx = np.array([-0.193-corr_factor,-0.109,-0.145,-0.8,-0.57,-0.166,-0.314,-1.957,-1.508,-1.354,-1.513,-1.829,-1.102,-0.279,-0.725,-0.332])
	zx = np.array([0.803,0.55,0.598,0.531,0.305,0.488,0.609,0.43,0.47,0.52,0.437,0.599,0.711,0.555,0.69,0.684])

	# Depletion factor of element x at a given F_star
	x_depl = Bx + Ax*(F_star - zx)

	obs_Mx_MH = np.power(10,solar+x_depl-12)*amu/amu_H

	if elem == 'Z':
		# Need error bars since C depletion has few observations 
		# Assume all C-20% in CO in dust or no C in dust for error bars
		# Set 'average' value
		min_val=0.2; max_val=1.; avg_val=(min_val+max_val)/2.
		obs_Mx_MH[0] = np.power(10,solar[0]+np.log10(avg_val)-12)*amu[0]/amu_H
		WNM_depl = np.sum(solar_Mx_MH - obs_Mx_MH)/total_Z	
		obs_Mx_MH[0] = np.power(10,solar[0]+np.log10(max_val)-12)*amu[0]/amu_H
		WNM_error = np.abs(np.sum(solar_Mx_MH - obs_Mx_MH)/total_Z - WNM_depl)
	elif elem in elems:
		index = elems.index(elem)
		elem_to_H = solar_Mx_MH[index] - obs_Mx_MH[index]
		WNM_depl = elem_to_H/solar_Mx_MH[index]
		WNM_error = 0.
	else:
		print("Given element is not included in Jenkins (2009)\n")
		return None, None


	return nH_dens, 1.-WNM_depl, WNM_error


def Jenkins_2009_Elem_Depl(elem,density='NH'):
	"""
	Gives element depletion data vs <nH> from sightlines in Jenkins (2009).

	Parameters
	----------
	elem : string
		Which element to get depletions for
	phy_dens : boolean
		If True use physical nH conversion from Zhukovska+16. If False use average sight line nH.
	density : string
		Desired density property which include NH (raw surface number density), <nH> (average sight line density),
		nH (physical density conversion from Zhukovska+16)

	Returns
	------
	elem_depl : list
		Element depletions for all sight lines
	elem_err : list
		Upper and lower errors for each depletion
	density : list
		Desired density property for each sight line
	"""

	if elem in config.ELEMENTS:
		elem_name = config.ELEMENT_NAMES[config.ELEMENTS.index(elem)]
	else:
		print("Element %s is not supported in Jenkins_2009_Elem_Depl()"%elem)
		return None,None,None

	if density not in ['NH','<nH>','nH']:
		print("Density parameter %s is not supported in Jenkins_2009_Elem_Depl(). Must be NH,<nH>, or nH"%density)
		return None,None,None

	# Get NElem data for the specified element and reference abundance
	NElem_data = ascii.read(OBS_DIR+'Jenkins09/elem_depletions.txt')
	fit_data = ascii.read(OBS_DIR+'Jenkins09/element_depletion_parameters.txt')
	ref_abund = fit_data['X/H'][fit_data['El']==elem].data[0] # in 12+log(X/H) units
	err_ref_abund = fit_data['e_X/H'][fit_data['El']==elem].data[0]

	# Fill in all the missing data for needed fields with NaNs, +/-infinity, or 0
	NElem_data['F*'].fill_value = np.nan; NElem_data['logNEl'].fill_value = np.nan; NElem_data['VComp'].fill_value = 0;
	NElem_data['E_logNEl'].fill_value = np.inf; NElem_data['e_logNEl'].fill_value = -np.inf;
	NElem_data['Res'].fill_value = 0;
	NElem_data = NElem_data.filled()

	# Only use sight lines with all the data we want for the specified element only
	elem_mask = (NElem_data['El']==elem_name) & (~np.isnan(NElem_data['logNEl']))

	# Some elements may need a correction factor for NEl, notably C should be reduced by half
	if elem=='C':
		corr_factor = -np.log10(2)
	else: corr_factor = 0.

	NEl = NElem_data['logNEl'][elem_mask].data+corr_factor
	# Get upper/lower limits and convert to upper/lower errors
	lower_NEl = NEl-(NElem_data['e_logNEl'][elem_mask].data+corr_factor); upper_NEl = (NElem_data['E_logNEl'][elem_mask].data+corr_factor)-NEl;
	F_star = NElem_data['F*'][elem_mask].data
	NEl = np.power(10,NEl)
	# Error propagation: 10^NEl for lower and upper limits/errors
	lower_NEl = np.log(10.)*NEl*lower_NEl; upper_NEl = np.log(10.)*NEl*upper_NEl;

	# Get IDs to match with sight line NH data
	Elem_ID1 =  NElem_data['Name'][elem_mask].data; Elem_ID2 = NElem_data['VComp'][elem_mask].data;
	Elem_ID = np.char.add(Elem_ID1,Elem_ID2.astype(str))

	# Get NHI and NH2 data
	NH_data = ascii.read(OBS_DIR+'Jenkins09/sightline_NH.txt')
	NH_data['logNHOb'].fill_value = np.nan; NH_data['VComp'].fill_value = 0;
	NH_data['e_logNHOb'].fill_value = -np.inf; NH_data['E_logNHOb'].fill_value = np.inf;
	NH_data = NH_data.filled()
	H_ID1 = NH_data['Name'].data; H_ID2 = NH_data['VComp'].data;
	H_ID = np.char.add(H_ID1,H_ID2.astype(str))
	NH_mask = np.in1d(H_ID,Elem_ID)

	# Dispose of any sight lines with only lower or upper limits on NH
	NH_mask = NH_mask & (~np.isnan(NH_data['logNHOb'].data))

	# Now need to remask element data to avoid these no NH sight lines
	H_ID = H_ID[NH_mask]
	elem_mask = np.in1d(Elem_ID, H_ID)
	NEl = NEl[elem_mask]; lower_NEl = lower_NEl[elem_mask]; upper_NEl = upper_NEl[elem_mask];
	F_star = F_star[elem_mask]

	# Get NH_neutral data for each sight line
	NH_n = NH_data['logNHOb'].data[NH_mask];
	lower_NH_n = NH_n-NH_data['e_logNHOb'].data[NH_mask]; upper_NH_n = NH_data['E_logNHOb'].data[NH_mask]-NH_n;
	NH_n=np.power(10,NH_n)
	lower_NH_n = np.log(10.)*NH_n*lower_NH_n; upper_NH_n = np.log(10.)*NH_n*upper_NH_n;
	# Deal with large errors greater than expected value
	mask = lower_NH_n>NH_n
	lower_NH_n[mask] = -np.inf

	# Convert reference abundance to ppm
	ref_abund = np.power(10,ref_abund-12)
	# Error propagation: 10^(ref-12) for uncertainty
	err_ref_abund = np.log(10.)*ref_abund*err_ref_abund;
	# Calculate linear depletion
	elem_depl = (NEl/NH_n)/ref_abund
	# Error propagation: add in percentile quadrature
	lower_elem_depl = elem_depl*np.sqrt(np.square(lower_NH_n/NH_n)+np.square(lower_NEl/NEl)+np.square(err_ref_abund/ref_abund))
	upper_elem_depl = elem_depl*np.sqrt(np.square(upper_NH_n/NH_n)+np.square(upper_NEl/NEl)+np.square(err_ref_abund/ref_abund))
	err_depl = np.stack((lower_elem_depl, upper_elem_depl))


	if density == 'NH':
		density_vals = NH_n
	else:
		# Now get density values from F* depletion parameter
		avg_nH = np.power(10,(F_star-0.772)/0.461)
		if density == 'nH':
			phys_nH = 147.234*np.power(avg_nH,1.054)
			# This conversion is only valid for a certain range of densities
			obs_range = np.array([10, 1E3])
			density_vals = phys_nH
		elif density == '<nH>':
			density_vals = avg_nH

	return elem_depl, err_depl, density_vals



def RomanDuval_2021_LMC_Elem_Depl(elem):

	if elem not in config.ELEMENTS:
		print("Element %s is not supported in RomanDuval_2021_LMC_Elem_Depl()" % elem)
		return None, None, None

	depl_data = ascii.read(OBS_DIR + 'RomanDuval21/LMC_depletions.txt')
	# First get all the elements since we want to sort by a given element
	elements = depl_data['element'].data
	elements = np.array([elem.replace('I', '') for elem in elements])
	elem_mask = elements == elem

	if len(elements[elem_mask])==0:
		print("RomanDuval_2021_LMC_Elem_Depl does not have data for %s depletions"%elem)
		return None,None,None

	NH_vals = np.power(10, depl_data['logN(H)'].data)[elem_mask]
	depletions = depl_data['d(X)'].data[elem_mask]
	depletion_error = depl_data['e_d(X)'].data[elem_mask]
	depletion_limits = np.array([depletions - depletion_error, depletions + depletion_error])

	depletions = np.power(10, depletions)
	depletion_limits = np.power(10, depletion_limits)

	# Some of these observations are solely upper or lower limits
	depletion_is_limit = depl_data['l_d(X)'].data[elem_mask]
	upper_limit = depletion_is_limit == '<'
	lower_limit = depletion_is_limit == '>'
	depletion_limits[1, lower_limit] = np.inf
	depletion_limits[0, lower_limit] = depletions[lower_limit]
	depletion_limits[0, upper_limit] = np.inf
	depletion_limits[1, upper_limit] = depletions[upper_limit]

	return depletions, depletion_limits, NH_vals


def Jenkins_Wallerstein_2017_SMC_Elem_Depl(elem):

	depl_data = ascii.read(OBS_DIR + 'JenkinsWallerstein17/SMC.csv')

	elems = ['Mg', 'Si', 'Fe']
	if elem not in elems:
		print("Jenkins_Wallerstein_2017_SMC_Elem_Depl does not have data for %s depletions" % elem)
		return None, None, None

	depletions = np.array(depl_data['d(' + elem + ')'].data)
	valid_mask = depletions != 0
	NH_vals = np.array(depl_data['log(NH)'])[valid_mask]
	error_lower = np.array(depl_data['d(' + elem + ')_l'])[valid_mask]
	error_upper = np.array(depl_data['d(' + elem + ')_u'])[valid_mask]
	depletions = depletions[valid_mask]

	depletion_limits = np.array([depletions - error_lower, depletions + error_upper])
	depletions = np.power(10, depletions)
	depletion_limits = np.power(10, depletion_limits)
	NH_vals = np.power(10, NH_vals)

	return depletions, depletion_limits, NH_vals

def Parvathi_2012_C_Depl(solar_abund='max', density='<nH>'):
	"""
	Gives C depletion data vs <nH> from sightlines in Parvathi+ (2012).

	Parameters
	----------
	solar_abund : string
		What to use as the assumed abundace of C 'max'=maximum of data set, 'solar'=Lodders03, 'young_star'=young star abundance from Sofia & Meyer (2001)

	Returns
	------
	C_depl : np.array
		C depletions for all sightlines
	C_err : array
		Errors for each C depeltions
	nH : array
		Average sight line density (cm^-3) for each sightline
	"""	

	solar_C = 228
	young_star_C = 358

	# Number abundances of gas-phase C along with their errors and average sightline densities
	gas_phase_C = np.array([382,202,116,365,275,112,173,138,193,290,85,464,131,321,317,93,92,99,98,215,69],dtype=float)
	C_error     = np.array([56, 24, 102, 94, 96, 19, 34, 24, 31, 58, 28, 57, 27, 70, 46, 32, 38, 36, 23, 54, 21],dtype=float)
	nH          = np.power(10,np.array([0.41,-0.73,-1.15,-0.47,-0.69,-0.41,0.08,-0.27,-0.13,-0.90,0.55,-0.03,0.04,-0.70,-0.92,0.53,0.61,0.66,1.11,-0.28,0.40]))
	NH			= np.power(10,np.array([21.46,21.16,21.02,21.34,20.66,21.44,21.41,21.48,21.15,21.00,21.44,21.25,21.32,21.33,21.19,21.36,21.68,21.73,21.77,21.47,21.68]))

	if solar_abund=='max':
		i_max = np.argmax(gas_phase_C)
		# Propagate error
		C_error = np.sqrt(np.power(C_error/gas_phase_C,2)+np.power(C_error[i_max]/gas_phase_C[i_max],2))
		C_depl = gas_phase_C/np.max(gas_phase_C)
		C_error *= C_depl
	elif solar_abund=='solar':
		C_depl = gas_phase_C/solar_C
		C_error /= solar_C
	elif solar_abund=='young_star':
		C_depl = gas_phase_C/young_star_C
		C_error /= young_star_C
	else:
		print("%s is not a valid argument for Parvathi_2012_C_Depl()"%solar_abund)
		return None,None,None

	if density == 'NH':
		density_vals = NH
	elif density=='<nH>':
		density_vals = nH

	return C_depl, C_error, density_vals






def Jenkins_2009_DZ_vs_dens(phys_dens=False, elem='Z', C_corr=True):
	"""
	Gives the total D/Z or individual element D/Z vs average sight light density from Jenkins (2009). Note that
	for C the depletion is increased by factor of 2 (if C_corr=True) due to findings in Sofia et al. (2011) and Parvathi et al. (2012).
	Can also output physical density instead using results from Zhukovska (2016).
	"""

	# This is the range the relation was observed, used when plotting to contrast with extrapolated range
	obs_range = np.array([np.power(10,-1.7), np.power(10,0.8)])
	# C data is especially scarce so limit the relation to only the observed range
	if elem == 'C':
		obs_range = np.array([np.power(10,-0.9), np.power(10,0.8)])

	avg_nH = np.logspace(-2,3,num=200)
	# Get physical nH value with conversion from Zhukovska (2016).
	# This may not be accurate so use with caution.
	if phys_dens:
		phys_nH = 147.234*np.power(avg_nH,1.054)
		# This conversion is only valid for a certain range of densities
		obs_range = np.array([10, 1E3])
		if elem == 'C':
			obs_range = np.array([17, 1E3])
	F_star = 0.772 + 0.461*np.log10(avg_nH) 

	amu_H = 1.008
	elems = ['C','N','O','Mg','Si','P','Cl','Ti','Cr','Mn','Fe','Ni','Cu','Zn','Ge','Kr']
	# All values are for C,N,O,Mg,Si,P,Cl,Ti,Cr,Mn,Fe,Ni,Cu,Zn,Ge,Kr respectively
	amu = np.array([12.01,14.01,16,24.3,28.085,30.973762,35.45,47.867,51.9961,54.938044,55.845,58.6934,63.546,65.38,72.63,83.798])
	# 12+log(X/H)_solar values
	solar = np.array([8.46,7.9,8.76,7.62,7.61,5.54,5.33,5,5.72,5.58,7.54,6.29,4.34,4.7,3.7,3.36])
	solar_Mx_MH = np.power(10,solar-12)*amu/amu_H
	total_Z = np.sum(solar_Mx_MH)
	# Fit parameters for depletions
	factor = 0.
	if C_corr:
		factor = np.log10(2)
	Ax = np.array([-0.101,0,-0.225,-0.997,-1.136,-0.945,-1.242,-2.048,-1.447,-0.857,-1.285,-1.49,-0.71,-0.61,-0.615,-0.166])
	Bx = np.array([-0.193-factor,-0.109,-0.145,-0.8,-0.57,-0.166,-0.314,-1.957,-1.508,-1.354,-1.513,-1.829,-1.102,-0.279,-0.725,-0.332])
	zx = np.array([0.803,0.55,0.598,0.531,0.305,0.488,0.609,0.43,0.47,0.52,0.437,0.599,0.711,0.555,0.69,0.684])

	# Depletion factor of element x at a given F_star
	x_depl = Bx + Ax*(F_star.reshape([-1,1]) - zx)

	# Decrease all converted depletions by 0.2 dex to account for contribution from WNM along line of sight
	if phys_dens and (elem in ['Mg','Si','Fe']):
		x_depl -= 0.2

	obs_Mx_MH = np.power(10,solar+x_depl-12)*amu/amu_H

	if elem == 'Z':
		dust_to_H = np.sum(solar_Mx_MH - obs_Mx_MH,axis=1)
		DZ_vals = dust_to_H/total_Z
	elif elem in elems:
		index = elems.index(elem)
		elem_to_H = solar_Mx_MH[index] - obs_Mx_MH[:,index]
		DZ_vals = elem_to_H/solar_Mx_MH[index]
	else:
		print("Given element is not included in Jenkins (2009)\n")
		return None, None

	
	if phys_dens:
		in_range = np.where(np.logical_and(phys_nH>=obs_range[0], phys_nH<=obs_range[1]))
		return phys_nH[in_range], DZ_vals[in_range]
	else:
		in_range = np.where(np.logical_and(avg_nH>=obs_range[0], avg_nH<=obs_range[1]))
		return avg_nH[in_range], DZ_vals[in_range]



# Data from Chiang+2021. Includes D/Z vs various spatially-resolved gas properties. Note this data main observes H2 dominated regions
def Chiang21_DZ_vs_param(param, bin_data=True, CO_opt='B13', bin_nums=10, log=True, goodSNR=True,aggregate_data=False):
	file_name = OBS_DIR+'Chiang21/Chiang+20_dat_v0.1.'+CO_opt+'.csv'
	data = np.genfromtxt(file_name,names=True,delimiter=',',dtype=None,encoding=None)
	DZ = np.power(10,data['dtm'])
	if param == 'sigma_gas' or param == 'sigma_gas_neutral':
		vals = np.power(10,data['gas'])
	elif param == 'sigma_H2':
		vals = np.power(10,data['h2'])
	elif param in ['sigma_stellar','sigma_star']:
		vals = np.power(10,data['star'])
	elif param == 'sigma_sfr':
		vals = np.power(10,data['sfr'])
	elif param == 'fH2':
		vals = data['fh2']
	elif param == 'sigma_Z':
		vals = data['metal_z']
	elif 'O/H' in param or param == 'Z':
		vals = data['metal']
		if param == 'Z':
			O_H = data['metal']
			# Convert from 12+log{0/H} to Z given Eq. 9 in Chiang2021
			vals = 1./0.51*config.ATOMIC_MASS[4]/(1.36*config.ATOMIC_MASS[0])*np.power(10,O_H-12.)
			vals /= config.SOLAR_Z
	elif param == 'sigma_dust':
		vals = data['dust']
	elif param in ['r','r25','r*']:
		# kpc distance to galaxy
		gal_distance = {'IC342': 2.29E3,'M101': 6.96E3,'M31': 0.79E3,'M33': 0.92E3,'NGC628': 9.77E3}
		if param == 'r':
			arcsec_to_rad = 4.848E-6
			vals = data['radius_arcsec']*arcsec_to_rad
		elif param == 'r25':
			vals = data['radius_r25']
		elif param == 'r*':
			vals = data['radius_r25']*0.2 # Assume r* ~ 0.2 * r25
	else:
		print("%s is not a valid param for Chiang_20_DZ_vs_param"%param)
		return

	gal = data['gal']
	IDs = np.unique(data['gal'])


	SNR = data['GOODSNR']

	# Check whether to use all data or only that with good SNR
	if goodSNR:
		mask = SNR == 1
		DZ = DZ[mask]
		gal = gal[mask]
		vals = vals[mask]

	data = dict()
	if aggregate_data: data['all'] = [vals,DZ]
	else:
		for gal_name in IDs:
			gal_vals = vals[gal==gal_name]
			if param == 'r':
				gal_vals *= gal_distance[gal_name]
			DZ_vals = DZ[gal==gal_name]
			if bin_data:
				mean_DZ = np.zeros(bin_nums - 1)
				# 16th and 84th percentiles
				std_DZ = np.zeros([bin_nums - 1,2])
				if log:
					val_bins = np.logspace(np.log10(np.min(gal_vals)), np.log10(np.max(gal_vals)), num=bin_nums)
				else:
					val_bins = np.linspace(np.min(gal_vals), np.max(gal_vals), num=bin_nums)
				param_vals = (val_bins[1:] + val_bins[:-1]) / 2.
				digitized = np.digitize(gal_vals,val_bins)
				for j in range(1,len(val_bins)):
					if len(gal_vals[digitized==j])==0:
						mean_DZ[j-1] = np.nan
						std_DZ[j-1,0] = np.nan; std_DZ[j-1,1] = np.nan;
						continue
					else:
						values = DZ_vals[digitized == j]
						mean_DZ[j-1] = np.mean(values)
						std_DZ[j-1] = np.percentile(values, [16,84])
				mask = np.logical_not(np.isnan(mean_DZ))
				data[gal_name] = [param_vals[mask], mean_DZ[mask], std_DZ[mask]]

			else:
				data[gal_name] = [gal_vals,DZ_vals]

	return data


# Similar to data from Chiang+21 but expanded to more galaxies and with varying spatial resolution. Currently unplublished.
def Chiang22_DZ_vs_param(param, bin_data=True, bin_nums=10, log=True, goodSNR=True, only_PG16S=True,aggregate_data=False):
	# Use just 2kpc resolution data set since this includes almost all galaxies in the sample with only a handful at
	# the other resolutions
	file_name = OBS_DIR+'Chiang22_unpub/DataAssembly_2kpc.csv'
	data = pd.read_csv(file_name)

	gal_names = data['objname']
	# We will stick to the B13 values which use Bolatto+13 alpha_CO factor to determine H2 amounts
	DZ = data['D/M (B13)']
	if param == 'sigma_gas' or param == 'sigma_gas_neutral':
		vals = data['SigmaGas (B13)']
	elif param == 'sigma_H2':
		vals = data['SigmaH2 (B13)']
	elif param in ['sigma_stellar','sigma_star']:
		vals = data['SigmaMstar']
	elif param == 'sigma_sfr':
		vals = data['SigmaSFR']
	elif param == 'fH2':
		vals = data['SigmaH2 (B13)']/data['SigmaGas (B13)']
	elif param == 'sigma_Z':
		vals = data['z']*data['SigmaGas (B13)']
	elif param == 'Z':
		vals = data['z']/config.SOLAR_Z
	elif 'O/H' in param:
		vals = data['12+log(O/H) (All)']
	elif param == 'sigma_dust':
		vals = data['dust']
	elif param == 'r':
		vals = data['Rg_kpc']
	elif param == 'r25':
		vals = data['Rg/R25']
	else:
		print("%s is not a valid param for Chiang_20_DZ_vs_param"%param)
		return

	# There is a dust mask for 1 sigma and 3 sigma values. Lets just use the 3 sigma values
	SNR = data['dust_mask_3sigma']
	#SNR = data['dust_mask_1sigma']

	# Check whether to use all data or only that with good SNR
	mask = np.ones(len(vals), dtype=bool)
	if goodSNR:
		mask = (mask) & (SNR == 1)
	# Not all galaxies have directly measured O/H, some are inferred with a mass-metallicity relation and a painted on radial distribution
	# If you only want directly measured O/H then the PG16S O/H measurements are the ones you want
	if only_PG16S:
		PG16S = data['12+log(O/H) (PG16S)']
		mask = (mask) & (~np.isnan(PG16S))

	DZ = DZ[mask]
	gal_names = gal_names[mask]
	vals = vals[mask]

	# After masking get unique galaxies in final data set
	IDs = np.unique(gal_names)

	data = dict()
	if aggregate_data: data['all'] = [vals,DZ]
	else:
		for gal_name in IDs:
			gal_vals = vals[gal_names==gal_name]
			DZ_vals = DZ[gal_names==gal_name]
			if bin_data:
				mean_DZ = np.zeros(bin_nums - 1)
				# 16th and 84th percentiles
				std_DZ = np.zeros([bin_nums - 1,2])
				if log:
					val_bins = np.logspace(np.log10(np.min(gal_vals+config.EPSILON)), np.log10(np.max(gal_vals)), num=bin_nums)
				else:
					val_bins = np.linspace(np.min(gal_vals), np.max(gal_vals), num=bin_nums)
				param_vals = (val_bins[1:] + val_bins[:-1]) / 2.
				digitized = np.digitize(gal_vals,val_bins)
				for j in range(1,len(val_bins)):
					if len(gal_vals[digitized==j])==0:
						mean_DZ[j-1] = np.nan
						std_DZ[j-1,0] = np.nan; std_DZ[j-1,1] = np.nan;
						continue
					else:
						values = DZ_vals[digitized == j]
						mean_DZ[j-1] = np.mean(values)
						std_DZ[j-1] = np.percentile(values, [16,84])
				mask = np.logical_not(np.isnan(mean_DZ))
				data[gal_name] = [param_vals[mask], mean_DZ[mask], std_DZ[mask]]

			else:
				data[gal_name] = [gal_vals,DZ_vals]

	return data