import numpy as np
import utils

SOLAR_Z= 0.02

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
	kpc_to_pc = 1E3;
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

	return r_val, sigma_dust


def Jenkins_2009_DZ_vs_dens(phys_dens=False, elem='Z'):
	"""
	Gives the total D/Z or individual element D/Z vs average sight light density from Jenkins (2009). Can also output physical density instead
	using results from Zhukovska (2016).
	"""

	avg_nH = np.logspace(-2,3,num=50)
	# Get physical nH value with conversion from Zhukovska (2016).
	# This may not be accurate so use with caution.
	phys_nH = 147.234*np.power(avg_nH,1.054)
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
	Ax = np.array([-0.101,0,-0.225,-0.997,-1.136,-0.945,-1.242,-2.048,-1.447,-0.857,-1.285,-1.49,-0.71,-0.61,-0.615,-0.166])
	Bx = np.array([-0.193,-0.109,-0.145,-0.8,-0.57,-0.166,-0.314,-1.957,-1.508,-1.354,-1.513,-1.829,-1.102,-0.279,-0.725,-0.332])
	zx = np.array([0.803,0.55,0.598,0.531,0.305,0.488,0.609,0.43,0.47,0.52,0.437,0.599,0.711,0.555,0.69,0.684])

	# Depletion factor of element x at a given F_star
	x_depl = Bx + Ax*(F_star.reshape([-1,1]) - zx)

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
		return 

	if phys_dens:
		return phys_nH, DZ_vals
	else:
		return avg_nH, DZ_vals


def Chiang_2020_dust_vs_radius(bin_data = True, DZ=True, phys_r=True):
	"""
	Gives the D/Z or dust surface density values vs radius for nearby galaxies from Chiang+(2020). Given max radius it will returned the
	binned data
	"""
	bin_nums = 40

	gal_names = ['IC342','M31','M33','M101','NGC628']
	gal_distance = np.array([2.29,0.79,0.92,6.96,9.77])*1E3 # kpc distance to galaxy

	data = np.genfromtxt("Chiang+20_dat.csv",names=True,delimiter=',',dtype=None)
	ID = data['gal']
	if DZ:
		vals = np.power(10,data['dtm'])
	else:
		vals = data['dust']
	if phys_r:
		arcsec_to_rad = 4.848E-6
		radius = data['radius_arcsec']*arcsec_to_rad
	else: 
		radius = data['radius_r25']

	data = dict()
	for i,name in enumerate(gal_names):
		if phys_r:
			r_data = radius[ID==name]*gal_distance[i]
		else:
			r_data = radius[ID==name]
		r_max = np.max(r_data)
		dust_data = vals[ID==name]
		# If given a max radius, bin the data for each galaxy
		if bin_data:
			mean_vals = np.zeros(bin_nums - 1)
			# 16th and 84th percentiles
			std_vals = np.zeros([bin_nums - 1,2])
			r_bins = np.linspace(0, r_max, num=bin_nums)
			r_vals = (r_bins[1:] + r_bins[:-1]) / 2.

			for j in range(bin_nums-1):
				# find all coordinates within shell
				r_min = r_bins[j]; r_max = r_bins[j+1];
				in_annulus = np.logical_and(r_data <= r_max, r_data > r_min)
				values = dust_data[in_annulus]
				if len(values) > 0:
					mean_vals[j] = np.mean(values)
					std_vals[j] = np.percentile(values, [16,84])
				else:
					mean_vals[j] = np.nan
					std_vals[j,0] = np.nan; std_vals[j,1] = np.nan;
			mask = np.logical_not(np.isnan(mean_vals))
			data[name] = [r_vals[mask], mean_vals[mask], std_vals[mask]]
		# Else just give the raw data for each galaxy	
		else:
			data[name] = [r_data,dust_data]

	return data


def Chiang_2020_DZ_vs_fH2(bin_data = True):
	data = np.genfromtxt("Chiang+20_dat.csv",names=True,delimiter=',',dtype=None)
	gal = data['gal']
	IDs = np.unique(data['gal'])
	DZ = np.power(10,data['dtm'])
	fH2 = data['fh2']
	data = dict()

	for gal_name in IDs:
		DZ_data = DZ[gal==gal_name]
		fH2_data = fH2[gal==gal_name]
		# Bin data if asked for
		if bin_data:
			bin_nums = 40
			mean_vals = np.zeros(bin_nums - 1)
			# 16th and 84th percentiles
			std_vals = np.zeros([bin_nums - 1,2])
			fH2_bins = np.linspace(np.min(fH2_data), np.max(fH2_data), num=bin_nums)
			fH2_vals = (fH2_bins[1:] + fH2_bins[:-1]) / 2.
			digitized = np.digitize(fH2_data,fH2_bins)
			for j in range(1,len(fH2_bins)):
				if len(fH2_data[digitized==j])==0:
					mean_vals[j-1] = np.nan
					std_vals[j-1,0] = np.nan; std_vals[j-1,1] = np.nan;
					continue
				else:
					values = DZ_data[digitized == j]
					mean_vals[j-1] = np.mean(values)
					std_vals[j-1] = np.percentile(values, [16,84])
			mask = np.logical_not(np.isnan(mean_vals))
			data[gal_name] = [fH2_vals[mask], mean_vals[mask], std_vals[mask]]

		else:
			data[gal_name] = [fH2_data,DZ_data]

	return data


	for id in IDs:
		data[id] = [fH2[gal==id],DZ[id==gal]]

	return data


def Chiang_2020_DZ_vs_Sigma_dust(bin_data = True):
	data = np.genfromtxt("Chiang+20_dat.csv",names=True,delimiter=',',dtype=None)
	gal = data['gal']
	IDs = np.unique(data['gal'])
	DZ = np.power(10,data['dtm'])
	dust = data['dust']

	data = dict()
	for gal_name in IDs:
		DZ_data = DZ[gal==gal_name]
		dust_surf = dust[gal==gal_name]
		# Bin data if asked for
		if bin_data:
			bin_nums = 40
			mean_vals = np.zeros(bin_nums - 1)
			# 16th and 84th percentiles
			std_vals = np.zeros([bin_nums - 1,2])
			surf_bins = np.logspace(np.log10(np.min(dust_surf)), np.log10(np.max(dust_surf)), num=bin_nums)
			surf_vals = (surf_bins[1:] + surf_bins[:-1]) / 2.
			digitized = np.digitize(dust_surf,surf_bins)
			for j in range(1,len(surf_bins)):
				if len(dust_surf[digitized==j])==0:
					mean_vals[j-1] = np.nan
					std_vals[j-1,0] = np.nan; std_vals[j-1,1] = np.nan;
					continue
				else:
					values = DZ_data[digitized == j]
					mean_vals[j-1] = np.mean(values)
					std_vals[j-1] = np.percentile(values, [16,84])
			mask = np.logical_not(np.isnan(mean_vals))
			data[gal_name] = [surf_vals[mask], mean_vals[mask], std_vals[mask]]

		else:
			data[gal_name] = [dust_surf,DZ_data]

	return data



def Chiang_2020_DZ_vs_Sigma_H2(bin_data = True):
	data = np.genfromtxt("Chiang+20_dat.csv",names=True,delimiter=',',dtype=None)
	gal = data['gal']
	IDs = np.unique(data['gal'])
	DZ = np.power(10,data['dtm'])
	H2 = np.power(10,data['h2'])

	data = dict()
	for gal_name in IDs:
		DZ_data = DZ[gal==gal_name]
		H2_surf = H2[gal==gal_name]
		# Bin data if asked for
		if bin_data:
			bin_nums = 40
			mean_vals = np.zeros(bin_nums - 1)
			# 16th and 84th percentiles
			std_vals = np.zeros([bin_nums - 1,2])
			surf_bins = np.logspace(np.log10(np.min(H2_surf)), np.log10(np.max(H2_surf)), num=bin_nums)
			surf_vals = (surf_bins[1:] + surf_bins[:-1]) / 2.
			digitized = np.digitize(H2_surf,surf_bins)
			for j in range(1,len(surf_bins)):
				if len(H2_surf[digitized==j])==0:
					mean_vals[j-1] = np.nan
					std_vals[j-1,0] = np.nan; std_vals[j-1,1] = np.nan;
					continue
				else:
					values = DZ_data[digitized == j]
					mean_vals[j-1] = np.mean(values)
					std_vals[j-1] = np.percentile(values, [16,84])
			mask = np.logical_not(np.isnan(mean_vals))
			data[gal_name] = [surf_vals[mask], mean_vals[mask], std_vals[mask]]

		else:
			data[gal_name] = [H2_surf,DZ_data]

	return data


def Chiang_2020_DZ_vs_Sigma_metals(bin_data = True):
	data = np.genfromtxt("Chiang+20_dat.csv",names=True,delimiter=',',dtype=None)
	gal = data['gal']
	IDs = np.unique(data['gal'])
	DZ = np.power(10,data['dtm'])
	metals = data['metal_z']/SOLAR_Z

	data = dict()
	for gal_name in IDs:
		DZ_data = DZ[gal==gal_name]
		metals_surf = metals[gal==gal_name]
		# Bin data if asked for
		if bin_data:
			bin_nums = 40
			mean_vals = np.zeros(bin_nums - 1)
			# 16th and 84th percentiles
			std_vals = np.zeros([bin_nums - 1,2])
			surf_bins = np.logspace(np.log10(np.min(metals_surf)), np.log10(np.max(metals_surf)), num=bin_nums)
			surf_vals = (surf_bins[1:] + surf_bins[:-1]) / 2.
			digitized = np.digitize(metals_surf,surf_bins)
			for j in range(1,len(surf_bins)):
				if len(metals_surf[digitized==j])==0:
					mean_vals[j-1] = np.nan
					std_vals[j-1,0] = np.nan; std_vals[j-1,1] = np.nan;
					continue
				else:
					values = DZ_data[digitized == j]
					mean_vals[j-1] = np.mean(values)
					std_vals[j-1] = np.percentile(values, [16,84])
			mask = np.logical_not(np.isnan(mean_vals))
			data[gal_name] = [surf_vals[mask], mean_vals[mask], std_vals[mask]]

		else:
			data[gal_name] = [metals_surf,DZ_data]

	return data


def Chiang_2020_DZ_vs_Sigma_gas(bin_data = True):
	data = np.genfromtxt("Chiang+20_dat.csv",names=True,delimiter=',',dtype=None)
	gal = data['gal']
	IDs = np.unique(data['gal'])
	DZ = np.power(10,data['dtm'])
	gas = np.power(10,data['gas'])

	data = dict()
	for gal_name in IDs:
		DZ_data = DZ[gal==gal_name]
		gas_surf = gas[gal==gal_name]
		# Bin data if asked for
		if bin_data:
			bin_nums = 40
			mean_vals = np.zeros(bin_nums - 1)
			# 16th and 84th percentiles
			std_vals = np.zeros([bin_nums - 1,2])
			surf_bins = np.logspace(np.log10(np.min(gas_surf)), np.log10(np.max(gas_surf)), num=bin_nums)
			surf_vals = (surf_bins[1:] + surf_bins[:-1]) / 2.
			digitized = np.digitize(gas_surf,surf_bins)
			for j in range(1,len(surf_bins)):
				if len(gas_surf[digitized==j])==0:
					mean_vals[j-1] = np.nan
					std_vals[j-1,0] = np.nan; std_vals[j-1,1] = np.nan;
					continue
				else:
					values = DZ_data[digitized == j]
					mean_vals[j-1] = np.mean(values)
					std_vals[j-1] = np.percentile(values, [16,84])
			mask = np.logical_not(np.isnan(mean_vals))
			data[gal_name] = [surf_vals[mask], mean_vals[mask], std_vals[mask]]

		else:
			data[gal_name] = [gas_surf,DZ_data]

	return data


def Chiang_2020_dust_surf_dens_vs_param(param):
	data = np.genfromtxt("Chiang+20_dat.csv",names=True,delimiter=',',dtype=None)
	if param == 'gas':
		vals = np.power(10,data['gas'])
	elif param == 'H2':
		vals = np.power(10,data['h2'])
	elif param == 'Z':
		vals = data['metal']
	elif param == 'DZ':
		vals = np.power(10,data['dtm'])
	else:
		print("%s is not a valid param for Chiang_2020_dust_surf_dens_vs_param"%param)
		return

	gal = data['gal']
	IDs = np.unique(data['gal'])
	dust = data['dust']

	data = dict()
	for id in IDs:
		data[id] = [vals[gal==id],dust[id==gal]]

	return data
