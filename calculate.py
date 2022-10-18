import numpy as np
import gizmo_library.config as config
import gizmo_library.utils as utils
from scipy.stats import binned_statistic_2d



def calc_binned_property_vs_property(property1, property2, snap, bin_nums=50, prop_lims=None):
	"""
	Calculates median and 16/84th-percentiles of property1 in relation to binned property2 for
	the given snapshot gas data

	Parameters
	----------
	property1: string
		Name of property to calculate median and percentiles for. Supported properties
		('D/Z','Z','nH','T','r','fH2','X_depletion')
	property2: string
		Name of property for property1 to be binned over
	snap : snapshot/galaxy
		Snapshot or Galaxy object from which particle data can be loaded
	bin_nums : int
		Number of bins to use for property2
	prop_lims : ndarray, optional
		Limits for property2 binning

	Returns
	-------
	bin_vals: ndarray
		Bins for property2
	mean_vals : ndarray
		Median of of property1 across property2 bins
	std_vals : ndarray
		16/84th-percentiles of property1 across property2 bins

	"""

	if property1 in ['sigma_star','sigma_stellar','stellar_Z','age'] or \
	   property2 in ['sigma_star','sigma_stellar','stellar_Z','age']:
		ptype = 4
	else:
		ptype = 0

	P = snap.loadpart(ptype)
	# Get property data
	data = np.zeros([2,P.npart])
	weights = P.get_property('M')
	for i, property in enumerate([property1,property2]):
		data[i] = P.get_property(property)

	if prop_lims is None:
		prop_lims = config.PROP_INFO[property2][1]
		log_bins  = config.PROP_INFO[property2][2]
	else:
		if prop_lims[1] > 30*prop_lims[0]: 	log_bins=True
		else:								log_bins=False

	bin_vals, mean_DZ, std_DZ = utils.bin_values(data[1], data[0], prop_lims, bin_nums=bin_nums, weight_vals=weights,
												 log=log_bins)

	return bin_vals, mean_DZ, std_DZ




def calc_phase_hist_data(property, snap, bin_nums=100):
	"""
	Calculate the 2D histogram for the given property and data from gas particle

	Parameters
	----------
	property : string
		Name of property to calculate phase plot for
	snap : snapshot/galaxy
		Snapshot or Galaxy object from which particle data can be loaded
	bin_nums: int, optional
		Number of bins to use

	Returns
	-------
	ret : tuple
		Binned 2D data with bin edges and bin numbers

	"""

	# Set up x and y data, limits, and bins
	G = snap.loadpart(0)
	nH_data = G.get_property('nH')
	T_data = G.get_property('T')
	nH_bin_lims = config.PROP_INFO['nH'][1]
	T_bin_lims = config.PROP_INFO['T'][1]
	if config.PROP_INFO['nH'][2]:
		nH_bins = np.logspace(np.log10(nH_bin_lims[0]),np.log10(nH_bin_lims[1]),bin_nums)
	else:
		nH_bins = np.linspace(nH_bin_lims[0], nH_bin_lims[1], bin_nums)
	if config.PROP_INFO['T'][2]:
		T_bins = np.logspace(np.log10(T_bin_lims[0]),np.log10(T_bin_lims[1]),bin_nums)
	else:
		T_bins = np.linspace(T_bin_lims[0], T_bin_lims[1], bin_nums)

	if property in ['M_H2','M_gas']:
		func = np.sum
	else:
		func = np.mean
	bin_data = G.get_property(property)
	ret = binned_statistic_2d(nH_data, T_data, bin_data, statistic=func, bins=[nH_bins, T_bins])
	# Need to catch case were np.sum is given empty array which will return zero
	if property in ['M_H2','M_gas']:
		ret.statistic[ret.statistic<=0] = np.nan

	return ret



def calc_binned_obs_property_vs_property(property1, property2, snap, r_max=20, pixel_res=2, bin_nums=50, prop_lims=None):
	"""
	First calculates mock observations of property1 and property2 for the given pixel resolution. Then
	calculates median and 16/84th-percentiles of property1 in relation to binned property2.

	Parameters
	----------
	property1: string
		Name of property to calculate median and percentiles for.
	property2: string
		Name of property for property1 to be binned over
	snap : snapshot/galaxy
		Snapshot or Galaxy object from which particle data can be loaded
	r_max : double, optional
		Maximum radius from the center of the simulated observation
	pixel_res : double, optional
		Size resolution of each pixel bin in kpc
	bin_nums : int, optional
		Number of bins for property2
	prop_lims : ndarray, optional
		Limits for property2 binning

	Returns
	-------
	bin_vals: ndarray
		Bins for property2
	mean_vals : ndarray
		Median of of property1 across property2 bins
	std_vals : ndarray
		16/84th-percentiles of property1 across property2 bins
	"""

	if prop_lims is None:
		prop_lims = config.PROP_INFO[property2][1]
		log_bins = config.PROP_INFO[property2][2]
	else:
		if prop_lims[1] > 30*prop_lims[0]:	log_bins=True
		else:								log_bins=False



	pixel_bins = int(np.ceil(2*r_max/pixel_res))
	x_bins = np.linspace(-r_max,r_max,pixel_bins)
	y_bins = np.linspace(-r_max,r_max,pixel_bins)
	x_vals = (x_bins[1:] + x_bins[:-1]) / 2.
	y_vals = (y_bins[1:] + y_bins[:-1]) / 2.
	pixel_area = pixel_res**2 * 1E6 # area of pixel in pc^2

	pixel_data = np.zeros([2,(pixel_bins-1)**2])
	for i, property in enumerate([property1,property2]):
		if property in ['sigma_sfr','sigma_stellar','sigma_star']:
			ptype = 4
		else:
			ptype = 0

		P = snap.loadpart(ptype)
		x = P.p[:,0]; y = P.p[:,1];

		if property == 'sigma_dust':
			bin_data = P.get_property('M_dust')
			ret = binned_statistic_2d(x, y, bin_data, statistic=np.sum, bins=[x_bins,y_bins]).statistic
			dust_pixel = ret.flatten()/pixel_area
			pixel_data[i] = dust_pixel
		elif property=='sigma_gas':
			bin_data = P.get_property('M_gas')
			ret = binned_statistic_2d(x, y, bin_data, statistic=np.sum, bins=[x_bins,y_bins]).statistic
			M_pixel = ret.flatten()/pixel_area
			pixel_data[i] = M_pixel
		elif property in ['sigma_stellar','sigma_star']:
			bin_data = P.get_property('M_stellar')
			ret = binned_statistic_2d(x, y, bin_data, statistic=np.sum, bins=[x_bins,y_bins]).statistic
			M_pixel = ret.flatten()/pixel_area
			pixel_data[i] = M_pixel
		elif property=='sigma_sfr':
			bin_data = P.get_property('M_sfr')
			ret = binned_statistic_2d(x, y, bin_data, statistic=np.sum, bins=[x_bins,y_bins]).statistic
			M_pixel = ret.flatten()/pixel_area
			pixel_data[i] = M_pixel
		elif property=='sigma_gas_neutral':
			bin_data = P.get_property('M_gas_neutral')
			ret = binned_statistic_2d(x, y, bin_data, statistic=np.sum, bins=[x_bins,y_bins]).statistic
			M_pixel = ret.flatten()/pixel_area
			pixel_data[i] = M_pixel
		elif property=='sigma_H2':
			bin_data = P.get_property('M_H2')
			ret = binned_statistic_2d(x, y, bin_data, statistic=np.sum, bins=[x_bins,y_bins]).statistic
			MH2_pixel = ret.flatten()/pixel_area
			pixel_data[i] = MH2_pixel
		elif property == 'sigma_Z' or property == 'sigma_metals':
			bin_data = P.get_property('M_metals')
			ret = binned_statistic_2d(x, y, bin_data, statistic=np.sum, bins=[x_bins,y_bins]).statistic
			Z_pixel = ret.flatten()/pixel_area
			pixel_data[i] = Z_pixel
		elif property == 'Z':
			bin_data = [P.get_property('M_metals'),P.get_property('M_gas')]
			ret = binned_statistic_2d(x, y, bin_data, statistic=np.sum, bins=[x_bins,y_bins]).statistic
			Z_pixel = ret[0].flatten()/(ret[1].flatten())/config.SOLAR_Z
			pixel_data[i] = Z_pixel
		elif property == 'O/H':
			bin_data = P.get_property('O/H')
			ret = binned_statistic_2d(x, y, bin_data, statistic=np.mean, bins=[x_bins,y_bins]).statistic
			OH_pixel = ret.flatten()
			pixel_data[i] = OH_pixel
		elif property == 'O/H_gas':
			bin_data = P.get_property('O/H_gas')
			ret = binned_statistic_2d(x, y, bin_data, statistic=np.mean, bins=[x_bins,y_bins]).statistic
			OH_pixel = ret.flatten()
			pixel_data[i] = OH_pixel
		elif property == 'O/H_ionized':
			bin_data = P.get_property('O/H')
			nH = P.get_property('nH')
			T = P.get_property('T')
			mask = (nH>=0.5) & (T>=7000) & (T<=15000)
			ret = binned_statistic_2d(x[mask], y[mask], bin_data[mask], statistic=np.mean, bins=[x_bins,y_bins]).statistic
			OH_pixel = ret.flatten()
			pixel_data[i] = OH_pixel
		elif property == 'O/H_gas_ionized':
			bin_data = P.get_property('O/H_gas')
			nH = P.get_property('nH')
			T = P.get_property('T')
			mask = (nH>=0.5) & (T>=7000) & (T<=15000)
			ret = binned_statistic_2d(x[mask], y[mask], bin_data[mask], statistic=np.mean, bins=[x_bins,y_bins]).statistic
			OH_pixel = ret.flatten()
			pixel_data[i] = OH_pixel
		elif property == 'fH2':
			bin_data = [P.get_property('M_H2'),P.get_property('M_gas')]
			ret = binned_statistic_2d(x, y, bin_data, statistic=np.sum, bins=[x_bins,y_bins]).statistic
			fH2_pixel = ret[0].flatten()/(ret[1].flatten())
			pixel_data[i] = fH2_pixel
		elif property in ['r','r25']:
			r_conversion = 1.
			if property == 'r25':
				try:
					r25 = snap.calc_stellar_scale_r()/0.2
					r_conversion = 1./r25
				except NameError:
					print("Using r25 only works for disk galaxy objects. Will default to galactocentric radius.")
			# Get the average r coordinate for each pixel in kpc
			pixel_r_vals = np.array([np.sqrt(np.power(np.abs(y_vals),2) + np.power(np.abs(x_vals[k]),2))*r_conversion for k in range(len(x_vals))]).flatten()
			pixel_data[i] = pixel_r_vals
			# Makes more sense to force the number of bins for this
			bin_nums = pixel_bins//2
		elif property == 'D/Z':
			bin_data = [P.get_property('M_dust'),P.get_property('M_metals')]
			ret = binned_statistic_2d(x, y, bin_data, statistic=np.sum, bins=[x_bins,y_bins]).statistic
			DZ_pixel = np.divide(ret[0].flatten(),ret[1].flatten(),where=ret[1].flatten()!=0)
			pixel_data[i] = DZ_pixel
		else:
			print("Property given to calc_binned_obs_property_vs_property() is not supported:",property)
			return None,None,None


	bin_vals, mean_vals, std_vals = utils.bin_values(pixel_data[1], pixel_data[0], prop_lims, bin_nums=bin_nums, weight_vals=None, log=log_bins)

	return bin_vals, mean_vals, std_vals



def calc_gal_int_params(property, snap, criteria='all'):
	"""
	Calculate the galaxy-integrated values given center and virial radius for multiple simulations/snapshots

	Parameters
	----------
	property: string
		Property to calculate the galaxy-integrated value for (D/Z, Z)
	snap : snapshot/galaxy
		Snapshot or Galaxy object from which particle data can be loaded
	criteria : string, optional
		Criteria for what gas/star particles to use for the given galaxy_integrated property and D/Z.
		Default is all particles.
		cold/neutral : Use only cold/neutral gas T<1000K
		hot/ionized: Use only ionized/hot gas
		molecular: Use only molecular gas


	Returns
	-------
	val : double
		Galaxy-integrated values for given property

	"""

	if property in ['M_star']:
		ptype = 4
	else:
		ptype = 0


	P = snap.loadpart(ptype)
	weights = P.get_property('M')
	prop_vals = P.get_property(property)
	mask = np.ones(len(prop_vals), dtype=int)
	if ptype==0:
		if 'cold' in criteria:
			T = P.get_property('T')
			mask = T < 1E3
		elif 'hot' in criteria:
			T = P.get_property('T')
			mask = T > 1E4
		elif 'molecular' in criteria:
			fH2 = P.get_property('fH2')
			mask = fH2 > 0.5
		else:
			if criteria!='all':
				print("Criteria %s used in calc_gal_int_params() is not supported. Defaulting to all."%criteria)
	if ptype==4:
		if 'young' in criteria:
			age = P.get_property('age')
			mask = age < 0.1
		elif 'old' in criteria:
			age = P.get_property('age')
			mask = age < 1.
		else:
			if criteria!='all':
				print("Criteria %s used in calc_gal_int_params() is not supported. Defaulting to all."%criteria)

	weights=weights[mask]
	prop_vals=prop_vals[mask]

	val = utils.weighted_percentile(prop_vals, percentiles=np.array([50]), weights=weights, ignore_invalid=True)

	return val



def calc_projected_prop(property, snap, side_lens, pixel_res=2, proj='xy'):
	"""
	Calculates the 2D projection of a give property given the projection orientation and resolution

	Parameters
	----------
	property: string
		Name of property to project
	snap : Snapshot/Halo
	    Snapshot data structure (either gas or star particle depending on property)
	side_len : list
		Size of box to consider (L1, L2, Lz), L1 ands L2 are side lengths and Lz depth is depth of box
	pixel_res : double
		Size resolution of each pixel bin in kpc
	proj : string
		What 2D coordinates you want to project (xy,yz,zx)

	Returns
	-------
	pixel_data : array
		Projected data for each pixel
	coord1_bins : array
		Bins for first coordinate
	coord2_bins : array
		Bins for second coordinate
	"""

	L1 = side_lens[0]; L2 = side_lens[1]; Lz = side_lens[2]

	if 'star' in property or 'stellar' in property or 'sfr' in property: P = snap.loadpart(4)
	else: 				   												 P = snap.loadpart(0)
	x = P.p[:,0];y=P.p[:,1];z=P.p[:,2]

	# Set up coordinates to project
	if   proj=='xy': coord1 = x; coord2 = y; coord3 = z;
	elif proj=='yz': coord1 = y; coord2 = z; coord3 = x;
	elif proj=='xz': coord1 = x; coord2 = z; coord3 = y;
	else:
		print("Projection must be xy, yz, or xz for calc_projected_prop()")
		return None

	# Only include particles in the box
	mask = (coord1>-L1) & (coord1<L1) & (coord2>-L2) & (coord2<L2) & (coord3>-Lz) & (coord3<Lz)

	pixel_bins = int(np.ceil(2*L1/pixel_res)) + 1
	coord1_bins = np.linspace(-L1,L1,pixel_bins)
	pixel_bins = int(np.ceil(2*L2/pixel_res)) + 1
	coord2_bins = np.linspace(-L2,L2,pixel_bins)



	# Get the data to be projected
	if property in ['D/Z','fH2','fMC']:
		if property == 'D/Z':
			proj_data1 = P.get_property('M_dust')
			proj_data2 = P.get_property('M_metals')
		elif property == 'fH2':
			proj_data1 = P.get_property('M_gas')
			proj_data2 = P.get_property('M_H2')
		else:
			proj_data1 = P.get_property('M_gas')
			proj_data2 = P.get_property('M_mc')
		binned_stats = binned_statistic_2d(coord1[mask], coord2[mask],[proj_data1[mask],proj_data2[mask]], statistic=np.sum, bins=[coord1_bins,coord2_bins])
		pixel_stats = binned_stats.statistic[0]/binned_stats.statistic[1]

	else:
		if   property == 'sigma_dust':  	proj_data = P.get_property('M_dust')
		elif property == 'sigma_gas': 		proj_data = P.get_property('M_gas')
		elif property == 'sigma_H2': 		proj_data = P.get_property('M_H2')
		elif property == 'sigma_metals': 	proj_data = P.get_property('M_metals')
		elif property == 'sigma_sil': 		proj_data = P.get_property('M_sil')
		elif property == 'sigma_sil+':  	proj_data = P.get_property('M_sil+')
		elif property == 'sigma_carb':  	proj_data = P.get_property('M_carb')
		elif property == 'sigma_SiC': 		proj_data = P.get_property('M_SiC')
		elif property == 'sigma_iron':  	proj_data = P.get_property('M_iron')
		elif property == 'sigma_ORes': 		proj_data = P.get_property('M_ORes')
		elif property == 'sigma_star':  	proj_data = P.get_property('M_star')
		elif property == 'sigma_sfr':  		proj_data = P.get_property('M_star_young')
		elif property == 'T':				proj_data = P.get_property('T')
		else:
			print("%s is not a supported parameter in calc_obs_projection()."%property)
			return None

		if 'sigma' in property:
			stats = np.nansum
			pixel_area = pixel_res**2 * 1E6 # area of pixel in pc^2
		else:
			stats = np.average
			pixel_area = 1.

		binned_stats = binned_statistic_2d(coord1[mask], coord2[mask], proj_data[mask], statistic=stats, bins=[coord1_bins,coord2_bins])
		pixel_stats = binned_stats.statistic/pixel_area

	return pixel_stats, coord1_bins, coord2_bins


def calc_radial_dens_projection(property, snap, rmax, rmin=0, proj='xy', bin_nums=50, log_bins=False):
	"""
	Calculates the 2D radial density projection of a give property given the projection orientation

	Parameters
	----------
	property: string
		Name of property to project
	snap : Snapshot/Halo
	    Snapshot data structure (either gas or star particle depending on property)
	rmax : double
		Maximum radius to calculate projection in kpc
	proj : string
		What 2D coordinates you want to project (xy,yz,zx)
	bin_nums : int
		Numbers of radial bins

	Returns
	-------
	sigma_vals : array
		Projected density data for each radial bin
	r_vals : array
		Center radial bins values
	"""

	if 'star' in property or 'stellar' in property or 'sfr' in property: P = snap.loadpart(4)
	else: 				   												 P = snap.loadpart(0)
	x = P.p[:,0];y=P.p[:,1];z=P.p[:,2]

	# Set up coordinates to project
	if   proj=='xy': coord1 = x; coord2 = y;
	elif proj=='yz': coord1 = y; coord2 = z;
	elif proj=='xz': coord1 = x; coord2 = z;
	else:
		print("Projection must be xy, yz, or xz for calc_projected_prop()")
		return None
	# Only include particles in the box
	coordr = np.sqrt(np.power(coord1,2)+np.power(coord2,2))
	mask = coordr<rmax
	coordr = coordr[mask]

	# Get the data to be projected
	if   property == 'sigma_dust':  	proj_data = P.get_property('M_dust')
	elif property == 'sigma_gas': 		proj_data = P.get_property('M_gas')
	elif property == 'sigma_H2': 		proj_data = P.get_property('M_H2')
	elif property == 'sigma_metals': 	proj_data = P.get_property('M_metals')
	elif property == 'sigma_sil': 		proj_data = P.get_property('M_sil')
	elif property == 'sigma_sil+':  	proj_data = P.get_property('M_sil+')
	elif property == 'sigma_carb':  	proj_data = P.get_property('M_carb')
	elif property == 'sigma_SiC': 		proj_data = P.get_property('M_SiC')
	elif property == 'sigma_iron':  	proj_data = P.get_property('M_iron')
	elif property == 'sigma_ORes': 		proj_data = P.get_property('M_ORes')
	elif property == 'sigma_star':  	proj_data = P.get_property('M_star')
	else:
		print("%s is not a supported parameter in calc_obs_projection()."%property)
		return None
	proj_data = proj_data[mask]


	if log_bins:
		r_bins = np.logspace(np.log10(rmin), np.log10(rmax), bin_nums)
	else:
		r_bins = np.linspace(rmin, rmax, bin_nums)
	r_vals = (r_bins[1:] + r_bins[:-1]) / 2.
	sigma_vals = np.zeros(len(r_vals))

	for j in range(bin_nums-1):
		# find all coordinates within shell
		r_min = r_bins[j]; r_max = r_bins[j+1];
		in_annulus = np.logical_and(coordr <= r_max, coordr > r_min)
		area = 4*np.pi*(r_max**2-r_min**2) * 1E6 # kpc^2
		sigma_vals[j] = np.sum(proj_data[in_annulus])/area

	return r_vals, sigma_vals