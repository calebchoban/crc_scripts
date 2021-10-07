import numpy as np
import gizmo_library.config as config
import gizmo_library.utils as utils
from scipy.stats import binned_statistic_2d


def get_particle_data(particle, property):
	"""
	Retrieves property data from particle, reducing or converting as needed

	Parameters
	----------
	particle: Particle
	    Particle to get data from
	property: string
		Name of property to get data for

	Returns
	-------
	data : ndarray
		Data for given property from the particle

	"""
	if property == 'M' or property == 'M_gas':
		data = particle.m*config.UnitMass_in_Msolar
	elif property == 'M_metals':
		data = particle.z[:,0]*particle.m*config.UnitMass_in_Msolar
	elif property == 'M_dust':
		data = particle.dz[:,0]*particle.m*config.UnitMass_in_Msolar
	elif property == 'M_sil':
		data = particle.spec[:,0]*particle.m*config.UnitMass_in_Msolar
	elif property == 'M_carb':
		data = particle.spec[:,1]*particle.m*config.UnitMass_in_Msolar
	elif property == 'M_SiC':
		data = particle.spec[:,2]*particle.m*config.UnitMass_in_Msolar
	elif property == 'M_iron':
		if particle.sp.Flag_DustSpecies>4:
			data = (particle.spec[:,3]+particle.spec[:,5])*particle.m*config.UnitMass_in_Msolar
		else:
			data = particle.spec[:,3]*particle.m*config.UnitMass_in_Msolar
	elif property == 'M_ORes':
		data = particle.spec[:,4]*particle.m*config.UnitMass_in_Msolar
	elif property == 'M_sil+':
		data = (particle.spec[:,0]+np.sum(particle.spec[:,2:],axis=1))*particle.m*config.UnitMass_in_Msolar
	elif property == 'fH2':
		data = particle.fH2
		data[data>1] = 1
	elif property == 'fMC':
		data = particle.fMC
		data[data>1] = 1
	elif property == 'CinCO':
		data = particle.CinCO/particle.z[:,2]
	elif property == 'nH':
		data = particle.rho*config.UnitDensity_in_cgs * (1. - (particle.z[:,0]+particle.z[:,1])) / config.H_MASS
	elif property == 'T':
		data = particle.T
	elif property == 'r':
		data = np.sqrt(np.power(particle.p[:,0],2) + np.power(particle.p[:,1],2))
	elif property == 'Z':
		data = particle.z[:,0]/config.SOLAR_Z
	elif property == 'Z_all':
		data = particle.z
	elif property == 'O/H':
		O = particle.z[:,4]/config.ATOMIC_MASS[4]; H = (1-(particle.z[:,0]+particle.z[:,1]))/config.ATOMIC_MASS[0]
		data = 12+np.log10(O/H)
	elif property == 'Si/C':
		data = particle.spec[:,0]/particle.spec[:,1]
	elif property == 'D/Z':
		data = particle.dz[:,0]/particle.z[:,0]
		data[data > 1] = 1.
	elif 'depletion' in property:
		elem = property.split('_')[0]
		if elem not in config.ELEMENTS:
			print('%s is not a valid element to calculate depletion for. Valid elements are'%elem)
			print(config.ELEMENTS)
			return None,None,None
		elem_indx = config.ELEMENTS.index(elem)
		data =  particle.dz[:,elem_indx]/particle.z[:,elem_indx]
		data[data > 1] = 1.
	else:
		print("Property given to get_particle_data is not supported:",property)
		return None

	return data



def calc_binned_property_vs_property(property1, property2, G, bin_nums=50, prop_lims=None):
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
	G : Particle
	    Gas particle to get data from
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

	# Get property data
	data = np.zeros([2,G.npart])
	for i, property in enumerate([property1,property2]):
		data[i] = get_particle_data(G, property)

	if prop_lims is None:
		prop_lims = config.PROP_INFO[property2][1]
		log_bins = config.PROP_INFO[property2][2]
	else:
		if prop_lims[1] > 30*prop_lims[0]: 	log_bins=True
		else:								log_bins=False

	bin_vals, mean_DZ, std_DZ = utils.bin_values(data[1], data[0], prop_lims, bin_nums=bin_nums, weight_vals=G.m, log=log_bins)

	return bin_vals, mean_DZ, std_DZ




def calc_phase_hist_data(property, G, bin_nums=100):
	"""
	Calculate the 2D histogram for the given property and data from gas particle

	Parameters
	----------
	property : string
		Name of property to calculate phase plot for
	G : Particle
	    Gas particle data to use
	bin_nums: int, optional
		Number of bins to use

	Returns
	-------
	ret : tuple
		Binned 2D data with bin edges and bin numbers

	"""

	# Set up x and y data, limits, and bins
	nH_data = get_particle_data(G,'nH')
	T_data = get_particle_data(G,'T')
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
	bin_data = get_particle_data(G,property)
	ret = binned_statistic_2d(nH_data, T_data, bin_data, statistic=func, bins=[nH_bins, T_bins])
	# Need to catch case were np.sum is given empty array which will return zero
	if property in ['M_H2','M_gas']:
		ret.statistic[ret.statistic<=0] = np.nan

	return ret



def calc_binned_obs_property_vs_property(property1, property2, G, r_max=20, pixel_res=2, bin_nums=50, prop_lims=None):
	"""
	First calculates mock observations of property1 and property2 for the given pixel resolution. Then
	calculates median and 16/84th-percentiles of property1 in relation to binned property2.

	Parameters
	----------
	property1: string
		Name of property to calculate median and percentiles for.
	property2: string
		Name of property for property1 to be binned over
	G : Particle
	    Particle data structure to get property data from
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

	# TODO: Add star particle data, currently only gas particles are supported

	if prop_lims is None:
		prop_lims = config.PROP_INFO[property2][1]
		log_bins = config.PROP_INFO[property2][2]
	else:
		if prop_lims[1] > 30*prop_lims[0]:	log_bins=True
		else:								log_bins=False



	x = G.p[:,0]; y = G.p[:,1];
	pixel_bins = int(np.ceil(2*r_max/pixel_res))
	x_bins = np.linspace(-r_max,r_max,pixel_bins)
	y_bins = np.linspace(-r_max,r_max,pixel_bins)
	x_vals = (x_bins[1:] + x_bins[:-1]) / 2.
	y_vals = (y_bins[1:] + y_bins[:-1]) / 2.
	pixel_area = pixel_res**2 * 1E6 # area of pixel in pc^2

	pixel_data = np.zeros([2,(pixel_bins-1)**2])
	for i, property in enumerate([property1,property2]):

		if property == 'sigma_dust':
			bin_data = get_particle_data(G,'M_dust')
			ret = binned_statistic_2d(x, y, bin_data, statistic=np.sum, bins=[x_bins,y_bins]).statistic
			dust_pixel = ret.flatten()/pixel_area
			pixel_data[i] = dust_pixel
		elif property=='sigma_gas':
			bin_data = get_particle_data(G,'M_gas')
			ret = binned_statistic_2d(x, y, bin_data, statistic=np.sum, bins=[x_bins,y_bins]).statistic
			M_pixel = ret.flatten()/pixel_area
			pixel_data[i] = M_pixel
		elif property=='sigma_H2':
			bin_data = get_particle_data(G,'M_H2')
			ret = binned_statistic_2d(x, y, bin_data, statistic=np.sum, bins=[x_bins,y_bins]).statistic
			MH2_pixel = ret.flatten()/pixel_area
			pixel_data[i] = MH2_pixel
		elif property == 'sigma_Z':
			bin_data = get_particle_data(G,'M_metals')
			ret = binned_statistic_2d(x, y, bin_data, statistic=np.sum, bins=[x_bins,y_bins]).statistic
			Z_pixel = ret.flatten()/pixel_area
			pixel_data[i] = Z_pixel
		elif property == 'fH2':
			bin_data = [get_particle_data(G,'M_H2'),get_particle_data(G,'M_gas')]
			ret = binned_statistic_2d(x, y, bin_data, statistic=np.sum, bins=[x_bins,y_bins]).statistic
			fH2_pixel = ret[0].flatten()/(ret[1].flatten())
			pixel_data[i] = fH2_pixel
		elif property == 'r':
			# Get the average r coordinate for each pixel in kpc
			pixel_r_vals = np.array([np.sqrt(np.power(np.abs(y_vals),2) + np.power(np.abs(x_vals[k]),2)) for k in range(len(x_vals))]).flatten()
			pixel_data[i] = pixel_r_vals
			# Makes more sense to force the number of bins for this
			bin_nums = pixel_bins//2
		elif property == 'D/Z':
			bin_data = [get_particle_data(G,'M_dust'),get_particle_data(G,'M_metals')]
			ret = binned_statistic_2d(x, y, bin_data, statistic=np.sum, bins=[x_bins,y_bins]).statistic
			DZ_pixel = np.divide(ret[0].flatten(),ret[1].flatten(),where=ret[1].flatten()!=0)
			pixel_data[i] = DZ_pixel
		else:
			print("Property given to calc_binned_obs_property_vs_property() is not supported:",property)
			return None,None,None


	bin_vals, mean_vals, std_vals = utils.bin_values(pixel_data[1], pixel_data[0], prop_lims, bin_nums=bin_nums, weight_vals=None, log=log_bins)

	return bin_vals, mean_vals, std_vals



def calc_gal_int_params(property, G):
	"""
	Calculate the galaxy-integrated values given center and virial radius for multiple simulations/snapshots

	Parameters
	----------
	property: string
		Property to calculate the galaxy-integrated value for (D/Z, Z)
	G : Particle
	    Particle gas data structure

	Returns
	-------
	val : double
		Galaxy-integrated values for given property

	"""

	if property == 'D/Z':
		val = utils.weighted_percentile(get_particle_data(G,'D/Z'), percentiles=np.array([50]), weights=G.m, ignore_invalid=True)
	elif property == 'Z':
		val = utils.weighted_percentile(get_particle_data(G,'Z'), percentiles=np.array([50]), weights=G.m, ignore_invalid=True)
	elif property == 'O/H':
		val = utils.weighted_percentile(get_particle_data(G,'O/H'), percentiles=np.array([50]), weights=G.m, ignore_invalid=True)
	elif property == 'M_gas':
		val = np.sum(G.m*config.UnitMass_in_Msolar)
	else:
		print("Property given to calc_gal_int_params is not supported:",property)
		return None

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

	if 'star' in property: P = snap.loadpart(4)
	else: 				   P = snap.loadpart(0)
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

	pixel_area = pixel_res**2 * 1E6 # area of pixel in pc^2

	# Get the data to be projected
	if property in ['D/Z','fH2','fMC']:
		if property == 'D/Z':
			proj_data1 = get_particle_data(P,'M_dust')
			proj_data2 = get_particle_data(P,'M_metals')
		elif property == 'fH2':
			proj_data1 = get_particle_data(P,'M_gas')
			proj_data2 = get_particle_data(P,'M_H2')
		elif property == 'fMC':
			proj_data1 = get_particle_data(P,'M_gas')
			proj_data2 = get_particle_data(P,'M_mc')
		binned_stats = binned_statistic_2d(coord1[mask], coord2[mask],[proj_data1[mask],proj_data2[mask]], statistic=np.sum, bins=[coord1_bins,coord2_bins])
		pixel_stats = binned_stats.statistic[0]/binned_stats.statistic[1]

	else:
		if   property == 'sigma_dust':  	proj_data = get_particle_data(P,'M_dust')
		elif property == 'sigma_gas': 		proj_data = get_particle_data(P,'M_gas')
		elif property == 'sigma_H2': 		proj_data = get_particle_data(P,'M_H2')
		elif property == 'sigma_metals': 	proj_data = get_particle_data(P,'M_metals')
		elif property == 'sigma_sil': 		proj_data = get_particle_data(P,'M_sil')
		elif property == 'sigma_sil+':  	proj_data = get_particle_data(P,'M_sil+')
		elif property == 'sigma_carb':  	proj_data = get_particle_data(P,'M_carb')
		elif property == 'sigma_SiC': 		proj_data = get_particle_data(P,'M_SiC')
		elif property == 'sigma_iron':  	proj_data = get_particle_data(P,'M_iron')
		elif property == 'sigma_ORes': 		proj_data = get_particle_data(P,'M_ORes')
		elif property == 'sigma_star':  	proj_data = get_particle_data(P,'M_star')
		else:
			print("%s is not a supported parameter in calc_obs_projection()."%property)
			return None

		binned_stats = binned_statistic_2d(coord1[mask], coord2[mask], proj_data[mask], statistic=np.sum, bins=[coord1_bins,coord2_bins])
		pixel_stats = binned_stats.statistic/pixel_area

	return pixel_stats, coord1_bins, coord2_bins