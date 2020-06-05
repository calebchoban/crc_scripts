import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
plt.switch_backend('agg')
from scipy.stats import binned_statistic_2d
import pickle
import os
from readsnap import readsnap
from astropy.table import Table
import gas_temperature as gas_temp
from tasz import *
import observations

# Set style of plots
plt.style.use('seaborn-talk')
# Set personal color cycle
Line_Colors = ["xkcd:blue", "xkcd:red", "xkcd:green", "xkcd:orange", "xkcd:violet", "xkcd:teal", "xkcd:brown"]
Marker_Colors = ["xkcd:violet", "xkcd:teal", "xkcd:brown", "xkcd:gold", "xkcd:tomato"]
Line_Styles = ['-','--',':','-.']
Marker_Style = ['o','^','X','s','*']
Line_Widths = [0.5,1.0,1.5,2.0,2.5,3.0]

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=Line_Colors)

# Large and small font sizes to be used for axes labels and legends
Large_Font=26
Small_Font=16


UnitLength_in_cm            = 3.085678e21   # 1.0 kpc/h
UnitMass_in_g               = 1.989e43  	# 1.0e10 solar masses/h
UnitMass_in_Msolar			= UnitMass_in_g / 1.989E33
UnitVelocity_in_cm_per_s    = 1.0e5   	    # 1 km/sec
UnitTime_in_s 				= UnitLength_in_cm / UnitVelocity_in_cm_per_s # 0.978 Gyr/h
UnitTime_in_Gyr 			= UnitTime_in_s /1e9/365./24./3600.
UnitEnergy_per_Mass 		= np.power(UnitLength_in_cm, 2) / np.power(UnitTime_in_s, 2)
UnitDensity_in_cgs 			= UnitMass_in_g / np.power(UnitLength_in_cm, 3)
H_MASS 						= 1.67E-24 # grams
SOLAR_Z						= 0.02
	
# Small number used to plot zeros on log graph
small_num = 1E-7
EPSILON = 1E-7


def set_labels(axis, xlabel, ylabel):
	"""
	Sets the labels and ticks for the given axis.

	Parameters
	----------
	axis : Matplotlib axis
	    Axis of plot
	xlabel : array-like
	    X axis label
	ylabel : array-like, optional
	    Y axis label

	Returns
	-------
	None
	"""
	axis.set_xlabel(xlabel, fontsize = Large_Font)
	axis.set_ylabel(ylabel, fontsize = Large_Font)
	axis.minorticks_on()
	axis.tick_params(axis='both',which='both',direction='in',right=True, top=True)
	axis.tick_params(axis='both', which='major', labelsize=Small_Font)
	axis.tick_params(axis='both', which='minor', labelsize=Small_Font)	


def calc_rotate_matrix(vec1, vec2):
	""""
	Gives the rotation matrix between two unit vectors
	"""
	a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
	v = np.cross(a, b)
	c = np.dot(a, b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
	return rotation_matrix



def weighted_percentile(a, percentiles=np.array([50, 16, 84]), weights=None):
	"""
	Calculates percentiles associated with a (possibly weighted) array

	Parameters
	----------
	a : array-like
	    The input array from which to calculate percents
	percentiles : array-like
	    The percentiles to calculate (0.0 - 100.0)
	weights : array-like, optional
	    The weights to assign to values of a.  Equal weighting if None
	    is specified

	Returns
	-------
	values : np.array
	    The values associated with the specified percentiles.  
	"""

	# First deal with empty array
	if len(a)==0:
		return np.full(len(percentiles), np.nan)

	# Standardize and sort based on values in a
	percentiles = percentiles
	if weights is None:
		weights = np.ones(a.size)
	idx = np.argsort(a)
	a_sort = a[idx]
	w_sort = weights[idx]

	# Get the percentiles for each data point in array
	p=1.*w_sort.cumsum()/w_sort.sum()*100
	# Get the value of a at the given percentiles
	values=np.interp(percentiles, p, a_sort)
	return values


def plot_observational_data(axis, param, log=True):
	"""
	Plots observational D/Z data vs the given param.

	Parameters
	----------
	axis : pyplot axis
		Axis on which to plot the data
	param: string
		Parameters to plot D/Z against (fH2, nH, Z, r, sigma_dust)
	log : boolean
		Plot on log scale

	Returns
	-------
	None

	"""
	if param == 'fH2':
		data = observations.Chiang_2020_DZ_vs_fH2(bin_data=True)
		for i, gal_name in enumerate(data.keys()):
			fH2_vals = data[gal_name][0]; mean_DZ = data[gal_name][1]; std_DZ = data[gal_name][2]
			if log:
				std_DZ[std_DZ == 0] = EPSILON
			axis.errorbar(fH2_vals, mean_DZ, yerr = np.abs(mean_DZ-np.transpose(std_DZ)), label=gal_name, c=Marker_Colors[i], fmt=Marker_Style[i], elinewidth=1, markersize=5)
	elif param == 'r':
		data = observations.Chiang_2020_dust_vs_radius(bin_data=True,DZ=True)
		for i, gal_name in enumerate(data.keys()):
			r_vals = data[gal_name][0]; mean_DZ = data[gal_name][1]; std_DZ = data[gal_name][2]
			if log:
				std_DZ[std_DZ == 0] = EPSILON
			axis.errorbar(r_vals, mean_DZ, yerr = np.abs(mean_DZ-np.transpose(std_DZ)), label=gal_name, c=Marker_Colors[i], fmt=Marker_Style[i], elinewidth=1, markersize=5)
	elif param == 'nH':
		dens_vals, DZ_vals = observations.Jenkins_2009_DZ_vs_dens(phys_dens=True)
		axis.scatter(dens_vals, DZ_vals, label='Jenkins09+phys_dens', c='xkcd:grey', marker=Marker_Style[0], s = 12)
		dens_vals, DZ_vals = observations.Jenkins_2009_DZ_vs_dens(phys_dens=False)
		axis.scatter(dens_vals, DZ_vals, label='Jenkins09', c='xkcd:grey', marker=Marker_Style[1], s = 12)
	elif param == 'dust':
		data = observations.Chiang_2020_DZ_vs_Sigma_dust(bin_data=True)
		for i, gal_name in enumerate(data.keys()):
			sigma_vals = data[gal_name][0]; mean_DZ = data[gal_name][1]; std_DZ = data[gal_name][2]
			if log:
				std_DZ[std_DZ == 0] = EPSILON
			axis.errorbar(sigma_vals, mean_DZ, yerr = np.abs(mean_DZ-np.transpose(std_DZ)), label=gal_name, c=Marker_Colors[i], fmt=Marker_Style[i], elinewidth=1, markersize=5)			
	elif param == 'Z':
		# TO DO: Add Remy-Ruyer D/Z vs Z observed data
		print "D/Z vs Z observations have not been implemented yet"



def DZ_vs_params(params, param_lims, gas, header, center_list, r_max_list, Lz_list=None, height_list=None, bin_nums=50, time=False, depletion=False, \
	          cosmological=True, labels=None, foutname='DZ_vs_param.png', std_bars=True, style='color', log=True, include_obs=True):
	"""
	Plots the average dust-to-metals ratio (D/Z) vs given parameters given code values of center and virial radius for multiple simulations/snapshots

	Parameters
	----------
	param: array
		Array of parameters to plot D/Z against (fH2, nH, Z, r)
	param_lims: array
		Limits for each parameter given in params
	gas : array
	    Array of snapshot gas data structures
	header : array
		Array of snapshot header structures
	center_list : array
		array of 3-D coordinate of center of circles
	r_max_list : array
		array of maximum radii
	Lz_list : array
		List of Lz unit vectors if selecting particles in disk
	height_list : array
		List of disk heights if applicable
	bin_nums : int
		Number of bins to use
	time : bool
		Print time in corner of plot (useful for movies)
	depletion: bool, optional
		Was the simulation run with the DEPLETION option
	cosmological : bool
		Is the simulation cosmological
	labels : array
		Array of labels for each data set
	foutname: str, optional
		Name of file to be saved
	std_bars : bool
		Include standard deviation bars for the data
	style : string
		Plotting style when plotting multiple data sets
		'color' - gives different color and linestyles to each data set
		'size' - make all lines solid black but with varying line thickness
	log : boolean
		Plot log of D/Z
	include_obs : boolean
		Overplot observed data if available

	Returns
	-------
	None
	"""	

	if len(gas) == 1:
		linewidths = np.full(len(gas),2)
		colors = ['xkcd:black' for i in range(len(gas))]
		linestyles = ['-' for i in range(len(gas))]
	elif style == 'color':
		linewidths = np.full(len(gas),2)
		colors = Line_Colors
		linestyles = Line_Styles
	elif style == 'size':
		linewidths = Line_Widths
		colors = ['xkcd:black' for i in range(len(gas))]
		linestyles = ['-' for i in range(len(gas))]
	else:
		print("Need to give a style when plotting more than one set of data. Currently 'color' and 'size' are supported.")
		return

	# Set up subplots based on number of parameters given
	if len(params) == 1:
		fig,axes = plt.subplots(1, 1, figsize=(14,10))
		axes = [axes]
	elif len(params)%2 == 0:
		fig,axes = plt.subplots(len(params)/2, 2, figsize=(24,len(params)/2*10))
	else:
		fig,axes = plt.subplots(1, 3, figsize=(3*14,10))

	for i, param in enumerate(params):
		# Set up for each plot
		axis = axes.flatten()[i]
		param_lim = param_lims[i]
		axis.set_xlim(param_lim)
		ylabel = 'D/Z Ratio'
		if param == 'fH2':
			xlabel = r'$f_{H2}$'
		elif param == 'r':
			xlabel = 'Radius (kpc)'
		elif param == 'nH':
			xlabel = r'$n_{H}$ (cm$^{-3}$)'
			axis.set_xscale('log')
		elif param == 'Z':
			xlabel = r'Z (Z$_{\odot}$)'
			axis.set_xscale('log')
		else:
			print("%s is not a valid parameter for DZ_vs_params()\n"%param)
			return()
		set_labels(axis,xlabel,ylabel)
		if log:
			axis.set_yscale('log')
			axis.set_ylim([0.01,1.0])
		else:
			axis.set_ylim([0,1])

		# First plot observational data if applicable
		if include_obs:
			plot_observational_data(axis, param, log=log)

		for j in range(len(gas)):
			G = gas[j]; H = header[j]; center = center_list[j]; r_max = r_max_list[j]; 
			if Lz_list != None:
				Lz_hat = Lz_list[j]; disk_height = height_list[j];
			else:
				Lz_hat = None; disk_height = None;

			mean_DZ,std_DZ,param_vals = calc_DZ_vs_param(param, param_lim, G, center, r_max, Lz_hat=Lz_hat, disk_height=disk_height, depletion=depletion)
			# Replace zeros with small values since we are taking the log of the values
			if log:
				std_DZ[std_DZ == 0] = EPSILON
				mean_DZ[mean_DZ == 0] = EPSILON

			# Only need to label the seperate simulations in the first plot
			if i==0:
				axis.plot(param_vals, mean_DZ, label=labels[j], linestyle=linestyles[j], color=colors[j], linewidth=linewidths[j])
			else:
				axis.plot(param_vals, mean_DZ, linestyle=linestyles[j], color=colors[j], linewidth=linewidths[j])
			if std_bars:
				axis.fill_between(param_vals, std_DZ[:,0], std_DZ[:,1], alpha = 0.4, color=colors[j])

		axis.legend(loc=0, fontsize=Small_Font, frameon=False)

	if time:
		if cosmological:
			z = H['redshift']
			axes[0].text(.05, .95, 'z = ' + '%.2g' % z, color="xkcd:black", fontsize = Large_Font, ha = 'left', transform=axes[0].transAxes)
		else:
			t = H['time']
			axes[0].text(.05, .95, 't = ' + '%2.2g Gyr' % t, color="xkcd:black", fontsize = Large_Font, ha = 'left', transform=axes[0].transAxes)		
	plt.savefig(foutname)
	plt.close()	


def calc_DZ_vs_param(param, param_lims, G, center, r_max, Lz_hat=None, disk_height=5, bin_nums=50, depletion=False):
	"""
	Calculate the average dust-to-metals ratio (D/Z) vs radius, density, and Z given code values of center and virial radius for multiple simulations/snapshots

	Parameters
	----------
	param: string
		Name of parameter to get D/Z values for
	G : dict
	    Snapshot gas data structure
	center : array
		3-D coordinate of center of circle
	r_max : double
		maximum radii of gas particles to use
	Lz_hat: array
		Unit vector of Lz to be used to mask only 
	disk_height: double
		Height of disk to mask if Lz_hat is given, default is 5 kpc
	bin_nums : int
		Number of bins to use
	depletion : bool, optional
		Was the simulation run with the DEPLETION option
	Returns
	-------
	mean_DZ : array
		Array of mean D/Z values vs parameter given
	std_DZ : array
		Array of 16th and 84th percentiles D/Z values
	param_vals : array
		Parameter values D/Z values are taken over
	"""	

	param_min = param_lims[0]; param_max = param_lims[1]; 

	coords = np.copy(G['p']) # Since we edit coords need to make a deep copy
	coords -= center
	# Get only data of particles in sphere/disk since those are the ones we care about
	# Also gives a nice speed-up
	# Get paticles in disk
	if Lz_hat != None:
		zmag = np.dot(coords,Lz_hat)
		r_z = np.zeros(np.shape(coords))
		r_z[:,0] = zmag*Lz_hat[0]
		r_z[:,1] = zmag*Lz_hat[1]
		r_z[:,2] = zmag*Lz_hat[2]
		r_s = np.subtract(coords,r_z)
		smag = np.sqrt(np.sum(np.power(r_s,2),axis=1))
		in_galaxy = np.logical_and(np.abs(zmag) <= disk_height, smag <= r_max)
	# Get particles in sphere otherwise
	else:
		in_galaxy = np.sum(np.power(coords,2),axis=1) <= np.power(r_max,2.)

	M = G['m'][in_galaxy]*1E10
	coords = coords[in_galaxy]
	if depletion:
		DZ = (G['dz'][:,0]/(G['z'][:,0]+G['dz'][:,0]))[in_galaxy]
	else:
		DZ = (G['dz'][:,0]/G['z'][:,0])[in_galaxy]

	mean_DZ = np.zeros(bin_nums - 1)
	# 16th and 84th percentiles
	std_DZ = np.zeros([bin_nums - 1,2])

	# Get D/Z values over number density of Hydrogen (nH)
	if param == 'nH':
		if depletion:
			nH = G['rho'][in_galaxy]*UnitDensity_in_cgs * ( 1. - (G['z'][:,0][in_galaxy]+G['z'][:,1]+G['dz'][:,0][in_galaxy])) / H_MASS
		else:
			nH = G['rho'][in_galaxy]*UnitDensity_in_cgs * ( 1. - (G['z'][:,0][in_galaxy]+G['z'][:,1][in_galaxy])) / H_MASS

		# Make bins for nH 
		nH_bins = np.logspace(np.log10(param_min),np.log10(param_max),bin_nums)
		param_vals = (nH_bins[1:] + nH_bins[:-1]) / 2.
		digitized = np.digitize(nH,nH_bins)

		for j in range(1,len(nH_bins)):
			if len(nH[digitized==j])==0:
				mean_DZ[j-1] = np.nan
				std_DZ[j-1,0] = np.nan; std_DZ[j-1,1] = np.nan;
				continue
			else:
				weights = M[digitized == j]
				values = DZ[digitized == j]
				mean_DZ[j-1],std_DZ[j-1,0],std_DZ[j-1,1] = weighted_percentile(values, weights=weights)

	# Get D/Z valus over radius of galaxy from the center
	elif param == 'r':
		r_bins = np.linspace(0, r_max, num=bin_nums)
		param_vals = (r_bins[1:] + r_bins[:-1]) / 2.

		for j in range(bin_nums-1):
			# find all coordinates within shell
			r_min = r_bins[j]; r_max = r_bins[j+1];

			# If disk get particles in annulus
			if Lz_hat!=None:
				zmag = np.dot(coords,Lz_hat)
				r_z = np.zeros(np.shape(coords))
				r_z[:,0] = zmag*Lz_hat[0]
				r_z[:,1] = zmag*Lz_hat[1]
				r_z[:,2] = zmag*Lz_hat[2]
				r_s = np.subtract(coords,r_z)
				smag = np.sqrt(np.sum(np.power(r_s,2),axis=1))
				in_shell = np.where((np.abs(zmag) <= disk_height) & (smag <= r_max) & (smag > r_min))
			# Else get particles in shell
			else:
				in_shell = np.logical_and(np.sum(np.power(coords,2),axis=1) <= np.power(r_max,2.), np.sum(np.power(coords,2),axis=1) > np.power(r_min,2.))
			weights = M[in_shell]
			values = DZ[in_shell]
			if len(values) > 0:
				mean_DZ[j],std_DZ[j,0],std_DZ[j,1] = weighted_percentile(values, weights=weights)
			else:
				mean_DZ[j] = np.nan
				std_DZ[j,0] = np.nan; std_DZ[j,1] = np.nan;

	# Get D/Z values vs total metallicty of gas
	elif param == 'Z':
		solar_Z = 0.02
		if depletion:
			Z = (G['z'][:,0]+G['dz'][:,0])[in_galaxy]/solar_Z
		else:
			Z = G['z'][:,0][in_galaxy]/solar_Z

		Z_bins = np.logspace(np.log10(param_min),np.log10(param_max),bin_nums)
		param_vals = (Z_bins[1:] + Z_bins[:-1]) / 2.
		digitized = np.digitize(Z,Z_bins)

		for j in range(1,len(Z_bins)):
			if len(Z[digitized==j])==0:
				mean_DZ[j-1] = np.nan
				std_DZ[j-1,0] = np.nan; std_DZ[j-1,1] = np.nan;
				continue
			else:
				weights = M[digitized == j]
				values = DZ[digitized == j]
				mean_DZ[j-1],std_DZ[j-1,0],std_DZ[j-1,1] = weighted_percentile(values, weights=weights)
	# Get D/Z values vs H2 mass fraction of gas
	elif param == 'fH2':
		NH1,NHion,NH2 = calc_H_fracs(G)
		fH2 = 2*NH2[in_galaxy]/(NH1[in_galaxy]+2*NH2[in_galaxy])
		fH2_bins = np.logspace(np.log10(param_min),np.log10(param_max),bin_nums)
		param_vals = (fH2_bins[1:] + fH2_bins[:-1]) / 2.
		digitized = np.digitize(fH2,fH2_bins)

		if depletion:
			nH = G['rho'][in_galaxy]*UnitDensity_in_cgs * ( 1. - (G['z'][:,0][in_galaxy]+G['z'][:,1]+G['dz'][:,0][in_galaxy])) / H_MASS
		else:
			nH = G['rho'][in_galaxy]*UnitDensity_in_cgs * ( 1. - (G['z'][:,0][in_galaxy]+G['z'][:,1][in_galaxy])) / H_MASS


		for j in range(1,len(fH2_bins)):
			if len(fH2[digitized==j])==0:
				mean_DZ[j-1] = np.nan
				std_DZ[j-1,0] = np.nan; std_DZ[j-1,1] = np.nan;
				continue
			else:
				weights = M[digitized == j]
				values = DZ[digitized == j]
				mean_DZ[j-1],std_DZ[j-1,0],std_DZ[j-1,1] = weighted_percentile(values, weights=weights)
	else:
		print("Parameter given to calc_DZ_vs_param is not supported:",param)
		return None,None,None

	return mean_DZ, std_DZ, param_vals


def observed_DZ_vs_param(params, param_lims, gas, header, center_list, r_max_list, Lz_list=None, \
			height_list=None, bin_nums=50, time=False, depletion=False, cosmological=True, labels=None, \
			foutname='obs_DZ_vs_param.png', std_bars=True, style='color', log=True, include_obs=True):
	"""
	Plots mock observations of dust-to-metals vs various parameters for multiple simulations 

	Parameters
	----------
	gas : array
	    Array of snapshot gas data structures
	header : array
		Array of snapshot header structures
	center_list : array
		array of 3-D coordinate of center of circles
	r_max_list : array
		array of maximum radii
	bin_nums : int
		Number of bins to use
	time : bool
		Print time in corner of plot (useful for movies)
	depletion: bool, optional
		Was the simulation run with the DEPLETION option
	cosmological : bool
		Is the simulation cosmological
	labels : array
		Array of labels for each data set
	foutname: str, optional
		Name of file to be saved
	std_bars : bool
		Include standard deviation bars for the data
	style : string
		Plotting style when plotting multiple data sets
		'color' - gives different color and linestyles to each data set
		'size' - make all lines solid black but with varying line thickness
	log : boolean
		Plot log of D/Z
	include_obs : boolean
		Overplot observed data if available

	Returns
	-------
	None
	"""	

	if len(gas) == 1:
		linewidths = np.full(len(gas),2)
		colors = ['xkcd:black' for i in range(len(gas))]
		linestyles = ['-' for i in range(len(gas))]
	elif style == 'color':
		linewidths = np.full(len(gas),2)
		colors = Line_Colors
		linestyles = Line_Styles
	elif style == 'size':
		linewidths = Line_Widths
		colors = ['xkcd:black' for i in range(len(gas))]
		linestyles = ['-' for i in range(len(gas))]
	else:
		print("Need to give a style when plotting more than one set of data. Currently 'color' and 'size' are supported.")
		return

	# Set up subplots based on number of parameters given
	if len(params) == 1:
		fig,axes = plt.subplots(1, 1, figsize=(14,10))
		axes = [axes]
	elif len(params)%2 == 0:
		fig,axes = plt.subplots(len(params)/2, 2, figsize=(24,len(params)/2*10))
	else:
		fig,axes = plt.subplots(1, 3, figsize=(3*14,10))

	for i, param in enumerate(params):
		# Set up for each plot
		axis = axes[i]
		param_lim = param_lims[i]
		axis.set_xlim(param_lim)
		ylabel = r'D/Z Ratio'
		if param == 'fH2':
			xlabel = r'$f_{H2}$'
		elif param == 'r':
			xlabel = 'Radius (kpc)'
		elif param == 'gas':
			xlabel = r'$\Sigma_{gas}$ (M$_{\odot}$ pc$^{-2}$)'
			axis.set_xscale('log')
		elif param == 'Z':
			xlabel = r'$\Sigma_{metals}$ (M$_{\odot}$ pc$^{-2}$)'
			axis.set_xscale('log')
		elif param == 'dust':
			xlabel = r'$\Sigma_{dust}$ (M$_{\odot}$ pc$^{-2}$)'
			axis.set_xscale('log')
		else:
			print("%s is not a valid parameter for dust_surface_dens_vs_param()\n"%param)
			return()
		set_labels(axis,xlabel,ylabel)
		if log:
			axis.set_yscale('log')
			axis.set_ylim([0.01,1.0])
		else:
			axis.set_ylim([0.,1.0])

		# First plot observational data if applicable
		if include_obs:
			plot_observational_data(axis, param, log=log)

		for j in range(len(gas)):
			G = gas[j]; H = header[j]; center = center_list[j]; r_max = r_max_list[j]; 
			if Lz_list != None:
				Lz_hat = Lz_list[j]; disk_height = height_list[j];
			else:
				Lz_hat = None; disk_height = None;

			mean_DZ,std_DZ,param_vals = calc_obs_DZ_vs_param(param, param_lim, G, center, r_max, Lz_hat=Lz_hat, disk_height=disk_height, depletion=depletion)
			# Replace zeros with small values since we are taking the log of the values
			if log:
				std_DZ[std_DZ == 0] = EPSILON
				mean_DZ[mean_DZ == 0] = EPSILON

			# Only need to label the seperate simulations in the first plot
			if i==0:
				axis.plot(param_vals, mean_DZ, label=labels[j], linestyle=linestyles[j], color=colors[j], linewidth=linewidths[j])
			else:
				axis.plot(param_vals, mean_DZ, linestyle=linestyles[j], color=colors[j], linewidth=linewidths[j])
			if std_bars:
				axis.fill_between(param_vals, std_DZ[:,0], std_DZ[:,1], alpha = 0.4, color=colors[j])

		axis.legend(loc=0, fontsize=Small_Font, frameon=False)

	if time:
		if cosmological:
			z = H['redshift']
			axes[0].text(.05, .95, 'z = ' + '%.2g' % z, color="xkcd:black", fontsize = Large_Font, ha = 'left', transform=axes[0].transAxes)
		else:
			t = H['time']
			axes[0].text(.05, .95, 't = ' + '%2.2g Gyr' % t, color="xkcd:black", fontsize = Large_Font, ha = 'left', transform=axes[0].transAxes)		
	plt.savefig(foutname)
	plt.close()	


def calc_obs_DZ_vs_param(param, param_lims, G, center, r_max, Lz_hat=None, disk_height=5, param_bins=50, pixel_res = 2, depletion=False):
	"""
	Calculate the average dust-to-metals ratio vs radius, gas , H2 , metal, or dust surface density
	given code values of center and viewing direction for multiple simulations/snapshots

	Parameters
	----------
	param: string
		Name of parameter to get D/Z values for
	G : dict
	    Snapshot gas data structure
	center : array
		3-D coordinate of center of circle
	r_max : double
		maximum radii of gas particles to use
	Lz_hat: array
		Unit vector of Lz to be used to mask only 
	disk_height: double
		Height of disk to mask if Lz_hat is given, default is 5 kpc
	param_bins : int
		Number of bins to use for physical param
	pixel_res : double
		Size resolution of each pixel bin in kpc
	depletion : bool, optional
		Was the simulation run with the DEPLETION option
	Returns
	-------
	mean_surf_dens : array
		Array of mean dust surface density values vs parameter given
	std_surf_dens : array
		Array of 16th and 84th percentiles dust surface density values
	param_vals : array
		Parameter values dust surface density values are taken over
	"""	

	param_min = param_lims[0]; param_max = param_lims[1]; 

	coords = np.copy(G['p']) # Since we edit coords need to make a deep copy
	coords -= center
	# Get only data of particles in sphere/disk since those are the ones we care about
	# Also gives a nice speed-up
	# Get paticles in disk
	if Lz_hat != None:
		zmag = np.dot(coords,Lz_hat)
		r_z = np.zeros(np.shape(coords))
		r_z[:,0] = zmag*Lz_hat[0]
		r_z[:,1] = zmag*Lz_hat[1]
		r_z[:,2] = zmag*Lz_hat[2]
		r_s = np.subtract(coords,r_z)
		smag = np.sqrt(np.sum(np.power(r_s,2),axis=1))
		in_galaxy = np.logical_and(np.abs(zmag) <= disk_height, smag <= r_max)
	# Get particles in sphere otherwise
	else:
		in_galaxy = np.sum(np.power(coords,2),axis=1) <= np.power(r_max,2.)

	NH1,NHion,NH2=calc_H_fracs(G)
	NH1=NH1[in_galaxy];NHion=NHion[in_galaxy];NH2=NH2[in_galaxy];
	M = G['m'][in_galaxy]*1E10
	coords = coords[in_galaxy]
	dust_mass = G['dz'][in_galaxy,0]*M
	if depletion:
		Z_mass = G['z'][in_galaxy,0] * M + dust_mass
	else:
		Z_mass = G['z'][in_galaxy,0] * M

	x = coords[:,0];y=coords[:,1];
	pixel_bins = int(np.ceil(2*r_max/pixel_res))
	x_bins = np.linspace(-r_max,r_max,pixel_bins)
	y_bins = np.linspace(-r_max,r_max,pixel_bins)
	x_vals = (x_bins[1:] + x_bins[:-1]) / 2.
	y_vals = (y_bins[1:] + y_bins[:-1]) / 2.
	pixel_area = pixel_res**2 * 1E6 # area of pixel in pc^2
	
	mean_DZ = np.zeros(param_bins - 1)
	std_DZ = np.zeros([param_bins - 1,2])


	if param == 'dust':
		ret = binned_statistic_2d(x, y, [Z_mass,dust_mass], statistic=np.sum, bins=[x_bins,y_bins]).statistic
		DZ_pixel = ret[1].flatten()/ret[0].flatten()
		dust_pixel = ret[1].flatten()/pixel_area

		# Now bin the data 
		dust_bins = np.logspace(np.log10(param_min),np.log10(param_max),param_bins)
		param_vals = (dust_bins[1:] + dust_bins[:-1]) / 2.
		digitized = np.digitize(dust_pixel,dust_bins)

		for j in range(1,len(dust_bins)):
			if len(dust_pixel[digitized==j])==0:
				mean_DZ[j-1] = np.nan
				std_DZ[j-1,0] = np.nan; std_DZ[j-1,1] = np.nan;
				continue
			else:
				values = DZ_pixel[digitized == j]
				mean_DZ[j-1],std_DZ[j-1,0],std_DZ[j-1,1] = weighted_percentile(values, weights=None)

	elif param=='gas':
		ret = binned_statistic_2d(x, y, [Z_mass,dust_mass,M], statistic=np.sum, bins=[x_bins,y_bins]).statistic
		DZ_pixel = ret[1].flatten()/ret[0].flatten()
		M_pixel = ret[2].flatten()/pixel_area

		# Now bin the data 
		gas_bins = np.logspace(np.log10(param_min),np.log10(param_max),param_bins)
		param_vals = (gas_bins[1:] + gas_bins[:-1]) / 2.
		digitized = np.digitize(M_pixel,gas_bins)

		for j in range(1,len(gas_bins)):
			if len(M_pixel[digitized==j])==0:
				mean_DZ[j-1] = np.nan
				std_DZ[j-1,0] = np.nan; std_DZ[j-1,1] = np.nan;
				continue
			else:
				values = DZ_pixel[digitized == j]
				mean_DZ[j-1],std_DZ[j-1,0],std_DZ[j-1,1] = weighted_percentile(values, weights=None)

	elif param == 'r':
		mean_DZ = np.zeros(pixel_bins/2 - 1)
		std_DZ = np.zeros([pixel_bins/2 - 1,2])
		ret = binned_statistic_2d(x, y, [Z_mass,dust_mass], statistic=np.sum, bins=[x_bins,y_bins],expand_binnumbers=True)
		DZ_pixel = ret.statistic[1].flatten()/ret.statistic[0].flatten()
		# Get the average r coordinate for each pixel in kpc
		pixel_r_vals = np.array([np.sqrt(np.power(np.abs(y_vals),2) + np.power(np.abs(x_vals[k]),2)) for k in range(len(x_vals))]).flatten()

		r_bins = np.linspace(0, r_max, num=pixel_bins/2)
		param_vals = (r_bins[1:] + r_bins[:-1]) / 2.

		for j in range(pixel_bins/2-1):
			# find all coordinates within shell
			r_min = r_bins[j]; r_max = r_bins[j+1];
			in_shell = np.where((pixel_r_vals <= r_max) & (pixel_r_vals > r_min))
			values = DZ_pixel[in_shell]
			if len(values) > 0:
				mean_DZ[j],std_DZ[j,0],std_DZ[j,1] = weighted_percentile(values, weights=None)
			else:
				mean_DZ[j] = np.nan
				std_DZ[j,0] = np.nan; std_DZ[j,1] = np.nan;

	elif param == 'fH2':
		MH1=NH1*H_MASS/UnitMass_in_Msolar
		MH2=2*NH2*H_MASS/UnitMass_in_Msolar
		ret = binned_statistic_2d(x, y, [Z_mass,dust_mass,MH2,MH1], statistic=np.sum, bins=[x_bins,y_bins]).statistic
		DZ_pixel = ret[1].flatten()/ret[0].flatten()
		fH2_pixel = ret[2].flatten()/(ret[2].flatten()+ret[3].flatten())

		# Now bin the data 
		fH2_bins = np.linspace(param_min,param_max,param_bins)
		param_vals = (fH2_bins[1:] + fH2_bins[:-1]) / 2.
		digitized = np.digitize(fH2_pixel,fH2_bins)

		for j in range(1,len(fH2_bins)):
			if len(fH2_pixel[digitized==j])==0:
				mean_DZ[j-1] = np.nan
				std_DZ[j-1,0] = np.nan; std_DZ[j-1,1] = np.nan;
				continue
			else:
				values = DZ_pixel[digitized == j]
				mean_DZ[j-1],std_DZ[j-1,0],std_DZ[j-1,1] = weighted_percentile(values, weights=None)

	elif param == 'Z':
		ret = binned_statistic_2d(x, y, [Z_mass,dust_mass], statistic=np.sum, bins=[x_bins,y_bins]).statistic
		DZ_pixel = ret[1].flatten()/ret[0].flatten()
		Z_pixel = ret[0].flatten()/pixel_area

		# Now bin the data 
		Z_bins = np.logspace(np.log10(param_min),np.log10(param_max),param_bins)
		param_vals = (Z_bins[1:] + Z_bins[:-1]) / 2.
		digitized = np.digitize(Z_pixel,Z_bins)

		for j in range(1,len(Z_bins)):
			if len(Z_pixel[digitized==j])==0:
				mean_DZ[j-1] = np.nan
				std_DZ[j-1,0] = np.nan; std_DZ[j-1,1] = np.nan;
				continue
			else:
				values = DZ_pixel[digitized == j]
				mean_DZ[j-1],std_DZ[j-1,0],std_DZ[j-1,1] = weighted_percentile(values, weights=None)
	else:
		print("Parameter given to calc_dust_dens_vs_param is not supported:",param)
		return None,None,None

	return mean_DZ, std_DZ, param_vals



def calc_H_fracs(G):
	# Analytic calculation of molecular hydrogen from Krumholz et al. (2018)


	Z = G['z'][:,0] #metal mass (everything not H, He)
	# dust mean mass per H nucleus
	mu_H = 2.3E-24# grams
	# standard effective number of particle kernel neighbors defined in parameters file
	N_ngb = 32.
	# Gas softening length
	hsml = G['h']*UnitLength_in_cm
	density = G['rho']*UnitDensity_in_cgs

	sobColDens = np.multiply(hsml,density) / np.power(N_ngb,1./3.) # Cheesy approximation of column density

	#  dust optical depth 
	tau = np.multiply(sobColDens,Z*1E-21/SOLAR_Z)/mu_H
	tau[tau==0]=EPSILON #avoid divide by 0

	chi = 3.1 * (1+3.1*np.power(Z/SOLAR_Z,0.365)) / 4.1 # Approximation

	s = np.divide( np.log(1+0.6*chi+0.01*np.power(chi,2)) , (0.6 *tau) )
	s[s==-4.] = -4.+EPSILON # Avoid divide by zero
	fH2 = np.divide((1 - 0.5*s) , (1+0.25*s)) # Fraction of Molecular Hydrogen from Krumholz & Knedin
	fH2[fH2<0] = 0 #Nonphysical negative molecular fractions set to 0

	fHe = G['z'][:,1]
	fMetals = G['z'][:,0]

	# NH1 = Mass * Fraction of Hydrogen * Fraction of Hydrogen that is Neutral * Fraction of Neutral Hydrogen that is HI / Mass_HI
	NH1 =  G['m'] * UnitMass_in_g * (1. - fHe - fMetals) * (G['nh']) * (1. - fH2) / H_MASS
	# NH2=  Mass * Fraction of Hydrogen * Fraction of Hydrogen that is Neutral * Fraction of Neutral Hydrogen that is H2 / Mass_HI
	NH2 =  G['m'] * UnitMass_in_g * (1. - fHe - fMetals) * (G['nh']) * fH2 / (2*H_MASS) #Gives Number of H2 molecules
	#NHion=Mass * Fraction of Hydrogen * Fraction of Ionized Hydrogen / Mass 
	NHion= G['m'] * UnitMass_in_g * (1. - fHe - fMetals) * (1.-G['nh']) / H_MASS
	
	return NH1,NHion,NH2




def nH_vs_fH2(gas, header, center_list, r_max_list,  Lz_list=None, height_list=None, foutname='nH_vs_fH2.png', style='color', labels=None, bin_nums=100):
	if len(gas) == 1:
		linewidths = np.full(len(gas),2)
		colors = Line_Colors
		linestyles = Line_Styles
	elif style == 'color':
		linewidths = np.full(len(gas),2)
		colors = Line_Colors
		linestyles = Line_Styles
	elif style == 'size':
		linewidths = Line_Widths
		colors = ['xkcd:black' for i in range(len(gas))]
		linestyles = ['-' for i in range(len(gas))]
	else:
		print("Need to give a style when plotting more than one set of data. Currently 'color' and 'size' are supported.")
		return

	plt.figure()
	axis = plt.gca()

	for i in range(len(gas)):
		G = gas[i]; H = header[i]; center = center_list[i]; r_max = r_max_list[i];
		if Lz_list != None:
			Lz_hat = Lz_list[i]; disk_height = height_list[i];
		else:
			Lz_hat = None; disk_height = None;

		coords = np.copy(G['p']) # Since we edit coords need to make a deep copy
		coords -= center
		# Get only data of particles in sphere/disk since those are the ones we care about
		# Also gives a nice speed-up
		# Get paticles in disk
		if Lz_hat != None:
			zmag = np.dot(coords,Lz_hat)
			r_z = np.zeros(np.shape(coords))
			r_z[:,0] = zmag*Lz_hat[0]
			r_z[:,1] = zmag*Lz_hat[1]
			r_z[:,2] = zmag*Lz_hat[2]
			r_s = np.subtract(coords,r_z)
			smag = np.sqrt(np.sum(np.power(r_s,2),axis=1))
			in_galaxy = np.logical_and(np.abs(zmag) <= disk_height, smag <= r_max)
		# Get particles in sphere otherwise
		else:
			in_galaxy = np.sum(np.power(coords,2),axis=1) <= np.power(r_max,2.)

		M = G['m'][in_galaxy]
		nH = G['rho'][in_galaxy]*UnitDensity_in_cgs * ( 1. - (G['z'][:,0][in_galaxy]+G['z'][:,1][in_galaxy])) / H_MASS

		NH1,NHion,NH2 = calc_H_fracs(G)
		fH2 = 2*NH2[in_galaxy]/(NH1[in_galaxy]+2*NH2[in_galaxy])

		mean_fH2 = np.zeros(bin_nums - 1)
		# 16th and 84th percentiles
		std_fH2 = np.zeros([bin_nums - 1,2])
		
		nH_bins = np.logspace(np.log10(1E-2),np.log10(1E3),bin_nums)
		nH_vals = (nH_bins[1:] + nH_bins[:-1]) / 2.
		digitized = np.digitize(nH,nH_bins)

		for j in range(1,len(nH_bins)):
			if len(nH[digitized==j])==0:
				mean_fH2[j-1] = np.nan
				std_fH2[j-1,0] = np.nan; std_fH2[j-1,1] = np.nan;
				continue
			else:
				weights = M[digitized == j]
				values = fH2[digitized == j]
				mean_fH2[j-1],std_fH2[j-1,0],std_fH2[j-1,1] = weighted_percentile(values, weights=weights)


		plt.plot(nH_vals, mean_fH2, label=labels[i], linestyle=linestyles[i], color=colors[i], linewidth=linewidths[i])
		plt.fill_between(nH_vals, std_fH2[:,0], std_fH2[:,1], alpha = 0.4, color=colors[i])


	plt.xscale('log')
	plt.yscale('log')
	plt.ylim([0.01,1])
	plt.xlim([1E-2,1E3])
	xlabel=r'$n_H (cm^{-3})$'; ylabel=r'$f_{H2}$';
	set_labels(axis, xlabel, ylabel)
	plt.savefig(foutname)




def inst_dust_prod(gas, header, center_list, r_max_list,  Lz_list=None, height_list=None, bin_nums=100, time=False, depletion=False, log = False, \
	           cosmological=True, Tmin=1, Tmax=1E5, Tcut=300, labels=None, foutname='inst_dust_prod.png', style='color', implementation='species', \
	           t_ref_factors = None):
	"""
	Make plot of instantaneous dust growth for a given snapshot depending on the dust evolution implementation used

	Parameters
	----------
	gas : array
	    Array of snapshot gas data structure
	header : array
		Array of snapshot header structure
	bin_nums: int
		Number of bins to use
	time : bool, optional
		Print time in corner of plot (useful for movies)
	style : string
		Plotting style when plotting multiple data sets
		'color' - gives different color and linestyles to each data set
		'size' - make all lines solid black but with varying line thickness

	Returns
	-------
	None
	"""
	if len(gas) == 1:
		linewidths = np.full(len(gas),2)
		colors = Line_Colors
		linestyles = Line_Styles
	elif style == 'color':
		linewidths = np.full(len(gas),2)
		colors = Line_Colors
		linestyles = Line_Styles
	elif style == 'size':
		linewidths = Line_Widths
		colors = ['xkcd:black' for i in range(len(gas))]
		linestyles = ['-' for i in range(len(gas))]
	else:
		print("Need to give a style when plotting more than one set of data. Currently 'color' and 'size' are supported.")
		return
	
	fig,axes = plt.subplots(1, 1, figsize=(12,10))

	for i in range(len(gas)):
		if t_ref_factors != None:
			t_ref_factor = t_ref_factors[i]
		else:
			t_ref_factor = 1.

		if isinstance(implementation, list):
			imp = implementation[i]
		else:
			imp = implementation

		G = gas[i]; H = header[i]; center = center_list[i]; r_max = r_max_list[i];
		if Lz_list != None:
			Lz_hat = Lz_list[i]; disk_height = height_list[i];
		else:
			Lz_hat = None; disk_height = None;

		coords = np.copy(G['p']) # Since we edit coords need to make a deep copy
		coords -= center
		# Get only data of particles in sphere/disk since those are the ones we care about
		# Also gives a nice speed-up
		# Get paticles in disk
		if Lz_hat != None:
			zmag = np.dot(coords,Lz_hat)
			r_z = np.zeros(np.shape(coords))
			r_z[:,0] = zmag*Lz_hat[0]
			r_z[:,1] = zmag*Lz_hat[1]
			r_z[:,2] = zmag*Lz_hat[2]
			r_s = np.subtract(coords,r_z)
			smag = np.sqrt(np.sum(np.power(r_s,2),axis=1))
			in_galaxy = np.logical_and(np.abs(zmag) <= disk_height, smag <= r_max)
		# Get particles in sphere otherwise
		else:
			in_galaxy = np.sum(np.power(coords,2),axis=1) <= np.power(r_max,2.)


		T = gas_temp.gas_temperature(G)
		T = T[in_galaxy]
		M = G['m'][in_galaxy]

		if depletion:
			nH = G['rho'][in_galaxy]*UnitDensity_in_cgs * ( 1. - (G['z'][:,0][in_galaxy]+G['z'][:,1]+G['dz'][:,0][in_galaxy])) / H_MASS
		else:
			nH = G['rho'][in_galaxy]*UnitDensity_in_cgs * ( 1. - (G['z'][:,0][in_galaxy]+G['z'][:,1][in_galaxy])) / H_MASS


		if depletion:
			DZ = (G['dz'][:,0]/(G['z'][:,0]+G['dz'][:,0]))[in_galaxy]
		else:
			DZ = (G['dz'][:,0]/G['z'][:,0])[in_galaxy]

		if imp == 'species':
			# First get silicates
			t_ref = 2.45E6*t_ref_factor;
			dust_formula_mass = 0.0
			atomic_mass = np.array([1.01, 2.0, 12.01, 14, 15.99, 20.2, 24.305, 28.086, 32.065, 40.078, 55.845])
			elem_num_dens = np.multiply(G['z'][in_galaxy,:len(atomic_mass)], G['rho'][in_galaxy, np.newaxis]*UnitDensity_in_cgs) / (atomic_mass*H_MASS)
			sil_elems_index = np.array([4,6,7,10]) # O,Mg,Si,Fe
			# number of atoms that make up one formula unit of silicate dust assuming an olivine, pyroxene mixture
			# with olivine fraction of 0.32 and Mg fraction of 0.8
			sil_num_atoms = np.array([3.63077,1.06,1.,0.570769]) # O, Mg, Si, Fe


			for k in range(4): dust_formula_mass += sil_num_atoms[k] * atomic_mass[sil_elems_index[k]];
			sil_num_dens = elem_num_dens[:,sil_elems_index] 
			# Find element with lowest number density factoring in number of atoms needed to make one formula unit of the dust species
			key = np.argmin(sil_num_dens / sil_num_atoms, axis = 1)
			key_num_dens = sil_num_dens[range(sil_num_dens.shape[0]),key]
			key_elem = sil_elems_index[key]
			key_DZ = G['dz'][in_galaxy,key_elem]/G['z'][in_galaxy,key_elem]
			key_M_dust = G['dz'][in_galaxy,key_elem]*M*1E10
			key_in_dust = sil_num_atoms[key]
			key_mass = atomic_mass[key_elem]
			cond_dens = 3.13
			growth_time = t_ref * key_in_dust * np.sqrt(key_mass) / dust_formula_mass * (cond_dens/3) * (1E-2/key_num_dens) * np.power(300/T,0.5)
			sil_dust_prod = (1.-key_DZ)*key_M_dust/growth_time
			sil_dust_prod[np.logical_or(key_DZ <= 0,key_DZ >= 1)] = 0.
			sil_dust_prod *= dust_formula_mass / atomic_mass[key_elem]
			sil_dust_prod[T>Tcut] = 0.

			# Now carbon dust
			CO_frac = 0.2; # Fraction of C locked in CO
			t_ref = 2.45E6*t_ref_factor
			key_elem = 2
			key_DZ = G['dz'][in_galaxy,key_elem]/((1-CO_frac)*G['z'][in_galaxy,key_elem])
			key_M_dust = G['dz'][in_galaxy,key_elem]*M*1E10
			key_in_dust = 1
			key_mass = atomic_mass[key_elem]
			dust_formula_mass = key_mass
			cond_dens = 2.25
			key_num_dens = elem_num_dens[:,key_elem]*(1-CO_frac)
			growth_time = t_ref * key_in_dust * np.sqrt(key_mass) / dust_formula_mass * (cond_dens/3) * (1E-2/key_num_dens) * np.power(300/T,0.5)
			carbon_dust_prod = (1.-key_DZ)*key_M_dust/growth_time
			carbon_dust_prod[np.logical_or(key_DZ <= 0,key_DZ >= 1)] = 0.
			carbon_dust_prod[T>Tcut] = 0.

			# Now iron dust
			t_ref = 2.45E6*t_ref_factor
			key_elem = 10
			key_DZ = G['dz'][in_galaxy,key_elem]/G['z'][in_galaxy,key_elem]
			key_M_dust = G['dz'][in_galaxy,key_elem]*M*1E10
			key_in_dust = 1
			dust_formulat_mass = atomic_mass[key_elem]
			cond_dens = 7.86
			key_num_dens = elem_num_dens[:,key_elem]
			key_mass = atomic_mass[key_elem]
			growth_time = t_ref * key_in_dust * np.sqrt(key_mass) / dust_formula_mass * (cond_dens/3) * (1E-2/key_num_dens) * np.power(300/T,0.5)
			iron_dust_prod = (1.-key_DZ)*key_M_dust/growth_time
			iron_dust_prod[np.logical_or(key_DZ <= 0,key_DZ >= 1)] = 0.
			iron_dust_prod[T>Tcut] = 0.


		# Elemental implementation
		else:
			t_ref = 0.2E9*t_ref_factor; T_ref = 20; dens_ref = H_MASS;
			dens = G['rho'][in_galaxy]*UnitDensity_in_cgs
			growth_time = t_ref * (dens_ref/dens) * np.power(T_ref/T,0.5)
			sil_DZ = G['dz'][:,[4,6,7,10]][in_galaxy]/G['z'][:,[4,6,7,10]][in_galaxy]
			sil_DZ[np.logical_or(sil_DZ <= 0,sil_DZ >= 1)] = 1.
			sil_dust_mass = np.multiply(G['dz'][:,[4,6,7,10]][in_galaxy],M[:,np.newaxis]*1E10)
			sil_dust_prod = np.sum((1.-sil_DZ)*sil_dust_mass/growth_time[:,np.newaxis],axis=1)
			CO_frac = 0.2; # Fraction of C locked in CO
			C_DZ = G['dz'][:,2][in_galaxy]/((1-CO_frac)*G['z'][:,2][in_galaxy])
			C_dust_mass = G['dz'][:,2][in_galaxy]*M*1E10
			carbon_dust_prod = (1.-C_DZ)*C_dust_mass/growth_time
			carbon_dust_prod[np.logical_or(C_DZ <= 0,C_DZ >= 1)] = 0.
			iron_dust_prod = np.zeros(len(sil_dust_prod))


		axes.hist(nH, bins=np.logspace(np.log10(1E-2),np.log10(1E3),bin_nums), weights=sil_dust_prod, histtype='step', cumulative=True, label=labels[i], color=colors[i], \
			         linewidth=linewidths[0], linestyle=linestyles[0])
		axes.hist(nH, bins=np.logspace(np.log10(1E-2),np.log10(1E3),bin_nums), weights=carbon_dust_prod, histtype='step', cumulative=True, label=labels[i], color=colors[i], \
			         linewidth=linewidths[0], linestyle=linestyles[1])
		axes.hist(nH, bins=np.logspace(np.log10(1E-2),np.log10(1E3),bin_nums), weights=iron_dust_prod, histtype='step', cumulative=True, label=labels[i], color=colors[i], \
			         linewidth=linewidths[0], linestyle=linestyles[2])

	lines = []
	for i in range(len(labels)):
		lines += [mlines.Line2D([], [], color=colors[i], label=labels[i])]
	lines += [mlines.Line2D([], [], color='xkcd:black', linestyle =linestyles[0],label='Silicate'), \
		      mlines.Line2D([], [], color='xkcd:black', linestyle =linestyles[1],label='Carbon'), \
		      mlines.Line2D([], [], color='xkcd:black', linestyle =linestyles[2],label='Iron')]

	axes.legend(handles=lines,loc=2, frameon=False)
	axes.set_xlim([1E-2,1E3])
	xlabel = r'$n_H$ (cm$^{-3}$)'; ylabel = r'Cumulative Inst. Dust Prod. $(M_{\odot}/yr)$'
	set_labels(axes,xlabel,ylabel)
	if log:
		axes.set_yscale('log')
		axes.set_ylim([1E-3,1E1])
	axes.set_xscale('log')

	plt.savefig(foutname)
	plt.close()


"""
def accretion_analysis_plot(gas, header, center_list, r_max_list,  Lz_list=None, height_list=None, bin_nums=100, time=False, depletion=False, \
	           cosmological=True, Tmin=1, Tmax=1E5, Tcut=300, labels=None, foutname='accretion_analysis.png',  style='color', implementation='species'):
	if len(gas) == 1:
		linewidths = np.full(len(gas),2)
		colors = Line_Colors
		linestyles = Line_Styles
	elif style == 'color':
		linewidths = np.full(len(gas),2)
		colors = Line_Colors
		linestyles = Line_Styles
	elif style == 'size':
		linewidths = Line_Widths
		colors = ['xkcd:black' for i in range(len(gas))]
		linestyles = ['-' for i in range(len(gas))]
	else:
		print("Need to give a style when plotting more than one set of data. Currently 'color' and 'size' are supported.")
		return
	
	fig,axes = plt.subplots(1, 2, figsize=(24,10))

	for i in range(len(gas)):
		if isinstance(implementation, list):
			imp = implementation[i]
		else:
			imp = implementation

		G = gas[i]; H = header[i]; center = center_list[i]; r_max = r_max_list[i];
		if Lz_list != None:
			Lz_hat = Lz_list[i]; disk_height = height_list[i];
		else:
			Lz_hat = None; disk_height = None;

		coords = np.copy(G['p']) # Since we edit coords need to make a deep copy
		coords -= center
		# Get only data of particles in sphere/disk since those are the ones we care about
		# Also gives a nice speed-up
		# Get paticles in disk
		if Lz_hat != None:
			zmag = np.dot(coords,Lz_hat)
			r_z = np.zeros(np.shape(coords))
			r_z[:,0] = zmag*Lz_hat[0]
			r_z[:,1] = zmag*Lz_hat[1]
			r_z[:,2] = zmag*Lz_hat[2]
			r_s = np.subtract(coords,r_z)
			smag = np.sqrt(np.sum(np.power(r_s,2),axis=1))
			in_galaxy = np.logical_and(np.abs(zmag) <= disk_height, smag <= r_max)
		# Get particles in sphere otherwise
		else:
			in_galaxy = np.sum(np.power(coords,2),axis=1) <= np.power(r_max,2.)


		T = gas_temp.gas_temperature(G)
		T = T[in_galaxy]
		M = G['m'][in_galaxy]

		if depletion:
			nH = G['rho'][in_galaxy]*UnitDensity_in_cgs * ( 1. - (G['z'][:,0][in_galaxy]+G['z'][:,1]+G['dz'][:,0][in_galaxy])) / H_MASS
		else:
			nH = G['rho'][in_galaxy]*UnitDensity_in_cgs * ( 1. - (G['z'][:,0][in_galaxy]+G['z'][:,1][in_galaxy])) / H_MASS


		if depletion:
			DZ = (G['dz'][:,0]/(G['z'][:,0]+G['dz'][:,0]))[in_galaxy]
		else:
			DZ = (G['dz'][:,0]/G['z'][:,0])[in_galaxy]

		if imp == 'species':
			t_ref = 4.24E6; sil_dust_mass = 143.78; Mg_mass = 24.305; Mg_in_sil = 1.06;
			n_Mg = G['z'][:,6][in_galaxy] * G['rho'][in_galaxy]*UnitDensity_in_cgs/(Mg_mass*H_MASS);
			Mg_DZ = G['dz'][:,6][in_galaxy]/G['z'][:,6][in_galaxy]
			Md = G['dz'][:,6][in_galaxy]*M*1E10
			growth_time = t_ref * Mg_in_sil * np.sqrt(Mg_mass) / sil_dust_mass * (3.13/3) * (1E-2/(n_Mg)) * np.power(300/T,0.5)
			dust_prod = (1.-Mg_DZ)*Md/growth_time
			dust_prod *= sil_dust_mass / Mg_mass
		else:
			t_ref = 0.2; T_ref = 20; nH_ref = 1.;
			growth_time = t_ref * (nH_ref/nH) * np.power(T_ref/T,0.5)
			Mg_DZ = G['dz'][:,6][in_galaxy]/G['z'][:,6][in_galaxy]
			Md = G['dz'][:,6][in_galaxy]*M*1E10
			dust_prod = (1.-Mg_DZ)*Md/growth_time
		dust_prod[dust_prod<0] = 0 
		# No dust growth if no dust
		dust_prod[G['dz'][:,6][in_galaxy]==0] = 0;


		hot_gas = T>Tcut
		axes[0].hist(np.log10(nH), bin_nums, weights=dust_prod, range=[np.log10(1E-2),np.log10(1E3)], histtype='step', cumulative=True, label="All Gas", color=colors[0], \
			         linewidth=linewidths[0])
		axes[0].hist(np.log10(nH[np.logical_not(hot_gas)]), bin_nums, weights=dust_prod[np.logical_not(hot_gas)], range=[np.log10(1E-2),np.log10(1E3)], histtype='step', \
			         cumulative=True, label="T < %i K" % Tcut, color=colors[1],linewidth=linewidths[0])
		axes[0].hist(np.log10(nH[hot_gas]), bin_nums, weights=dust_prod[hot_gas], range=[np.log10(1E-2),np.log10(1E3)], histtype='step', cumulative=True, \
			         label="T > %i K" % Tcut, color=colors[2],linewidth=linewidths[0])
		axes[0].legend(loc=2)
		xlabel = r'Log $n_H$ (cm$^{-3}$)'; ylabel = r'Cumulative Inst. Dust Prod. $(M_{\odot,sil}/yr)$'
		set_labels(axes[0],xlabel,ylabel)

		nH_lims = [0.1,1,10,100]
		names = [r'$n_H>0.1$ cm$^{-3}$ ',r'$n_H>1$ cm$^{-3}$    ',r'$n_H>10$ cm$^{-3}$  ',r'$n_H>100$ cm$^{-3}$']
		for j,nH_lim in enumerate(nH_lims):
			mask = nH>=nH_lim
			frac = 1.0*len(nH[mask])/len(nH)
			names[j] += r' $f_{gas}$ = %.2g' % frac
			axes[1].hist(np.log10(T[mask]), bin_nums, range=[np.log10(Tmin),np.log10(Tmax)], density=True, histtype='step', cumulative=True, label=names[j], \
				       linestyle=linestyles[j], color=colors[j], linewidth=linewidths[0], weights = M[mask])
			axes[1].axvline(x=np.log10(Tcut))
		xlabel = r'Log T (K)'; ylabel = "Fraction of Gas < T"
		set_labels(axes[1],xlabel,ylabel)	
		axes[1].legend(loc=2)


		axes[2].hist(np.log10(growth_time/1E9), bin_nums, weights=M, range=[np.log10(1E-6),np.log10(1E1)], histtype='step', cumulative=True, density=True, \
					 linewidth=linewidths[0], color=colors[0], label='All Gas')
		axes[2].hist(np.log10(growth_time[np.logical_not(hot_gas)]/1E9), bin_nums, weights=M[np.logical_not(hot_gas)], range=[np.log10(1E-6),np.log10(1E1)], histtype='step', cumulative=True, density=True, \
					 linewidth=linewidths[0], color=colors[1], label="T < %i K" % Tcut)
		axes[2].hist(np.log10(growth_time[hot_gas]/1E9), bin_nums, weights=M[hot_gas], range=[np.log10(1E-6),np.log10(1E1)], histtype='step', cumulative=True, density=True, \
					 linewidth=linewidths[0], color=colors[2], label="T > %i K" % Tcut)
		xlabel =r' $\tau_{grow}$ (Gyr)'; ylabel = r'Fraction of Gas < $\tau_{grow}$'
		set_labels(axes[2],xlabel,ylabel)	
		axes[2].legend(loc=2)
		
		plt.savefig(labels[i]+'_'+foutname)
		plt.close()
"""




def binned_phase_plot(param, gas, header, center_list, r_max_list, Lz_list=None, height_list=None, bin_nums=100, time=False, depletion=False, cosmological=True, \
			   nHmin=1E-3, nHmax=1E3, Tmin=1E1, Tmax=1E5, numbins=200, thecmap='hot', vmin=1E-8, vmax=1E-4, labels =None, log=False, foutname='phase_plot.png'):
	"""
	Plots the temperate-density has phase

	Parameters
	----------
	param: string
		What parameterto bin 2d-historgram over
	gas : array
	    Array of snapshot gas data structure
	header : array
		Array of snapshot header structure
	bin_nums: int
		Number of bins to use
	depletion: bool, optional
		Was the simulation run with the DEPLETION option

	Returns
	-------
	None
	"""

	for i in range(len(gas)):
		G = gas[i]; H = header[i]; center = center_list[i]; r_max = r_max_list[i];
		if Lz_list != None:
			Lz_hat = Lz_list[i]; disk_height = height_list[i];
		else:
			Lz_hat = None; disk_height = None;

		coords = np.copy(G['p']) # Since we edit coords need to make a deep copy
		coords -= center
		# Get only data of particles in sphere/disk since those are the ones we care about
		# Also gives a nice speed-up
		# Get paticles in disk
		if Lz_hat != None:
			zmag = np.dot(coords,Lz_hat)
			r_z = np.zeros(np.shape(coords))
			r_z[:,0] = zmag*Lz_hat[0]
			r_z[:,1] = zmag*Lz_hat[1]
			r_z[:,2] = zmag*Lz_hat[2]
			r_s = np.subtract(coords,r_z)
			smag = np.sqrt(np.sum(np.power(r_s,2),axis=1))
			in_galaxy = np.logical_and(np.abs(zmag) <= disk_height, smag <= r_max)
		# Get particles in sphere otherwise
		else:
			in_galaxy = np.sum(np.power(coords,2),axis=1) <= np.power(r_max,2.)

		if param == 'DZ':
			if depletion:
				values = (G['dz'][:,0]/(G['z'][:,0]+G['dz'][:,0]))[in_galaxy]
			else:
				values = (G['dz'][:,0]/G['z'][:,0])[in_galaxy]
			bar_label = 'D/Z Ratio in Pixel'
		else:
			print("Parameter given to binned_phase_plot is not supported:",param)
			return

		if depletion:
			nH = np.log10(G['rho']*UnitDensity_in_cgs * ( 1. - (G['z'][:,0]+G['z'][:,1]+G['dz'][:,0])) / H_MASS)[in_galaxy]
		else:
			nH = np.log10(G['rho']*UnitDensity_in_cgs * ( 1. - (G['z'][:,0]+G['z'][:,1])) / H_MASS)[in_galaxy]
		T = np.log10(gas_temp.gas_temperature(G))
		T = T[in_galaxy]
		M = G['m'][in_galaxy]

		# Bin data across nH and T parameter space
		nH_bins = np.linspace(np.log10(nHmin), np.log10(nHmax), numbins)
		T_bins = np.linspace(np.log10(Tmin), np.log10(Tmax), numbins)
		ret = binned_statistic_2d(nH, T, values, statistic=np.mean, bins=[nH_bins, T_bins])

		fig = plt.figure()
		ax = plt.gca()
		plt.subplot(111, facecolor='xkcd:black')
		if log:
			plt.imshow(ret.statistic.T, origin='bottom', cmap=plt.get_cmap(thecmap), norm=mpl.colors.LogNorm(), vmin=vmin, vmax=vmax, extent=[np.log10(nHmin),np.log10(nHmax),np.log10(Tmin),np.log10(Tmax)])
			#plt.hist2d(nH, T, range=np.log10([[nHmin,nHmax],[Tmin,Tmax]]), bins=numbins, cmap=plt.get_cmap(thecmap), norm=mpl.colors.LogNorm(), weights=weight, vmin=vmin, vmax=vmax) 
		else:
			plt.imshow(ret.statistic.T, origin='bottom', cmap=plt.get_cmap(thecmap), vmin=vmin, vmax=vmax, extent=[np.log10(nHmin),np.log10(nHmax),np.log10(Tmin),np.log10(Tmax)])
			#plt.hist2d(nH, T, range=np.log10([[nHmin,nHmax],[Tmin,Tmax]]), bins=numbins, cmap=plt.get_cmap(thecmap), weights=weight, vmin=vmin, vmax=vmax) 
		cbar = plt.colorbar()
		cbar.ax.set_ylabel(bar_label, fontsize=Large_Font)


		plt.xlabel(r'log $n_{H} ({\rm cm}^{-3})$', fontsize=Large_Font) 
		plt.ylabel(r'log T (K)', fontsize=Large_Font)
		plt.tight_layout()
		if time:
			if cosmological:
				z = H['redshift']
				ax.text(.95, .95, 'z = ' + '%.2g' % z, color="xkcd:white", fontsize = 16, ha = 'right', transform=ax.transAxes)
			else:
				t = H['time']
				ax.text(.95, .95, 't = ' + '%2.1g Gyr' % t, color="xkcd:white", fontsize = 16, ha = 'right', transform=ax.transAxes)	
		plt.savefig(labels[i]+'_'+foutname)
		plt.close()



def DZ_vs_time(dataname='data.pickle', data_dir='data/', foutname='DZ_vs_time.png', time=True, cosmological=True, log=True):
	"""
	Plots the average dust-to-metals ratio (D/Z) vs time from precompiled data

	Parameters
	----------
	dataname : str
		Name of data file
	datadir: str
		Directory of data
	foutname : str
		Name of file to be saved

	Returns
	-------
	None
	"""

	with open(data_dir+dataname, 'rb') as handle:
		data = pickle.load(handle)
	
	if cosmological:
		if time:
			time_data = data['time']
		else:
			time_data = -1.+1./data['a_scale']
	else:
		time_data = data['time']

	mean_DZ = data['DZ_ratio'][:,0]
	std_DZ = data['DZ_ratio'][:,1:]
	if log:
		# Replace zeros in with small numbers
		std_DZ[std_DZ==0.] = small_num
		std_DZ = np.log10(std_DZ)
		mean_DZ = np.log10(mean_DZ)

	ax=plt.figure()
	plt.plot(time_data, mean_DZ)
	plt.fill_between(time_data, std_DZ[:,0], std_DZ[:,1],alpha = 0.4)

	# Set y-axis label
	if log:
		y_label = "Log D/Z Ratio"
	else:
		y_label = "D/Z Ratio"
	plt.ylabel(y_label,fontsize = Large_Font)

	# Set y limits
	if log:
		DZ_min = -1.0
		DZ_max = 0.0
	else:
		DZ_min = 0.0
		DZ_max = 1.0

	plt.ylim([DZ_min,DZ_max])

	if time or not cosmological:
		plt.xlabel('t (Gyr)',fontsize = Large_Font)
		plt.xscale('log')
	else:
		plt.xlabel('z',fontsize = Large_Font)
		plt.gca().invert_xaxis()
		plt.xscale('log')

	plt.savefig(foutname)
	plt.close()

def all_data_vs_time(dataname='data.pickle', data_dir='data/', foutname='all_data_vs_time.png', time=False, cosmological=True, log=True):
	"""
	Plots all time averaged data vs time from precompiled data

	Parameters
	----------
	dataname : str
		Name of data file
	datadir: str
		Directory of data
	foutname: str
		Name of file to be saved

	Returns
	-------
	None
	"""

	Z_solar = 0.02
	species_names = ['Silicates','Carbon','SiC','Iron','O Bucket']
	source_names = ['Accretion','SNe Ia', 'SNe II', 'AGB']

	with open(data_dir+dataname, 'rb') as handle:
		data = pickle.load(handle)
	
	if cosmological:
		if time:
			time_data = data['time']
		else:
			time_data = -1.+1./data['a_scale']
	else:
		time_data = data['time']

	num_species = np.shape(data['spec_frac'])[1]

	sfr = data['sfr'] 
	# Get mean and std, and make sure to set zero std to small number
	mean_DZ = data['DZ_ratio'][:,0]; std_DZ = data['DZ_ratio'][:,1:];
	mean_Z = data['metallicity'][:,0]/Z_solar; std_Z = data['metallicity'][:,1:]/Z_solar;
	mean_spec = data['spec_frac'][:,:,0]; std_spec = data['spec_frac'][:,:,1:];
	mean_source = data['source_frac'][:,:,0]; std_source = data['source_frac'][:,:,1:];
	mean_sil_to_C = data['sil_to_C_ratio'][:,0]; std_sil_to_C = data['sil_to_C_ratio'][:,1:];

	if log:
		# Replace zeros in with small numbers
		std_DZ[std_DZ==0.] = small_num;
		std_DZ = np.log10(std_DZ); 
		mean_DZ = np.log10(mean_DZ);

	fig,axes = plt.subplots(2, 3, sharex='all', figsize=(24,12))

	axes[0,0].plot(time_data, mean_DZ)
	axes[0,0].set_xlim([time_data[1],time_data[-1]])
	axes[0,0].fill_between(time_data, std_DZ[:,0], std_DZ[:,1],alpha = 0.4)
	axes[0,0].set_xscale('log')

	axes[0,1].plot(time_data, sfr)
	axes[0,1].set_xlim([time_data[1],time_data[-1]])
	axes[0,1].set_ylabel(r'SFR $(M_{\odot}/yr)$',fontsize = Large_Font)
	axes[0,1].set_ylim([0.0001,0.1])
	axes[0,1].set_xscale('log')
	axes[0,1].set_yscale('log')


	axes[0,2].plot(time_data, np.log10(mean_Z))
	axes[0,02].set_xlim([time_data[1],time_data[-1]])
	axes[0,2].fill_between(time_data, std_Z[:,0], std_Z[:,1],alpha = 0.4)
	axes[0,2].set_xscale('log')
	axes[0,2].set_ylim([-2.,1.])
	axes[0,2].set_ylabel(r'Log Z $(Z_{\odot})$',fontsize = Large_Font)

	for i in range(num_species):
		axes[1,0].plot(time_data, mean_spec[:,i], label=species_names[i])
		axes[1,0].fill_between(time_data, std_spec[:,i,0], std_spec[:,i,1],alpha = 0.4)
	axes[1,0].set_xlim([time_data[1],time_data[-1]])
	axes[1,0].set_ylabel(r'Species Mass Fraction',fontsize = Large_Font)
	axes[1,0].set_ylim([0,1])
	axes[1,0].set_xscale('log')
	axes[1,0].legend(loc=0, fontsize=Small_Font, frameon=False)

	for i in range(4):
		axes[1,1].plot(time_data, mean_source[:,i], label=source_names[i])
		axes[1,1].fill_between(time_data, std_source[:,i,0], std_source[:,i,1],alpha = 0.4)
	axes[1,1].set_xlim([time_data[1],time_data[-1]])
	axes[1,1].set_ylabel(r'Source Mass Fraction',fontsize = Large_Font)
	axes[1,1].set_ylim([1E-2,1.1])
	axes[1,1].set_yscale('log')
	axes[1,1].set_xscale('log')
	axes[1,1].legend(loc=0, fontsize=Small_Font, frameon=False)

	axes[1,2].plot(time_data, mean_sil_to_C)
	axes[1,2].fill_between(time_data, std_sil_to_C[:,0], std_sil_to_C[:,1],alpha = 0.4)
	axes[1,2].set_xlim([time_data[1],time_data[-1]])
	axes[1,2].set_ylabel(r'Silicates to Carbon Ratio',fontsize = Large_Font)
	axes[1,2].set_ylim([0,5])
	axes[1,2].set_xscale('log')

	# Set tick size
	for ax in axes.flatten():
		ax.tick_params(axis='both', which='major', labelsize=Small_Font)
		ax.tick_params(axis='both', which='minor', labelsize=Small_Font)


	# Set y-axis label for D/Z and Z
	if log:
		axes[0,0].set_ylabel(r'Log D/Z Ratio',fontsize = Large_Font)	
	else:
		axes[0,0].set_ylabel(r'D/Z Ratio',fontsize = Large_Font)

	# Set y limits
	if log:
		val_min = -1.0
		val_max = 0.0
	else:
		val_min = 0.0
		val_max = 1.0

	axes[0,0].set_ylim([val_min,val_max])


	if time or not cosmological:
		axes[1,0].set_xlabel('t (Gyr)',fontsize = Large_Font)
		axes[1,1].set_xlabel('t (Gyr)',fontsize = Large_Font)
		axes[1,2].set_xlabel('t (Gyr)',fontsize = Large_Font)
	else:
		axes[1,0].set_xlabel('z',fontsize = Large_Font)
		axes[1,1].set_xlabel('z',fontsize = Large_Font)
		axes[1,2].set_xlabel('z',fontsize = Large_Font)
		plt.gca().invert_xaxis()
		

	plt.tight_layout()

	plt.savefig(foutname)
	plt.close()

	
def compare_runs_vs_time(datanames=['data.pickle'], data_dir='data/', foutname='compare_runs_vs_time.png', \
	                     labels=["fiducial"], time=False, cosmological=True, log=True):
	"""
	Plots all time averaged data vs time from precompiled data for a set of simulation runs

	Parameters
	----------
	dataname : list
		List of data file names for sims
	datadir: str
		Directory of data
	foutname: str
		Name of file to be saved

	Returns
	-------
	None
	"""

	Z_solar = 0.02
	species_names = ['Silicates','Carbon','SiC','Iron','O Bucket']
	source_names = ['Accretion','SNe Ia', 'SNe II', 'AGB']

	fig,axes = plt.subplots(2, 2, sharex='all', figsize=(24,12))
	lines = [] # List of line styles used for plot legend
	for i,dataname in enumerate(datanames):

		lines += [mlines.Line2D([], [], color='xkcd:black',
                          linestyle=Line_Styles[i], label=labels[i])]

		with open(data_dir+dataname, 'rb') as handle:
			data = pickle.load(handle)
		
		if cosmological:
			if time:
				time_data = data['time']
			else:
				time_data = -1.+1./data['a_scale']
		else:
			time_data = data['time']

		num_species = np.shape(data['spec_frac'])[1]

		# Get mean and std, and make sure to set zero std to small number
		mean_DZ = data['DZ_ratio'][:,0]; std_DZ = data['DZ_ratio'][:,1:];
		mean_spec = data['spec_frac'][:,:,0]; std_spec = data['spec_frac'][:,:,1:];
		mean_source = data['source_frac'][:,:,0]; std_source = data['source_frac'][:,:,1:];
		mean_sil_to_C = data['sil_to_C_ratio'][:,0]; std_sil_to_C = data['sil_to_C_ratio'][:,1:];

		# If plotting log of D/Z
		if log:
			# Replace zeros in with small numbers
			std_DZ[std_DZ==0.] = small_num; 
			std_DZ = np.log10(std_DZ);
			mean_DZ = np.log10(mean_DZ);	
		
		axes[0,0].plot(time_data, mean_DZ, color='xkcd:black', linestyle=Line_Styles[i])

		for j in range(num_species):
			axes[0,1].plot(time_data, mean_spec[:,j], label=species_names[j], color=Line_Colors[j], linestyle=Line_Styles[i])


		for j in range(4):
			axes[1,0].plot(time_data, mean_source[:,j], label=source_names[j], color=Line_Colors[j], linestyle=Line_Styles[i])

		axes[1,1].plot(time_data, mean_sil_to_C, color='xkcd:black', linestyle=Line_Styles[i])


	# Set y-axis label
	if log:
		y_label = "Log D/Z Ratio"
	else:
		y_label = "D/Z Ratio"

	# Set y limits
	if log:
		DZ_min = -1.0
		DZ_max = 0.0
	else:
		DZ_max = 1.0
		DZ_min = 0.0

	axes[0,0].set_xlim([time_data[1],time_data[-1]])
	axes[0,0].set_ylabel(y_label,fontsize = Large_Font)
	axes[0,0].set_ylim([DZ_min,DZ_max])
	axes[0,0].set_xscale('log')

	axes[0,1].set_xlim([time_data[1],time_data[-1]])
	axes[0,1].set_ylabel(r'Species Mass Fraction',fontsize = Large_Font)
	axes[0,1].set_ylim([0,1])
	axes[0,1].set_xscale('log')
	spec_lines = []
	for i in range(num_species):
		spec_lines += [mlines.Line2D([], [], color=Line_Colors[i], label=species_names[i])]
	axes[0,1].legend(handles=spec_lines,loc=0, frameon=False)

	axes[1,0].set_xlim([time_data[1],time_data[-1]])
	axes[1,0].set_ylabel(r'Source Mass Fraction',fontsize = Large_Font)
	axes[1,0].set_ylim([1E-2,1.1])
	axes[1,0].set_yscale('log')
	axes[1,0].set_xscale('log')
	source_lines = []
	for i in range(4):
		source_lines += [mlines.Line2D([], [], color=Line_Colors[i], label=source_names[i])]
	axes[1,0].legend(handles=source_lines, loc=0, frameon=False)

	axes[1,1].set_xlim([time_data[1],time_data[-1]])
	axes[1,1].set_ylabel(r'Silicates to Carbon Ratio',fontsize = Large_Font)
	axes[1,1].set_ylim([0,5])
	axes[1,1].set_xscale('log')


	if time or not cosmological:
		axes[1,0].set_xlabel('t (Gyr)',fontsize = Large_Font)
		axes[1,1].set_xlabel('t (Gyr)',fontsize = Large_Font)
	else:
		axes[1,0].set_xlabel('z',fontsize = Large_Font)
		axes[1,1].set_xlabel('z',fontsize = Large_Font)
		plt.gca().invert_xaxis()

	# Set tick size
	for ax in axes.flatten():
		ax.tick_params(axis='both', which='major', labelsize=Small_Font)
		ax.tick_params(axis='both', which='minor', labelsize=Small_Font)

	
	# Create the legend for the different runs
	axes[0,0].legend(handles=lines, loc=0, frameon=False)

	"""
	fig.legend(handles=lines,   		# The line objects
           loc="upper center",  		# Position of legend
           borderaxespad=0.1,   		# Small spacing around legend box
           ncol=len(lines),    			# Make the legend stretch horizontally across the plot
           fontsize=24,
           bbox_to_anchor=(0.5, .95), 	# Pin the legend to just above the plots
		   frameon=False)
	"""
	plt.savefig(foutname)
	plt.close()


def compile_dust_data(snap_dir, foutname='data.pickle', data_dir='data/', mask=False, halo_dir='', Rvir_frac = 1., \
                      r_max = None, Lz_hat = None, disk_height = None, overwrite=False, cosmological=True, startnum=0, \
                      endnum=600, implementation='species', depletion=False):
	"""
	Compiles all the dust data needed for time evolution plots from all of the snapshots 
	into a small file.

	Parameters
	----------
	snap_dir : string
		Name of directory with snapshots to be used 

	Returns
	-------
	None

	"""

	if os.path.isfile(data_dir + foutname) and not overwrite:
		"Data exists already. \n If you want to overwrite it use the overwrite param."
	else:
		# First create ouput directory if needed
		try:
			# Create target Directory
			os.mkdir(data_dir)
			print("Directory ", data_dir, " Created")
		except:
			print("Directory ", data_dir, " already exists")


		print("Fetching data now...")
		length = endnum-startnum+1
		# Need to load in the first snapshot to see how many dust species there are
		if implementation=='species':
			G = readsnap(snap_dir, startnum, 0, cosmological=cosmological)
			if G['k']==-1:
				print("No snapshot found in directory")
				print("Snap directory:", snap_dir)
				return
			species_num = np.shape(G['spec'])[1]
			print("There are %i dust species"%species_num)
		else:
			species_num = 2
		# Most data comes with mean of values and 16th and 84th percentile
		DZ_ratio = np.zeros([length,3])
		sil_to_C_ratio = np.zeros([length,3])
		sfr = np.zeros(length)
		metallicity = np.zeros([length,3])
		time = np.zeros(length)
		a_scale = np.zeros(length)
		source_frac = np.zeros((length,4,3))
		spec_frac = np.zeros((length,species_num,3))


		# Go through each of the snapshots and get the data
		for i, num in enumerate(range(startnum, endnum+1)):
			print(num)
			G = readsnap(snap_dir, num, 0, cosmological=cosmological)
			H = readsnap(snap_dir, num, 0, header_only=True, cosmological=cosmological)
			S = readsnap(snap_dir, num, 4, cosmological=cosmological)

			if G['k']==-1:
				print("No snapshot found in directory")
				print("Snap directory:", snap_dir)
				return

			if mask:
				coords = G['p']
				center = np.zeros(3)
				if cosmological:
					halo_data = Table.read(halo_dir,format='ascii')
					# Convert to physical units
					xpos =  halo_data['col7'][num-1]*H['time']/H['hubble']
					ypos =  halo_data['col8'][num-1]*H['time']/H['hubble']
					zpos =  halo_data['col9'][num-1]*H['time']/H['hubble']
					rvir = halo_data['col13'][num-1]*H['time']/H['hubble']
					
					#TODO : Add ability to only look at particles in disk using angular momentum vector from
					# halo file
					
					center = np.array([xpos,ypos,zpos])
					coords -= center
					if r_max == None:
						print("Using AHF halo as spherical mask with radius of ",str(Rvir_frac)," * Rvir.")
						r_max = rvir*Rvir_frac
					else:
						print("Using AHF halo as spherical mask with radius of ",str(r_max)," kpc.")

					in_galaxy = np.sum(np.power(coords,2),axis=1) <= np.power(r_max,2.)
				else:
					if r_max == None:
						print("Must give maximum radius r_max for non-cosmological simulations!")
						return
					# Recenter coords at center of periodic box
					boxsize = H['boxsize']
					mask1 = coords > boxsize/2; mask2 = coords <= boxsize/2
					coords[mask1] -= boxsize/2; coords[mask2] += boxsize/2;
					center = np.average(coords, weights = G['m'], axis = 0)
					coords -= center
					# Check if mask should be sphere or disk if Lz_hat is given it's a disk
					if Lz_hat != None:
						zmag = np.dot(coords,Lz_hat)
						r_z = np.zeros(np.shape(coords))
						r_z[:,0] = zmag*Lz_hat[0]
						r_z[:,1] = zmag*Lz_hat[1]
						r_z[:,2] = zmag*Lz_hat[2]
						r_s = np.subtract(coords,r_z)
						smag = np.sqrt(np.sum(np.power(r_s,2),axis=1))
						in_galaxy = np.logical_and(np.abs(zmag) <= disk_height, smag <= r_max)
					# Get particles in sphere otherwise
					else:
						in_galaxy = np.sum(np.power(coords,2),axis=1) <= np.power(r_max,2.)

				for key in G.keys():
					if key != 'k':
						G[key] = G[key][in_galaxy]
				# Check if there are any star particles
				if S['k']!=-1:
					coords = S['p']
					if not cosmological:
						boxsize = H['boxsize']
						mask1 = coords > boxsize/2; mask2 = coords <= boxsize/2
						coords[mask1] -= boxsize/2; coords[mask2] += boxsize/2;

					coords -= center

					# Check if mask should be sphere or disk if Lz_hat is given it's a disk
					if Lz_hat != None:
						zmag = np.dot(coords,Lz_hat)
						r_z = np.zeros(np.shape(coords))
						r_z[:,0] = zmag*Lz_hat[0]
						r_z[:,1] = zmag*Lz_hat[1]
						r_z[:,2] = zmag*Lz_hat[2]
						r_s = np.subtract(coords,r_z)
						smag = np.sqrt(np.sum(np.power(r_s,2),axis=1))
						in_galaxy = np.logical_and(np.abs(zmag) <= disk_height, smag <= r_max)
					# Get particles in sphere otherwise
					else:
						in_galaxy = np.sum(np.power(coords,2),axis=1) <= np.power(r_max,2.)

					S['age'] = S['age'][in_galaxy]
					S['m'] = S['m'][in_galaxy]

			M = G['m']
			omeganot = H['omega0']
			h = H['hubble']
			if cosmological:
				a_scale[i] = H['time']
				time[i] = tfora(H['time'], omeganot, h)
			else:
				time[i] = H['time']

			if depletion:
				metallicity[i] = weighted_percentile(G['z'][:,0]+G['dz'][:,0], weights=M)
			else:
				metallicity[i] = weighted_percentile(G['z'][:,0], weights=M)

			for j in range(4):
				source_frac[i,j] = weighted_percentile(G['dzs'][:,j], weights=M)
				source_frac[i,j][source_frac[i,j]==0] = small_num


			if implementation == 'species':
				# Need to mask all rows with nan and inf values for average to work
				for j in range(species_num):
					spec_frac_vals = G['spec'][:,j]/G['dz'][:,0]
					is_num = np.logical_and(~np.isnan(spec_frac_vals), ~np.isinf(spec_frac_vals))
					spec_frac[i,j] = weighted_percentile(spec_frac_vals[is_num], weights =M[is_num])
					spec_frac[i,j][spec_frac[i,j]==0] = small_num

				sil_to_C_vals = G['spec'][:,0]/G['spec'][:,1]
				is_num = np.logical_and(~np.isnan(sil_to_C_vals), ~np.isinf(sil_to_C_vals))
				sil_to_C_ratio[i] = weighted_percentile(sil_to_C_vals[is_num], weights =M[is_num])
				sil_to_C_ratio[i][sil_to_C_ratio[i]==0] = small_num

			elif implementation == 'elemental':
				# Need to mask nan and inf values for average to work
				spec_frac_vals = (G['dz'][:,4]+G['dz'][:,6]+G['dz'][:,7]+G['dz'][:,10])/G['dz'][:,0]
				is_num = np.logical_and(~np.isnan(spec_frac_vals), ~np.isinf(spec_frac_vals))
				spec_frac[i,0] = weighted_percentile(spec_frac_vals[is_num], weights =M[is_num])
				spec_frac[i,0][spec_frac[i,0]==0] = small_num

				spec_frac_vals = G['dz'][:,2]/G['dz'][:,0]
				is_num = np.logical_and(~np.isnan(spec_frac_vals), ~np.isinf(spec_frac_vals))
				spec_frac[i,1] = weighted_percentile(spec_frac_vals[is_num], weights =M[is_num])
				spec_frac[i,1][spec_frac[i,1]==0] = small_num

				sil_to_C_vals = (G['dz'][:,4]+G['dz'][:,6]+G['dz'][:,7]+G['dz'][:,10])/G['dz'][:,2]
				is_num = np.logical_and(~np.isnan(sil_to_C_vals), ~np.isinf(sil_to_C_vals))
				sil_to_C_ratio[i] = weighted_percentile(sil_to_C_vals[is_num], weights =M[is_num])
				sil_to_C_ratio[i][sil_to_C_ratio[i]==0] = small_num

			if depletion:
				DZ_vals = G['dz'][:,0]/(G['z'][:,0]+G['dz'][:,0])
			else:
				DZ_vals = G['dz'][:,0]/G['z'][:,0]
			DZ_ratio[i] = weighted_percentile(DZ_vals, weights=M)
			DZ_ratio[i][DZ_ratio[i]==0] = small_num

			# Calculate SFR as all stars born within the last 100 Myrs
			if S['k']!=-1:
				if cosmological:
					formation_time = tfora(S['age'], omeganot, h)
					current_time = time[i]
				else:
					formation_time = S['age']*UnitTime_in_Gyr
					current_time = time[i]*UnitTime_in_Gyr

				time_interval = 100E-3 # 100 Myr
				new_stars = (current_time - formation_time) < time_interval
				sfr[i] = np.sum(S['m'][new_stars]) * UnitMass_in_Msolar / (time_interval*1E9)   # Msun/yr

		if cosmological:
			data = {'time':time,'a_scale':a_scale,'DZ_ratio':DZ_ratio,'sil_to_C_ratio':sil_to_C_ratio,'metallicity':metallicity,'source_frac':source_frac,'spec_frac':spec_frac,'sfr':sfr}
		else:
			data = {'time':time,'DZ_ratio':DZ_ratio,'sil_to_C_ratio':sil_to_C_ratio,'metallicity':metallicity,'source_frac':source_frac,'spec_frac':spec_frac,'sfr':sfr}
		with open(data_dir+foutname, 'wb') as handle:
			pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)