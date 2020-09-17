import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
plt.switch_backend('agg')
from scipy.stats import binned_statistic_2d
from scipy.optimize import curve_fit
import pickle
import os
import observations as obs
from analytic_dust_yields import *
import plot_setup as plt_set

import gizmo_library.config as config
import gizmo_library.utils as utils



def plot_observational_data(axis, param, elem=None, log=True, CO_opt='S12', goodSNR=True):
	"""
	Plots observational D/Z data vs the given param.

	Parameters
	----------
	axis : Matplotlib axis
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
		data = obs.Chiang_20_DZ_vs_param(param, CO_opt=CO_opt, bin_nums=30, log=True, goodSNR=goodSNR)
		for i, gal_name in enumerate(data.keys()):
			fH2_vals = data[gal_name][0]; mean_DZ = data[gal_name][1]; std_DZ = data[gal_name][2]
			if log:
				std_DZ[std_DZ == 0] = config.EPSILON
			axis.errorbar(fH2_vals, mean_DZ, yerr = np.abs(mean_DZ-np.transpose(std_DZ)), label=gal_name, c=config.MARKER_COLORS[i], fmt=config.MARKER_STYLES[i], elinewidth=1, markersize=6,zorder=2)
	elif param == 'r':
		data = obs.Chiang_20_DZ_vs_param(param, bin_data=True, CO_opt=CO_opt, phys_r=True, bin_nums=30, log=False, goodSNR=goodSNR)
		for i, gal_name in enumerate(data.keys()):
			r_vals = data[gal_name][0]; mean_DZ = data[gal_name][1]; std_DZ = data[gal_name][2]
			if log:
				std_DZ[std_DZ == 0] = config.EPSILON
			axis.errorbar(r_vals, mean_DZ, yerr = np.abs(mean_DZ-np.transpose(std_DZ)), label=gal_name, c=config.MARKER_COLORS[i], fmt=config.MARKER_STYLES[i], elinewidth=1, markersize=6,zorder=2)
	elif param == 'r25':
		data = obs.Chiang_20_DZ_vs_param('r', bin_data=True, CO_opt=CO_opt, phys_r=False, bin_nums=30, log=False, goodSNR=goodSNR)
		for i, gal_name in enumerate(data.keys()):
			r_vals = data[gal_name][0]; mean_DZ = data[gal_name][1]; std_DZ = data[gal_name][2]
			if log:
				std_DZ[std_DZ == 0] = config.EPSILON
			axis.errorbar(r_vals, mean_DZ, yerr = np.abs(mean_DZ-np.transpose(std_DZ)), label=gal_name, c=config.MARKER_COLORS[i], fmt=config.MARKER_STYLES[i], elinewidth=1, markersize=6,zorder=2)
	elif param == 'nH':
		dens_vals, DZ_vals = obs.Jenkins_2009_DZ_vs_dens(phys_dens=True)
		axis.plot(dens_vals, DZ_vals, label='Jenkins09 w/ Phys. Dens.', c='xkcd:black', linestyle=config.LINE_STYLES[0], linewidth=config.LINE_WIDTHS[5], zorder=2)
		dens_vals, DZ_vals = obs.Jenkins_2009_DZ_vs_dens(phys_dens=False)
		axis.plot(dens_vals, DZ_vals, label='Jenkins09', c='xkcd:black', linestyle=config.LINE_STYLES[1], linewidth=config.LINE_WIDTHS[5], zorder=2)
	elif param == 'sigma_dust':
		data = obs.Chiang_20_DZ_vs_param(param, bin_data=True, CO_opt=CO_opt, bin_nums=30, log=True, goodSNR=goodSNR)
		for i, gal_name in enumerate(data.keys()):
			sigma_vals = data[gal_name][0]; mean_DZ = data[gal_name][1]; std_DZ = data[gal_name][2]
			if log:
				std_DZ[std_DZ == 0] = config.EPSILON
			axis.errorbar(sigma_vals, mean_DZ, yerr = np.abs(mean_DZ-np.transpose(std_DZ)), label=gal_name, c=config.MARKER_COLORS[i], fmt=config.MARKER_STYLES[i], elinewidth=1, markersize=6,zorder=2)	
	elif param == 'sigma_gas':
		data = obs.Chiang_20_DZ_vs_param(param, bin_data=True, CO_opt=CO_opt, bin_nums=30, log=True, goodSNR=True)
		for i, gal_name in enumerate(data.keys()):
			sigma_vals = data[gal_name][0]; mean_DZ = data[gal_name][1]; std_DZ = data[gal_name][2]
			if log:
				std_DZ[std_DZ == 0] = config.EPSILON
			axis.errorbar(sigma_vals, mean_DZ, yerr = np.abs(mean_DZ-np.transpose(std_DZ)), label=gal_name, c=config.MARKER_COLORS[i], fmt=config.MARKER_STYLES[i], elinewidth=1, markersize=6,zorder=2)	
		if not goodSNR:
			data = obs.Chiang_20_DZ_vs_param(param, bin_data=False, CO_opt=CO_opt, log=True, goodSNR=False)
			for i, gal_name in enumerate(data.keys()):
				sigma_vals = data[gal_name][0]; DZ = data[gal_name][1]
				axis.scatter(sigma_vals, DZ, c=config.MARKER_COLORS[i], marker=config.MARKER_STYLES[i], s=2, zorder=0, alpha=0.4)	
	elif param == 'sigma_H2':
		data = obs.Chiang_20_DZ_vs_param(param, bin_data=True, CO_opt=CO_opt, bin_nums=30, log=True, goodSNR=goodSNR)
		for i, gal_name in enumerate(data.keys()):
			sigma_vals = data[gal_name][0]; mean_DZ = data[gal_name][1]; std_DZ = data[gal_name][2]
			if log:
				std_DZ[std_DZ == 0] = config.EPSILON
			axis.errorbar(sigma_vals, mean_DZ, yerr = np.abs(mean_DZ-np.transpose(std_DZ)), label=gal_name, c=config.MARKER_COLORS[i], fmt=config.MARKER_STYLES[i], elinewidth=1, markersize=6,zorder=2)	
	elif param == 'depletion':
		dens_vals, DZ_vals = obs.Jenkins_2009_DZ_vs_dens(elem=elem, phys_dens=False)
		axis.plot(dens_vals, 1.-DZ_vals, label='Jenkins09', c='xkcd:black', linestyle=config.LINE_STYLES[1], linewidth=config.LINE_WIDTHS[5], zorder=2)
		dens_vals, DZ_vals = obs.Jenkins_2009_DZ_vs_dens(elem=elem, phys_dens=True)
		axis.plot(dens_vals, 1.-DZ_vals, label='Jenkins09 w/ Phys. Dens.', c='xkcd:black', linestyle=config.LINE_STYLES[0], linewidth=config.LINE_WIDTHS[5], zorder=2)
	elif param == 'sigma_Z':
		# TO DO: Add Remy-Ruyer D/Z vs Z observed data
		print("D/Z vs Z observations have not been implemented yet")
	else:
		print("D/Z vs %s observational data is not available."%param)

	return



def DZ_vs_params(params, snaps, bin_nums=50, time=None, labels=None, foutname='DZ_vs_param.png', std_bars=True, style='color', include_obs=True, CO_opt='S12'):
	"""
	Plots the average dust-to-metals ratio (D/Z) vs given parameters given code values of center and virial radius for multiple simulations/snapshots

	Parameters
	----------
	params: array
		Array of parameters to plot D/Z against (fH2, nH, Z, r, r25)
	snaps : array
	    Array of snapshots to plot
	bin_nums : int
		Number of bins to use
	time : string
		Option for printing time in corner of plot (None, one, all)
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
	include_obs : boolean
		Overplot observed data if available

	Returns
	-------
	None

	"""	

	# Get plot stylization
	linewidths,colors,linestyles = plt_set.setup_plot_style(len(snaps), style=style)

	# Set up subplots based on number of parameters given
	fig,axes = plt_set.setup_figure(len(params))

	for i, x_param in enumerate(params):
		# Set up for each plot
		axis = axes[i]
		y_param = 'DZ'
		plt_set.setup_axis(axis, x_param, y_param)

		# First plot observational data if applicable
		if include_obs: plot_observational_data(axis, x_param, CO_opt=CO_opt, goodSNR=True);

		for j,snap in enumerate(snaps):
			G = snap.loadpart(0)
			H = snap.loadheader()
			param_vals,mean_DZ,std_DZ = calc_DZ_vs_param(x_param, G, bin_nums=bin_nums)

			# Only need to label the seperate simulations in the first plot
			if i==0 and labels is not None: label = labels[j];
			else:    						label = None;
			axis.plot(param_vals, mean_DZ, label=label, linestyle=linestyles[j], color=colors[j], linewidth=linewidths[j], zorder=3)
			if std_bars:
				axis.fill_between(param_vals, std_DZ[:,0], std_DZ[:,1], alpha = 0.3, color=colors[j], zorder=1)

		# Setup legend
		if include_obs: ncol=2;
		else: 			ncol=1;
		axis.legend(loc=0, fontsize=config.SMALL_FONT, frameon=False, ncol=ncol)

		# Print time in corner of plot if applicable
		if time=='one' and i==0:
			time_str = 'z = ' + '%.2g' % H.redshift if snap.cosmological else 't = ' + '%2.2g Gyr' % H.time
			axis.text(.05, .95, time_str, color="xkcd:black", fontsize = config.SMALL_FONT, ha = 'left', transform=axis.transAxes, zorder=4)	
		elif time=='all':
			time_str = 'z = ' + '%.2g' % H.redshift if snap.cosmological else 't = ' + '%2.2g Gyr' % H.time
			axis.text(.05, .95, time_str, color="xkcd:black", fontsize = config.SMALL_FONT, ha = 'left', transform=axis.transAxes, zorder=4)			

	plt.tight_layout()	
	plt.savefig(foutname)
	plt.close()

	return




def calc_DZ_vs_param(param, G, bin_nums=50, param_lims=None, elem='Z'):
	"""
	Calculate the average dust-to-metals ratio (D/Z) vs radius, density, and Z given code values of center and virial radius for multiple simulations/snapshots

	Parameters
	----------
	param: string
		Name of parameter to get D/Z values for
	G : dict
	    Snapshot gas data structure
	bin_nums : int
		Number of bins to use
	elem : string, optional
		Can specify the D/Z ratio for a specific element. Default is all metals.

	Returns
	-------
	mean_DZ : array
		Array of mean D/Z values vs parameter given
	std_DZ : array
		Array of 16th and 84th percentiles D/Z values
	param_vals : array
		Parameter values D/Z values are taken over

	"""	

	# First make sure the element given is valid
	if elem not in config.ELEMENTS:
		print('%s is not a valid element to calculate D/Z for. Valid elements are'%elem)
		print(config.ELEMENT)
		return None,None,None
	
	elem_indx = config.ELEMENTS.index(elem)

	if param_lims is None:
		param_lims = config.PARAM_INFO[param][1]
		log_bins = config.PARAM_INFO[param][2]
	else:
		if param_lims[1] > 20*param_lims[0]: 	log_bins=True
		else:								  	log_bins=False

	# Get D/Z values over number density of Hydrogen (nH)
	if param == 'nH':
		nH = G.rho*config.UnitDensity_in_cgs * (1. - (G.z[:,0]+G.z[:,1])) / config.H_MASS
		bin_data = nH
		log_bins=True
	# Get D/Z values over gas temperature
	elif param == 'T':
		T = G.T
		bin_data = T
		log_bins=True
	# Get D/Z valus over radius of galaxy from the center
	elif param == 'r' or param == 'r25':
		# TODO: implement r25 bins and r for halo objects 
		r = np.sqrt(np.power(G.p[:,0],2) + np.power(G.p[:,1],2))
		bin_data = r
		if param_lims[1]>40: log_bins=True
		else:			     log_bins=False
	# Get D/Z values vs total metallicty of gas
	elif param == 'Z':
		Z = G.z[:,0]/config.SOLAR_Z
		bin_data = Z
		log_bins=True
	# Get D/Z values vs H2 mass fraction of gas
	elif param == 'fH2':
		fH2 = utils.calc_fH2(G)
		bin_data = fH2
		log_bins=False
	else:
		print("Parameter given to calc_DZ_vs_param is not supported:",param)
		return None,None,None

	DZ = G.dz[:,elem_indx]/G.z[:,elem_indx]
	DZ[DZ > 1] = 1.
	
	bin_vals, mean_DZ, std_DZ = utils.bin_values(bin_data, DZ, param_lims, bin_nums=bin_nums, weight_vals=G.m, log=log_bins)

	return bin_vals, mean_DZ, std_DZ



def observed_DZ_vs_param(params, snaps, pixel_res=2, bin_nums=50, time=None, labels=None, foutname='obs_DZ_vs_param.png', \
						 std_bars=True, style='color', include_obs=True, CO_opt='S12'):
	"""
	Plots mock observations of dust-to-metals vs various parameters for multiple simulations 

	Parameters
	----------
	params: array
		Array of parameters to plot D/Z against (fH2, nH, Z, r, r25)
	snaps : array
	    Array of snapshots to plot
	pixel_res : double, optional
		Resolution of simulated pixels in kpc
	bin_nums : int, optional
		Number of bins to use
	time : string, optional
		Option for printing time in corner of plot (None, one, all)
	labels : array, optional
		Array of labels for each data set
	foutname: str, optional, optional
		Name of file to be saved
	std_bars : bool, optional
		Include standard deviation bars for the data
	style : string, optional
		Plotting style when plotting multiple data sets
		'color' - gives different color and linestyles to each data set
		'size' - make all lines solid black but with varying line thickness
	include_obs : boolean, optional
		Overplot observed data if available

	Returns
	-------
	None

	"""	

	# Get plot stylization
	linewidths,colors,linestyles = plt_set.setup_plot_style(len(snaps), style=style)

	# Set up subplots based on number of parameters given
	fig,axes = plt_set.setup_figure(len(params))

	for i, x_param in enumerate(params):
		# Set up for each plot
		axis = axes[i]
		y_param = 'DZ'
		plt_set.setup_axis(axis, x_param, y_param)

		# First plot observational data if applicable
		if include_obs: plot_observational_data(axis, x_param, CO_opt=CO_opt, goodSNR=True);

		for j,snap in enumerate(snaps):
			G = snap.loadpart(0)
			H = snap.loadheader()
			r_max = snap.get_rmax()
			param_vals,mean_DZ,std_DZ = calc_obs_DZ_vs_param(x_param, G, r_max, bin_nums=bin_nums, pixel_res=pixel_res)

			# Only need to label the seperate simulations in the first plot
			if i==0 and labels is not None: label = labels[j];
			else:    						label = None;
			axis.plot(param_vals, mean_DZ, label=label, linestyle=linestyles[j], color=colors[j], linewidth=linewidths[j], zorder=3)
			if std_bars:
				axis.fill_between(param_vals, std_DZ[:,0], std_DZ[:,1], alpha = 0.3, color=colors[j], zorder=1)

		# Setup legend
		if include_obs: ncol=2;
		else: 			ncol=1;
		axis.legend(loc=0, fontsize=config.SMALL_FONT, frameon=False, ncol=ncol)

		# Print time in corner of plot if applicable
		if time=='one' and i==0:
			time_str = 'z = ' + '%.2g' % H.redshift if snap.cosmological else 't = ' + '%2.2g Gyr' % H.time
			axis.text(.05, .95, time_str, color="xkcd:black", fontsize = config.LARGE_FONT, ha = 'left', transform=axis.transAxes, zorder=4)	
		elif time=='all':
			time_str = 'z = ' + '%.2g' % H.redshift if snap.cosmological else 't = ' + '%2.2g Gyr' % H.time
			axis.text(.05, .95, time_str, color="xkcd:black", fontsize = config.LARGE_FONT, ha = 'left', transform=axis.transAxes, zorder=4)			

	plt.tight_layout()	
	plt.savefig(foutname)
	plt.close()	

	return


def calc_obs_DZ_vs_param(param, G, r_max, pixel_res=2, bin_nums=50, param_lims=None):
	"""
	Calculate the average dust-to-metals ratio vs radius, gas , H2 , metal, or dust surface density
	given code values of center and viewing direction for multiple simulations/snapshots

	Parameters
	----------
	param: string
		Name of parameter to get D/Z values for
	G : Particle
	    Snapshot gas data structure
	r_max : double
		Maximum radius from the center of the simulated observation
	pixel_res : double
		Size resolution of each pixel bin in kpc
	bin_nums : int
		Number of bins for param
	param_bins : int
		Number of bins to use for physical param

	Returns
	-------
	mean_surf_dens : array
		Array of mean dust surface density values vs parameter given
	std_surf_dens : array
		Array of 16th and 84th percentiles dust surface density values
	param_vals : array
		Parameter values dust surface density values are taken over
	"""	

	if param_lims is None:
		param_lims = config.PARAM_INFO[param][1]
		log_bins = config.PARAM_INFO[param][2]
	else:
		if param_lims[1] > 20*param_lims[0]: 	log_bins=True
		else:								  	log_bins=False

	M = G.m*config.UnitMass_in_Msolar
	dust_mass = G.dz[:,0]*M
	Z_mass = G.z[:,0]*M

	x = G.p[:,0];y=G.p[:,1];
	pixel_bins = int(np.ceil(2*r_max/pixel_res))
	x_bins = np.linspace(-r_max,r_max,pixel_bins)
	y_bins = np.linspace(-r_max,r_max,pixel_bins)
	x_vals = (x_bins[1:] + x_bins[:-1]) / 2.
	y_vals = (y_bins[1:] + y_bins[:-1]) / 2.
	pixel_area = pixel_res**2 * 1E6 # area of pixel in pc^2

	# This data will always be needed so set it here
	bin_data = [Z_mass,dust_mass]

	if param == 'sigma_dust':
		ret = binned_statistic_2d(x, y, bin_data, statistic=np.sum, bins=[x_bins,y_bins]).statistic
		DZ_pixel = ret[1].flatten()/ret[0].flatten()
		dust_pixel = ret[1].flatten()/pixel_area
		pixel_data = dust_pixel

	elif param=='sigma_gas':
		bin_data += [M]
		ret = binned_statistic_2d(x, y, bin_data, statistic=np.sum, bins=[x_bins,y_bins]).statistic
		DZ_pixel = ret[1].flatten()/ret[0].flatten()
		M_pixel = ret[2].flatten()/pixel_area
		pixel_data = M_pixel

	elif param=='sigma_H2':
		MH2 = utils.calc_fH2(G)*G.m*(G.z[:,0]+G.z[:,1])
		bin_data += [MH2]
		ret = binned_statistic_2d(x, y, bin_data, statistic=np.sum, bins=[x_bins,y_bins]).statistic
		DZ_pixel = ret[1].flatten()/ret[0].flatten()
		MH2_pixel = ret[2].flatten()/pixel_area
		pixel_data = MH2_pixel

	elif param == 'sigma_Z':
		ret = binned_statistic_2d(x, y, bin_data, statistic=np.sum, bins=[x_bins,y_bins]).statistic
		DZ_pixel = ret[1].flatten()/ret[0].flatten()
		Z_pixel = ret[0].flatten()/pixel_area
		pixel_data = Z_pixel

	elif param == 'fH2':
		fH2 = utils.calc_fH2(G)
		MH2 = fH2*G.m*(G.z[:,0]+G.z[:,1])
		MH1 = (1-fH2)*G.m*(G.z[:,0]+G.z[:,1])
		bin_data += [MH2,MH1]
		ret = binned_statistic_2d(x, y, bin_data, statistic=np.sum, bins=[x_bins,y_bins]).statistic
		DZ_pixel = ret[1].flatten()/ret[0].flatten()
		fH2_pixel = ret[2].flatten()/(ret[2].flatten()+ret[3].flatten())
		pixel_data = fH2_pixel

	elif param == 'r':
		mean_DZ = np.zeros(pixel_bins/2 - 1)
		std_DZ = np.zeros([pixel_bins/2 - 1,2])
		ret = binned_statistic_2d(x, y, bin_data, statistic=np.sum, bins=[x_bins,y_bins],expand_binnumbers=True)
		DZ_pixel = ret.statistic[1].flatten()/ret.statistic[0].flatten()
		# Get the average r coordinate for each pixel in kpc
		pixel_r_vals = np.array([np.sqrt(np.power(np.abs(y_vals),2) + np.power(np.abs(x_vals[k]),2)) for k in range(len(x_vals))]).flatten()
		pixel_data = pixel_r_vals
		# Makes more sense to force the number of bins for this
		bin_nums = pixel_bins/2
	else:
		print("Parameter given to calc_obs_DZ_vs_param is not supported:",param)
		return None,None,None

	bin_vals, mean_DZ, std_DZ = utils.bin_values(pixel_data, DZ_pixel, param_lims, bin_nums=bin_nums, weight_vals=None, log=log_bins)

	return bin_vals, mean_DZ, std_DZ



def elem_depletion_vs_param(elems, param, snaps, bin_nums=50, time=None, labels=None, \
			foutname='obs_elem_dep_vs_dens.png', std_bars=True, style='color', include_obs=True):
	"""
	Plots mock observations of specified elemental depletion vs various parameters for multiple simulations 

	Parameters
	----------
	elems : array
		Array of which elements you want to plot depletions for
	params: string
		Parameters to plot depletion against (fH2, nH)
	snaps : array
	    Array of snapshots to plot
	pixel_res : double, optional
		Resolution of simulated pixels in kpc
	bin_nums : int, optional
		Number of bins to use
	time : string, optional
		Option for printing time in corner of plot (None, all)
	labels : array, optional
		Array of labels for each data set
	foutname: str, optional, optional
		Name of file to be saved
	std_bars : bool, optional
		Include standard deviation bars for the data
	style : string, optional
		Plotting style when plotting multiple data sets
		'color' - gives different color and linestyles to each data set
		'size' - make all lines solid black but with varying line thickness
	include_obs : boolean, optional
		Overplot observed data if available

	Returns
	-------
	None
	"""	

	# Get plot stylization
	linewidths,colors,linestyles = plt_set.setup_plot_style(len(snaps), style=style)

	# Set up subplots based on number of parameters given
	fig,axes = plt_set.setup_figure(len(elems))

	for i,elem in enumerate(elems):
		axis = axes[i]
		plt_set.setup_axis(axis, param, 'depletion')

		if include_obs and param == 'nH':
			plot_observational_data(axis, param='depletion', elem=elem)

		for j,snap in enumerate(snaps):
			G = snap.loadpart(0)
			H = snap.loadheader()
			param_vals,mean_DZ,std_DZ = calc_DZ_vs_param(param, G, bin_nums=bin_nums, elem=elem)

			# Only need to label the seperate simulations in the first plot
			if i==0 and labels is not None: label = labels[j];
			else:    						label = None;
			axis.plot(param_vals, 1.-mean_DZ, label=label, linestyle=linestyles[j], color=colors[j], linewidth=linewidths[j], zorder=3)
			if std_bars:
				axis.fill_between(param_vals, 1.-std_DZ[:,0], 1.-std_DZ[:,1], alpha = 0.3, color=colors[j], zorder=1)

			# Setup legend
			if i == 0:
				if include_obs: ncol=2;
				else: 			ncol=1;
				axis.legend(loc=0, fontsize=config.SMALL_FONT, frameon=False, ncol=ncol)

		axis.text(.10, .30, elem, color="xkcd:black", fontsize = 2*config.LARGE_FONT, ha = 'center', va = 'center', transform=axis.transAxes)

		# Print time in corner of plot if applicable
		if time=='all':
			time_str = 'z = ' + '%.2g' % H.redshift if snap.cosmological else 't = ' + '%2.2g Gyr' % H.time
			axis.text(.05, .95, time_str, color="xkcd:black", fontsize = config.LARGE_FONT, ha = 'left', transform=axis.transAxes, zorder=4)	

	plt.tight_layout()
	plt.savefig(foutname)

	return



def dust_data_vs_time(params, data_objs, foutname='dust_data_vs_time.png',labels=None, style='color', time=True):
	"""
	Plots all time averaged data vs time from precompiled data for a set of simulation runs

	Parameters
	----------
	params : array
		List of parameters to plot over time
	data_objs : array
		Array of Dust_Evo objects with data to plot
	foutname: str, optional
		Name of file to be saved

	Returns
	-------
	None
	"""

	# Get plot stylization
	linewidths,colors,linestyles = plt_set.setup_plot_style(len(data_objs), style=style)

	# Set up subplots based on number of parameters given
	fig,axes = plt_set.setup_figure(len(params))

	for i, y_param in enumerate(params):
		# Set up for each plot
		axis = axes[i]		
		if time:
			x_param = 'time'
		else:
			x_param = 'redshift'
		plt_set.setup_axis(axis, x_param, y_param)

		param_labels=None
		if y_param == 'DZ':
			param_id = 'DZ_ratio'
		elif y_param == 'source_frac':
			param_id = 'source_frac'
			param_labels = ['Accretion','SNe Ia', 'SNe II', 'AGB']
		elif y_param=='spec_frac':
			param_id = 'spec_frac'
			param_labels = ['Silicates','Carbon','SiC','Iron','O Reservoir']
		elif y_param == 'Si/C':
			param_id = 'sil_to_C_ratio'
		else:
			print("%s is not a valid parameter for dust_data_vs_time()\n"%y_param)
			return()

		for j,data in enumerate(data_objs):
			
			if time:
				time_data = data.get_data('time')
			else:
				time_data = data.get_data('redshift')

			# Check if parameter has subparameters
			data_vals = data.get_data(y_param)
			if param_labels is None:
				axis.plot(time_data, data_vals, color='xkcd:black', linestyle=config.LINE_STYLES[j], label=labels[j], zorder=3)
			else:
				for k in range(np.shape(data_vals)[1]):
					axis.plot(time_data, data_vals[:,k], color=config.LINE_COLORS[k], linestyle=config.LINE_STYLES[j], zorder=3)
			axis.set_xlim([time_data[0],time_data[-1]])
		# Only need to label the seperate simulations in the first plot
		if i==0 and len(data_objs)>1:
			axis.legend(loc=0, frameon=False, fontsize=config.SMALL_FONT)
		# If there are subparameters need to make their own legend
		if param_labels is not None:
			param_lines = []
			for j, label in enumerate(param_labels):
				param_lines += [mlines.Line2D([], [], color=config.LINE_COLORS[j], label=label)]
			axis.legend(handles=param_lines, loc=0, frameon=False, fontsize=config.SMALL_FONT)

	plt.tight_layout()
	plt.savefig(foutname)
	plt.close()




def binned_phase_plot(param, snap, bin_nums=200, time=None, color_map='inferno', hist_proj=True, foutname='phase_plot.png'):
	"""
	Plots the a 2D histogram for nH vs T using the specified parameter as weights

	Parameters
	----------
	params : array
		The parameters for the x axis, y axis, and weights respectively
	snap : Snapshot/Halo/Disk
	    Snapshot gas data structure
	bin_nums: int, optional
		Number of bins to use
	color_map : string, optional
		Color mapping for plot
	hist_proj : boolean, optional
		Add additional 1D histogram projection along x and y axis
	founame : string, optional
		File name for saved figure

	Returns
	-------
	None

	"""

	# TODO : Include 1D hist projects along each axis
	# Make this work for dynamic x and y params notable D/Z vs Z
	# Also fix cbar ticks for log space. Might be an issue with matplotlib version

	fig,axes = plt_set.setup_2D_hist_fig(hist_proj=hist_proj)
	axis_2D_hist = axes[0]
	plt_set.setup_axis(axis_2D_hist, 'nH', 'T')
	axis_2D_hist.set_facecolor('xkcd:grey')

	G = snap.loadpart(0)
	H = snap.loadheader()

	ret = calc_phase_hist(param, G, bin_nums=bin_nums)
	param_lims = config.PARAM_INFO[param][1]
	log_param = config.PARAM_INFO[param][2]
	if log_param:
		norm = mpl.colors.LogNorm()
	else:
		norm = None

	X, Y = np.meshgrid(ret.x_edge, ret.y_edge)
	img = axis_2D_hist.pcolormesh(X, Y, ret.statistic.T, cmap=plt.get_cmap(color_map), vmin=param_lims[0], vmax=param_lims[1], norm=norm)
	axis_2D_hist.autoscale('tight')

	bar_label =  config.PARAM_INFO[param][0]
	plt_set.setup_colorbar(img, axis_2D_hist, bar_label)

	# Print time in corner of plot if applicable
	if time=='all':
		time_str = 'z = ' + '%.2g' % H.redshift if snap.cosmological else 't = ' + '%2.2g Gyr' % H.time
		axis_2D_hist.text(.05, .95, time_str, color="xkcd:black", fontsize = config.LARGE_FONT, ha = 'left', transform=axis_2D_hist.transAxes, zorder=4)	
	plt.tight_layout()

	plt.savefig(foutname)
	plt.close()



def calc_phase_hist(param, G, bin_nums=100):
	"""
	Calculate the 2D histogram for the given params and data from gas particle

	Parameters
	----------
	params : array
		The parameters for the x axis, y axis, and weights respectively
	snap : Snapshot/Halo/Disk
	    Snapshot gas data structure
	bin_nums: int, optional
		Number of bins to use
	color_map : string, optional
		Color mapping for plot
	hist_proj : boolean, optional
		Add additional 1D histogram projection along x and y axis
	founame : string, optional
		File name for saved figure

	Returns
	-------
	None

	"""

	# Set up x and y data, limits, and bins
	nH_data = G.rho*config.UnitDensity_in_cgs * (1. - (G.z[:,0]+G.z[:,1])) / config.H_MASS
	T_data = G.T
	nH_bin_lims = config.PARAM_INFO['nH'][1]
	T_bin_lims = config.PARAM_INFO['T'][1]
	if config.PARAM_INFO['nH'][2]:
		nH_bins = np.logspace(np.log10(nH_bin_lims[0]),np.log10(nH_bin_lims[1]),bin_nums)
	else:
		nH_bins = np.linspace(nH_bin_lims[0], nH_bin_lims[1], bin_nums)
	if config.PARAM_INFO['T'][2]:
		T_bins = np.logspace(np.log10(T_bin_lims[0]),np.log10(T_bin_lims[1]),bin_nums)
	else:
		T_bins = np.linspace(T_bin_lims[0], T_bin_lims[1], bin_nums)

	func = np.mean
	if param == 'Z':
		Z = G.z[:,0]/config.SOLAR_Z
		bin_data = Z
	elif param == 'fH2':
		fH2 = utils.calc_fH2(G)
		bin_data = fH2
	elif param == 'M':
		M = G.m*1E10
		bin_data = M
		func = np.sum
	elif param == 'DZ':
		DZ = G.dz[:,0]/G.z[:,0]
		DZ[DZ > 1] = 1.
		bin_data = DZ
	else:
		print("Parameter given to calc_phase_hist is not supported:",param)
		return None
	
	ret = binned_statistic_2d(nH_data, T_data, bin_data, statistic=func, bins=[nH_bins, T_bins])

	return ret




def compare_dust_creation(Z_list, dust_species, data_dirc, FIRE_ver=2, transition_age = 0.03753, style='color', foutname='creation_routine_compare.pdf'):
	"""
	Plots comparison of stellar dust creation for the given stellar metallicities

	Parameters
	----------
	Z_list : list
		List of metallicities to compare in solar units
	dust_species : list
		List of dust species to plot individually (carbon, silicates, iron, SiC, silicates+)
	data_dirc: string
		Name of directory to store calculated yields files
	FIRE_ver : int
		Version of FIRE metals yields to use in calculations
	transition_age : double
		Age at which stellar yields switch from O/B to AGB stars

	Returns
	-------
	None

	"""

	# First create ouput directory if needed
	try:
	    # Create target Directory
	    os.mkdir(data_dirc)
	    print("Directory " + data_dirc +  " Created")
	except:
	    print("Directory " + data_dirc +  " already exists")

	# Get plot stylization
	linewidths,colors,linestyles = plt_set.setup_plot_style(len(Z_list), style=style)

	N = 10000 # number of steps 
	max_t = 10. # max age of stellar population to compute yields

	time_step = max_t/N
	time = np.arange(0,max_t,time_step)

	# First make simulated data if it hasn't been made already
	for Z in Z_list:
		name = '/elem_Z_'+str(Z).replace('.','-')+'_cum_yields.pickle'
		if not os.path.isfile(data_dirc + name):
			cum_yields, cum_dust_yields, cum_species_yields = totalStellarYields(max_t,N,Z, routine="elemental")
			pickle.dump({"time": time, "yields": cum_yields, "elem": cum_dust_yields, "spec": cum_species_yields}, open(data_dirc + name, "wb" ))

		name = '/spec_Z_'+str(Z).replace('.','-')+'_cum_yields.pickle'
		if not os.path.isfile(data_dirc +name):
			cum_yields, cum_dust_yields, cum_species_yields = totalStellarYields(max_t,N,Z, routine="species")
			pickle.dump({"time": time, "yields": cum_yields, "elem": cum_dust_yields, "spec": cum_species_yields}, open(data_dirc + name, "wb" ))


	# Compare routine carbon yields between routines
	# Set up subplots based on number of parameters given
	fig,axes = plt_set.setup_figure(len(dust_species))
	x_param = 'time'; y_param = 'cum_dust_prod'
	x_lim = [0,max_t]

	for i, species in enumerate(dust_species):
		axis = axes[i]
		plt_set.setup_axis(axis, x_param, y_param, x_lim=x_lim)
		if species == 'carbon':
			name = 'Carbonaceous'
			indices = np.array([1])
		elif species == 'silicates':
			name = 'Silicates'
			indices = np.array([0])
		elif species == 'silicates+':
			name = 'Silicates+Others'
			indices = np.array([0,2,3])
		elif species == 'iron':
			name = 'Iron'
			indices = np.array([3])
		elif species == 'SiC':
			name = 'SiC'
			indices = np.array([2])
		else:
			print("%s is not a valid dust species for compare_dust_creation()\n"%species)
			return()

		# Add extra lines emphazising the time regimes for SNe II or AGB+SNe Ia
		axis.axvline(transition_age, color='xkcd:grey',lw=3)
		# Only need labels and legend for first plot
		if i == 0:
			y_arrow = 1E-5
			axis.annotate('AGB+SNe Ia', va='center', xy=(transition_age, y_arrow), xycoords="data", xytext=(2*transition_age, y_arrow), 
			            arrowprops=dict(arrowstyle='<-',color='xkcd:grey', lw=3), size=config.LARGE_FONT, color='xkcd:grey')
			axis.annotate('SNe II', va='center', xy=(transition_age, y_arrow/2), xycoords="data", xytext=(0.2*transition_age, y_arrow/2), 
			            arrowprops=dict(arrowstyle='<-',color='xkcd:grey', lw=3), size=config.LARGE_FONT, color='xkcd:grey')
			# Make legend
			lines = []
			for j in range(len(Z_list)):
				lines += [mlines.Line2D([], [], color=colors[j], label=r'Z = %.2g $Z_{\odot}$' % Z_list[j])]
			lines += [mlines.Line2D([], [], color='xkcd:black', linestyle=linestyles[0], label='Elemental'), mlines.Line2D([], [], color='xkcd:black', linestyle=linestyles[1],label='Species')]
			axis.legend(handles=lines, frameon=True, ncol=2, loc='center left', bbox_to_anchor=(0.025,1.0), framealpha=1, fontsize=config.SMALL_FONT)
		#  Add label for dust species
		axis.text(.95, .05, name, color="xkcd:black", fontsize = config.LARGE_FONT, ha = 'right', transform=axis.transAxes)

		for j,Z in enumerate(Z_list):
			name = '/elem_Z_'+str(Z).replace('.','-')+'_cum_yields.pickle'
			data = pickle.load(open(data_dirc + name, "rb" ))
			time = data['time']; cum_yields = data['yields']; cum_dust_yields = data['elem']; cum_species_yields = data['spec'];
			elem_cum_spec = np.sum(cum_species_yields[:,indices], axis=1)
			axis.loglog(time, elem_cum_spec, color = colors[j], linestyle = linestyles[0], nonposy = 'clip', linewidth = linewidths[j])

			name = '/spec_Z_'+str(Z).replace('.','-')+'_cum_yields.pickle'
			data = pickle.load(open(data_dirc + name, "rb" ))
			time = data['time']; cum_yields = data['yields']; cum_dust_yields = data['elem']; cum_species_yields = data['spec'];
			spec_cum_spec = np.sum(cum_species_yields[:,indices], axis=1)
			axis.loglog(time, spec_cum_spec, color = colors[j], linestyle = linestyles[1], nonposy = 'clip', linewidth = linewidths[j])

		axis.set_ylim([1E-7,1E-2])
		axis.set_xlim([time[0], time[-1]])

	plt.savefig(foutname, format='pdf', transparent=False, bbox_inches='tight')
	plt.close()




def DZ_var_in_pixel(gas, header, center_list, r_max_list, Lz_list=None, \
			height_list=None, pixel_res=2, time=False, depletion=False, cosmological=True, labels=None, \
			foutname='DZ_variation_per_pixel.png', style='color', log=True):
	"""
	Plots variation of dust-to-metals in each observed pixels for multiple simulations 

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
	pixel_res : double
		Size of pixels in kpc
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
	style : string
		Plotting style when plotting multiple data sets
		'color' - gives different color and linestyles to each data set
		'size' - make all lines solid black but with varying line thickness
	log : boolean
		Plot log of D/Z

	Returns
	-------
	None
	"""	

	# Get plot stylization
	linewidths,colors,linestyles = plt_set.setup_plot_style(len(gas), style=style)

	fig = plt.figure() 
	axis = plt.gca()
	ylabel = r'D/Z Ratio'
	xlabel = r'Pixel Num'
	plt_set.setup_labels(axis,xlabel,ylabel)
	if log:
		axis.set_yscale('log')
		axis.set_ylim([0.01,1.0])
	else:
		axis.set_ylim([0.,1.0])


	for j in range(len(gas)):
		G = gas[j]; H = header[j]; center = center_list[j]; r_max = r_max_list[j]; 
		if Lz_list != None:
			Lz_hat = Lz_list[j]; disk_height = height_list[j];
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
		coords = coords[in_galaxy]
		dust_mass = G['dz'][in_galaxy,0]*M
		if depletion:
			Z_mass = G['z'][in_galaxy,0] * M + dust_mass
		else:
			Z_mass = G['z'][in_galaxy,0] * M

		x = coords[:,0];y=coords[:,1];
		pixel_bins = int(np.ceil(2*r_max/pixel_res))+1
		x_bins = np.linspace(-r_max,r_max,pixel_bins)
		y_bins = np.linspace(-r_max,r_max,pixel_bins)
		x_vals = (x_bins[1:] + x_bins[:-1]) / 2.
		y_vals = (y_bins[1:] + y_bins[:-1]) / 2.
		pixel_area = pixel_res**2 * 1E6 # area of pixel in pc^2

		ret = binned_statistic_2d(x, y, [Z_mass,dust_mass], statistic=np.sum, bins=[x_bins,y_bins], expand_binnumbers=True)
		data = ret.statistic
		DZ_pixel = data[1].flatten()/data[0].flatten()
		binning = ret.binnumber
		pixel_num = np.arange(len(DZ_pixel.flatten()))
		mean_DZ = np.zeros(len(pixel_num))
		std_DZ = np.zeros([len(pixel_num),2])
		for y in range(len(y_vals)):
			for x in range(len(x_vals)):
				binx = binning[0]; biny = binning[1]
				in_pixel = np.logical_and(binx == x+1, biny == y+1)
				values = dust_mass[in_pixel]/Z_mass[in_pixel]
				weights = M[in_pixel]
				mean_DZ[y*len(x_vals)+x],std_DZ[y*len(x_vals)+x,0],std_DZ[y*len(x_vals)+x,1] = utils.weighted_percentile(values, weights=weights)

		# Now set pixel indices so we start at the center pixel and spiral outward
		N = np.sqrt(np.shape(pixel_num)[0])
		startx = N/2; starty = startx
		dirs = [(0, -1), (-1, 0), (0, 1), (1, 0)]
		x, y = startx, starty
		size = N * N
		k, indices = 0, []
		while len(indices) < size:
			for l in xrange((k % 2) * 2, (k % 2) * 2 + 2):
				dx, dy = dirs[l]
				for _ in xrange(k + 1):
					if 0 <= x < N and 0 <= y < N:
						indices += [int(y*N+x)]
					x, y = x + dx, y + dy
			k+=1

		axis.errorbar(pixel_num, mean_DZ[indices], yerr = np.abs(mean_DZ[indices]-np.transpose(std_DZ[indices])), c=colors[j], fmt=MARKER_STYLE[0], elinewidth=1, markersize=2)
		axis.plot(pixel_num, DZ_pixel[indices], label=labels[j], linestyle=linestyles[j], color=colors[j], linewidth=linewidths[j])
		

	axis.legend(loc=0, fontsize=utils.SMALL_FONT, frameon=False)
	plt.savefig(foutname)
	plt.close()	



def dust_acc_diag(params, snaps, bin_nums=100, labels=None, foutname='dust_acc_diag.png', style='color', implementation='species'):
	"""
	Make plot of instantaneous dust growth for a given snapshot depending on the dust evolution implementation used

	Parameters
	----------
	params : array
		List of parameters to plot diagnostics for (inst_dust_prod, growth_timescale, )
	snaps : array
	    Array of snapshots to use
	bin_nums: int
		Number of bins to use
	style : string
		Plotting style when plotting multiple data sets
		'color' - gives different color and linestyles to each data set
		'size' - make all lines solid black but with varying line thickness

	Returns
	-------
	None
	"""

	# Get plot stylization
	linewidths,colors,linestyles = plt_set.setup_plot_style(len(snaps), style=style)

	# Set up subplots based on number of parameters given
	fig,axes = plt_set.setup_figure(len(params))

	for i,param in enumerate(params):
		axis = axes[i]
		if param == 'inst_dust_prod':
			plt_set.setup_axis(axis, 'nH', param)
		if param == 'g_timescale':
			plt_set.setup_axis(axis, param, 'g_timescale_frac')


		for j,snap in enumerate(snaps):
			if isinstance(implementation, list):
				imp = implementation[j]
			else:
				imp = implementation

			G =	snap.loadpart(0)
			H = snap.loadheader()

			nH = G.rho*config.UnitDensity_in_cgs * ( 1. - (G.z[:,0]+G.z[:,1])) / config.H_MASS

			if param == 'inst_dust_prod':
				weight_vals = calc_dust_acc(G,implementation=imp, CNM_thresh=1.0, CO_frac=0.2, nano_iron=False)
				x_vals = dict.fromkeys(weight_vals.keys(), nH)
			elif param == 'g_timescale':
				if imp == 'species':
					x_vals = calc_spec_acc_timescale(G, CNM_thresh=1.0, nano_iron=False)
				else:
					x_vals = calc_elem_acc_timescale(G)
				for key in x_vals.keys(): 
					x_vals[key]*=1E-9
				weight_vals=dict.fromkeys(x_vals.keys(), np.full(len(nH),1./len(nH)))
			else:
				print('%s is not a valid parameter for dust_growth_diag()'%param)
				return
			lines = []
			for k in range(len(labels)):
				lines += [mlines.Line2D([], [], color=colors[k], label=labels[k])]

			# Set up bins based on limits and scale of x axis
			limits = axis.get_xlim()
			scale_str = axis.get_xaxis().get_scale()
			if scale_str == 'log':
				bins = np.logspace(np.log10(limits[0]),np.log10(limits[1]),bin_nums)
			else:
				bins = np.linspace(limits[0],limits[1],bin_nums)

			for k,key in enumerate(sorted(weight_vals.keys())):
				axis.hist(x_vals[key], bins=bins, weights=weight_vals[key], histtype='step', cumulative=True, label=labels[j], color=colors[j], \
				         linewidth=linewidths[0], linestyle=linestyles[k])
				lines += [mlines.Line2D([], [], color='xkcd:black', linestyle =linestyles[k],label=key)]

			# Want legend only on first plot
			if i == 0:
				axis.legend(handles=lines,loc=2, frameon=False)


	plt.savefig(foutname)
	plt.close()   

