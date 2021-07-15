import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colorbar import Colorbar
from scipy.stats import binned_statistic_2d
import pickle
import os

import observations.dust_obs as obs
import analytical_models.stellar_yields as st_yields
import plot_setup as plt_set
import gizmo_library.config as config
import gizmo_library.utils as utils
import calculate as calc




def plot_observational_data(axis, property, elem=None, log=True, CO_opt='B13', goodSNR=True):
	"""
	Plots observational D/Z data vs the given property.

	Parameters
	----------
	axis : Matplotlib axis
		Axis on which to plot the data
	property: string
		Parameters to plot D/Z against (fH2, nH, Z, r, sigma_dust)
	log : boolean
		Plot on log scale
	good_SNR : boolean
		Restrict data to only that with good signal-to-noise if applicable

	Returns
	-------
	None

	"""

	if property == 'fH2':
		data = obs.Chiang_20_DZ_vs_param(property, CO_opt=CO_opt, bin_nums=30, log=True, goodSNR=goodSNR)
		for i, gal_name in enumerate(data.keys()):
			fH2_vals = data[gal_name][0]; mean_DZ = data[gal_name][1]; std_DZ = data[gal_name][2]
			if log:
				std_DZ[std_DZ == 0] = config.EPSILON
			axis.errorbar(fH2_vals, mean_DZ, yerr = np.abs(mean_DZ-np.transpose(std_DZ)), label=gal_name, c=config.MARKER_COLORS[i],
						  fmt=config.MARKER_STYLES[i], elinewidth=config.BASE_ELINEWIDTH, ms=config.BASE_MARKERSIZE, zorder=2)

	elif property == 'r':
		data = obs.Chiang_20_DZ_vs_param(property, bin_data=True, CO_opt=CO_opt, phys_r=True, bin_nums=15, log=False, goodSNR=goodSNR)
		for i, gal_name in enumerate(data.keys()):
			r_vals = data[gal_name][0]; mean_DZ = data[gal_name][1]; std_DZ = data[gal_name][2]
			if log:
				std_DZ[std_DZ == 0] = config.EPSILON
			axis.errorbar(r_vals, mean_DZ, yerr = np.abs(mean_DZ-np.transpose(std_DZ)), label=gal_name, c=config.MARKER_COLORS[i],
						  fmt=config.MARKER_STYLES[i], elinewidth=config.BASE_ELINEWIDTH, ms=config.BASE_MARKERSIZE, zorder=2)

	elif property == 'r25':
		data = obs.Chiang_20_DZ_vs_param('r', bin_data=True, CO_opt=CO_opt, phys_r=False, bin_nums=30, log=False, goodSNR=goodSNR)
		for i, gal_name in enumerate(data.keys()):
			r_vals = data[gal_name][0]; mean_DZ = data[gal_name][1]; std_DZ = data[gal_name][2]
			if log:
				std_DZ[std_DZ == 0] = config.EPSILON
			axis.errorbar(r_vals, mean_DZ, yerr = np.abs(mean_DZ-np.transpose(std_DZ)), label=gal_name, c=config.MARKER_COLORS[i], fmt=config.MARKER_STYLES[i],
						   elinewidth=config.BASE_ELINEWIDTH, ms=config.BASE_MARKERSIZE,zorder=2)

	elif property == 'nH':
		# Plot J09 with Zhuk16/18 conversion to physical density
		dens_vals, DZ_vals = obs.Jenkins_2009_DZ_vs_dens(phys_dens=True, C_corr=True)
		axis.plot(dens_vals, DZ_vals, label=r'J09 $n_{\rm H}$', c='xkcd:black', linestyle=config.LINE_STYLES[0], linewidth=config.BASE_LINEWIDTH, zorder=2)
		# Add data point for WNM depletion from Jenkins (2009) comparison to Savage and Sembach (1996) with error bars accounting for possible
		# range of C depeletions since they are not observed for this regime
		nH_val, WNM_depl, WNM_error = obs.Jenkins_Savage_2009_WNM_Depl('Z', C_corr=True)
		axis.errorbar(nH_val, 1.-WNM_depl, yerr=WNM_error, label='WNM', c='xkcd:black', fmt='D',  elinewidth=config.BASE_ELINEWIDTH, ms=config.BASE_MARKERSIZE ,zorder=2)
		axis.plot(np.logspace(np.log10(nH_val), np.log10(dens_vals[0])),np.logspace(np.log10(1.-WNM_depl), np.log10(DZ_vals[0])),
				  c='xkcd:black', linestyle=':', linewidth=config.BASE_LINEWIDTH, zorder=2)
		# Plot J09 relation to use as upper bound
		dens_vals, DZ_vals = obs.Jenkins_2009_DZ_vs_dens(phys_dens=False, C_corr=True)
		axis.plot(dens_vals, DZ_vals, label=r'J09 $\left< n_{\rm H} \right>$', c='xkcd:black', linestyle=config.LINE_STYLES[1], linewidth=config.BASE_LINEWIDTH, zorder=2)
	
	elif property == 'sigma_dust':
		data = obs.Chiang_20_DZ_vs_param(property, bin_data=True, CO_opt=CO_opt, bin_nums=30, log=True, goodSNR=goodSNR)
		for i, gal_name in enumerate(data.keys()):
			sigma_vals = data[gal_name][0]; mean_DZ = data[gal_name][1]; std_DZ = data[gal_name][2]
			if log:
				std_DZ[std_DZ == 0] = config.EPSILON
			axis.errorbar(sigma_vals, mean_DZ, yerr = np.abs(mean_DZ-np.transpose(std_DZ)), label=gal_name, c=config.MARKER_COLORS[i],
						  fmt=config.MARKER_STYLES[i], elinewidth=config.BASE_ELINEWIDTH, ms=config.BASE_MARKERSIZE, zorder=2)

	elif property == 'sigma_gas':
		data = obs.Chiang_20_DZ_vs_param(property, bin_data=True, CO_opt=CO_opt, bin_nums=15, log=True, goodSNR=True)
		for i, gal_name in enumerate(data.keys()):
			sigma_vals = data[gal_name][0]; mean_DZ = data[gal_name][1]; std_DZ = data[gal_name][2]
			if log:
				std_DZ[std_DZ == 0] = config.EPSILON
			axis.errorbar(sigma_vals, mean_DZ, yerr = np.abs(mean_DZ-np.transpose(std_DZ)), label=gal_name, c=config.MARKER_COLORS[i],
						  fmt=config.MARKER_STYLES[i], elinewidth=config.BASE_ELINEWIDTH, ms=config.BASE_MARKERSIZE, zorder=2)
		if not goodSNR:
			data = obs.Chiang_20_DZ_vs_param(property, bin_data=False, CO_opt=CO_opt, log=True, goodSNR=False)
			for i, gal_name in enumerate(data.keys()):
				sigma_vals = data[gal_name][0]; DZ = data[gal_name][1]
				axis.scatter(sigma_vals, DZ, c=config.MARKER_COLORS[i], marker=config.MARKER_STYLES[i], s=config.BASE_MARKERSIZE**2, zorder=0, alpha=0.4)
	
	elif property == 'sigma_H2':
		data = obs.Chiang_20_DZ_vs_param(property, bin_data=True, CO_opt=CO_opt, bin_nums=30, log=True, goodSNR=goodSNR)
		for i, gal_name in enumerate(data.keys()):
			sigma_vals = data[gal_name][0]; mean_DZ = data[gal_name][1]; std_DZ = data[gal_name][2]
			if log:
				std_DZ[std_DZ == 0] = config.EPSILON
			axis.errorbar(sigma_vals, mean_DZ, yerr = np.abs(mean_DZ-np.transpose(std_DZ)), label=gal_name, c=config.MARKER_COLORS[i],
						  fmt=config.MARKER_STYLES[i], elinewidth=config.BASE_ELINEWIDTH, ms=config.BASE_MARKERSIZE, zorder=2)

	elif property == 'depletion':
		if elem == 'C':
			# J09 C relation is derived from only a handful of sightlines over a limited range so just plot the relation over the observed range and no
			# WNM depletion approximation
			dens_vals, DZ_vals = obs.Jenkins_2009_DZ_vs_dens(elem=elem, phys_dens=False, C_corr=True)
			axis.plot(dens_vals, 1.-DZ_vals, label=r'J09 $\left< n_{\rm H} \right>$', c='xkcd:black', linestyle=config.LINE_STYLES[1], linewidth=config.BASE_LINEWIDTH, zorder=2)
			dens_vals, DZ_vals = obs.Jenkins_2009_DZ_vs_dens(elem=elem, phys_dens=True, C_corr=True)
			axis.plot(dens_vals, 1.-DZ_vals, label=r'J09 $n_{\rm H}$', c='xkcd:black', linestyle=config.LINE_STYLES[0], linewidth=config.BASE_LINEWIDTH, zorder=2)
			# Add in data from Parvathi which sampled twice as many sightlines 
			C_depl, C_error, nH_vals = obs.Parvathi_2012_C_Depl(solar_abund='max')
			axis.errorbar(nH_vals,C_depl, yerr = C_error, label='Parvathi+12', fmt='o', c='xkcd:black', elinewidth=config.BASE_ELINEWIDTH, ms=config.BASE_MARKERSIZE, zorder=2)
		else:
			dens_vals, DZ_vals = obs.Jenkins_2009_DZ_vs_dens(elem=elem, phys_dens=False, C_corr=False)
			axis.plot(dens_vals, 1.-DZ_vals, label=r'J09 $\left< n_{\rm H} \right>$', c='xkcd:black', linestyle=config.LINE_STYLES[1], linewidth=config.BASE_LINEWIDTH, zorder=2)
			dens_vals, DZ_vals = obs.Jenkins_2009_DZ_vs_dens(elem=elem, phys_dens=True, C_corr=False)
			axis.plot(dens_vals, 1.-DZ_vals, label=r'J09 $n_{\rm H}$', c='xkcd:black', linestyle=config.LINE_STYLES[0], linewidth=config.BASE_LINEWIDTH, zorder=2)
			# Add data point for WNM depletion from Jenkins (2009) comparison to Savage and Sembach (1996)
			nH_val, WNM_depl,_ = obs.Jenkins_Savage_2009_WNM_Depl(elem, C_corr=False)
			axis.scatter(nH_val,WNM_depl, marker='D',c='xkcd:black', zorder=2, label='WNM', s=config.BASE_MARKERSIZE**2)
			axis.plot(np.logspace(np.log10(nH_val), np.log10(dens_vals[0])),np.logspace(np.log10(WNM_depl), np.log10(1-DZ_vals[0])), c='xkcd:black', linestyle=':', linewidth=config.BASE_LINEWIDTH, zorder=2)

	elif property == 'Z' or property=='O/H':
		if property == 'Z':
			key_name = 'metal_z'; unit_conv = config.SOLAR_Z
		else:
			key_name = 'metal'; unit_conv = 1.
		data = obs.galaxy_integrated_DZ('R14')
		Z_vals = data[key_name].values/unit_conv; DZ_vals = data['dtm'].values
		if log:
			DZ_vals[DZ_vals == 0] = config.EPSILON
		axis.scatter(Z_vals, DZ_vals, label='Rémy-Ruyer+14', s=config.BASE_MARKERSIZE**2, marker=config.MARKER_STYLES[0],
					 facecolors='none', linewidths=config.LINE_WIDTHS[1], edgecolors=config.MARKER_COLORS[0], zorder=2)

		data = obs.galaxy_integrated_DZ('DV19')
		Z_vals = data[key_name].values/unit_conv; DZ_vals = data['dtm'].values
		if log:
			DZ_vals[DZ_vals == 0] = config.EPSILON
		axis.scatter(Z_vals, DZ_vals, label='De Vis+19', s=config.BASE_MARKERSIZE**2, marker=config.MARKER_STYLES[1], facecolors='none',
					 linewidths=config.LINE_WIDTHS[1], edgecolors=config.MARKER_COLORS[1], zorder=2)

		data = obs.galaxy_integrated_DZ('PH20')
		Z_vals = data[key_name].values/unit_conv; DZ_vals = data['dtm'].values
		lim_mask = data['limit'].values==1
		if log:
			DZ_vals[DZ_vals == 0] = config.EPSILON
		axis.scatter(Z_vals[~lim_mask], DZ_vals[~lim_mask], label='Péroux & Howk 19', s=config.BASE_MARKERSIZE**2,
					 marker=config.MARKER_STYLES[2], facecolors='none', linewidths=config.LINE_WIDTHS[1],
					 edgecolors=config.MARKER_COLORS[2], zorder=2)
		yerr = DZ_vals[lim_mask]*(1-10**-0.1) # Set limit bars to be the same size in log space
		axis.errorbar(Z_vals[lim_mask], DZ_vals[lim_mask], yerr=yerr, uplims=True, ms=config.BASE_MARKERSIZE, mew=config.LINE_WIDTHS[1],
					  fmt=config.MARKER_STYLES[2], mfc='none', mec=config.MARKER_COLORS[2], elinewidth=config.LINE_WIDTHS[1],
					  ecolor=config.MARKER_COLORS[2], zorder=2)

	else:
		print("D/Z vs %s observational data is not available."%property)
		return None

	return



def galaxy_int_DZ_vs_prop(properties, snaps, labels=None, foutname='gal_int_DZ_vs_param.png', style='color', include_obs=True):
	"""
	Plots the galaxy integrate dust-to-metals ratio (D/Z) vs given parameters for multiple simulations/snapshots

	Parameters
	----------
	properties: list
		List of properties to plot D/Z against (fH2, Mgas, Z)
	snaps : list
	    List of snapshots to plot
	labels : list
		List of labels for each data set
	foutname: str, optional
		Name of file to be saved
	style : string, optional
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
	fig,axes = plt_set.setup_figure(len(properties))

	labels_handles = {}
	for i, x_prop in enumerate(properties):
		# Set up for each plot
		axis = axes[i]
		y_prop = 'D/Z'
		plt_set.setup_axis(axis, x_prop, y_prop, y_lim = [5E-3,5], y_log=True)

		# First plot observational data if applicable
		if include_obs: plot_observational_data(axis, x_prop, goodSNR=True);

		for j,snap in enumerate(snaps):
			G = snap.loadpart(0)
			DZ_val = calc.calc_gal_int_params(y_prop, G)
			prop_val = calc.calc_gal_int_params(x_prop, G)

			print(prop_val,DZ_val)

			axis.scatter(prop_val, DZ_val, label=labels[j], marker='o', color=colors[j], s=(1.1*config.BASE_MARKERSIZE)**2, zorder=3)

		# Check labels and handles between this and last axis. Any differences should be added to a new legend
		hands, labs = axis.get_legend_handles_labels()
		new_lh = dict(zip(labs, hands))
		for key in labels_handles.keys(): new_lh.pop(key,0);
		if len(new_lh)>0:
			ncol = 2 if len(new_lh) > 4 else 1
			axis.legend(new_lh.values(), new_lh.keys(), loc='best', fontsize=config.SMALL_FONT, frameon=False, ncol=ncol)
		labels_handles = dict(zip(labs, hands))

	plt.tight_layout()
	plt.savefig(foutname)
	plt.close()

	return



def plot_prop_vs_prop(xprops, yprops, snaps, bin_nums=50, labels=None,
					  foutname='prop_vs_prop.png', std_bars=True, style='color-line',
					  include_obs=True):
	"""
	Plots the between two properties for multiple simulations/snapshots

	Parameters
	----------
	xprops: list
		List of x-axis properties to plot
	yprops: list
		List of y-axis properties to plot
	snaps : list
	    List of snapshots to plot
	bin_nums : int, optional
		Number of bins to use
	labels : list, optional
		List of labels for each snapshot
	foutname: string, optional
		Name of file to be saved
	std_bars : boolean
		Include standard deviation bars for the data
	style : string
		Plotting style when plotting multiple data sets
		'color-line' - gives different color and linestyles to each data set
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
	fig,axes = plt_set.setup_figure(len(xprops), sharey=True)

	labels_handles = {}

	for i, x_prop in enumerate(xprops):
		# Set up for each plot
		axis = axes[i]
		y_prop = yprops[i]
		plt_set.setup_axis(axis, x_prop, y_prop)

		# First plot observational data if applicable
		if include_obs and y_prop=='D/Z': plot_observational_data(axis, x_prop, goodSNR=True);

		for j,snap in enumerate(snaps):
			G = snap.loadpart(0)
			x_vals,y_mean,y_std = calc.calc_binned_property_vs_property(y_prop, x_prop, G, bin_nums=bin_nums)

			# Only need to label the separate simulations in the first plot
			if i==0 and labels is not None: label = labels[j];
			else:    						label = None;
			axis.plot(x_vals, y_mean, label=label, linestyle=linestyles[j], color=colors[j], linewidth=linewidths[j], zorder=3)
			if std_bars:
				axis.fill_between(x_vals, y_std[:,0], y_std[:,1], alpha = 0.3, color=colors[j], zorder=1)

		# Check labels and handles between this and last axis. Any differences should be added to a new legend
		hands, labs = axis.get_legend_handles_labels()
		new_lh = dict(zip(labs, hands))
		for key in labels_handles.keys(): new_lh.pop(key,0);
		if len(new_lh)>0:
			ncol = 2 if len(new_lh) > 4 else 1
			axis.legend(new_lh.values(), new_lh.keys(), loc='best', fontsize=config.SMALL_FONT, frameon=False, ncol=ncol)
		labels_handles = dict(zip(labs, hands))

	plt.tight_layout()
	plt.savefig(foutname)
	plt.close()

	return



def plot_elem_depletion_vs_prop(elems, prop, snaps, bin_nums=50, labels=None, \
			foutname='obs_elem_dep_vs_dens.png', std_bars=True, style='color', include_obs=True):
	"""
	Plots binned relations of specified elemental depletion vs various properties for multiple simulations

	Parameters
	----------
	elems : list
		List of which elements you want to plot depletions for
	prop: string
		Property to plot depletion against (fH2, nH)
	snaps : list
	    List of snapshots to plot
	bin_nums : int, optional
		Number of bins to use
	labels : list, optional
		List of labels for each data set
	foutname: string, optional
		Name of file to be saved
	std_bars : boolean, optional
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

	labels_handles = {}
	for i,elem in enumerate(elems):
		axis = axes[i]
		plt_set.setup_axis(axis, prop, elem+'_depletion')

		if include_obs and prop == 'nH':
			plot_observational_data(axis, property='depletion', elem=elem)

		for j,snap in enumerate(snaps):
			G = snap.loadpart(0)
			prop_vals,mean_DZ,std_DZ = calc.calc_binned_property_vs_property(elem+'_depletion', prop, G, bin_nums=bin_nums)

			axis.plot(prop_vals, 1.-mean_DZ, label=labels[j], linestyle=linestyles[j], color=colors[j], linewidth=linewidths[j], zorder=3)
			if std_bars:
				axis.fill_between(prop_vals, 1.-std_DZ[:,0], 1.-std_DZ[:,1], alpha = 0.3, color=colors[j], zorder=1)

		# Check labels and handles between this and last axis. Any differences should be added to a new legend
		hands, labs = axis.get_legend_handles_labels()
		new_lh = dict(zip(labs, hands))
		for key in labels_handles.keys(): new_lh.pop(key,0);
		if len(new_lh)>0:
			ncol = 2 if len(new_lh) > 4 else 1
			axis.legend(new_lh.values(), new_lh.keys(), loc='lower left', fontsize=config.SMALL_FONT, frameon=False, ncol=ncol)
		labels_handles = dict(zip(labs, hands))

		# Add label for each element
		axis.text(.10, .4, elem, color=config.BASE_COLOR, fontsize = 2*config.LARGE_FONT, ha = 'center', va = 'center', transform=axis.transAxes)

	plt.tight_layout()
	plt.savefig(foutname)
	plt.close()

	return



def binned_phase_plot(prop, snaps, bin_nums=200, labels=None, color_map=config.BASE_CMAP, foutname='phase_plot.png'):
	"""
	Plots the a 2D histogram for nH vs T using the specified parameter as weights for the given snaps

	Parameters
	----------
	prop : string
		The property for the x axis, y axis, and weights respectively
	snaps : list
	     List of Snapshot/Halo/Disk objects
	bin_nums: int, optional
		Number of bins to use
	labels: list, optional
		List of labels for each snap
	color_map : string, optional
		Color mapping for plot
	founame : string, optional
		File name for saved figure

	Returns
	-------
	None

	"""

	fig,axes = plt_set.setup_figure(len(snaps), sharey='row')

	for i, snap in enumerate(snaps):
		# Set up for each plot
		axis = axes[i]
		plt_set.setup_axis(axis, 'nH', 'T', x_lim=[1.1E-3,0.9E3], y_lim=[1.1*1E1,2E6])
		axis.set_facecolor('xkcd:light grey')

		G = snap.loadpart(0)

		ret = calc.calc_phase_hist_data(prop, G, bin_nums=bin_nums)
		prop_lims = config.PROP_INFO[prop][1]
		log_param = config.PROP_INFO[prop][2]
		if log_param:
			norm = mpl.colors.LogNorm(vmin=prop_lims[0], vmax=prop_lims[1], clip=True)
		else:
			norm = mpl.colors.Normalize(vmin=prop_lims[0], vmax=prop_lims[1], clip=True)

		X, Y = np.meshgrid(ret.x_edge, ret.y_edge)
		img = axis.pcolormesh(X, Y, ret.statistic.T, cmap=plt.get_cmap(color_map), norm=norm)
		axis.autoscale('tight')

		# Print label in corner of plot if applicable
		if labels!=None:
			label = labels[i]
			axis.text(.95, .95, label, color=config.BASE_COLOR, fontsize=config.EXTRA_LARGE_FONT, ha='right', va='top', transform=axis.transAxes, zorder=4)

	# Add color bar to last axis
	bar_label =  config.PROP_INFO[prop][0]
	plt_set.setup_colorbar(img, axes[-1], bar_label)

	plt.tight_layout()
	plt.savefig(foutname)
	plt.close()

	return




def plot_obs_prop_vs_prop(xprops, yprops, snaps, pixel_res=2, bin_nums=50, labels=None, foutname='obs_DZ_vs_param.png', \
						 std_bars=True, style='color-linestyle', include_obs=True):
	"""
	Plots mock observations of one property versus another for multiple snapshots

	Parameters
	----------
	xprops: list
		List of x-axis properties to plot
	yprops: list
		List of y-axis properties to plot
	snaps : list
	    List of snapshots to plot
	pixel_res : double, optional
		Resolution of simulated pixels in kpc
	bin_nums : int, optional
		Number of bins to use
	labels : list, optional
		Array of labels for each data set
	foutname: string, optional
		Name of file to be saved
	std_bars : boolean, optional
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
	fig,axes = plt_set.setup_figure(len(xprops), sharey=True)

	labels_handles = {}
	for i, x_prop in enumerate(xprops):
		# Set up for each plot
		axis = axes[i]
		y_prop = yprops[i]
		plt_set.setup_axis(axis, x_prop, y_prop)

		# First plot observational data if applicable
		if include_obs: plot_observational_data(axis, x_prop, goodSNR=True);

		for j,snap in enumerate(snaps):
			G = snap.loadpart(0)
			try:
				r_max = snap.get_rmax()
			except:
				r_max = 20
			x_vals,y_mean,y_std = calc.calc_binned_obs_property_vs_property(y_prop,x_prop, G, r_max=r_max, bin_nums=bin_nums, pixel_res=pixel_res)

			# Only need to label the separate simulations in the first plot
			axis.plot(x_vals, y_mean, label=labels[j], linestyle=linestyles[j], color=colors[j], linewidth=linewidths[j], zorder=3)
			if std_bars:
				axis.fill_between(x_vals, y_std[:,0], y_std[:,1], alpha = 0.3, color=colors[j], zorder=1)

		# Check labels and handles between this and last axis. Any differences should be added to a new legend
		hands, labs = axis.get_legend_handles_labels()
		new_lh = dict(zip(labs, hands))
		for key in labels_handles.keys(): new_lh.pop(key,0);
		if len(new_lh)>0:
			ncol = 2 if len(new_lh) > 4 else 1
			axis.legend(new_lh.values(), new_lh.keys(), loc='best', fontsize=config.SMALL_FONT, frameon=False, ncol=ncol)
		labels_handles = dict(zip(labs, hands))

	plt.tight_layout()	
	plt.savefig(foutname)
	plt.close()	

	return




def dmol_vs_props(mol_params, properties, snaps, labels=None, bin_nums=50,
				   foutname='dmol_vs_props.png', std_bars=True):
	"""
	Plots the molecular parameters (fH2, fMC, CinCO) vs the parameters given for one snapshot

	Parameters
	----------
	mol_params: list
		List of molecular parameter to plot (fH2, fMC, CinCO) on y axis
	properties: list
		List of parameters to plot D/Z against (fH2, nH, Z, r, r25) on x axis
	snaps : list
	    List of snapshots to plot
	bin_nums : int
		Number of bins to use
	labels : list
		list of labels for each data set
	foutname: str, optional
		Name of file to be saved
	std_bars : bool
		Include standard deviation bars for the data
	style : string
		Plotting style when plotting multiple data sets
		'color' - gives different color and linestyles to each data set
		'size' - make all lines solid black but with varying line thickness

	Returns
	-------
	None

	"""

	# Get plot stylization
	linewidths,colors,linestyles = plt_set.setup_plot_style(len(snaps), properties=mol_params, style='color-linestyle')
	# Set up subplots based on number of parameters given
	fig,axes = plt_set.setup_figure(len(properties), sharey=True)

	for i, x_prop in enumerate(properties):
		# Set up axes and label legends for each plot
		axis = axes[i]
		if len(mol_params) > 1:
			y_prop = 'mass_frac'
		else:
			y_prop = mol_params[0]

		plt_set.setup_axis(axis, x_prop, y_prop)

		for j,snap in enumerate(snaps):
			G = snap.loadpart(0)
			for k,mol_param in enumerate(mol_params):
				if len(mol_params) == 1:
					index = j
				else:
					index = j*len(mol_params)+k

				x_vals,y_mean,y_std = calc.calc_binned_property_vs_property(mol_param, x_prop, G, bin_nums=bin_nums)
				axis.plot(x_vals, y_mean, color=colors[index], linestyle=linestyles[index], linewidth=linewidths[index], zorder=3)
				if std_bars:
					axis.fill_between(x_vals, y_std[:,0], y_std[:,1], alpha = 0.3, color=colors[index], zorder=1)

		# Need to manually set legend handles
		handles = []
		for j, prop in enumerate(mol_params):
			handles += [mlines.Line2D([], [], color=config.BASE_COLOR, linewidth=config.BASE_LINEWIDTH, linestyle=config.LINE_STYLES[j],label=config.PROP_INFO[prop][0])]
		for j, label in enumerate(labels):
			handles += [mlines.Line2D([], [], color=config.LINE_COLORS[j], linewidth=config.BASE_LINEWIDTH, linestyle=config.BASE_LINESTYLE,label=label)]
		# Setup legend on first axis
		if i == 0 and len(handles)>1:
			axis.legend(handles=handles, loc=0, fontsize=config.SMALL_FONT, frameon=False)

	plt.tight_layout()
	plt.savefig(foutname)
	plt.close()

	return



def dust_data_vs_time(params, data_objs, foutname='dust_data_vs_time.png',labels=None, style='color', time=True):
	"""
	Plots all time averaged data vs time from precompiled data for a set of simulation runs

	Parameters
	----------
	params : list
		List of parameters to plot over time
	data_objs : list
		List of Dust_Evo objects with data to plot
	foutname: str, optional
		Name of file to be saved

	Returns
	-------
	None
	"""

	# Get plot stylization
	linewidths,colors,linestyles = plt_set.setup_plot_style(len(data_objs), style=style)

	# Set up subplots based on number of parameters given
	fig,axes = plt_set.setup_figure(len(params), orientation='vertical', sharex=True)

	for i, y_param in enumerate(params):
		# Set up for each plot
		axis = axes[i]		
		if time:
			x_param = 'time'
		else:
			x_param = 'redshift'
		plt_set.setup_axis(axis, x_param, y_param)

		param_labels=None
		if y_param == 'D/Z':
			loc = 'upper left'
		elif y_param == 'source_frac':
			param_labels = config.DUST_SOURCES
			loc = 'lower left'
		elif y_param=='spec_frac':
			param_labels = config.DUST_SPECIES
			loc = 'upper right'
		elif y_param == 'Si/C':
			loc = 'upper right'
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
				axis.plot(time_data, data_vals, color=config.BASE_COLOR, linestyle=config.LINE_STYLES[j], label=labels[j], linewidth=config.BASE_LINEWIDTH, zorder=3)
			else:
				for k in range(np.shape(data_vals)[1]):
					axis.plot(time_data, data_vals[:,k], color=config.LINE_COLORS[k], linestyle=config.LINE_STYLES[j], linewidth=config.BASE_LINEWIDTH, zorder=3)
			axis.set_xlim([time_data[1],time_data[-1]])
		# Only need to label the seperate simulations in the first plot
		if i==0 and len(data_objs)>1:
			axis.legend(loc='upper left', frameon=False, fontsize=config.SMALL_FONT)
		# If there are subparameters need to make their own legend
		if param_labels is not None:
			param_labels = param_labels[:np.shape(data_vals)[1]]
			param_lines = []
			for j, label in enumerate(param_labels):
				param_lines += [mlines.Line2D([], [], color=config.LINE_COLORS[j], label=label, linewidth=config.BASE_LINEWIDTH,)]
			axis.legend(handles=param_lines, loc=loc, frameon=False, fontsize=config.SMALL_FONT)

	plt.tight_layout()
	plt.savefig(foutname)
	plt.close()





def compare_dust_creation(Z_list, dust_species, data_dirc, FIRE_ver=2, style='color-linestyle',
						  foutname='creation_routine_compare.png', reload=False):
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
	reload : boolean
		Recalculate the simulated dust data instead of loading from any saved files

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

	time_step = max_t/(N+1)
	time = np.arange(time_step,max_t,time_step)

	# First make simulated data if it hasn't been made already
	for Z in Z_list:
		name = '/FIRE'+str(FIRE_ver)+'_elem_Z_'+str(Z).replace('.','-')+'_cum_yields.pickle'
		if not os.path.isfile(data_dirc + name) or reload:
			cum_yields, cum_dust_yields, cum_species_yields = st_yields.totalStellarYields(max_t,N,Z,FIRE_ver=FIRE_ver,routine="elemental")
			pickle.dump({"time": time, "yields": cum_yields, "elem": cum_dust_yields, "spec": cum_species_yields}, open(data_dirc + name, "wb" ))

		name = '/FIRE'+str(FIRE_ver)+'_spec_Z_'+str(Z).replace('.','-')+'_cum_yields.pickle'
		if not os.path.isfile(data_dirc +name) or reload:
			cum_yields, cum_dust_yields, cum_species_yields = st_yields.totalStellarYields(max_t,N,Z,FIRE_ver=FIRE_ver,routine="species")
			pickle.dump({"time": time, "yields": cum_yields, "elem": cum_dust_yields, "spec": cum_species_yields}, open(data_dirc + name, "wb" ))


	# Set transition age which is different for each version of FIRE
	transition_age = 0.03753 if FIRE_ver<=2 else 0.044


	# Compare routine carbon yields between routines
	# Set up subplots based on number of parameters given
	fig,axes = plt_set.setup_figure(len(dust_species), sharey=True)
	x_param = 'star_age'; y_param = 'cum_dust_prod'
	x_lim = [time_step,max_t]
	x_lim=None

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
		axis.axvline(transition_age, color='xkcd:grey',lw=config.BASE_ELINEWIDTH)
		# Only need labels and legend for first plot
		if i == 0:
			y_arrow = 1E-5
			axis.annotate('AGB+SNe Ia', va='center', xy=(transition_age, y_arrow), xycoords="data", xytext=(2*transition_age, y_arrow), 
			            arrowprops=dict(arrowstyle='<-',color='xkcd:grey', lw=config.BASE_ELINEWIDTH), size=config.SMALL_FONT*1.1, color='xkcd:grey')
			axis.annotate('SNe II', va='center', xy=(transition_age, y_arrow/3), xycoords="data", xytext=(0.15*transition_age*1.1, y_arrow/3),
			            arrowprops=dict(arrowstyle='<-',color='xkcd:grey', lw=config.BASE_ELINEWIDTH), size=config.SMALL_FONT*1.1, color='xkcd:grey')
			# Make legend
			lines = []
			for j in range(len(Z_list)):
				lines += [mlines.Line2D([], [], color=colors[j], label=r'Z = %.2g $Z_{\odot}$' % Z_list[j], linewidth=config.BASE_LINEWIDTH)]
			lines += [mlines.Line2D([], [], color=config.BASE_COLOR, linestyle=linestyles[0], label='Elemental', linewidth=config.BASE_LINEWIDTH),
					  mlines.Line2D([], [], color=config.BASE_COLOR, linestyle=linestyles[1],label='Species', linewidth=config.BASE_LINEWIDTH)]
			legend = axis.legend(handles=lines, frameon=True, ncol=2, loc='center left', bbox_to_anchor=(0.025,1.0), framealpha=1, fontsize=config.SMALL_FONT*0.9, edgecolor=config.BASE_COLOR)
			legend.get_frame().set_lw(config.AXIS_BORDER_WIDTH)
		#  Add label for dust species
		axis.text(.95, .05, name, color=config.BASE_COLOR, fontsize = config.LARGE_FONT, ha = 'right', transform=axis.transAxes)

		for j,Z in enumerate(Z_list):
			name = '/FIRE'+str(FIRE_ver)+'_elem_Z_'+str(Z).replace('.','-')+'_cum_yields.pickle'
			data = pickle.load(open(data_dirc + name, "rb" ))
			time = data['time']; cum_yields = data['yields']; cum_dust_yields = data['elem']; cum_species_yields = data['spec'];
			elem_cum_spec = np.sum(cum_species_yields[:,indices], axis=1)
			axis.loglog(time, elem_cum_spec, color = colors[j], linestyle = linestyles[0], nonpositive = 'clip', linewidth = linewidths[j])

			name = '/FIRE'+str(FIRE_ver)+'_spec_Z_'+str(Z).replace('.','-')+'_cum_yields.pickle'
			data = pickle.load(open(data_dirc + name, "rb" ))
			time = data['time']; cum_yields = data['yields']; cum_dust_yields = data['elem']; cum_species_yields = data['spec'];
			spec_cum_spec = np.sum(cum_species_yields[:,indices], axis=1)
			axis.loglog(time, spec_cum_spec, color = colors[j], linestyle = linestyles[1], nonpositive = 'clip', linewidth = linewidths[j])

		axis.set_ylim([1E-7,1E-2])

	plt.tight_layout()
	plt.savefig(foutname, transparent=False, bbox_inches='tight')
	plt.close()



def compare_FIRE_metal_yields(Z, elems, foutname='FIRE_yields_comparison.png'):
	

	# Get plot stylization
	linewidths,colors,linestyles = plt_set.setup_plot_style(len(elems))

	N = 10000 # number of steps 
	max_t = 10. # max age of stellar population to compute yields

	time_step = max_t/N
	time = np.arange(0,max_t,time_step)

	# Compare routine carbon yields between routines
	# Set up subplots based on number of parameters given
	fig,axes = plt_set.setup_figure(1)
	x_param = 'star_age'; y_param = 'cum_metal_yield'

	axis = axes[0]
	plt_set.setup_axis(axis, x_param, y_param)

	FIRE2_yields,_,_ = st_yields.onlySNeYields(max_t, N, Z, FIRE_ver=2)
	FIRE3_yields,_,_ = st_yields.onlySNeYields(max_t, N, Z, FIRE_ver=3)

	lines = []
	lines += [mlines.Line2D([], [], color=config.BASE_COLOR, linestyle=linestyles[0], label='FIRE-2', linewidth=linewidths[0]),
			  mlines.Line2D([], [], color=config.BASE_COLOR, linestyle=linestyles[1], label='FIRE-3', linewidth=linewidths[1])]

	for i, elem in enumerate(elems):
		elem_indx = config.ELEMENTS.index(elem)
		
		plt.plot(time,FIRE2_yields[:,elem_indx], c=colors[i], linestyle=linestyles[0], linewidth=linewidths[0])
		plt.plot(time,FIRE3_yields[:,elem_indx], c=colors[i], linestyle=linestyles[1], linewidth=linewidths[1])
		
		lines += [mlines.Line2D([], [], color=colors[i], label=elems[i], linewidth=linewidths[0])]
		

	axis.legend(handles=lines, loc='upper left', frameon=False, ncol=2, fontsize=config.SMALL_FONT)
	plt.savefig(foutname, transparent=False, bbox_inches='tight')
	plt.close()




def dust_projections(param, snap, bin_nums=100, param_lims=None):
	"""
	Face-on and edge-on projection of dust in galaxy snapshot.

	Parameters
	----------
	param: list
		List of dust parameters (D/Z,carbon,silicate,total)
	snap : dict
	    Snapshot gas data structure
	bin_nums : int
		Number of bins to use
	param_lims : list
		Limits for each dust parameter

	Returns
	-------
	None

	"""	
	fig,axes = plt_set.setup_2D_hist_fig(hist_proj=hist_proj)
	axis_2D_hist = axes[0]
	plt_set.setup_axis(axis_2D_hist, 'nH', 'T')
	axis_2D_hist.set_facecolor('xkcd:grey')

	G = snap.loadpart(0)
	H = snap.loadheader()

	ret = calc_phase_hist(param, G, bin_nums=bin_nums)
	param_lims = config.PROP_INFO[param][1]
	log_param = config.PROP_INFO[param][2]
	if log_param:
		norm = mpl.colors.LogNorm(vmin=param_lims[0], vmax=param_lims[1], clip=True)
	else:
		norm = mpl.colors.Normalize(vmin=param_lims[0], vmax=param_lims[1], clip=True)

	X, Y = np.meshgrid(ret.x_edge, ret.y_edge)
	img = axis_2D_hist.pcolormesh(X, Y, ret.statistic.T, cmap=plt.get_cmap(color_map), norm=norm)
	axis_2D_hist.autoscale('tight')

	bar_label =  config.PROP_INFO[param][0]
	plt_set.setup_colorbar(img, axis_2D_hist, bar_label)

	# Print time in corner of plot if applicable
	if time=='all':
		time_str = 'z = ' + '%.2g' % H.redshift if snap.cosmological else 't = ' + '%2.2g Gyr' % H.time
		axis_2D_hist.text(.05, .95, time_str, color=config.BASE_COLOR, fontsize = config.LARGE_FONT, ha = 'left', transform=axis_2D_hist.transAxes, zorder=4)	
	plt.tight_layout()

	plt.savefig(foutname)
	plt.close()

	return





def snap_projection(params, snap, param_lims, L=None, Lz=None, pixel_res=0.1, labels=None, color_map=config.BASE_CMAP, foutname='snap_projection.png', **kwargs):
	"""
	Plots face on and edge on projections for one snapshots for the chosen parameters

	Parameters
	----------
	param: list
		List of dust parameters (D/Z,carbon,silicate,total)
	snaps : array
	    Array of snapshots to plot
	bin_nums : int
		Number of bins to use
	time : string
		Option for printing time in corner of plot (None, one, all)
	foutname: str, optional
		Name of file to be saved
	# other available keywords:
    ## cen  - center of the image
    ## L    - side length along x and y direction
    ## Lz   - depth along z direction, for particle trimming
    ## Nx   - number of pixels along x and y direction, default 250
    ## vmin - minimum scaling of the colormap
    ## vmax - maximum scaling of the colormap
    ## theta, phi - viewing angle
	Returns
	-------
	None

	"""	

	# Set up subplots based on number of parameters given
	fig,axes = plt_set.setup_projection(len(params), L, Lz=Lz)

	for i, param in enumerate(params):
		ax1 = axes[i,0]; ax2 = axes[i,1]; cbar_ax = axes[i,2];

		param_lim = config.PROP_INFO[param][1]
		log_param = config.PROP_INFO[param][2]
		if log_param:
			norm = mpl.colors.LogNorm(vmin=np.power(10,-0.075)*param_lim[0], vmax=np.power(10,+0.075)*param_lim[1], clip=True)
		else:
			norm = mpl.colors.Normalize(vmin=np.power(10,-0.075)*param_lim[0], vmax=np.power(10,+0.075)*param_lim[1], clip=True)
		cbar_label =  config.PROP_INFO[param][0]



		# If first plot add scale bar

		# If labels given add to corner of each projection
		if labels != None:
			ax1.annotate(labels[i], (0.975,0.975), xycoords='axes fraction', color='xkcd:white', ha='right', va='top', fontsize=config.EXTRA_LARGE_FONT)

		# Plot x-y projection on top
		pixel_stats, xedges, yedges = calc_obs_projection(param, snap, [L,L,L], pixel_res=pixel_res, proj='xy')
		pixel_stats[np.logical_or(pixel_stats<=0,np.isnan(pixel_stats))] = config.EPSILON
		img = ax1.imshow(pixel_stats.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
           aspect='equal', interpolation='bicubic', cmap=plt.get_cmap(color_map), norm=norm)
		ax1.set_xlim(xedges[-1], xedges[0])
		ax1.set_ylim(yedges[-1], yedges[0])

		# Plot x-z projection below
		pixel_stats, xedges, yedges = calc_obs_projection(param, snap, [Lz,L,L], pixel_res=pixel_res, proj='xz')
		pixel_stats[np.logical_or(pixel_stats<=0,np.isnan(pixel_stats))] = config.EPSILON
		ax2.imshow(pixel_stats.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
           aspect='auto', interpolation='bicubic', cmap=plt.get_cmap(color_map), norm=norm)
		ax2.set_xlim(xedges[-1], xedges[0])
		ax2.set_ylim(yedges[-1], yedges[0])

		cbar = Colorbar(ax = cbar_ax, mappable = img, orientation = 'horizontal', ticklocation = 'bottom')
		cbar.ax.set_xlabel(cbar_label, fontsize=config.LARGE_FONT)
		cbar.ax.minorticks_off() 
		cbar.ax.tick_params(axis='both',which='both',direction='in', pad=10)
		cbar.ax.tick_params(axis='both', which='major', labelsize=config.SMALL_FONT, length=8, width=2)
		cbar.outline.set_linewidth(2)


	plt.savefig(foutname)
	plt.close()


def calc_obs_projection(param, snap, side_lens, pixel_res=2, proj='xy'):
	"""
	Calculate the average dust-to-metals ratio vs radius, gas , H2 , metal, or dust surface density
	given code values of center and viewing direction for multiple simulations/snapshots

	Parameters
	----------
	param: string
		name of parameter to project
	snap : Snapshot
	    Simulations snapshot
	side_len : list
		Size of box to consider (L1, L2, Lz), L1 ands L2 are side lengths and Lz depth is depth of box
	pixel_res : double
		Size resolution of each pixel bin in kpc
	bin_nums : int
		Number of bins for param
	proj : string
		What coordinates you want to project (xy,yz,zx)

	Returns
	-------
	pixel_data : array
		Projected data for each pixel
	coord1_bins : array
		Bins for first coordinate
	coord2_bins : array
		Bins for second coordinate
	"""	

	L1 = side_lens[0]
	L2 = side_lens[1]
	Lz = side_lens[2]

	if 'star' in param:
		S = snap.loadpart(4)
		M = S.m*config.UnitMass_in_Msolar
		x = S.p[:,0];y=S.p[:,1];z=S.p[:,2]
	else:
		G = snap.loadpart(0)
		M = G.m*config.UnitMass_in_Msolar
		x = G.p[:,0];y=G.p[:,1];z=G.p[:,2]

	# Set up coordinates to project
	if   proj=='xy': coord1 = x; coord2 = y; coord3 = z; 
	elif proj=='yz': coord1 = y; coord2 = z; coord3 = x; 
	elif proj=='xz': coord1 = x; coord2 = z; coord3 = y; 
	else:
		print("Projection must be xy, yz, or xz for calc_obs_projection()")
		return None

	# Only include particles in the box
	mask = (coord1>-L1) & (coord1<L1) & (coord2>-L2) & (coord2<L2) & (coord3>-Lz) & (coord3<Lz)

	pixel_bins = int(np.ceil(2*L1/pixel_res)) + 1
	coord1_bins = np.linspace(-L1,L1,pixel_bins)
	pixel_bins = int(np.ceil(2*L2/pixel_res)) + 1
	coord2_bins = np.linspace(-L2,L2,pixel_bins)

	pixel_area = pixel_res**2 * 1E6 # area of pixel in pc^2

	# Get the data to be projected
	if param in ['D/Z','fH2','fMC']:
		if param == 'D/Z':
			proj_data1 = G.dz[:,0]*M
			proj_data2 = G.z[:,0]*M
		#TODO: Add support for fH2 and fMC

		binned_stats = binned_statistic_2d(coord1[mask], coord2[mask],[proj_data1[mask],proj_data2[mask]], statistic=np.sum, bins=[coord1_bins,coord2_bins])
		pixel_stats = binned_stats.statistic[0]/binned_stats.statistic[1]

	else:
		if   param == 'sigma_dust': proj_data = G.dz[:,0]*M
		elif param == 'sigma_gas': 	proj_data = M
		elif param == 'sigma_H2': 	proj_data = G.fH2*G.m*(1-(G.z[:,0]+G.z[:,1]))
		elif param == 'sigma_Z': 	proj_data = G.z[:,0]*M
		elif param == 'sigma_sil': 	proj_data = G.spec[:,0]*M
		elif param == 'sigma_sil+': proj_data = (np.sum(G.spec,axis=1)-G.spec[:,1])*M
		elif param == 'sigma_carb': proj_data = G.spec[:,1]*M
		elif param == 'sigma_SiC': 	proj_data = G.spec[:,2]*M
		elif param == 'sigma_iron': proj_data = G.spec[:,3]*M
		elif param == 'sigma_O': 	proj_data = G.spec[:,4]*M
		elif param == 'sigma_star': proj_data = M
		else:
			print("%s is not a supported parameter in calc_obs_projection()."%param)
			return None

		binned_stats = binned_statistic_2d(coord1[mask], coord2[mask], proj_data[mask], statistic=np.sum, bins=[coord1_bins,coord2_bins])
		pixel_stats = binned_stats.statistic/pixel_area

	return pixel_stats, coord1_bins, coord2_bins




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
				lines += [mlines.Line2D([], [], color=config.BASE_COLOR, linestyle =linestyles[k],label=key)]

			# Want legend only on first plot
			if i == 0:
				axis.legend(handles=lines,loc=2, frameon=False)


	plt.savefig(foutname)
	plt.close()



def DZ_var_in_pixel(gas, header, center_list, r_max_list, Lz_list=None, \
			height_list=None, pixel_res=2, depletion=False, cosmological=True, labels=None, \
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


	axis.legend(loc=0, fontsize=config.SMALL_FONT, frameon=False)
	plt.savefig(foutname)
	plt.close()