import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import binned_statistic_2d
import pickle
import os

from .observations import dust_obs as obs
from .analytical_models import stellar_yields as st_yields
from .analytical_models import dust_accretion as dust_acc
from . import plot_utils as plt_set
from . import config
from . import math_utils
from . import data_calc_utils as calc

# Check if Phil's visualization module is installed
# import importlib
# vis_installed = importlib.util.find_spec("visualization") is not None
# if vis_installed:
# 	from visualization.image_maker import image_maker, edgeon_faceon_projection
#
#
# def create_visualization(snapdir, snapnum, image_key='star', fov=50, pixels=2048, **kwargs):
# 	"""
# 	Wrapper for Phil's mock Hubble visualization routine. Documentation can be found here:
# 	https://bitbucket.org/phopkins/pfh_python/src/master/ and check the docstring of
# 	=visualization.image_maker.image_maker for more details and options.
#
# 	Parameters
# 	----------
# 	snapdir: string
# 		Directory of snapshot
# 	snapnum: int
# 		Number of snapshot
# 	image_key : boolean
# 		Restrict data to only that with good signal-to-noise if applicable
# 	fov: float
# 		Sets physical size of image in kpc
# 	pixels: int
# 		Number of pixels for image, sets resolution
#
#
# 	Returns
# 	-------
# 	None
#
# 	"""
#
# 	# TODO: Incorporate Alex Gurvich's FIRE studio here
#
# 	# First check if Phil's routine is installed
# 	if not vis_installed:
# 		print("The visualization routine is not installed! Go to https://bitbucket.org/phopkins/pfh_python/src/master/ \
# 			  and follow the instructions to install the routine.")
# 		return
#
# 	edgeon_image = edgeon_faceon_projection(snapdir, snapnum, centering='', field_of_view=fov,image_key=image_key,
# 										pixels=pixels, edgeon=True, load_dust=True, **kwargs)
# 	faceon_image = edgeon_faceon_projection(snapdir, snapnum, centering='', field_of_view=fov, image_key=image_key,
# 											pixels=pixels, faceon=True, load_dust=True, **kwargs)
#
# 	return



def plot_extragalactic_dust_obs(axis, property1, property2, goodSNR=True, aggregate_data=False, show_raw_data=True):
	"""
	Plots extragalactic observational data vs the given property.

	Parameters
	----------
	axis : Matplotlib axis
		Axis on which to plot the data
	property1: string
		Dust property
	property2:
		Galaxy property
	good_SNR : boolean
		Restrict data to only that with good signal-to-noise if applicable
	aggregate_data : boolean
		Aggregate all the resolved data and make a histogram. Only applicable for resolved extragalactic D/Z data

	Returns
	-------
	None

	"""

	data_to_use = ''

	log=False
	if axis.get_yaxis().get_scale()=='log':
		log=True

	if property1 == 'D/Z':
		if property2 in ['fH2','r','r25','sigma_gas','sigma_H2','sigma_stellar','sigma_star','sigma_dust','sigma_sfr',
						 'sigma_gas_neutral','sigma_Z','Z'] or 'O/H' in property2:
			data_to_use = 'Chiang22'
			#data_to_use = 'Chiang21'
	elif property1 == 'sigma_dust':
		if property2 == 'r':
			data_to_use = 'Menard10'
	elif 'D/H' in property1:
		if property2 == 'sigma_gas_neutral':
			data_to_use = 'Clark23'
	else:
		print("%s vs %s extragalactic observational data is not available."%(property1,property2))
		return None

	if data_to_use in ['Chiang21','Chiang22']:
		if aggregate_data:
			# Plot 2D hist of all pixel data across the sample of galaxies using new data set from I-Da Chiang
			if data_to_use == 'Chiang22':
				data = obs.Chiang22_DZ_vs_param(property2, bin_data=False, log=log, goodSNR=True, only_PG16S=True, aggregate_data=aggregate_data)
			else:
				data = obs.Chiang21_DZ_vs_param(property2, bin_data=False, log=log, goodSNR=True, aggregate_data=aggregate_data)

			prop_vals = data['all'][0]; DZ_vals = data['all'][1];
			bin_nums=50

			xlog = axis.get_xaxis().get_scale()=='log'; xlims = config.get_prop_limits(property2)
			ylog = axis.get_yaxis().get_scale()=='log'; ylims = config.get_prop_limits(property1)
			x_bins = np.logspace(np.log10(xlims[0]),np.log10(xlims[1]),bin_nums) if xlog else np.linspace(xlims[0],xlims[1],bin_nums)
			y_bins = np.logspace(np.log10(ylims[0]),np.log10(ylims[1]),bin_nums) if ylog else np.linspace(ylims[0],ylims[1],bin_nums)

			lognorm = mpl.colors.LogNorm(clip=True)
			# Need minimum value to not plot null bins
			counts, xedges, yedges, img = axis.hist2d(prop_vals,DZ_vals,cmap=config.BASE_CMAP,norm=lognorm, bins=[x_bins,y_bins],cmin=config.EPSILON)

		else:
			# First plot binned values with 16/84-percentile errors
			if data_to_use == 'Chiang22':
				data = obs.Chiang22_DZ_vs_param(property2, bin_data=True, bin_nums=10, log=True, goodSNR=True, only_PG16S=True)
			else:
				data = obs.Chiang21_DZ_vs_param(property2, bin_data=True, bin_nums=10, log=True, goodSNR=True)
			for i, gal_name in enumerate(data.keys()):
				prop_vals = data[gal_name][0]; mean_DZ = data[gal_name][1]; std_DZ = data[gal_name][2]
				if log:
					std_DZ[std_DZ == 0] = config.EPSILON
				axis.errorbar(prop_vals, mean_DZ, yerr = np.abs(mean_DZ-np.transpose(std_DZ)), label=gal_name,
							  c=config.MARKER_COLORS[i], mec=config.MARKER_COLORS[i], ecolor='xkcd:dark grey',
							   mew=0.75*config.BASE_ELINEWIDTH, fmt=config.MARKER_STYLES[i], mfc='xkcd:white',
							  elinewidth=config.BASE_ELINEWIDTH, ms=config.BASE_MARKERSIZE, zorder=2)
			if show_raw_data:
				# Second plot raw pixel data in the background
				if data_to_use == 'Chiang22':
					data = obs.Chiang22_DZ_vs_param(property2, bin_data=False, log=True, goodSNR=True, only_PG16S=True)
				else:
					data = obs.Chiang21_DZ_vs_param(property2, bin_data=False, log=True, goodSNR=True)
				for i, gal_name in enumerate(data.keys()):
					prop_vals = data[gal_name][0]; mean_DZ = data[gal_name][1];
					axis.errorbar(prop_vals, mean_DZ, fmt=config.MARKER_STYLES[i], c=config.MARKER_COLORS[i],
								ms=0.3*config.BASE_MARKERSIZE, mew=0.3*config.BASE_ELINEWIDTH, mfc=config.MARKER_COLORS[i],
								mec=config.MARKER_COLORS[i], zorder=0, alpha=0.25)
	elif data_to_use == 'Menard10':
		r_vals, sigma_dust = obs.Menard_2010_dust_dens_vs_radius(1E-3, 0)
		axis.plot(r_vals,sigma_dust,label=r'Ménard+10', c='xkcd:black', linestyle=config.LINE_STYLES[0], linewidth=config.BASE_LINEWIDTH*1.5, zorder=2)
	elif data_to_use == 'Clark23':
		# The standard deviation for these median values is huge so just show the uncertainity in the median
		data = obs.Clark_2023_DtH_vs_SigmaH(error_bars='unc')
		for i, gal_name in enumerate(data.keys()):
			bin_edges = data[gal_name][0]; median_DtH = data[gal_name][1]; unc_DtH = data[gal_name][2]
			axis.errorbar(bin_edges, median_DtH, yerr=unc_DtH, label=gal_name, c=config.MARKER_COLORS[i], 
				mec=config.MARKER_COLORS[i], ecolor=config.MARKER_COLORS[i], mew=0.5*config.BASE_ELINEWIDTH, 
				fmt=config.MARKER_STYLES[i], mfc='xkcd:white', elinewidth=0.5*config.BASE_ELINEWIDTH, 
				ms=0.5*config.BASE_MARKERSIZE, zorder=2)

	return




def plot_depl_dust_obs(axis, property1, property2):
	"""
	Plots Milky Way observational data vs the given property.

	Parameters
	----------
	axis : Matplotlib axis
		Axis on which to plot the data
	property1: string
		Dust property
	property2:
		Gas property

	Returns
	-------
	None

	"""

	data_to_use = ''

	log=False
	if axis.get_yaxis().get_scale()=='log':
		log=True

	if property1 == 'D/Z':
		# Plot MW D/Z sight line data derived from depletion measurements in Jenkins09
		if property2 == 'nH' or property2 == 'nH_neutral':
			data_to_use = 'Jenkins09_DZ'
	elif 'depletion' in property1:
		elem = property1.split('_')[0]
		if elem not in config.ELEMENTS:
			print("%s is not a valid element depletion for MW observational data."%property1)
			return None
		if 'nH' in property2:
			data_to_use = 'Jenkins09_depl_nH'
		elif 'NH' in property2:
			data_to_use = ['Jenkins09_depl_NH', 'RD21_depl_NH','JW17_depl_NH']
	else:
		print("%s vs %s Milky Way observational data is not available."%(property1,property2))
		return None

	if 'Jenkins09_DZ' in data_to_use:
		# Plot J09 with Zhukovska+16/18 conversion to physical density
		dens_vals, DZ_vals = obs.Jenkins_2009_DZ_vs_dens(phys_dens=True, C_corr=True)
		axis.plot(dens_vals, DZ_vals, label=r'J09 $n_{\rm H,neutral}^{\rm Z16}$', c='xkcd:black', linestyle=config.LINE_STYLES[0], linewidth=config.BASE_LINEWIDTH*1.5, zorder=2)
		# Add data point for WNM depletion from Jenkins (2009) comparison to Savage and Sembach (1996) with error bars accounting for possible
		# range of C depeletions since they are not observed for this regime
		nH_val, WNM_depl, WNM_error = obs.Jenkins_Savage_2009_WNM_Depl('Z')
		axis.errorbar(nH_val, 1.-WNM_depl, yerr=WNM_error, label='WNM', c='xkcd:black', fmt='D',  elinewidth=config.BASE_ELINEWIDTH*1.5, ms=config.BASE_MARKERSIZE*1.5 ,zorder=2)
		axis.plot(np.logspace(np.log10(nH_val), np.log10(dens_vals[0])),np.logspace(np.log10(1.-WNM_depl), np.log10(DZ_vals[0])),
				  c='xkcd:black', linestyle=':', linewidth=config.BASE_LINEWIDTH*1.5, zorder=2)
		# Plot J09 relation to use as upper bound
		dens_vals, DZ_vals = obs.Jenkins_2009_DZ_vs_dens(phys_dens=False, C_corr=True)
		axis.plot(dens_vals, DZ_vals, label=r'J09 $\left< n_{\rm H} \right>_{\rm neutral}^{\rm min}$', c='xkcd:black', linestyle=config.LINE_STYLES[1], linewidth=config.BASE_LINEWIDTH*1.5, zorder=2)

	if 'Jenkins09_depl_nH' in data_to_use:
		elem = property1.split('_')[0]
		if elem == 'C':
			# Plot raw Jenkins data since there are so few sightlines and fit is quite bad
			C_depl, C_error, nH_vals = obs.Jenkins_2009_Elem_Depl(elem,density='<nH>')
			axis.errorbar(nH_vals,C_depl, yerr = C_error, label='Jenkins09', fmt='o', c='xkcd:black', elinewidth=config.BASE_ELINEWIDTH, ms=config.BASE_MARKERSIZE, mew=config.BASE_ELINEWIDTH,
					  mfc='xkcd:white', mec='xkcd:black', zorder=2)
			# Add in data from Parvathi which sampled twice as many sightlines
			C_depl, C_error, nH_vals = obs.Parvathi_2012_C_Depl(solar_abund='max', density='<nH>')
			axis.errorbar(nH_vals,C_depl, yerr = C_error, label='Parvathi+12', fmt='^', c='xkcd:black', elinewidth=config.BASE_ELINEWIDTH, ms=config.BASE_MARKERSIZE,
						  mew=config.BASE_ELINEWIDTH, mfc='xkcd:white', mec='xkcd:black' , zorder=2)
			# Add in shaded region for 20-40% of C in CO bars
			axis.fill_between([5E2,1E4],[0.2,0.2], [0.4,0.4], facecolor="none", hatch="x", edgecolor="xkcd:black", lw=0,
							  label='CO', zorder=2, alpha=0.99)

		else:
			dens_vals, DZ_vals = obs.Jenkins_2009_DZ_vs_dens(elem=elem, phys_dens=False, C_corr=False)
			axis.plot(dens_vals, 1.-DZ_vals, label=r'J09 $\left< n_{\rm H} \right>_{\rm neutral}^{\rm min}$', c='xkcd:black', linestyle=config.LINE_STYLES[1], linewidth=config.BASE_LINEWIDTH*1.5, zorder=2)
			dens_vals, DZ_vals = obs.Jenkins_2009_DZ_vs_dens(elem=elem, phys_dens=True, C_corr=False)
			axis.plot(dens_vals, 1.-DZ_vals, label=r'J09 $n_{\rm H,neutral}^{\rm Z16}$', c='xkcd:black', linestyle=config.LINE_STYLES[0], linewidth=config.BASE_LINEWIDTH*1.5, zorder=2)
			# Add data point for WNM depletion from Jenkins (2009) comparison to Savage and Sembach (1996)
			nH_val, WNM_depl,_ = obs.Jenkins_Savage_2009_WNM_Depl(elem)
			axis.scatter(nH_val,WNM_depl, marker='D',c='xkcd:black', zorder=2, label='WNM', s=(config.BASE_MARKERSIZE*1.5)**2)
			axis.plot(np.logspace(np.log10(nH_val), np.log10(dens_vals[0])),np.logspace(np.log10(WNM_depl), np.log10(1-DZ_vals[0])), c='xkcd:black', linestyle=':', linewidth=config.BASE_LINEWIDTH*1.5, zorder=2)

	if 'Jenkins09_depl_NH' in data_to_use:
		elem = property1.split('_')[0]

		depl, depl_err, NH_vals = obs.Jenkins_2009_Elem_Depl(elem,density='NH')
		lower_lim = np.isinf(depl_err[1,:])
		depl_err[1,lower_lim]=depl[lower_lim]*(1-10**-0.1)
		upper_lim = np.isinf(depl_err[0,:])
		depl_err[0,upper_lim]=depl[upper_lim]*(1-10**-0.1)
		no_lim = ~upper_lim & ~lower_lim
		# First plot points without limits then plot the limits individually, helps with the marker image used for the legend
		axis.errorbar(NH_vals[no_lim], depl[no_lim], yerr=depl_err[:,no_lim], label='MW', fmt='o', c='xkcd:medium grey',
					  elinewidth=0.5*config.BASE_ELINEWIDTH, ms=0.5*config.BASE_MARKERSIZE, mew=0.5*config.BASE_ELINEWIDTH,
					  mfc='xkcd:white', mec='xkcd:medium grey', zorder=-1, alpha=1)
		axis.errorbar(NH_vals[lower_lim], depl[lower_lim], yerr=depl_err[:,lower_lim], lolims=True, fmt='o', c='xkcd:medium grey',
			  elinewidth=0.5*config.BASE_ELINEWIDTH, ms=0.5*config.BASE_MARKERSIZE, mew=0.5*config.BASE_ELINEWIDTH,
			  mfc='xkcd:white', mec='xkcd:medium grey', zorder=-1, alpha=1)
		axis.errorbar(NH_vals[upper_lim], depl[upper_lim], yerr=depl_err[:,upper_lim], uplims=True, fmt='o', c='xkcd:medium grey',
					  elinewidth=0.5*config.BASE_ELINEWIDTH, ms=0.5*config.BASE_MARKERSIZE, mew=0.5*config.BASE_ELINEWIDTH,
					  mfc='xkcd:white', mec='xkcd:medium grey', zorder=-1, alpha=1)
		if elem=='C':
			C_depl, C_error, C_NH_vals = obs.Parvathi_2012_C_Depl(solar_abund='max', density='NH')
			NH_vals = np.append(NH_vals,C_NH_vals); depl = np.append(depl,C_depl);
			axis.errorbar(C_NH_vals,C_depl, yerr = C_error, label='Parvathi+12', fmt='^', c='xkcd:medium grey', elinewidth=0.5*config.BASE_ELINEWIDTH,
						  ms=0.5*config.BASE_MARKERSIZE, mew=0.5*config.BASE_ELINEWIDTH, mfc='xkcd:medium grey', mec='xkcd:medium grey' , zorder=2, alpha=1)
			# Add in shaded region for 20-40% of C in CO bars
			upper = config.get_prop_limits(property2)[1]
			axis.fill_between([np.power(10,21.75),upper],[0.2,0.2], [0.4,0.4], facecolor="none", hatch="x", edgecolor="xkcd:black",
							  lw=0, label='CO', zorder=2, alpha=0.99)

		# Now bin the data
		bin_lims = [np.min(NH_vals),np.max(NH_vals)]
		# Don't want to plot below this anyways
		if bin_lims[0]<1E18: bin_lims[0] = 1E18
		if bin_lims[1]>1E22: bin_lims[1] = 1E22
		# Set bins to be ~0.2 dex in size since range of data varies for each element
		bin_nums = int((np.log10(bin_lims[1])-np.log10(bin_lims[0]))/0.33)
		NH_vals,mean_depl_X,std_depl_X = math_utils.bin_values(NH_vals, depl, bin_lims, bin_nums=bin_nums, weight_vals=None, log=True)
		# Get rid of any bins with too few points to even get error bars for
		bad_mask = ~np.isnan(std_depl_X[:,0]) & ~np.isnan(std_depl_X[:,1]) & (std_depl_X[:,0]!=mean_depl_X) & (std_depl_X[:,1]!=mean_depl_X)
		NH_vals=NH_vals[bad_mask]; mean_depl_X=mean_depl_X[bad_mask]; std_depl_X=std_depl_X[bad_mask,:];
		axis.errorbar(NH_vals, mean_depl_X, yerr=np.abs(mean_depl_X-std_depl_X.T), label='Binned Obs.', fmt='s', c='xkcd:black',
			  elinewidth=config.BASE_ELINEWIDTH, ms=config.BASE_MARKERSIZE, mew=config.BASE_ELINEWIDTH,
			  mfc='xkcd:white', mec='xkcd:black', zorder=3, alpha=1)

	if 'RD21_depl_NH' in data_to_use:
		elem = property1.split('_')[0]
		# Stop here if element not observed
		if elem in ['O', 'Mg', 'Si', 'Fe']:
			depl, depl_err, NH_vals = obs.RomanDuval_2021_LMC_Elem_Depl(elem)
			lower_lim = np.isinf(depl_err[1, :])
			depl_err[1, lower_lim] = depl[lower_lim] * (1 - 10 ** -0.1)
			upper_lim = np.isinf(depl_err[0, :])
			depl_err[0, upper_lim] = depl[upper_lim] * (1 - 10 ** -0.1)
			no_lim = ~upper_lim & ~lower_lim
			# First plot points without limits then plot the limits individually, helps with the marker image used for the legend
			axis.errorbar(NH_vals[no_lim], depl[no_lim], yerr=depl_err[:, no_lim], label='LMC', fmt='o',
						  c='xkcd:light gold',
						  elinewidth=0.5 * config.BASE_ELINEWIDTH, ms=0.5 * config.BASE_MARKERSIZE,
						  mew=0.5 * config.BASE_ELINEWIDTH,
						  mfc='xkcd:white', mec='xkcd:light gold', zorder=-1, alpha=1)
			axis.errorbar(NH_vals[lower_lim], depl[lower_lim], yerr=depl_err[:, lower_lim], lolims=True, fmt='o',
						  c='xkcd:light gold',
						  elinewidth=0.5 * config.BASE_ELINEWIDTH, ms=0.5 * config.BASE_MARKERSIZE,
						  mew=0.5 * config.BASE_ELINEWIDTH,
						  mfc='xkcd:white', mec='xkcd:light gold', zorder=-1, alpha=1)
			axis.errorbar(NH_vals[upper_lim], depl[upper_lim], yerr=depl_err[:, upper_lim], uplims=True, fmt='o',
						  c='xkcd:light gold',
						  elinewidth=0.5 * config.BASE_ELINEWIDTH, ms=0.5 * config.BASE_MARKERSIZE,
						  mew=0.5 * config.BASE_ELINEWIDTH,
						  mfc='xkcd:white', mec='xkcd:light gold', zorder=-1, alpha=1)

			# Now bin the data
			bin_lims = [np.min(NH_vals), np.max(NH_vals)]
			# Don't want to plot below this anyways
			if bin_lims[0] < 1E18: bin_lims[0] = 1E18
			if bin_lims[1] > 1E22: bin_lims[1] = 1E22
			# Set bins to be ~0.2 dex in size since range of data varies for each element
			bin_nums = int((np.log10(bin_lims[1]) - np.log10(bin_lims[0])) / 0.33)
			NH_vals, mean_depl_X, std_depl_X = math_utils.bin_values(NH_vals, depl, bin_lims, bin_nums=bin_nums,
																weight_vals=None, log=True)
			# Get rid of any bins with too few points to even get error bars for
			bad_mask = ~np.isnan(std_depl_X[:, 0]) & ~np.isnan(std_depl_X[:, 1]) & (std_depl_X[:, 0] != mean_depl_X) & (
						std_depl_X[:, 1] != mean_depl_X)
			NH_vals = NH_vals[bad_mask];
			mean_depl_X = mean_depl_X[bad_mask];
			std_depl_X = std_depl_X[bad_mask, :];
			axis.errorbar(NH_vals, mean_depl_X, yerr=np.abs(mean_depl_X - std_depl_X.T), fmt='s',c='xkcd:dark gold',
						  elinewidth=config.BASE_ELINEWIDTH, ms=config.BASE_MARKERSIZE, mew=config.BASE_ELINEWIDTH,
						  mfc='xkcd:white', mec='xkcd:dark gold', zorder=3, alpha=1)

	if 'JW17_depl_NH' in data_to_use:
		elem = property1.split('_')[0]
		# Stop here if element not observed
		if elem in ['Mg', 'Si', 'Fe']:
			depl, depl_err, NH_vals = obs.Jenkins_Wallerstein_2017_SMC_Elem_Depl(elem)
			# First plot points without limits then plot the limits individually, helps with the marker image used for the legend
			axis.errorbar(NH_vals, depl, yerr=depl_err, label='SMC', fmt='o',
						  c='xkcd:jade',
						  elinewidth=0.5 * config.BASE_ELINEWIDTH, ms=0.5 * config.BASE_MARKERSIZE,
						  mew=0.5 * config.BASE_ELINEWIDTH,
						  mfc='xkcd:white', mec='xkcd:jade', zorder=-1, alpha=1)
			# Now bin the data
			bin_lims = [np.min(NH_vals), np.max(NH_vals)]
			# Don't want to plot below this anyways
			if bin_lims[0] < 1E18: bin_lims[0] = 1E18
			if bin_lims[1] > 1E22: bin_lims[1] = 1E22
			# Set bins to be ~0.2 dex in size since range of data varies for each element
			bin_nums = int((np.log10(bin_lims[1]) - np.log10(bin_lims[0])) / 0.25)
			NH_vals, mean_depl_X, std_depl_X = math_utils.bin_values(NH_vals, depl, bin_lims, bin_nums=bin_nums,
																weight_vals=None, log=True)
			# Get rid of any bins with too few points to even get error bars for
			bad_mask = ~np.isnan(std_depl_X[:, 0]) & ~np.isnan(std_depl_X[:, 1]) & (std_depl_X[:, 0] != mean_depl_X) & (
					std_depl_X[:, 1] != mean_depl_X)
			NH_vals = NH_vals[bad_mask];
			mean_depl_X = mean_depl_X[bad_mask];
			std_depl_X = std_depl_X[bad_mask, :];
			axis.errorbar(NH_vals, mean_depl_X, yerr=np.abs(mean_depl_X - std_depl_X.T), fmt='s', c='xkcd:dark aqua',
						  elinewidth=config.BASE_ELINEWIDTH, ms=config.BASE_MARKERSIZE, mew=config.BASE_ELINEWIDTH,
						  mfc='xkcd:white', mec='xkcd:dark aqua', zorder=3, alpha=1)

	return


def plot_galaxy_int_observational_data(axis, property):
	"""
	Plots galaxy integrated observational D/Z data vs the given property.

	Parameters
	----------
	axis : Matplotlib axis
		Axis on which to plot the data
	property: string
		Parameters to plot D/Z against (Z, O/H)

	Returns
	-------
	None

	"""

	# First check if axis scale is log or not. If so all zeros are converted to small numbers.
	log=False
	if axis.get_yaxis().get_scale()=='log':
		log=True

	if property == 'Z' or 'O/H' in property:
		if property == 'Z':
			key_name = 'metal_z_solar'
		else:
			key_name = 'metal'
		data = obs.galaxy_integrated_DZ('R14')
		Z_vals = data[key_name].values; DZ_vals = data['dtm'].values
		if log:
			DZ_vals[DZ_vals == 0] = config.EPSILON
		axis.scatter(Z_vals, DZ_vals, label='Rémy-Ruyer+14', s=(0.5*config.BASE_MARKERSIZE)**2, marker=config.MARKER_STYLES[0],
					 facecolors='none', linewidths=config.LINE_WIDTHS[3], edgecolors=config.MARKER_COLORS[0], zorder=2)
		# Also plot broken power law in Table 1
		a = 2.21; alpha1=1; b = 0.96; alpha2=3.1; xt = 8.1; xsol=8.69;
		x_vals = np.linspace(7,9.5,100)
		y_vals = np.zeros(len(x_vals))
		y_vals[x_vals<=xt] = b+alpha2*(xsol-x_vals[x_vals<=xt]); y_vals[x_vals>xt] = a+alpha1*(xsol-x_vals[x_vals>xt]);
		y_vals = 10**y_vals
		# Convert from gas-to-dust ratio to dust-to-metals ratio
		metal_vals = 10**(x_vals - 12.0) * 16.0 / 1.008 / 0.51 / 1.36
		y_vals = 1./y_vals/metal_vals
		axis.plot(x_vals, y_vals, color=config.MARKER_COLORS[0],linestyle='--')

		data = obs.galaxy_integrated_DZ('DV19')
		Z_vals = data[key_name].values; DZ_vals = data['dtm'].values
		if log:
			DZ_vals[DZ_vals == 0] = config.EPSILON
		axis.scatter(Z_vals, DZ_vals, label='De Vis+19', s=(0.5*config.BASE_MARKERSIZE)**2, marker=config.MARKER_STYLES[1], facecolors='none',
					 linewidths=config.LINE_WIDTHS[3], edgecolors=config.MARKER_COLORS[1], zorder=2)
		# Also plot power law in Table 1
		a = 2.45; b = -23.3;
		x_vals = np.linspace(7,9.5,100)
		y_vals = 10**(a*x_vals+b)
		# Convert from dust-to-gas ratio to dust-to-metals ratio
		metal_vals = 10**(x_vals - 12.0) * 16.0 / 1.008 / 0.51 / 1.36
		y_vals /= metal_vals
		axis.plot(x_vals, y_vals, color=config.MARKER_COLORS[1],linestyle='--')

		# data = obs.galaxy_integrated_DZ('PH20')
		# Z_vals = data[key_name].values; DZ_vals = data['dtm'].values
		# lim_mask = data['limit'].values==1
		# if log:
		# 	DZ_vals[DZ_vals == 0] = config.EPSILON
		# axis.scatter(Z_vals[~lim_mask], DZ_vals[~lim_mask], label='Péroux & Howk 19', s=(0.5*config.BASE_MARKERSIZE)**2,
		# 			 marker=config.MARKER_STYLES[2], facecolors='none', linewidths=config.LINE_WIDTHS[1],
		# 			 edgecolors=config.MARKER_COLORS[2], zorder=2)
		# yerr = DZ_vals[lim_mask]*(1-10**-0.1) # Set limit bars to be the same size in log space
		# axis.errorbar(Z_vals[lim_mask], DZ_vals[lim_mask], yerr=yerr, uplims=True, ms=(0.5*config.BASE_MARKERSIZE), mew=config.LINE_WIDTHS[1],
		# 			  fmt=config.MARKER_STYLES[2], mfc='none', mec=config.MARKER_COLORS[2], elinewidth=config.LINE_WIDTHS[1],
		# 			  ecolor=config.MARKER_COLORS[2], zorder=2)

	else:
		print("D/Z vs %s galaxy-integrated observational data is not available."%property)
		return None

	return



def galaxy_int_DZ_vs_prop(properties, snaps, labels=None, foutname='gal_int_DZ_vs_param.png', style='color',
						  include_obs=True, criteria='all'):
	"""
	Plots the galaxy integrate dust-to-metals ratio (D/Z) vs given parameters for multiple simulations/snapshots

	Parameters
	----------
	properties: list
		List of properties to plot D/Z against (fH2, M_gas, Z)
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
	criteria : string, optional
		Criteria for what gas/star particles to use for the given galaxy_integrated property and D/Z.
		Default is all particles.
		cold/neutral : Use only cold/neutral gas T<1000K
		hot/ionized: Use only ionized/hot gas
		molecular: Use only molecular gas

	Returns
	-------
	None

	"""

	# Also allow groupings of plots

	# Get plot stylization
	linewidths,colors,linestyles = plt_set.setup_plot_style(len(snaps), style=style)

	# Set up subplots based on number of parameters given
	fig,axes,dims = plt_set.setup_figure(len(properties))

	labels_handles = {}
	for i, x_prop in enumerate(properties):
		# Set up for each plot
		axis = axes[i]
		y_prop = 'D/Z'
		plt_set.setup_axis(axis, x_prop, y_prop, y_lim = [5E-3,5], y_log=True)

		# First plot observational data if applicable
		if include_obs: plot_galaxy_int_observational_data(axis, x_prop);

		for j,snap in enumerate(snaps):
			label = labels[j] if labels!=None else None
			DZ_val = calc.calc_gal_int_params(y_prop, snap, criteria)
			prop_val = calc.calc_gal_int_params(x_prop, snap, criteria)

			axis.scatter(prop_val, DZ_val, label=label, marker=config.BASE_MARKERSTYLE, color=colors[j], s=(1.25*config.BASE_MARKERSIZE)**2,
						 edgecolors=config.BASE_COLOR, linewidths=config.LINE_WIDTHS[2], zorder=3)

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


def plot_galaxy_int_prop_vs_prop_over_time(x_props, y_props, snaps, labels=None, x_criteria='all', y_criteria='all',
										   foutname='prop_vs_prop_over_time.png', style='color-line', include_obs=True,
										   connect_points=True, axes_args=None):
	"""
    Plots the galaxy integrated values between two properties for either a single simulation over time or multiple simulations over time.

    Parameters
    ----------
    xprops: list
        List of x-axis properties to plot
    yprops: list
        List of y-axis properties to plot
    snaps : list
        2D list of snapshots were each row is multiple snapshots at different times for a single simulation
    labels : list, optional
        List of labels for each simulation
    x_criteria : string or list
        Mask or criteria used for calculating galaxy integrated x properties (i.e molecular, cold, warm, hot, neutral, ionized)
    y_criteria : string or list
        Mask or criteria used for calculating galaxy integrated y properties (i.e molecular, cold, warm, hot, neutral, ionized)
    foutname: string, optional
        Name of file to be saved
    style : string
        Plotting style when plotting multiple data sets
        'color-line' - gives different color and linestyles to each data set
        'size' - make all lines solid black but with varying line thickness
    include_obs : boolean
        Overplot observed data if available
    connect_points: boolean
    	Add connecting lines between points for each sim
    axes_args : list
    	List of dictionaries containing arguments to be passed to axis setup functions

    Returns
    -------
    None

    """
	snaps = np.array(snaps)
	# Get plot stylization
	linewidths, colors, linestyles = plt_set.setup_plot_style(np.shape(snaps)[0], style=style)
	# Set up subplots based on number of parameters given
	sharex = 'col' if len(set(x_props)) == 1 else False
	sharey = 'row' if len(set(y_props)) == 1 else False
	fig, axes, dims = plt_set.setup_figure(len(x_props), sharex=sharex, sharey=sharey)

	labels_handles = {}
	for i in range(len(x_props)):
		x_prop = x_props[i]
		x_crit = x_criteria[i] if isinstance(x_criteria, list) else x_criteria
		y_prop = y_props[i]
		y_crit = y_criteria[i] if isinstance(y_criteria, list) else y_criteria

		# Set up for each plot
		axis = axes[i]
		axis_args = axes_args[i] if axes_args is not None else {}
		plt_set.setup_axis(axis, x_prop, y_prop, **axis_args)

		# First plot observational data if applicable
		if include_obs: plot_galaxy_int_observational_data(axis, x_prop);

		for j, snap_group in enumerate(snaps):
			if i == 0 and labels is not None:
				label = labels[j];
			else:
				label = None;
			y_vals = []
			x_vals = []
			for k, snap in enumerate(snap_group):
				y_vals += [calc.calc_gal_int_params(y_prop, snap, y_crit)]
				x_vals += [calc.calc_gal_int_params(x_prop, snap, x_crit)]
			axis.scatter(x_vals, y_vals, label=label, marker=config.MARKER_STYLES[j], color=colors[j],
						 s=(1.25 * config.BASE_MARKERSIZE) ** 2,
						 edgecolors=config.BASE_COLOR, linewidths=config.LINE_WIDTHS[2], zorder=3)
			if connect_points:
				axis.plot(x_vals, y_vals, linestyle='--', c='xkcd:grey', linewidth=config.LINE_WIDTHS[3], zorder=2)

		# Check labels and handles between this and last axis. Any differences should be added to a new legend
		hands, labs = axis.get_legend_handles_labels()
		new_lh = dict(zip(labs, hands))
		for key in labels_handles.keys(): new_lh.pop(key, 0);
		if len(new_lh) > 0:
			ncol = 2 if len(new_lh) > 4 else 1
			axis.legend(new_lh.values(), new_lh.keys(), loc='upper left', fontsize=config.SMALL_FONT, frameon=False,
						ncol=ncol)
		labels_handles = dict(zip(labs, hands))

	plt.tight_layout()
	plt.savefig(foutname, bbox_inches="tight")
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
	fig,axes,dims = plt_set.setup_figure(len(xprops), sharey=True)

	labels_handles = {}

	for i, x_prop in enumerate(xprops):
		# Set up for each plot
		axis = axes[i]
		y_prop = yprops[i]
		plt_set.setup_axis(axis, x_prop, y_prop)

		# First plot observational data if applicable
		if include_obs and y_prop=='D/Z': plot_depl_dust_obs(axis, y_prop, x_prop);

		for j,snap in enumerate(snaps):
			x_vals,y_mean,y_std = calc.calc_binned_property_vs_property(y_prop, x_prop, snap, bin_nums=bin_nums)

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



def plot_elem_depletion_vs_prop(elems, prop, snaps, bin_nums=50, labels=None, foutname='obs_elem_dep_vs_dens.png',
								std_bars=True, style='color', include_obs=True, outside_legend=False):
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
	fig,axes,dims = plt_set.setup_figure(len(elems), orientation='vertical')

	labels_handles = {}
	rollover_handles = {}
	for i,elem in enumerate(elems):
		axis = axes[i]
		plt_set.setup_axis(axis, prop, elem+'_depletion')

		for j,snap in enumerate(snaps):
			prop_vals,mean_DZ,std_DZ = calc.calc_binned_property_vs_property(elem+'_depletion', prop, snap, bin_nums=bin_nums)

			label = labels[j] if labels is not None else None
			axis.plot(prop_vals, 1.-mean_DZ, label=label, linestyle=linestyles[j], color=colors[j], linewidth=linewidths[j], zorder=3)

			if std_bars:
				axis.fill_between(prop_vals, 1.-std_DZ[:,0], 1.-std_DZ[:,1], alpha = 0.3, color=colors[j], zorder=1)

		# Only need to label the separate simulations once for external legend so do it after the first plot
		if i == 0 and outside_legend and len(snaps) > 1:
			fig.legend(bbox_to_anchor=(0.5, 1.), loc='lower center', frameon=False,
					ncol=int(np.ceil(len(snaps) / 2)), borderaxespad=0., fontsize=config.LARGE_FONT)
			# Make sure these aren't added to legends inside the plots
			hands, labs = axis.get_legend_handles_labels()
			labels_handles = dict(zip(labs, hands))

		if include_obs: plot_depl_dust_obs(axis, elem+'_depletion', prop)

		# Check labels and handles between this and last axis. Any differences should be added to a new legend
		# Also set limit to 4 legend elements per plot, rollover extra elements to next plot
		hands, labs = axis.get_legend_handles_labels()
		new_lh = dict(zip(labs, hands))
		for key in labels_handles.keys():
			if key not in rollover_handles.keys():
				new_lh.pop(key,0)
		if len(new_lh)>0:
			ncol = 2 if len(new_lh) > 3 else 1
			loc = 'lower left' if elem!='C' else 'upper right'
			# If over 4 legend elements put extra into rollover for next plot
			# Also reserve first plot legend for just simulation labels
			if i == 0 and not outside_legend:
				cap = len(snaps)
				if cap < 4:
					ncol = 1
			else:
				cap = 4
			if len(new_lh) > cap:
				rollover_handles = dict(zip(list(new_lh.keys())[cap:], list(new_lh.values())[cap:]))
				axis.legend(list(new_lh.values())[:cap], list(new_lh.keys())[:cap], loc=loc, fontsize=config.SMALL_FONT, frameon=False,
						ncol=ncol)
			else:
				rollover_handles = {}
				axis.legend(new_lh.values(), new_lh.keys(), loc=loc, fontsize=config.SMALL_FONT, frameon=False,
						ncol=ncol)

			labels_handles = dict(zip(labs, hands))

		# Add label for each element
		axis.text(.10, .4, elem, color=config.BASE_COLOR, fontsize = config.EXTRA_LARGE_FONT, ha = 'center', va = 'center', transform=axis.transAxes)

	plt.tight_layout()
	plt.savefig(foutname, bbox_inches="tight")
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

	fig,axes,dims = plt_set.setup_figure(len(snaps), sharey='row')

	for i, snap in enumerate(snaps):
		# Set up for each plot
		axis = axes[i]
		plt_set.setup_axis(axis, 'nH', 'T')
		axis.set_facecolor('xkcd:light grey')

		ret = calc.calc_phase_hist_data(prop, snap, bin_nums=bin_nums)
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



def comparison_binned_phase_plot(prop, main_snap, snaps, bin_nums=200, labels=None, color_map=config.BASE_DIVERGING_CMAP,
								 foutname='difference_phase_plot.png'):
	"""
	Plots the a 2D histogram for nH vs T of the first snap using the specified parameter as weights, and then
	plots the differences between the succeeding snaps  and the first snap.

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

	fig,axes,dims = plt_set.setup_figure(len(snaps), sharey='row', sharex='col')

	primary_ret = calc.calc_phase_hist_data(prop, main_snap, bin_nums=bin_nums)
	for i, snap in enumerate(snaps):
		# Set up for each plot
		axis = axes[i]
		plt_set.setup_axis(axis, 'nH', 'T')
		axis.set_facecolor('xkcd:light grey')

		ret = calc.calc_phase_hist_data(prop, snap, bin_nums=bin_nums)
		prop_lims = config.PROP_INFO[prop][1]
		prop_lims[0] = -prop_lims[1]
		log_param = config.PROP_INFO[prop][2]
		if log_param:
			norm = mpl.colors.SymLogNorm(1.0, vmin=prop_lims[0], vmax=prop_lims[1], clip=True,base=10)
		else:
			norm = mpl.colors.Normalize(vmin=prop_lims[0], vmax=prop_lims[1], clip=True)

		X, Y = np.meshgrid(ret.x_edge, ret.y_edge)
		img = axis.pcolormesh(X, Y, ret.statistic.T-primary_ret.statistic.T, cmap=plt.get_cmap(color_map), norm=norm)
		axis.autoscale('tight')

		# Print label in corner of plot if applicable
		if labels!=None:
			label = labels[i]
			axis.text(.95, .95, label, color=config.BASE_COLOR, fontsize=config.EXTRA_LARGE_FONT, ha='right', va='top', transform=axis.transAxes, zorder=4)

	# Add color bar to last axis
	bar_label =  "Difference " + config.PROP_INFO[prop][0]
	plt_set.setup_colorbar(img, axes[-1], bar_label)

	plt.tight_layout()
	plt.savefig(foutname)
	plt.close()

	return




def plot_sightline_depletion_vs_prop(elems, prop, sightline_data_files, bin_data=True, bin_nums=20, labels=None, foutname='sightline_depl_vs_prop.png', \
						 std_bars=True, style='color-linestyle', include_obs=True, outside_legend=False):
	"""
	Plots binned relations of specified elemental depletion vs various properties for multiple simulations

	Parameters
	----------
	elems : list
		List of which elements you want to plot depletions for
	prop: string
		Property to plot depletion against (fH2, nH)
	sightline_data_files : list
	    List of file names to pull sightline data from
	bin_data : boolean, optional
		Bin sight line data instead of scatter plotting
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
	linewidths,colors,linestyles = plt_set.setup_plot_style(len(sightline_data_files), style=style)

	# Set up subplots based on number of parameters given
	fig,axes,dims = plt_set.setup_figure(len(elems), orientation='vertical')

	labels_handles = {}
	rollover_handles = {}
	for i,elem in enumerate(elems):
		axis = axes[i]
		plt_set.setup_axis(axis, prop, elem+'_depletion')

		for j,data_file in enumerate(sightline_data_files):

			# Load in sight line data
			data = pickle.load(open(data_file, "rb" ))
			elem_indx = config.ELEMENTS.index(elem)
			depl_X = data['depl_X'][:,elem_indx]
			NH = data[prop]


			if bin_data:
				lower,upper = axis.get_xlim()
				NH_vals,mean_depl_X,std_depl_X = math_utils.bin_values(NH, depl_X, [lower, upper], bin_nums=bin_nums, weight_vals=None, log=True)
				axis.plot(NH_vals, mean_depl_X, label=labels[j], linestyle=linestyles[j], color=colors[j], linewidth=linewidths[j], zorder=3)
				if std_bars:
					axis.fill_between(NH_vals, std_depl_X[:,0], std_depl_X[:,1], alpha = 0.2, color=colors[j], zorder=1)
			else:
				axis.scatter(NH, depl_X, label=labels[j], c=colors[j], marker=config.MARKER_STYLES[j], s=2*config.BASE_MARKERSIZE, zorder=3, alpha=0.4)

		# Only need to label the separate simulations once for external legend so do it after the first plot
		if i == 0 and outside_legend and len(sightline_data_files) > 1:
			fig.legend(bbox_to_anchor=(0.5, 1.), loc='lower center', frameon=False,
					ncol=int(np.ceil(len(sightline_data_files) / 2)), borderaxespad=0., fontsize=config.LARGE_FONT)
			# Make sure these aren't added to legends inside the plots
			hands, labs = axis.get_legend_handles_labels()
			labels_handles = dict(zip(labs, hands))

		if include_obs: plot_depl_dust_obs(axis, elem + '_depletion', prop)

		# Check labels and handles between this and last axis. Any differences should be added to a new legend
		# Also set limit to 4 legend elements per plot, rollover extra elements to next plot
		hands, labs = axis.get_legend_handles_labels()
		new_lh = dict(zip(labs, hands))
		for key in labels_handles.keys():
			if key not in rollover_handles.keys():
				new_lh.pop(key,0)
		if len(new_lh)>0:
			ncol = 2 if len(new_lh) > 2 else 1
			# If over 4 legend elements put extra into rollover for next plot
			if len(new_lh) > 4:
				rollover_handles = dict(zip(list(new_lh.keys())[4:], list(new_lh.values())[4:]))
				axis.legend(list(new_lh.values())[:4], list(new_lh.keys())[:4], loc='lower left', fontsize=config.SMALL_FONT, frameon=False,
						ncol=ncol, markerscale=2.)
			else:
				rollover_handles = {}
				axis.legend(new_lh.values(), new_lh.keys(), loc='lower left', fontsize=config.SMALL_FONT, frameon=False,
						ncol=ncol, markerscale=2.)

			labels_handles = dict(zip(labs, hands))

		# Add label for each element
		axis.text(.10, .3, elem, color=config.BASE_COLOR, fontsize = config.EXTRA_LARGE_FONT, ha = 'center', va = 'center', transform=axis.transAxes)

	plt.tight_layout()
	plt.savefig(foutname, bbox_inches="tight")
	plt.close()

	return




def plot_obs_prop_vs_prop(xprops, yprops, snaps, pixel_res=[2], bin_nums=10, labels=None, foutname='obs_DZ_vs_param.png', \
						 std_bars=True, style='color-linestyle', include_obs=True, aggregate_obs=False, mask_prop=None,
						 show_raw_data=True, r_max=10, loud=False, axes_args=None, xbin_lims=None, xprop_criteria='all', yprop_criteria='all'):
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
	mask_prop : string
		Will mask pixels which do not meet a gas property criteria (for now only fH2>0.1)
	load : bool
		Print out various intermediate data products. Useful when adjusting bin numbers and data ranges.

	Returns
	-------
	None

	"""	

	# Get plot stylization
	linewidths,colors,linestyles = plt_set.setup_plot_style(len(snaps), style=style)

	# Set up subplots based on number of parameters given
	if any(yprop != yprops[0] for yprop in yprops):
		sharey = False
	else:
		sharey= True
	fig,axes,dims = plt_set.setup_figure(len(xprops), sharey=sharey)

	labels_handles = {}; rollover_handles = {};
	for i, x_prop in enumerate(xprops):
		# Set up for each plot
		axis = axes[i]
		axis_args = axes_args[i] if axes_args is not None else {}
		y_prop = yprops[i]
		xbin_lim = xbin_lims[i] if xbin_lims is not None else None
		plt_set.setup_axis(axis, x_prop, y_prop, **axis_args)

		resolution = pixel_res[i]

		# First plot observational data if applicable
		if include_obs:
			plot_extragalactic_dust_obs(axis, y_prop, x_prop, goodSNR=True, aggregate_data=aggregate_obs, show_raw_data=True);
			# Keep track of labels and handles created from observational data so we can make a separate legend
			hands, labs = axis.get_legend_handles_labels()
			new_obs_lh = dict(zip(labs, hands))
		else: new_obs_lh = {}

		for j,snap in enumerate(snaps):
			# Only need to label the separate simulations in the first plot
			label = labels[j] if (labels!=None and i==0) else None
			if  isinstance(mask_prop, list):
					mask = mask_prop[j]
			else:
					mask = mask_prop

			x_vals,y_mean,y_std,raw_data = calc.calc_binned_obs_property_vs_property(y_prop,x_prop, snap, r_max=r_max, bin_nums=bin_nums,
																			pixel_res=resolution,mask_prop=mask,prop_lims=xbin_lim,
																			prop1_criteria=xprop_criteria, prop2_criteria=yprop_criteria)

			if loud:
				print("Snap: %s x_prop: %s y_prop: %s"%(label,x_prop,y_prop))
				print("Binned values....")
				print("x_bins: ", x_vals); print("y_mean: ", y_mean); print("y_std: ", y_std)

			axis.plot(x_vals, y_mean, label=label, linestyle=linestyles[j], color=colors[j], linewidth=linewidths[j], zorder=3)
			if std_bars:
				axis.fill_between(x_vals, y_std[:,0], y_std[:,1], alpha = 0.3, color=colors[j], zorder=1)
			if show_raw_data:
				axis.scatter(raw_data[1], raw_data[0], marker='o',c=colors[j], s=(config.BASE_MARKERSIZE*0.1)**2, zorder=1)


		# Check labels and handles between this and last axis. Any differences should be added to a new legend
		# Also set limit to 4 legend elements per plot, rollover extra elements to next plot
		hands, labs = axis.get_legend_handles_labels()
		new_lh = dict(zip(labs, hands))
		# First separate data and observation legend labels so we can make separate legends for each
		if i==0 and labels!=None:
			data_lh = new_lh.copy()
			for key in new_obs_lh.keys(): data_lh.pop(key,0)
			axis.legend(data_lh.values(), data_lh.keys(), loc='upper left', fontsize=config.SMALL_FONT, frameon=False,
						ncol=len(data_lh.keys())//4+1)
			rollover_handles = new_obs_lh.copy()
		# Now add legends for observations labels and let them rollover if there are too many for one plot
		else:
			for key in labels_handles.keys():
				if key not in rollover_handles.keys():
					new_lh.pop(key,0)
			if len(new_lh)>0:
				ncol = 2 if len(new_lh) > 3 else 1
				# If over 4 legend elements put extra into rollover for next plot
				# Also reserve first plot legend for just simulation labels
				cap = 6
				if len(new_lh) > cap:
					rollover_handles = dict(zip(list(new_lh.keys())[cap:], list(new_lh.values())[cap:]))
					axis.legend(list(new_lh.values())[:cap], list(new_lh.keys())[:cap], loc='upper left',
								fontsize=config.SMALL_FONT, frameon=False,ncol=ncol)
				else:
					rollover_handles = {}
					axis.legend(new_lh.values(), new_lh.keys(), loc='upper left', fontsize=config.SMALL_FONT, frameon=False,
							ncol=ncol)

			labels_handles = dict(zip(labs, hands))

	plt.tight_layout()	
	plt.savefig(foutname)
	plt.close()	

	return




def plot_radial_proj_prop(props, snaps, rmax=20, rmin=0.1, bin_nums=50, log_bins=False, labels=None, foutname='radial_proj_prop.png', \
						   style='color-linestyle', include_obs=True):
	"""
	Plots radial projections of properties for multiple snapshots

	Parameters
	----------
	props: list
		List of properties to plot radial projections for
	snaps : list
	    List of snapshots to plot
	rmax : double
		Maximum radius of projection
	rmin : double
		Minimum radius of projection
	bin_nums : int, optional
		Number of bins to use
	log_bins : boolean, optional
		Use logarithmic radial bins
	labels : list, optional
		Array of labels for each data set
	foutname: string, optional
		Name of file to be saved
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
	fig,axes,dims = plt_set.setup_figure(len(props))

	labels_handles = {}
	for i, y_prop in enumerate(props):
		# Set up for each plot
		axis = axes[i]
		x_prop = 'r'
		plt_set.setup_axis(axis, x_prop, y_prop, x_lim=[rmin,rmax], x_log=log_bins)

		# First plot observational data if applicable
		if include_obs: plot_extragalactic_dust_obs(axis, y_prop, x_prop, goodSNR=True);

		for j,snap in enumerate(snaps):
			label = labels[j] if labels!=None else None
			x_vals,y_vals= calc.calc_radial_dens_projection(y_prop, snap, rmax, rmin = rmin, bin_nums=bin_nums,
															proj='xy',log_bins=log_bins)

			# Only need to label the separate simulations in the first plot
			axis.plot(x_vals, y_vals, label=label, linestyle=linestyles[j], color=colors[j], linewidth=linewidths[j], zorder=3)

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




def dmol_vs_props(mol_params, properties, snaps, labels=None, bin_nums=50, std_bars=True,
				   foutname='dmol_vs_props.png'):
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
	std_bars : bool
		Include standard deviation bars for the data
	foutname: str, optional
		Name of file to be saved
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
	fig,axes,dims = plt_set.setup_figure(len(properties), sharey=True)

	for i, x_prop in enumerate(properties):
		# Set up axes and label legends for each plot
		axis = axes[i]
		if len(mol_params) > 1:
			y_prop = 'mass_frac'
		else:
			y_prop = mol_params[0]

		plt_set.setup_axis(axis, x_prop, y_prop)

		for j,snap in enumerate(snaps):
			for k,mol_param in enumerate(mol_params):
				if len(mol_params) == 1:
					index = j
				else:
					index = j*len(mol_params)+k

				x_vals,y_mean,y_std = calc.calc_binned_property_vs_property(mol_param, x_prop, snap, bin_nums=bin_nums)
				axis.plot(x_vals, y_mean, color=colors[index], linestyle=linestyles[index], linewidth=linewidths[index], zorder=3)
				if std_bars:
					axis.fill_between(x_vals, y_std[:,0], y_std[:,1], alpha = 0.3, color=colors[index], zorder=1)

		# Need to manually set legend handles
		handles = []
		for j, prop in enumerate(mol_params):
			handles += [mlines.Line2D([], [], color=config.BASE_COLOR, linewidth=config.BASE_LINEWIDTH, linestyle=config.LINE_STYLES[j],label=config.PROP_INFO[prop][0])]
		if labels!=None:
			for j, label in enumerate(labels):
				handles += [mlines.Line2D([], [], color=config.LINE_COLORS[j], linewidth=config.BASE_LINEWIDTH, linestyle=config.BASE_LINESTYLE,label=label)]
		# Setup legend on first axis
		if i == 0 and len(handles)>1:
			axis.legend(handles=handles, loc=0, fontsize=config.SMALL_FONT, frameon=False)

	plt.tight_layout()
	plt.savefig(foutname)
	plt.close()

	return


def dust_data_vs_time(params, data_objs, stat='median', subsample='all', foutname='dust_data_vs_time.png', labels=None,
					  style='color-linestyle', redshift=False, outside_legend=False):
	"""
    Plots all time averaged data vs time from precompiled data for a set of simulation runs

    Parameters
    ----------
    params : list
        List of parameters to plot over time
    data_objs : list
        List of Dust_Evo objects with data to plot
    stat : string
        Statistic for given param if available (median or total)
    subsample : string or list
        Subsampling for parameter (all, cold, hot, neutral, molecular)
    foutname: str, optional
        Name of file to be saved
    outside_legend: bool
        Plot legend outside of figure. Useful for when you have a lot of plots.

    Returns
    -------
    None
    """

	# Get plot stylization
	linewidths, colors, linestyles = plt_set.setup_plot_style(len(data_objs), style=style)

	# Set up subplots based on number of parameters given
	fig, axes, dims = plt_set.setup_figure(len(params), orientation='vertical', sharex=True)

	for i, y_param in enumerate(params):
		# Set up for each plot
		axis = axes[i]

		# Check if data is from cosmological sims to determine time units
		if data_objs[0].cosmological and redshift:
			x_param = 'redshift_plus_1'
		else:
			x_param = 'time'
		plt_set.setup_axis(axis, x_param, y_param)

		# Since we are plotting with time, also want to make twinned axis for redshift to time or vice versa
		tick_labels = False
		if i < dims[1]:
			tick_labels = True
		plt_set.make_axis_secondary_time(axis, x_param, snapshot=data_objs[0], tick_labels=tick_labels)

		param_labels = []
		sub_y_params = []
		renorm = True
		if y_param == 'D/Z':
			loc = 'upper left'
		elif y_param == 'source_frac':
			param_labels = np.array(config.DUST_SOURCES)
			sub_y_params = np.array(['source_acc', 'source_SNeII', 'source_AGB','source_SNeIa'])
			loc = 'upper right'
			sub_colors = config.LINE_COLORS
		elif y_param == 'spec_frac':
			param_labels = np.array(config.DUST_SPECIES)
			sub_y_params = np.array(['spec_sil', 'spec_carb', 'spec_iron', 'spec_ORes', 'spec_SiC', 'spec_ironIncl'])
			loc = 'upper right'
			sub_colors = config.SECOND_LINE_COLORS
			# Check which of these dust species are in the simulations given
			in_sim = np.zeros(len(sub_y_params))
			for j, data in enumerate(data_objs):
				if isinstance(subsample, list):
					sample = subsample[j]
				else:
					sample = subsample
				for k, sub_param in enumerate(sub_y_params):
					data_vals = data.get_data(sub_param, subsample=sample, statistic=stat)
					if data_vals is not None and not np.all((data_vals == 0) | (np.isnan(data_vals))):
						in_sim[k] += 1
			param_labels = param_labels[in_sim > 0]
			sub_y_params = sub_y_params[in_sim > 0]
			print("These species in sims given:", param_labels)
		elif y_param == 'spec_frac_Si/C':
			param_labels = np.array(config.DUST_SPECIES_SIL_CARB)
			sub_y_params = np.array(['spec_sil+', 'spec_carb'])
			loc = 'upper right'
			sub_colors = config.SECOND_LINE_COLORS
		elif '/H_all' in y_param and y_param.split('/')[0] in config.ELEMENTS:
			param_labels = np.array(['total', 'dust'])
			elem = y_param.split('/')[0]
			sub_y_params = np.array([elem + '/H', elem + '/H_dust'])
			loc = 'lower right'
			renorm = False
			sub_colors = config.SECOND_LINE_COLORS
		elif y_param == 'Si/C':
			loc = 'upper right'
		else:
			loc = 'upper left'

		for j, data in enumerate(data_objs):
			if isinstance(subsample, list):
				sample = subsample[j]
			else:
				sample = subsample
			label = labels[j] if labels != None else None
			# Chose to plot vs cosmic time or redshift
			if x_param == 'time' and data_objs[0].cosmological and not redshift:
				time_data = math_utils.quick_lookback_time(data.get_data(x_param))
			else:
				time_data = data.get_data(x_param)
				if data_objs[0].cosmological and redshift:
					time_data += 1.

			# Check if parameter has subparameters
			if len(param_labels) == 0:
				data_vals = data.get_data(y_param, subsample=sample, statistic=stat)
				axis.plot(time_data, data_vals, color=colors[j], linestyle=linestyles[j], label=label,
						  linewidth=config.BASE_LINEWIDTH, zorder=-1)
			else:
				data_vals = []
				for sub_param in sub_y_params:
					data_vals += [data.get_data(sub_param, subsample=sample, statistic=stat)]

				data_vals = np.array(data_vals)
				# Renormalize just in case since some of the older snapshots aren't normalized
				if renorm:
					data_vals = data_vals / np.sum(data_vals, axis=0)[np.newaxis, :]
					data_vals[np.isnan(data_vals)] = 0.
				for k in range(np.shape(data_vals)[0]):
					# Ignore properties that are zero for the entire simulation and don't plot them
					if not np.all(data_vals[k, :] == 0):
						axis.plot(time_data, data_vals[k, :], color=sub_colors[k], linestyle=linestyles[j],
								  linewidth=config.BASE_LINEWIDTH, zorder=-1)
		# Only need to label the separate simulations in the first plot
		if i == 0 and len(data_objs) > 1:
			if outside_legend:
				fig.legend(bbox_to_anchor=(0.5, 1.), loc='lower center', frameon=False,
						   ncol=int(np.ceil(len(data_objs) / 2)), borderaxespad=0., fontsize=config.LARGE_FONT)
			else:
				axis.legend(loc='best', frameon=False, fontsize=config.SMALL_FONT)

		# If there are subparameters need to make their own legend
		if len(param_labels) > 0:
			# param_labels = param_labels[:np.shape(data_vals)[1]]
			param_lines = []
			for j, label in enumerate(param_labels):
				param_lines += [
					mlines.Line2D([], [], color=sub_colors[j], label=label, linewidth=config.BASE_LINEWIDTH, )]
			axis.legend(handles=param_lines, loc=loc, frameon=False, fontsize=config.SMALL_FONT)

	plt.tight_layout()
	# Need to specify the bbox_inches for external legends or else they get cutoff
	plt.savefig(foutname, bbox_inches="tight")
	plt.close()


def dust_data_vs_time_seperate_sims(params, data_objs, stat = 'median', subsample = ['all'],
									foutname = 'sim_dust_data_vs_time.pdf', sim_labels = None, sample_labels = None,
									style = 'color-linestyle',redshift = False, sample_linewidths = None):
	"""
	Plots given param data vs time from precompiled data for a set of simulation runs, plotting each simulation on a
	separate column
	
	Parameters
	----------
	params : list
		List of parameters to plot over time
	data_objs : list
		List of Dust_Evo objects with data to plot
	stat : string
		Statistic for given param if available (median or total)
	subsample : list
		Subsampling for parameter (all, cold, hot, neutral, molecular)
	foutname: str, optional
		Name of file to be saved
	sim_labels: list
		Names for given data_obs/sims which will appear on the top row of plots
	sample_labels: list
		Names for each given subsample to appear in legend
	style: string
		How each subsample will be differentiated (color, linestyle, size, or combination)
	redshift: bool
		Plot time as redshift instead of cosmic time
	sample_linewidths: list
		Linewidths for each subsample
	
	Returns
	-------
	None
	"""

	# Get plot stylization
	linewidths, colors, linestyles = plt_set.setup_plot_style(len(subsample), style=style)
	if sample_linewidths is not None:
		linewidths = sample_linewidths
	num_sims = len(data_objs)
	num_params = len(params)
	# Set up subplots based on number of parameters given
	fig, axes = plt.subplots(num_params, num_sims, figsize=(
	num_sims * config.BASE_FIG_XSIZE * config.FIG_XRATIO, num_params * config.BASE_FIG_YSIZE * config.FIG_YRATIO),
							 squeeze=True, sharex=True, sharey='row')
	axes = list(filter(None, axes.flat))
	dims = [num_sims, num_params]

	for i, data in enumerate(data_objs):
		for j, y_param in enumerate(params):
			# Set up for each plot
			axis = axes[i + j * num_sims]

			# Check if data is from cosmological sims to determine time units
			if data_objs[0].cosmological and redshift:
				x_param = 'redshift_plus_1'
			else:
				x_param = 'time'
			plt_set.setup_axis(axis, x_param, y_param)

			# Since we are plotting with time, also want to make twinned axis for redshift to time or vice versa
			tick_labels = False
			if j == 0:
				tick_labels = True
			plt_set.make_axis_secondary_time(axis, x_param, snapshot=data_objs[0], tick_labels=tick_labels)

			param_labels = []
			sub_y_params = []
			if y_param == 'D/Z':
				loc = 'upper left'
			elif y_param == 'source_frac':
				param_labels = np.array(config.DUST_SOURCES)
				sub_y_params = np.array(['source_acc', 'source_SNeII', 'source_AGB', 'source_SNeIa'])
				loc = 'upper right'
				sub_colors = config.LINE_COLORS
			elif y_param == 'spec_frac':
				param_labels = np.array(config.DUST_SPECIES)
				sub_y_params = np.array(['spec_sil', 'spec_carb', 'spec_iron', 'spec_ORes', 'spec_SiC', 'spec_ironIncl'])
				loc = 'upper right'
				sub_colors = config.SECOND_LINE_COLORS
				# Check which of these dust species are in the simulations given
				in_sim = np.zeros(len(sub_y_params))
				if isinstance(subsample, list):
					sample = subsample[0]
				else:
					sample = subsample
				for k, sub_param in enumerate(sub_y_params):
					data_vals = data.get_data(sub_param, subsample=sample, statistic=stat)
					if data_vals is not None and not np.all((data_vals == 0) | (np.isnan(data_vals))):
						in_sim[k] += 1
				param_labels = param_labels[in_sim > 0]
				sub_y_params = sub_y_params[in_sim > 0]
				print("These species in sims given:", param_labels)
			elif y_param == 'spec_frac_Si/C':
				param_labels = np.array(config.DUST_SPECIES_SIL_CARB)
				sub_y_params = np.array(['spec_sil+', 'spec_carb'])
				loc = 'upper right'
				sub_colors = config.SECOND_LINE_COLORS
			elif '/H_all' in y_param and y_param.split('/')[0] in config.ELEMENTS:
				param_labels = np.array(['total', 'dust'])
				elem = y_param.split('/')[0]
				sub_y_params = np.array([elem + '/H', elem + '/H_dust'])
				loc = 'lower right'
				sub_colors = config.SECOND_LINE_COLORS
			elif y_param == 'Si/C':
				loc = 'upper right'
			else:
				loc = 'upper left'

			for k, sample in enumerate(subsample):
				sample = subsample[k]
				label = sample_labels[k] if sample_labels != None else None
				# Choose to plot vs cosmic time or redshift
				if x_param == 'time' and data_objs[0].cosmological and not redshift:
					time_data = math_utils.quick_lookback_time(data.get_data(x_param))
				else:
					time_data = data.get_data(x_param)
					if data_objs[0].cosmological and redshift:
						time_data += 1.

				# Check if parameter has subparameters
				if len(param_labels) == 0:
					data_vals = data.get_data(y_param, subsample=sample, statistic=stat)
					axis.plot(time_data, data_vals, color=config.BASE_COLOR, linestyle=linestyles[k], label=label,
							  linewidth=linewidths[k], zorder=-1)
				else:
					data_vals = []
					for sub_param in sub_y_params:
						data_vals += [data.get_data(sub_param, subsample=sample, statistic=stat)]

					data_vals = np.array(data_vals)
					# Renormalize just in case since some of the older snapshots aren't normalized
					# if renorm:
					# 	data_vals = data_vals / np.sum(data_vals, axis=0)[np.newaxis, :]
					# 	data_vals[np.isnan(data_vals)] = 0.
					for m in range(np.shape(data_vals)[0]):
						# Ignore properties that are zero for the entire simulation and don't plot them
						if not np.all(data_vals[m, :] == 0):
							data_vals[data_vals==0] = np.nan
							axis.plot(time_data, data_vals[m, :], color=sub_colors[m], linestyle=linestyles[k],
									  linewidth=linewidths[k], zorder=-1)

				# Add label for each sim on the first row of plots
				if j == 0:
					axis.text(.05, .9, sim_labels[i], color='xkcd:medium grey', fontsize=config.LARGE_FONT, ha='left',
							  va='top', transform=axis.transAxes, zorder=1)

				# Only need to label the separate simulations in the first plot
				if i == 0 and j == 0 and len(data_objs) > 1:
					axis.legend(loc='best', frameon=False, fontsize=config.SMALL_FONT)

				# If there are subparameters need to make their own legend but only for the first sim
				if len(param_labels) > 0 and i == 0:
					# param_labels = param_labels[:np.shape(data_vals)[1]]
					param_lines = []
					for k, label in enumerate(param_labels):
						param_lines += [
							mlines.Line2D([], [], color=sub_colors[k], label=label, linewidth=config.BASE_LINEWIDTH, )]
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
	fig,axes,dims = plt_set.setup_figure(len(dust_species), sharey=True)
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


	plt.tight_layout()
	plt.savefig(foutname, transparent=False, bbox_inches='tight')
	plt.close()



def compare_FIRE_winds(Z_list, style='color-linestyle', foutname='wind_routine_compare.png'):


	N = 1000 # number of steps
	max_t = 10. # max age of stellar population to compute yields
	min_t = 1E-4


	FIRE_ver = 2
	names = ['base', 'SB99fix', 'AGBshift']
	labels = ['Base FIRE-2', 'SB99 Fix', 'AGB Shift']


	# Compare routine carbon yields between routines
	# Set up subplots based on number of parameters given
	fig,axes,dims = plt_set.setup_figure(4, orientation='vertical', sharex=True)
	x_param = 'star_age'; y_params = ['wind_rate','wind_vel','wind_E','cum_wind_E']

	# Get plot stylization
	linewidths,colors,linestyles = plt_set.setup_plot_style(len(y_params), style=style)

	# Make lines for legend
	lines = []
	for i in range(len(Z_list)):
		lines += [mlines.Line2D([], [], color=config.BASE_COLOR, label=r'Z = %.2g $Z_{\odot}$' % Z_list[i], linewidth= linewidths[i])]
	for i in range(len(names)):
		lines += [mlines.Line2D([], [], color=colors[i], linestyle=linestyles[i], label=labels[i], linewidth=config.BASE_LINEWIDTH)]

	for i, name in enumerate(names):
		for j, Z in enumerate(Z_list):
			time, wind_rate, wind_vel, wind_E, cum_windE = st_yields.stellar_winds(min_t, max_t,N,Z,FIRE_ver=FIRE_ver,AGB_change=name)
			data = [wind_rate, wind_vel, wind_E, cum_windE]
			for k, y_param in enumerate(y_params):
				axis = axes[k]
				if i == 0:
					plt_set.setup_axis(axis, x_param, y_param)
					if k==0:
						legend = axis.legend(handles=lines, frameon=True, ncol=2, loc='center left', bbox_to_anchor=(0.025,1.0), framealpha=1, fontsize=config.SMALL_FONT*0.9, edgecolor=config.BASE_COLOR)
						legend.get_frame().set_lw(config.AXIS_BORDER_WIDTH)

				axis.loglog(time, data[k], color = colors[i], linestyle = linestyles[i], nonpositive = 'clip', linewidth = linewidths[j])


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
	fig,axes,dims = plt_set.setup_figure(1)
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




def snap_projection(props, snap, L=None, Lz=None, pixel_res=0.1, labels=None, color_map=config.BASE_CMAP,
					foutname='snap_projection.png', param_lims=None, **kwargs):
	"""
	Plots face on and edge on projections for one snapshots for the chosen parameters

	Parameters
	----------
	props: list
		List of properties
	snap : Snapshot
	    Snapshot to pull data from
	L : float
		Length size in kpc for x and y directions
	Lz : float
		Length size in kpc for z direction
	pixel_res : float
	    Resolution of pixels in kpc
	labels : list
		List of labels for each projection
	color_map : string or list
		Single colormap for all plots or list of colormaps for each
	foutname: string
		Name of file to be saved
	# other available keywords:
    ## cen  - center of the image
    ## L    - side length along x and y direction
    ## Lz   - depth along z direction, for particle trimming
    ## Nx   - number of pixels along x and y direction, default 250
    ## theta, phi - viewing angle
	Returns
	-------
	None

	"""	
	# TODO : Add ability to only put colorbar on right side if multiple plots if they all share the same projected data but are from different snapshots
	# TODO : Add ability to plot multiple snapshots

	# Set up subplots based on number of parameters given
	fig,axes = plt_set.setup_projection(len(props), L, Lz=Lz)

	for i, param in enumerate(props):
		ax1 = axes[0,i]; ax2 = axes[1,i]
		if isinstance(color_map, str):
			cmap = color_map
		else:
			cmap = color_map[i]


		param_lim = param_lims[i] if param_lims is not None else config.PROP_INFO[param][1]
		log_param = config.PROP_INFO[param][2]
		if log_param:
			norm = mpl.colors.LogNorm(vmin=np.power(10,-0.075)*param_lim[0], vmax=np.power(10,+0.075)*param_lim[1], clip=True)
		else:
			norm = mpl.colors.Normalize(vmin=param_lim[0], vmax=param_lim[1], clip=True)
		cbar_label =  config.PROP_INFO[param][0]

		# If labels given add to corner of each projection
		if labels != None:
			ax1.annotate(labels[i], (0.975,0.975), xycoords='axes fraction', color='xkcd:white', ha='right', va='top', fontsize=config.EXTRA_LARGE_FONT)

		# Plot x-y projection on top
		pixel_stats, xedges, yedges = calc.calc_projected_prop(param, snap, [L,L,L], pixel_res=pixel_res, proj='xy')
		pixel_stats[np.logical_or(pixel_stats<=0,np.isnan(pixel_stats))] = config.EPSILON
		img = ax1.imshow(pixel_stats.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
           aspect='equal', interpolation='bicubic', cmap=plt.get_cmap(cmap), norm=norm)

		# Plot x-z projection below
		pixel_stats, xedges, yedges = calc.calc_projected_prop(param, snap, [L,Lz,L], pixel_res=pixel_res, proj='xz')
		pixel_stats[np.logical_or(pixel_stats<=0,np.isnan(pixel_stats))] = config.EPSILON
		ax2.imshow(pixel_stats.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
           aspect='equal', interpolation='bicubic', cmap=plt.get_cmap(cmap), norm=norm)

		# Add color bar to the bottom of each set of x-y and x-z projections
		cbar = plt.colorbar(mappable = img, ax=[ax1,ax2], fraction=.05, pad=0.01, location='bottom',use_gridspec=True)
		cbar.ax.set_xlabel(cbar_label, fontsize=config.LARGE_FONT)
		cbar.ax.minorticks_on()
		cbar.ax.tick_params(axis='both',which='both',direction='in')
		cbar.ax.tick_params(axis='both', which='major', labelsize=config.SMALL_FONT, length=4*config.AXIS_BORDER_WIDTH, width=config.AXIS_BORDER_WIDTH)
		cbar.ax.tick_params(axis='both', which='minor', labelsize=config.SMALL_FONT, length=2*config.AXIS_BORDER_WIDTH, width=config.AXIS_BORDER_WIDTH/2)
		cbar.outline.set_linewidth(config.AXIS_BORDER_WIDTH)

		for key, value in kwargs.items():
			if key == 'show_rvir' and value == True:
				# Plot cirlce for r_vir
				circle1 = plt.Circle((0,0),snap.rvir,linestyle='dashed',edgecolor='xkcd:red',facecolor='none',zorder=10,
									 linewidth=config.LINE_WIDTHS[1])
				circle2 = plt.Circle((0,0),snap.rvir,linestyle='dashed',edgecolor='xkcd:red',facecolor='none',zorder=10,
									 linewidth=config.LINE_WIDTHS[1])
				ax1.add_patch(circle1)
				ax2.add_patch(circle2)

	plt.savefig(foutname)
	plt.close()




def dust_acc_diag(params, snaps, bin_nums=100, labels=None, foutname='dust_acc_diag.png', style='color', implementation='species'):
	"""
	Make plot of instantaneous dust growth for a given snapshot depending on the dust evolution implementation used

	Parameters
	----------
	params : list
		List of parameters to plot diagnostics for (inst_dust_prod, growth_timescale, )
	snaps : list
	    List of snapshots to use
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
	linewidths,colors,linestyles = plt_set.setup_plot_style(len(snaps), properties=config.DUST_SPECIES, style=style)

	# Set up subplots based on number of parameters given
	fig,axes,dims = plt_set.setup_figure(len(params))

	for i,param in enumerate(params):
		axis = axes[i]
		if param == 'inst_dust_prod':
			plt_set.setup_axis(axis, 'nH', param, x_lim=[1.1E0,0.9E3])
		if param == 'g_timescale':
			plt_set.setup_axis(axis, param, 'g_timescale_frac')


		for j,snap in enumerate(snaps):
			if isinstance(implementation, list):
				imp = implementation[j]
			else:
				imp = implementation

			G =	snap.loadpart(0)
			nH = G.rho*config.UnitDensity_in_cgs * ( 1. - (G.z[:,0]+G.z[:,1])) / config.H_MASS

			if param == 'inst_dust_prod':
				weight_vals = dust_acc.calc_dust_acc(G,implementation=imp, nano_iron=True, O_res=True)
				x_vals = dict.fromkeys(weight_vals.keys(), nH)
				density = False
			elif param == 'g_timescale':
				if imp == 'species':
					x_vals = dust_acc.calc_spec_acc_timescale(G, nano_iron=True)
				else:
					x_vals = dust_acc.calc_elem_acc_timescale(G)
				for key in x_vals.keys():
					x_vals[key] = x_vals[key][x_vals[key]>0]
					x_vals[key]*=1E-9
				weight_vals=dict.fromkeys(x_vals.keys(), np.full(len(nH),1./len(nH)))
				density = True
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

			for k,spec in enumerate(config.DUST_SPECIES):
				if spec in x_vals.keys():
					print(spec,param)
					print(np.sum(weight_vals[spec]))
					axis.hist(x_vals[spec], bins=bins, weights=weight_vals[spec], histtype='step', cumulative=True, density=density, label=labels[j], color=colors[k], \
							 linewidth=linewidths[j], linestyle=linestyles[j])
					lines += [mlines.Line2D([], [], color=config.BASE_COLOR, linestyle =linestyles[j],label=spec)]

		# Need to manually set legend handles
		handles = []
		for j, label in enumerate(labels):
			handles += [mlines.Line2D([], [], color=config.BASE_COLOR, linewidth=config.BASE_LINEWIDTH, linestyle=linestyles[j],label=label)]
		for j, spec in enumerate(config.DUST_SPECIES):
			if spec != 'Iron Inclusions' and spec != 'SiC':
				handles += [mlines.Line2D([], [], color=colors[j], linewidth=config.BASE_LINEWIDTH, linestyle=config.BASE_LINESTYLE,label=spec)]

		# Setup legend on first axis
		if i == 0 and len(handles)>1:
			axis.legend(handles=handles, loc='best', fontsize=config.SMALL_FONT, frameon=False, ncol=2)


	plt.tight_layout()
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
				mean_DZ[y*len(x_vals)+x],std_DZ[y*len(x_vals)+x,0],std_DZ[y*len(x_vals)+x,1] = math_utils.weighted_percentile(values, weights=weights)

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



def plot_binned_grain_distribution(snaps, y_props='dn/da', mask_criterias=None, labels=None, foutname='binned_grain_size_dist.png', 
						 std_bars=True, style='color-linestyle'):
	"""
	Plots grain size distribution binned across gas particles.

	Parameters
	----------
	snaps : list
	    List of snapshots to plot
	y_prop: string
		Choose whether to plot grain size probability or grain mass propability and which dust species
	mask_criterias: list
		Criterias for what particles to include (i.e. hot/warm/cold gas) for each snap
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
	Returns
	-------
	None

	"""

	# Get plot stylization
	linewidths,colors,linestyles = plt_set.setup_plot_style(len(snaps), style=style)

	# Set up subplots based on number of parameters given
	fig,axes,dims = plt_set.setup_figure(1, sharey=True)
	axis = axes[0]

	x_prop='grain_size'
	plt_set.setup_axis(axis, x_prop, y_props[0])


	for j,snap in enumerate(snaps):
			y_prop = y_props[j]
			if mask_criterias is None: mask_criteria = 'all'
			else: 					   mask_criteria=mask_criterias[j]
			x_vals,y_mean,y_std = calc.calc_binned_grain_distribution(y_prop, mask_criteria, snap)

			if labels is not None: label = labels[j];
			else:    			   label = None;
			axis.plot(x_vals, y_mean, label=label, linestyle=linestyles[j], color=colors[j], linewidth=linewidths[j], zorder=3)
			if std_bars:
				axis.fill_between(x_vals, y_std[:,0], y_std[:,1], alpha = 0.3, color=colors[j], zorder=1)

		
	axis.legend(loc='best', fontsize=config.SMALL_FONT, frameon=False)

	plt.tight_layout()
	plt.savefig(foutname)
	plt.close()

	return



def plot_grain_size_bins(snap, y_prop='dn/da', particle_ids=None, foutname='grain_size_dist.png', 
						 style='color-linestyle'):
	"""
	Plots the full grain size distribution for a few select particles. Useful for testing.

	Parameters
	----------
	snaps : list
	    List of snapshots to plot
	y_prop: string
		Choose whether to plot grain size probability or grain mass propability and which dust species
	particle_ids: list
		List of particle ids to plot. If none are given then chose a particle from each of the ISM phases.
	labels : list, optional
		List of labels for each snapshot
	foutname: string, optional
		Name of file to be saved
	style : string
		Plotting style when plotting multiple data sets
		'color-line' - gives different color and linestyles to each data set
		'size' - make all lines solid black but with varying line thickness
	Returns
	-------
	None

	"""

	G = snap.loadpart(0)
	IDs = G.id

	if particle_ids is None:
		# Get plot stylization
		linewidths,colors,linestyles = plt_set.setup_plot_style(3, style=style)
		T = G.get_property('T')
		ISM_phases = [0,1E3,1E4,1E7]
		particle_ids = np.zeros(len(ISM_phases)-1, dtype='int')
		for i in range(len(ISM_phases)-1):

			mask = (T>ISM_phases[i]) & (T<ISM_phases[i+1])
			if len(IDs[mask]) > 0:
				particle_ids[i] = IDs[mask][0]
		print("Particle IDs chosen (0 means no particles in that ISM phase):",particle_ids)
	else:
		linewidths,colors,linestyles = plt_set.setup_plot_style(len(particle_ids), style=style)

	# Set up subplots based on number of parameters given
	fig,axes,dims = plt_set.setup_figure(1, sharey=True)
	axis = axes[0]

	x_prop='grain_size'
	plt_set.setup_axis(axis, x_prop, y_prop)

	bin_nums = G.get_property('grain_bin_nums')
	bin_slopes = G.get_property('grain_bin_slopes')
	bin_edges = snap.Grain_Bin_Edges
	bin_centers = snap.Grain_Bin_Centers

	# default to silicates for now
	spec_num=0
	for j,id in enumerate(particle_ids):
		index = np.where(IDs == id)[0][0]
		for k in range(snap.Flag_GrainSizeBins):
			bin_num = bin_nums[index][spec_num][k]; 
			bin_slope = bin_slopes[index][spec_num][k]; 
			# Since the grain bin number and slopes can be extremely large, rounding errors can cause negative values near the bin edges
			# that seem large but are small compared to the number of grains. To avoid this when plotting we dont include the 1% around the bin edges.
			grain_sizes = np.logspace(np.log10(bin_edges[k]*1.01),np.log10(bin_edges[k+1]*0.99))
			if y_prop=='dn/da':
				dist_vals = (bin_num/(bin_edges[k+1]-bin_edges[k])+bin_slope*(grain_sizes-bin_centers[k]))
			else: 
				dist_vals = np.power(grain_sizes,4)*(bin_num/(bin_edges[k+1]-bin_edges[k])+bin_slope*(grain_sizes-bin_centers[k]))

			if k==0: label = 'id ' + str(particle_ids[j])
			else: label = None

			axis.plot(grain_sizes,dist_vals, label=label, linestyle=linestyles[j], color=colors[j], linewidth=linewidths[j], zorder=3)

		
	axis.legend(loc='best', fontsize=config.SMALL_FONT, frameon=False)

	plt.tight_layout()
	plt.savefig(foutname)
	plt.close()

	return