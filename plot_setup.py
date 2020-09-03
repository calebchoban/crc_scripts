import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
plt.switch_backend('agg')
from scipy.stats import binned_statistic_2d
from scipy.optimize import curve_fit
import pickle
import os
from readsnap import readsnap
from astropy.table import Table
import gas_temperature as gas_temp
from tasz import *
from observations import *
from analytic_dust_yields import *

# Set style of plots
plt.style.use('seaborn-talk')
# Set personal color cycle
Line_Colors = ["xkcd:blue", "xkcd:red", "xkcd:green", "xkcd:orange", "xkcd:violet", "xkcd:teal", "xkcd:brown"]
Line_Colors = ["xkcd:azure", "xkcd:tomato", "xkcd:green", "xkcd:orchid", "xkcd:teal", "xkcd:sienna"]
Marker_Colors = ["xkcd:orange", "xkcd:teal", "xkcd:sienna", "xkcd:gold", "xkcd:magenta"]
Line_Styles = ['-','--',':','-.']
Marker_Style = ['o','^','X','s','v']
Line_Widths = [0.5,1.0,1.5,2.0,2.5,3.0]

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=Line_Colors)

# Large and small font sizes to be used for axes labels and legends
Large_Font=26
Small_Font=16	

# Houses labels, limits, and if they should be plotted in log space for possible parameters
PARAM_INFO  				= {'fH2': [r'$f_{H2}$', 									[0,1.], 		False],
								 'r': ['Radius (kpc)', 									[0,20,], 		False],
							   'r25': [r'Radius (R$_{25}$)', 							[0,2], 			False],
					     'sigma_gas': [r'$\Sigma_{gas}$ (M$_{\odot}$ pc$^{-2}$)', 		[1E0,1E2], 		True],
						   'sigma_Z': [r'$\Sigma_{metals}$ (M$_{\odot}$ pc$^{-2}$)', 	[1E-3,1E0], 	True],
						'sigma_dust': [r'$\Sigma_{dust}$ (M$_{\odot}$ pc$^{-2}$)', 		[1E-3,1E0], 	True],
						  'sigma_H2': [r'$\Sigma_{H2}$ (M$_{\odot}$ pc$^{-2}$)', 		[1E-3,1E0], 	True],
						  	  'time': ['Time (Gyr)',									[1E-2,1E1],		True],
						  'redshift': ['z',												[1E-1,100],		True],
						        'nH': [r'$n_{H}$ (cm$^{-3}$)', 							[1E-2,1E3], 	True],
						         'T': [r'T (K)', 										[1E1,1E5], 		True],
						         'Z': [r'Z (Z$_{\odot}$)', 								[1E-3,5E0], 	True],
						        'DZ': ['D/Z Ratio', 									[0,1], 			False],
						 'depletion': [r'[X/H]$_{gas}$', 								[1E-3,1E0], 	True],
				     'cum_dust_prod': [r'Cumulative Dust Ratio $(M_{dust}/M_{\star})$', [1E-6,1E-2], 	True],
					'inst_dust_prod': [r'Cumulative Inst. Dust Prod. $(M_{\odot}/yr)$', [0,10], 		False],
					   'source_frac': ['Source Mass Fraction', 							[1E-2,1E0], 	True],
					     'spec_frac': ['Species Mass Fraction', 						[0,1], 			False],
					          'Si/C': ['Sil-to-C Ratio', 								[0,1], 			False]
					     }



def setup_plot_style(num_sets, style='color'):
	"""
	Sets up the line widths, colors, and line styles that will be used for plotting
	the given number of data sets and desired style.

	Parameters
	----------
	num_sets : int
		Number of data sets that will be plotted
	style : string
		The style which will be used to differentiate the data sets. 'color' gives 
		each a different color, 'size' gives each a different line width.

	Returns
	-------
	linewidths : array
		List of linewidths for each data set.
	colors : array
		List of colors for each data set.		
	linestyles : array
		List of linestyles for each data set.

	"""	


	if num_sets == 1:
		linewidths = np.full(num_sets,2)
		colors = Line_Colors
		linestyles = Line_Styles
	elif style == 'color':
		linewidths = np.full(num_sets,2)
		colors = Line_Colors
		linestyles = Line_Styles
	elif style == 'size':
		linewidths = Line_Widths
		colors = ['xkcd:black' for i in range(num_sets)]
		linestyles = ['-' for i in range(num_sets)]
	else:
		print("Need to give a style when plotting more than one set of data. Currently 'color' and 'size' differentiation are supported.")
		return None,None,None

	return linewidths, colors, linestyles



def setup_figure(num_plots):
	"""
	Sets up the figure size and subplot layout based on number of plots

	Parameters
	----------
	num_plots : int
		Number of plots to be plotted

	Returns
	-------
	fig : Pyplot figure
		Pyplot figure which houses plots
	axes: list
		List of axes for each plot
	"""	
	if num_plots == 1:
		fig,axes = plt.subplots(1, 1, figsize=(14/1.2,10/1.2))
		axes = np.array([axes])
	elif num_plots%2 == 0:
		fig,axes = plt.subplots(num_plots/2, 2, figsize=(28/1.2,num_plots/2*10/1.2), squeeze=True)
	elif num_plots%3 == 0:
		fig,axes = plt.subplots(num_plots/3, 3, figsize=(3*14/1.2,num_plots/3*10/1.2), squeeze=True)
	else:
		fig,axes = plt.subplots(int(np.ceil(num_plots/3.)), 3, figsize=(3*14/1.2,np.ceil(num_plots/3.)*10/1.2), squeeze=True)\
	
	axes=axes.flat
	# Delete any excess axes
	if len(axes) > num_plots:
		for i in range(len(axes)-num_plots):
			fig.delaxes(axes[-(i+1)])
	return fig,axes


def setup_axis(axis, x_param, y_param, x_lim=None, x_log=None, y_lim=None, y_log=None):
	"""
	Sets up the axes for plot given x and y param and optional limits

	Parameters
	----------
	axis : Matplotlib axis
	    Axis of plot
	x_param : string
		Parameter to be plotted on x axis
	y_param : string
		Parameter to be plotted on y axis
	x_lim : array
		Limits for x axis
	x_log : boolean
		Explicitly set x axis to linear or log space, otherwise go with default for x_param
	y_lim : array
		Limits for y axis
	y_log : boolean
		Explicitly set y axis to linear or log space, otherwise go with default for y_param

	Returns
	-------
	None

	"""	

	# Setup x axis
	if x_param not in PARAM_INFO.keys():
		print("%s is not a valid parameter\n"%x_param)
		print("Valid parameters are:")
		print(PARAM_INFO.keys())
		return
	x_info = PARAM_INFO[x_param]
	xlabel = x_info[0]
	if x_lim == None:
		x_lim = x_info[1]
	if x_info[2] and (x_log or x_log==None):
		axis.set_xscale('log')
	axis.set_xlim(x_lim)

	# Setup y axis
	if y_param not in PARAM_INFO.keys():
		print("%s is not a valid parameter\n"%y_param)
		print("Valid parameters are:")
		print(PARAM_INFO.keys())
		return
	y_info = PARAM_INFO[y_param]
	ylabel = y_info[0]
	if y_lim == None:
		y_lim = y_info[1]
	if y_info[2] and (y_log or y_log==None):
		axis.set_yscale('log')
	axis.set_ylim(y_lim)

	# Set axis labels and ticks
	setup_labels(axis,xlabel,ylabel)

	return


def setup_labels(axis, xlabel, ylabel):
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
	axis.tick_params(axis='both', which='major', labelsize=Small_Font, length=8, width=2)
	axis.tick_params(axis='both', which='minor', labelsize=Small_Font, length=4, width=1)	
	for axe in ['top','bottom','left','right']:
  		axis.spines[axe].set_linewidth(2)