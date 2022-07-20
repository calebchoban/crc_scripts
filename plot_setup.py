import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from gizmo_library import config
from gizmo_library import utils


def setup_plot_style(snap_nums, properties=[], style='color-linestyle'):
	"""
	Sets up the handles for plot legend with specified style and parameters being plotted.

	Parameters
	----------
	snap_nums : int
		Number of snapshots you will be plotting
	properties: list
		List of properties that will also be plotted on the same plot if applicable
	style : string
		The style which will be used to differentiate the data sets.
		'color': each snapshot has different color
		'size': each snapshot has different line width
		'linestyle': each snapshot had different line style
		'color-linestyle': each snapshot had different color and line style

	Returns
	-------
	linewidths : list
		List of linewidths for each data set.
	colors : list
		List of colors for each data set.
	linestyles : list
		List of linestyles for each data set.

	"""

	# First case just plotting one snapshot so no need for legend handles
	if snap_nums == 1 and len(properties)<=1:
		linewidths = np.full(1,config.BASE_LINEWIDTH)
		colors = [config.BASE_COLOR]
		linestyles = [config.BASE_LINESTYLE]
	# In case we are plotting one snapshot with multiple properties on one plot follow the style choice to differentiate properties
	elif snap_nums == 1:
		colors = [config.BASE_COLOR]*len(properties)
		linestyles = [config.BASE_LINESTYLE]*len(properties)
		linewidths = [config.BASE_LINEWIDTH]*len(properties)
		if 'color' in style:
			colors = config.LINE_COLORS[:len(properties)]
		if 'linestyle' in style:
			linestyles = config.LINE_STYLES[:len(properties)]
		if 'size' in style:
			linewidths = config.LINE_WIDTHS[:len(properties)]
	# If we are just dealing with multiple snapshots follow the style choice to differentiate each of the snapshots
	elif len(properties) <= 1:
		colors = [config.BASE_COLOR]*snap_nums
		linestyles = [config.BASE_LINESTYLE]*snap_nums
		linewidths = [config.BASE_LINEWIDTH]*snap_nums
		if 'color' in style:
			colors = config.LINE_COLORS[:snap_nums]
		if 'linestyle' in style:
			linestyles = config.LINE_STYLES[:snap_nums]
		if 'size' in style:
			linewidths = config.LINE_WIDTHS[:snap_nums]
	# If there are multiple snapshots and properties then need to force each snapshot as a different linestyle and each parameter as a different color
	else:
		colors = []; linewidths = []; linestyles = [];
		for i in range(snap_nums):
			colors += [config.LINE_COLORS[i]]*len(properties)
			linewidths += [config.BASE_LINEWIDTH]*len(properties)
			linestyles += config.LINE_STYLES[:len(properties)]

	return linewidths, colors, linestyles





def setup_legend_handles(snap_nums, snap_labels=[], properties=[], style='color-line'):
	"""
	Sets up the handles for plot legend with specified style and parameters being plotted.

	Parameters
	----------
	snap_nums : int
		Number of snapshots you will be plotting
	snap_labels : list
		List of labels for snapshots to be plotted. Only need to provide if you want legend handles.
	properties: list
		List of properties that will also be plotted on the same plot if applicable
	style : string
		The style which will be used to differentiate the data sets.
		'color': each snapshot has different color
		'size': each snapshot has different line width
		'line': each snapshot had different line style
		'color-line': each snapshot had different color and line style

	Returns
	-------
	label_handles : dict
		Dictionary with labels and their corresponding handles to be given to plot legend
	linewidths : list
		List of linewidths for each data set.
	colors : list
		List of colors for each data set.
	linestyles : list
		List of linestyles for each data set.

	"""

	handles = []
	labels = []
	# First case just plotting one snapshot so no need for legend handles
	if snap_nums == 1 and len(properties)<=1:
		linewidths = np.full(1,config.BASE_LINEWIDTH)
		colors = [config.BASE_COLOR]
		linestyles = [config.BASE_LINESTYLE]
	# In case we are plotting one snapshot with multiple properties on one plot follow the style choice to differentiate properties
	elif snap_nums == 1:
		colors = [config.BASE_COLOR]*len(properties)
		linestyles = [config.BASE_LINESTYLE]*len(properties)
		linewidths = [config.BASE_LINEWIDTH]*len(properties)
		if 'color' in style:
			colors = config.LINE_COLORS[:len(properties)]
		if 'line' in style:
			linestyles = config.LINE_STYLES[:len(properties)]
		if 'size' in style:
			linewidths = config.LINE_WIDTHS[:len(properties)]
		for i, prop in enumerate(properties):
			handles += [mlines.Line2D([], [], color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i], label=config.PROP_INFO[prop][0])]
			labels += [config.PROP_INFO[prop][0]]
	# If we are just dealing with multiple snapshots follow the style choice to differentiate each of the snapshots
	elif len(properties) <= 1:
		colors = [config.BASE_COLOR]*snap_nums
		linestyles = [config.BASE_LINESTYLE]*snap_nums
		linewidths = [config.BASE_LINEWIDTH]*snap_nums
		if 'color' in style:
			colors = config.LINE_COLORS[:snap_nums]
		if 'line' in style:
			linestyles = config.LINE_STYLES[:snap_nums]
		if 'size' in style:
			linewidths = config.LINE_WIDTHS[:snap_nums]
		for i, label in enumerate(snap_labels):
			handles += [mlines.Line2D([], [], color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i], label=label)]
			labels += [label]
	# If there are multiple snapshots and properties then need to force each snapshot as a different linestyle and each parameter as a different color
	else:
		colors = []; linewidths = []; linestyles = [];
		for i, label in enumerate(snap_labels):
			colors += [config.LINE_COLORS[i]]*len(properties)
			linewidths += [config.BASE_LINEWIDTH]*len(properties)
			linestyles += config.LINE_STYLES[:len(properties)]
		for i, prop in enumerate(properties):
			handles += [mlines.Line2D([], [], color=config.BASE_COLOR, linestyle=config.LINE_STYLES[i], label=config.PROP_INFO[prop][0])]
			labels += [config.PROP_INFO[prop][0]]
		for i, label in enumerate(snap_labels):
			handles += [mlines.Line2D([], [], color=config.LINE_COLORS[i], linestyle=config.BASE_LINESTYLE,label=label)]
			labels += [label]

	label_handles = dict(zip(labels, handles))
	return label_handles, linewidths, colors, linestyles



def setup_figure(num_plots, orientation='horizontal', sharex=False, sharey=False):
	"""
	Sets up the figure size and subplot layout based on number of plots for a normal square aspect ratio plot

	Parameters
	----------
	num_plots : int
		Number of plots to be plotted
	orientation : string, optional
		Choose horizontal or vertical orientation when there are multiple subplots
	sharex,sharey : bool or {'none', 'all', 'row', 'col'}, default: False
		Same as sharex, sharey for matplotlib.pyplot.subplots

	Returns
	-------
	fig : Figure
		Matplotlib figure which houses plots
	axes: list
		List of axes for each plot
	"""

	if orientation not in ['vertical', 'horizontal']:
		print("Orientation must be either vertical or horizontal for setup_figure(). Assuming horizontal for now.")

	# add extra padding when there are axes labels
	if (sharex and orientation=='vertical') or (sharey and orientation=='horizontal'):
		label_pad = 0.
	else:
		label_pad = 0.

	if num_plots == 1:
		fig,axes = plt.subplots(1, 1, figsize=(config.BASE_FIG_XSIZE*config.FIG_XRATIO,config.BASE_FIG_YSIZE*config.FIG_YRATIO))
		axes = np.array([axes])
	elif num_plots%2 == 0:
		if orientation == 'vertical':
			fig,axes = plt.subplots(2, num_plots//2, figsize=(num_plots/2*config.BASE_FIG_XSIZE*config.FIG_XRATIO,2*(1+label_pad)*config.BASE_FIG_YSIZE*config.FIG_YRATIO),
									squeeze=True, sharex=sharex, sharey=sharey)
		else:
			fig,axes = plt.subplots(num_plots//2, 2, figsize=(2*(1+label_pad)*config.BASE_FIG_XSIZE*config.FIG_XRATIO,num_plots/2*config.BASE_FIG_YSIZE*config.FIG_YRATIO),
									squeeze=True, sharex=sharex, sharey=sharey)
	elif num_plots%3 == 0:
		if orientation == 'vertical':
			fig,axes = plt.subplots(3, num_plots//3, figsize=(num_plots/3*config.BASE_FIG_XSIZE*config.FIG_XRATIO,3*(1+label_pad)*config.BASE_FIG_YSIZE*config.FIG_YRATIO),
									squeeze=True, sharex=sharex, sharey=sharey)
		else:
			fig,axes = plt.subplots(num_plots//3, 3, figsize=(3*(1+label_pad)*config.BASE_FIG_XSIZE*config.FIG_XRATIO,num_plots/3*config.BASE_FIG_YSIZE*config.FIG_YRATIO),
									squeeze=True, sharex=sharex, sharey=sharey)
	else:
		dim_num = 3 # default number of plots in specified orientation
		if orientation == 'vertical':
			fig,axes = plt.subplots(dim_num, int(np.ceil(num_plots/dim_num)), figsize=(np.ceil(num_plots/dim_num)*config.BASE_FIG_XSIZE*config.FIG_XRATIO,dim_num*(1+label_pad)*config.BASE_FIG_YSIZE*config.FIG_YRATIO),
						squeeze=True, sharex=sharex, sharey=sharey)
			# Need to delete extra axes and reshow tick labels if axes were shared
			axes[(dim_num-1)-(dim_num-num_plots%dim_num),num_plots//dim_num].xaxis.set_tick_params(which='both', labelbottom=True, labeltop=False)
			for i in range(dim_num-num_plots%dim_num):
				fig.delaxes(axes[dim_num-1-i, num_plots//dim_num])
				axes[dim_num-1-i, num_plots//dim_num] = None
		else:
			fig,axes = plt.subplots(int(np.ceil(num_plots/dim_num)), dim_num, figsize=(dim_num*(1)*config.BASE_FIG_XSIZE*config.FIG_XRATIO,np.ceil(num_plots/dim_num)*config.BASE_FIG_YSIZE*config.FIG_YRATIO),
									squeeze=True, sharex=sharex, sharey=sharey)
			# Need to delete extra axes and reshow tick labels if axes were shared
			for i in range(dim_num-num_plots%dim_num):
				axes[num_plots//dim_num-1,dim_num-1-i].xaxis.set_tick_params(which='both', labelbottom=True, labeltop=False)
				fig.delaxes(axes[num_plots//dim_num,dim_num-1-i])
				axes[num_plots//dim_num,dim_num-1-i] = None

	# Get rid of the axes we may have deleted
	axes = list(filter(None, axes.flat))

	return fig,axes



def setup_axis(axis, x_prop, y_prop, x_lim=None, x_log=None, y_lim=None, y_log=None):
	"""
	Sets up the axis plot given x and y properties and optional limits

	Parameters
	----------
	axis : Matplotlib axis
	    Axis of plot
	x_prop : string
		Property to be plotted on x axis
	y_prop : string
		Property to be plotted on y axis
	x_lim : list
		Limits for x axis
	x_log : boolean
		Explicitly set x axis to linear or log space, otherwise go with default for x_param
	y_lim : list
		Limits for y axis
	y_log : boolean
		Explicitly set y axis to linear or log space, otherwise go with default for y_param

	Returns
	-------
	None

	"""

	# Setup x axis
	if x_prop not in config.PROP_INFO.keys():
		print("%s is not a valid property for plot_setup\n"%x_prop)
		print("Valid properties are:")
		print(config.PROP_INFO.keys())
		return
	x_info = config.PROP_INFO[x_prop]
	xlabel = x_info[0]
	if x_lim == None:
		x_lim = x_info[1]
	if x_info[2] or x_log:
		axis.set_xscale('log')
	axis.set_xlim(x_lim)

	# Setup y axis
	if y_prop not in config.PROP_INFO.keys():
		print("%s is not a valid property plot_setup\n"%y_prop)
		print("Valid properties are:")
		print(config.PROP_INFO.keys())
		return
	y_info = config.PROP_INFO[y_prop]
	ylabel = y_info[0]
	if y_lim == None:
		y_lim = y_info[1]
	if y_info[2] or y_log:
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
	xlabel : string
	    X axis label
	ylabel : string
	    Y axis label

	Returns
	-------
	None

	"""

	# Check if there are tick labels for each axis, not having them means the axes are shared and so
	# we don't want to add a label
	if (len(axis.get_xaxis().get_ticklabels())!=0):
		axis.set_xlabel(xlabel, fontsize = config.LARGE_FONT)
	if (len(axis.get_yaxis().get_ticklabels())!=0):
		axis.set_ylabel(ylabel, fontsize = config.LARGE_FONT)

	# If plotting redshift need to make some specific changes
	if xlabel == config.PROP_INFO['redshift'][0]:
		# First manually set major ticks since ususal redshift range is small but still logarithmically spaced
		xlims = axis.get_xlim()
		major_xticks = range(int(xlims[0]),int(xlims[1])-1,-1)
		axis.set_xticks(major_xticks,minor=False)
		axis.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
		axis.tick_params(axis='both',which='both',direction='in',right=True, top=True)
		axis.tick_params(axis='both', which='major', labelsize=config.SMALL_FONT, length=4*config.AXIS_BORDER_WIDTH, width=config.AXIS_BORDER_WIDTH)
		# Make sure not to put minor ticks on redshift axis
		axis.tick_params(axis='y', which='minor', labelsize=config.SMALL_FONT, length=2*config.AXIS_BORDER_WIDTH,width=config.AXIS_BORDER_WIDTH/2)

		# TODO: Add ability to plot cosmic time on top of plot and share this top axis across multiple subplots
		# For redshift need to set tick locations and add secondary x-axis for cosmic time
		# axis2 = axis.twiny()
		# axis2.set_xlim(utils.quick_lookback_time(1/xlims[0]),utils.quick_lookback_time(1/xlims[1]))
		# axis2.set_yscale("linear")
		# if (len(axis.get_xaxis().get_ticklabels())!=0):
		# 	axis2.set_xlabel('Cosmic Time [Gyr]', fontsize = config.LARGE_FONT)
		#
		# axis2.tick_params(axis='x',which='both',direction='in',top=True)
		# axis2.tick_params(axis='x', which='major', labelsize=config.SMALL_FONT, length=4*config.AXIS_BORDER_WIDTH, width=config.AXIS_BORDER_WIDTH)
		# axis2.tick_params(axis='x', which='minor', labelsize=config.SMALL_FONT, length=2*config.AXIS_BORDER_WIDTH,width=config.AXIS_BORDER_WIDTH/2)

	else:
		axis.minorticks_on()
		axis.tick_params(axis='both',which='both',direction='in',right=True, top=True)
		axis.tick_params(axis='both', which='major', labelsize=config.SMALL_FONT, length=4*config.AXIS_BORDER_WIDTH, width=config.AXIS_BORDER_WIDTH)
		axis.tick_params(axis='both', which='minor', labelsize=config.SMALL_FONT, length=2*config.AXIS_BORDER_WIDTH, width=config.AXIS_BORDER_WIDTH/2)

	for axe in ['top','bottom','left','right']:
		axis.spines[axe].set_linewidth(config.AXIS_BORDER_WIDTH)



def setup_colorbar(image, axis, label):
	"""
	Sets the colorbar and its label and ticks for the given axis.

	Parameters
	----------
	image : mappable
		The image this colorbar is associated with
	axis : Matplotlib axis
	    Axis of plot
	label : string
	    Color bar label


	Returns
	-------
	None

	"""

	divider = make_axes_locatable(axis)
	cax = divider.append_axes("right", size="5%", pad=0.0)
	cbar = plt.colorbar(image, cax=cax)
	cbar.ax.set_ylabel(label, fontsize=config.LARGE_FONT)
	cbar.ax.minorticks_on()
	cbar.ax.tick_params(axis='both',which='both',direction='in',right=True)
	cbar.ax.tick_params(axis='both', which='major', labelsize=config.SMALL_FONT, length=4*config.AXIS_BORDER_WIDTH, width=config.AXIS_BORDER_WIDTH)
	cbar.ax.tick_params(axis='both', which='minor', labelsize=config.SMALL_FONT, length=2*config.AXIS_BORDER_WIDTH, width=config.AXIS_BORDER_WIDTH/2)
	cbar.outline.set_linewidth(config.AXIS_BORDER_WIDTH)

	return




def setup_2D_hist_fig(hist_proj = True):

	if not hist_proj:
		fig,axes = plt.subplots(1, 1, figsize=(14/1.2,10/1.2))
		axes = np.array([axes])

	else:
		fig = plt.figure(1, figsize=(14/1.2,10/1.2))
		# definitions for the axes
		left, width = 0.1, 0.65
		bottom, height = 0.1, 0.65
		bottom_h = left_h = left + width + 0.02

		rect_scatter = [left, bottom, width, height]
		rect_histx = [left, bottom_h, width, 0.2]
		rect_histy = [left_h, bottom, 0.2, height]

		# start with a rectangular Figure
		plt.figure(1, figsize=(8, 8))

		axHist2D = plt.axes(rect_scatter)
		axHistx = plt.axes(rect_histx)
		axHisty = plt.axes(rect_histy)

		# no labels
		axHistx.xaxis.set_major_formatter(nullfmt)
		axHisty.yaxis.set_major_formatter(nullfmt)

		axes = np.array([axHist2D,axHistx,axHisty])

	return fig,axes



def setup_projection(num_plots,L,Lz=None):

	axes = []
	base_size = 10
	# Ratio of color bar size to projection plot
	cbar_ratio = 0.05

	# X-Y and X-Z Projection
	if Lz != None:
		# gs = gridspec.GridSpec(3,num_plots,height_ratios=[L,Lz,cbar_ratio*L])
		# #gs.update(hspace=0.025,wspace=0.025,top=0.99, bottom=0.075, left=0.025, right=0.975)
		# ratio = 1+1.0*Lz/L + 2*cbar_ratio
		# fig=plt.figure(figsize=(num_plots*base_size,ratio*base_size))
		# for i in range(num_plots):
		# 	ax1 = plt.subplot(gs[0,i])
		# 	#ax1.xaxis.set_visible(False)
		# 	#ax1.yaxis.set_visible(False)
		# 	ax1.set_aspect('equal', adjustable='box')
		# 	ax2 = plt.subplot(gs[1,i])
		# 	#ax2.xaxis.set_visible(False)
		# 	#ax2.yaxis.set_visible(False)
		# 	ax2.set_aspect('equal', adjustable='box')
		# 	cbarax = plt.subplot(gs[2,i])
		# 	axes += [[ax1,ax2,cbarax]]

		# fig, axs = plt.subplots(nrows=3, ncols=1, gridspec_kw={'hspace':0.1, 'height_ratios':[L,Lz,cbar_ratio*L]},
		# 						figsize=(12*1.1, 12*(1+cbar_ratio+Lz/L)))
		# ax1 = axs[0]
		# ax1.set_xlim([-L,L])
		# ax1.set_ylim([-L,L])
		# ax1.xaxis.set_visible(False)
		# ax1.yaxis.set_visible(False)
		# ax1.set_aspect('equal', adjustable='box')
		# ax2 = axs[1]
		# ax2.set_ylim([-Lz,Lz])
		# ax2.set_xlim([-L,L])
		# ax2.xaxis.set_visible(False)
		# ax2.yaxis.set_visible(False)
		# ax2.set_aspect('equal', adjustable='box')
		#
		#
		# ax3 = axs[2]
		# ax3.set_ylim([-cbar_ratio*L,cbar_ratio*L])
		# ax3.set_xlim([-L,L])
		#
		# axes += [axs]
		#plt.tight_layout()

		fig, axs = plt.subplots(nrows=2, ncols=num_plots, gridspec_kw={'hspace':0.01,'wspace':0.1,'height_ratios':[L,Lz]},
								figsize=[num_plots*config.BASE_FIG_SIZE,(L+Lz)/L*(1+2*cbar_ratio)*config.BASE_FIG_SIZE])
		# Deal with only one projection being plotted
		if num_plots==1:
			axs = np.array([[axs[0]],[axs[1]]])

		for i in range(num_plots):
			ax1 = axs[0,i]
			ax1.set_xlim([-L,L])
			ax1.set_ylim([-L,L])
			ax1.xaxis.set_visible(False)
			ax1.yaxis.set_visible(False)
			ax1.set_aspect('equal', adjustable='box')
			for axe in ['top','bottom','left','right']:
				ax1.spines[axe].set_linewidth(config.AXIS_BORDER_WIDTH)
			ax2 = axs[1,i]
			ax2.set_ylim([-Lz,Lz])
			ax2.set_xlim([-L,L])
			ax2.xaxis.set_visible(False)
			ax2.yaxis.set_visible(False)
			ax2.set_aspect('equal', adjustable='box')
			for axe in ['top','bottom','left','right']:
					ax2.spines[axe].set_linewidth(config.AXIS_BORDER_WIDTH)

			# Add scale bar
			bar, label = find_scale_bar(L)
			ax1.plot([-0.7*L-bar/2,-0.7*L+bar/2], [-0.87*L,-0.87*L], '-', c='xkcd:white', lw=config.BASE_LINEWIDTH)
			ax1.annotate(label, (0.15,0.05), xycoords='axes fraction', color='xkcd:white', ha='center', va='top', fontsize=config.SMALL_FONT)

		axes = axs

	# Only X-Y Projection
	else:
		gs = gridspec.GridSpec(2,num_plots,height_ratios=[L,cbar_ratio*L],hspace=0.0,wspace=0.025,top=0.975, bottom=0.025, left=0.025, right=0.975)
		ratio = 1.+cbar_ratio
		fig=plt.figure(figsize=(num_plots*base_size,ratio*base_size))
		for i in range(num_plots):
			ax = plt.subplot(gs[0,i])
			ax1.xaxis.set_visible(False)
			ax1.yaxis.set_visible(False)
			cbarax = plt.subplot(gs[1,i])
			axes += [[ax,cbarax]]


	
	return fig, np.array(axes)



# find appropriate scale bar and label
def find_scale_bar(L):

	if (L>=10000):
		bar = 1000.; label = '1 Mpc'
	elif (L>=1000):
		bar = 100.; label = '100 kpc'
	elif (L>=500):
		bar = 50.; label = '50 kpc'
	elif (L>=200):
		bar = 20.; label = '20 kpc'
	elif (L>=100):
		bar = 10.; label = '10 kpc'
	elif (L>=50):
		bar = 5.; label = '5 kpc'
	elif (L>=20):
		bar = 2.; label = '2 kpc'
	elif (L>=10):
		bar = 1.; label = '1 kpc'
	elif (L>=5):
		bar = 0.5; label = '500 pc'
	elif (L>=2):
		bar = 0.2; label = '200 pc'
	elif (L>=1):
		bar = 0.1; label = '100 pc'
	elif (L>0.5):
		bar = 0.05; label = '50 pc'
	elif (L>0.2):
		bar = 0.02; label = '20 pc'
	elif (L>0.1):
		bar = 0.01; label = '10 pc'
	elif (L>0.05):
		bar = 0.005; label = '5 pc'
	elif (L>0.02):
		bar = 0.002; label = '2 pc'
	elif (L>0.01):
		bar = 0.001; label = '1 pc'
	else:
		bar = 0.0005; label = '0.5 pc'

	return bar, label