from config import *


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
		colors = LINE_COLORS
		linestyles = LINE_STYLES
	elif style == 'color':
		linewidths = np.full(num_sets,2)
		colors = LINE_COLORS
		linestyles = LINE_STYLES
	elif style == 'size':
		linewidths = LINE_WIDTHS
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

	axis.set_xlabel(xlabel, fontsize = LARGE_FONT)
	axis.set_ylabel(ylabel, fontsize = LARGE_FONT)
	axis.minorticks_on()
	axis.tick_params(axis='both',which='both',direction='in',right=True, top=True)
	axis.tick_params(axis='both', which='major', labelsize=SMALL_FONT, length=8, width=2)
	axis.tick_params(axis='both', which='minor', labelsize=SMALL_FONT, length=4, width=1)	
	for axe in ['top','bottom','left','right']:
  		axis.spines[axe].set_linewidth(2)