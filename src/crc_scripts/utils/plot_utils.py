from copy import copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .. import config
from . import math_utils


def setup_plot_style(num_datasets, num_sub_datasets=1, style='color-linestyle'):
    """
    This will give you lists of linestyles, colors, and linewidths to be used when plotting the specified number of datasets.
    Depending on the style specified each dataset will have a unique linestyles, colors, and/or linewidths.
    If you are plotting sub

    Parameters
    ----------
    num_datasets : int
        Number of datasets you will be plotting. As an example this could be the number of individual snapshots or masks for particles.
    num_sub_datasets: int
        Number of sub datasets you will have on each plot. For example you are plotting multiple snapshots with different particle masks on the same plot.
    style : string
        The style which will be used to differentiate the datasets.
        'color': each dataset has a different color
        'size': each dataset has a different line width
        'linestyle': each dataset has a different line style
        These can be combined by passing one string such as 'color-linestyle'.

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
    if num_datasets == 1 and num_sub_datasets<=1:
        linewidths = np.full(1,config.BASE_LINEWIDTH)
        colors = [config.BASE_COLOR]
        linestyles = [config.BASE_LINESTYLE]
    # In case we are plotting one snapshot with multiple properties on one plot follow the style choice to differentiate properties
    elif num_datasets == 1:
        colors = [config.BASE_COLOR]*num_sub_datasets
        linestyles = [config.BASE_LINESTYLE]*num_sub_datasets
        linewidths = [config.BASE_LINEWIDTH]*num_sub_datasets
        if 'color' in style:
            colors = config.LINE_COLORS[:num_sub_datasets]
        if 'linestyle' in style:
            linestyles = config.LINE_STYLES[:num_sub_datasets]
        if 'size' in style:
            linewidths = config.LINE_WIDTHS[:num_sub_datasets]
    # If we are just dealing with multiple snapshots follow the style choice to differentiate each of the snapshots
    elif num_sub_datasets <= 1:
        colors = [config.BASE_COLOR]*num_datasets
        linestyles = [config.BASE_LINESTYLE]*num_datasets
        linewidths = [config.BASE_LINEWIDTH]*num_datasets
        if 'color' in style:
            colors = config.LINE_COLORS[:num_datasets]
        if 'linestyle' in style:
            linestyles = config.LINE_STYLES[:num_datasets]
        if 'size' in style:
            linewidths = config.LINE_WIDTHS[:num_datasets]
    # If there are multiple snapshots and properties then need to force each snapshot as a different linestyle and each parameter as a different color
    else:
        colors = []; linewidths = []; linestyles = [];
        for i in range(num_datasets):
            colors += [config.LINE_COLORS[i]]*num_sub_datasets
            linewidths += [config.BASE_LINEWIDTH]*num_sub_datasets
            linestyles += config.LINE_STYLES[:num_sub_datasets]

    return linewidths, colors, linestyles




# def setup_legend_handles(snap_nums, snap_labels=[], properties=[], style='color-line'):
#     """
#     Sets up the handles for plot legend with specified style and parameters being plotted.

#     Parameters
#     ----------
#     snap_nums : int
#         Number of snapshots you will be plotting
#     snap_labels : list
#         List of labels for snapshots to be plotted. Only need to provide if you want legend handles.
#     properties: list
#         List of properties that will also be plotted on the same plot if applicable
#     style : string
#         The style which will be used to differentiate the data sets.
#         'color': each snapshot has different color
#         'size': each snapshot has different line width
#         'line': each snapshot had different line style
#         'color-line': each snapshot had different color and line style

#     Returns
#     -------
#     label_handles : dict
#         Dictionary with labels and their corresponding handles to be given to plot legend
#     linewidths : list
#         List of linewidths for each data set.
#     colors : list
#         List of colors for each data set.
#     linestyles : list
#         List of linestyles for each data set.

#     """

#     handles = []
#     labels = []
#     # First case just plotting one snapshot so no need for legend handles
#     if snap_nums == 1 and len(properties)<=1:
#         linewidths = np.full(1,config.BASE_LINEWIDTH)
#         colors = [config.BASE_COLOR]
#         linestyles = [config.BASE_LINESTYLE]
#     # In case we are plotting one snapshot with multiple properties on one plot follow the style choice to differentiate properties
#     elif snap_nums == 1:
#         colors = [config.BASE_COLOR]*len(properties)
#         linestyles = [config.BASE_LINESTYLE]*len(properties)
#         linewidths = [config.BASE_LINEWIDTH]*len(properties)
#         if 'color' in style:
#             colors = config.LINE_COLORS[:len(properties)]
#         if 'line' in style:
#             linestyles = config.LINE_STYLES[:len(properties)]
#         if 'size' in style:
#             linewidths = config.LINE_WIDTHS[:len(properties)]
#         for i, prop in enumerate(properties):
#             handles += [mlines.Line2D([], [], color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i], label=config.PROP_INFO[prop][0])]
#             labels += [config.PROP_INFO[prop][0]]
#     # If we are just dealing with multiple snapshots follow the style choice to differentiate each of the snapshots
#     elif len(properties) <= 1:
#         colors = [config.BASE_COLOR]*snap_nums
#         linestyles = [config.BASE_LINESTYLE]*snap_nums
#         linewidths = [config.BASE_LINEWIDTH]*snap_nums
#         if 'color' in style:
#             colors = config.LINE_COLORS[:snap_nums]
#         if 'line' in style:
#             linestyles = config.LINE_STYLES[:snap_nums]
#         if 'size' in style:
#             linewidths = config.LINE_WIDTHS[:snap_nums]
#         for i, label in enumerate(snap_labels):
#             handles += [mlines.Line2D([], [], color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i], label=label)]
#             labels += [label]
#     # If there are multiple snapshots and properties then need to force each snapshot as a different linestyle and each parameter as a different color
#     else:
#         colors = []; linewidths = []; linestyles = [];
#         for i, label in enumerate(snap_labels):
#             colors += [config.LINE_COLORS[i]]*len(properties)
#             linewidths += [config.BASE_LINEWIDTH]*len(properties)
#             linestyles += config.LINE_STYLES[:len(properties)]
#         for i, prop in enumerate(properties):
#             handles += [mlines.Line2D([], [], color=config.BASE_COLOR, linestyle=config.LINE_STYLES[i], label=config.PROP_INFO[prop][0])]
#             labels += [config.PROP_INFO[prop][0]]
#         for i, label in enumerate(snap_labels):
#             handles += [mlines.Line2D([], [], color=config.LINE_COLORS[i], linestyle=config.BASE_LINESTYLE,label=label)]
#             labels += [label]

#     label_handles = dict(zip(labels, handles))
#     return label_handles, linewidths, colors, linestyles



def setup_figure(num_plots, orientation=config.DEFAULT_PLOT_ORIENTATION, sharex=False, sharey=False, yx_ratio=1., ncols=None, 
                 sqeezespace=0.05):
    """
    Sets up the figure size and subplot layout based on number of plots for a normal square aspect ratio plot

    Parameters
    ----------
    num_plots : int
        Number of plots to be plotted
    orientation : string, optional
        Choose 'horizontal' or 'vertical' orientation when there are multiple subplots
    sharex,sharey : bool or {'none', 'all', 'row', 'col'}, optional
        Same as sharex, sharey for matplotlib.pyplot.subplots
    yx_ratio: float, optional
        Scale relative size between y and x axis. Default is 1:1.
    ncols: int, optional
        Force number of columns given multiple plots. This overrids orientation.
    sqeezespace: float, optional
        The space between plots which are sharing axis as given by sharex and sharey.

    Returns
    -------
    fig : Figure
        Matplotlib figure which houses plots
    axes : list
        List of axes for each plot
    dims : list
        Number of plot rows and columns of figure.
    """

    if orientation not in ['vertical', 'horizontal']:
        print("Orientation must be either vertical or horizontal for setup_figure(). Assuming horizontal for now.")

    yx_ratio *= config.BASE_AXES_RATIO

    if num_plots == 1:
        fig,axes = plt.subplots(1, 1, figsize=(config.BASE_FIG_SIZE,config.BASE_FIG_SIZE*yx_ratio))
        axes = np.array([axes])
        dims = np.array([1,1])
    else:
        # If number of columns specified there is only one thing to do
        if ncols is None:
            # Default 2 or 3
            if num_plots%2 == 0 and num_plots<5:
                if orientation == 'vertical': ncols = num_plots//2+1
                else: ncols = 2
            else:
                if orientation == 'vertical': ncols = num_plots//3+1
                else: ncols = 3
        nrows = int(np.ceil(num_plots/ncols))
        fig,axes = plt.subplots(nrows, ncols, figsize=(ncols*config.BASE_FIG_SIZE,np.ceil(num_plots/ncols)*config.BASE_FIG_SIZE*yx_ratio),
                                    squeeze=True, sharex=sharex, sharey=sharey)
        # Need to delete extra axes and reshow tick labels if axes were shared
        if num_plots%ncols > 0:
            for i in range(ncols-num_plots%ncols):
                axes[num_plots//ncols-1,ncols-1-i].xaxis.set_tick_params(which='both', labelbottom=True, labeltop=False)
                fig.delaxes(axes[num_plots//ncols,ncols-1-i])
                axes[num_plots//ncols,ncols-1-i] = None
        dims = np.array([num_plots//ncols,ncols])

    # Get rid of the axes we may have deleted
    axes = list(filter(None, axes.flat))

    # Squish axes together if they are the same
    if sharex:
        fig.subplots_adjust(hspace=sqeezespace)
    if sharey:
        fig.subplots_adjust(wspace=sqeezespace)

    return fig,axes,dims


def add_artists(axis, artists):
    """
    Adds the given artists to the given axis. Useful when you want to add a line or something to a plot.

    Parameters
    ----------
    axis : Axis
        Matplotlib axis to add artists to
    artists : list
        List of Matplotlib artists to be added

    Returns
    -------
    None

    """
    if artists is not None and artists[0] is not None:
        for artist in artists:
            # Just in case you reused the same artist across axes make a copy and add that instead
            axis.add_artist(copy(artist))
    return


def setup_axis(axis, x_prop, y_prop, x_label=None, y_label=None, x_lim=None, x_log=None, y_lim=None, y_log=None, 
               artists_to_add=None, face_color='xkcd:white'):
    """
    Sets up the axis plot given x and y properties. Supported properties are listed in config.PROP_INFO and unless given, this
    is where the limits for each axis and log/linear scale are determined. 

    Parameters
    ----------
    axis : Axis
        Axis of plot
    x_prop : string
        Property to be plotted on x axis
    y_prop : string
        Property to be plotted on y axis
    x_label : string, optional
        Label for x axis (will override default label for give x_prop)
    y_label : string, optional
        Label for y axis (will override default label for give y_prop)
    x_lim : list, optional
        Limits for x axis
    x_log : boolean, optional
        Explicitly set x axis to linear or log space, otherwise go with default for x_param
    y_lim : list, optional
        Limits for y axis
    y_log : boolean, optional
        Explicitly set y axis to linear or log space, otherwise go with default for y_param
    artitst_to_add : list, optional
        List of matplotlib artists objects to be added to axis
    face_color : string, optional
        Axis background color

    Returns
    -------
    None

    """

    # Setup x axis
    if x_prop not in config.PROP_INFO.keys() and (x_label is None or x_lim is None):
        print("%s is not a supported property for plot_setup\n"%x_prop)
        print("Either give x_label and x_lim to make your own or choose from supported properties.")
        print("Valid properties are:")
        print(config.PROP_INFO.keys())
        return
    if x_lim == None:
        x_lim = config.get_prop_limits(x_prop)
    if (x_log is not None and x_log) or (x_log is None and config.get_prop_if_log(x_prop)):
        axis.set_xscale('log')
    else:
        axis.ticklabel_format(axis='x',style='plain')
    axis.set_xlim(x_lim)

    # Setup y axis
    if y_prop not in config.PROP_INFO.keys() and (y_label is None or y_lim is None):
        print("%s is not a supported property for plot_setup\n"%y_prop)
        print("Either give y_label and y_lim to make your own or choose from supported properties.")
        print("Valid properties are:")
        print(config.PROP_INFO.keys())
        return
    if y_lim == None:
        y_lim = config.get_prop_limits(y_prop)
    if (y_log is not None and y_log) or (y_log is None and config.get_prop_if_log(y_prop)):
        axis.set_yscale('log')
    else:
        axis.ticklabel_format(axis='y',style='plain')
    axis.set_ylim(y_lim)

    axis.set_facecolor(face_color)

    # Set axis labels and ticks
    setup_labels(axis,x_prop,y_prop, x_label=x_label, y_label=y_label)
    # Add given artist to axis
    add_artists(axis,artists_to_add)

    return


def setup_labels(axis, x_prop, y_prop, x_label=None, y_label=None,):
    """
    Sets the axis labels based on the given properties. Ticks are set so they face inwards and have minor and major ticks.
    Special ticks are given for redshift only.

    Parameters
    ----------
    axis : Matplotlib axis
        Axis of plot
    x_prop : string
        Name of property on x axis
    y_prop : string
        Name of property on y axis
    x_label : string, optional
        X axis label (overrides default label for x_prop)
    y_label : string, optional
        Y axis label (overrides default label for y_prop)

    Returns
    -------
    None

    """

    # Check if there are tick labels for each axis, not having them means the axes are shared and so
    # we don't want to add a label
    if (len(axis.get_xaxis().get_ticklabels())!=0):
        label = x_label if x_label is not None else config.get_prop_label(x_prop)
        axis.set_xlabel(label, fontsize = config.LARGE_FONT)
    if (len(axis.get_yaxis().get_ticklabels())!=0):
        label = y_label if y_label is not None else config.get_prop_label(y_prop)
        axis.set_ylabel(label, fontsize = config.LARGE_FONT)

    # If plotting redshift need to make some specific changes
    if x_prop in ['redshift','redshift_plus_1']:
        # First manually set major ticks since usual redshift range is small but still logarithmically spaced
        xlims = axis.get_xlim()
        major_xticks = range(int(xlims[0]),int(xlims[1])-1,-1)
        axis.set_xticks(major_xticks,minor=False)
        axis.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        axis.tick_params(axis='both',which='both',direction='in',right=True, top=True)
        axis.tick_params(axis='both', which='major', labelsize=config.SMALL_FONT, length=4*config.AXIS_BORDER_WIDTH, width=config.AXIS_BORDER_WIDTH)
        # Make sure not to put minor ticks on redshift axis
        axis.tick_params(axis='x', which='minor', labelsize=config.SMALL_FONT, length=2*config.AXIS_BORDER_WIDTH,width=config.AXIS_BORDER_WIDTH/2)
    else:
        axis.minorticks_on()
        axis.tick_params(axis='both',which='both',direction='in',right=True, top=True)
        axis.tick_params(axis='both', which='major', labelsize=config.SMALL_FONT, length=4*config.AXIS_BORDER_WIDTH, width=config.AXIS_BORDER_WIDTH)
        axis.tick_params(axis='both', which='minor', labelsize=config.SMALL_FONT, length=2*config.AXIS_BORDER_WIDTH, width=config.AXIS_BORDER_WIDTH/2)

    for axe in ['top','bottom','left','right']:
        axis.spines[axe].set_linewidth(config.AXIS_BORDER_WIDTH)



def make_axis_secondary_time(axis, time_name, snapshot=None, tick_labels=True):
    '''
    Make secondary x-axis for time. The secondary time parameter is chosen based on what time is already plotted and the scale
    of the x-axis. If physical time (e.g. Gyr) is plotted then redshift/redshift+1 if plotted on the second axis if the x-axis 
    scale is linear/log. If redshift/redshift+1 is plotted then lookback time is plotted on secondary axis.


    Parameters
    ----------
    axis : Axis
        Pyplot axis you want to add the secondary time axis to. This axis must already have the primary x-axis time limits set.
    time_name : str
        The type of time on existing (primary) axis: 'time', 'time_lookback', 'redshift', 'redshift_plus_1'. 
    snapshot : Snapshot
        One of the snapshots to be plotted which hold cosmological constants used for time conversion. If set to None then
        cosmological constants are assumed.
    tick_labels : boolean
        Set whether the seconday axis should include tick labels
    
    Returns
    -------
    None

    '''

    # If physical time then we want redshift for secondary axis
    if time_name in ['time', 'time_lookback']:
        time_limits = axis.get_xlim()
        # If axis is log scale need to plot z+1
        if axis.get_xaxis().get_scale()=='log':
            axis_2_name = 'redshift_plus_1'
            # Depending on the timespan may want more redshift tick labels
            if time_limits[1] < 1:
                axis_2_tick_labels = ['13','12','11','10','9','8','7']
            elif time_limits[1] < 3:
                axis_2_tick_labels = ['11','9','8', '7', '6', '5', '4', '3', '2', '1.5', '1.2', '1']
            else:
                axis_2_tick_labels = ['7', '5', '4', '3', '2', '1.5', '1.2', '1']
        else:
            axis_2_name = 'redshift'
            # Depending on the timespan may want more redshift tick labels
            if time_limits[1] < 1:
                axis_2_tick_labels = ['12','11','10','9','8','7','6']
            elif time_limits[1] < 3:
                axis_2_tick_labels = ['10','8', '7', '6', '5', '4', '3', '2', '1', '0.5', '0.2', '0']
            else:
                axis_2_tick_labels = ['6', '4', '3', '2', '1', '0.5', '0.2', '0']
        axis_2_tick_values = np.array([float(v) for v in axis_2_tick_labels])
        # get an interpolation function to covert from physical time to redshift
        conv_func = math_utils.get_time_conversion_spline(time_name,axis_2_name,sp=snapshot)
        axis_2_tick_locations = conv_func(axis_2_tick_values)

    elif time_name in ['redshift','redshift_plus_1']:
        axis_2_name = 'time_lookback'
        axis_2_tick_labels = ['0', '2', '4', '6', '8', '10', '11', '12', '12.5', '13']
        axis_2_tick_values = np.array([float(v) for v in axis_2_tick_labels])
        conv_func = math_utils.get_time_conversion_spline(time_name,'timelookback_',sp=snapshot)
        axis_2_tick_locations = conv_func(axis_2_tick_values)
    
    else:
        print("make_axis_secondary_time() failed because %s is not a supported time property."%time_name)


    axis2 = axis.twiny()
    # Need to turn on minor ticks so that y-axis will have them
    axis2.minorticks_on()
    axis2.set_xscale(axis.get_xaxis().get_scale())
    # Force scalar notation for labels or else scientific notation might pop up
    axis2.get_xaxis().set_major_formatter(mticker.ScalarFormatter()) 
    axis2.set_xticks(axis_2_tick_locations,minor=False)
    if tick_labels:
        axis2.set_xticklabels(axis_2_tick_labels)
        axis2.set_xlabel(config.get_prop_label(axis_2_name), fontsize = config.LARGE_FONT, labelpad=9)
    else:
        axis2.set_xticklabels([])
    axis2.set_xlim(axis.get_xlim()) # Need to reset limits to twinned axis after making ticks
    axis2.xaxis.set_minor_locator(mticker.NullLocator()) # Turn off minor ticks which reappear when setting xlimits
    axis2.tick_params(axis='x', which='major',direction='in', labelsize=config.SMALL_FONT, length=4*config.AXIS_BORDER_WIDTH, width=config.AXIS_BORDER_WIDTH)
    



def setup_colorbar(image, axis, label):
    """
    Sets the colorbar and its label and ticks for the given axis.

    Parameters
    ----------
    image : Matplotlib mappable
        The image this colorbar is associated with
    axis : Axis
        Axis of plot to have color bar added
    label : string
        Color bar label


    Returns
    -------
    None

    """
    
    # Added another axis for the colorbar to existing axis
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.0)
    # Add colobar to new axis and give it a mappable image so it knows what color map to use
    cbar = plt.colorbar(image, cax=cax)
    cbar.ax.set_ylabel(label, fontsize=config.LARGE_FONT)
    # Set ticks to the way I like them
    cbar.ax.minorticks_on()
    cbar.ax.tick_params(axis='both',which='both',direction='in',right=True)
    cbar.ax.tick_params(axis='both', which='major', labelsize=config.SMALL_FONT, length=4*config.AXIS_BORDER_WIDTH, width=config.AXIS_BORDER_WIDTH)
    cbar.ax.tick_params(axis='both', which='minor', labelsize=config.SMALL_FONT, length=2*config.AXIS_BORDER_WIDTH, width=config.AXIS_BORDER_WIDTH/2)
    cbar.outline.set_linewidth(config.AXIS_BORDER_WIDTH)

    return


def setup_proj_figure(num_plots,add_sub_projs,add_colorbars=True,height_ratios=[5,1]):
    """
    Sets up figure for projection plots given the number of plots you want and if they are going to include sub-projections. 

    Parameters
    ----------
    num_plots : int
        The number of projections you want.
    add_sub_projs : bool
        Do you want secondary projections for each projection (use this if you want an additional edge-on projection)  
    add_colorbars : bool
        Toggle whether each projection has a colorbar.
    height_ratios : list, optional
        Shape (2) ratio between height of projection and sub_projection if set. Default 5 to 1.
        


    Returns
    -------
    fig: Figure
        Matplotlib Figure with projection plots.
    axes: list
        List of axes for each plot in figure.

    """


    axes = []
    height_ratios = np.array(height_ratios)
    # Default height of color bar to 1/10th of main projection
    cbar_height = 0.05*height_ratios[0]
    width_space = 0.02*height_ratios[0]
    height_space = 0.02*height_ratios[0]
    # Check whether you want another projection below the first (usually edge-on with a disk)
    if add_sub_projs:
        if add_colorbars:
            fig, axes = plt.subplots(nrows=3, ncols=num_plots, gridspec_kw={'hspace':height_space,'wspace':width_space,'height_ratios':np.append(height_ratios,[cbar_height])},
                                    figsize=[num_plots*config.BASE_FIG_SIZE,
                                            (height_ratios[0]+height_ratios[1]+cbar_height)/height_ratios[0]*config.BASE_FIG_SIZE])
        else:
            fig, axes = plt.subplots(nrows=2, ncols=num_plots, gridspec_kw={'hspace':height_space,'wspace':width_space,'height_ratios':height_ratios},
                                    figsize=[num_plots*config.BASE_FIG_SIZE,
                                            (height_ratios[0]+height_ratios[1])/height_ratios[0]*config.BASE_FIG_SIZE])     
        # Deal with only one projection being plotted
        if num_plots==1:
            axes = np.array([[axes[0],axes[1],axes[2]]])
        else:
            axes = np.array(axes).T
        for axis_set in axes: 
            axis_set[0].set_aspect('equal', adjustable='box')
            axis_set[1].set_aspect('equal', adjustable='box')
            # Just to give a preview of what the whole figure will look like
            axis_set[1].set_ylim(0,height_ratios[1]/height_ratios[0])

    # Only one projection
    else:
        nrows=1
        if add_colorbars:
            height_ratios=np.append(height_ratios,[cbar_height])
            nrows=2
        gs = gridspec.GridSpec(nrows,num_plots,height_ratios=height_ratios,hspace=height_space,wspace=width_space,top=0.975, bottom=0.025, left=0.025, right=0.975)
        ratio = (height_ratios[0]+height_ratios[1])/height_ratios[0] if add_colorbars else 1
        fig=plt.figure(figsize=(num_plots*config.BASE_FIG_SIZE,ratio* config.BASE_FIG_SIZE))
        for i in range(num_plots):
            ax = plt.subplot(gs[0,i])
            cbarax = plt.subplot(gs[1,i])
            axes += [[ax,cbarax]]
        axes = np.array(axes)
    
    return fig, axes


def setup_proj_axis(axes, main_L, sub_L=None, axes_visible=False):
    """
    Sets up axis, ticks, and scale bar for specified projection size. 

    Parameters
    ----------
    axes : list
        List with projection axes you want to setup. Expects one/two axes if sub_projection is true/false.
    main_L : float
        Size of main projection (L x L)
    su_L : float, optional
        Size of subprojection (L x sub_L)
    axes_visible : bool, optional
        Set whether you want axes with ticks visible along projections.

    """

    if sub_L is None:
        ax1 = axes[0]
        ax1.set_xlim([-main_L,main_L])
        ax1.set_ylim([-main_L,main_L])
        if not axes_visible:
            ax1.xaxis.set_visible(False)
            ax1.yaxis.set_visible(False)
        for axe in ['top','bottom','left','right']:
            ax1.spines[axe].set_linewidth(config.AXIS_BORDER_WIDTH)
    # If there is a subprojection set that up as well
    else:
        ax1 = axes[0]
        ax1.set_xlim([-main_L,main_L])
        ax1.set_ylim([-main_L,main_L])
        for axe in ['top','bottom','left','right']:
            ax1.spines[axe].set_linewidth(config.AXIS_BORDER_WIDTH)

        ax2 = axes[1]
        ax2.set_ylim([-sub_L, sub_L])
        ax2.set_xlim([-main_L,main_L])
        for axe in ['top','bottom','left','right']:
                ax2.spines[axe].set_linewidth(config.AXIS_BORDER_WIDTH)
        
        if not axes_visible:
            ax1.xaxis.set_visible(False)
            ax1.yaxis.set_visible(False) 
            ax2.xaxis.set_visible(False)
            ax2.yaxis.set_visible(False)

    # Add scale bar to main projection
    bar, label = find_scale_bar(main_L)
    ax1.plot([-0.7*main_L-bar/2,-0.7*main_L+bar/2], [-0.87*main_L,-0.87*main_L], '-', c='xkcd:white', lw=config.BASE_LINEWIDTH)
    ax1.annotate(label, (0.15,0.05), xycoords='axes fraction', color='xkcd:white', ha='center', va='top', fontsize=config.SMALL_FONT)


def setup_proj_colorbar(property, fig, caxis, mappable=None, cmap='magma', label=None, limits=None, log=False):
    """
    Sets up colorbar for given projection. 

    Parameters
    ----------
    property : string
        Name of property displayed by colorbar. Supported properties are listed in config.PROP_INFO and unless given, this
    is where the limits for each axis and log/linear scale are determined. 
    fig : Figure
        Figure housing projection we are making colorbar for.
    caxis : Axis
        Axis where colorbar will be put.
    mappable : mappable, optional
        Mappable image to get colorbar data from. 
    cmap : string, optional
        Name of colormap to use.
    label : string, optional
        Name of label for colorbar. Will override default in config.PROP_INFO.
    limits : list, optional
        Shape (2) list with limits for colorbar. Will override default in config.PROP_INFO.
    log : bool, optional
        Set whether to be in log or linear scale. Will override default in config.PROP_INFO.

    """


    # Setup x axis
    if property not in config.PROP_INFO.keys() and (label is None and limits is None):
        print("%s is not a supported property for setup_proj_colorbar"%property)
        print("Either give label and limits to make your own or choose from supported properties.")
        print("Valid properties are:")
        print(config.PROP_INFO.keys())
        return
    label = config.get_prop_label(property)

    if mappable is None:
        if limits == None:
            limits = config.get_prop_limits(property)
        if config.get_prop_if_log(property) or log:
            norm = mpl.colors.LogNorm(vmin=limits[0], vmax=limits[1], clip=True)
        else:
            norm = mpl.colors.Normalize(vmin=limits[0], vmax=limits[1], clip=True)
        cbar = fig.colorbar(mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=caxis, orientation='horizontal')
    else: 
        cbar = fig.colorbar(mappable = mappable, cax=caxis, orientation='horizontal')
    cbar.ax.set_xlabel(label, fontsize=config.LARGE_FONT)
    cbar.ax.minorticks_on()
    cbar.ax.tick_params(axis='both',which='both',direction='in')
    cbar.ax.tick_params(axis='both', which='major', labelsize=config.SMALL_FONT, length=4*config.AXIS_BORDER_WIDTH, width=config.AXIS_BORDER_WIDTH)
    cbar.ax.tick_params(axis='both', which='minor', labelsize=config.SMALL_FONT, length=2*config.AXIS_BORDER_WIDTH, width=config.AXIS_BORDER_WIDTH/2)
    cbar.outline.set_linewidth(config.AXIS_BORDER_WIDTH)

    return cbar


def find_scale_bar(L):
    """
    Determines the appropriate scale bar to use in a projection given the size of the projection.

    Parameters
    ----------
    L : float
        Physical size of projection space in kpc

    Returns
    -------
    bar : float
        Size of scale bar
    label : string
        Label for scale bar

    """

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