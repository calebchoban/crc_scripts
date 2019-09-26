import numpy as np
import matplotlib.pyplot as plt





def DZ_vs_dens(P, H, mask=[], time=False):
	"""
	Plots the average dust-to-metals ratio (D/Z) vs density 

	Parameters
	----------
	P : dict
	    Snapshot data structure
	H : dict
		Snapshot header structure
	mask : np.array, optional
	    Mask for which particles to use in plot
	time : bool, optional
		Print time in corner of plot (useful for movies)

	Returns
	-------
	None
	"""


def DZ_vs_r(P, H, center, radius, mask=[], time=False):
	"""
	Plots the average dust-to-metals ratio (D/Z) vs radius in a given circle

	Parameters
	----------
	P : dict
	    Snapshot data structure
	H : dict
		Snapshot header structure
	center: array
		3-D coordinate of center of circle
	radius: double
		Radius of circle in kpc
	mask : np.array, optional
	    Mask for which particles to use in plot
	time : bool, optional
		Print time in corner of plot (useful for movies)

	Returns
	-------
	None
	"""	

def DZ_vs_time(redshift_range):
	"""
	Plots the average dust-to-metals ratio (D/Z) vs time from precompiled data

	Parameters
	----------
	redshift_range : array
		Range of redshift for plot 

	Returns
	-------
	None
	"""

def compile_dust_data(snap_dir):
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