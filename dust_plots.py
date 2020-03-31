import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
plt.switch_backend('agg')
import pickle
import os
from readsnap import readsnap
from astropy.table import Table
import gas_temperature as gas_temp
from tasz import *

# Set style of plots
plt.style.use('seaborn-talk')
# Set personal color cycle
colors = ["xkcd:blue", "xkcd:red", "xkcd:green", "xkcd:orange", "xkcd:violet", "xkcd:teal", "xkcd:brown"]
linestyles = ['-','--',':','-.']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)


UnitLength_in_cm            = 3.085678e21   # 1.0 kpc/h
UnitMass_in_g               = 1.989e43  	# 1.0e10 solar masses/h
UnitMass_in_Msolar			= UnitMass_in_g / 1.989E33
UnitVelocity_in_cm_per_s    = 1.0e5   	    # 1 km/sec
UnitTime_in_s 				= UnitLength_in_cm / UnitVelocity_in_cm_per_s # 0.978 Gyr/h
UnitTime_in_Gyr 			= UnitTime_in_s /1e9/365./24./3600.
UnitEnergy_per_Mass 		= np.power(UnitLength_in_cm, 2) / np.power(UnitTime_in_s, 2)
UnitDensity_in_cgs 			= UnitMass_in_g / np.power(UnitLength_in_cm, 3)
H_MASS 						= 1.67E-24 # grams

# Small number used to plot zeros on log graph
small_num = 1E-7

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



def phase_plot(G, H, mask=True, time=False, depletion=False, cosmological=True, nHmin=1E-6, nHmax=1E3, Tmin=1E1, Tmax=1E8, numbins=200, thecmap='hot', vmin=1E-8, vmax=1E-4, foutname='phase_plot.png'):
	"""
	Plots the temperate-density has phase

	Parameters
	----------
	G : dict
	    Snapshot gas data structure
	H : dict
		Snapshot header structure
	mask : np.array, optional
		Mask for which particles to use in plot, default mask=True means all values are used
	bin_nums: int
		Number of bins to use
	depletion: bool, optional
		Was the simulation run with the DEPLETION option

	Returns
	-------
	None
	"""
	if depletion:
		nH = np.log10(G['rho'][mask]*UnitDensity_in_cgs * ( 1. - (G['z'][:,0][mask]+G['z'][:,1]+G['dz'][:,0][mask])) / H_MASS)
	else:
		nH = np.log10(G['rho'][mask]*UnitDensity_in_cgs * ( 1. - (G['z'][:,0][mask]+G['z'][:,1][mask])) / H_MASS)
	T = np.log10(gas_temp.gas_temperature(G))
	T = T[mask]
	M = G['m'][mask]

	ax = plt.figure()
	plt.subplot(111, facecolor='xkcd:black')
	plt.hist2d(nH, T, range=np.log10([[nHmin,nHmax],[Tmin,Tmax]]), bins=numbins, cmap=plt.get_cmap(thecmap), norm=mpl.colors.LogNorm(), weights=M, vmin=vmin, vmax=vmax) 
	cbar = plt.colorbar()
	cbar.ax.set_ylabel(r'Mass in pixel $(10^{10} M_{\odot}/h)$')


	plt.xlabel(r'log $n_{H} ({\rm cm}^{-3})$') 
	plt.ylabel(r'log T (K)')
	plt.tight_layout()
	if time:
		if cosmological:
			z = H['redshift']
			ax.text(.85, .825, 'z = ' + '%.2g' % z, color="xkcd:white", fontsize = 16, ha = 'right')
		else:
			t = H['time']
			ax.text(.75, .9, 't = ' + '%2.1g Gyr' % t, color="xkcd:white", fontsize = 16, ha = 'right')	
	plt.savefig(foutname)
	plt.close()



def DZ_vs_dens(G, H, mask=True, bin_nums=30, time=False, depletion=False, cosmological=True, nHmin=1E-2, nHmax=1E3, foutname='DZ_vs_dens.png'):
	"""
	Plots the average dust-to-metals ratio (D/Z) vs density 

	Parameters
	----------
	G : dict
	    Snapshot gas data structure
	H : dict
		Snapshot header structure
	mask : np.array, optional
	    Mask for which particles to use in plot, default mask=True means all values are used
	bin_nums: int
		Number of bins to use
	time : bool, optional
		Print time in corner of plot (useful for movies)
	depletion: bool, optional
		Was the simulation run with the DEPLETION option

	Returns
	-------
	None
	"""

	# TODO : Replace standard deviation with WEIGHTED percentiles for the 16th and 84th precentile 
	#        to better plot error in log space


	if depletion:
		nH = G['rho'][mask]*UnitDensity_in_cgs * ( 1. - (G['z'][:,0][mask]+G['z'][:,1]+G['dz'][:,0][mask])) / H_MASS
	else:
		nH = G['rho'][mask]*UnitDensity_in_cgs * ( 1. - (G['z'][:,0][mask]+G['z'][:,1][mask])) / H_MASS
	D = G['dz'][mask]
	M = G['m'][mask]
	if depletion:
		DZ = G['dz'][:,0][mask]/(G['z'][:,0][mask]+G['dz'][:,0][mask])
	else:
		DZ = G['dz'][:,0][mask]/(G['z'][:,0][mask])

	# Make bins for nH 
	nH_bins = np.logspace(np.log10(nHmin),np.log10(nHmax),bin_nums)
	nH_vals = (nH_bins[1:] + nH_bins[:-1]) / 2.
	digitized = np.digitize(nH,nH_bins)
	mean_DZ = np.zeros(bin_nums - 1)
	# 16th and 84th percentiles
	std_DZ = np.zeros([bin_nums - 1,2])

	for i in range(1,len(nH_bins)):
		if len(nH[digitized==i])==0:
			mean_DZ[i-1] = np.nan
			std_DZ[i-1,0] = np.nan; std_DZ[i-1,1] = np.nan;
			continue
		else:
			weights = M[digitized == i]
			values = DZ[digitized == i]
			mean_DZ[i-1],std_DZ[i-1,0],std_DZ[i-1,1] = weighted_percentile(values, weights=weights)

	# Replace zeros with small values since we are taking the log of the values
	std_DZ[std_DZ == 0] = small_num

	ax=plt.figure()
	# Now take the log value of the binned statistics
	plt.plot(nH_vals, np.log10(mean_DZ))
	plt.fill_between(nH_vals, np.log10(std_DZ[:,0]), np.log10(std_DZ[:,1]),alpha = 0.4)
	plt.xlabel(r'$n_H (cm^{-3})$')
	plt.ylabel(r'Log D/Z Ratio')
	plt.ylim([-3.0,0.])
	plt.xlim([nHmin, nHmax])
	plt.xscale('log')
	if time:
		if cosmological:
			z = H['redshift']
			ax.text(.85, .825, 'z = ' + '%.2g' % z, color="xkcd:black", fontsize = 16, ha = 'right')
		else:
			t = H['time']
			ax.text(.85, .825, 't = ' + '%2.1g Gyr' % t, color="xkcd:black", fontsize = 16, ha = 'right')			
	plt.savefig(foutname)
	plt.close()




def DZ_vs_Z(G, H, mask=True, bin_nums=30, time=False, depletion=False, cosmological=True,Zmin=1E-4, Zmax=1e1, foutname='DZ_vs_Z.png'):
	"""
	Plots the average dust-to-metals ratio (D/Z) vs Z for masked particles

	Parameters
	----------
	G : dict
	    Snapshot gas data structure
	H : dict
		Snapshot header structure
	mask : np.array, optional
	    Mask for which particles to use in plot, default mask=True means all values are used
	bin_nums: int
		Number of bins to use
	time : bool, optional
		Print time in corner of plot (useful for movies)
	depletion: bool, optional
		Was the simulation run with the DEPLETION option

	Returns
	-------
	None
	"""
	solar_Z = 0.02
	if depletion:
		DZ = (G['dz'][:,0]/(G['z'][:,0]+G['dz'][:,0]))[mask]
		Z = (G['z'][:,0]+G['dz'][:,0])[mask]/solar_Z
	else:
		DZ = (G['dz'][:,0]/G['z'][:,0])[mask]
		Z = G['z'][:,0][mask]/solar_Z
	M = G['m'][mask]

	Z_bins = np.logspace(np.log10(Zmin),np.log10(Zmax),bin_nums)
	Z_vals = (Z_bins[1:] + Z_bins[:-1]) / 2.
	digitized = np.digitize(Z,Z_bins)
	mean_DZ = np.zeros(bin_nums-1)
	# 16th and 84th percentiles
	std_DZ = np.zeros([bin_nums - 1,2])


	for i in range(1,len(Z_bins)):
		if len(Z[digitized==i])==0:
			mean_DZ[i-1] = np.nan
			std_DZ[i-1,0] = np.nan; std_DZ[i-1,1] = np.nan;
			continue
		else:
			weights = M[digitized == i]
			values = DZ[digitized == i]
			mean_DZ[i-1],std_DZ[i-1,0],std_DZ[i-1,1] = weighted_percentile(values, weights=weights)
	# Replace zeros with small values since we are taking the log of the values
	std_DZ[std_DZ == 0] = small_num

	ax=plt.figure()
	plt.plot(Z_vals, np.log10(mean_DZ))
	plt.fill_between(Z_vals, np.log10(std_DZ[:,0]), np.log10(std_DZ[:,1]), alpha = 0.4)
	plt.xlabel(r'Metallicity $(Z_{\odot})$')
	plt.ylabel("Log D/Z Ratio")
	if time:
		if cosmological:
			z = H['redshift']
			ax.text(.85, .825, 'z = ' + '%.2g' % z, color="xkcd:black", fontsize = 16, ha = 'right')
		else:
			t = H['time']
			ax.text(.85, .825, 't = ' + '%2.1g Gyr' % t, color="xkcd:black", fontsize = 16, ha = 'right')		
	plt.xlim([Z_vals[0],Z_vals[-1]])
	plt.xscale('log')
	plt.ylim([-3.0,0.5])
	plt.savefig(foutname)
	plt.close()	




def DZ_vs_r(G, H, center, r_max, bin_nums=50, time=False, depletion=False, cosmological=True,  foutname='DZ_vs_r.png'):
	"""
	Plots the average dust-to-metals ratio (D/Z) vs radius given code values of center and virial radius

	Parameters
	----------
	G : dict
	    Snapshot gas data structure
	H : dict
		Snapshot header structure
	center: array
		3-D coordinate of center of circle
	r_max: double
		maximum radius
	bin_nums: int
		Number of bins to use
	time : bool, optional
		Print time in corner of plot (useful for movies)
	depletion: bool, optional
		Was the simulation run with the DEPLETION option
	foutname: str, optional
		Name of file to be saved

	Returns
	-------
	None
	"""	

	if depletion:
		DZ = G['dz'][:,0]/(G['z'][:,0]+G['dz'][:,0])
	else:
		DZ = G['dz'][:,0]/G['z'][:,0]
	
	r_bins = np.linspace(0, r_max, num=bin_nums)
	r_vals = (r_bins[1:] + r_bins[:-1]) / 2.
	mean_DZ = np.zeros(bin_nums-1)
	# 16th and 84th percentiles
	std_DZ = np.zeros([bin_nums - 1,2])

	coords = G['p']
	M = G['m']
	# Get only data of particles in sphere since those are the ones we care about
	# Also gives a nice speed-up
	in_sphere = np.power(coords[:,0] - center[0],2.) + np.power(coords[:,1] - center[1],2.) + np.power(coords[:,2] - center[2],2.) <= np.power(r_max,2.)
	M=M[in_sphere]
	DZ=DZ[in_sphere]
	coords=coords[in_sphere]

	for i in range(bin_nums-1):
		# find all coordinates within shell
		r_min = r_bins[i]; r_max = r_bins[i+1];
		in_shell = np.logical_and(np.power(coords[:,0] - center[0],2.) + np.power(coords[:,1] - center[1],2.) + np.power(coords[:,2] - center[2],2.) <= np.power(r_max,2.),
									np.power(coords[:,0] - center[0],2.) + np.power(coords[:,1] - center[1],2.) + np.power(coords[:,2] - center[2],2.) > np.power(r_min,2.))
		weights = M[in_shell]
		values = DZ[in_shell]
		if len(values) > 0:
			mean_DZ[i],std_DZ[i,0],std_DZ[i,1] = weighted_percentile(values, weights=weights)
		else:
			mean_DZ[i] = np.nan
			std_DZ[i,0] = np.nan; std_DZ[i,1] = np.nan;
		
	# Replace zeros with small values since we are taking the log of the values
	std_DZ[std_DZ == 0] = small_num

	ax=plt.figure()
	plt.plot(r_vals, np.log10(mean_DZ))
	plt.fill_between(r_vals, np.log10(std_DZ[:,0]), np.log10(std_DZ[:,1]), alpha = 0.4)
	plt.xlabel("Radius (kpc)")
	plt.ylabel("Log D/Z Ratio")
	if time:
		if cosmological:
			z = H['redshift']
			ax.text(.85, .825, 'z = ' + '%.2g' % z, color="xkcd:black", fontsize = 16, ha = 'right')
		else:
			t = H['time']
			ax.text(.85, .825, 't = ' + '%2.1g Gyr' % t, color="xkcd:black", fontsize = 16, ha = 'right')		
	plt.xlim([r_vals[0],r_vals[-1]])
	plt.ylim([-3.0,0.])
	plt.savefig(foutname)
	plt.close()




def DZ_vs_time(dataname='data.pickle', data_dir='data/', foutname='DZ_vs_time.png', time=True, cosmological=True):
	"""
	Plots the average dust-to-metals ratio (D/Z) vs time from precompiled data

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
	# Replace zeros in with small numbers
	std_DZ[std_DZ==0.] = small_num

	ax=plt.figure()
	plt.plot(time_data, np.log10(mean_DZ))
	plt.fill_between(time_data, np.log10(std_DZ[:,0]), np.log10(std_DZ[:,1]),alpha = 0.4)
	plt.ylabel(r'Log D/Z Ratio')
	plt.ylim([-3.0,0.])
	if time or not cosmological:
		plt.xlabel('t (Gyr)')
		plt.xscale('log')
	else:
		plt.xlabel('z')
		plt.gca().invert_xaxis()
		plt.xscale('log')

	plt.savefig(foutname)
	plt.close()

def all_data_vs_time(dataname='data.pickle', data_dir='data/', foutname='all_data_vs_time.png', time=False, cosmological=True):
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

	num_species = np.shape(data['spec'])[1]

	sfr = data['sfr'] 
	# Get mean and std, and make sure to set zero std to small number
	mean_DZ = data['DZ_ratio'][:,0]; std_DZ = data['DZ_ratio'][:,1:];
	mean_Z = data['metallicity'][:,0]/Z_solar; std_Z = data['metallicity'][:,1:]/Z_solar;
	mean_spec = data['spec_frac'][:,:,0]; std_spec = data['spec_frac'][:,:,1:];
	mean_source = data['source_frac'][:,:,0]; std_source = data['source_frac'][:,:,1:];
	mean_sil_to_C = data['sil_to_C_ratio'][:,0]; std_sil_to_C = data['sil_to_C_ratio'][:,1:];

	fig,axes = plt.subplots(2, 3, sharex='all', figsize=(24,12))

	axes[0,0].plot(time_data, np.log10(mean_DZ))
	axes[0,0].fill_between(time_data, np.log10(std_DZ[:,0]), np.log10(std_DZ[:,1]),alpha = 0.4)
	axes[0,0].set_ylabel(r'Log D/Z Ratio')
	axes[0,0].set_ylim([-3.0,0.])
	axes[0,0].set_xscale('log')

	axes[0,1].plot(time_data, sfr)
	axes[0,1].set_ylabel(r'SFR $(M_{\odot}/yr)$')
	axes[0,1].set_ylim([0.0001,0.1])
	axes[0,1].set_xscale('log')
	axes[0,1].set_yscale('log')


	axes[0,2].plot(time_data, np.log10(mean_Z))
	axes[0,2].fill_between(time_data, np.log10(std_Z[:,0]), np.log10(std_Z[:,1]),alpha = 0.4)
	axes[0,2].set_ylabel(r'Log Z $(Z_{\odot})$')
	axes[0,2].set_ylim([np.log10(1E-2),1])
	axes[0,2].set_xscale('log')

	for i in range(num_species):
		axes[1,0].plot(time_data, mean_spec[:,i], label=species_names[i])
		axes[1,0].fill_between(time_data, std_spec[:,i,0], std_spec[:,i,1],alpha = 0.4)
	axes[1,0].set_ylabel(r'Species Mass Fraction')
	axes[1,0].set_ylim([1E-3,1])
	axes[1,0].set_yscale('log')
	axes[1,0].set_xscale('log')
	axes[1,0].legend(loc=4)

	for i in range(4):
		axes[1,1].plot(time_data, mean_source[:,i], label=source_names[i])
		axes[1,1].fill_between(time_data, std_source[:,i,0], std_source[:,i,1],alpha = 0.4)
	axes[1,1].set_ylabel(r'Source Mass Fraction')
	axes[1,1].set_ylim([1E-2,1])
	axes[1,1].set_yscale('log')
	axes[1,1].set_xscale('log')
	axes[1,1].legend(loc=4)

	axes[1,2].plot(time_data, mean_sil_to_C)
	axes[1,2].fill_between(time_data, std_sil_to_C[:,0], std_sil_to_C[:,1],alpha = 0.4)
	axes[1,2].set_ylabel(r'Silicates to Carbon Ratio')
	axes[1,2].set_ylim([1E-2,1E1])
	axes[1,2].set_yscale('log')
	axes[1,2].set_xscale('log')


	if time or not cosmological:
		axes[1,0].set_xlabel('t (Gyr)')
		axes[1,1].set_xlabel('t (Gyr)')
		axes[1,2].set_xlabel('t (Gyr)')
	else:
		axes[1,0].set_xlabel('z')
		axes[1,1].set_xlabel('z')
		axes[1,2].set_xlabel('z')
		plt.gca().invert_xaxis()
		

	plt.tight_layout()

	plt.savefig(foutname)
	plt.close()

	
def compare_runs_vs_time(datanames=['data.pickle'], data_dir='data/', foutname='compare_runs_vs_time.png', labels=["fiducial"], time=False, cosmological=True):
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
                          linestyle=linestyles[i], label=labels[i])]

		with open(data_dir+dataname, 'rb') as handle:
			data = pickle.load(handle)
		
		if cosmological:
			if time:
				time_data = data['time']
			else:
				time_data = -1.+1./data['a_scale']
		else:
			time_data = data['time']

		num_species = np.shape(data['spec'])[1]

		# Get mean and std, and make sure to set zero std to small number
		mean_DZ = data['DZ_ratio'][:,0]; std_DZ = data['DZ_ratio'][:,1:];
		mean_spec = data['spec_frac'][:,:,0]; std_spec = data['spec_frac'][:,:,1:];
		mean_source = data['source_frac'][:,:,0]; std_source = data['source_frac'][:,:,1:];
		mean_sil_to_C = data['sil_to_C_ratio'][:,0]; std_sil_to_C = data['sil_to_C_ratio'][:,1:];
		
		axes[0,0].plot(time_data, np.log10(mean_DZ), color=colors[0], linestyle=linestyles[i])

		for j in range(num_species):
			axes[0,1].plot(time_data, mean_spec[:,j], label=species_names[j], color=colors[j], linestyle=linestyles[i])


		for j in range(4):
			axes[1,0].plot(time_data, mean_source[:,j], label=source_names[j], color=colors[j], linestyle=linestyles[i])

		axes[1,1].plot(time_data, mean_sil_to_C, color=colors[0], linestyle=linestyles[i])


	axes[0,0].set_ylabel(r'Log D/Z Ratio')
	axes[0,0].set_ylim([-3.0,0.])
	axes[0,0].set_xscale('log')

	axes[0,1].set_ylabel(r'Species Mass Fraction')
	axes[0,1].set_ylim([1E-3,1])
	axes[0,1].set_yscale('log')
	axes[0,1].set_xscale('log')
	spec_lines = []
	for i in range(num_species):
		spec_lines += [mlines.Line2D([], [], color=colors[i], label=species_names[i])]
	axes[0,1].legend(handles=spec_lines,loc=4)

	axes[1,0].set_ylabel(r'Source Mass Fraction')
	axes[1,0].set_ylim([1E-2,1])
	axes[1,0].set_yscale('log')
	axes[1,0].set_xscale('log')
	source_lines = []
	for i in range(4):
		source_lines += [mlines.Line2D([], [], color=colors[i], label=source_names[i])]
	axes[1,0].legend(handles=source_lines, loc=4)

	axes[1,1].set_ylabel(r'Silicates to Carbon Ratio')
	axes[1,1].set_ylim([1E-2,1E1])
	axes[1,1].set_yscale('log')
	axes[1,1].set_xscale('log')


	if time or not cosmological:
		axes[1,0].set_xlabel('t (Gyr)')
		axes[1,1].set_xlabel('t (Gyr)')
	else:
		axes[1,0].set_xlabel('z')
		axes[1,1].set_xlabel('z')
		plt.gca().invert_xaxis()

	# Create the legend for the different runs
	fig.legend(handles=lines,   		# The line objects
           loc="upper center",  		# Position of legend
           borderaxespad=0.1,   		# Small spacing around legend box
           ncol=len(lines),    			# Make the legend stretch horizontally across the plot
           fontsize=24,
           bbox_to_anchor=(0.5, .95)) 	# Pin the legend to just above the plots

	plt.savefig(foutname)
	plt.close()


def compile_dust_data(snap_dir, foutname='data.pickle', data_dir='data/', mask=False, halo_dir='', Rvir_frac = 1., r_max = None, overwrite=False, cosmological=True, startnum=0, endnum=600, implementation='species', depletion=False):
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
				if cosmological:
					halo_data = Table.read(halo_dir,format='ascii')
					# Convert to physical units
					xpos =  halo_data['col7'][num-1]*H['time']/H['hubble']
					ypos =  halo_data['col8'][num-1]*H['time']/H['hubble']
					zpos =  halo_data['col9'][num-1]*H['time']/H['hubble']
					rvir = halo_data['col13'][num-1]*H['time']/H['hubble']
					center = np.array([xpos,ypos,zpos])
					if r_max == None:
						print("Using AHF halo as spherical mask with radius of ",str(Rvir_frac)," * Rvir.")
						r_max = rvir*Rvir_frac
					else:
						print("Using AHF halo as spherical mask with radius of ",str(r_max)," kpc.")

				else:
					if r_max == None:
						print("Must give maximum radius r_max for noncosmological simulations!")
						return
					# Recenter coords at center of periodic box
					boxsize = H['boxsize']
					mask1 = coords > boxsize/2; mask2 = coords <= boxsize/2
					coords[mask1] -= boxsize/2; coords[mask2] += boxsize/2;
					center = np.average(coords, weights = G['m'], axis = 0)

				# Keep data for gas and star particles with coordinates within a sphere of radius r_max
				in_sphere = np.power(coords[:,0] - center[0],2.) + np.power(coords[:,1] - center[1],2.) + np.power(coords[:,2] - center[2],2.) <= np.power(r_max,2.)
				for key in G.keys():
					if key != 'k':
						G[key] = G[key][in_sphere]
				# Check if there are any star particles
				if S['k']!=-1:
					coords = S['p']
					in_sphere = np.power(coords[:,0] - center[0],2.) + np.power(coords[:,1] - center[1],2.) + np.power(coords[:,2] - center[2],2.) <= np.power(r_max,2.)
					S['age'] = S['age'][in_sphere]
					S['m'] = S['m'][in_sphere]

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
				spec_frac_vals = G['dz'][:,2]/G['dz'][:,0]
				is_num = np.logical_and(~np.isnan(spec_frac_vals), ~np.isinf(spec_frac_vals))
				spec_frac[i,0] = weighted_percentile(spec_frac_vals[is_num], weights =M[is_num])
				spec_frac[i,0][spec_frac[i,0]==0] = small_num

				spec_frac_vals = (G['dz'][:,4]+G['dz'][:,6]+G['dz'][:,7]+G['dz'][:,10])/G['dz'][:,0]
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