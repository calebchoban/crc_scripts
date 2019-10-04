import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
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
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["xkcd:blue", "xkcd:red", "xkcd:green", "xkcd:orange", "xkcd:violet", "xkcd:teal", "xkcd:brown"])


UnitLength_in_cm            = 3.085678e21   # 1.0 kpc/h
UnitMass_in_g               = 1.989e43  	# 1.0e10 solar masses/h
UnitMass_in_Msolar			= UnitMass_in_g / 1.989E33
UnitVelocity_in_cm_per_s    = 1.0e5   	    # 1 km/sec
UnitTime_in_s 				= UnitLength_in_cm / UnitVelocity_in_cm_per_s
UnitTime_in_Gyr 			= UnitTime_in_s /1e9/365./24./3600.
UnitEnergy_per_Mass 		= np.power(UnitLength_in_cm, 2) / np.power(UnitTime_in_s, 2)
UnitDensity_in_cgs 			= UnitMass_in_g / np.power(UnitLength_in_cm, 3)
H_MASS 						= 1.67E-24 # grams

def phase_plot(G, H, mask=True, time=False, depletion=False, nHmin=1E-6, nHmax=1E3, Tmin=1E1, Tmax=1E8, numbins=200, thecmap='hot', vmin=1E-8, vmax=1E-4, foutname='phase_plot.png'):
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
		z = H['redshift']
		ax.text(.75, .9, 'z = ' + '%.2g' % z, color="xkcd:white", fontsize = 16, ha='right')
	plt.savefig(foutname)



def DZ_vs_dens(G, H, mask=True, bin_nums=30, time=False, depletion=False, nHmin=1E-2, nHmax=1E3, foutname='DZ_vs_dens.png'):
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
			mean_DZ[i] = np.nan
			std_DZ[i,0] = np.nan; std_DZ[i,1] = np.nan;
			continue
		else:
			weights = M[digitized == i]
			values = DZ[digitized == i]
			mean_DZ[i] = np.average(values,weights=weights)
			#variance = np.dot(weights, (values - mean_DZ[i]) ** 2) / weights.sum()
			#std_DZ[i] = np.sqrt(variance)
			std_DZ[i,0],std_DZ[i,1] = np.percentile(values, [18,84])

	# Replace zeros with small values since we are taking the log of the values
	std_DZ[std_DZ == 0] = 1E-5

	ax=plt.figure()
	# Now take the log value of the binned statistics
	plt.plot(nH_vals, np.log10(mean_DZ))
	plt.fill_between(nH_vals, np.log10(std_DZ[:,0]), np.log10(std_DZ[:,1]),alpha = 0.4)
	plt.xlabel(r'$n_H (cm^{-3})$')
	plt.ylabel(r'Log D/Z Ratio')
	plt.ylim([-2.0,0.])
	plt.xlim([nHmin, nHmax])
	plt.xscale('log')
	if time:
		z = H['redshift']
		ax.text(.85, .825, 'z = ' + '%.2g' % z, color="xkcd:black", fontsize = 16, ha = 'right')
	plt.savefig(foutname)
	plt.close()



def DZ_vs_r(G, H, center, Rvir, bin_nums=50, time=False, depletion=False, Rvir_frac = 1., foutname='DZ_vs_r.png'):
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
	Rvir: double
		Virial radius of circle
	bin_nums: int
		Number of bins to use
	time : bool, optional
		Print time in corner of plot (useful for movies)
	depletion: bool, optional
		Was the simulation run with the DEPLETION option
	Rvir_frac: int, optional
		Max radius for plot as fraction of virial radius
	foutname: str, optional
		Name of file to be saved

	Returns
	-------
	None
	"""	

	# TODO : Replace standard deviation with WEIGHTED percentiles for the 16th and 84th precentile 
	#        to better plot error in log space


	if depletion:
		DZ = G['dz'][:,0]/(G['z'][:,0]+G['dz'][:,0])
	else:
		DZ = G['dz'][:,0]/G['z'][:,0]

	r_bins = np.linspace(0, Rvir*Rvir_frac, num=bin_nums)
	r_vals = (r_bins[1:] + r_bins[:-1]) / 2.
	mean_DZ = np.zeros(bin_nums-1)
	# 16th and 84th percentiles
	std_DZ = np.zeros([bin_nums - 1,2])

	coords = G['p']
	M = G['m']
	# Get only data of particles in sphere since those are the ones we care about
	# Also gives a nice speed-up
	in_sphere = np.power(coords[:,0] - center[0],2.) + np.power(coords[:,1] - center[1],2.) + np.power(coords[:,2] - center[2],2.) <= np.power(Rvir*Rvir_frac,2.)
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
			mean_DZ[i] = np.average(values,weights=weights)
			std_DZ[i,0],std_DZ[i,1] = np.percentile(values, [18,84])
		else:
			mean_DZ[i] = np.nan
			std_DZ[i,0] = np.nan; std_DZ[i,1] = np.nan;
		
		#std_DZ[i] = np.sqrt(np.dot(weights, (values - mean_DZ[i]) ** 2) / weights.sum())
		

	# Convert coordinates to physical units
	r_vals *= H['time'] * H['hubble']  # kpc

	# Replace zeros with small values since we are taking the log of the values
	std_DZ[std_DZ == 0] = 1E-5

	ax=plt.figure()
	plt.plot(r_vals, np.log10(mean_DZ))
	plt.fill_between(r_vals, np.log10(std_DZ[:,0]), np.log10(std_DZ[:,1]), alpha = 0.4)
	plt.xlabel("Radius (kpc)")
	plt.ylabel("Log D/Z Ratio")
	if time:
		z = H['redshift']
		ax.text(.85, .825, 'z = ' + '%.2g' % z, color="xkcd:black", fontsize = 16, ha = 'right')
	plt.xlim([r_vals[0],r_vals[-1]])
	plt.ylim([-2.0,0.])
	plt.savefig(foutname)
	plt.close()




def DZ_vs_time(dataname='data.pickle', data_dir='data/', foutname='DZ_vs_time.png', time=True, omeganot=0.272, h=0.72):
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
	
	redshift = data['redshift']
	if time:
		time_data = tfora(1./(1.+redshift), omeganot, h)
	else:
		time_data = redshift
	mean_DZ = data['DZ_ratio'][:,0]
	std_DZ = data['DZ_ratio'][:,1:]
	# Replace zeros in with small numbers
	std_DZ[std_DZ==0.] = 1E-5

	ax=plt.figure()
	plt.plot(time_data, np.log10(mean_DZ))
	plt.fill_between(time_data, np.log10(std_DZ[:,0]), np.log10(std_DZ[:,1]),alpha = 0.4)
	plt.ylabel(r'Log D/Z Ratio')
	plt.ylim([-2.0,0.])
	if time:
		plt.xlabel('t (Gyr)')
		plt.xscale('log')
	else:
		plt.xlabel('z')
		plt.gca().invert_xaxis()
		plt.xscale('log')

	plt.savefig(foutname)
	plt.close()

def all_data_vs_time(dataname='data.pickle', data_dir='data/', foutname='all_data_vs_time.png', time=False, omeganot=0.272, h=0.72):
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

	species_names = ['Carbon','Silicates','SiC','Iron']
	source_names = ['Accretion','SNe Ia', 'SNe II', 'AGB']

	with open(data_dir+dataname, 'rb') as handle:
		data = pickle.load(handle)
	
	redshift = data['redshift']
	if time:
		time_data = tfora(1./(1.+redshift), omeganot, h)
	else:
		time_data = redshift
	sfr = data['sfr'] 
	mean_DZ = data['DZ_ratio'][:,0]; std_DZ = data['DZ_ratio'][:,1:];
	mean_Z = data['metallicity'][:,0]; std_Z = data['metallicity'][:,1:];
	mean_spec = data['spec_frac'][:,:,0]; std_spec = data['spec_frac'][:,:,1:];
	mean_source = data['source_frac'][:,:,0]; std_source = data['source_frac'][:,:,1:];
	mean_sil_to_C = data['sil_to_C_ratio'][:,0]; std_sil_to_C = data['sil_to_C_ratio'][:,1:];

	print data['metallicity']

	fig,axes = plt.subplots(2, 3, sharex='all', figsize=(24,12))

	axes[0,0].plot(time_data, np.log10(mean_DZ))
	axes[0,0].fill_between(time_data, np.log10(std_DZ[:,0]), np.log10(std_DZ[:,1]),alpha = 0.4)
	axes[0,0].set_ylabel(r'Log D/Z Ratio')
	axes[0,0].set_ylim([-2.0,0.])
	axes[0,0].set_xscale('log')

	axes[0,1].plot(time_data, sfr)
	axes[0,1].set_ylabel(r'SFR $(M_{\odot}/yr)$')
	axes[0,1].set_ylim([0.0001,0.1])
	axes[0,1].set_xscale('log')
	axes[0,1].set_yscale('log')


	axes[0,2].plot(time_data, np.log10(mean_Z))
	axes[0,2].fill_between(time_data, np.log10(std_Z[:,0]), np.log10(std_Z[:,1]),alpha = 0.4)
	axes[0,2].set_ylabel(r'Log Z')
	axes[0,2].set_ylim([np.log10(2E-6),-2.])
	axes[0,2].set_xscale('log')

	print mean_spec
	for i in range(4):
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


	if time:
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

	



def compile_dust_data(snap_dir, foutname='data.pickle', data_dir='data/', mask=False, halo_dir='', Rvir_frac = 1., overwrite=False, startnum=0, endnum=600, implementation='species', depletion=False):
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
		    print "Directory " + data_dir +  " Created " 
		except:
		    print "Directory " + data_dir +  " already exists"

		print "Fetching data now..."
		length = endnum-startnum+1
		# Most data comes with mean of values and 16th and 84th percentile
		DZ_ratio = np.zeros([length,3])
		sil_to_C_ratio = np.zeros([length,3])
		sfr = np.zeros(length)
		metallicity = np.zeros([length,3])
		redshift = np.zeros(length)
		source_frac = np.zeros((length,4,3))
		spec_frac = np.zeros((length,4,3))

		# Go through each of the snapshots and get the data
		for i, num in enumerate(range(startnum, endnum+1)):
			print num
			G = readsnap(snap_dir, num, 0)
			H = readsnap(snap_dir, num, 0, header_only=True)
			S = readsnap(snap_dir, num, 4)

			if mask:
				print "Using AHF halo as spherical mask with radius of " + str(Rvir_frac) + " * Rvir."
				halo_data = Table.read(halo_dir,format='ascii')
				xpos =  halo_data['col7'][num-1]
				ypos =  halo_data['col8'][num-1]
				zpos =  halo_data['col9'][num-1]
				rvir = halo_data['col13'][num-1]*Rvir_frac
				center = np.array([xpos,ypos,zpos])

				# Keep data for particles with coordinates within a sphere of radius Rvir
				coords = G['p']
				in_sphere = np.power(coords[:,0] - center[0],2.) + np.power(coords[:,1] - center[1],2.) + np.power(coords[:,2] - center[2],2.) <= np.power(rvir,2.)
				for key in G.keys():
					if key != 'k':
						G[key] = G[key][in_sphere]
				coords = S['p']
				in_sphere = np.power(coords[:,0] - center[0],2.) + np.power(coords[:,1] - center[1],2.) + np.power(coords[:,2] - center[2],2.) <= np.power(rvir,2.)
				S['age'] = S['age'][in_sphere]
				S['m'] = S['m'][in_sphere]

			M = G['m']
			redshift[i] = H['redshift']
			h = H['hubble']

			if depletion:
				metallicity[i,0] = np.average(G['z'][:,0]+G['dz'][:,0], weights=M)
				metallicity[i,1],metallicity[i,2] = np.percentile(G['z'][:,0]+G['dz'][:,0], [16,84])
			else:
				metallicity[i,0] = np.average(G['z'][:,0], weights=M)
				metallicity[i,1],metallicity[i,2] = np.percentile(G['z'][:,0], [16,84])

			source_frac[i,:,0] = np.average(G['dzs'], axis = 0, weights=M)
			source_frac[i,:,1],source_frac[i,:,2] = np.percentile(G['dzs'], [16,84], axis=0)
			if implementation == 'species':
				# Need to mask nan values for average to work
				spec_frac_vals = G['spec']/G['dz'][:,0]
				not_nan = ~np.isnan(spec_frac_vals)
				spec_frac[i,:,0] = np.average(spec_frac_vals[not_nan], axis = 0, weights=M[not_nan])
				spec_frac[i,:,1],spec_frac[i,:,2] = np.percentile(spec_frac_vals[not_nan], [16,84], axis=0)
				# Need to mask nan values for average to work
				sil_to_C_vals = G['spec'][:,1]/G['spec'][:,0]
				not_nan = ~np.isnan(sil_to_C_vals)
				sil_to_C_ratio[i,0] = np.average(sil_to_C_vals[not_nan], weights=M[not_nan])
				sil_to_C_ratio[i,1],sil_to_C_ratio[i,2] = np.percentile(sil_to_C_vals[not_nan], [16,84], axis=0)
			elif implementation == 'elemental':
				# Need to mask nan values for average to work
				spec_frac_vals = G['dz'][:,2]/G['dz'][:,0]
				not_nan = ~np.isnan(spec_frac_vals)
				spec_frac[i,0,0] = np.average(spec_frac_vals[not_nan], weights=M[not_nan])
				spec_frac[i,0,1], spec_frac[i,0,2] = np.percentile(spec_frac_vals[not_nan], [16,84])
				spec_frac_vals = (G['dz'][:,4]+G['dz'][:,6]+G['dz'][:,7]+G['dz'][:,10])/G['dz'][:,0]
				not_nan = ~np.isnan(spec_frac_vals)
				spec_frac[i,1,0] = np.average(spec_frac_vals[not_nan], weights=M[not_nan])
				spec_frac[i,1,1],spec_frac[i,1,2] = np.percentile(spec_frac_vals[not_nan], [16,84])
				spec_frac[i,2] = 0.
				spec_frac[i,2,0]=0.; spec_frac[i,2,2]=0.;
				spec_frac[i,3] = 0.
				spec_frac[i,3,0]=0.; spec_frac[i,3,2]=0.;

				sil_to_C_vals = (G['dz'][:,4]+G['dz'][:,6]+G['dz'][:,7]+G['dz'][:,10])/G['dz'][:,2]
				not_nan = ~np.isnan(sil_to_C_vals)
				sil_to_C_ratio[i,0] = np.average(sil_to_C_vals[not_nan], weights=M[not_nan])
				sil_to_C_ratio[i,1],sil_to_C_ratio[i,2] = np.percentile(sil_to_C_vals[not_nan], [16,84])

			if depletion:
				DZ_vals = G['dz'][:,0]/(G['z'][:,0]+G['dz'][:,0])
			else:
				DZ_vals = G['dz'][:,0]/G['z'][:,0]
			DZ_ratio[i,0] = np.average(DZ_vals, weights=M)
			DZ_ratio[i,1],DZ_ratio[i,2] = np.percentile(DZ_vals, [16,84])
			# Calculate SFR as all stars born within the last 20 Myrs
			formation_time = tfora(S['age'], H['omega0'], h)
			current_time = tfora(H['time'], H['omega0'], h)
			time_interval = 100E-3 # 100 Myr
			new_stars = (current_time - formation_time) < time_interval
			sfr[i] = np.sum(S['m'][new_stars]) * UnitMass_in_Msolar * h / (time_interval*1E9)   # Msun/yr

		data = {'redshift':redshift,'DZ_ratio':DZ_ratio,'sil_to_C_ratio':sil_to_C_ratio,'metallicity':metallicity,'source_frac':source_frac,'spec_frac':spec_frac,'sfr':sfr}
		with open(data_dir+foutname, 'wb') as handle:
			pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)