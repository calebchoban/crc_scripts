import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import os
from readsnap import readsnap
from astropy.table import Table
import gas_temperature as gas_temp
# Set style of plots
plt.style.use('seaborn-talk')
# Set personal color cycle
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["xkcd:blue", "xkcd:red", "xkcd:green", "xkcd:orange", "xkcd:violet", "xkcd:teal", "xkcd:brown"])


UnitLength_in_cm            = 3.085678e21   # 1.0 kpc/h
UnitMass_in_g               = 1.989e43  	# 1.0e10 solar masses/h
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
	digitized = np.digitize(nH,nH_bins)
	mean_DZ = np.zeros(len(nH_bins - 1))
	std_DZ = np.zeros(len(nH_bins - 1))

	for i in range(1,len(nH_bins)):
		if len(nH[digitized==i])==0:
			mean_DZ[i] = np.nan
			std_DZ[i] = np.nan
			continue
		else:
			weights = M[digitized == i]
			values = DZ[digitized == i]
			mean_DZ[i] = np.average(values,weights=weights)
			variance = np.dot(weights, (values - mean_DZ[i]) ** 2) / weights.sum()
			std_DZ[i] = np.sqrt(variance)

	ax=plt.figure()
	# Now take the log value of the binned statistics
	plt.plot(nH_bins[1:], np.log10(mean_DZ[1:]))
	plt.fill_between(nH_bins[1:], np.log10(mean_DZ[1:] - std_DZ[1:]), np.log10(mean_DZ[1:] + std_DZ[1:]),alpha = 0.4)
	plt.xlabel(r'$n_H (cm^{-3})$')
	plt.ylabel(r'Log D/Z Ratio')
	plt.ylim([-2.5,0.])
	plt.xlim([nHmin, nHmax])
	plt.xscale('log')
	if time:
		z = H['redshift']
		ax.text(.85, .825, 'z = ' + '%.2g' % z, color="xkcd:black", fontsize = 16, ha = 'right')
	plt.savefig(foutname)
	plt.close()



def DZ_vs_r(G, H, center, radius, mask=[], time=False):
	"""
	Plots the average dust-to-metals ratio (D/Z) vs radius in a given circle

	Parameters
	----------
	G : dict
	    Snapshot gas data structure
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



def compile_dust_data(snap_dir, foutname='data.pickle', data_dir='data/', mask=False, halo_dir='', Rvir_frac = 1., overwrite=False, startnum=0, endnum=600, implementation='species'):
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
		DZ_ratio = np.zeros(length)
		sil_to_C_ratio = np.zeros(length)
		sfr = np.zeros(length)
		metallicity = np.zeros(length)
		redshift = np.zeros(length)
		source_frac = np.zeros((length,4))
		spec_frac = np.zeros((length,4))

		total_star_mass = 0
		prev_time = 0
		# Go through each of the snapshots and get the data
		for i, num in enumerate(range(startnum, endnum+1)):
			print num
			G = readsnap(snap_dir, num, 0)
			H = readsnap(snap_dir, num, 0, header_only=True)
			S = readsnap(snap_dir, num, 4)

			if mask:
				print "Using AHF halo as spherical mask with radius of " + str(Rvir_frac) + " * Rvir."
				halo_data = Table.read(halo_dir,format='ascii')
				xpos =  halo_data['col7'][num]
				ypos =  halo_data['col8'][num]
				zpos =  halo_data['col9'][num]
				rvir = halo_data['col13'][num]*Rvir_frac
				center = np.array([xpos,ypos,zpos])

				coords = G['p']
				# coordinates within a sphere of radius Rvir
				in_sphere = np.power(coords[:,0] - center[0],2.) + np.power(coords[:,1] - center[1],2.) + np.power(coords[:,2] - center[2],2.) <= np.power(rvir,2.)
				for key in G.keys():
					if key != 'k':
						G[key] = G[key][in_sphere]

			M = G['m']
			redshift[i] = H['redshift']
			metallicity[i] = np.average(G['z'][:,0], weights=M)
			source_frac[i] = np.average(G['dzs'], axis = 0, weights=M)
			if implementation == "species":
				spec_frac[i] = np.average(G['spec'], axis = 0, weights=M)
				sil_to_C_ratio = np.average(G['spec'][:,1]/G['spec'][:,0], weights=M)
			elif implementation == 'elemental':
				spec_frac[i,0] = np.average(G['dz'][:,2], weights=M)
				spec_frac[i,1] = np.average(G['dz'][:,4]+G['dz'][:,6]+G['dz'][:,7]+G['dz'][:,10], weights=M)
				spec_frac[i,2] = 0.
				spec_frac[i,3] = 0.
				sil_to_C_ratio = np.average((G['dz'][:,4]+G['dz'][:,6]+G['dz'][:,7]+G['dz'][:,10])/G['dz'][:,2], weights=M)

			DZ_ratio[i] = np.average(G['dz'][:,0]/G['z'][:,0], weights=M)
			# TODO : Fix sfr so it only counts all new stars born within the last 20 Myr
			sfr[i] = (np.sum(S['m']) - total_star_mass) / (H['time']-prev_time)
			total_star_mass = S['m']

		data = {'redshift':redshift,'DZ_ratio':DZ_ratio,'sil_to_C_ratio':sil_to_C_ratio,'metallicity':metallicity,'source_frac':source_frac,'spec_frac':spec_frac,'sfr':sfr}

		# TODO : Switch to saving data in hdf5 using h5py
		with open(data_dir+foutname, "w") as write_file:
		    pickle.dump(data, write_file)