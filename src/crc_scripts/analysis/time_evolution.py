import os
import numpy as np
import pickle
from ..math_utils import weighted_percentile, quick_lookback_time
from ..io.snapshot import Snapshot

# This is a class that compiles the evolution data of a Snapshot/Halo/Disk
# over a specified time for a simulation

class Dust_Evo(object):

	def __init__(self, sdir, snap_lims, cosmological=1, totals=None, medians=None,
				 median_subsamples=None, star_totals=None, dirc='./', name_prefix=''):
		# Set property totals and property medians you want from each snapshot. Also set median masks which will
		# take subsampled medians based on gas properties
		if totals is None:
			self.totals = ['M_gas','M_H2','M_gas_neutral','M_dust','M_metals','M_sil','M_carb',
							'M_SiC','M_iron','M_ORes','M_SNeIa_dust','M_SNeII_dust','M_AGB_dust','M_acc_dust',
						   'f_cold', 'f_warm','f_hot']
		else:
			self.totals = totals
		if medians is None:
			self.medians = ['D/Z','Z','dz_acc','dz_SNeIa','dz_SNeII','dz_AGB','dz_sil','dz_carb',
							'dz_SiC','dz_iron','dz_ORes','CinCO','fdense','fH2',
							'Z_C','Z_O','Z_Mg','Z_Si','Z_Fe',
							'C/H','C/H_gas','O/H','O/H_gas','Mg/H','Mg/H_gas','Si/H','Si/H_gas','Fe/H','Fe/H_gas']
		else:
			self.medians = medians
		if median_subsamples is None:
			self.median_subsamples = ['all','cold','warm','hot','neutral','molecular']
		else:
			self.median_subsamples = median_subsamples
		if star_totals is None:
			self.star_totals = ['M_star','sfr','sfr_10Myr','sfr_100Myr']
		else:
			self.star_totals = star_totals

		self.sdir = sdir
		self.snap_lims = snap_lims
		self.num_snaps = (snap_lims[1]+1)-snap_lims[0]
		self.cosmological = cosmological
		self.hubble=None
		self.omega=None
		self.dirc = dirc
		# Determines if you want to look at the Snapshot/Halo/Disk
		self.setHalo=False
		self.setDisk=False

		# Get the basename of the directory the snapshots are stored in
		self.basename = os.path.basename(os.path.dirname(os.path.normpath(sdir)))
		# Name for saving object
		self.name = 'dust_evo_'+name_prefix+'_'+self.basename+'_snaps'

		# Check if object file has already been created if so load that first instead of creating a new one
		if not os.path.isfile(self.dirc + self.name + '.pickle'):
			self.dust_evo_data = Dust_Evo_Data(sdir, snap_lims, self.totals, self.medians, self.median_subsamples,
											   self.star_totals, cosmological=cosmological)
		else:
			with open(self.dirc+self.name + '.pickle', 'rb') as handle:
				self.dust_evo_data = pickle.load(handle)
			print("Dust_Evo data already exists so loading that first....")

			# If previously saved data set has different snap limits need to update limits before loading
			if not np.array_equiv(self.snap_lims, self.dust_evo_data.snap_lims):
				if self.snap_lims[0]<self.dust_evo_data.snap_lims[0] or self.snap_lims[0]>self.dust_evo_data.snap_lims[1]:
					print('Given snapshot limits differ from previously saved data structure')
					print("Previous limits:",self.dust_evo_data.snap_lims)
					print("New limits:",self.snap_lims)
					print("Adding new snapshots now......")
					self.dust_evo_data.change_snap_lims(self.snap_lims)
					if self.dust_evo_data.haloIDs is not None:
						print("Since the initial limits were given an array of halo IDs make sure you call setHalo() again "
							  "with an array of halo IDs that match the new limits.")
						self.dust_evo_data.setHalo = False
						self.setHalo = False

			new_props = False
			# If previously saved data is missing properties or subsamples then update property list
			missing_props = list(set(self.totals) - set(self.dust_evo_data.total_props))
			if len(missing_props) > 0:
				print('New Totals:',missing_props)
				self.dust_evo_data.add_total_property(missing_props)
				new_props=True
			missing_props = list(set(self.medians) - set(list(list(self.dust_evo_data.median_props))))
			if len(missing_props) > 0:
				print('New Medians:',missing_props)
				self.dust_evo_data.add_median_property(missing_props)
				new_props=True
			missing_props = list(set(self.median_subsamples) - set(self.dust_evo_data.subsamples))
			if len(missing_props) > 0:
				print('New Subsamples:',missing_props)
				self.dust_evo_data.add_subsample(missing_props)
				new_props=True
			if new_props:
				print("Some properties are missing from the previously saved data structure. Adding them now...")

		self.k = 0

		return





	# Set data to only include particles in specified halo
	def set_halo(self, **kwargs):
		self.dust_evo_data.set_halo(**kwargs)
		self.setHalo=True

		return


	# Set data to only include particles in specified disk
	def set_disk(self, **kwargs):
		self.dust_evo_data.set_disk(**kwargs)
		self.setDisk=True

		return


	def load(self, increment=5):

		if self.k: return

		if increment < 1:
			increment = 1

		while not self.dust_evo_data.all_snaps_loaded:
			ok = self.dust_evo_data.load(increment=increment)
			if not ok:
				print("Ran into an error when attempting to load data....")
				return
			self.save()

		self.hubble=self.dust_evo_data.hubble
		self.omega=self.dust_evo_data.omega

		self.k = 1
		return


	def save(self):

		# First create directory if needed
		if not os.path.isdir(self.dirc):
			os.mkdir(self.dirc)
			print("Directory " + self.dirc +  " Created ")

		with open(self.dirc+self.name+'.pickle', 'wb') as handle:
			pickle.dump(self.dust_evo_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

		return

	# Returns the specified data or derived data field if possible
	def get_data(self, data_name, subsample='all', statistic='median'):

		if statistic not in ['total','median']:
			print("Warning: Statistic must be either total or median.")
			return None

		if not self.dust_evo_data.all_snaps_loaded:
			print("Warning: Not all snapshots have been loaded! All unloaded values will be zero!")

		if data_name in self.totals:
			data = self.dust_evo_data.total_data[data_name]
		elif data_name in self.medians:
			if subsample in self.median_subsamples:
				data = self.dust_evo_data.median_data[subsample][data_name]
			else:
				print('No data for given median subsample %s is available for %s.'%(subsample,data_name))
				return None
		elif data_name in self.star_totals:
			data = self.dust_evo_data.star_total_data[data_name]
		elif data_name == 'time':
			data = self.dust_evo_data.time
		elif data_name == 'redshift' and self.dust_evo_data.cosmological:
			data = self.dust_evo_data.redshift
		elif 'source' in data_name:
			if 'total' in statistic:
				if 'source_acc' in data_name:
					data = self.dust_evo_data.total_data['M_acc_dust']/self.dust_evo_data.total_data['M_dust']
				elif 'source_SNeIa' in data_name:
					data = self.dust_evo_data.total_data['M_SNeIa_dust']/self.dust_evo_data.total_data['M_dust']
				elif 'source_SNeII' in data_name:
					data = self.dust_evo_data.total_data['M_SNeII_dust']/self.dust_evo_data.total_data['M_dust']
				elif 'source_AGB' in data_name:
					data = self.dust_evo_data.total_data['M_AGB_dust']/self.dust_evo_data.total_data['M_dust']
				else:
					print(data_name," is not in the dataset.")
					return None
				data[np.isnan(data)] = 0
			elif 'median' in statistic:
				if 'source_acc' in data_name:
					data = self.dust_evo_data.median_data[subsample]['dz_acc']
				elif 'source_SNeIa' in data_name:
					data = self.dust_evo_data.median_data[subsample]['dz_SNeIa']
				elif 'source_SNeII' in data_name:
					data = self.dust_evo_data.median_data[subsample]['dz_SNeII']
				elif 'source_AGB' in data_name:
					data = self.dust_evo_data.median_data[subsample]['dz_AGB']
				else:
					print(data_name," is not in the dataset.")
					return None
			else:
				print(data_name," is not in the dataset.")
				return None
		elif 'spec' in data_name:
			if 'total' in statistic:
				if 'spec_sil' in data_name:
					data = self.dust_evo_data.total_data['M_sil']/self.dust_evo_data.total_data['M_dust']
				elif 'spec_carb' in data_name:
					data = self.dust_evo_data.total_data['M_carb']/self.dust_evo_data.total_data['M_dust']
				elif 'spec_SiC' in data_name:
					data = self.dust_evo_data.total_data['M_SiC']/self.dust_evo_data.total_data['M_dust']
				elif 'spec_iron' in data_name and 'spec_ironIncl' not in data_name:
					data = self.dust_evo_data.total_data['M_iron']/self.dust_evo_data.total_data['M_dust']
				elif 'spec_ORes' in data_name:
					data = self.dust_evo_data.total_data['M_ORes']/self.dust_evo_data.total_data['M_dust']
				else:
					print(data_name," is not in the dataset.")
					return None
				data[np.isnan(data)] = 0
			elif 'median' in statistic:
				if 'spec_sil' in data_name:
					data = self.dust_evo_data.median_data[subsample]['dz_sil']
				elif 'spec_carb' in data_name:
					data = self.dust_evo_data.median_data[subsample]['dz_carb']
				elif 'spec_SiC' in data_name:
					data = self.dust_evo_data.median_data[subsample]['dz_SiC']
				elif 'spec_iron' in data_name and 'spec_ironIncl' not in data_name:
					data = self.dust_evo_data.median_data[subsample]['dz_iron']
				elif 'spec_ORes' in data_name:
					data = self.dust_evo_data.median_data[subsample]['dz_ORes']
				elif 'spec_sil+' in data_name:
					data = self.dust_evo_data.median_data[subsample]['dz_sil']+self.dust_evo_data.median_data[subsample]['dz_SiC']+\
							self.dust_evo_data.median_data[subsample]['dz_iron']+self.dust_evo_data.median_data[subsample]['dz_ORes']
				else:
					print(data_name," is not in the dataset.")
					return None
			else:
				print(data_name," is not in the dataset.")
				return None
		elif data_name in ['C/H_dust','O/H_dust','Mg/H_dust','Si/H_dust','Fe/H_dust']:
			base_name = data_name.split('_')[0]
			total = self.dust_evo_data.median_data[subsample][base_name]
			gas = self.dust_evo_data.median_data[subsample][base_name+'_gas']
			data = 12 + np.log10(np.power(10,total-12) - np.power(10,gas-12))
		elif data_name in ['Z_C_dust','Z_O_dust','Z_Mg_dust','Z_Si_dust','Z_Fe_dust']:
			base_name = data_name.split('_')[0] + '_' + data_name.split('_')[1]
			total = self.dust_evo_data.median_data[subsample][base_name]
			gas = self.dust_evo_data.median_data[subsample][base_name+'_gas']
			data = total-gas
		elif 'Si/C' in data_name:
			if 'total' in statistic:
				data = (self.dust_evo_data.total_data['M_sil']+self.dust_evo_data.total_data['M_SiC']+ \
						self.dust_evo_data.total_data['M_iron']+self.dust_evo_data.total_data['M_ORes'])/self.dust_evo_data.total_data['M_carb']
			elif 'median' in statistic:
				data = (self.dust_evo_data.median_data[subsample]['dz_sil']+self.dust_evo_data.median_data[subsample]['dz_SiC']+\
							self.dust_evo_data.median_data[subsample]['dz_iron']+self.dust_evo_data.median_data[subsample]['dz_ORes']) \
							/ self.dust_evo_data.median_data[subsample]['dz_carb']
			else:
				print(data_name, " is not in the dataset with given statistic.")
				return None
		else:
			print(data_name," is not in the dataset.")
			return None

		return data.copy()



class Dust_Evo_Data(object):

	def __init__(self, sdir, snap_lims, totals, medians, median_subsamples, star_totals, cosmological=1):
		self.sdir = sdir
		self.snap_lims = snap_lims
		self.num_snaps = int((snap_lims[1]+1)-snap_lims[0])
		self.snap_loaded = np.zeros(self.num_snaps,dtype=bool)
		# Load first snap to get cosmological parameters
		self.cosmological = cosmological
		sp = Snapshot(self.sdir, snap_lims[0], cosmological=self.cosmological)
		self.hubble = sp.hubble
		self.omega = sp.omega
		self.time = np.zeros(self.num_snaps)
		if self.cosmological:
			self.redshift = np.zeros(self.num_snaps)
			self.scale_factor = np.zeros(self.num_snaps)


		# Populate the data dictionaries
		self.total_data = {key : np.zeros(self.num_snaps) for key in totals}
		self.star_total_data = {key : np.zeros(self.num_snaps) for key in star_totals}
		# Populate the gas property fields corresponding with each mask
		self.median_data = {sub_key : {key : np.zeros(self.num_snaps) for key in medians} for sub_key in median_subsamples}

		self.setHalo=False
		self.setDisk=False
		self.load_kwargs = {}
		self.set_kwargs = {}
		self.use_halfmass_radius = False
		# Used when dominate halo changes during sim
		self.haloIDs = None

		self.all_snaps_loaded=False

		self.total_props = totals
		self.star_total_props = star_totals
		self.median_props = medians
		self.subsamples = median_subsamples

		# Used to track what properties are loaded and used to selectively load certain properties if they are added later
		self.totals_loaded = np.zeros(len(totals))
		self.star_totals_loaded = np.zeros(len(star_totals))
		self.median_loaded = np.zeros(len(medians))
		self.subsamples_loaded = np.zeros(len(median_subsamples))


		return


	def change_snap_lims(self, snap_lims):
		prepend_snaps = 0
		append_snaps = 0
		if snap_lims[0]<self.snap_lims[0]:
			prepend_snaps = self.snap_lims[0]-snap_lims[0]
			self.snap_lims[0] = snap_lims[0]
		if snap_lims[1]>self.snap_lims[1]:
			append_snaps = snap_lims[1]-self.snap_lims[1]
			self.snap_lims[1] = snap_lims[1]
		self.num_snaps = (snap_lims[1]+1)-snap_lims[0]

		prepend=np.zeros(prepend_snaps); append=np.zeros(append_snaps);
		self.time = np.append(np.concatenate((prepend,self.time)),append)
		if self.cosmological:
			self.redshift = np.append(np.concatenate((prepend,self.redshift)),append)
		self.snap_loaded = np.append(np.concatenate((prepend,self.snap_loaded)),append)

		# Populate the data dictionaries
		for key in self.total_data.keys():
			self.total_data[key] = np.append(np.concatenate((prepend,self.total_data[key])),append)
		for key in self.star_total_data.keys():
			self.star_total_data[key] = np.append(np.concatenate((prepend,self.star_total_data[key])),append)
		# Populate the gas property fields corresponding with each mask
		for sub_key in self.median_data.keys():
			for key in self.median_data[sub_key]:
				self.median_data[sub_key][key] = np.append(np.concatenate((prepend,self.median_data[sub_key][key])),append)

		self.all_snaps_loaded=False
		self.totals_loaded = np.zeros(len(self.totals_loaded))
		self.star_totals_loaded = np.zeros(len(self.star_totals_loaded))
		self.median_loaded = np.zeros(len(self.median_loaded))
		self.subsamples_loaded = np.zeros(len(self.subsamples_loaded))

	def add_total_property(self,properties):
		self.total_props += properties
		self.totals_loaded = np.append(self.totals_loaded,np.zeros(len(properties)))
		for prop in properties:
			self.total_data[prop] = np.zeros(self.num_snaps)

		self.snap_loaded = np.zeros(self.num_snaps,dtype=bool)
		self.all_snaps_loaded=False


	def add_median_property(self,properties):
		self.median_props += properties
		self.median_loaded = np.append(self.median_loaded,np.zeros(len(properties)))
		for subsample in self.median_data.keys():
			for prop in properties:
				self.median_data[subsample][prop] = np.zeros(self.num_snaps)

		self.snap_loaded = np.zeros(self.num_snaps,dtype=bool)
		self.all_snaps_loaded=False

	def add_subsample(self,subsamples):
		self.subsamples += subsamples
		self.subsamples_loaded = np.append(self.subsamples_loaded,np.zeros(len(subsamples)))
		for subsample in subsamples:
			self.median_data[subsample] = {key : np.zeros(self.num_snaps) for key in self.median_props}

		self.snap_loaded = np.zeros(self.num_snaps,dtype=bool)
		self.all_snaps_loaded=False


	# Set to include particles in specified halo
	# Give array of halo IDs equal to number of snapshots if the halo ID for the final main halo changes during the sim
	def set_halo(self, load_kwargs={}, set_kwargs={}, ids=None, use_halfmass_radius=False):
		if not self.setHalo:
			if ids is not None and len(ids)!=self.num_snaps:
				print("The array of halo IDs given does not equal the number of snapshots set for the given snap limits! These need to match.")
				return
			elif ids is not None:
				self.haloIDs = ids
			self.setHalo=True
			self.load_kwargs = load_kwargs
			self.set_kwargs = set_kwargs
			self.use_halfmass_radius = use_halfmass_radius

		return

	# Set to include particles in specified disk
	def set_disk(self, load_kwargs={}, set_kwargs={}):
		if not self.setDisk:
			self.setDisk=True
			self.load_kwargs = load_kwargs
			self.set_kwargs = set_kwargs

		return


	def load(self, increment=5):
		# Load total masses of different gases/stars and then calculated the median and 16/86th percentiles for
		# gas properties for each snapshot. Only loads set increment number of snaps at a time.

		if not self.setHalo and not self.setDisk:
			print("Need to call set_halo() or set_disk() to specify halo/disk to load time evolution data.")
			return 0


		snaps_loaded=0
		for i, snum in enumerate(np.arange(self.snap_lims[0],self.snap_lims[1]+1)):


			# Stop loading if already loaded set increment so it can be saved
			if snaps_loaded >= increment:
				return 1
			# Skip already loaded snaps
			if self.snap_loaded[i]:
				continue

			print('Loading snap',snum,'...')
			sp = Snapshot(self.sdir, snum, cosmological=self.cosmological)
			self.hubble = sp.hubble
			self.omega = sp.omega
			self.time[i] = sp.time
			if self.cosmological:
				self.redshift[i] = sp.redshift
				self.scale_factor[i] = sp.scale_factor
				self.time[i] = quick_lookback_time(sp.time, sp=sp)
			# Calculate the data fields for either all particles in the halo and all particles in the disk
			if self.setHalo:
				if self.haloIDs is not None:
					self.load_kwargs['id'] = self.haloIDs[i]
					print("For snap %i using Halo ID %i"%(snum,self.haloIDs[i]))
				gal = sp.loadhalo(**self.load_kwargs)
				if self.use_halfmass_radius:
					half_mass_radius = gal.get_half_mass_radius(rvir_frac=0.5)
					gal.set_zoom(rout=3.*half_mass_radius, kpc=True)
				else:
					gal.set_zoom(**self.set_kwargs)
			else:
				gal = sp.loaddisk(**self.load_kwargs)
				gal.set_disk(**self.set_kwargs)

			print('Loading gas data....')
			G = gal.loadpart(0)
			G_mass = G.get_property('M')
			T = G.get_property('T'); nH = G.get_property('nH'); fnh = G.get_property('nh'); fH2 = G.get_property('fH2')
			masks = {}
			for name in self.median_data.keys():
				if name == 'all':
					mask = np.ones(len(G_mass), dtype=bool)
				elif name == 'cold':
					mask = T <= 1E3
				elif name == 'warm':
					mask = (T<1E4) & (T>=1E3)
				elif name == 'hot':
					mask = T >= 1E4
				elif name == 'neutral':
					mask = fnh > 0.5
				elif name =='molecular':
					mask = fH2*fnh > 0.5
				else:
					print("Median subsampling %s is not supported so assuming all"%name)
					mask = np.ones(len(G_mass), dtype=bool)
				masks[name] = mask


			# First do totals
			for j, prop in enumerate(self.total_props):
				if not self.totals_loaded[j]:
					prop_mass = G.get_property(prop)
					self.total_data[prop][i] = np.nansum(prop_mass)


			# Now calculate medians for gas properties
			for j, subsample in enumerate(self.subsamples):
				for k, prop in enumerate(self.median_props):
					if not self.subsamples_loaded[j] or not self.median_loaded[k]:
						prop_vals = G.get_property(prop)
						mask = masks[subsample]
						# Deal with properties that are more than one value
						if len(prop_vals[mask]>0):
							self.median_data[subsample][prop][i] = weighted_percentile(prop_vals[mask], percentiles=[50],
															weights=G_mass[mask], ignore_invalid=True)
						else:
							self.median_data[subsample][prop][i] = np.nan


			# Finally do star properties if there are any
			S = gal.loadpart(4)
			print('Loading star data....')
			if S.npart == 0:
				for prop in self.star_total_data.keys():
					self.star_total_data[prop][i] = 0.
			else:
				S_mass = S.get_property('M')
				for j, prop in enumerate(self.star_total_props):
					if not self.star_totals_loaded[j]:
						if prop == 'M_star':
							self.star_total_data[prop][i] = np.nansum(S_mass)
						elif prop == 'sfr' or prop == 'sfr_100Myr':
							age = S.get_property('age')
							self.star_total_data[prop][i] = np.nansum(S_mass[age<=0.1])/1E8 # M_sol/yr
						elif prop == 'sfr_10Myr':
							age = S.get_property('age')
							self.star_total_data[prop][i] = np.nansum(S_mass[age<=0.01])/1E7 # M_sol/yr


			# snap all loaded
			self.snap_loaded[i]=True
			snaps_loaded+=1

		self.all_snaps_loaded=True

		self.totals_loaded = np.ones(len(self.totals_loaded))
		self.star_totals_loaded = np.ones(len(self.star_totals_loaded))
		self.subsamples_loaded = np.ones(len(self.subsamples_loaded))
		self.median_loaded = np.ones(len(self.median_loaded))

		return 1