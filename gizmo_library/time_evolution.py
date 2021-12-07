import os
import numpy as np
from .utils import weighted_percentile
import pickle
from .snapshot import Snapshot

# This is a class that compiles the evolution data of a Snapshot/Halo/Disk
# over a specified time for a simulation

class Dust_Evo(object):

	def __init__(self, sdir, snap_lims, cosmological=0, periodic_bound_fix=False, dust_depl=False,
				 statistic='average' , dirc='./', name=None):
		# If a name is given then data has already been saved so just load that
		if name != None:
			if os.path.isfile(dirc+name):
				with open(dirc+name, 'rb') as handle:
					self.data = pickle.load(handle)
			else:
				print("Preexisting file %s given but it does not exist!"%name)
				return
		else:
			self.data = Dust_Evo_Data(sdir, snap_lims, cosmological=cosmological, periodic_bound_fix=periodic_bound_fix, dust_depl=dust_depl, statistic=statistic)

		self.stat = statistic
		self.sdir = sdir
		self.snap_lims = snap_lims
		self.num_snaps = (snap_lims[1]+1)-snap_lims[0]
		self.cosmological = cosmological
		self.dirc = dirc
		# In case the sim was non-cosmological and used periodic BC which causes
		# galaxy to be split between the 4 corners of the box
		self.pb_fix = False
		if periodic_bound_fix and cosmological==0:
			self.pb_fix=True
		# Determines if you want to look at the Snapshot/Halo/Disk
		self.setHalo=False
		self.setDisk=False

		# Get the basename of the directory the snapshots are stored in
		self.basename = os.path.basename(os.path.dirname(os.path.normpath(sdir)))
		# Name for saving object
		self.name = 'dust_evo_'+statistic+'_'+self.data.dust_impl+'_'+self.basename+'_snaps_'+str(snap_lims[0])+'_to_'+str(snap_lims[1])

		self.k = 0

		return



	# Set data to only include particles in specified halo
	def set_halo(self, **kwargs):
		self.data.set_halo(**kwargs)
		self.setHalo=True

		return


	# Set data to only include particles in specified disk
	def set_disk(self, **kwargs):
		self.data.set_disk(**kwargs)
		self.setDisk=True

		return


	def load(self):

		if self.k: return

		# Check if previously saved if so load that and we are done
		if os.path.isfile(self.dirc+self.name+'.pickle'):
			with open(self.dirc+self.name+'.pickle', 'rb') as handle:
				self.data = pickle.load(handle)
			self.k = 1
			return

		if self.stat == 'average':
			self.data.load_average()
		elif self.stat == 'total':
			self.data.load_total()
		else:
			print('%s is not a valid statistic for Dust_Evo'%self.stat)
			return

		self.k = 1

		return


	def save(self):

		# First create directory if needed
		if not os.path.isdir(self.dirc):
			os.mkdir(self.dirc)
			print("Directory " + self.dirc +  " Created ")

		with open(self.dirc+self.name+'.pickle', 'wb') as handle:
			pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

		return

	# Returns the specified data
	def get_data(self, data_name):

		if data_name == 'D/Z':
			data = self.data.dz
		elif data_name == 'Z':
			data = self.data.z
		elif data_name == 'source_frac':
			data = self.data.source
		elif data_name == 'spec_frac':
			data = self.data.spec
		elif data_name == 'Si/C':
			data = self.data.spec[:,0]/self.data.spec[:,1]
		elif data_name == 'time':
			data = self.data.time
		elif data_name == 'redshift':
			data = self.data.redshift
		else:
			print(data_name," is not a defined data name")
			return

		return data


	# TODO : Add ability to add on more snapshots to object which is helpful if loading has to be done in batch jobs




class Dust_Evo_Data(object):

	def __init__(self, sdir, snap_lims, cosmological=0, periodic_bound_fix=False, dust_depl=False, statistic='average'):
		self.sdir = sdir
		self.stat = statistic
		self.snap_lims = snap_lims
		self.num_snaps = (snap_lims[1]+1)-snap_lims[0]
		self.cosmological = cosmological
		self.time = np.zeros(self.num_snaps)
		if self.cosmological:
			self.redshift = np.zeros(self.num_snaps)


		# In case the sim was non-cosmological and used periodic BC which causes
		# galaxy to be split between the 4 corners of the box
		self.pb_fix = False
		if periodic_bound_fix and cosmological==0:
			self.pb_fix=True
		# Determines if you want to look at the Snapshot/Halo/Disk
		self.setHalo=False
		self.setDisk=False


		# Load the first snapshot to check needed array sizes
		sp = Snapshot(self.sdir, self.snap_lims[0], cosmological=self.cosmological, periodic_bound_fix=self.pb_fix)
		self.dust_impl = sp.dust_impl
		self.m = np.zeros(self.num_snaps)
		self.z = np.zeros(self.num_snaps)
		self.dz = np.zeros(self.num_snaps)
		self.spec = np.zeros([self.num_snaps, sp.Flag_DustSpecies])
		self.source = np.zeros([self.num_snaps, 4])

		return


	# Set data to only include particles in specified halo
	def set_halo(self, **kwargs):
		self.setHalo=True
		self.kwargs = kwargs

		return

	# Set data to only include particles in specified disk
	def set_disk(self, **kwargs):
		self.setDisk=True
		self.kwargs = kwargs

		return


	def load_total(self):
		for i, snum in enumerate(np.arange(self.snap_lims[0],self.snap_lims[1]+1)):
			sp = Snapshot(self.sdir, snum, cosmological=self.cosmological, periodic_bound_fix=self.pb_fix)
			self.time[i] = sp.time
			if self.cosmological: self.redshift[i] = sp.redshift
			if self.setHalo:
				gal = sp.loadhalo(**self.kwargs)
			elif self.setDisk:
				gal = sp.loaddisk(**self.kwargs)
			else:
				print("Need to specify halo or disk before loading data")
				return
			gas = gal.loadpart(0)

			self.z[i] = np.nansum(gas.z[:,0]*gas.m)/np.nansum(gas.m)
			self.dz[i] = np.nansum(gas.dz[:,0]*gas.m)/np.nansum(gas.z[:,0]*gas.m)
			self.spec[i] = np.nansum(gas.spec*gas.m[:,np.newaxis], axis=0)/np.nansum(gas.dz[:,0]*gas.m)
			self.source[i] = np.nansum(gas.dzs*gas.dz[:,0]*gas.m[:,np.newaxis], axis=0)/np.nansum(gas.dz[:,0]*gas.m)

		return


	def load_average(self):
		for i, snum in enumerate(np.arange(self.snap_lims[0],self.snap_lims[1]+1)):
			sp = Snapshot(self.sdir, snum, cosmological=self.cosmological, periodic_bound_fix=self.pb_fix)
			self.time[i] = sp.time
			if self.cosmological: self.redshift[i] = sp.redshift
			if self.setHalo:
				gal = sp.loadhalo(**self.kwargs)
			elif self.setDisk:
				gal = sp.loaddisk(**self.kwargs)
			else:
				print("Need to specify halo or disk before loading data")
				return
			gas = gal.loadpart(0)

			self.z[i] = weighted_percentile(gas.z[:,0], percentiles=[50], weights=gas.m, ignore_invalid=True)
			self.dz[i] = weighted_percentile(gas.dz[:,0]/gas.z[:,0], percentiles=[50], weights=gas.m, ignore_invalid=True)
			for j in range(sp.Flag_DustSpecies):
				self.spec[i,j] = weighted_percentile(gas.spec[:,j]/gas.dz[:,0], percentiles=[50], weights=gas.m, ignore_invalid=True)
			for j in range(4):
				self.source[i,j] = weighted_percentile(gas.dzs[:,j], percentiles=[50], weights=gas.m, ignore_invalid=True)

		return