import os
import numpy as np
from .utils import weighted_percentile
from . import config
import pickle
from .snapshot import Snapshot
import yt

from shutil import copyfile
import h5py


# This is a class that creates mock sight lines from a given Snapshot using yt
# TODO: Add Edan'el's sight line creator to replace using yt

class Sight_Lines(object):

	def __init__(self, sdir, snum, cosmological=0, dust_impl='species', periodic_bound_fix=False, dirc='./', name=None):

		self.sdir = sdir
		self.snap_num = snum
		self.cosmological = cosmological
		self.dirc = dirc
		self.dust_impl = dust_impl
		# In case the sim was non-cosmological and used periodic BC which causes
		# galaxy to be split between the 4 corners of the box
		self.pb_fix = False
		if periodic_bound_fix and cosmological==0:
			self.pb_fix=True


		# Get the basename of the directory the snapshots are stored in
		self.basename = os.path.basename(os.path.dirname(os.path.normpath(sdir)))
		# Name for saving object
		if name == None:
			self.name = 'sightlines_'+self.dust_impl+'_'+self.basename+'_snap_'+str(self.snap_num)+'.pickle'
		else:
			self.name = name

		# Check if data has already been saved and if so load that instead
		if os.path.isfile(dirc+name):
			with open(dirc+name, 'rb') as handle:
				self.sightline_data = pickle.load(handle)
			print("Preexisting file %s already exists and is loaded!"%name)
			self.k = 1
			return
		else:
			print("Preexisting file does not exist, so you to load in the data."%name)
			self.k = 0


		if self.pb_fix:
			# First need to load in and fix periodic coordinates since non-cosmo sims with
			# Periodic coordinates have funky coordinates
			self.snap_name = 'snapshot_'+str(self.snap_num)+'.hdf5'
			copyfile(self.sdir+self.snap_name, './fixed_'+self.snap_name)
			self.snap_name='fixed_'+self.snap_name

			f = h5py.File('./'+self.snap_name, 'r+')
			grp = f['PartType0']
			old_p = grp['Coordinates']
			new_p = old_p[...].copy()
			boxsize = f['Header'].attrs['BoxSize']
			mask1 = new_p > boxsize/2; mask2 = new_p <= boxsize/2
			new_p[mask1] -= boxsize/2; new_p[mask2] += boxsize/2;
			old_p[...]=new_p
			f.close()

			full_dir = './'+self.snap_name
		else:
			full_dir = sdir+'/'+self.snap_name


		# Load data with yt
		self.ds = yt.load(full_dir)
		_, self.center = self.ds.find_max(('gas', 'density'))

		return


	# Creates random start and end points and distance for sight line given start galactic radius and center coordinates of galaxy
	def ray_points(self, center, radius, dist_lims):
		solar_r = radius
		solar_theta = np.random.random()*2*np.pi
		start = np.array([solar_r*np.cos(solar_theta), solar_r*np.sin(solar_theta),0])+center
		# Random sight line distance of 0.1-2kpc
		distance = dist_lims[0]+np.random.random()*dist_lims[1]
		theta = np.random.random()*2*np.pi
		end =  np.array([distance*np.cos(theta), distance*np.sin(theta),0])+start

		return start,end,distance

	# Create N number of sightlines
	def create_sightlines(self, N=100, radius=config.SOLAR_GAL_RADIUS, dist_lims=[0.1,1.9]):
		if self.k: return

		NH_all = np.zeros(N)
		NH_neutral = np.zeros(N)
		NH2 = np.zeros(N)
		NX_gas=np.zeros((N,len(config.ELEMENTS)))
		NX_dust=np.zeros((N,len(config.ELEMENTS)))
		distances = np.zeros(N)
		points = np.zeros((N,2,3))

		# Make a ray
		for i in range(N):
			if i%10==0: print(i)
			start, end, distance = self.ray_points(self.center.data, radius, dist_lims)
			points[i] = np.array([start,end])
			distances[i] = distance
			ray = self.ds.ray(start, end)
			NH_all[i] = np.sum(ray["gas", "density"]*ray['dts']*(distance*config.Kpc_to_cm*self.ds.units.cm)*(1.-ray["gas", "metallicity"]-ray["gas", "He_metallicity"])/(config.H_MASS*self.ds.units.g))
			NH_neutral[i] = np.sum(ray["gas", "density"]*ray['dts']*(distance*config.Kpc_to_cm*self.ds.units.cm)*ray["PartType0", "NeutralHydrogenAbundance"]*(1.-ray["gas", "metallicity"]-ray["gas", "He_metallicity"])/(config.H_MASS*self.ds.units.g))
			if ('PartType0', 'MolecularMassFraction') in self.ds.field_list:
				fH2 = ray[('PartType0', 'MolecularMassFraction')]
			else:
				fH2 = ray[('PartType0', 'DustMolecular')][:,0]
			NH2[i] = np.sum(ray["gas", "density"]*ray['dts']*(distance*config.Kpc_to_cm*self.ds.units.cm)*fH2*(1.-ray["gas", "metallicity"]-ray["gas", "He_metallicity"])/(config.H_MASS*self.ds.units.g))

			# Go through each each element
			for j in range(1,len(config.ELEMENTS)):

				elem = config.ELEMENTS[j]
				NX_gas[i,j] = np.sum(ray["gas", "density"]*(ray["gas", elem+"_metallicity"]-ray["PartType0", "DustMetallicity"][:,j])*ray['dts']*distance*config.Kpc_to_cm*self.ds.units.cm)
				NX_dust[i,j] = np.sum(ray["gas", "density"]*ray["PartType0", "DustMetallicity"][:,j]*ray['dts']*distance*config.Kpc_to_cm*self.ds.units.cm)

			# Now add them all up for total Z
			NX_gas[i,0] = np.sum(NX_gas[i,1:])
			NX_dust[i,0] = np.sum(NX_dust[i,1:])

		depl_X = NX_gas/(NX_gas+NX_dust)
		self.sightline_data = {'NH_neutral':NH_neutral,'NH':NH_all,'NX_gas':NX_gas,'NX_dust':NX_dust,'depl_X':depl_X,'points':points,'distance':distances}
		pickle.dump(self.sightline_data, open(self.name, "wb" ))

		# Delete extra snap if pb_fix work around used
		if self.pb_fix:
			os.remove(self.snap_name)

		return