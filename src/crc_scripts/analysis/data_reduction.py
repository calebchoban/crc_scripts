import os
import numpy as np
import pickle
from .. import config
from ..math_utils import weighted_percentile, quick_lookback_time
from ..io.snapshot import Snapshot
from .. import data_calc_utils as calc_utils
from ..io.snap_utils import check_snap_exist

import time

# This is a class that compiles the evolution data of a Snapshot/Halo
# over a specified time for a simulation

class Buffer(object):

    def __init__(self, sdir, snap_nums, cosmological=1, gas_props=None, gas_subsamples=None, 
                 star_props=None, star_subsamples=None,data_dirc='reduced_data/', 
                save_w_sims=True, halohist_file=None):
        # Set property totals and property medians you want from each snapshot. Also set median masks which will
        # take subsampled medians based on gas properties
        if gas_props is None:
            self.gas_properties = ['M_gas','M_H2','M_gas_neutral','M_dust','M_metals','M_sil','M_carb',
                            'M_SiC','M_iron','M_ORes','M_SNeIa_dust','M_SNeII_dust','M_AGB_dust','M_acc_dust',
                            'D/Z','Z','dz_acc','dz_SNeIa','dz_SNeII','dz_AGB','dz_sil','dz_carb',
                            'dz_SiC','dz_iron','dz_ORes','CinCO','fdense','fH2',
                            'Z_C','Z_O','Z_Mg','Z_Si','Z_Fe',
                            'C/H','C/H_gas','O/H','O/H_gas','Mg/H','Mg/H_gas','Si/H','Si/H_gas','Fe/H','Fe/H_gas']
        else: 
            self.gas_properties = gas_props
        self.gas_subsamples = ['all','cold','warm','hot','neutral','molecular'] if gas_subsamples is None else gas_subsamples
        
        if star_props is None:
            self.star_properties = ['M_star','M_star_10Myr','M_star_100Myr']
        else: 
            self.star_properties = star_props
        self.star_subsamples = ['all']  if star_subsamples is None else star_subsamples

        self.sdir = sdir
        self.snap_nums = snap_nums
        self.num_snaps = len(snap_nums)
        self.cosmological = cosmological
        self.halohist_file = halohist_file

        # Determines if you want to look at the Snapshot/Halo
        self.setHalo=False

        # Get the basename of the directory the snapshots are stored in
        self.basename = os.path.basename(os.path.dirname(os.path.normpath(sdir)))
        self.name = self.basename+'_reduced_data'
        if save_w_sims:
            self.data_dirc = sdir[:-7]+data_dirc
        else:
            self.data_dirc = './'+data_dirc
        

        self.k = 0

        return


    # Set data to only include particles in specified halo
    def set_halo(self, mode='AHF', hdir='', rout=1, kpc=False, use_halfmass_radius=False):
        # Make a unique name given parameters so we can tell what part of the halo was considered
        self.name += '_'+ mode + '_'
        if kpc and not use_halfmass_radius:
            self.name += str(rout) + 'kpc'
        elif use_halfmass_radius:
            self.name += str(rout) + 'R1/2'
        else:
            self.name += str(rout) + 'Rvir'
        self.name += '.pickle'
        # Store the arguements so we can pass them on later
        args = locals()
        args.pop('self')
        self.halo_args = args
        self.setHalo=True

        return


    def load(self, increment=2, override=False):

        if self.k: return

        # Check if object file has already been created if so load that first instead of creating a new one
        if not os.path.isfile(self.data_dirc+self.name) or override:
            self.reduced_data = Reduced_Data(self.sdir, self.snap_nums, self.gas_properties, self.gas_subsamples, self.star_properties, 
                                             self.star_subsamples, cosmological=self.cosmological,halohist_file=self.halohist_file)
            self.reduced_data.set_halo(**self.halo_args)
        else:
            with open(self.data_dirc+self.name, 'rb') as handle:
                self.reduced_data = pickle.load(handle)
            print("Reduced data already exists for the given halo setup so loading that first....")
            self.reduced_data.check_props_and_snaps(self.snap_nums, self.gas_properties, self.gas_subsamples, self.star_properties, 
                                             self.star_subsamples)



        if increment < 1:
            increment = 1

        while not self.reduced_data.all_snaps_loaded:
            ok = self.reduced_data.load(increment=increment)
            if not ok:
                print("Ran into an error when attempting to load data....")
                return
            self.save()

        print("All done reducing snapshot data. Please clap.....")
        self.k = 1
        return


    def save(self):
        # First create directory if needed
        if not os.path.isdir(self.data_dirc):
            os.mkdir(self.data_dirc)
            print("Directory " + self.data_dirc +  " Created ")

        with open(self.data_dirc+self.name, 'wb') as handle:
            pickle.dump(self.reduced_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return

    # Returns the specified data or derived data field if possible
    def get_data(self, prop, subsample='all',statistic='total'):

        if not self.reduced_data.all_snaps_loaded:
            print("Warning: Not all snapshots have been loaded! All unloaded values will be zero!")

        data_key = prop + '_' + subsample
        reduced_data = self.reduced_data.data
        if data_key in reduced_data:
            data = reduced_data[data_key]
        elif prop == 'time':
            data = self.reduced_data.time
        elif prop == 'redshift' and self.reduced_data.cosmological:
            data = self.reduced_data.redshift
        elif prop in ['sfr_10Myr','sfr']:
            data = self.reduced_data['M_sfr_10Myr_'+subsample]/1E7
        elif prop == 'sfr_100Myr':
            data = self.reduced_data['M_sfr_100Myr_'+subsample]/1E8
        elif 'source' in prop:
            if 'total' in statistic:
                if 'source_acc' in prop:
                    data = self.reduced_data['M_acc_dust_'+subsample]/self.reduced_data['M_dust_'+subsample]
                elif 'source_SNeIa' in prop:
                    data = self.reduced_data['M_SNeIa_dust_'+subsample]/self.reduced_data['M_dust_'+subsample]
                elif 'source_SNeII' in prop:
                    data = self.reduced_data['M_SNeII_dust_'+subsample]/self.reduced_data['M_dust_'+subsample]
                elif 'source_AGB' in prop:
                    data = self.reduced_data['M_AGB_dust_'+subsample]/self.reduced_data['M_dust_'+subsample]
                else:
                    print(prop," is not in the dataset.")
                    return None
                data[np.isnan(data)] = 0
            elif 'median' in statistic:
                if 'source_acc' in prop:
                    data = self.reduced_data['dz_acc_'+subsample]
                elif 'source_SNeIa' in prop:
                    data = self.reduced_data['dz_SNeIa_'+subsample]
                elif 'source_SNeII' in prop:
                    data = self.reduced_data['dz_SNeII_'+subsample]
                elif 'source_AGB' in prop:
                    data = self.reduced_data['dz_AGB_'+subsample]
                else:
                    print(prop," is not in the dataset.")
                    return None
            else:
                print(prop," is not in the dataset.")
                return None
        elif 'spec' in prop:
            if 'total' in statistic:
                if 'spec_sil' in prop:
                    data = self.reduced_data['M_sil_'+subsample]/self.reduced_data['M_dust_'+subsample]
                elif 'spec_carb' in prop:
                    data = self.reduced_data['M_carb_'+subsample]/self.reduced_data['M_dust_'+subsample]
                elif 'spec_SiC' in prop:
                    data = self.reduced_data['M_SiC_'+subsample]/self.reduced_data['M_dust_'+subsample]
                elif 'spec_iron' in prop and 'spec_ironIncl' not in prop:
                    data = self.reduced_data['M_iron_'+subsample]/self.reduced_data['M_dust_'+subsample]
                elif 'spec_ORes' in prop:
                    data = self.reduced_data['M_ORes_'+subsample]/self.reduced_data['M_dust_'+subsample]
                else:
                    print(prop," is not in the dataset.")
                    return None
                data[np.isnan(data)] = 0
            elif 'median' in statistic:
                if 'spec_sil' in prop:
                    data = self.reduced_data[subsample]['dz_sil']
                elif 'spec_carb' in prop:
                    data = self.reduced_data[subsample]['dz_carb']
                elif 'spec_SiC' in prop:
                    data = self.reduced_data[subsample]['dz_SiC']
                elif 'spec_iron' in prop and 'spec_ironIncl' not in prop:
                    data = self.reduced_data[subsample]['dz_iron']
                elif 'spec_ORes' in prop:
                    data = self.reduced_data[subsample]['dz_ORes']
                elif 'spec_sil+' in prop:
                    data = self.reduced_data[subsample]['dz_sil']+self.reduced_data[subsample]['dz_SiC']+\
                            self.reduced_data[subsample]['dz_iron']+self.reduced_data[subsample]['dz_ORes']
                else:
                    print(prop," is not in the dataset.")
                    return None
            else:
                print(prop," is not in the dataset.")
                return None
        elif prop in ['C/H_dust','O/H_dust','Mg/H_dust','Si/H_dust','Fe/H_dust']:
            base_name = prop.split('_')[0]
            total = self.reduced_data[base_name+'_'+subsample]
            gas = self.reduced_data[base_name+'_gas_'+subsample]
            data = 12 + np.log10(np.power(10,total-12) - np.power(10,gas-12))
        elif prop in ['Z_C_dust','Z_O_dust','Z_Mg_dust','Z_Si_dust','Z_Fe_dust']:
            base_name = prop.split('_')[0] + '_' + prop.split('_')[1]
            total = self.reduced_data[base_name+'_'+subsample]
            gas = self.reduced_data[base_name+'_gas_'+subsample]
            data = total-gas
        elif 'Si/C' in prop:
            if 'total' in statistic:
                data = (self.reduced_data['M_sil_'+subsample]+self.reduced_data['M_SiC_'+subsample]+ \
                        self.reduced_data['M_iron_'+subsample]+self.reduced_data['M_ORes_'+subsample])/self.reduced_data['M_carb_'+subsample]
            elif 'median' in statistic:
                data = (self.reduced_data['dz_sil_'+subsample]+self.reduced_data['dz_SiC_'+subsample]+\
                            self.reduced_data['dz_iron_'+subsample]+self.reduced_data['dz_ORes_'+subsample]) \
                            / self.reduced_data['dz_carb_'+subsample]
            else:
                print(prop, " is not in the dataset with given statistic.")
                return None
        else:
            print(prop," is not in the dataset.")
            return None

        return data.copy()



class Reduced_Data(object):

    def __init__(self, sdir, snap_nums, gas_props=None, gas_subsamples=['all'], star_props=None, star_subsamples=['all'],
                  cosmological=1, halohist_file=None):
        self.sdir = sdir
        self.snaps = np.sort(snap_nums)
        self.num_snaps = len(self.snaps)
        self.snap_loaded = np.zeros(self.num_snaps,dtype=bool)

        # First check that all these snaps exist
        missing_file = False
        for snap_num in snap_nums:
            if not check_snap_exist(sdir,snap_num):
                missing_file = True
        if missing_file: 
            print("The above files are missing. Check the snap directory and numbers given.")
            return

        # Load first snap to get cosmological parameters
        self.cosmological = cosmological
        sp = Snapshot(self.sdir, self.snaps[0], cosmological=self.cosmological)
        self.FIRE_ver = sp.FIRE_ver
        self.hubble = sp.hubble
        self.omega = sp.omega
        self.time = np.zeros(self.num_snaps)
        if self.cosmological:
            self.redshift = np.zeros(self.num_snaps)
            self.scale_factor = np.zeros(self.num_snaps)

        # Populate the data dictionaries
        prop_list = []
        for prop in gas_props:
            for sample in gas_subsamples:
                prop_list += [prop + '_' + sample]
        for prop in star_props:
            for sample in star_subsamples:
                prop_list += [prop + '_' + sample]
        
        self.gas_props = gas_props
        self.gas_subsamples = gas_subsamples
        self.star_props = star_props
        self.star_subsamples = star_subsamples
        self.data = {key : np.zeros(self.num_snaps) for key in prop_list}

        self.setHalo=False
        self.load_kwargs = {}
        self.set_kwargs = {}
        self.use_halfmass_radius = False
        # Used when dominate halo changes during sim
        self.halohist_file = halohist_file
        if self.halohist_file is None:
            self.haloIDs = np.array([-1]*self.num_snaps,dtype=int)
        else:
            if not os.path.isfile(halohist_file):
                print("%s halo history file does not exist."%halohist_file)
                return
            self.haloIDs = np.zeros(self.num_snaps,dtype=int)
            halo_IDs = np.loadtxt(self.halohist_file, usecols=(1,), unpack=True,dtype=int)
            halo_redshifts = np.loadtxt(self.halohist_file, usecols=(0,), unpack=True,dtype=float)
            ref_redshifts = 1/np.loadtxt(os.path.join(config.BASE_DIR,'FIRE'+str(self.FIRE_ver)+'_snapshot_scale-factors.txt'))-1
            ref_redshifts = ref_redshifts[snap_nums] 
            # Now find matching redshifts between snaps we have and redshifts listed in halo history
            for i, z in enumerate(ref_redshifts):
                idx = np.nanargmin(np.abs(z - halo_redshifts))
                self.haloIDs[i] = halo_IDs[idx]

        self.all_snaps_loaded=False
        
        return


    def check_props_and_snaps(self, snap_nums, gas_props, gas_subsamples, star_props, star_subsamples):
        # Deal with new snap numbers being added
        new_snaps = np.setdiff1d(snap_nums,self.snaps)
        if len(new_snaps)>0:
            print("New snaps listed below added. Will need to load these in")
            print(new_snaps)

            self.all_snaps_loaded = False
            self.num_snaps += len(new_snaps)

            if self.halohist_file is None:
               self.haloIDs = [-1]*self.num_snaps 
            else:
                halo_IDs = np.loadtxt(self.halohist_file, usecols=(1,), unpack=True,dtype=int)
                halo_redshifts = np.loadtxt(self.halohist_file, usecols=(0,), unpack=True,dtype=float)
                ref_redshifts = 1/np.loadtxt(os.path.join(config.BASE_DIR,'FIRE'+str(self.FIRE_ver)+'_snapshot_scale-factors.txt'))-1

            for i, n in enumerate(new_snaps):
                # Find where we need to insert the new snap to be in order
                index = np.searchsorted(self.snaps, n)

                self.snaps = np.insert(self.snaps, index, n)
                self.snap_loaded = np.insert(self.snap_loaded,index,0)
                self.time = np.insert(self.time,index,0)
                self.redshift = np.insert(self.redshift,index,0)
                self.scale_factor = np.insert(self.scale_factor,index,0)
                for key in self.data.keys():
                    self.data[key] = np.insert(self.data[key],index,0)
                # Now none of the data is fully loaded so reset this
                self.data_loaded = np.zeros(len(self.data.keys()))
                # Add new halo ID for the new snap
                if self.halohist_file is not None:
                    ref_redshift = ref_redshifts[n] 
                    idx = np.nanargmin(np.abs(ref_redshift - halo_redshifts))
                    self.haloIDs = np.insert(self.haloIDs,index,halo_IDs[idx])

        # Add any new gas and star properties
        prop_list = []
        for prop in gas_props:
            for sample in gas_subsamples:
                prop_list += [prop + '_' + sample]
        for prop in star_props:
            for sample in star_subsamples:
                prop_list += [prop + '_' + sample]
        new_props = np.setdiff1d(prop_list,list(self.data.keys()))
        if len(new_props)>0:
            print("New properties listed below added. Will need to reload all snaps.")
            print(new_props)

            new_data = {key : np.zeros(self.num_snaps) for key in new_props}
            self.data.update(new_data)
            # Need to reload all snaps
            self.snap_loaded = np.zeros(len(self.snap_loaded))
            self.all_snaps_loaded=False

        return


    # Set to include particles in specified halo
    # Give array of halo IDs equal to number of snapshots if the halo ID for the final main halo changes during the sim
    def set_halo(self, mode='AHF', hdir='', rout=1, kpc=False, use_halfmass_radius=False):
        if not self.setHalo:
            self.setHalo=True
            self.load_kwargs = {'mode':mode, 'hdir':hdir}
            self.set_kwargs = {'rout':rout, 'kpc':kpc}
            self.use_halfmass_radius = use_halfmass_radius
            return 1
        else:
            return 0


    def load(self, increment=5):
        # Load total masses of different gases/stars and then calculated the median and 16/86th percentiles for
        # gas properties for each snapshot. Only loads set increment number of snaps at a time.

        if not self.setHalo:
            print("Need to call set_halo() to specify halo to load time evolution data.")
            return 0

        snaps_loaded=0
        for i, snum in enumerate(self.snaps):


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
            # Calculate the data fields for either all particles in the halo
            if self.setHalo:
                self.load_kwargs['id'] = self.haloIDs[i]
                print("For snap %i using Halo ID %i"%(snum,self.haloIDs[i]))
                gal = sp.loadhalo(**self.load_kwargs)
                if self.use_halfmass_radius:
                    half_mass_radius = gal.get_half_mass_radius(rvir_frac=0.5)
                    gal.set_zoom(rout=3.*half_mass_radius, kpc=True)
                else:
                    gal.set_zoom(**self.set_kwargs)

            print('Loading data....')

            # First do totals
            for subsample in self.gas_subsamples:
                # Only need to get the mask once
                sample_mask = calc_utils.get_particle_mask(0,gal,mask_criteria=subsample)
                for prop in self.gas_props:
                    data_key = prop + '_' + subsample
                    self.data[data_key][i] = calc_utils.calc_gal_int_params(prop,gal,mask=sample_mask)
            for subsample in self.star_subsamples:
                sample_mask = calc_utils.get_particle_mask(4,gal,mask_criteria=subsample)
                for prop in self.star_props:
                    data_key = prop + '_' + subsample
                    self.data[data_key][i] = calc_utils.calc_gal_int_params(prop,gal,mask=sample_mask)
            # snap all loaded
            self.snap_loaded[i]=True
            snaps_loaded+=1

        self.all_snaps_loaded=True

        return 1