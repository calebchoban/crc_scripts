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
                save_w_sims=True, halohist_file=None, base_FIRE=False):
        # Set property totals and property medians you want from each snapshot. Also set median masks which will
        # take subsampled medians based on gas properties
        if gas_props is None:
            if not base_FIRE:
                self.gas_properties = ['M_gas','M_H2','M_gas_neutral','M_dust','M_metals','M_sil','M_carb',
                                'M_SiC','M_iron','M_ORes','M_SNeIa_dust','M_SNeII_dust','M_AGB_dust','M_acc_dust',
                                'D/Z','Z','dz_acc','dz_SNeIa','dz_SNeII','dz_AGB','dz_sil','dz_carb',
                                'dz_SiC','dz_iron','dz_ORes','CinCO','fdense','fH2',
                                'Z_C','Z_O','Z_Mg','Z_Si','Z_Fe',
                                'C/H','C/H_gas','O/H','O/H_gas','Mg/H','Mg/H_gas','Si/H','Si/H_gas','Fe/H','Fe/H_gas']
            else:
                self.gas_properties = ['M_gas','M_H2','M_gas_neutral','M_metals','Z','fH2']
        else: 
            self.gas_properties = gas_props
        self.gas_subsamples = ['all','cold','warm','hot','coronal','neutral','molecular','ionized'] if gas_subsamples is None else gas_subsamples
        
        if star_props is None:
            self.star_properties = ['M_star','M_star_10Myr','M_star_100Myr','r_1/2']
        else: 
            self.star_properties = star_props
        self.star_subsamples = ['all']  if star_subsamples is None else star_subsamples

        self.sdir = sdir
        self.snap_nums = np.sort(snap_nums)
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


    def load(self, increment=2, override=False, verbose=True):

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
            ok = self.reduced_data.load(increment=increment, verbose=verbose)
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
    def get_data(self, prop, subsample='all',statistic='total',snap_nums=None):

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
            data = reduced_data['M_star_10Myr_'+subsample]/1E7
        elif prop == 'sfr_100Myr':
            data = reduced_data['M_star_100Myr_'+subsample]/1E8
        elif prop == 'ssfr':
            data = (reduced_data['M_star_10Myr_'+subsample]/0.01)/reduced_data['M_star_'+subsample] # Gyr^-1
        elif prop in ['f_cold','f_warm','f_hot','f_H2','f_neutral','f_coronal','f_ionized']:
            if 'cold' in prop: data = reduced_data['M_gas_cold']/reduced_data['M_gas_all']
            elif 'warm' in prop: data = reduced_data['M_gas_warm']/reduced_data['M_gas_all']
            elif 'hot' in prop: data = reduced_data['M_gas_hot']/reduced_data['M_gas_all']
            elif 'H2' in prop: data = reduced_data['M_H2_all']/reduced_data['M_gas_all']
            elif 'neutral' in prop: data = reduced_data['M_gas_neutral_all']/reduced_data['M_gas_all']
            elif 'coronal' in prop: data = reduced_data['M_gas_coronal']/reduced_data['M_gas_all']
            elif 'ionized' in prop: data = reduced_data['M_gas_ionzed']/reduced_data['M_gas_all']
        elif 'source' in prop:
            if 'total' in statistic:
                if 'source_frac' in prop:
                    data = [reduced_data['M_acc_dust_'+subsample],reduced_data['M_acc_dust_'+subsample],reduced_data['M_AGB_dust_'+subsample],
                            reduced_data['M_SNeIa_dust_'+subsample]]/reduced_data['M_dust_'+subsample]
                elif 'source_acc' in prop:
                    data = reduced_data['M_acc_dust_'+subsample]/reduced_data['M_dust_'+subsample]
                elif 'source_SNeII' in prop:
                    data = reduced_data['M_SNeII_dust_'+subsample]/reduced_data['M_dust_'+subsample]
                elif 'source_AGB' in prop:
                    data = reduced_data['M_AGB_dust_'+subsample]/reduced_data['M_dust_'+subsample]
                elif 'source_SNeIa' in prop:
                    data = reduced_data['M_SNeIa_dust_'+subsample]/reduced_data['M_dust_'+subsample]
                else:
                    print(prop," is not in the dataset.")
                    return None
                data[np.isnan(data)] = 0
            elif 'median' in statistic:
                if 'source_frac' in prop:
                    data = [reduced_data['dz_acc_'+subsample],reduced_data['dz_SNeII_'+subsample],reduced_data['dz_AGB_'+subsample],
                            reduced_data['dz_SNeIa_'+subsample]]
                elif 'source_acc' in prop:
                    data = reduced_data['dz_acc_'+subsample]
                elif 'source_SNeII' in prop:
                    data = reduced_data['dz_SNeII_'+subsample]
                elif 'source_AGB' in prop:
                    data = reduced_data['dz_AGB_'+subsample]
                elif 'source_SNeIa' in prop:
                    data = reduced_data['dz_SNeIa_'+subsample]
                else:
                    print(prop," is not in the dataset.")
                    return None
            else:
                print(prop," is not in the dataset.")
                return None
        elif 'spec' in prop:
            if 'total' in statistic:
                if 'spec_frac' in prop:
                    if 'spec_frac+'==prop:
                        data = [reduced_data['M_sil_'+subsample]+reduced_data['M_SiC_'+subsample]+reduced_data['M_iron_'+subsample]+\
                                reduced_data['M_ORes_'+subsample],reduced_data['M_carb_'+subsample],]/reduced_data['M_dust_'+subsample]
                    else:
                        data = [reduced_data['M_sil_'+subsample],reduced_data['M_carb_'+subsample],reduced_data['M_SiC_'+subsample],
                                reduced_data['M_iron_'+subsample],reduced_data['M_ORes_'+subsample]]/reduced_data['M_dust_'+subsample]
                elif 'spec_sil' in prop:
                    data = reduced_data['M_sil_'+subsample]/reduced_data['M_dust_'+subsample]
                elif 'spec_carb' in prop:
                    data = reduced_data['M_carb_'+subsample]/reduced_data['M_dust_'+subsample]
                elif 'spec_SiC' in prop:
                    data = reduced_data['M_SiC_'+subsample]/reduced_data['M_dust_'+subsample]
                elif 'spec_iron' in prop and 'spec_ironIncl' not in prop:
                    data = reduced_data['M_iron_'+subsample]/reduced_data['M_dust_'+subsample]
                elif 'spec_ORes' in prop:
                    data = reduced_data['M_ORes_'+subsample]/reduced_data['M_dust_'+subsample]
                else:
                    print(prop," is not in the dataset.")
                    return None
                data[np.isnan(data)] = 0
            elif 'median' in statistic:
                if 'spec_frac' in prop:
                    if 'spec_frac+'==prop:
                        data = [reduced_data['dz_sil_'+subsample]+reduced_data['dz_SiC_'+subsample]+reduced_data['dz_iron_'+subsample]+\
                                reduced_data['dz_ORes_'+subsample],reduced_data['dz_carb_'+subsample]]
                    else:
                        data = [reduced_data['dz_sil_'+subsample],reduced_data['dz_carb_'+subsample],reduced_data['dz_SiC_'+subsample],
                            reduced_data['dz_iron_'+subsample],reduced_data['dz_ORes_'+subsample]]
                elif 'spec_sil' in prop:
                    data = reduced_data['dz_sil_'+subsample]
                elif 'spec_carb' in prop:
                    data = reduced_data['dz_carb_'+subsample]
                elif 'spec_SiC' in prop:
                    data = reduced_data['dz_SiC_'+subsample]
                elif 'spec_iron' in prop and 'spec_ironIncl' not in prop:
                    data = reduced_data['dz_iron_'+subsample]
                elif 'spec_ORes' in prop:
                    data = reduced_data['dz_ORes_'+subsample]
                elif 'spec_sil+' in prop:
                    data = reduced_data['dz_sil_'+subsample]+reduced_data['dz_SiC_'+subsample]+\
                            reduced_data['dz_iron_'+subsample]+reduced_data['dz_ORes_'+subsample]
                else:
                    print(prop," is not in the dataset.")
                    return None
            else:
                print(prop," is not in the dataset.")
                return None
        elif prop in ['C/H_dust','O/H_dust','Mg/H_dust','Si/H_dust','Fe/H_dust']:
            base_name = prop.split('_')[0]
            total = reduced_data[base_name+'_'+subsample]
            gas = reduced_data[base_name+'_gas_'+subsample]
            data = 12 + np.log10(np.power(10,total-12) - np.power(10,gas-12))
        elif prop in ['Z_C_dust','Z_O_dust','Z_Mg_dust','Z_Si_dust','Z_Fe_dust']:
            base_name = prop.split('_')[0] + '_' + prop.split('_')[1]
            total = reduced_data[base_name+'_'+subsample]
            gas = reduced_data[base_name+'_gas_'+subsample]
            data = total-gas
        elif 'Si/C' in prop:
            if 'total' in statistic:
                data = (reduced_data['M_sil_'+subsample]+reduced_data['M_SiC_'+subsample]+ \
                        reduced_data['M_iron_'+subsample]+reduced_data['M_ORes_'+subsample])/reduced_data['M_carb_'+subsample]
            elif 'median' in statistic:
                data = (reduced_data['dz_sil_'+subsample]+reduced_data['dz_SiC_'+subsample]+\
                            reduced_data['dz_iron_'+subsample]+reduced_data['dz_ORes_'+subsample]) \
                            / reduced_data['dz_carb_'+subsample]
            else:
                print(prop, " is not in the dataset with given statistic.")
                return None
        else:
            print(prop," is not in the dataset.")
            return None
        
        # Only return specified snaps
        data = data.copy()
        if snap_nums is not None:
            intersect = np.intersect1d(snap_nums, self.reduced_data.snaps)
            if np.array_equal(intersect.sort(), snap_nums.sort()):
                indices = np.isin(self.reduced_data.snaps,snap_nums)
                if len(np.shape(data))>1:
                    data = data[:,indices]
                else:
                    data = data[indices]
            else:
                print("Missing snaps in loaded data. Only those listed below are included.")
                print(intersect,snap_nums)
                return

        return data



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
            col_names = list(np.genfromtxt(self.halohist_file,skip_header=0,max_rows = 1,dtype=str,comments='@'))
            halo_snums = np.loadtxt(halohist_file,skiprows=1,usecols=[col_names.index('snum')])
            halo_IDs =   np.loadtxt(halohist_file,skiprows=1,usecols=[col_names.index('ID')])
            for i,snap_num in enumerate(snap_nums):
                if snap_num in halo_snums:
                    self.haloIDs[i] = halo_IDs[halo_snums==snap_num]
                else:
                    print("WARNING: The requested snapshot %i is not included in the provided halo history file!"%snap_num +
                          "Either fix the file or don't specify a file to assume the largest halo for every snap.")
                    return

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
                col_names = list(np.genfromtxt(self.halohist_file,skip_header=0,max_rows = 1,dtype=str,comments='@'))
                halo_snums = np.loadtxt(self.halohist_file,skiprows=1,usecols=[col_names.index('snum')])
                halo_IDs =   np.loadtxt(self.halohist_file,skiprows=1,usecols=[col_names.index('ID')])

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
                # Now the data is not loaded so reset this
                self.data_loaded = np.zeros(len(self.data.keys()))
                # Add new halo ID for the new snap
                if self.halohist_file is not None:
                    idx = np.searchsorted(halo_snums, n)
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
            # Add new props not already in reduced_data
            self.gas_props = self.gas_props+list(set(gas_props)-set(self.gas_props))
            self.gas_subsamples = self.gas_subsamples+list(set(gas_subsamples)-set(self.gas_subsamples))
            self.star_props = self.star_props+list(set(star_props)-set(self.star_props))
            self.star_subsamples = self.star_subsamples+list(set(star_subsamples)-set(self.star_subsamples))
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


    def load(self, increment=5, verbose=True):
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

            if verbose: print('Loading snap',snum,'...')
            start=time.time()
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
                if verbose: print("For snap %i using Halo ID %i"%(snum,self.haloIDs[i]))
                gal = sp.loadhalo(**self.load_kwargs)
                if self.use_halfmass_radius:
                    half_mass_radius = gal.get_half_mass_radius(rvir_frac=0.5)
                    gal.set_zoom(rout=3.*half_mass_radius, kpc=True)
                else:
                    gal.set_zoom(**self.set_kwargs)


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
                    if prop == 'r_1/2':
                        self.data[data_key][i] = gal.get_half_mass_radius(within_radius=None, geometry='spherical', ptype=4, rvir_frac=0.5)
                    else:
                        self.data[data_key][i] = calc_utils.calc_gal_int_params(prop,gal,mask=sample_mask)
            # snap all loaded
            self.snap_loaded[i]=True
            snaps_loaded+=1

            end=time.time()
            if verbose: print("Took %.3f secs to load this snap:"%(end-start))

        self.all_snaps_loaded=True

        return 1