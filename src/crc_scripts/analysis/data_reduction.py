import os
import numpy as np
import pickle
from .. import config
from ..io.snapshot import Snapshot
from ..utils import data_calc_utils as calc_utils
from ..utils.math_utils import weighted_percentile, quick_lookback_time
from ..utils.snap_utils import check_snap_exist
import time


# This is a class that compiles the evolution data of a Snapshot/Halo
# over a specified time for a simulation

class MultiSnapDataIO(object):
    """
    This class handles the initialization and IO of reducing gas and star particle data from given snapshot numbers. 
    The specified gas and star properties along with specified particle subsamples are calculated by and stored in 
    a MultiSnapReducedData object which is a file to disk.
    The file can be loaded and updated with new snapshots or properties.
    """

    def __init__(self, 
                 sdir:str, 
                 snap_nums:list, 
                 gas_props:list=None, 
                 gas_subsamples:list=None, 
                 star_props:list=None, 
                 star_subsamples:list=None, 
                 save_dir:str=None, 
                 halohist_file:str=None):
        """
        Parameters
        ----------
        sdir : str
            Directory where the simulation snapshots are stored.
        snap_nums : list
            List of snapshot numbers to reduce.
        gas_props : list, optional 
            List of gas properties to reduce. If None, defaults to commonly used values, determined based on simulation flags (mainly for dust).
        gas_subsamples : list, optional
            List of gas subsamples to reduce. If None, defaults to ['all','cold','warm','hot','coronal','neutral','molecular','ionized'].
        star_props : list, optional
            List of star properties to reduce. If None, defaults to ['M_star','M_form_10Myr','M_form_100Myr','r1/2_stars'].
        star_subsamples : list, optional
            List of star subsamples to reduce. If None, defaults to ['all'].
        save_dir : str, optional
            Directory to save the reduced data. If None, defaults to the directory where the snapshots are stored.
        halohist_file : str, optional
            Path to the AHF halo history file used to determine where the galaxy center is for each snapshot. 
            This is necessary for simulations where the main halo between snapshots such as dwarf-mass galaxies and at high-z.
            If None, defaults to determining the rough galactic center from dense gas/young star locations.
        """
        
        self.loaded = False
        self.sdir = sdir
        self.snap_nums = np.sort(snap_nums)
        self.num_snaps = len(snap_nums)
        self.halohist_file = halohist_file

        # Determines if you want to look at the entire Snapshot or a specific Halo
        self.setHalo=False

        # Get the basename of the directory the snapshots are stored in
        self.basename = os.path.basename(os.path.dirname(os.path.normpath(sdir)))
        self.name = self.basename+'_reduced_data'
        # Set the basename of the directory the reduced data is saved to
        if save_dir is None:
            self.data_dirc = sdir[:-7]+'reduced_data/'
        else:
            self.data_dirc = './'+save_dir


        # Open up the first provided snapshot to get load the header info on the simulation
        if not check_snap_exist(self.sdir,self.snap_nums[0]):
            raise ValueError("The first snapshot number does not exist. Check the snap directory and numbers given.")
        self.snap = Snapshot(self.sdir, self.snap_nums[0])
        self.cosmological = self.snap.cosmological
    

        if gas_props is None:
            self.gas_properties = ['M_gas','M_H2','M_gas_neutral','M_metals','Z','fH2','O/H','r1/2_gas','r1/2_neutral','r1/2_H2']
            if self.snap.Flag_DustSpecies:
                # All dust simulations have these
                self.gas_properties += ['M_dust','D/Z','O/H_gas','r1/2_dust',
                                        'M_SNeIa_dust','M_SNeII_dust','M_AGB_dust','M_acc_dust',
                                        'dz_SNeIa','dz_SNeII','dz_AGB','dz_acc']
                if self.snap.Flag_DustSpecies >= 2:
                    # silicates and carbonaceous dust
                    self.gas_properties += ['M_sil','M_carb','dz_sil','dz_carb']
                if self.snap.Flag_DustSpecies >= 3:
                    # Metallic iron dust
                    self.gas_properties += ['M_iron','dz_iron']
                if self.snap.Flag_DustSpecies >= 5:
                    # Silicon carbide dust and O reservoir dust species
                    self.gas_properties += ['M_SiC','M_ORes','dz_SiC','dz_ORes']

                if not self.snap.Flag_GrainSizeBins:
                    # Only tracked for fixed grain size simulations
                    self.gas_properties += ['CinCO','fdense']
                else:
                    # For simulations with evolving grain sizes
                    self.gas_properties += ['M_grain_small', 'M_grain_large','f_STL',
                                            'M_grain_small_sil','M_grain_large_sil','M_grain_small_carb','M_grain_large_carb','M_grain_small_iron','M_grain_large_iron']
        else:
            self.gas_properties = gas_props

        # Subsampling of gas properties
        self.gas_subsamples = ['all','cold','warm','hot','coronal','neutral','molecular','ionized'] if gas_subsamples is None else gas_subsamples
        
        if star_props is None:
            self.star_properties = ['M_star','Z','M_form_10Myr','M_form_100Myr','r1/2_stars']
        else: 
            self.star_properties = star_props
        self.star_subsamples = ['all']  if star_subsamples is None else star_subsamples


        return


    # Set data to only include particles in specified halo
    def set_halo(self, 
                 mode:str='AHF', 
                 rout:float=1, 
                 kpc:bool=False, 
                 use_halfmass_radius:bool=False):
        """
        Set the halo arguments you want when loading each snapshot to determined which particles 
        are in the galactic halo. These arguments will be used when loading each snapshot.
        This function must be called before loading snapshot data.


        Parameters
        ----------
        mode : str
            Set how the galactic halo properties will be determined. Default 'AHF' uses AHF halo files.
        rout : float
            The radius of the halo to include in kpc or Rvir. Default is 1.
        kpc : bool
            If True, rout is in kpc. If False, rout is in Rvir.
        use_halfmass_radius : bool
            If True, use the half mass radius of the halo instead of the virial radius. Default is False.
        """

        if mode =='AHF' and not self.cosmological:
            raise ValueError("AHF mode only works for cosmological simulations. Use 'dense_gas' or 'young_star' mode instead.")


        # Make a unique name given parameters so we can tell what part of the halo was considered
        self.name += '_'+ mode + '_'
        if kpc and not use_halfmass_radius:
            self.name += str(rout) + 'kpc'
        elif use_halfmass_radius:
            self.name += str(rout) + 'R1/2'
        else:
            self.name += str(rout) + 'Rvir'
        self.name += '.pickle'
        # Store the arguments so we can pass them on later
        args = locals()
        args.pop('self')
        self.halo_args = args
        self.setHalo=True

        return


    def load(self, 
             increment:int=2, 
             overwrite:bool=False, 
             verbose:bool=True):
        """
        Parameters
        ----------
        increment : int
            Number of snapshots between periodic saves of the reduced data. Default is 2.
        overwrite : bool
            If True, overwrite the existing reduced data file. Default is False.
        verbose : bool
            If True, print out the progress of the loading. Default is True.
        """

        # Check if the data has already been loaded
        if self.loaded: return

        # Check if object file has already been created if so load that first instead of creating a new one
        if not os.path.isfile(self.data_dirc+self.name) or overwrite:
            self.reduced_data = MultiSnapReducedData(self.sdir, self.snap_nums, self.gas_properties, self.gas_subsamples, self.star_properties, 
                                             self.star_subsamples,halohist_file=self.halohist_file)
            if self.setHalo:
                self.reduced_data.set_halo(**self.halo_args)
            else:
                raise Exception("No halo specified. Set halo using set_halo() to specify halo before loading snapshots.")
        else:
            with open(self.data_dirc+self.name, 'rb') as handle:
                self.reduced_data = pickle.load(handle)
            print("Reduced data already exists for the given halo setup so loading that first....")
            self.reduced_data.compare_and_update_props_and_snaps(self.snap_nums, self.gas_properties, self.gas_subsamples, self.star_properties, 
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
        self.loaded=True
        return


    def save(self):
        """
        Saves the reduced data to a pickle file. The file is saved in the directory specified by self.data_dirc.
        """

        # First create directory if needed
        if not os.path.isdir(self.data_dirc):
            os.mkdir(self.data_dirc)
            print("Directory " + self.data_dirc +  " created for reduced data file.")

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
            data = reduced_data['M_form_10Myr_'+subsample]/1E7
        elif prop == 'sfr_100Myr':
            data = reduced_data['M_form_100Myr_'+subsample]/1E8
        elif prop == 'ssfr':
            data = (reduced_data['M_form_10Myr_'+subsample]/0.01)/reduced_data['M_star_'+subsample] # Gyr^-1
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
                print(prop," with subsample",subsample, "is not in the dataset with given statistic.")
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



class MultiSnapReducedData(object):
    """
    This class reduces and stores gas and star particle data from snapshots. 
    MultiSnapDataIO deals with the initialization and IO of this class and saves it
    to disk for easy access and modification.
    """

    def __init__(self, 
                 sdir:str, 
                 snap_nums:list, 
                 gas_props:list, 
                 gas_subsamples:list, 
                 star_props:list, 
                 star_subsamples:list,
                 halohist_file:str):
        """
        Parameters
        ----------
        sdir : str
            Directory where the simulation snapshots are stored.
        snap_nums : list
            List of snapshot numbers to reduce.
        gas_props : list 
            List of gas properties to reduce. If None, defaults to commonly used values, determined based on simulation flags (mainly for dust).
        gas_subsamples : list
            List of gas subsamples to reduce. If None, defaults to ['all','cold','warm','hot','coronal','neutral','molecular','ionized'].
        star_props : list
            List of star properties to reduce. If None, defaults to ['M_star','M_form_10Myr','M_form_100Myr','r1/2_stars'].
        star_subsamples : list
            List of star subsamples to reduce. If None, defaults to ['all'].
        halohist_file : str
            Path to the AHF halo history file used to determine where the galaxy center is for each snapshot. 
            This is necessary for simulations where the main halo between snapshots such as dwarf-mass galaxies and at high-z.
            If None, defaults to determining the rough galactic center from dense gas/young star locations.
        """
        

        self.sdir = sdir
        self.snaps = np.sort(snap_nums)
        self.num_snaps = len(self.snaps)
        self.snap_loaded = np.zeros(self.num_snaps,dtype=bool)

        # First check that all these snaps exist
        missing_file = False
        for snap_num in snap_nums:
            if not check_snap_exist(sdir,snap_num):
                missing_file = True
                break
        if missing_file: 
            print(f"Snapshot {snap_num} is missing. Check the snap directory and numbers given.")
            return

        # Load first snap to get cosmological parameters
        sp = Snapshot(self.sdir, self.snaps[0])
        self.cosmological = sp.cosmological
        self.FIRE_ver = sp.FIRE_ver
        self.time = np.zeros(self.num_snaps)
        if self.cosmological:
            self.hubble = sp.hubble
            self.omega = sp.omega
            self.redshift = np.zeros(self.num_snaps)
            self.scale_factor = np.zeros(self.num_snaps)

        # The reduced data is stored in a dictionary with keys corresponding to each property
        # for each subsample
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

        # Arguments used for each snapshot when determining the halo and what particles are within the halo
        self.setHalo=False
        self.load_kwargs = {}
        self.set_kwargs = {}
        self.use_halfmass_radius = False

        # If an AHF halo history file is given use it to determine the halo ID of the main halo for each snapshot
        # used to determine the center of the halo and its virial radius
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
                          "Either fix the file or don't specify an AHF halo history file to assume the largest halo for every snap.")
                    return

        self.all_snaps_loaded=False
        
        return


    def compare_and_update_props_and_snaps(self, 
                                            snap_nums:list, 
                                            gas_props:list, 
                                            gas_subsamples:list, 
                                            star_props:list, 
                                            star_subsamples:list):
        """
        Compares snapshot numbers and properties stored by the object with the ones given by the user.
        If new snapshots and/or properties are specified, the object will update to include them.

        Parameters
        ----------
        snap_nums : list
            List of snapshot numbers to reduce.
        gas_props : list   
            List of gas properties to reduce.
        gas_subsamples : list
            List of gas subsamples to reduce.
        star_props : list  
            List of star properties to reduce.
        star_subsamples : list
            List of star subsamples to reduce.
        """


        # Check snapshot number and add any new snap numbers
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


    def set_halo(self, 
                 mode:str='AHF', 
                 rout:float=1, 
                 kpc:bool=False, 
                 use_halfmass_radius:bool=False):
        """
        Set the halo arguments you want when loading each snapshot to determined which particles 
        are in the galactic halo. These arguments will be used when loading each snapshot.


        Parameters
        ----------
        mode : str
            Set how the galactic halo properties will be determined. Default 'AHF' uses AHF halo files.
        rout : float
            The radius of the halo to include in kpc or Rvir. Default is 1.
        kpc : bool
            If True, rout is in kpc. If False, rout is in Rvir.
        use_halfmass_radius : bool
            If True, use the half mass radius of the halo instead of the virial radius. Default is False.
        """

        if not self.setHalo:
            self.setHalo=True
            self.load_kwargs = {'mode':mode}
            self.set_kwargs = {'rout':rout, 'kpc':kpc}
            self.use_halfmass_radius = use_halfmass_radius
            return 1
        else:
            return 0


    def load(self, 
             increment:int=5, 
             verbose:bool=True):
        """
        Loads a N incremental number of snapshots and reduces the data for each. The N number of snapshots loaded are the
        lowest Nth number snapshots which have not been loaded.

        Parameters
        ----------
        increment : int
            Number of snapshots to load at a time. Default is 5.
        verbose : bool
            If True, print out the progress of the loading. Default is True.
        """

        # Keep track of how many snapshots we have loaded
        snaps_loaded=0
        for i, snum in enumerate(self.snaps):

            # Stop loading if already loaded set increment so it can be saved
            if snaps_loaded >= increment:
                return 1
            # Skip already loaded snaps
            if self.snap_loaded[i]:
                continue

            # load in snapshot and general time data
            if verbose: print('Loading snap',snum,'...')
            start=time.time()
            sp = Snapshot(self.sdir, snum)
            if sp.cosmological:
                self.redshift[i] = sp.redshift
                self.scale_factor[i] = sp.scale_factor
                self.time[i] = quick_lookback_time(sp.time, sp=sp)
            else:
                self.time[i] = sp.time

            # Get the specified halo object from the snapshot which is used to get particles in the halo
            if self.setHalo:
                self.load_kwargs['id'] = self.haloIDs[i]
                if verbose and self.cosmological: print("For snap %i using Halo ID %i"%(snum,self.haloIDs[i]))
                gal = sp.loadhalo(**self.load_kwargs)
                if self.use_halfmass_radius:
                    half_mass_radius = calc_utils.calc_half_mass_radius(0, gal, within_radius=None, geometry='spherical', rvir_frac=0.5)
                    gal.set_zoom(rout=3.*half_mass_radius, kpc=True)
                else:
                    gal.set_zoom(**self.set_kwargs)
            else:
                raise Exception("No halo specified. Set halo using set_halo() to specify halo before loading snapshots.")


            # Calculate each gas particle property for each subsample
            ptype=0
            for subsample in self.gas_subsamples:
                # Get subsample mask
                sample_mask = calc_utils.get_particle_mask(ptype,gal,mask_criteria=subsample)
                for property in self.gas_props:
                    data_key = property + '_' + subsample
                    # If no particles match the subsample mask, set the value to NaN
                    if np.all(sample_mask==False): galaxy_value = np.nan
                    elif 'r1/2' in property:
                        gas_particles = gal.loadpart(ptype)
                        if 'H2' in property: weights = gas_particles.get_property('H2_fraction')
                        elif 'neutral' in property: weights = gas_particles.get_property('H_neutral_fraction')
                        else: weights = None
                        galaxy_value = calc_utils.calc_half_mass_radius(ptype, gal, within_radius=None, 
                                                                                  geometry='spherical', rvir_frac=0.5, mass_weight = weights)
                    else:
                        P = gal.loadpart(ptype)
                        prop_vals = P.get_property(property)[sample_mask]
                        if prop_vals.ndim != 1:
                            raise ValueError(f"Property {property} array must be one-dimensional.")
                        # Galaxy-integrated masses are total masses so this is a simple sum
                        if 'M_' in property:
                            galaxy_value = np.sum(prop_vals)
                        # For all other properties we want the mass-weighted median
                        else:
                            weights = P.get_property('M')
                            weights=weights[sample_mask]
                            galaxy_value = weighted_percentile(prop_vals, percentiles=np.array([50]), weights=weights, ignore_invalid=True)
                        
                    self.data[data_key][i] = galaxy_value

            # Calculate each star particle property for each subsample
            ptype=4
            for subsample in self.star_subsamples:
                sample_mask = calc_utils.get_particle_mask(ptype,gal,mask_criteria=subsample)
                for property in self.star_props:
                    data_key = property + '_' + subsample
                    if 'r1/2' in property:
                        galaxy_value = calc_utils.calc_half_mass_radius(ptype, gal, within_radius=None, geometry='spherical', rvir_frac=0.5)
                    else:
                        P = gal.loadpart(ptype)
                        prop_vals = P.get_property(property)[sample_mask]
                        if prop_vals.ndim != 1:
                            raise ValueError(f"Property {property} array must be one-dimensional.")
                        # Galaxy-integrated masses are total masses so this is a simple sum
                        if 'M_' in property:
                            galaxy_value = np.sum(prop_vals)
                        # For all other properties we want the mass-weighted median
                        else:
                            weights = P.get_property('M')
                            weights=weights[sample_mask]
                            galaxy_value = weighted_percentile(prop_vals, percentiles=np.array([50]), weights=weights, ignore_invalid=True)
                    self.data[data_key][i] = galaxy_value

            # Track which snapshots have been loaded
            self.snap_loaded[i]=True
            snaps_loaded+=1

            end=time.time()
            if verbose: print("Took %.3f secs to load this snap:"%(end-start))

        self.all_snaps_loaded=True

        return 1