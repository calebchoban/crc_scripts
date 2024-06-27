import h5py
import numpy as np
from .. import config
from ..utils import coordinate_utils
from ..utils.snap_utils import get_snap_file_name
from ..utils.math_utils import approx_gas_temperature,get_stellar_ages,case_insen_compare

class Header:

    def __init__(self, sp):

        # no snapshot, no particle info
        self.sp = sp
        self.k = -1 if sp.k==-1 else 0

        return

    def load(self):

        if (self.k!=0): return

        sp = self.sp
        self.k = 1
        self.npart = sp.npart
        self.time = sp.time
        if sp.cosmological:
            self.scale_factor = sp.scale_factor
        else: self.scale_factor = 1
        self.redshift = sp.redshift
        self.boxsize = sp.boxsize
        self.hubble = sp.hubble

        return


class Particle:

    def __init__(self, sp, ptype):

        # no snapshot, no particle info
        self.sp = sp
        self.k = -1 if sp.k==-1 else 0
        self.ptype = ptype

        self.time = sp.time
        if sp.cosmological:
            self.scale_factor = sp.scale_factor
        else: self.scale_factor = 1
        self.redshift = sp.redshift
        self.boxsize = sp.boxsize
        self.hubble = sp.hubble

        # Used to make sure particles aren't orientated twice
        self.orientated = False

        # To be used for centering if wanted
        self.center_position = None
        self.center_velocity = None
        self.principal_axes_vectors = None
        self.principal_axes_ratios = None

        # Where all the particle data will be stored
        self.data = {}

        return


    def load(self):

        if (self.k!=0): return

        # class basic info
        sp = self.sp

        ptype = self.ptype
        npart = sp.npart[ptype]
        self.npart = npart

        # no particle in this ptype
        if (npart==0): return

        # A dictionary of particle attributes stored in snapshots. Keys are the attribute names in given snap 
        # and the values are what I will call them. Note this is not a complete list and some attributes house
        # multiple data structures and so have multiple value names
        property_dict = {
            # all particles ----------
            'ParticleIDs': 'id',  # indexing starts at 0
            'Coordinates': 'position',  # [kpc]
            'Velocities': 'velocity',  # [km/s physics/peculiar]
            'Masses': 'mass',  # [M_sun]
            'Potential': 'potential',  # [km^2 / s^2]
            # grav acceleration for dark matter and stars, grav + hydro acceleration for gas
            'Acceleration': 'acceleration',  # [km/s / Gyr]
            # gas ----------
            'InternalEnergy': 'temperature',  # [K] (converted from stored internal energy)
            'Density': 'density',  # [M_sun / kpc^3]
            'Pressure': 'pressure',  # [M_sun / kpc / Gyr^2]
            # 'SoundSpeed': 'sound_speed',  # [km/s]
            'SmoothingLength': 'size',  # radius of kernel (smoothing length) [kpc]
            'ElectronAbundance': 'electron_fraction',  # average number of free electrons per proton
            # fraction of hydrogen that is neutral (not ionized)
            'NeutralHydrogenAbundance': 'H_neutral_fraction',
            'MolecularMassFraction': 'H2_fraction',  # fraction of mass that is molecular
            'CoolingRate': 'cool_rate', # [M_sun / yr]
            'HeatingRate': 'heat_rate', # [M_sun / yr]
            'NetHeatingRateQ': 'net_heat_Q', # [M_sun / yr]
            'HydroHeatingRate': 'hydro_heat_rate', # [M_sun / yr]
            'MetalCoolingRate': 'metal_cool_rate', # [M_sun / yr]
            'DustCoolingRate': 'dust_cool_rate', # [M_sun / yr]
            'PElecHeatingRate': 'photo_heat_rate', # [M_sun / yr]
            'StarFormationRate': 'sfr',  # [M_sun / yr]
            # 'MagneticField': 'magnetic_field',  # 3-D magnetic field [Gauss]
            # star/gas ----------
            # id.generation and id.child initialized to 0 for all gas cells
            # each time a gas cell splits into two:
            #   'self' particle retains id.child, other particle gets id.child += 2 ^ id.generation
            #   both particles get id.generation += 1
            # allows maximum of 30 generations, then restarts at 0
            #   thus, particles with id.child > 2^30 are not unique anymore
            'ParticleChildIDsNumber': 'id_child',
            'ParticleIDGenerationNumber': 'id_generation',
            # mass fraction of individual elements ----------
            # 0 = all metals (everything not H, He)
            # 1 = He, 2 = C, 3 = N, 4 = O, 5 = Ne, 6 = Mg, 7 = Si, 8 = S, 9 = Ca, 10 = Fe
            # currently this only loads in 0-10 so no r-process or age-tracers
            'Metallicity': 'Z',  # linear mass fraction
            # stars ----------
            # 'time' when star particle formed
            # for cosmological runs, = scale-factor; for non-cosmological runs, = time [Gyr/h]
            'StellarFormationTime': 'sft', # (age [Gyr] is also calculated from this)
            # dust ----------
            # some dust headers store multiple properties
            # mass fraction of individual elements locked in dust
            # 0 = all metals (everything not H, He)
            # 1 = He, 2 = C, 3 = N, 4 = O, 5 = Ne, 6 = Mg, 7 = Si, 8 = S, 9 = Ca, 10 = Fe
            # Last 4 elements are the fraction of the dust mass from dust creation sources
            # 11 = accretion, 12 = SNe Ia, 13 = Sne II, 14 = AGB
            'DustMetallicity': ['dust_Z','dust_source'],
            # mass fraction of dust species, length depends on dust model 
            # 0 = silicates, 1 = carbonaceous, 2 = SiC, 3 = free-flying iron, 4 = O reservoir, 5 = iron inclusions
            'DustSpeciesAbundance': 'dust_spec',
            # dense H2 gas mass fraction and fraction of gas=phase C locked in CO
            'DustMolecularSpeciesFractions': ['dense_H2_frac','C_in_CO'],
            # clumping factor (C_2) from unresolved turbulent mixing
            'ClumpingFactor': 'clumping_factor',
            # temperature of dust grains from FIRE-3
            'Dust_Temperature': 'dust_temp',
            # parameters for grain size bins with linear slopes. Each species has N grain size bins specified at sim runtime
            # total number of dust grains in grain size bin
            'DustBinNumbers': 'grain_bin_num',
            # slope of grain size bin
            'DustBinSlope': 'grain_bin_slope'
        }

        # First initialize all the arrays
        snapfile = get_snap_file_name(sp.sdir,sp.snum,sp.nsnap,0)
        f = h5py.File(snapfile, 'r')
        snap_ptype = f['PartType%d'%ptype]
        for header_key, prop_key in property_dict.items():
            # If not in the attribute keys that means it doesnt exist and we can skip it
            if header_key not in snap_ptype:
                continue
            # Get shape and dtype of each data structure so we can intialize them
            data_shape = snap_ptype[header_key].shape
            data_dtype = snap_ptype[header_key].dtype
            
            if len(data_shape) == 1:
                prop_shape = self.npart
            elif len(data_shape) == 2:
                prop_shape = [self.npart, data_shape[1]]
                if header_key == 'DustMetallicity':
                    prop_shape = [[self.npart, 11],[self.npart, 4]]
                if header_key == 'DustMolecularSpeciesFractions':
                    prop_shape = [[self.npart],[self.npart]]
            elif len(data_shape) == 2:
                prop_shape = [self.npart, data_shape[1], data_shape[2]]
            # Initialize everything to -1
            if type(prop_key) is list:
                for i,prop in enumerate(prop_key):
                    self.data[prop] = np.zeros(prop_shape[i],dtype=data_dtype)-1
            else:
                self.data[prop_key] = np.zeros(prop_shape,dtype=data_dtype)-1
        f.close()

        # Read in particle data from snapshot or multiple subsnapshots if snapshot is spread across multiple files
        # Indices for particle data used for subsnaps
        part_index_lo = 0; part_index_hi = 0;
        for i in range(sp.nsnap):
            snapfile = get_snap_file_name(sp.sdir,sp.snum,sp.nsnap,i)
            f = h5py.File(snapfile, 'r')
            npart_this = f['Header'].attrs['NumPart_ThisFile'][ptype]
            if npart_this <=0: continue  # if there are no particles of this type in the snap no reason to continue
            part_index_hi = part_index_lo + npart_this
            snap_ptype = f['PartType%d'%ptype]

            # step through header keys in snapshot
            for ptype_key in snap_ptype:
                # only keep the ones we want
                if ptype_key in property_dict:
                    data_key = property_dict[ptype_key]
                    if len(snap_ptype[ptype_key].shape) == 1:
                        self.data[data_key][part_index_lo:part_index_hi] = snap_ptype[ptype_key]
                    elif len(snap_ptype[ptype_key].shape) == 2:
                        # deal with properties that house multiple data structures
                        if type(data_key) is list:
                            data_index_lo = 0; data_index_hi = 0;
                            for key in data_key:
                                if len(self.data[key].shape)==1:
                                    data_index_hi = data_index_lo + 1
                                    # Get rid of pesky extra dimension
                                    self.data[key][part_index_lo:part_index_hi] = snap_ptype[ptype_key][:,data_index_lo:data_index_hi].flatten()
                                else:
                                    data_index_hi = data_index_lo + self.data[key].shape[1]
                                    self.data[key][part_index_lo:part_index_hi] = snap_ptype[ptype_key][:,data_index_lo:data_index_hi]
                                data_index_lo = data_index_hi
                        else:
                            self.data[data_key][part_index_lo:part_index_hi] = snap_ptype[ptype_key]
                    elif len(snap_ptype[ptype_key].shape) == 3:
                        self.data[data_key][part_index_lo:part_index_hi] = snap_ptype[ptype_key]
            f.close()
            part_index_lo = part_index_hi

        # Now convert from code units to physical units (kpc, M_sun, Gyr)
        hubble = sp.hubble # non-cosmological simulations have hubble units!
        ascale = sp.time if (sp.cosmological) else 1.0

        mass_conversion = self.sp.UnitMass_in_Msolar / hubble  # multiple by this for [M_sun]
        length_conversion = ascale / hubble  # multiply for [kpc physical]
        time_conversion = 1 / hubble  # multiply by this for [Gyr]
        velocity_conversion = np.sqrt(ascale) # multiply for km/s
        density_conversion = self.sp.UnitDensity_in_CGS * 1/hubble/(length_conversion**3)
        internal_energy_conversion = self.sp.UnitVelocity_In_CGS**2
        
        if 'position' in self.data:
            self.data['position'] *= length_conversion
        if 'velocity' in self.data:
            self.data['velocity'] *= velocity_conversion
        if 'acceleration' in self.data:
            # convert to [km/s / Gyr]
            self.data['acceleration'] *= hubble
        if 'mass' in self.data:
            self.data['mass'] *= mass_conversion
        if 'size' in self.data:
            self.data['size'] *= length_conversion
            # size in snapshot is full extent of the kernal (radius of compact support)
            # convert to mean interparticle spacing = volume^(1/3)
            self.data['size'] *= (np.pi / 3) ** (1 / 3) / 2  # 0.5077
        if 'density' in self.data:
            self.data['density'] *= density_conversion
        if 'temperature' in self.data:
            # Get temperature from interal energy, make sure to convert it to physical energy first
            self.data['temperature'] = approx_gas_temperature(self.data['temperature']*internal_energy_conversion,self.data['electron_fraction'])
        if 'potential' in self.data:
            # convert to [km^2 / s^2 physical]
            self.data['potential'] *= ascale
        if 'sft' in self.data:
            sft = self.data['sft']*time_conversion if not sp.cosmological else self.data['sft']
            self.data['age'] = get_stellar_ages(sft, sp=sp)
        if 'pressure' in self.data:
             # convert to [M_sun / kpc / Gyr^2]
            self.data['pressure'] *= (mass_conversion / length_conversion / time_conversion**2)

        # Special dust conversions below
        if (sp.Flag_DustSpecies) and (sp.Flag_DustSpecies<1):
            # For dust evo model which does not tracak specific dust species, 'Elemental' in Choban et al. (2022)
            # Build carbonaceous and silicates from their respective elements for the
            self.data['dust_spec'] = np.zero(npart,2)-1
            self.data['dust_spec'][:,0] = self.data['dust_Z'][:,4]+self.data['dust_Z'][:,6]+self.data['dust_Z'][:,7]+self.data['dust_Z'][:,10]
            self.data['dust_spec'][:,1] = self.data['dust_Z'][:,2]

        if 'grain_bin_num' in self.data and 'grain_bin_slope' in self.data:
            # Need to convert to actual linear values. 
            # Grain numbers are in log10 form 
            # Grain slopes are in log10 form but the sign represent if it's positive or negative
            # THIS WILL PROBABLY CHANGE TO BE MASS INSTEAD OF SLOPE FOR FINAL RUNS
            self.data['grain_bin_num'] = np.power(10,self.data['grain_bin_num'].reshape((npart, sp.Flag_DustSpecies,sp.Flag_GrainSizeBins)),dtype='double')
            self.data['grain_bin_slope'] = self.data['grain_bin_slope'].reshape((npart, sp.Flag_DustSpecies,sp.Flag_GrainSizeBins))
            self.data['grain_bin_slope'] = np.sign(self.data['grain_bin_slope'])*np.power(10,np.abs(self.data['grain_bin_slope']),dtype='double') / (config.cm_to_um*config.cm_to_um)
            # No dust grains are denoted by -1 in snapshots
            no_dust =  self.data['grain_bin_num']==0.1 
            self.data['grain_bin_num'][no_dust] = 0; self.data['grain_bin_slope'][no_dust] = 0;
            # Since dn/da is normalized to the dust mass in the code, need to multiply by h factor
            self.data['grain_bin_num'] *= hubble
            self.data['grain_bin_slope'] *= hubble
        
        self.k = 1
        return
        

    # Reduce the particle data to only the masked particles
    def mask(self, mask):

        for prop in self.data.keys():
            self.data[prop] = self.data[prop][mask]

        self.npart = len(self.data['mass'])
        
        return

    def orientate(self, center_pos=None, center_vel=None, principal_vec=None):

        if self.orientated: return

        if center_vel is not None and center_pos is not None:
            # convert to be relative to galaxy center [km / s]
            self.data['velocity'] = coordinate_utils.get_velocity_differences(
                        self.data['velocity'], center_vel)
            if principal_vec is not None:
                # convert to be aligned with galaxy principal axes
                self.data['velocity'] = coordinate_utils.get_coordinates_rotated(self.data['velocity'], principal_vec)

        if center_pos is not None:
            # convert to be relative to galaxy center [kpc physical]
            self.data['position'] = coordinate_utils.get_distances(self.data['position'], center_pos,self.boxsize)
            if principal_vec is not None:
                # convert to be aligned with galaxy principal axes
                self.data['position'] = coordinate_utils.get_coordinates_rotated(
                    self.data['position'], principal_vec)

        self.orientated=1

        return


    # Gets derived properties from particle data
    def get_property(self, property):

        data = self.data
        # Nothing to do here if there are no particles
        numpart = len(data['mass'])
        if numpart == 0:
            return np.zeros(0)
        
        # Default to -1 for unsupported properties
        prop_data = np.full(numpart,-1,dtype=float)

        # First properties that all particle types have
        # If it corresponds to a data key just pull that
        # Except for Z since you usually only want Z_total
        if case_insen_compare(property,data.keys()) and not case_insen_compare(property,'Z'):
            # Make case insensitive
            casefold_keys = [item.casefold() for item in data.keys()]
            idx = casefold_keys.index(property.casefold())
            prop_data = data[list(data.keys())[idx]]
        elif case_insen_compare(property,['M','mass','M_gas','M_star','M_dm','M_stellar']):
            prop_data = data['mass']
        elif case_insen_compare(property,['coords','position']):
            prop_data = data['position']
        elif case_insen_compare(property,['r_spherical','r','radius']):
            prop_data = np.sum(data['position'] ** 2, axis=1) ** 0.5
        elif case_insen_compare(property,'r_cylindrical'):
            prop_data = np.sum(data['position'][:, :2] ** 2, axis=1) ** 0.5
        elif case_insen_compare(property,['v','vel','velocity']):
            prop_data = data['velocity']
        elif self.ptype == 0:
            if case_insen_compare(property,['h','size','scale_length']):
                prop_data = data['size']
            elif case_insen_compare(property,'M_gas_neutral'):
                prop_data = data['mass']*data['H_neutral_fraction']
            elif case_insen_compare(property,['m_mol','m_h2']) and 'H2_fraction' in data:
                prop_data =  data['mass']*data['H_neutral_fraction']*data['H2_fraction']
            elif case_insen_compare(property,'fH2') and 'H2_fraction' in data:
                prop_data = data['H2_fraction']
                prop_data[prop_data>1] = 1
            elif case_insen_compare(property,'M_gas_ionized'):
                prop_data = data['mass']*(1-data['H_neutral_fraction'])
            elif case_insen_compare(property,'M_metals'):
                prop_data = data['mass']*data['Z'][:,0]
            elif case_insen_compare(property,'nH'):
                prop_data = data['density'] * (1. - (data['Z'][:,0]+data['Z'][:,1])) / config.H_MASS
            elif case_insen_compare(property,['nh','f_nh','fnh','f_neutral']):
                prop_data = data['H_neutral_fraction']
            elif case_insen_compare(property,'nH_neutral'):
                prop_data = (data['density'] * (1. - (data['Z'][:,0]+data['Z'][:,1])) / config.H_MASS)*data['H_neutral_fraction']
            elif case_insen_compare(property,['T','temperature']):
                prop_data = data['temperature']
            elif case_insen_compare(property,'Z'):
                SOLAR_Z = self.sp.solar_abundances[0]
                prop_data = data['Z'][:,0]/SOLAR_Z
            elif case_insen_compare(property,'Z_all'):
                prop_data = data['Z']
            elif case_insen_compare(property,'Z_O'):
                prop_data = data['Z'][:,4]/self.sp.solar_abundances[4]
            elif case_insen_compare(property,'Z_C'):
                prop_data = data['Z'][:,2]/self.sp.solar_abundances[2]
            elif case_insen_compare(property,'Z_Mg'):
                prop_data = data['Z'][:,6]/self.sp.solar_abundances[6]
            elif case_insen_compare(property,'Z_Si'):
                prop_data = data['Z'][:,7]/self.sp.solar_abundances[7]
            elif case_insen_compare(property,'Z_Fe'):
                prop_data = data['Z'][:,10]/self.sp.solar_abundances[10]
            elif case_insen_compare(property,'O/H'):
                O = data['Z'][:,4]/config.ATOMIC_MASS[4]; H = (1-(data['Z'][:,0]+data['Z'][:,1]))/config.ATOMIC_MASS[0]
                prop_data = 12+np.log10(O/H)
            elif case_insen_compare(property,'O/H_offset'):
                offset=0.2 # This is roughly difference in O/H_solar between AG89 (8.93) and Asplund+09 protosolar(8.73). Ma+16 finds FIRE gives 9.00.
                O = data['Z'][:,4]/config.ATOMIC_MASS[4]; H = (1-(data['Z'][:,0]+data['Z'][:,1]))/config.ATOMIC_MASS[0]
                prop_data = 12+np.log10(O/H)-offset
            elif case_insen_compare(property,'C/H'):
                C = data['Z'][:,2]/config.ATOMIC_MASS[2]; H = (1-(data['Z'][:,0]+data['Z'][:,1]))/config.ATOMIC_MASS[0]
                prop_data = 12+np.log10(C/H)
            elif case_insen_compare(property,'Mg/H'):
                Mg = data['Z'][:,6]/config.ATOMIC_MASS[6]; H = (1-(data['Z'][:,0]+data['Z'][:,1]))/config.ATOMIC_MASS[0]
                prop_data = 12+np.log10(Mg/H)
            elif case_insen_compare(property,'Si/H'):
                Si = data['Z'][:,7]/config.ATOMIC_MASS[7]; H = (1-(data['Z'][:,0]+data['Z'][:,1]))/config.ATOMIC_MASS[0]
                prop_data = 12+np.log10(Si/H)
            elif case_insen_compare(property,'Fe/H'):
                Fe = data['Z'][:,10]/config.ATOMIC_MASS[10]; H = (1-(data['Z'][:,0]+data['Z'][:,1]))/config.ATOMIC_MASS[0]
                prop_data = 12+np.log10(Fe/H)
            elif case_insen_compare(property,'f_cold'):
                prop_data = np.sum(data['mass'][data['temperature']<=1E3])/np.sum(data['mass'])
            elif case_insen_compare(property,'f_warm'):
                prop_data = np.sum(data['mass'][(data['temperature']<1E4) & (data['temperature']>=1E3)])/np.sum(data['mass'])
            elif case_insen_compare(property,'f_hot'):
                prop_data = np.sum(data['mass'][data['temperature']>=1E4])/np.sum(data['mass'])
            elif self.sp.Flag_DustSpecies:
                if case_insen_compare(property,'M_dust'):
                    prop_data = data['dust_Z'][:,0]*data['mass']
                elif case_insen_compare(property,'M_sil'):
                    prop_data = data['dust_spec'][:,0]*data['mass']
                elif case_insen_compare(property,'M_carb'):
                    prop_data = data['dust_spec'][:,1]*data['mass']
                elif case_insen_compare(property,'M_SiC'):
                    if self.sp.Flag_DustSpecies>2:
                        prop_data = data['dust_spec'][:,2]*data['mass']
                elif case_insen_compare(property,'M_iron'):
                    if self.sp.Flag_DustSpecies>5:
                        prop_data = (data['dust_spec'][:,3]+data['dust_spec'][:,5])*data['mass']
                    elif self.sp.Flag_DustSpecies>2:
                        prop_data = data['dust_spec'][:,3]*data['mass']
                elif case_insen_compare(property,'M_ORes'):
                    if self.sp.Flag_DustSpecies>=5:
                        prop_data = data['dust_spec'][:,4]*data['mass']
                elif case_insen_compare(property,'M_sil+'):
                    prop_data = (data['dust_spec'][:,0]+np.sum(data['dust_spec'][:,2:],axis=1))*data['mass']
                elif case_insen_compare(property,'dz_sil'):
                    prop_data = data['dust_spec'][:,0]/data['dust_Z'][:,0]
                elif case_insen_compare(property,'dz_carb'):
                    prop_data = data['dust_spec'][:,1]/data['dust_Z'][:,0]
                elif case_insen_compare(property,'dz_SiC'):
                    if self.sp.Flag_DustSpecies>2:
                        prop_data = data['dust_spec'][:,2]/data['dust_Z'][:,0]
                elif case_insen_compare(property,'dz_iron'):
                    if self.sp.Flag_DustSpecies>5:
                        prop_data = (data['dust_spec'][:,3]+data['dust_spec'][:,5])/data['dust_Z'][:,0]
                    elif self.sp.Flag_DustSpecies>2:
                        prop_data = data['dust_spec'][:,3]/data['dust_Z'][:,0]
                elif case_insen_compare(property,'dz_ORes'):
                    if self.sp.Flag_DustSpecies>=5:
                        prop_data = data['dust_spec'][:,4]/data['dust_Z'][:,0]
                elif case_insen_compare(property,'M_acc_dust'):
                    prop_data = data['dust_source'][:,0]*data['dust_Z'][:,0]*data['mass']
                elif case_insen_compare(property,'M_SNeIa_dust'):
                    prop_data = data['dust_source'][:,1]*data['dust_Z'][:,0]*data['mass']
                elif case_insen_compare(property,'M_SNeII_dust'):
                    prop_data = data['dust_source'][:,2]*data['dust_Z'][:,0]*data['mass']
                elif case_insen_compare(property,'M_AGB_dust'):
                    prop_data = data['dust_source'][:,3]*data['dust_Z'][:,0]*data['mass']
                elif case_insen_compare(property,'dz_acc'):
                    prop_data = data['dust_source'][:,0]
                elif case_insen_compare(property,'dz_SNeIa'):
                    prop_data = data['dust_source'][:,1]
                elif case_insen_compare(property,'dz_SNeII'):
                    prop_data = data['dust_source'][:,2]
                elif case_insen_compare(property,'dz_AGB'):
                    prop_data = data['dust_source'][:,3]
                elif case_insen_compare(property,'fdense'):
                    prop_data = data['dense_H2_frac']
                    prop_data[prop_data > 1] = 1
                elif case_insen_compare(property,['C_in_CO','CinCO']):
                    prop_data = data['C_in_CO']/data['Z'][:,2]
                elif case_insen_compare(property,'Z_O_gas'):
                    prop_data = (data['Z'][:,4]-data['dust_Z'][:,4])/self.sp.solar_abundances[4]
                elif case_insen_compare(property,'Z_C_gas'):
                    prop_data = (data['Z'][:,2]-data['dust_Z'][:,2])/self.sp.solar_abundances[2]
                elif case_insen_compare(property,'Z_Mg_gas'):
                    prop_data = (data['Z'][:,6]-data['dust_Z'][:,6])/self.sp.solar_abundances[6]
                elif case_insen_compare(property,'Z_Si_gas'):
                    prop_data = (data['Z'][:,7]-data['dust_Z'][:,7])/self.sp.solar_abundances[7]
                elif case_insen_compare(property,'Z_Fe_gas'):
                    prop_data = (data['Z'][:,10]-data['dust_Z'][:,10])/self.sp.solar_abundances[10]
                elif case_insen_compare(property,'O/H_gas_offset'):
                    offset=0.2 # This is roughly difference in O/H_solar between AG89 (8.93) and Asplund+09 (8.69). Ma+16 finds FIRE gives 9.00.
                    O = (data['Z'][:,4]-data['dust_Z'][:,4])/config.ATOMIC_MASS[4]; H = (1-(data['Z'][:,0]+data['Z'][:,1]))/config.ATOMIC_MASS[0]
                    prop_data = 12+np.log10(O/H)-offset
                elif case_insen_compare(property,'O/H_gas'):
                    O = (data['Z'][:,4]-data['dust_Z'][:,4])/config.ATOMIC_MASS[4]; H = (1-(data['Z'][:,0]+data['Z'][:,1]))/config.ATOMIC_MASS[0]
                    prop_data = 12+np.log10(O/H)
                elif case_insen_compare(property,'C/H_gas'):
                    C = (data['Z'][:,2]-data['dust_Z'][:,2])/config.ATOMIC_MASS[2]; H = (1-(data['Z'][:,0]+data['Z'][:,1]))/config.ATOMIC_MASS[0]
                    prop_data = 12+np.log10(C/H)
                elif case_insen_compare(property,'Mg/H_gas'):
                    Mg = (data['Z'][:,6]-data['dust_Z'][:,6])/config.ATOMIC_MASS[6]; H = (1-(data['Z'][:,0]+data['Z'][:,1]))/config.ATOMIC_MASS[0]
                    prop_data = 12+np.log10(Mg/H)
                elif case_insen_compare(property,'Si/H_gas'):
                    Si = (data['Z'][:,7]-data['dust_Z'][:,7])/config.ATOMIC_MASS[7]; H = (1-(data['Z'][:,0]+data['Z'][:,1]))/config.ATOMIC_MASS[0]
                    prop_data = 12+np.log10(Si/H)
                elif case_insen_compare(property,'Fe/H_gas'):
                    Fe = (data['Z'][:,10]-data['dust_Z'][:,10])/config.ATOMIC_MASS[10]; H = (1-(data['Z'][:,0]+data['Z'][:,1]))/config.ATOMIC_MASS[0]
                    prop_data = 12+np.log10(Fe/H)
                elif case_insen_compare(property,'Si/C'):
                    prop_data = data['dust_spec'][:,0]/data['dust_spec'][:,1]
                elif case_insen_compare(property,'D/Z'):
                    prop_data = data['dust_Z'][:,0]/data['Z'][:,0]
                    prop_data[prop_data > 1] = 1.
                elif 'depletion' in property:
                    elem = property.split('_')[0]
                    if elem not in config.ELEMENTS:
                        print('%s is not a valid element to calculate depletion for. Valid elements are'%elem)
                        print(config.ELEMENTS)
                    elem_indx = config.ELEMENTS.index(elem)
                    prop_data =  data['dust_Z'][:,elem_indx]/data['Z'][:,elem_indx]
                    prop_data[prop_data > 1] = 1.
                elif case_insen_compare(property,['T_dust']):
                    prop_data = data['dust_temp']
                elif self.sp.Flag_GrainSizeBins:
                    if case_insen_compare(property,'grain_bin_num'):
                        prop_data = data['grain_bin_num'];
                    elif case_insen_compare(property,'grain_bin_slope'):
                        prop_data = data['grain_bin_slope'];
                    # Note all dust grain size distribution data is normalized by the total grain number
                    elif case_insen_compare(property,'dn/da'):
                        # Gives normalized dn/da at the center of the grain bins 
                        N_total = np.sum(data['grain_bin_num'],axis=2)
                        prop_data = data['grain_bin_num']/np.ediff1d(self.sp.Grain_Bin_Edges)/N_total;
                    elif case_insen_compare(property,'dm/da'):
                        # Gives a^4 dn/da at the center of the grain bins (note this is in the usual observer convention of mass probability density per log a dm/dloga where the extra factor of a comes from the 1/loga)
                        N_total = np.sum(data['grain_bin_num'],axis=2)
                        prop_data = np.power(self.sp.Grain_Bin_Centers,4)*data['grain_bin_num']/np.ediff1d(self.sp.Grain_Bin_Edges)/N_total;
                    elif case_insen_compare(property,'sil_dn/da'):
                        N_total = np.sum(data['grain_bin_num'],axis=2)[:,0,np.newaxis]
                        prop_data = data['grain_bin_num'][:,0,:]/np.ediff1d(self.sp.Grain_Bin_Edges)/N_total;
                    elif case_insen_compare(property,'sil_dm/da'):
                        N_total = np.sum(data['grain_bin_num'],axis=2)[:,0,np.newaxis]
                        prop_data = np.power(self.sp.Grain_Bin_Centers,4)*data['grain_bin_num'][:,0,:]/np.ediff1d(self.sp.Grain_Bin_Edges)/N_total;
                    elif case_insen_compare(property,'carb_dn/da'):
                        N_total = np.sum(data['grain_bin_num'],axis=2)[:,1,np.newaxis]
                        prop_data = data['grain_bin_num'][:,1,:]/np.ediff1d(self.sp.Grain_Bin_Edges)/N_total;
                    elif case_insen_compare(property,'carb_dm/da'):
                        N_total = np.sum(data['grain_bin_num'],axis=2)[:,1,np.newaxis]
                        prop_data = np.power(self.sp.Grain_Bin_Centers,4)*data['grain_bin_num'][:,1,:]/np.ediff1d(self.sp.Grain_Bin_Edges)/N_total;
                    elif case_insen_compare(property,'SiC_dn/da'):
                        N_total = np.sum(data['grain_bin_num'],axis=2)[:,2,np.newaxis]
                        prop_data = data['grain_bin_num'][:,2,:]/np.ediff1d(self.sp.Grain_Bin_Edges)/N_total;
                    elif case_insen_compare(property,'SiC_dm/da'):
                        N_total = np.sum(data['grain_bin_num'],axis=2)[:,2,np.newaxis]
                        prop_data = np.power(self.sp.Grain_Bin_Centers,4)*data['grain_bin_num'][:,2,:]/np.ediff1d(self.sp.Grain_Bin_Edges)/N_total;
                    elif case_insen_compare(property,'iron_dn/da'):
                        N_total = np.sum(data['grain_bin_num'],axis=2)[:,3,np.newaxis]
                        prop_data = data['grain_bin_num'][:,3,:]/np.ediff1d(self.sp.Grain_Bin_Edges)/N_total;
                    elif case_insen_compare(property,'iron_dm/da'):
                        N_total = np.sum(data['grain_bin_num'],axis=2)[:,3,np.newaxis]
                        prop_data = np.power(self.sp.Grain_Bin_Centers,4)*data['grain_bin_num'][:,3,:]/np.ediff1d(self.sp.Grain_Bin_Edges)/N_total;

        elif self.ptype in [1,2,3]:
            if case_insen_compare(property,['h','size','scale_length']):
                prop_data = data['size']

        elif self.ptype==4:
            if case_insen_compare(property,['M_star_young','M_stellar_young','M_star_10Myr']):
                # Assume young stars are < 10 Myr old
                prop_data = data['mass'].copy()
                age = data['age']
                prop_data[age>0.01] = 0.
            elif case_insen_compare(property,['M_star_100Myr']):
                prop_data = data['mass'].copy()
                age = data['age']
                prop_data[age>0.1] = 0.        
            elif case_insen_compare(property,'Z'):
                SOLAR_Z = self.sp.solar_abundances[0]
                prop_data = data['Z'][:,0]/SOLAR_Z
            elif case_insen_compare(property,'Z_all'):
                prop_data = data['Z']
            elif case_insen_compare(property,'O/H'):
                O = data['Z'][:,4]/config.ATOMIC_MASS[4]; H = (1-(data['Z'][:,0]+data['Z'][:,1]))/config.ATOMIC_MASS[0]
                prop_data = 12+np.log10(O/H)
            elif case_insen_compare(property,'age'):
                prop_data = data['age']
            elif case_insen_compare(property,'sft'):
                prop_data = data['sft']

        if np.all(prop_data==-1):
            print("Property %s given to Particle with ptype %i is not supported. Returning -1 array."%(property,self.ptype))
            return prop_data
        else:
            # Make sure to return a copy of the data so the snapshot data cannot be altered
            return prop_data.copy()
