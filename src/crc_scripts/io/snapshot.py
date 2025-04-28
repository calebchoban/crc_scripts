import numpy as np
import h5py
from .. import config
from .particle import Particle
from .galaxy import Halo,Disk
from .AHF import AHF
from ..utils.snap_utils import check_snap_exist,get_snap_file_name

class Snapshot:
    """Snapshot object that reads and stores snapshot data, and implements various methods to load halo and particle data."""

    def __init__(self, sdir, snum, cosmological=True):
        """
        Construct a Snapshot instance.

        Parameters
        ----------
        sdir : string
            Directory to snapshot file (usually SIM_DIR/output/)
        snum: int
            Number of snapshot file to be loaded
        cosmological : bool, optional
            Set if snapshot is from a cosmological or non-cosmological simulation. 
            This can be determined from the header of new simulation snapshots but not old simulations.

        Returns
        -------
        Snapshot
            Snapshot instance created from given snapshot file.
        """

        self.sdir = sdir
        self.snum = snum
        self.nsnap = check_snap_exist(sdir,snum)
        self.k = -1 if self.nsnap==0 else 1
        
        if (self.k==-1): 
            raise Exception("Snapshot %s does not exist." % (sdir + "/snapshot_%03d.hdf5" % snum))

        # now, read snapshot header if it exists
        snapfile = get_snap_file_name(sdir,snum,self.nsnap,0)
        f = h5py.File(snapfile, 'r')
        if 'ComovingIntegrationOn' in f['Header'].attrs.keys():
            self.cosmological=f['Header'].attrs['ComovingIntegrationOn']
            if cosmological!=self.cosmological: 
                print("WARNING: Snapshot is either cosmological and you specified it as non-cosmological or vice versa.")
                print("Defaulting to snapshot.")
        else:
            self.cosmological=cosmological
        self.npart = f['Header'].attrs['NumPart_Total']
        self.time = f['Header'].attrs['Time']
        self.redshift = f['Header'].attrs['Redshift']
        self.boxsize = f['Header'].attrs['BoxSize']
        if self.cosmological:
            self.scale_factor = self.time
            self.omega = f['Header'].attrs.get('Omega_Matter',0)
            self.omega_lambda = f['Header'].attrs.get('Omega_Lambda',0)
            # For old FIRE snapshots
            if self.omega==0: self.omega = f['Header'].attrs.get('Omega0',0)
        else:
            self.scale_factor = 1.
        self.hubble = f['Header'].attrs['HubbleParam']
        if 'Solar_Abundances_Adopted' in f['Header'].attrs.keys():
            self.solar_abundances = f['Header'].attrs['Solar_Abundances_Adopted']
        else: # Old snaps without abundances are from FIRE-1/2 which use the AG89 abundances
            self.solar_abundances = config.AG89_ABUNDANCES
        if self.solar_abundances[0] == 0.02:
            self.FIRE_ver = 2
        else:
            self.FIRE_ver = 3

        # Get code units in CGS if available. If not just assume usual FIRE units
        if 'UnitLength_In_CGS' in f['Header'].attrs.keys():
            # Note the saved header units have a factor of 1/h
            self.UnitLength_In_CGS = f['Header'].attrs['UnitLength_In_CGS']*self.hubble
            self.UnitMass_In_CGS = f['Header'].attrs['UnitMass_In_CGS']*self.hubble
            self.UnitVelocity_In_CGS = f['Header'].attrs['UnitVelocity_In_CGS']
        else:
            self.UnitLength_In_CGS = config.UnitLength_in_cm
            self.UnitMass_In_CGS = config.UnitMass_in_g
            self.UnitVelocity_In_CGS = config.UnitVelocity_in_cm_per_s
        self.UnitMass_in_Msolar = self.UnitMass_In_CGS / config.Msolar_to_g
        self.UnitDensity_in_CGS = self.UnitMass_In_CGS / np.power(self.UnitLength_In_CGS, 3)

        self.Flag_Sfr = f['Header'].attrs['Flag_Sfr']
        self.Flag_Cooling = f['Header'].attrs['Flag_Cooling']
        self.Flag_StellarAge = f['Header'].attrs['Flag_StellarAge']
        self.Flag_Metals = f['Header'].attrs['Flag_Metals']
        # Deal with old flag tags
        if f['Header'].attrs.get('ISMDustChem_NumberOfSpecies', 0):
            self.Flag_DustSpecies = f['Header'].attrs.get('ISMDustChem_NumberOfSpecies', 0)
        elif f['Header'].attrs.get('Flag_Dust_Species',0):
            self.Flag_DustSpecies = f['Header'].attrs.get('Flag_Dust_Species',0)
        else:
            self.Flag_DustSpecies = f['Header'].attrs.get('Flag_Species', 0)
        self.Flag_Sfr = f['Header'].attrs['Flag_Sfr']

        # Check if there is info on the chemical composition of silicates since this can change
        if 'Silicates_Element_Key' in f['Header'].attrs.keys():
            self.Silicates_Element_Key = f['Header'].attrs['Silicates_Element_Key'] # The indices of the elements that make up silicates
        if 'Silicates_Element_Number' in f['Header'].attrs.keys():
            self.Silicates_Element_Number = f['Header'].attrs['Silicates_Element_Number'] # The number of each element in silicates

        if f['Header'].attrs.get('ISMDustChem_Num_Grain_Size_Bins',0):
            print("This snap has grain size bins!")
            self.Grain_Size_Max = f['Header'].attrs['ISMDustChem_Grain_Size_Max'] * config.cm_to_um
            self.Grain_Size_Min = f['Header'].attrs['ISMDustChem_Grain_Size_Min'] * config.cm_to_um
            self.Flag_GrainSizeBins = f['Header'].attrs['ISMDustChem_Num_Grain_Size_Bins']
            bin_size = np.power(10,np.log10( self.Grain_Size_Max/self.Grain_Size_Min)/self.Flag_GrainSizeBins)
            self.Grain_Bin_Edges = np.zeros(self.Flag_GrainSizeBins+1)
            self.Grain_Bin_Centers = np.zeros(self.Flag_GrainSizeBins)
            for i in range(self.Flag_GrainSizeBins+1):
                self.Grain_Bin_Edges[i] = pow(bin_size,i)*self.Grain_Size_Min
            for i in range(self.Flag_GrainSizeBins):
                self.Grain_Bin_Centers[i] = ( self.Grain_Bin_Edges[i+1]+ self.Grain_Bin_Edges[i])/2.
        else:
                self.Flag_GrainSizeBins = 0

        f.close()

        # correct for cosmological runs
        if (self.cosmological==1): self.boxsize *= (self.scale_factor/self.hubble)

        # initialize particle types
        self.gas = Particle(self, 0)
        self.DM = Particle(self, 1)
        self.disk = Particle(self, 2)
        self.bulge = Particle(self, 3)
        self.star = Particle(self, 4)
        self.BH = Particle(self, 5)
        self.part = [self.gas,self.DM,self.disk,self.bulge,self.star,self.BH]

        # initialize catalogs and/or halos
        if (self.cosmological==1):
            # we support AHF
            self.AHF = AHF(self)
            self.AHFhaloIDs = []
            self.AHFhalos = []
            self.AHFdiskIDs = []
            self.AHFdisks = []

        # non-cosmological snapshot has only one galaxy
        self.halo = Halo(self)


        return


    def loadpart(self, ptype):
        """
        Loads all particle data for the given particle type in the snapshot and returns it in a Particle object instance.

        Parameters
        ----------
        pytpe: int
            Particle type (0=gas,1=high-res DM,2/3=dummy particles,4=stars,5=sink particles) to load.

        Returns
        -------
        Particle
            Particle instance with ptype particle data.  
            
        """
        part = self.part[ptype]
        part.load()

        return part


    def loadAHF(self, hdir=None):
        """ 
        Loads and returns AHF object with Amiga Halo Finder data. 

        Parameters
        ----------
        hdir : string, optional
            Directory to corresponding AHF file for the Snapshot object. If None is given, it will look for the AHF files.

        Returns
        -------
        AHF
            AHF instance created from corresponding AHF file for the Snapshot object.        
        """

        # non-cosmological snapshots do not have AHF attribute
        if (self.cosmological==0): return None

        AHF = self.AHF
        AHF.load(hdir=hdir)

        return AHF
    

    def loadhalo(self, id=-1, mode='AHF', hdir=None):
        """ 
        Loads and returns Halo/Galaxy object for a galactic halo in the snapshot. How the center and size of the hao is determined depends on the mode. AHF uses AHF data while all other methods use particle data to get an approximate center.

        Parameters
        ----------
        id : int, optional
            ID for the galactic halo used in AHF. Usually AHF output has halos ordered from most to least masive starting at 0. id=-1 will use the most massive, high-res, halo. High-res means high-res DM particles since some halos are dominated by low-res particles we don't want.
        mode : string, optional
            Set to 'AHF' to use AHF output for getting halo data, else a general halo is computed based on particle positions.
        hdir : string, optional
             Directory to corresponding AHF file for the Snapshot object. If None is given, it will look for the AHF files.

        Returns
        -------
        Halo
            Halo instance representing galactic halo in snapshot.        
        """

        # non-cosmological or not using AHF so use the only galaxy attribute
        if self.cosmological==0 or mode!='AHF':
            hl = self.halo
            hl.load(mode)

            return hl
        
        # cosmological, use AHF
        if mode=='AHF':
            id = self.AHF.get_valid_halo_id(id, hdir=hdir)
            if id<0: return
        
            if id in self.AHFhaloIDs:
                index = self.AHFhaloIDs.index(id)
                hl = self.AHFhalos[index]
            else:
                hl = Halo(self, id=id)
                self.AHFhaloIDs.append(id)
                self.AHFhalos.append(hl)

        hl.load(mode)
        return hl


    def loaddisk(self, id=-1, mode='AHF', hdir=None, rmax=20, height=5):
        """ 
        Loads and returns Halo/Disk object for a galactic disk in the snapshot. How the center is determined depends on the mode. AHF uses AHF data while all other methods use particle data to get an approximate center. The normal vector of the disk is deteremined from particle data. Warning that this won't work well if the galaxy isn't a disk.

        Parameters
        ----------
        id : int, optional
            ID for the galactic halo used in AHF. Usually AHF output has halos ordered from most to least masive starting at 0. id=-1 will use the most massive, high-res, halo. High-res means high-res DM particles since some halos are dominated by low-res particles we don't want.
        mode : string, optional
            Set to 'AHF' to use AHF output for getting halo data, else a general halo is computed based on particle positions.
        hdir : string, optional
            Directory to corresponding AHF file for the Snapshot object. If None is given, it will look for the AHF files.
        rmax : int, optional
            Maximum radius of galactic disk in kpc.
        height : int, optional
            Maximum height of galactic disk in kpc.

        Returns
        -------
        Halo/Disk
            Halo/Disk instance representing galactic disk in snapshot.        
        """
       
        # non-cosmological or not using AHF so use the only galaxy attribute
        if self.cosmological==0 or mode!='AHF':
            disk = Disk(self, id=id,rmax=rmax,height=height)
            disk.load()

            return disk
        
        # cosmological, use AHF
        if (mode=='AHF'):
            id = self.AHF.get_valid_halo_id(id, hdir=hdir)
            if (id<0): return
        
            if id in self.AHFdiskIDs:
                index = self.AHFdiskIDs.index(id)
                disk = self.AHFdisks[index]
            else:
                disk = Disk(self, id=id,rmax=rmax,height=height)
                self.AHFdiskIDs.append(id)
                self.AHFdisks.append(disk)
            disk.load()

        else:
            disk = Disk(self, id=id,rmax=rmax,height=height)
    
                    
        disk.load(mode)
    
        return disk