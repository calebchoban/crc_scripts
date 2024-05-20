import numpy as np
import h5py
import os
from .. import config
from .particle import Header,Particle
from .galaxy import Halo,Disk
from .AHF import AHF
from ..utils import math_utils
from ..utils.snap_utils import check_snap_exist,get_snap_file_name

class Snapshot:

    def __init__(self, sdir, snum, cosmological=1):

        self.sdir = sdir
        self.snum = snum
        self.cosmological = cosmological
        self.nsnap = check_snap_exist(sdir,snum)
        self.k = -1 if self.nsnap==0 else 1

        # now, read snapshot header if it exists
        if (self.k==-1): return
        snapfile = get_snap_file_name(sdir,snum,self.nsnap,0)
        f = h5py.File(snapfile, 'r')
        self.npart = f['Header'].attrs['NumPart_Total']
        self.time = f['Header'].attrs['Time']
        self.redshift = f['Header'].attrs['Redshift']
        self.boxsize = f['Header'].attrs['BoxSize']
        if cosmological:
            self.scale_factor = self.time
            self.omega = f['Header'].attrs.get('Omega_Matter',0)
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

        if f['Header'].attrs.get('ISMDustChem_Num_Grain_Size_Bins',0):
            print("This snap has grain size bins!")
            self.Grain_Size_Max = f['Header'].attrs['ISMDustChem_Grain_Size_Max']
            self.Grain_Size_Min = f['Header'].attrs['ISMDustChem_Grain_Size_Min']
            self.Flag_GrainSizeBins = f['Header'].attrs['ISMDustChem_Num_Grain_Size_Bins']
            bin_size = np.power(10,np.log10( self.Grain_Size_Max/self.Grain_Size_Min)/self.Flag_GrainSizeBins)
            self.Grain_Bin_Edges = np.zeros(self.Flag_GrainSizeBins+1)
            self.Grain_Bin_Centers = np.zeros(self.Flag_GrainSizeBins)
            for i in range(self.Flag_GrainSizeBins+1):
                self.Grain_Bin_Edges[i] = pow(bin_size,i)*self.Grain_Size_Min * config.cm_to_um
            for i in range(self.Flag_GrainSizeBins):
                self.Grain_Bin_Centers[i] = ( self.Grain_Bin_Edges[i+1]+ self.Grain_Bin_Edges[i])/2.
        else:
                self.Flag_GrainSizeBins = 0

        f.close()

        # correct for cosmological runs
        if (self.cosmological==1): self.boxsize *= (self.time/self.hubble)

        # initialize particle types
        self.header = Header(self)
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

        part = self.part[ptype]
        part.load()

        return part


    def loadheader(self):

        header = self.header
        header.load()

        return header


    def loadAHF(self, hdir=None):
        
        # non-cosmological snapshots do not have AHF attribute
        if (self.cosmological==0): return None

        AHF = self.AHF
        AHF.load(hdir=hdir)

        return AHF
    

    def loadhalo(self, id=-1, mode='AHF', hdir=None):
    

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


    # SFH in the whole box
    def get_SFH(self, dt=0.01, cum=0):

        if(self.k==-1): return None, None

        part = self.star; part.load()
        if (part.k==-1): return None, None

        sft, m = part.sft, part.m
        t, sfr = math_utils.SFH(sft, m, dt=dt, cum=cum, sp=self)

        return t, sfr