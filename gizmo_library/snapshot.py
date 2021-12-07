import numpy as np
import h5py
import os
from . import utils
from . import config
from .particle import Header,Particle
from .galaxy import Halo,Disk
from .AHF import AHF

class Snapshot:

    def __init__(self, sdir, snum, cosmological=0, periodic_bound_fix=False):

        self.sdir = sdir
        self.snum = snum
        self.cosmological = cosmological
        self.nsnap = self.check_snap_exist()
        self.k = -1 if self.nsnap==0 else 1

        # In case the sim was non-cosmological and used periodic BC which causes
        # galaxy to be split between the 4 corners of the box
        self.pb_fix = False
        if periodic_bound_fix and cosmological==0:
            self.pb_fix=True


        # now, read snapshot header if it exists
        if (self.k==-1): return
        snapfile = self.get_snap_file_name(0)
        f = h5py.File(snapfile, 'r')
        self.npart = f['Header'].attrs['NumPart_Total']
        self.time = f['Header'].attrs['Time']
        self.redshift = f['Header'].attrs['Redshift']
        self.boxsize = f['Header'].attrs['BoxSize']
        if cosmological:
            self.scale_factor = self.time
            self.omega = f['Header'].attrs.get('Omega_Matter',0)
            if self.omega==0: f['Header'].attrs.get('Omega0',0)
        self.hubble = f['Header'].attrs['HubbleParam']
        self.Flag_Sfr = f['Header'].attrs['Flag_Sfr']
        self.Flag_Cooling = f['Header'].attrs['Flag_Cooling']
        self.Flag_StellarAge = f['Header'].attrs['Flag_StellarAge']
        self.Flag_Metals = f['Header'].attrs['Flag_Metals']
        self.Flag_DustMetals = f['Header'].attrs.get('Flag_Dust',0)
        self.Flag_DustSpecies = f['Header'].attrs.get('Flag_Species',0)
        if(self.Flag_DustSpecies==0 and self.Flag_DustMetals !=0): self.Flag_DustSpecies=2 # just generalized silicate and carbonaceous
        # Determine if the snapshot came from a simulations with on-the-fly dust
        if self.Flag_Metals and self.Flag_DustSpecies>2:
            self.dust_impl = 'species'
        elif self.Flag_Metals:
            self.dust_impl = 'elemental'
        else:
            self.dust_impl = None
        self.Flag_Sfr = f['Header'].attrs['Flag_Sfr']
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
        t, sfr = utils.SFH(sft, m, dt=dt, cum=cum, sp=self)

        return t, sfr


    # this function only used once when loading a snapshot
    def check_snap_exist(self):
    
        sdir = self.sdir
        snum = self.snum
        
        # single file case
        snapfile = sdir + "/snapshot_%03d.hdf5" %snum
        if (os.path.isfile(snapfile)): return 1
        
        # multiple files
        snapfile = sdir + "/snapdir_%03d/snapshot_%03d.0.hdf5" %(snum,snum)
        if (os.path.isfile(snapfile)):
            f = h5py.File(snapfile, 'r')
            nsnap = f['Header'].attrs['NumFilesPerSnapshot']
        else:
            print("Snapshot does not exist.")
            return 0
        
        for i in np.arange(1,nsnap,1):
            snapfile = sdir + "/snapdir_%03d/snapshot_%03d.%d.hdf5" %(snum,snum,i)
            if (not os.path.isfile(snapfile)):
                print("Snapshot is not complete.")
                return 0
        
        return nsnap


    # this function returns snapshot file name
    def get_snap_file_name(self, i):
    
        sdir = self.sdir
        snum = self.snum
        
        if (self.nsnap==1):
            snapfile = sdir + "/snapshot_%03d.hdf5" %snum
        else:
            snapfile = sdir + "/snapdir_%03d/snapshot_%03d.%d.hdf5" %(snum,snum,i)

        return snapfile
