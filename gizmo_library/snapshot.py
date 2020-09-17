import numpy as np
import h5py
import os
import utils
import config
from particle import Header,Particle
from galaxy import Halo,Disk

class Snapshot:

    def __init__(self, sdir, snum, cosmological=0, periodic_bound_fix=False, dust_depl=False):

        self.sdir = sdir
        self.snum = snum
        self.cosmological = cosmological
        self.nsnap = self.check_snap_exist()
        self.k = -1 if self.nsnap==0 else 1
        self.Flag_DustDepl = dust_depl

        # In case the sim was non-cosmological and used periodic BC which causes
        # galaxy to be split between the 4 corners of the box
        self.pb_fix = False
        if periodic_bound_fix and cosmological==0:
            self.pb_fix=True
        
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
            self.AHF = AHF.AHF(self)
            self.AHFhaloIDs = []
            self.AHFhalos = []
        else:
            # non-cosmological snapshot has only one galaxy
            self.halo = Halo(self)

        # now, read snapshot header if it exists
        if (self.k==-1): return
        snapfile = self.get_snap_file_name(0)
        f = h5py.File(snapfile, 'r')
        self.npart = f['Header'].attrs['NumPart_Total']
        self.time = f['Header'].attrs['Time']
        self.redshift = f['Header'].attrs['Redshift']
        self.boxsize = f['Header'].attrs['BoxSize']
        self.omega = f['Header'].attrs['Omega0']
        self.hubble = f['Header'].attrs['HubbleParam']
        self.Flag_Sfr = f['Header'].attrs['Flag_Sfr']
        self.Flag_Cooling = f['Header'].attrs['Flag_Cooling']
        self.Flag_StellarAge = f['Header'].attrs['Flag_StellarAge']
        self.Flag_Metals = f['Header'].attrs['Flag_Metals']
        try:
            self.Flag_DustMetals = f['Header'].attrs["Flag_Dust"]
            self.Flag_DustSpecies = f['Header'].attrs['Flag_Species']
        except:
            self.Flag_DustMetals = 0
            self.Flag_DustSpecies = 0
        # Determine if the snapshot came from a simulations with on-the-fly dust
        if self.Flag_Metals and self.Flag_DustSpecies:
            self.dust_impl = 'species'
        elif self.Flag_Metals:
            self.dust_impl = 'elemental'
        else:
            self.dust_impl = None
        self.Flag_Sfr = f['Header'].attrs['Flag_Sfr']
        f.close()

        # correct for cosmological runs
        if (self.cosmological==1): self.boxsize *= (self.time/self.hubble)

        return


    def loadpart(self, ptype):

        part = self.part[ptype]
        part.load()

        return part


    def loadheader(self):

        header = self.header
        header.load()

        return header


    def viewpart(self, ptype, field='None', method='simple', **kwargs):
        
        if (self.k==-1): return -1
    
        part = self.part[ptype]; part.load()
        if (part.k==-1): return -1

        if 'cen' not in kwargs: kwargs['cen'] = self.boxsize*np.ones(3)/2
        if 'L' not in kwargs: kwargs['L'] = self.boxsize/2
        
        # add time label for the image
        kwargs['time'] = r"$z=%.1f$" % self.redshift if self.cosmological else r"$t=%.1f$" %self.time
        
        # check which field to show
        h, wt = None, part.m
        if (ptype==0):
            # for gas, check if we want a field from the list below
            if (field in ['nh','ne','z']) and (field in dir(part)): wt*=getattr(part,field)
        if (ptype==4):
            # for stars, check if we want luminosity from some band
            import colors
            if field in colors.colors_available:
                wt*=colors.colors_table(part.age,part.z/0.02,band=field)
        if (method=='smooth'):
            h = part.h if ptype==0 else utils.get_particle_hsml(part.p[:,0],part.p[:,1],part.p[:,2])

        # now, call the routine
        import visual
        H = visual.make_projected_image(part.p, wt, h=h, method=method, **kwargs)
        
        return H


    def loadAHF(self, hdir=None):
        
        # non-cosmological snapshots do not have AHF attribute
        if (self.cosmological==0): return None

        AHF = self.AHF
        AHF.load(hdir=hdir)

        return AHF
    

    def loadhalo(self, id=-1, mode='AHF', hdir=None, nclip=1000):
    
        # non-cosmological, use the only galaxy attribute
        if (self.cosmological==0):
            hl = self.halo
            hl.load()

            return hl
        
        # cosmological, use AHF
        if (mode=='AHF'):
            id = self.AHF.get_valid_halo_id(id, hdir=hdir)
            if (id<0): return
        
            if (id in self.AHFhaloIDs):
                index = self.AHFhaloIDs.index(id)
                hl = self.AHFhalos[index]
            else:
                hl = Halo(self, id=id)
                self.AHFhaloIDs.append(id)
                self.AHFhalos.append(hl)
        
        # cosmological, use rockstar
        else:
        
            id = self.rockstar.get_valid_halo_id(id)
            if (id<0): return
        
            if (id in self.rockstarhaloIDs):
                index = self.rockstarhaloIDs.index(id)
                hl = self.rockstarhalos[index]
            else:
                hl = Halo(self, id=id)
                self.rockstarhaloIDs.append(id)
                self.rockstarhalos.append(hl)
                    
        hl.load(mode=mode, nclip=nclip)
    
        return hl


    def loaddisk(self, id=-1, mode='AHF', hdir=None, nclip=1000, rmax=20, height=5):
    
        # non-cosmological, use the only galaxy attribute
        if (self.cosmological==0):
            disk = Disk(self, id=id,rmax=rmax,height=height)
            disk.load()

            return disk
        
        # cosmological, use AHF
        if (mode=='AHF'):
            id = self.AHF.get_valid_halo_id(id, hdir=hdir)
            if (id<0): return
        
            if (id in self.AHFhaloIDs):
                index = self.AHFhaloIDs.index(id)
                disk = self.AHFhalos[index]
            else:
                disk = Disk(self, id=id,rmax=rmax,height=height)
                self.AHFhaloIDs.append(id)
                self.AHFhalos.append(disk)
        
        # cosmological, use rockstar
        else:
        
            id = self.rockstar.get_valid_halo_id(id)
            if (id<0): return
        
            if (id in self.rockstarhaloIDs):
                index = self.rockstarhaloIDs.index(id)
                disk = self.rockstarhalos[index]
            else:
                disk = Disk(self, id=id,rmax=rmax,height=height)
                self.rockstarhaloIDs.append(id)
                self.rockstarhalos.append(disk)
                    
        disk.load(mode=mode, nclip=nclip)
    
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
