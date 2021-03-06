import numpy as np
import h5py
import os
import AHF
import utils
import rockstar
from galaxy import Halo
from particle import Header,Particle

class Snapshot:

    def __init__(self, sdir, snum, cosmological=0):
        
        self.sdir = sdir
        self.snum = snum
        self.cosmological = cosmological
        self.nsnap = self.check_snap_exist()
        self.k = -1 if self.nsnap==0 else 1
        
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
            # we support both AHF and rockstar
            self.AHF = AHF.AHF(self)
            self.rockstar = rockstar.rockstar(self)
            self.AHFhaloIDs = []
            self.AHFhalos = []
            self.rockstarhaloIDs = []
            self.rockstarhalos = []
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
        self.Flag_Sfr = f['Header'].attrs['Flag_Sfr']
        f.close()

        # correct for (non-)cosmological runs
        if (self.cosmological==0): self.time /= self.hubble
        if (self.cosmological==1): self.boxsize *= (self.time/self.hubble)

        return


    def loadpart(self, ptype, header_only=0):

        part = self.header if header_only else self.part[ptype]
        part.load()

        return part
    
    
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
    
    
    def runAHF(self, AHFDir=None):
        
        # non-cosmological snapshots do not have AHF attribute
        if (self.cosmological==0): return
        
        AHF = self.AHF
        AHF.runAHF(AHFDir=AHFDir)
    
        return
    
    
    def loadrockstar(self, nclip=1000):

        # non-cosmological snapshots do not have rockstar attribute
        if(self.cosmological==0): return

        rockstar = self.rockstar
        rockstar.load(nclip=nclip)
    
        return rockstar
    
    
    def runrockstar(self, DM_only=False, edm=1.0e-4, FOF_FRACTION=0.7, FOF_LINKING_LENGTH=0.28, MIN_HALO_PARTICLES=10):
        
        # non-cosmological snapshots do not have AHF attribute
        if (self.cosmological==0): return
        
        rockstar = self.rockstar
        rockstar.runrockstar(DM_only=DM_only, edm=edm, FOF_FRACTION=FOF_FRACTION, FOF_LINKING_LENGTH=FOF_LINKING_LENGTH, MIN_HALO_PARTICLES=MIN_HALO_PARTICLES)
    
        return
    

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
