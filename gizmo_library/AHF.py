import numpy as np
import os
import utils
import glob


class AHF:

    def __init__(self, sp):

        # AHF catalog may exist even if the snapshot doesn't
        self.sp = sp
        self.k = 0 if sp.cosmological else -1

        return


    def load(self, hdir=None):

        if (self.k!=0): return

        # load AHF catalog
        sp = self.sp
        if hdir is None: hdir = sp.sdir + "/AHF_halos"
        print('halo file directory is', hdir)
        hfile = hdir + "/snap%03d*.AHF_halos" %sp.snum
        flist = glob.glob(hfile)
        # Check for different halo file name format
        if len(flist) == 0:
            hfile = hdir + "/snapshot_%03d*.AHF_halos" %sp.snum
            flist = glob.glob(hfile)

        # no valid file, leave self.k=0
        if (len(flist)==0): return
        hfile = flist[0]
	    
        # read the blocks
        hinv = 1.0/sp.hubble 
        ascale = sp.time

        # Now check if halo file is old or new AHF version since the columns change
        old = True
        try:
            np.loadtxt(hfile, usecols=(83,), unpack=True)
        except:
            old = False

        if old:
            ID, hostHalo, npart, n_gas, n_star = \
                    np.loadtxt(hfile, usecols=(0,1,4,52,72,), unpack=True, dtype='int')
            Mvir, M_gas, M_star = hinv*np.loadtxt(hfile, usecols=(3,44,64,), unpack=True)
            Xc, Yc, Zc, Rvir, Rmax = ascale*hinv*np.loadtxt(hfile, usecols=(5,6,7,11,12,), unpack=True)
            Vmax = np.loadtxt(hfile, usecols=(16,), unpack=True) # velocity in km/s
            Lx, Ly, Lz = np.loadtxt(hfile, usecols=(23,24,25,), unpack=True)
            fMhires = np.loadtxt(hfile, usecols=(37,), unpack=True)
        
        else:
            ID, hostHalo, npart, n_gas, n_star = \
                    np.loadtxt(hfile, usecols=(0,1,4,43,63,), unpack=True, dtype='int')
            Mvir, M_gas, M_star = hinv*np.loadtxt(hfile, usecols=(3,53,73,), unpack=True)
            Xc, Yc, Zc, Rvir, Rmax = ascale*hinv*np.loadtxt(hfile, usecols=(5,6,7,11,12,), unpack=True)
            Vmax = np.loadtxt(hfile, usecols=(16,), unpack=True) # velocity in km/s
            Lx, Ly, Lz = np.loadtxt(hfile, usecols=(23,24,25,), unpack=True)
            fMhires = np.loadtxt(hfile, usecols=(37,), unpack=True)

        # now write to class
        self.k = 1
        self.hdir = hdir
        self.nhalo = len(ID)
        self.ID = ID
        self.hostHalo = hostHalo
        self.npart = npart
        self.n_gas = n_gas
        self.n_star = n_star
        self.Mvir = Mvir
        self.M_gas = M_gas
        self.M_star = M_star
        self.Xc = Xc
        self.Yc = Yc
        self.Zc = Zc
        self.Rvir = Rvir
        self.Rmax = Rmax
        self.Vmax = Vmax
        self.fMhires = fMhires
        self.Lhat = [Lx,Ly,Lz]


        return

            
    # load AHF particles
    def loadpart(self, id=0):
    
        self.load()
        if (self.k!=1): return [],[] # no halo catalog
        
        if ((id<0)|(id>=self.nhalo)): return [],[] # id not valid
        
        pfile = self.hdir + "/snap%03d*AHF_particles" %self.sp.snum
        flist = glob.glob(pfile)
        if (len(flist)==0): return [],[] # no particle file
        PFile = flist[0]
    
        NDummy = np.sum(self.npart[0:id]) + id
        NPart = self.npart[id]
        PartID, PType = utils.loadAHFpart(PFile, NDummy, NPart)
    
        return PartID, PType


    # this function is only called when loading a halo
    def get_valid_halo_id(self, id, hdir=None):
    
        self.load(hdir=hdir) # load catalog
        if (self.k!=1): return -1 # no catalog present
        if (id<0) or (id>=self.nhalo):
            ok = self.fMhires>0.99
            return self.ID[ok][np.argmax(self.Mvir[ok])]

        return id
