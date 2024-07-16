import numpy as np
import glob
import os


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
        if hdir is None: hdir = os.path.dirname(os.path.normpath(sp.sdir)) + "/halo/ahf/output"
        print("Looking for snapshot's corresponding AHF file")
        hfile_formats = [hdir + "/snap%03d*.AHF_halos" %sp.snum, hdir + "/snapshot_%03d*.AHF_halos" %sp.snum]
        for hfile in hfile_formats:
            flist = glob.glob(hfile)
            if (len(flist)) != 0:
                break

        # no valid file, leave self.k=0
        if (len(flist)==0): 
            print("No valid AHF halo file.")
            return
        else:
            hfile = flist[0]
            print("AHF file found" + hfile)
	    
        # read the blocks
        hinv = 1.0/sp.hubble 
        ascale = sp.time

        # Now check if halo file is old or new AHF version since the columns change
        old = True
        try:
            np.loadtxt(hfile, usecols=(83,), unpack=True)
        except:
            old = False

        # Note data is converted to physical units
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
            Mvir, M_gas, M_star = hinv*np.loadtxt(hfile, usecols=(3,44,64,), unpack=True)
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
        self.Lhat = np.array([Lx,Ly,Lz])

        return


    # this function is only called when loading a halo
    def get_valid_halo_id(self, id, hdir=None):
    
        self.load(hdir=hdir) # load catalog
        if (self.k!=1): return -1 # no catalog present
        if (id<0) or (id>=self.nhalo):
            ok = self.fMhires>0.99
            return self.ID[ok][np.argmax(self.Mvir[ok])]

        return id

