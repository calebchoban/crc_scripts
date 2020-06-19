import numpy as np
import os
import utils
import glob


# change this if needed
AHF_HOME = os.environ['HOME'] + "/pylib/gizmo_lib/AHF/"


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

        # Now check if halo file is form old or new AHF version since the columns change
        old = True
        try:
            np.loadtxt(hfile, usecols=(83,), unpack=True, dtype='int')
        except:
            old = False
        old=True
        if old:
            ID, hostHalo, npart, n_gas, n_star = \
                    np.loadtxt(hfile, usecols=(0,1,4,52,72,), unpack=True, dtype='int')
            Mvir, M_gas, M_star = hinv*np.loadtxt(hfile, usecols=(3,44,64,), unpack=True)
            Xc, Yc, Zc, Rvir, Rmax = ascale*hinv*np.loadtxt(hfile, usecols=(5,6,7,11,12,), unpack=True)
            Vmax = np.loadtxt(hfile, usecols=(16,), unpack=True) # velocity in km/s
            fMhires = np.loadtxt(hfile, usecols=(37,), unpack=True)
        
        else:
            ID, hostHalo, npart, n_gas, n_star = \
                    np.loadtxt(hfile, usecols=(0,1,4,43,63,), unpack=True, dtype='int')
            Mvir, M_gas, M_star = hinv*np.loadtxt(hfile, usecols=(3,53,73,), unpack=True)
            Xc, Yc, Zc, Rvir, Rmax = ascale*hinv*np.loadtxt(hfile, usecols=(5,6,7,11,12,), unpack=True)
            Vmax = np.loadtxt(hfile, usecols=(16,), unpack=True) # velocity in km/s
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


    def runAHF(self, AHFDir=None):

        # try load AHF catalog, see if it already exists
        self.load()
        if (self.k!=0): return

        # no snapshot, just return
        sp = self.sp
        if (sp.k==-1): return

        # single file or multiple files
        FL = 1 if sp.nsnap>1 else 0
        SnapDir = sp.sdir
        SnapNum = sp.snum

        # create converted binary files
        if (FL == 1):
            SnapFile = SnapDir + "/snapdir_%03d" % SnapNum
            SnapFile_converted = SnapDir + "/snap_converteddir_%03d" % SnapNum
            os.system("mkdir "+SnapFile_converted)
        else:
            SnapFile = SnapDir + "/snapshot_%03d.hdf5" % SnapNum
            SnapFile_converted = SnapDir + "/snap_convertedshot_%03d" % SnapNum

        # create parameter file for AHF
        if AHFDir is None: AHFDir = SnapDir + "/AHFHalos"
        if (not os.path.exists(AHFDir)): os.system("mkdir -p %s" %AHFDir)
        InputFile = AHFDir + "/AMIGA.input%03d" % SnapNum
    
        # do the actual run
        CONVERT = "ipython " + AHF_HOME + "HDF5converter.py " + SnapFile
        os.system(CONVERT)
        createINPUT = "ipython " +  AHF_HOME + "createAMIGAinput.py " + SnapDir + " %03d %d" %(SnapNum, FL)
        os.system(createINPUT)
        RUNAHF = "AHF-v1.0-069 " + InputFile # note you need $AHF_069 variable
        os.system(RUNAHF)
    
        # finally, delete the converted binary file
        os.system("rm -r "+SnapFile_converted)
    
        return


    # cross match between AHF and rockstar
    def AHFrockstar(self):
        
        sp = self.sp
        
        # load halo catalogs
        AHF = sp.loadAHF()
        rockstar = sp.loadrockstar()
        
        # if one of them does not exist, just return
        if (self.k!=1) or (rockstar.k!=1): return
        
        # now do the cross match
        id_rockstar = -1*np.ones(len(AHF.ID), dtype='int')
        dl_rockstar = np.zeros(len(AHF.ID))
        phalo_rockstar = np.zeros((len(rockstar.id),3))
        phalo_rockstar[:,0] = rockstar.x
        phalo_rockstar[:,1] = rockstar.y
        phalo_rockstar[:,2] = rockstar.z
        
        from scipy.spatial import cKDTree
        # find the nearest use cKDTree algorithm
        tree = cKDTree(phalo_rockstar)
        for id_AHF in AHF.ID:
            dl, id = tree.query([AHF.Xc[id_AHF],AHF.Yc[id_AHF],AHF.Zc[id_AHF]])
            id_rockstar[id_AHF]=id; dl_rockstar[id_AHF]=dl
        
        # save the record
        self.id_rockstar = id_rockstar
        self.dl_rockstar = dl_rockstar
        
        return


    # this function is only called when loading a halo
    def get_valid_halo_id(self, id, hdir=None):
    
        self.load(hdir=hdir) # load catalog
        if (self.k!=1): return -1 # no catalog present
        if (id<0) or (id>=self.nhalo):
            ok = self.fMhires>0.99
            return self.ID[ok][np.argmax(self.Mvir[ok])]

        return id

