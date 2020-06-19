import numpy as np
import h5py
import os


# This routine supports the 'quick' mode of rockstar
# and links the rockstar catalog with the snapshot.
# Note:
#   (1) currently it only works for single-file snapshots,
#   (2) rockstar only works on dark matter particles, so
#       we correct for baryonic fraction, and
#   (3) the DM particle mass is read from 'MassTable' in the header.


class rockstar:

    def __init__(self, sp):
        
        self.sp = sp
        self.k = 0 if sp.cosmological else -1
        
        return
    
    
    def load(self, nclip=1000):
        
        if (self.k!=0): return
        
        # load rockstar catalog
        sp = self.sp
        
        hdir = sp.sdir + "/rockstar"
        hfile = hdir + "/halo_%03d.ascii" %sp.snum
        
        # if file does not exist, just return
        if (not os.path.exists(hfile)): return
        
        # read the blocks
        hinv = 1.0/sp.hubble
        ascale = sp.time
        id, num_p = np.loadtxt(hfile, usecols=(0,1,), unpack=True, dtype='int')
        mvir, mbound_vir = hinv*np.loadtxt(hfile, usecols=(2,3,), unpack=True)
        rvir, rvmax = ascale*hinv*np.loadtxt(hfile, usecols=(4,6,), unpack=True)
        vmax = np.loadtxt(hfile, usecols=(5,), unpack=True)
        x, y, z = ascale*1.0e3*hinv*np.loadtxt(hfile, usecols=(8,9,10,), unpack=True)
        
        # Note: rockstar has spurious small halos with down to ten particles, so we clip halos with particle number less than nclip. However, this should be used with caution.
        print("Clipped halos with less than %d particles." %nclip)
        ok, = np.where(num_p>=nclip)
        
        # now write to class
        self.k =1
        self.hdir = hdir
        self.nhalo = len(ok)
        self.nclip = nclip
        self.id = np.arange(len(ok))
        self.rockstarid = id[ok]
        self.num_p = num_p[ok]
        self.mvir = mvir[ok]
        self.mbound_vir = mbound_vir[ok]
        self.rvir = rvir[ok]
        self.vmax = vmax[ok]
        self.rvmax = rvmax[ok]
        self.x = x[ok]
        self.y = y[ok]
        self.z = z[ok]

        return


    def rockstar_snapshot_convert(self, DM_only=False):
    
        sp = self.sp
        
        # define file names
        snapfile = sp.get_snap_file_name(0)
        snapfile_converted = sp.sdir + "/snap_convertedshot_%03d.hdf5" %sp.snum
        print("Converting snapshot %s for rockstar." %snapfile)
        
        # need DM particle mass
        fsnap = h5py.File(snapfile, 'r')
        MDM = fsnap['PartType1/Masses'][0] # get DM particle mass
        if (DM_only==False): MDM *= 1.1832 # correct for baryonic fraction
        fsnap.close()
        
        # copy the header
        fsnap = h5py.File(snapfile, 'r')
        fsnap_converted = h5py.File(snapfile_converted, 'w')
        header = fsnap['Header']
        header_converted = fsnap_converted.create_group('Header')
        for name in header.attrs.keys():
            data = header.attrs[name]
            if (name=='MassTable'): data[1] = MDM
            if (name=='NumPart_ThisFile'): data = sp.npart
            header_converted.attrs[name] = data
        fsnap.close()
        fsnap_converted.close()
        
        # copy the DM particle
        Coordinates = np.zeros((sp.npart[1],3), dtype='float32')
        Velocities = np.zeros((sp.npart[1],3), dtype='float32')
        ParticleIDs = np.zeros(sp.npart[1], dtype='int32')
        nL = 0
        for i in range(sp.nsnap):
            snapfile = sp.get_snap_file_name(i)
            fsnap = h5py.File(snapfile, 'r')
            npart_this = fsnap['Header'].attrs['NumPart_ThisFile'][1]
            nR = nL + npart_this
            
            Coordinates[nL:nR] = fsnap['PartType1/Coordinates'][...]
            Velocities[nL:nR] = fsnap['PartType1/Velocities'][...]
            ParticleIDs[nL:nR] = fsnap['PartType1/ParticleIDs'][...]
            nL = nR
            fsnap.close()
        
        fsnap_converted = h5py.File(snapfile_converted, 'r+')
        part1_converted = fsnap_converted.create_group('PartType1')
        part1_converted.create_dataset('Coordinates', data=Coordinates)
        part1_converted.create_dataset('Velocities', data=Velocities)
        part1_converted.create_dataset('ParticleIDs', data=ParticleIDs)
        fsnap_converted.close()
        print("Snapshot converted.")

        return


    def rockstar_create_param_file(self, edm=1.0e-4, FOF_FRACTION=0.7, FOF_LINKING_LENGTH=0.28, MIN_HALO_PARTICLES=10):
        
        sp = self.sp
        
        # create the directory
        hdir = sp.sdir + "/rockstar"
        os.system("mkdir -p %s" %hdir)
        
        # write the parameter file
        fname = hdir + "/input%03d.cfg" %sp.snum
        f = open(fname, 'w')
        f.write('FILE_FORMAT = "AREPO"\n')
        f.write('SCALE_NOW = %f\n' %sp.time)
        f.write('h0 = %f\n' %sp.hubble)
        f.write('Ol = %f\n' %(1-sp.omega))
        f.write('Om = %f\n' %sp.omega)
        f.write('AREPO_LENGTH_CONVERSION = 0.001\n')
        f.write('AREPO_MASS_CONVERSION = 1e+10\n')
        f.write('FORCE_RES = %.3e\n' %edm)
        f.write('\n')
        f.write('FOF_FRACTION = %f\n' %FOF_FRACTION)
        f.write('FOF_LINKING_LENGTH = %f\n' %FOF_LINKING_LENGTH)
        f.write('MIN_HALO_PARTICLES = %d\n' %MIN_HALO_PARTICLES)
        f.close()

        return


    def runrockstar(self, DM_only=False, edm=1.0e-4, FOF_FRACTION=0.7, FOF_LINKING_LENGTH=0.28, MIN_HALO_PARTICLES=10):
        
        sp = self.sp
        if (sp.k==-1): print("Snapshot does not exist."); return
        
        # specify dirs and file names
        hdir = sp.sdir + "/rockstar" # rockstar catalog directory
        snapfile_converted = sp.sdir + "/snap_convertedshot_%03d.hdf5" %sp.snum
        paramfile = sp.sdir + "/rockstar/input%03d.cfg" %sp.snum
        
        # now, run it
        self.rockstar_snapshot_convert(DM_only=DM_only)
        self.rockstar_create_param_file(edm=edm, FOF_FRACTION=FOF_FRACTION, FOF_LINKING_LENGTH=FOF_LINKING_LENGTH, MIN_HALO_PARTICLES=MIN_HALO_PARTICLES)
        command = "rockstar -c %s %s" %(paramfile,snapfile_converted)
        os.system(command)
        
        # clean up
        command = "mv halos_0.0.ascii %s/halo_%03d.ascii" %(hdir,sp.snum)
        os.system(command)
        command = "mv halos_0.0.bin %s/halo_%03d.bin" %(hdir,sp.snum)
        os.system(command)
        command = "rm rockstar.cfg"
        os.system(command)
        command = "rm %s" %snapfile_converted
        os.system(command)

        return
    
    
    def issubhalo(self, id, nclip=1000):
        
        self.load(nclip=nclip)
        
        # halo catalog does not exist
        if (self.k!=1): return
        
        xc, yc, zc, mvir = self.x[id], self.y[id], self.z[id], self.mvir[id]
        dc = np.sqrt((self.x-xc)**2+(self.y-yc)**2+(self.z-zc)**2)
        # mark it a subhalo if it lives in a more massive halo
        ok = np.argwhere((dc<self.rvir)&(mvir<self.mvir))
        host = self.id[ok][np.argmax(self.mvir[ok])] if len(ok)>0 else -1
    
        return host


    def get_valid_halo_id(self, id):
        
        self.load() # load catalog
        if (self.k!=1): return -1 # catalog does not exist
        if (id<0) or (id>=self.nhalo):
            return np.argmax(self.mvir)

        return id
