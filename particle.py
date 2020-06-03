import numpy as np
import h5py
import os
import utils
import config

import matplotlib.pyplot as plt
plt.switch_backend('agg')

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
        self.redshift = sp.redshift
        self.boxsize = sp.boxsize
        self.omega = sp.omega
        self.hubble = sp.hubble

        return


class Particle:

    def __init__(self, sp, ptype):

        # no snapshot, no particle info
        self.sp = sp
        self.k = -1 if sp.k==-1 else 0
        self.ptype = ptype

        return


    def load(self):

        if (self.k!=0): return

        # class basic info
        sp = self.sp
        ptype = self.ptype
        npart = sp.npart[ptype]

        # no particle in this ptype
        if (npart==0): return

        # get correction for scale factor

        hinv = 1.0
        ascale = 1.0
        if (sp.cosmological):
            ascale = sp.time
            hinv = 1.0 / sp.hubble

        # now, load basic particle info
        p = np.zeros((npart,3), dtype='float')
        v = np.zeros((npart,3), dtype='float')
        m = np.zeros(npart, dtype='float')
        id = np.zeros(npart, dtype='int')

        if (ptype==0):
            h = np.zeros(npart, dtype='float')
            u = np.zeros(npart, dtype='float')
            rho = np.zeros(npart, dtype='float')
            if (sp.Flag_Cooling):
                nh = np.zeros(npart, dtype='float')
                ne = np.zeros(npart, dtype='float')
            if (sp.Flag_Metals):
                z = np.zeros(npart, dtype='float')
            if (sp.Flag_DustMetals):
                dz = np.zeros((npart,sp.Flag_DustMetals-4), dtype='float')
                dzs = np.zeros((npart,4), dtype='float')
            if (sp.Flag_DustSpecies):
                spec = np.zeros((npart,4), dtype='float')
            if (sp.Flag_Sfr):
                sfr = np.zeros(npart, dtype='float')
            
        if (ptype==4):
            if (sp.Flag_StellarAge):
                sft = np.zeros(npart, dtype='float')
            if (sp.Flag_Metals):
                z = np.zeros(npart, dtype='float')

        # do the reading
        nL = 0
        for i in range(sp.nsnap):
            snapfile = sp.get_snap_file_name(i)
            f = h5py.File(snapfile, 'r')
            npart_this = f['Header'].attrs['NumPart_ThisFile'][ptype]
            nR = nL + npart_this

            grp = f['PartType%d'%ptype]
            p[nL:nR] = grp['Coordinates'][...]
            v[nL:nR] = grp['Velocities'][...]
            m[nL:nR] = grp['Masses'][...]
            id[nL:nR] = grp['ParticleIDs'][...]

            if (ptype==0):
                h[nL:nR] = grp['SmoothingLength'][...]
                u[nL:nR] = grp['InternalEnergy'][...]
                rho[nL:nR] = grp['Density'][...]
                if (sp.Flag_Cooling):
                    nh[nL:nR] = grp['NeutralHydrogenAbundance'][...]
                    ne[nL:nR] = grp['ElectronAbundance'][...]
                if (sp.Flag_Metals):
                    z[nL:nR] = grp['Metallicity'][:,0]
                if (sp.Flag_DustMetals):
                    dz[nL:nR] = grp['DustMetallicity'][:,:sp.Flag_DustMetals-4]
                    dzs[nL:nR] = grp['DustMetallicity'][:,sp.Flag_DustMetals-4:]
                if (sp.Flag_DustSpecies):
                    spec[nL:nR] = grp['DustSpecies'][...]
                if (sp.Flag_Sfr):
                    sfr[nL:nR] = grp['StarFormationRate'][...]
            
            if (ptype==4):
                if (sp.Flag_StellarAge):
                    sft[nL:nR] = grp['StellarFormationTime'][...]
                if (sp.Flag_Metals):
                    z[nL:nR] = grp['Metallicity'][:,0]
            
            f.close()
            nL = nR

        # construct the final dictionary
        p *= (ascale*hinv)
        # Fix coordinates for non-cosmological sims run in periodic box
        if not sp.cosmological:
            mask1 = p > sp.boxsize/2; mask2 = p <= sp.boxsize/2
            p[mask1] -= sp.boxsize/2; p[mask2] += sp.boxsize/2;
        v *= np.sqrt(ascale)
        m *= hinv
        self.k = 1
        self.npart = npart
        self.p = p
        self.v = v
        self.m = m
        self.id = id

        if (ptype==0):
            h *= (ascale*hinv)
            rho *= (hinv/(ascale*hinv)**3)
            self.h = h
            self.u = u
            self.rho = rho
            if (sp.Flag_Cooling):
                T = utils.gas_temperature(u, ne)
                self.T = T
                self.ne = ne
                self.nh = nh
            if (sp.Flag_Metals):
                self.z = z
            if (sp.Flag_DustMetals):
                self.dz = dz
                self.dzs = dzs
            if (sp.Flag_DustSpecies):
                self.spec = spec
            if (sp.Flag_Sfr):
                self.sfr = sfr
    
        if (ptype==4):
            if (sp.Flag_StellarAge):
                if (sp.cosmological==0): sft *= hinv
                self.sft = sft
                self.age = utils.get_stellar_ages(sft, sp=sp)
            if (sp.Flag_Metals):
                self.z = z

        return
