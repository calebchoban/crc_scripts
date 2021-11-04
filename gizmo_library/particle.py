import numpy as np
import h5py
import os
from . import utils
from . import config


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
        self.hubble = sp.hubble

        return


class Particle:

    def __init__(self, sp, ptype):

        # no snapshot, no particle info
        self.sp = sp
        self.k = -1 if sp.k==-1 else 0
        self.ptype = ptype

        # Used to make sure particles isn't centered twice
        self.centered = False

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
                z = np.zeros((npart,sp.Flag_Metals), dtype='float')
            if (sp.Flag_DustMetals):
                dz = np.zeros((npart,sp.Flag_DustMetals-4), dtype='float')
                dzs = np.zeros((npart,4), dtype='float')
                fH2 = np.zeros(npart, dtype='float')
                fMC = np.zeros(npart, dtype='float')
                CinCO = np.zeros(npart, dtype='float')
            if (sp.Flag_DustSpecies):
                spec = np.zeros((npart,sp.Flag_DustSpecies), dtype='float')
            if (sp.Flag_Sfr):
                sfr = np.zeros(npart, dtype='float')
            
        if (ptype==4):
            if (sp.Flag_StellarAge):
                sft = np.zeros(npart, dtype='float')
            if (sp.Flag_Metals):
                z = np.zeros((npart,sp.Flag_Metals), dtype='float')

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
                    z[nL:nR] = grp['Metallicity']
                if (sp.Flag_DustMetals):
                    dz[nL:nR] = grp['DustMetallicity'][:,:sp.Flag_DustMetals-4]
                    dzs[nL:nR] = grp['DustMetallicity'][:,sp.Flag_DustMetals-4:]
                if 'DustMolecular' in grp:
                    fMC = grp['DustMolecular'][:,0]
                    CinCO = grp['DustMolecular'][:,1]
                    if 'MolecularMassFraction' in grp:
                        fH2 = grp['MolecularMassFraction'][...]
                    # Deal with the inbetween case when molecular data was all in one place
                    # Can probably delete this soon since only a few sims have this specific output
                    else: 
                        fH2 = grp['DustMolecular'][:,0]
                        fMC = grp['DustMolecular'][:,1]
                        CinCO = grp['DustMolecular'][:,2]
                if (sp.Flag_DustSpecies>2):
                    spec[nL:nR] = grp['DustSpecies'][...]
                elif (sp.Flag_DustMetals and sp.Flag_DustSpecies==2):
                    spec[nL:nR,0] = dz[nL:nR,4]+dz[nL:nR,6]+dz[nL:nR,7]+dz[nL:nR,10]
                    spec[nL:nR,1] = dz[nL:nR,2]
                if (sp.Flag_Sfr):
                    sfr[nL:nR] = grp['StarFormationRate'][...]
            
            if (ptype==4):
                if (sp.Flag_StellarAge):
                    sft[nL:nR] = grp['StellarFormationTime'][...]
                if (sp.Flag_Metals):
                    z[nL:nR] = grp['Metallicity']
            
            f.close()
            nL = nR

        # construct the final dictionary
        p *= (ascale*hinv)
        v *= np.sqrt(ascale)
        m *= hinv
        self.k = 1
        self.npart = npart
        self.p = p
        if self.sp.pb_fix:
            self.pb_fix()
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
                    if (sp.Flag_DustDepl):
                        self.z += self.dz
                    if (sp.Flag_DustSpecies):
                        self.spec = spec
                    self.fH2 = fH2
                    self.fMC = fMC
                    self.CinCO = CinCO
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
        

    # Reduce the particle data to only the masked particles
    def mask(self, mask):
        self.p = self.p[mask]
        self.v = self.v[mask]
        self.m = self.m[mask]
        self.npart = len(self.m)
        self.id = self.id[mask]

        if (self.ptype==0):
            self.h = self.h[mask]
            self.u = self.u[mask]
            self.rho = self.rho[mask]
            if (self.sp.Flag_Cooling):
                self.T = self.T[mask]
                self.ne = self.ne[mask]
                self.nh = self.nh[mask]
            if (self.sp.Flag_Metals):
                self.z = self.z[mask]
            if (self.sp.Flag_DustMetals):
                self.dz = self.dz[mask]
                self.dzs = self.dzs[mask]
                if (self.sp.Flag_DustSpecies):
                    self.spec = self.spec[mask]
                self.fH2 = self.fH2[mask]
                self.fMC = self.fMC[mask]
                self.CinCO = self.CinCO[mask]
            if (self.sp.Flag_Sfr):
                self.sfr = self.sfr[mask]
    
        if (self.ptype==4):
            if (self.sp.Flag_StellarAge):
                if (self.sp.cosmological==0):
                    self.sft = self.sft[mask]
                    self.age = self.age[mask]
            if (self.sp.Flag_Metals):
                self.z = self.z[mask]

        return


    # Centers coordinates given origin coordinates
    def center(self, origin):

        if self.centered: return

        self.centered = True
        self.p -= origin

        return


    # Rotates particle fields given the rotation matrix
    def rotate(self, rotation_matrix):
        self.p = np.dot(self.p,rotation_matrix)
        self.v = np.dot(self.v,rotation_matrix)

        return


    # Fixes coordinate issue for non-cosmological periodic BCs
    def pb_fix(self):
        p = self.p       
        boxsize = self.sp.boxsize
        mask1 = p > boxsize/2; mask2 = p <= boxsize/2
        p[mask1] -= boxsize/2; p[mask2] += boxsize/2;
        self.p = p

        return

    # Gets derived properties from particle data
    def get_property(self, property):

        if property in ['M','M_gas','M_star']:
            data = self.m*config.UnitMass_in_Msolar
        elif property == 'M_gas_neutral':
            data = self.m*self.nh*config.UnitMass_in_Msolar
        elif property == 'M_metals':
            data = self.z[:,0]*self.m*config.UnitMass_in_Msolar
        elif property == 'M_dust':
            data = self.dz[:,0]*self.m*config.UnitMass_in_Msolar
        elif property == 'M_sil':
            data = self.spec[:,0]*self.m*config.UnitMass_in_Msolar
        elif property == 'M_carb':
            data = self.spec[:,1]*self.m*config.UnitMass_in_Msolar
        elif property == 'M_SiC':
            data = self.spec[:,2]*self.m*config.UnitMass_in_Msolar
        elif property == 'M_iron':
            if self.sp.Flag_DustSpecies>4:
                data = (self.spec[:,3]+self.spec[:,5])*self.m*config.UnitMass_in_Msolar
            else:
                data = self.spec[:,3]*self.m*config.UnitMass_in_Msolar
        elif property == 'M_ORes':
            data = self.spec[:,4]*self.m*config.UnitMass_in_Msolar
        elif property == 'M_sil+':
            data = (self.spec[:,0]+np.sum(self.spec[:,2:],axis=1))*self.m*config.UnitMass_in_Msolar
        elif property == 'fH2':
            data = self.fH2
            data[data>1] = 1
        elif property == 'fMC':
            data = self.fMC
            data[data>1] = 1
        elif property == 'CinCO':
            data = self.CinCO/self.z[:,2]
        elif property == 'nH':
            data = self.rho*config.UnitDensity_in_cgs * (1. - (self.z[:,0]+self.z[:,1])) / config.H_MASS
        elif property == 'nH_neutral':
            data = (self.rho*config.UnitDensity_in_cgs * (1. - (self.z[:,0]+self.z[:,1])) / config.H_MASS)*self.nh
        elif property == 'T':
            data = self.T
        elif property == 'r':
            data = np.sqrt(np.power(self.p[:,0],2) + np.power(self.p[:,1],2))
        elif property == 'Z':
            data = self.z[:,0]/config.SOLAR_Z
        elif property == 'Z_all':
            data = self.z
        elif property == 'O/H':
            O = self.z[:,4]/config.ATOMIC_MASS[4]; H = (1-(self.z[:,0]+self.z[:,1]))/config.ATOMIC_MASS[0]
            data = 12+np.log10(O/H)
        elif property == 'Si/C':
            data = self.spec[:,0]/self.spec[:,1]
        elif property == 'D/Z':
            data = self.dz[:,0]/self.z[:,0]
            data[data > 1] = 1.
        elif 'depletion' in property:
            elem = property.split('_')[0]
            if elem not in config.ELEMENTS:
                print('%s is not a valid element to calculate depletion for. Valid elements are'%elem)
                print(config.ELEMENTS)
                return None
            elem_indx = config.ELEMENTS.index(elem)
            data =  self.dz[:,elem_indx]/self.z[:,elem_indx]
            data[data > 1] = 1.
        else:
            print("Property given to Particle is not supported:",property)
            return None

        return data