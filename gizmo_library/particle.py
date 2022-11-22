import numpy as np
import h5py
from . import utils
from . import config
from . import coordinate

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
        if sp.cosmological:
            self.scale_factor = sp.scale_factor
        else: self.scale_factor = 1
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

        self.time = sp.time
        if sp.cosmological:
            self.scale_factor = sp.scale_factor
        else: self.scale_factor = 1
        self.redshift = sp.redshift
        self.boxsize = sp.boxsize
        self.hubble = sp.hubble

        # Used to make sure particles aren't orientated twice
        self.orientated = False

        # To be used for centering if wanted
        self.center_position = None
        self.center_velocity = None
        self.principal_axes_vectors = None
        self.principal_axes_ratios = None

        return


    def load(self):

        if (self.k!=0): return

        # class basic info
        sp = self.sp

        ptype = self.ptype
        npart = sp.npart[ptype]
        self.npart = npart

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
                fdense = np.zeros(npart, dtype='float')
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
            if npart_this <=0: continue  # if there are no particles of this type in the snap no reason to continue
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
                    fdense[nL:nR] = grp['DustMolecular'][:,0]
                    CinCO[nL:nR] = grp['DustMolecular'][:,1]
                    if 'MolecularMassFraction' in grp:
                        fH2[nL:nR] = grp['MolecularMassFraction'][...]
                    # Deal with the in between case when molecular data was all in one place
                    # Can probably delete this soon since only a few sims have this specific output
                    elif len(grp['DustMolecular'][:])>=3:
                        print("Loading old dust molecular...")
                        fH2[nL:nR] = grp['DustMolecular'][:,0]
                        fdense[nL:nR] = grp['DustMolecular'][:,1]
                        CinCO[nL:nR] = grp['DustMolecular'][:,2]
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
                z_he = z[:,1]; z_tot=z[:,0]
                T = utils.approx_gas_temperature(u,ne)
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
                    self.fH2 = fH2
                    self.fdense = fdense
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
                self.fdense = self.fdense[mask]
                self.CinCO = self.CinCO[mask]
            if (self.sp.Flag_Sfr):
                self.sfr = self.sfr[mask]
    
        if (self.ptype==4):
            if (self.sp.Flag_StellarAge):
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

    def orientate(self, center_pos=None, center_vel=None, principal_vec=None):

        if self.orientated: return

        print('adjusting particle coordinates to be relative to galaxy center')
        print('  and aligned with the principal axes\n')

        if center_vel is not None and center_pos is not None:
            # convert to be relative to galaxy center [km / s]
            self.v = coordinate.get_velocity_differences(
                        self.v, center_vel, self.p, center_pos,
                        self.boxsize, self.scale_factor, self.hubble)
            if principal_vec is not None:
                # convert to be aligned with galaxy principal axes
                self.v = coordinate.get_coordinates_rotated(self.v, principal_vec)

        if center_pos is not None:
            # convert to be relative to galaxy center [kpc physical]
            self.p = coordinate.get_distances(self.p, center_pos,
                self.boxsize, self.scale_factor)
            if principal_vec is not None:
                # convert to be aligned with galaxy principal axes
                self.p = coordinate.get_coordinates_rotated(
                    self.p, principal_vec)

        self.orientated=1

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

        if self.ptype==0:
            if property=='M' or property=='M_gas' or property=='m':
                data = self.m*config.UnitMass_in_Msolar
            elif property=='h':
                data = self.h
            elif property == 'M_gas_neutral':
                data = self.m*self.nh*config.UnitMass_in_Msolar
            elif property == 'M_mol' or property == 'M_H2':
                data = self.m*self.fH2*self.nh*config.UnitMass_in_Msolar
            elif property == 'M_metals':
                data = self.z[:,0]*self.m*config.UnitMass_in_Msolar
            elif property == 'M_dust':
                data = self.dz[:,0]*self.m*config.UnitMass_in_Msolar
            elif property == 'M_sil':
                data = self.spec[:,0]*self.m*config.UnitMass_in_Msolar
            elif property == 'M_carb':
                data = self.spec[:,1]*self.m*config.UnitMass_in_Msolar
            elif property == 'M_SiC':
                if self.sp.Flag_DustSpecies>2:
                    data = self.spec[:,2]*self.m*config.UnitMass_in_Msolar
                else:
                    data = np.zeros(len(self.m))
            elif property == 'M_iron':
                if self.sp.Flag_DustSpecies>5:
                    data = (self.spec[:,3]+self.spec[:,5])*self.m*config.UnitMass_in_Msolar
                elif self.sp.Flag_DustSpecies>2:
                    data = self.spec[:,3]*self.m*config.UnitMass_in_Msolar
                else:
                    data = np.zeros(len(self.m))
            elif property == 'M_ORes':
                if self.sp.Flag_DustSpecies>=5:
                    data = self.spec[:,4]*self.m*config.UnitMass_in_Msolar
                else:
                    data = np.zeros(len(self.m))
            elif property == 'M_sil+':
                data = (self.spec[:,0]+np.sum(self.spec[:,2:],axis=1))*self.m*config.UnitMass_in_Msolar
            elif property == 'dz_sil':
                data = self.spec[:,0]/self.dz[:,0]
            elif property == 'dz_carb':
                data = self.spec[:,1]/self.dz[:,0]
            elif property == 'dz_SiC':
                if self.sp.Flag_DustSpecies>2:
                    data = self.spec[:,2]/self.dz[:,0]
                else:
                    data = np.zeros(len(self.m))
            elif property == 'dz_iron':
                if self.sp.Flag_DustSpecies>5:
                    data = (self.spec[:,3]+self.spec[:,5])/self.dz[:,0]
                elif self.sp.Flag_DustSpecies>2:
                    data = self.spec[:,3]/self.dz[:,0]
                else:
                    data = np.zeros(len(self.m))
            elif property == 'dz_ORes':
                if self.sp.Flag_DustSpecies>=5:
                    data = self.spec[:,4]/self.dz[:,0]
                else:
                    data = np.zeros(len(self.m))
            elif property == 'M_acc_dust':
                data = self.dzs[:,0]*self.dz[:,0]*self.m*config.UnitMass_in_Msolar
            elif property == 'M_SNeIa_dust':
                data = self.dzs[:,1]*self.dz[:,0]*self.m*config.UnitMass_in_Msolar
            elif property == 'M_SNeII_dust':
                data = self.dzs[:,2]*self.dz[:,0]*self.m*config.UnitMass_in_Msolar
            elif property == 'M_AGB_dust':
                data = self.dzs[:,3]*self.dz[:,0]*self.m*config.UnitMass_in_Msolar
            elif property == 'dz_acc':
                data = self.dzs[:,0]
            elif property == 'dz_SNeIa':
                data = self.dzs[:,1]
            elif property == 'dz_SNeII':
                data = self.dzs[:,2]
            elif property == 'dz_AGB':
                data = self.dzs[:,3]
            elif property == 'fH2':
                data = self.fH2
                data[data>1] = 1
            elif property == 'fdense':
                data = self.fdense
                data[data>1] = 1
            elif property == 'CinCO':
                data = self.CinCO/self.z[:,2]
            elif property == 'nH':
                data = self.rho*config.UnitDensity_in_cgs * (1. - (self.z[:,0]+self.z[:,1])) / config.H_MASS
            elif property == 'nh':
                data = self.nh
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
            elif property == 'Z_O':
                data = self.z[:,4]/config.SOLAR_MASSFRAC[4]
            elif property == 'Z_O_gas':
                data = (self.z[:,4]-self.dz[:,4])/config.SOLAR_MASSFRAC[4]
            elif property == 'Z_C':
                data = self.z[:,2]/config.SOLAR_MASSFRAC[2]
            elif property == 'Z_C_gas':
                data = (self.z[:,2]-self.dz[:,2])/config.SOLAR_MASSFRAC[2]
            elif property == 'Z_Mg':
                data = self.z[:,6]/config.SOLAR_MASSFRAC[6]
            elif property == 'Z_Mg_gas':
                data = (self.z[:,6]-self.dz[:,6])/config.SOLAR_MASSFRAC[6]
            elif property == 'Z_Si':
                data = self.z[:,7]/config.SOLAR_MASSFRAC[7]
            elif property == 'Z_Si_gas':
                data = (self.z[:,7]-self.dz[:,7])/config.SOLAR_MASSFRAC[7]
            elif property == 'Z_Fe':
                data = self.z[:,10]/config.SOLAR_MASSFRAC[10]
            elif property == 'Z_Fe_gas':
                data = (self.z[:,10]-self.dz[:,10])/config.SOLAR_MASSFRAC[10]
            elif property == 'O/H':
                O = self.z[:,4]/config.ATOMIC_MASS[4]; H = (1-(self.z[:,0]+self.z[:,1]))/config.ATOMIC_MASS[0]
                data = 12+np.log10(O/H)
            elif property == 'O/H_offset':
                offset=0.3 # This is roughly difference in O/H_solar between AG89 (8.93) and Asplund+09 (8.69). Ma+16 finds FIRE gives 9.00.
                O = self.z[:,4]/config.ATOMIC_MASS[4]; H = (1-(self.z[:,0]+self.z[:,1]))/config.ATOMIC_MASS[0]
                data = 12+np.log10(O/H)-offset
            elif property == 'O/H_gas_offset':
                offset=0.3 # This is roughly difference in O/H_solar between AG89 (8.93) and Asplund+09 (8.69). Ma+16 finds FIRE gives 9.00.
                O = (self.z[:,4]-self.dz[:,4])/config.ATOMIC_MASS[4]; H = (1-(self.z[:,0]+self.z[:,1]))/config.ATOMIC_MASS[0]
                data = 12+np.log10(O/H)-offset
            elif property == 'O/H_gas':
                O = (self.z[:,4]-self.dz[:,4])/config.ATOMIC_MASS[4]; H = (1-(self.z[:,0]+self.z[:,1]))/config.ATOMIC_MASS[0]
                data = 12+np.log10(O/H)
            elif property == 'C/H':
                C = self.z[:,2]/config.ATOMIC_MASS[2]; H = (1-(self.z[:,0]+self.z[:,1]))/config.ATOMIC_MASS[0]
                data = 12+np.log10(C/H)
            elif property == 'C/H_gas':
                C = (self.z[:,2]-self.dz[:,2])/config.ATOMIC_MASS[2]; H = (1-(self.z[:,0]+self.z[:,1]))/config.ATOMIC_MASS[0]
                data = 12+np.log10(C/H)
            elif property == 'Mg/H':
                Mg = self.z[:,6]/config.ATOMIC_MASS[6]; H = (1-(self.z[:,0]+self.z[:,1]))/config.ATOMIC_MASS[0]
                data = 12+np.log10(Mg/H)
            elif property == 'Mg/H_gas':
                Mg = (self.z[:,6]-self.dz[:,6])/config.ATOMIC_MASS[6]; H = (1-(self.z[:,0]+self.z[:,1]))/config.ATOMIC_MASS[0]
                data = 12+np.log10(Mg/H)
            elif property == 'Si/H':
                Si = self.z[:,7]/config.ATOMIC_MASS[7]; H = (1-(self.z[:,0]+self.z[:,1]))/config.ATOMIC_MASS[0]
                data = 12+np.log10(Si/H)
            elif property == 'Si/H_gas':
                Si = (self.z[:,7]-self.dz[:,7])/config.ATOMIC_MASS[7]; H = (1-(self.z[:,0]+self.z[:,1]))/config.ATOMIC_MASS[0]
                data = 12+np.log10(Si/H)
            elif property == 'Fe/H':
                Fe = self.z[:,10]/config.ATOMIC_MASS[10]; H = (1-(self.z[:,0]+self.z[:,1]))/config.ATOMIC_MASS[0]
                data = 12+np.log10(Fe/H)
            elif property == 'Fe/H_gas':
                Fe = (self.z[:,10]-self.dz[:,10])/config.ATOMIC_MASS[10]; H = (1-(self.z[:,0]+self.z[:,1]))/config.ATOMIC_MASS[0]
                data = 12+np.log10(Fe/H)
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
            elif property == 'f_cold':
                data = np.sum(self.m[self.T<=1E3])/np.sum(self.m)
            elif property == 'f_warm':
                data = np.sum(self.m[(self.T<1E4) & (self.T>=1E3)])/np.sum(self.m)
            elif property == 'f_hot':
                data = np.sum(self.m[self.T>=1E4])/np.sum(self.m)
            else:
                print("Property %s given to Particle with ptype %i is not supported"%(property,self.ptype))
                return None

        elif self.ptype in [1,2,3]:
            if property=='M' or property=='M_dm' or property=='m':
                data = self.m*config.UnitMass_in_Msolar
            elif property=='h':
                data = self.h
            else:
                print("Property %s given to Particle with ptype %i is not supported"%(property,self.ptype))
                return None

        elif self.ptype==4:
            if property in ['M','M_star','M_stellar']:
                data = self.m*config.UnitMass_in_Msolar
            elif property in ['M_star_young','M_stellar_young','M_sfr']:
                # Assume young stars are < 10 Myr old
                data = self.m*config.UnitMass_in_Msolar
                age = self.age
                data[age>0.01] = 0.
            elif property == 'Z':
                data = self.z[:,0]/config.SOLAR_Z
            elif property == 'Z_all':
                data = self.z
            elif property == 'O/H':
                O = self.z[:,4]/config.ATOMIC_MASS[4]; H = (1-(self.z[:,0]+self.z[:,1]))/config.ATOMIC_MASS[0]
                data = 12+np.log10(O/H)
            elif property == 'age':
                data = self.age
            else:
                print("Property %s given to Particle with ptype %i is not supported"%(property,self.ptype))
                return None

        else:
            print("Property %s given to Particle with ptype %i is not supported"%(property,self.ptype))
            return None

        # Make sure to return a copy of the data
        return data.copy()