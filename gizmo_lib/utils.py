import numpy as np
import h5py
import os
import ctypes

# This is a dummy class.
class DUMMY:
    
    def __init__(self, hubble=0.68, omega=0.31):
        
        self.cosmological = 1
        self.hubble = hubble
        self.omega = omega
        
        return


sp = DUMMY(hubble=0.68, omega=0.31)


# cosmic time in a flat cosmology
def quick_lookback_time(a, sp=sp):
    
    h = sp.hubble
    omega = sp.omega
    
    x = omega / (1.0-omega) / (a*a*a)
    t = (2.0/(3.0*np.sqrt(1.0-omega))) * np.log(np.sqrt(x)/(-1.0+np.sqrt(1.+x)))
    t *= (13.777*(0.71/h)) # in Gyr
    
    return t


# calculate stellar ages
def get_stellar_ages(sft, sp=sp):
    
    if (sp.cosmological==1):
        t_form = quick_lookback_time(sft, sp=sp)
        t_now = quick_lookback_time(sp.time, sp=sp)
        age = t_now - t_form # in Gyr
    else:
        age = sp.time - sft # code already in Gyr
    
    return age


# calculate star formation history
def SFH(sft, m, dt=0.01, cum=0, sp=sp):
    
    if (sp.cosmological==1):
        tform = quick_lookback_time(sft, sp=sp)
    else:
        tform = sft

    # cumulative SFH from all particles
    index = np.argsort(tform)
    tform_sorted, m_sorted = tform[index], m[index]
    m_cum = np.cumsum(m_sorted)

    # get a time grid
    tmin, tmax = np.min(tform), np.max(tform)
    t = np.linspace(tmin, tmax, 1000)
    if (cum==1):
        sfr = 1.0e10*np.interp(t, tform_sorted, m_cum) # cumulative SFH, in Msun
    else:
        sfh_later = np.interp(t, tform_sorted, m_cum)
        sfh_former = np.interp(t-dt, tform_sorted, m_cum)
        sfr = 10.0*(sfh_later-sfh_former)/dt # in Msun per yr
    
    return t, sfr


# gas mean molecular weight
def gas_mu(ne):
    
    XH = 0.76
    YHe = (1.0-XH) / (4.0*XH)
    
    return (1.0+4.0*YHe) / (1.0+YHe+ne)


def gas_temperature(u, ne, keV=0):
    
    gamma = 5.0/3.0
    mH = 1.6726e-24
    kB = 1.3806e-16
    
    mu = gas_mu(ne);
    T = mu*mH*(gamma-1.0)*(u*1.e10)/kB
    
    # in units keV
    if (keV==1):
        keV = 1.60218e-9
        T *= (kB/keV)
    
    return T


# smoothing length for collisionless particles
import gadget_lib.load_stellar_hsml as starhsml

def get_particle_hsml(x, y, z, DesNgb=32, Hmax=0.):
    return starhsml.get_particle_hsml(x, y, z, DesNgb=DesNgb, Hmax=Hmax)


# get luminosity-to-mass ratio for a stellar population
import colors

def colors_table(age_in_Gyr, metallicity_in_solar_units, band=0, SALPETER_IMF=0,
                 CHABRIER_IMF=1, UNITS_SOLAR_IN_BAND=1):

    return colors.colors_table(age_in_Gyr, metallicity_in_solar_units, band=band,
                           SALPETER_IMF=SALPETER_IMF, CHABRIER_IMF=CHABRIER_IMF,
                           UNITS_SOLAR_IN_BAND=UNITS_SOLAR_IN_BAND)


# load AHF particles
def loadAHFpart(PFile, NDummy, NPart):
    
    # call the C function
    exec_call = os.environ['PYLIB'] + "/clib/loadAHFpart/loadAHFpart.so"
    load_routine = ctypes.cdll[exec_call]
            
    # prepare the returned arrays
    out_cast = ctypes.c_int*NPart
    PartID_ctype = out_cast()
    PType_ctype = out_cast()
                    
    # call the routine
    load_routine.loadAHFpart(PFile, ctypes.c_int(NDummy),
                             ctypes.c_int(NPart),
                             PartID_ctype, PType_ctype)
    PartID = np.ctypeslib.as_array(PartID_ctype)
    PType = np.ctypeslib.as_array(PType_ctype)
                                
    return PartID, PType


def match(x, y, bool=False):
    
    import ctypes

    a = np.copy(x); a = np.array(a, dtype='int32')
    b = np.copy(y); b = np.array(b, dtype='int32')
    aindex = np.argsort(a); asorted = a[aindex]; alen = len(a)
    bindex = np.argsort(b); bsorted = b[bindex]; blen = len(b)
                    
    if ((alen==0) | (blen==0)):
        subx = np.array([], dtype='int32'); suby = np.array([],dtype='int32')
    else:
        asorted_ctype = asorted.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        bsorted_ctype = bsorted.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        alen_ctype = ctypes.c_int(alen)
        blen_ctype = ctypes.c_int(blen)
        suba_ctype = (ctypes.c_int*alen)()
        subb_ctype = (ctypes.c_int*blen)()
    
        exec_call = os.environ['PYLIB'] + "/pylib/clib/match/match.so"
        match_routine = ctypes.cdll[exec_call]
        match_routine.match(asorted_ctype, bsorted_ctype,
                            alen_ctype, blen_ctype,
                            suba_ctype, subb_ctype)
        
        suba = np.ctypeslib.as_array(suba_ctype).reshape(-1)
        subb = np.ctypeslib.as_array(subb_ctype).reshape(-1)
        subx = aindex[suba>0]; suby = bindex[subb>0]
        
    if (bool):
        xbool = np.zeros(alen, dtype='bool')
        ybool = np.zeros(blen, dtype='bool')
        xbool[subx] = True; ybool[suby] = True
        return xbool, ybool
        
    return subx, suby
