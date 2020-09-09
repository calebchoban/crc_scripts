import config
import numpy as np


def weighted_percentile(a, percentiles=np.array([50, 16, 84]), weights=None):
    """
    Calculates percentiles associated with a (possibly weighted) array

    Parameters
    ----------
    a : array-like
        The input array from which to calculate percents
    percentiles : array-like
        The percentiles to calculate (0.0 - 100.0)
    weights : array-like, optional
        The weights to assign to values of a.  Equal weighting if None
        is specified

    Returns
    -------
    values : np.array
        The values associated with the specified percentiles.  
    """

    # First deal with empty array
    if len(a)==0:
        return np.full(len(percentiles), np.nan)

    # Standardize and sort based on values in a
    percentiles = percentiles
    if weights is None:
        weights = np.ones(a.size)
    idx = np.argsort(a)
    a_sort = a[idx]
    w_sort = weights[idx]

    # Get the percentiles for each data point in array
    p=1.*w_sort.cumsum()/w_sort.sum()*100
    # Get the value of a at the given percentiles
    values=np.interp(percentiles, p, a_sort)
    return values


def set_labels(axis, xlabel, ylabel):
    """
    Sets the labels and ticks for the given axis.

    Parameters
    ----------
    axis : Matplotlib axis
        Axis of plot
    xlabel : array-like
        X axis label
    ylabel : array-like, optional
        Y axis label

    Returns
    -------
    None
    """
    axis.set_xlabel(xlabel, fontsize = Large_Font)
    axis.set_ylabel(ylabel, fontsize = Large_Font)
    axis.minorticks_on()
    axis.tick_params(axis='both',which='both',direction='in',right=True, top=True)
    axis.tick_params(axis='both', which='major', labelsize=Small_Font)
    axis.tick_params(axis='both', which='minor', labelsize=Small_Font)


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


# returns gas particles temperature in Kelvin
def gas_temperature(u, ne, keV=0):

    g_gamma= 5.0/3.0
    g_minus_1= g_gamma-1.0

    mu = gas_mu(ne);
    MeanWeight= mu*config.PROTONMASS
    T= MeanWeight/config.BoltzMann_ergs * g_minus_1 * u * config.UnitVelocity_in_cm_per_s**2

    # do we want units of keV?  (0.001 factor converts from eV to keV)
    if (keV==1):
        T *= config.T_to_keV;

    return T


# returns H1,H2,and Hion masses
def calc_fH2(G):
    # Analytic calculation of molecular hydrogen from Krumholz et al. (2018)

    Z = G['z'][:,0] #metal mass (everything not H, He)
    # dust mean mass per H nucleus
    mu_H = 2.3E-24# grams
    # standard effective number of particle kernel neighbors defined in parameters file
    N_ngb = 32.
    # Gas softening length
    hsml = G['h']*config.UnitLength_in_cm
    density = G['rho']*config.UnitDensity_in_cgs

    sobColDens = np.multiply(hsml,density) / np.power(N_ngb,1./3.) # Cheesy approximation of column density

    #  dust optical depth 
    tau = np.multiply(sobColDens,Z*1E-21/SOLAR_Z)/mu_H
    tau[tau==0]=EPSILON #avoid divide by 0

    chi = 3.1 * (1+3.1*np.power(Z/SOLAR_Z,0.365)) / 4.1 # Approximation

    s = np.divide( np.log(1+0.6*chi+0.01*np.power(chi,2)) , (0.6 *tau) )
    s[s==-4.] = -4.+EPSILON # Avoid divide by zero
    fH2 = np.divide((1 - 0.5*s) , (1+0.25*s)) # Fraction of Molecular Hydrogen from Krumholz & Knedin
    fH2[fH2<0] = 0 #Nonphysical negative molecular fractions set to 0
    
    return fH2


# returns the rotation matrix between two vectors. To be used to rotate galaxies
def calc_rotate_matrix(vec1, vec2):
    """"
    Gives the rotation matrix between two unit vectors
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix