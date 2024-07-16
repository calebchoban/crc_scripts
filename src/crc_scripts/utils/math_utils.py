from .. import config
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

def weighted_percentile(a, percentiles=np.array([50, 16, 84]), weights=None, ignore_invalid=True):
    """
    Calculates percentiles associated with a (possibly weighted) array

    Parameters
    ----------
    a : ndarray
        The 1D input array from which to calculate percents
    percentiles : ndarray
        The percentiles to calculate (0.0 - 100.0)
    weights : ndarray, optional
        The weights to assign to values of a. Equal weighting if None
        is specified
    ignore_invalid : boolean, optional
        Set whether invalid values (inf,NaN) are not considered in calculation

    Returns
    -------
    values : ndarray
        The values associated with the specified percentiles.  
    """

    # First deal with empty array
    if len(a)==0:
        return np.full(len(percentiles), 0.)

    if weights is None:
        weights = np.ones(a.size)
    if ignore_invalid:
        mask_a = np.ma.masked_invalid(a) 
        weights = weights[~mask_a.mask]
        a = mask_a[~mask_a.mask]
        # Deal with the case that no data is valid
        if len(a) == 0:
            return np.zeros(len(percentiles))

    # Standardize and sort based on values in a
    idx = np.argsort(a)
    a_sort = a[idx]
    w_sort = weights[idx]

    # Get the percentiles for each data point in array
    p=w_sort.cumsum()
    p=p/p[-1]*100
    # Get the value of a at the given percentiles
    values=np.interp(percentiles, p, a_sort)
    return values


def bin_values(bin_data, data_vals, bin_lims, bin_nums=50, weight_vals=None, log=True):
    """
    Bins data_vals given bin_data

    Parameters
    ----------
    bin_data : ndarray
        Data over which binning will take place
    data_vals : ndarray
        Data to be binned
    bin_lims : ndarray
        Limits for bins
    bin_nums : int
        Number of bins
    weight_vals : ndarray, optional
        The weights to be sued when binning data_vals.  Equal weighting if None
        is specified
    log : boolean, optional
        Set if the binning is over log space

    Returns
    -------
    bin_vals: array
        Values of bins used
    mean_data : array
        50th percentile of data binned
    std_data : array
        16th and 84th percentiles of data binned

    """

    if log:
        bins = np.logspace(np.log10(bin_lims[0]),np.log10(bin_lims[1]),bin_nums)
    else:
        bins = np.linspace(bin_lims[0], bin_lims[1], bin_nums)

    bin_vals = (bins[1:] + bins[:-1]) / 2.
    digitized = np.digitize(bin_data,bins)

    mean_data = np.zeros(bin_nums - 1)
    # 16th and 84th percentiles
    std_data = np.zeros([bin_nums - 1,2])

    for i in range(1,len(bins)):
        if len(bin_data[digitized==i])==0:
            mean_data[i-1] = np.nan
            std_data[i-1,0] = np.nan; std_data[i-1,1] = np.nan;
            continue
        else:
            weights = weight_vals[digitized == i] if weight_vals is not None else None
            values = data_vals[digitized == i]
            mean_data[i-1],std_data[i-1,0],std_data[i-1,1] = weighted_percentile(values, weights=weights)

    return bin_vals, mean_data, std_data


# cosmic time in a flat cosmology
def quick_lookback_time(a, sp=None, redshift=False):
    """
    Quick calculation for the physical lookback time given a scale length or redshift.

    Parameters
    ----------
    a : float
        Scale length or redshift you want the look back time for.
    sp : Snapshot, optional
        Snapshot to pull cosmological constants from. If none is given these uses default values.
    redshift : bool, optional
        Set whether tou are giving a redshift or scale length.

    Returns
    -------
    time : float
        Calculated look back time.
    """

    if redshift:
        a = 1/(1+a)

    # Assume usual Hubble and omega values
    if sp == None:
        h = config.HUBBLE
        omega = config.OMEGA_MATTER
    else:
        h = sp.hubble
        omega = sp.omega
    
    x = omega / (1.0-omega) / (a*a*a)
    time = (2.0/(3.0*np.sqrt(1.0-omega))) * np.log(np.sqrt(x)/(-1.0+np.sqrt(1.+x)))
    time *= (13.777*(0.71/h)) # in Gyr
    
    return time

def get_time_conversion_spline(time_name_get, time_name_input, sp=None):
    """
    Calculates an interpolation spline from one time measure to the other. 
    Accepted time measures are 'time','time_lookback','redshift','reshift_plus_1'.

    Parameters
    ----------
    time_name_get : string
        Time measure you want to get the conversion for.
    time_name_input : string
        Time measure you are converting from.
    sp : Snapshot, optional
        Snapshot to pull cosmological constants from. If none is given these uses default values.

    Returns
    -------
    conv_func : function
        Conversion spline function.
    """

    a_vals = np.logspace(np.log10(0.01),np.log10(1),100)
    t_vals = quick_lookback_time(a_vals, sp=sp, redshift=False)
    universe_age = quick_lookback_time(1, sp=sp, redshift=False)

    if time_name_input=='time':
        x_vals = t_vals
    elif time_name_input=='time_lookback':
        x_vals = universe_age - t_vals
    elif time_name_input=='redshift':
        x_vals = 1./a_vals-1.
    elif time_name_input=='redshift_plus_1':
        x_vals = (1./a_vals-1.)+1
    else:
        x_vals = a_vals

    if time_name_get=='time':
        y_vals = t_vals
    elif time_name_get=='time_lookback':
        y_vals = universe_age - t_vals
    elif time_name_get=='redshift':
        y_vals = 1./a_vals-1.
    elif time_name_get=='redshift_plus_1':
        y_vals = (1./a_vals-1.)+1
    else:
        y_vals = a_vals

    conv_func = interp1d(x_vals,y_vals, kind='cubic',fill_value="extrapolate")

    return conv_func



def get_stellar_ages(sft, sp):
    """
    Calculates age of star particles given their formation time.

    Parameters
    ----------
    sft : string
        Formation time of star particle.
    sp : Snapshot
        Snapshot to pull cosmological constants from or to determine this isn't cosmological.

    Returns
    -------
    age : ndarray
        Age of star particle
    """

    if (sp.cosmological==1):
        t_form = quick_lookback_time(sft, sp=sp)
        t_now = quick_lookback_time(sp.time, sp=sp)
        age = t_now - t_form # in Gyr
    else:
        age = sp.time - sft # code already in Gyr
    
    return age


# calculate star formation history
# Assuming following units
# sft [ascale/Gyr for cosmological/non-cosmological]
# mass [M_solar]
# dt [Gyr]
def SFH(sft, m, sp, dt=0.01, cum=False):
    """
    Calculates the archeological star formation history from the given star particles formation times and masses.

    Parameters
    ----------
    sft : ndarray
        Formation times for star particles
    m : ndarray
        Masses of star particles
    sp : Snapshot
        Snapshot to pull cosmological constants from or to determine this isn't cosmological.
    dt : float, optional
        Time bin sizes in Gyr
    cum : bool, optional
        Make a cumulative SFH, i.e. archeological stellar mass evolution
    
    Returns
    -------
    time : ndarray
        Time array starting at first star formed and ending at last star formed.
    sfr : ndarray
        Star formation rate or cumulative stellar mass at each time.
    """

    
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
    time = np.linspace(tmin, tmax, 1000)
    if (cum):
        sfr = np.interp(time, tform_sorted, m_cum) # cumulative SFH, in Msun
    else:
        sfh_later = np.interp(time, tform_sorted, m_cum)
        sfh_former = np.interp(time-dt, tform_sorted, m_cum)
        sfr = (sfh_later-sfh_former)/dt/1E9 # in Msun per yr
    
    return time, sfr


# returns gas particles temperature in Kelvin assuming fully-atomic with no metal correction
def approx_gas_temperature(u, ne, keV=0):
    """
    Calculates the approximate temperature of gas particle given their internal energy and electron density.

    Parameters
    ----------
    u : ndarray
        Internal energy for gas particles
    ne : ndarray
        Electrong density for gas particles
    keV : bool, optional
        Output in keV instead of Kelvin
    
    Returns
    -------
    T : ndarray
        Temperature of gas particles
    """

    g_gamma= 5.0/3.0
    g_minus_1= g_gamma-1.0

    XH = 0.76
    YHe = (1.0-XH) / (4.0*XH)

    #return 1. / (XH*0.5 + (1-XH)/4. + 1./(16.+12.)) # This is now used in FIRE for the temperature floor
    mu = (1.0+4.0*YHe) / (1.0+YHe+ne)

    MeanWeight= mu*config.PROTONMASS
    T = MeanWeight/config.BoltzMann_ergs * g_minus_1 * u

    # do we want units of keV? 
    if (keV==1):
        T *= config.T_to_keV;

    return T


# returns the rotation matrix between two vectors. To be used to rotate galaxies
def calc_rotate_matrix(vec1, vec2):
    """"
    Calculates the rotation matrix between two unit vectors.

    Parameters
    ----------
    vec1 : ndarray
        First unit vector
    vec2 : ndarray
        Second unit vector you want to rotate vec1 to.

    Returns
    -------
    rotation_matrix : ndarray
        Rotation matrix between the two vectors
    """

    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def fit_exponential(x_data, y_data, guess=None, bounds=None):
    """"
    Fits exponential disk to galactocentric vs sufrace density data

    Parameters
    ----------
    x_data : ndarray
        Galactocentric radius data
    y_data : ndarray
        Surface density data
    guess : list, optional
        Inital guesses for curve_fit
    bouds : list, optional
        Bounds for values curve_fit can use

    Returns
    -------
    curve_fit : ndarray
        Parameters for fitted exponential
    """
    def exp_func(x, coeff, scale_l, offset):
        return coeff*np.exp(-x/scale_l)+offset

    return curve_fit(exp_func,x_data,y_data, p0=guess, bounds=bounds)

# This fits a sersic+exponential disk profile to galactocentric vs sufrace density data
def fit_bulge_and_disk(x_data, y_data, guess=None, bounds=None, bulge_profile='de_vauc', no_exp=False):
    """"
    Fits a sersic+exponential disk profile to galactocentric vs sufrace density data

    Parameters
    ----------
    x_data : ndarray
        Galactocentric radius data
    y_data : ndarray
        Surface density data
    guess : list, optional
        Inital guesses for curve_fit
    bouds : list, optional
        Bounds for values curve_fit can use
    bulge_profile : string, optional
        Type of function you want to fit ('sersic', de_vauc') for the bulge
    no_exp : bool, optional
        Set to not fit an exponential
    Returns
    -------
    curve_fit : ndarray
        Parameters for fitted profile
    """
    def sersic_and_exp_func(x, coeff1,coeff2,sersic_l,disk_l, sersic_index):
        return np.log10(coeff1*np.exp(-np.power(x/sersic_l,1./sersic_index))+coeff2*np.exp(-x/disk_l))
    def de_vaucouleurs_and_exp_func(x, coeff1,coeff2,sersic_l,disk_l):
        return np.log10(coeff1*np.exp(-np.power(x/sersic_l,1./4.))+coeff2*np.exp(-x/disk_l))
    def sersic_func(x, coeff1,sersic_l, sersic_index):
        return np.log10(coeff1*np.exp(-np.power(x/sersic_l,1./sersic_index)))

    y_data[y_data <= 0] = config.EPSILON
    if no_exp:
        return curve_fit(sersic_func, x_data, np.log10(y_data), p0=guess, bounds=bounds)
    if bulge_profile=='sersic':
        return curve_fit(sersic_and_exp_func,x_data,np.log10(y_data), p0=guess, bounds=bounds)
    if bulge_profile=='de_vauc':
        return curve_fit(de_vaucouleurs_and_exp_func, x_data, np.log10(y_data), p0=guess, bounds=bounds)


# Returns the dust metallicity from the total mass of dust grains by summing the mass in each grain size bin
def get_grain_mass(G):
    """"
    Returns the dust metallicity from the total mass of dust grains by summing the mass in each grain size bin

    Parameters
    ----------
    G : Particle
        Gas particle you want to check the grain metallicity for.

    Returns
    -------
    dust_z : ndarray
        Dust metallicty given by grain size bins
    """


    # Use so numpy-fu to do this in one line
    num_grains=G.get_property('grain_bin_num')
    slope = G.get_property('grain_bin_slope')
    alower = G.sp.Grain_Bin_Edges[:-1]
    aupper = G.sp.Grain_Bin_Edges[1:]
    acenter= G.sp.Grain_Bin_Centers
    bulk_dens=config.DUST_BULK_DENS / np.power(config.cm_to_um,3)
    bulk_dens=bulk_dens[np.newaxis,:,np.newaxis]
    total_grain_mass = np.sum(4*np.pi*bulk_dens/3*((num_grains/(4*(aupper-alower))-slope*acenter/4)*(pow(aupper,4)-pow(alower,4))+slope/5*(pow(aupper,5)-pow(alower,5))),axis=2)
    total_grain_mass *= config.grams_to_Msolar

    # Return the dust metallicity
    return total_grain_mass / G.get_property('m')[:,np.newaxis]

# Check if string is same as other string or in list of other strings in a case insensitive manner
def case_insen_compare(item1, item2):
    """"
    Check if string is same as other string or in list of other strings in a case insensitive manner

    Parameters
    ----------
    item1 : string
        String to look for
    item2 : list, string
        String or list of strings to check for item1

    Returns
    -------
    compare : bool
        Whether item1 is in item2
    """
    if type(item2) is str:
        item2=[item2]
    return item1.casefold() in map(str.casefold, item2)