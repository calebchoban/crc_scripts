import numpy as np
from scipy.integrate import quad
from .. import config
from ..utils.math_utils import weighted_percentile


def MRN_dnda(a):
    return np.power(a,-3.5)

def MRN_dmda(a):
    return 4/3*np.pi * 1 * np.power(a,3)*np.power(a,-3.5)

def MRN_dmda_update(a,da):
    return 4/3*np.pi * 1 * np.power(a+da,3)*np.power(a,-3.5)

# Determines the change in grain distribution given a constant change in grain size
def change_in_grain_distribution(da,amin=1E-9,amax=5E-6,bin_num=1000):
    # Assume MRN distribution 
    a = np.logspace(np.log10(amin),np.log10(amax),bin_num+1)
    # Append bins beyond min and max
    a=np.append(a,np.inf); a=np.append(-np.inf,a)
    bin_num+=2
    N_bin = np.zeros(bin_num); M_bin = np.zeros(bin_num)
    N_update = np.zeros(bin_num); M_update = np.zeros(bin_num)
    for i in range(bin_num):
        # Nothing outside of the min/max size range
        if i==0 or i == bin_num-1:
            N_bin[i]=0
            M_bin[i]=0
        else:
            N_bin[i] = quad(MRN_dnda,a[i],a[i+1])[0]
            M_bin[i] = quad(MRN_dmda,a[i],a[i+1])[0]

    for j in range(bin_num):
        aj_upper = a[j+1]; aj_lower = a[j];
        #print('j',j,aj_lower,aj_upper)
        for i in range(bin_num):
            ai_upper = a[i+1]; ai_lower = a[i];
            #print('i',i,ai_lower,ai_upper)
            intersect = [np.max([aj_lower-da, ai_lower]), np.min([aj_upper-da, ai_upper])]
            if intersect[0]>intersect[1] or intersect[0]>amax or intersect[1]<amin: 
                #print('no intersect')
                continue
            else:
               # Nothing beyond min/max grain size
               #print('int before',intersect)
               intersect[0]=np.max([intersect[0],amin])
               intersect[1]=np.min([intersect[1],amax])
               #print('int',intersect)
               N_update[j] += quad(MRN_dnda,intersect[0],intersect[1])[0]
               M_update[j] += quad(MRN_dmda_update,intersect[0],intersect[1],args=(da))[0]

    #print(N_bin,N_update)
    print(np.sum(N_bin),np.sum(N_update))
    #print(M_bin,M_update)
    print(np.sum(M_bin),np.sum(M_update))
    print('Change in number:',np.sum(N_update[1:-1])/np.sum(N_bin))
    print('Change in mass:',np.sum(M_update[1:-1])/np.sum(M_bin))


# Returns dnda and grain size data points for plotting
def get_grain_size_dist(snap, spec_ind, mask=None, mass=False, points_per_bin=1):
    """
    Calculates the grain size probability distribution (number or mass) of a dust species from a snapshot. 
    Specifically gives the mean and standard deviation. 

    Parameters
    ----------
    snap : snapshot/galaxy
        Snapshot or Galaxy object from which particle data can be loaded
    spec_ind: int
        Number for dust species you want the distribution for.
    mask : ndarray
        Boolean array to mask particles. Set to None for all particles.
    mass : bool
        Return grain mass probabiltiy distribution instead of grain number.
    points_per_bin : int
        Number of data points you want in each grain size bin. If 1, will use the center of each bin.
        Note this uses the bin slopes, so this won't always be pretty to look at.

    Returns
    -------
    grain_size_points: ndarray
        Grain size data points.
    mean_dist_points : ndarray
        Mean dn/da or dm/da values at correspoinding grain size points.
    std_dist_points : ndarray
        Standard deviation values of dn/da or dm/da.
    """	

    G = snap.loadpart(0)
    bin_nums = G.get_property('grain_bin_num')
    bin_slopes = G.get_property('grain_bin_slope')
    bin_edges = snap.Grain_Bin_Edges
    bin_centers = snap.Grain_Bin_Centers
    num_bins = snap.Flag_GrainSizeBins
    # internal density for given dust species
    bulk_dens = config.DUST_BULK_DENS[spec_ind];
    if mask is None: mask = np.ones(G.npart,dtype=bool)
    num_part = len(G.get_property('M_gas')[mask])
    


    grain_size_points = np.zeros(points_per_bin*num_bins)
    dist_points = np.zeros([num_part,points_per_bin*num_bins])

    # Need to normalize the distributions to one, so we are just considering their shapes
    # Add extra dimension for numpy math below
    total_N = np.sum(bin_nums[mask,spec_ind],axis=1)[:,np.newaxis]
    total_M = (G.get_property('M_gas')[mask]*G.get_property('dust_spec')[mask,spec_ind]*config.Msolar_to_g)[:,np.newaxis]

    for i in range(num_bins):
        bin_num = bin_nums[mask,spec_ind][:,i]; 
        bin_slope = bin_slopes[mask,spec_ind][:,i]; 
        # Add extra dimension for numpy math below
        bin_num = bin_num[:,np.newaxis]
        bin_slope = bin_slope[:,np.newaxis]
        
        # If one point per bin, set it to the center of the bin
        if points_per_bin == 1: x_points = np.array([bin_centers[i]])
        else: x_points = np.logspace(np.log10(bin_edges[i]*1.02),np.log10(bin_edges[i+1]*0.98),points_per_bin) # shave off the very edges of each bin since they can be near zero
        grain_size_points[i*points_per_bin:(i+1)*points_per_bin] = x_points

        if not mass:
            dist_points[:,i*points_per_bin:(i+1)*points_per_bin] = (bin_num/(bin_edges[i+1]-bin_edges[i])+bin_slope*(x_points-bin_centers[i]))/total_N
        else:
            dist_points[:,i*points_per_bin:(i+1)*points_per_bin] = (4/3*np.pi*bulk_dens*np.power(x_points,4)*(bin_num/(bin_edges[i+1]-bin_edges[i])+bin_slope*(x_points-bin_centers[i])))/total_M

    # If we have more than one particle want to return an average distribution
    if num_part > 1:
        # Weight each particle by their the total dust species mass
        weights = G.get_property('M_gas')[mask] * G.get_property('dust_spec')[mask,spec_ind]
        mean_dist_points = np.zeros(len(grain_size_points)); std_dist_points = np.zeros([len(grain_size_points),2]);
        # Get the mean and std for each x point
        for i in range(len(grain_size_points)):
            points = dist_points[:,i]
            mean_dist_points[i], std_dist_points[i,0], std_dist_points[i,1] = weighted_percentile(points, percentiles=np.array([50, 16, 84]), weights=weights, ignore_invalid=True)
        return grain_size_points, mean_dist_points, std_dist_points
    else: 
        std_dist_points = np.array([dist_points[0],dist_points[0]])
        return grain_size_points, dist_points[0], std_dist_points # Get rid of extra dimension if only one particle
