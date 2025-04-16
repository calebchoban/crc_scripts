import numpy as np
import pandas as pd
import os

from ... import config
from ...config import dust_species_properties
from ...io.snapshot import Snapshot
from ..math_utils import weighted_percentile


# Returns the mass in grain bins determined from their number and slope
def get_grain_bin_mass(snap: Snapshot, species='silicates'):
    G = snap.loadpart(0)
    bin_nums = G.get_property('grain_bin_num')
    bin_slopes = G.get_property('grain_bin_slope')
    bin_edges = snap.Grain_Bin_Edges
    bin_centers = snap.Grain_Bin_Centers
    num_bins = snap.Flag_GrainSizeBins

    if species == 'silicates': spec_ind = 0
    elif species == 'carbonaceous': spec_ind = 1
    elif species == 'iron': spec_ind = 2
    else: assert 0, "Dust species not supported"

    spec_props = dust_species_properties(species)
    rho_c = spec_props['rho_c']/(config.cm_to_um**3) # g/cm^3 to g/um^3 since grain radii are in um

    bin_masses = np.zeros([G.npart,num_bins])

    for i in range(num_bins):
        bin_num = bin_nums[:,spec_ind][:,i]; 
        bin_slope = bin_slopes[:,spec_ind][:,i]; 

        bin_upper = bin_edges[i+1]
        bin_lower = bin_edges[i]
        bin_center = bin_centers[i]
        bin_mass = 4*np.pi*rho_c/3*((bin_num/(4*(bin_upper-bin_lower))-bin_slope*bin_center/4)*(np.power(bin_upper,4)-np.power(bin_lower,4))+bin_slope/5*(np.power(bin_upper,5)-np.power(bin_lower,5)))

        bin_masses[:,i] = bin_mass

    return bin_masses


# Returns dnda and grain size data points for plotting
def get_grain_size_dist(snap, species='silicates', mask=None, mass=False, points_per_bin=1,percentiles = [50, 16, 84]):
    """
    Calculates the normalized grain size probability distribution (number or mass) of a dust species from a snapshot. 
    Gives the mean and standard deviation of the distrition for all particles. 

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
    # Physical properties of dust species needed for calculations
    spec_props = dust_species_properties(species)
    rho_c = spec_props['rho_c']/(config.cm_to_um**3) # g/cm^3 to g/um^3 since grain radii are in um

    if species == 'silicates': spec_ind = 0
    elif species == 'carbonaceous': spec_ind = 1
    elif species == 'iron': spec_ind = 2
    else: assert 0, "Dust species not supported"

    if mask is None: mask = np.ones(G.npart,dtype=bool)
    num_part = len(G.get_property('M_gas')[mask])

    grain_size_points = np.zeros(points_per_bin*num_bins)
    dist_points = np.zeros([num_part,points_per_bin*num_bins])

    # Need to normalize the distributions to one, so we are just considering their shapes
    # Add extra dimension for numpy math below
    total_N = np.sum(bin_nums[mask,spec_ind],axis=1)[:,np.newaxis]
    total_M = (G.get_property('M_gas')[mask]*G.get_property('dust_spec')[mask,spec_ind]*config.Msolar_to_g)[:,np.newaxis]
    no_dust = (total_N[:,0] == 0) | (total_M[:,0] == 0)

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

        dist_points[no_dust,i*points_per_bin:(i+1)*points_per_bin] = 0
        if not mass:
            dist_points[~no_dust,i*points_per_bin:(i+1)*points_per_bin] = (bin_num[~no_dust]/(bin_edges[i+1]-bin_edges[i])+bin_slope[~no_dust]*(x_points-bin_centers[i]))/total_N[~no_dust]
        else:
            dist_points[~no_dust,i*points_per_bin:(i+1)*points_per_bin] = (4/3*np.pi*rho_c*np.power(x_points,4)*(bin_num[~no_dust]/(bin_edges[i+1]-bin_edges[i])+bin_slope[~no_dust]*(x_points-bin_centers[i])))/total_M[~no_dust]

    # If we have more than one particle want to return an average distribution
    if num_part > 1:
        # Weight each particle by their the total dust species mass
        weights = G.get_property('M_gas')[mask] * G.get_property('dust_spec')[mask,spec_ind]
        mean_dist_points = np.zeros(len(grain_size_points)); std_dist_points = np.zeros([len(grain_size_points),2]);
        # Get the mean and std for each x point
        for i in range(len(grain_size_points)):
            points = dist_points[:,i]
            mean_dist_points[i], std_dist_points[i,0], std_dist_points[i,1] = weighted_percentile(points, percentiles=np.array(percentiles), weights=weights, ignore_invalid=True)
        return grain_size_points, mean_dist_points, std_dist_points
    else: 
        std_dist_points = np.array([dist_points[0],dist_points[0]])
        return grain_size_points, dist_points[0], std_dist_points # Get rid of extra dimension if only one particle




def get_dust_optical_properties(species: str):
    """
    Returns a table of optical properties (Qabs, Qscat, Qext) vs wavelength (w(micron)) for the given dust species for various grain sizes (radius(micron)).

    Parameters:
    - species (str): The type of dust species. Supported values are 'silicates' and 'carbonaceous'.

    Returns:
    - optical_properties (pd.DataFrame): Optical properties vs wavelength of the dust species at specific grain sizes.

    Raises:
    - AssertionError: If the given species is not supported.
    """

    data_dirc = os.path.dirname(__file__) + '/data/'
    if species == 'silicates':
        file_name = 'silicates.dat'
    elif species == 'carbonaceous':
        file_name = 'graphite.dat'
    else:
        assert 0, "Species type not supported"

    # Initialize variables
    tables = []
    current_table = []
    header_found = False
    radius_values = []

    # Open the file and process it line by line
    with open(data_dirc+file_name, 'r') as file:
        for line in file:
            # Check if the line contains the header for a new table
            if "w(micron)" in line:
                # If a table is already being processed, save it
                if current_table:
                    # Add the radius as a column to the current table
                    df = pd.DataFrame(current_table, columns=['w(micron)', 'Q_abs', 'Q_sca', 'g=<cos>'])
                    df['radius(micron)'] = radius_values[-1]  # Use the last radius value
                    df['Q_ext'] = df['Q_abs'] + df['Q_sca']  # Add Q_ext column
                    tables.append(df)
                    current_table = []
                header_found = True  # Mark that a header has been found
            elif "radius(micron)" in line:  # Extract radius value
                try:
                    radius = float(line.split('=')[0].strip())
                    radius_values.append(radius)
                except ValueError:
                    pass
            elif header_found and line.strip():  # Process data lines after the header
                try:
                    current_table.append([float(x) for x in line.split()])
                except ValueError:
                    # Skip lines that cannot be parsed as data
                    pass

        # Add the last table if it exists
        if current_table:
            df = pd.DataFrame(current_table, columns=['w(micron)', 'Q_abs', 'Q_sca', 'g=<cos>'])
            df['radius(micron)'] = radius_values[-1]  # Use the last radius value
            df['Q_ext'] = df['Q_abs'] + df['Q_sca']  # Add Q_ext column
            tables.append(df)


    # Merge all tables into one
    optical_properties = pd.concat(tables, ignore_index=True)

    return optical_properties