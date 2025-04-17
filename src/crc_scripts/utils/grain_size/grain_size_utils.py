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
    Gives the mean and standard deviation of the distribution for all particles. 

    Parameters
    ----------
    snap : snapshot/galaxy
        Snapshot or Galaxy object from which particle data can be loaded
    species: str
        Name of species you want size distribution for. (silicates, carbonaceous, or iron)
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

    # The optical properties files are formatted as tables of Qsca and Qabs vs wavelength for different grain sizes
    # General format is 
    # 1.000E-03 = radius(micron) Astronomical silicate, smoothed UV      
    # w(micron)  Q_abs     Q_sca     g=<cos>
    # ...        ...       ...       ...
    # Need to open the file and read line by line and then merge into one larage table since this is not a standard table format
    with open(data_dirc+file_name, 'r') as file:
        for line in file:
            if "radius(micron)" in line:  # Extract radius value before header
                try:
                    radius = float(line.split('=')[0].strip())
                    radius_values.append(radius)
                    header_found=True
                except ValueError:
                    pass
            elif "w(micron)" in line: # Skip actual header
                continue
            elif header_found and line.strip():  # If line isn't whitespace then it must be a line from the table
                try:
                    current_table.append([float(x) for x in line.split()])
                except ValueError:
                    # Skip lines that cannot be parsed as data
                    pass
            elif header_found and not line.strip(): # End of table is whitespace
                df = pd.DataFrame(current_table, columns=['w(micron)', 'Q_abs', 'Q_sca', 'g=<cos>'])
                df['radius(micron)'] = radius_values[-1]  # Use the last radius value
                df['Q_ext'] = df['Q_abs'] + df['Q_sca']  # Add Q_ext column
                tables.append(df)
                current_table = []
                header_found = False

        # Add the last table if it exists
        if current_table:
            df = pd.DataFrame(current_table, columns=['w(micron)', 'Q_abs', 'Q_sca', 'g=<cos>'])
            df['radius(micron)'] = radius_values[-1]  # Use the last radius value
            df['Q_ext'] = df['Q_abs'] + df['Q_sca']  # Add Q_ext column
            tables.append(df)

    # Merge all tables into one
    optical_properties = pd.concat(tables, ignore_index=True)

    return optical_properties


def get_extinction_curve(snap, species='silicates', mask=None, percentiles = [50, 16, 84]):
    """
    Calculates the extinction curve normalized by Av for each gas particle from a snapshot. 
    Gives the mean and standard deviation of the curves for all particles. 

    Parameters
    ----------
    snap : snapshot/galaxy
        Snapshot or Galaxy object from which particle data can be loaded
    species: str
        Species you want to extinction curve for. (silicates, carbonaceous, or all)
    mask : ndarray
        Boolean array to mask particles. Set to None for all particles.
    mass : bool
        Return grain mass probabiltiy distribution instead of grain number.
    percentiles : list
        Percentiles to calculate for the extinction curve. Default is [50, 16, 84].

    Returns
    -------
    wavelength_points: ndarray
        Wavelength data points in micron.
    mean_curve_points : ndarray
        Mean A_lambda/A_V values at correspoinding wavelength points.
    std_curve_points : ndarray
        Standard deviation values of A_lambda/A_V.
    """	

    # Find the closest wavelength to the V band wavelength and the closest radii to grain size bin centers
    # from our snapshot in the optical properties table 
    lambda_V = 0.5470 # V band wavelength in microns
    # Need wavelengths for A_lambda values. Assuming all dust species tables have the same wavelengths
    # Make sure this is the same order as appears in the first subtable
    # WARNING: If using numpy.unique the wavelengths order in the table is no preserved
    optical_property = get_dust_optical_properties('silicates')
    unique_wavelengths = optical_property['w(micron)'].values[optical_property['radius(micron)']==1E-3] # 1E-3 is the first radius in the table

    # Load snapshot gas particle data and grain size bin data
    G = snap.loadpart(0)
    bin_nums = G.get_property('grain_bin_num')
    bin_centers = snap.Grain_Bin_Centers
    num_bins = snap.Flag_GrainSizeBins

    if species == 'silicates': 
        spec_ind = [0]
        optical_properties = [get_dust_optical_properties(species)]
    elif species == 'carbonaceous': 
        spec_ind = [1]
        optical_properties = [get_dust_optical_properties(species)]
    elif species == 'iron': 
        spec_ind = [2]
        optical_properties = [get_dust_optical_properties(species)]
    elif species == 'all': 
        spec_ind = [0,1,2]
        optical_properties = [get_dust_optical_properties('silicates'),
                              get_dust_optical_properties('carbonaceous'),
                              get_dust_optical_properties('silicates')] # Assuming iron has silicate properties
    else: assert 0, "Dust species not supported"

    # Apply given mask or use all particles
    if mask is None: mask = np.ones(G.npart,dtype=bool)
    num_part = len(G.get_property('M_gas')[mask])

    A_lambda_total = np.zeros([num_part,len(unique_wavelengths)])
    A_V_total = np.zeros([num_part]) # Extinction in V band (5470 Angstrom)

    for i,spec in enumerate(spec_ind):
        # Load in Qextinction data for the given species
        optical_property = optical_properties[i]
        Qext = optical_property['Q_ext'].values
        grain_radii = optical_property['radius(micron)'].values
        wavelengths = optical_property['w(micron)'].values
        # Find the closest radii in the optical properties table to the bin centers
        unique_radii = pd.unique(grain_radii)
        closest_indices = [np.abs(unique_radii - bin_center).argmin() for bin_center in bin_centers]
        closest_radii = unique_radii[closest_indices]
        # Find the closest wavelength in the optical properties table to the V band wavelength
        unique_wavelengths = pd.unique(wavelengths)
        V_wavelength = unique_wavelengths[np.abs(unique_wavelengths - lambda_V).argmin()]


        spec_bin_num = bin_nums[mask,spec]

        A_lambda_spec = np.zeros([num_part,len(unique_wavelengths)])
        A_V_spec = np.zeros([num_part]) # Extinction in V band (5470 Angstrom) for one species

        for j in range(num_bins):
            bin_num = spec_bin_num[:,j]
            bin_center = bin_centers[j]
            closest_radius = closest_radii[j]

            Qext_acenter = Qext[grain_radii==closest_radius]
            Qext_V_acenter = Qext[(grain_radii==closest_radius) & (wavelengths == V_wavelength)]

            A_lambda_spec += bin_center*bin_center*bin_num[:,np.newaxis]*Qext_acenter # Add extra dimension for numpy math below
            A_V_spec += bin_center*bin_center*bin_num*Qext_V_acenter

        A_lambda_total += A_lambda_spec
        A_V_total += A_V_spec

    A_lambda_norm = A_lambda_total/A_V_total[:,np.newaxis] # Normalize by A_V

    # If we have more than one particle want to return an average extinction
    if num_part > 1:
        # Weight each particle by their the total dust species mass
        if species == 'all':
            weights = G.get_property('M_dust')[mask]
        else:
            weights = G.get_property('M_gas')[mask] * G.get_property('dust_spec')[mask,spec_ind[0]]
        mean_dist_points = np.zeros(len(unique_wavelengths)); std_dist_points = np.zeros([len(unique_wavelengths),2]);
        # Get the mean and std for each x point
        for i in range(len(unique_wavelengths)):
            points = A_lambda_norm[:,i]
            mean_dist_points[i], std_dist_points[i,0], std_dist_points[i,1] = weighted_percentile(points, percentiles=np.array(percentiles), weights=weights, ignore_invalid=True)
        return unique_wavelengths, mean_dist_points, std_dist_points
    else: 
        std_dist_points = np.array([A_lambda_norm[0],A_lambda_norm[0]])
        return unique_wavelengths, A_lambda_norm[0], std_dist_points # Get rid of extra dimension if only one particle