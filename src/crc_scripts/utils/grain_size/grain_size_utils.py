import numpy as np
import pandas as pd
import os
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.integrate import quad

from ... import config
from ...config import dust_species_properties
from ...io.snapshot import Snapshot
from ...io.particle import Particle
from ..math_utils import weighted_percentile


def MRN_dnda(a):
    return np.power(a,-3.5)

def MRN_dmdloga(a, rho_c=1):
    return 4/3*np.pi * rho_c * np.power(a,4)*np.power(a,-3.5)

def lognorm_dnda(a, a_norm=0.1*config.um_to_cm, sigma_a=0.6):
    return 1/a * np.exp(-np.power(np.log(a/a_norm),2) / (2*sigma_a*sigma_a))


# Returns the mass in grain bins determined from their number and slope for the given particles in the given Particle object
def get_grain_bin_mass(particle: Particle):
    snap = particle.sp
    bin_nums = particle.get_property('grain_bin_num')
    bin_slopes = particle.get_property('grain_bin_slope')
    upper_edges = snap.Grain_Bin_Edges[1:]
    lower_edges = snap.Grain_Bin_Edges[:-1]
    bin_centers = snap.Grain_Bin_Centers

    species = ['silicates', 'carbonaceous', 'iron']
    spec_indices=[0,1,2]

    # Calculate grain bin mass from numbers and slopes
    grain_bin_mass = np.zeros((particle.npart, snap.Flag_DustSpecies, snap.Flag_GrainSizeBins),dtype='double')
    for i,spec in enumerate(species):
        spec_ind = spec_indices[i]
        spec_bin_numbers = bin_nums[:,spec_ind,:]; spec_bin_slopes = bin_slopes[:,spec_ind,:]
        spec_props = config.dust_species_properties(spec)
        rho_c = spec_props['rho_c']/(config.cm_to_um**3) # g/cm^3 to g/um^3 since grain radii are in um
        no_dust = (spec_bin_numbers <= 0)
        spec_bin_mass = 4*np.pi*rho_c/3*((spec_bin_numbers/(4*(upper_edges-lower_edges))-spec_bin_slopes*bin_centers/4)*(np.power(upper_edges,4)-np.power(lower_edges,4))+spec_bin_slopes/5*(np.power(upper_edges,5)-np.power(lower_edges,5)))
        spec_bin_mass[no_dust] = 0
        grain_bin_mass[:,spec_ind,:] = spec_bin_mass

    return grain_bin_mass


# Returns the slope in grain bins determined from their grain number and mass for the given particles in the given Particle object
def get_grain_bin_slope(particle: Particle):
    snap = particle.sp
    bin_nums = particle.get_property('grain_bin_num')
    bin_masses = particle.get_property('grain_bin_mass')
    upper_edges = snap.Grain_Bin_Edges[1:]
    lower_edges = snap.Grain_Bin_Edges[:-1]
    bin_centers = snap.Grain_Bin_Centers

    species = ['silicates', 'carbonaceous', 'iron']
    spec_indices=[0,1,2]

    # Calculate grain bin mass from numbers and slopes
    grain_bin_slopes = np.zeros((particle.npart, snap.Flag_DustSpecies, snap.Flag_GrainSizeBins),dtype='double')
    for i,spec in enumerate(species):
        spec_ind = spec_indices[i]
        spec_bin_numbers = bin_nums[:,spec_ind,:]; spec_bin_masses = bin_masses[:,spec_ind,:]
        spec_props = config.dust_species_properties(spec)
        rho_c = spec_props['rho_c']/(config.cm_to_um**3) # g/cm^3 to g/um^3 since grain radii are in um
        no_dust = (spec_bin_numbers <= 0) | (spec_bin_masses <= 0)

        spec_bin_slopes = (3*spec_bin_masses/(4*np.pi*rho_c)-spec_bin_numbers/(4*(upper_edges-lower_edges))*(pow(upper_edges,4)-pow(lower_edges,4))) / ((pow(upper_edges,5)-pow(lower_edges,5))/5-bin_centers/4*(pow(upper_edges,4)-pow(lower_edges,4)));
        spec_bin_slopes[no_dust] = 0
        grain_bin_slopes[:,spec_ind,:] = spec_bin_slopes

    return grain_bin_slopes



def get_grain_size_distribution(snap: Snapshot,
                                species: str ='silicates', 
                                mask: list = None, 
                                points_per_bin: int = 1, 
                                std_percentiles: list = [16, 84]):
    """
    Calculates the normalized grain size probability distribution (dn/da and dm/dloga) of a dust species from a snapshot. 
    Gives the mean and standard deviation of the distribution for all particles. 
    Note this is determined by calculating the normalized distributions for all particles and then calculating the percentiles
    with the dust species masses as weights.

    Parameters
    ----------
    snap : Snapshot
        Snapshot or Galaxy object from which particle data can be loaded
    species: str
        Name of species you want size distribution for. (silicates, carbonaceous, or iron)
    mask : list
        Boolean array to mask particles. Set to None for all particles.
    points_per_bin : int, optional
        Number of data points you want in each grain size bin. If 1, will use the center of each bin.
        Note this uses the bin slopes, so this won't be pretty to look at.
    std_percentiles : list, optional
        Percentiles of the standard deviation you want.

    Returns
    -------
    grain_size_points: list
        Grain size data points.
    percentile_dnda : list
        Median and percentiles of dn/da at corresponding grain size points.
    percentile_dmdloga : list 
        Median and percentiles of dm/dloga at corresponding grain size points.
    """	

    percentiles = [50] + std_percentiles # Add the median to the percentiles
    G = snap.loadpart(0)
    if mask is None: mask = np.ones(G.npart,dtype=bool)
    num_part = len(G.get_property('M_gas')[mask])
    bin_edges = snap.Grain_Bin_Edges
    bin_centers = snap.Grain_Bin_Centers
    num_bins = snap.Flag_GrainSizeBins
    if species == 'silicates': spec_ind = 0
    elif species == 'carbonaceous': spec_ind = 1
    elif species == 'iron': spec_ind = 2
    else: assert 0, "Dust species %s not supported"%species

    bin_nums = G.get_property('grain_bin_num')[mask,spec_ind]
    bin_slopes = G.get_property('grain_bin_slope')[mask,spec_ind]
    total_species_mass = G.get_property('M_gas')[mask]*G.get_property('dust_spec')[mask,spec_ind]*config.Msolar_to_g

    # internal density for given dust species
    # Physical properties of dust species needed for calculations
    spec_props = dust_species_properties(species)
    rho_c = spec_props['rho_c']/(config.cm_to_um**3) # g/cm^3 to g/um^3 since grain radii are in um
    

    grain_size_vals = np.zeros(points_per_bin*num_bins)
    dnda_vals = np.zeros([num_part,points_per_bin*num_bins])
    dmdloga_vals = np.zeros([num_part,points_per_bin*num_bins])

    # Need to normalize the distributions by total number and total mass, since we are only considering their shapes
    total_N = np.sum(bin_nums,axis=1)[:,np.newaxis]
    total_M = total_species_mass[:,np.newaxis]
    no_dust = (total_N[:,0] == 0) | (total_M[:,0] == 0)

    # Determine grain size, dn/da, and dm/dloga values for points in each bin
    for i in range(num_bins):
        bin_num = bin_nums[:,i,np.newaxis]; # Add extra dimension for numpy math below
        bin_slope = bin_slopes[:,i,np.newaxis]; 
        # If one point per bin, set it to the center of the bin
        if points_per_bin == 1: x_points = np.array([bin_centers[i]])
        else: x_points = np.logspace(np.log10(bin_edges[i]*1.02),np.log10(bin_edges[i+1]*0.98),points_per_bin) # shave off the very edges of each bin since they can be near zero
        grain_size_vals[i*points_per_bin:(i+1)*points_per_bin] = x_points

        dnda_vals[no_dust,i*points_per_bin:(i+1)*points_per_bin] = 0
        dmdloga_vals[no_dust,i*points_per_bin:(i+1)*points_per_bin] = 0
        dnda_vals[~no_dust,i*points_per_bin:(i+1)*points_per_bin] = (bin_num[~no_dust]/(bin_edges[i+1]-bin_edges[i])+bin_slope[~no_dust]*(x_points-bin_centers[i]))/total_N[~no_dust]
        dmdloga_vals[~no_dust,i*points_per_bin:(i+1)*points_per_bin] = (4/3*np.pi*rho_c*np.power(x_points,4)*(bin_num[~no_dust]/(bin_edges[i+1]-bin_edges[i])+bin_slope[~no_dust]*(x_points-bin_centers[i])))/total_M[~no_dust]

    # Determine percentile distribution values from all of the the particles
    # Weight each particle by their the total dust species mass
    weights = total_species_mass/config.Msolar_to_g # Convert to smaller units to prevent overflow
    percentile_dnda = np.zeros([len(percentiles),points_per_bin*num_bins])
    percentile_dmdloga = np.zeros([len(percentiles),points_per_bin*num_bins])
    # Get percentiles for each point in each bin
    for i in range(len(grain_size_vals)):
        percentile_dnda[:,i] = weighted_percentile(dnda_vals[:,i], percentiles=percentiles, weights=weights, ignore_invalid=True)
        percentile_dmdloga[:,i] = weighted_percentile(dmdloga_vals[:,i], percentiles=percentiles, weights=weights, ignore_invalid=True)

    return grain_size_vals, percentile_dnda, percentile_dmdloga



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

    

def calculate_extinction_curve(snap: Snapshot, 
                               species: str = 'silicates', 
                               mask: list = None, 
                               std_percentiles: list = [16, 84],
                               bin_subsamples: int = 1):
    """
    Calculates the median and percentile extinction curve normalized by the extinction 
    in the visible band (A_lambda / A_V) from the gas cell grain size distributions 
    in the given snapshot. Can specify only contributions from a given species 
    (silicates, carbonaceous, or iron) or the total extinction from all species. The median 
    and percentiles are calculated as such. A normalized extinction curve is calculated 
    for each gas cell given its grain size distribution for each dust species. 
    The median and percentiles are then calculated from all of the gas cells weighted by
    their total dust mass.

    Parameters
    ----------
    snap : snapshot/galaxy
        Snapshot or Galaxy object from which particle data can be loaded
    species: str
        Species you want to extinction curve for. (silicates, carbonaceous, or all). 
        Note this is still normalized by the total A_V from all species.
    mask : ndarray
        Boolean array to mask particles. Set to None for all particles.
    std_percentiles : list
        Standard deviation percentiles to be calculated from the extinction curves from all 
        particles used in extinction curve calculation. 
        Default is [16, 84].
    bin_subsamples : int
        Number of dn/da subsamples from each grain size bin to be used for calculating extinction. 
        Default of 1 means the grain size distribution at only the centers of each bin are used 
        to calculate the extinction curves. If set to N>1, the grain size distribution at N points 
        is used. This is useful when you have a small number of grain size bins 
        (i.e. your bins cover a large range in grain sizes).


    Returns
    -------
    wavelength_points: list
        Wavelength data points in micron.
    A_lambda_points : list
        Median and percentile A_lambda/A_V values at corresponding wavelength points.
    """	    

    N_wave_bins = 500 # Number of wavelength bins for interpolation
    percentiles = [50] + std_percentiles # Add the median to the percentiles

    lambda_V = 0.5470 # V band wavelength in microns
    # Need wavelengths for A_lambda values. Assuming all dust species tables have the same wavelengths
    # Make sure this is the same order as appears in the first subtable
    # WARNING: If using numpy.unique the wavelengths order in the table is not preserved
    optical_property = get_dust_optical_properties('silicates')
    unique_wavelengths = optical_property['w(micron)'].values[optical_property['radius(micron)']==np.min(optical_property['radius(micron)'])] # use the smallest grain radius table to get the corresponding Qext wavelengths
    # Extend the wavelength grid for interpolation of Qext
    unique_wavelengths = np.logspace(np.log10(np.min(unique_wavelengths)), np.log10(np.max(unique_wavelengths)), N_wave_bins)

    # Load snapshot gas particle data and grain size bin data
    G = snap.loadpart(0)
    if mask is None: mask = np.ones(G.npart,dtype=bool)
    num_part = len(G.get_property('M_gas')[mask])
    bin_nums = G.get_property('grain_bin_num')[mask]
    bin_slopes = G.get_property('grain_bin_slope')[mask]
    dust_masses = G.get_property('M_dust')[mask]
    bin_centers = snap.Grain_Bin_Centers
    bin_edges = snap.Grain_Bin_Edges
    num_bins = snap.Flag_GrainSizeBins

    dust_species = ['silicates', 'carbonaceous', 'iron']
    spec_indices = [0,1,2]
    num_species = len(dust_species)
    optical_properties = [get_dust_optical_properties('silicates'),
                          get_dust_optical_properties('carbonaceous'),
                          get_dust_optical_properties('silicates')] # Assuming iron has silicate properties
    

    # If species is specified we will exclude all other species from 
    # the A_lambda calculation but still include them for A_V normalization
    if species == 'silicates': 
        exclude_spec_ind = [1,2]
    elif species == 'carbonaceous': 
        exclude_spec_ind = [0,2]
    elif species == 'iron': 
        exclude_spec_ind = [0,1]
    elif species == 'all': 
        exclude_spec_ind = []
    else: assert 0, "Dust species not supported"

    # Determine grain size and dn/da values for each dust species
    grain_size_vals = np.zeros(bin_subsamples*num_bins)
    dnda_vals = np.zeros([num_part,num_species,bin_subsamples*num_bins])

    # Determine dn/da values for points in each bin for each dust species
    for i in spec_indices:
        spec_bin_nums = bin_nums[:,i]
        spec_bin_slopes = bin_slopes[:,i]
        spec_dnda_vals = np.zeros([num_part, num_bins*bin_subsamples])

        for j in range(num_bins):
            bin_num = spec_bin_nums[:,j]
            bin_slope = spec_bin_slopes[:,j]
            
            if bin_subsamples == 1: x_points = np.array([bin_centers[j]])
            else: x_points = np.logspace(np.log10(bin_edges[j]*1.02),np.log10(bin_edges[j+1]*0.98),bin_subsamples) # shave off the very edges of each bin since they can be near zero
            grain_size_vals[j*bin_subsamples:(j+1)*bin_subsamples] = x_points

            spec_dnda_vals[:,j*bin_subsamples:(j+1)*bin_subsamples] = (bin_num[:,np.newaxis]/(bin_edges[j+1]-bin_edges[j])+bin_slope[:,np.newaxis]*(x_points[np.newaxis,:]-bin_centers[j]))
            
        dnda_vals[:,i,:] = spec_dnda_vals
        

    # Calculate extinction coefficient interpolation functions for each dust species from Qext data tables
    spec_Qext = []
    for i in range(num_species):
        # Load in Q extinction data for the given species
        optical_property = optical_properties[i]
        Qext = optical_property['Q_ext'].values
        table_grain_radii = optical_property['radius(micron)'].values
        table_wavelengths = optical_property['w(micron)'].values
        # RGI interpolator expects the 0th dimension to be strictly in ascending order
        # Interpolation of 2 variable Qext function requires we reorganize Qext data into a 2D grid
        # of grain radii and wavelengths and a 2D matrix of Qext values corresponding to the grid points
        # Make 2D grid from grain radii and wavelengths
        unique_table_radii = np.sort(pd.unique(table_grain_radii))
        unique_table_wavelengths = np.sort(pd.unique(table_wavelengths))
        # Make 2D matrix of Qext values corresponding to the grid points
        Qext_matrix = np.zeros([len(unique_table_radii),len(unique_table_wavelengths)])
        for k in range(len(unique_table_radii)):
            for l in range(len(unique_table_wavelengths)):
                Qext_matrix[k,l] = Qext[(table_grain_radii==unique_table_radii[k]) & (table_wavelengths == unique_table_wavelengths[l])]
        # Create the interpolation function
        Qext = RGI((unique_table_radii,unique_table_wavelengths), Qext_matrix, method='cubic', bounds_error=False) 
        spec_Qext += [Qext]


    # Calculate the extinction curve for each particle
    A_lambda_total = np.zeros([num_part,N_wave_bins])
    A_V_total = np.zeros(num_part) # Extinction in V band (5470 Angstrom)
    for i,spec_ind in enumerate(spec_indices):
        # Calculate the dust species A_lambda and A_V 
        A_lambda_spec = np.zeros([num_part,N_wave_bins])
        A_V_spec =  np.zeros(num_part) # Extinction in V band (5470 Angstrom) for one species
        Qext = spec_Qext[i]
        spec_dnda = dnda_vals[:,i,:]

        # We are approximating the integral Qext(a,lambda) * dn/da(a) da from a_min to a_max 
        # as a sum over grain bins Qext(a_i,center,lambda) * dn/da(a_i,center) * (a_i,upper - a_i,lower)
        for j in range(num_bins):
            bin_upper = bin_edges[j+1]
            bin_lower = bin_edges[j]

            # Need to know the extent of the bin (or subsamples of the bin) and the centers of the bin
            # (or centers of the subsamples) to calculate the extinction curve
            in_bin_mask = (grain_size_vals >= bin_lower) & (grain_size_vals < bin_upper)
            grain_sizes_in_bin = grain_size_vals[in_bin_mask]
            if (len(grain_sizes_in_bin) == 1): # Only one point in each bin which is the center
                size_diff_in_bin = np.array([bin_upper-bin_lower])
                grain_centers_in_bin = grain_sizes_in_bin
            else:
                size_diff_in_bin = grain_sizes_in_bin[1:] - grain_sizes_in_bin[:-1]
                grain_centers_in_bin = (grain_sizes_in_bin[1:] + grain_sizes_in_bin[:-1])/2
            dnda_in_bin = spec_dnda[:,in_bin_mask]

            # Calculate A_lambda only for species we are not excluding
            if spec_ind not in exclude_spec_ind:
                grain_wave_vals = np.zeros([N_wave_bins,2])
                for l,grain_size_center in enumerate(grain_centers_in_bin):
                    grain_wave_vals[:,0] = grain_size_center
                    grain_wave_vals[:,1] = unique_wavelengths
                    A_lambda_spec += grain_size_center*grain_size_center*dnda_in_bin[:,l,np.newaxis]*size_diff_in_bin[l]*Qext([grain_wave_vals])[0][np.newaxis,:]
                    
            # Calculate A_V for all species since we normalize by total A_V and want to know the relative contributions of each species
            for l,grain_size_center in enumerate(grain_centers_in_bin):
                A_V_spec += grain_size_center*grain_size_center*dnda_in_bin[:,l]*size_diff_in_bin[l]*Qext([grain_size_center,lambda_V])[0]


        A_lambda_total += A_lambda_spec
        A_V_total += A_V_spec

    # Normalize by A_V for each particle
    A_lambda_norm = A_lambda_total/A_V_total[:,np.newaxis] 

    # Calculate the percentiles for each wavelength point across all particles
    percentile_A_lambda = np.zeros([len(percentiles),N_wave_bins])
    for i in range(N_wave_bins):
        weights = dust_masses # Weight each particle extinction by their the total dust mass
        A_lambda_vals = A_lambda_norm[:,i]
        percentile_A_lambda[:,i] = weighted_percentile(A_lambda_vals, percentiles=percentiles, weights=weights, ignore_invalid=True)


    return unique_wavelengths, percentile_A_lambda
    
