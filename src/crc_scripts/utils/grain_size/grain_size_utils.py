
# Need this to avoid circular imports during type checking
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ...io.snapshot import Snapshot
    from ...io.galaxy import Halo
    from ...io.particle import Particle

import numpy as np
import pandas as pd
import os
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.special import erfc,erf

from ... import config
from ...config import dust_species_properties
from ..math_utils import weighted_percentile
from ...analytical_models.grain_size_evo import grain_relative_velocity, v_coagulation, m_shatter, m_coagulation




def MRN_dnda(a):
    return np.power(a,-3.5)

def MRN_dmdloga(a, rho_c=1):
    return 4/3*np.pi * rho_c * np.power(a,4)*np.power(a,-3.5)

def lognorm_dnda(a, a_norm=0.1*config.um_to_cm, sigma_a=0.6):
    return 1/a * np.exp(-np.power(np.log(a/a_norm),2) / (2*sigma_a*sigma_a))


def get_grain_bin_mass(particle: Particle):
    """
    Calculates the mass of dust grains (grams) in size bins for the given gas particles.

    Parameters:
    -----------
    particle : Particle
        An instance of the Particle class containing gas properties.

    Returns:
    --------
    numpy.ndarray
        A 3D array of shape (# of particles, # of dust species, # of grain size bins) representing the mass of grains in each bin for each dust species for each gas particle.

    """
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


def get_grain_bin_slope(particle: Particle):
    """
    Calculates the grain bin slopes for the given gas particles.

    Parameters:
    -----------
    particle : Particle
        An instance of the Particle class containing gas properties.

    Returns:
    --------
    numpy.ndarray
        A 3D array of shape (# of particles, # of dust species, # of grain size bins) containing the calculated grain bin slopes for each particle and dust species.

    """
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


def get_dust_accretion_rate(particles:Particle, 
                            T_cutoff:float=300,
                            scaling_factor:float=1.0, 
                            bin_subsamples:int=1,
                            factor_clumping:bool=True):
    """
    Determines the mass rate of dust growth from gas-dust accretion for the given gas particles.
    Parameters:
    - particles (Particle): The particle object containing properties of the gas particles.
    - T_cutoff (float): The temperature cutoff for accretion in Kelvin.
    - scaling_factor (float): A scaling factor for the accretion rate.
    - bin_subsamples(int): The number of subsampled points for each bin in the grain size distribution.
    Set > 1 for small number of bins
    - factor_clumping (bool): Whether to include the clumping factor in the calculation.

    Returns:
    - list: A list of accretion rates for each particle in the particles object.
    """

    # Get the physical properties of each particle needed to calculate rates
    npart = particles.npart
    nH = particles.get_property('nH')
    rho = particles.get_property('density')
    temp = particles.get_property('temperature')
    M = particles.get_property('mach_number')
    b = 0.5 # turbulence mode ratio assumed to be constant in sims
    sigma = np.sqrt(np.log(1+b*b*M*M))
    metallicity = particles.get_property('Z_all')
    dust_metallicity = particles.get_property('dust_Z')
    dust_bin_numbers = particles.get_property('grain_bin_num')
    dust_bin_slopes = particles.get_property('grain_bin_slope')

    # Global bin properties
    bin_num = particles.sp.Flag_GrainSizeBins
    # All grain sizes need to be in units of cm
    a_edges = particles.sp.Grain_Bin_Edges * config.um_to_cm 
    a_centers = particles.sp.Grain_Bin_Centers * config.um_to_cm 

    # The rate change in grain size for each bin for each gas particle
    dadt = np.zeros([npart,bin_num])
    # The rate change in mass for each bin for each gas particle
    dMbin_dt = np.zeros([npart,bin_num])

    # Need to step though each 
    species = ['silicates' , 'carbonaceous', 'iron']
    for i,spec in enumerate(species):
        spec_bin_number = dust_bin_numbers[:,i]
        spec_bin_slope = dust_bin_slopes[:,i]

        # Physical properties of dust species needed for calculations
        spec_props = dust_species_properties(spec)
        key_element_index = spec_props['key_element_index']
        dust_atomic_weight = spec_props['dust_atomic_weight']
        key_mass = spec_props['key_mass']
        key_num_atoms = spec_props['key_num_atoms'] 
        rho_c = spec_props['rho_c']
        nH_max = spec_props['nH_max']

        # number abundance of key element factoring in depletion into dust
        key_abundance = metallicity[:,key_element_index]
        key_depl_frac = dust_metallicity[:,key_element_index]/metallicity[:,key_element_index]
        key_num_dens = rho * key_abundance * (1 - key_depl_frac) / (key_mass*config.PROTONMASS)

        # Determine clumping factor due to subresolved gas-dust clumping using assumed Mach number
        if factor_clumping:
            temp_clump_factor = 1/(np.exp(sigma*sigma)/2 * (1 + erf((3/2*sigma*sigma + np.log(nH_max/nH)) / (np.sqrt(2)*sigma))))
            eff_clump_factor = np.exp(sigma*sigma)/2 * erfc((3/2*sigma*sigma-np.log(nH_max/nH)) / (np.sqrt(2)*sigma))
        else:
            temp_clump_factor = np.ones(npart)
            eff_clump_factor = np.ones(npart)

        # Dense gas fraction used for calculating the effective Coulomb enhancement factor
        # In dense molecular gas all gas-phase metals are neutral so no Coulomb enhancement
        nH_dense = 1E3
        fdense = 1/2+1/2*erf((sigma*sigma/2 - np.log(nH_dense/nH))/(np.sqrt(2)*sigma));

        # Simple prescription for Coulomb enhancement in each grain size bin
        Coulomb_enhancement = np.ones(len(a_centers))
        if species == 'silicates':
            Coulomb_enhancement[a_centers*config.cm_to_um<0.01] = 10
            Coulomb_enhancement[a_centers*config.cm_to_um>0.01] = 0.5
        elif species == 'carbonaceous':
            Coulomb_enhancement[a_centers*config.cm_to_um<0.01] = 3
            Coulomb_enhancement[a_centers*config.cm_to_um>0.01] = 0
        elif species == 'iron':
            Coulomb_enhancement[a_centers*config.cm_to_um<0.01] = 20
            Coulomb_enhancement[a_centers*config.cm_to_um>0.01] = 1

        Coulomb_enhancement = (1-fdense[:,np.newaxis])*Coulomb_enhancement[np.newaxis,:] + fdense[:,np.newaxis]

        # Accretion occurs below a critical temperature
        temp_mask = temp*temp_clump_factor <= T_cutoff
        dadt_ref = 1.91249E-4 # reference change in grain size in cm/Gyr assuming purely hard-sphere type encounters
        # Change in grain size for each bin in cm/Gyr
        dadt[temp_mask] = scaling_factor * dadt_ref * (dust_atomic_weight / (key_num_atoms * np.sqrt(key_mass))) * key_num_dens[temp_mask,np.newaxis] * np.sqrt(temp[temp_mask,np.newaxis] * temp_clump_factor[temp_mask,np.newaxis]) / rho_c * Coulomb_enhancement[temp_mask] * eff_clump_factor[temp_mask,np.newaxis];

        # Change in mass 
        # For simplicity assume all grains in a bin have the same size as the bin center
        for j in range(bin_num):
            # Assume all grains in a bin have the same size as the bin center
            if bin_subsamples==1: dMbin_dt[:,j] += dadt[:,j] * 4 * np.pi * rho_c * np.power(a_centers[j],2) * spec_bin_number[:,j] # g/Gyr
            # Subsample each bin into M linearly spaced points. Using bin number and slope, find number of grains in subsample
            # and assume they have a single size equal to the subsample center
            else:
                # For subsampling, we need to sum over all the subsampled points in the bin
                a_edges_in_bin = np.linspace(a_edges[j], a_edges[j+1], bin_subsamples+1)
                size_diff_in_bin = a_edges_in_bin[1:] - a_edges_in_bin[:-1]
                a_centers_in_bin = (a_edges_in_bin[1:] + a_edges_in_bin[:-1])/2
                for k in range(bin_subsamples):
                    number_in_subsample = (spec_bin_number[:,j]/(a_edges[j+1]-a_edges[j]) + spec_bin_slope[:,j]*a_centers[j]) * (size_diff_in_bin[k]) + spec_bin_slope[:,j]*(np.square(a_edges_in_bin[k+1])-np.square(a_edges_in_bin[k]))/2
                    dMbin_dt[:,j] += dadt[:,j] * 4 * np.pi * rho_c * np.power(a_centers_in_bin[k],2) * number_in_subsample # g/Gyr

    # Convert to more useful units Msol/yr
    dMbin_dt *= config.grams_to_Msolar / 1E9
    dM_total = np.sum(dMbin_dt,axis=1) # total change in mass for each gas particle
    return dM_total



def get_dust_sputtering_rate(particles:Particle, 
                             T_cutoff:float=1E4,
                             scaling_factor:float=1.0, 
                             bin_subsamples:int=1,
                             factor_clumping:bool=True):
    """
    Determines the mass rate of dust destruction from thermal sputtering for the given gas particles.
    Parameters:
    - particles (Particle): The particle object containing properties of the gas particles.
    - T_cutoff (float): The temperature cutoff for accretion in Kelvin.
    - scaling_factor (float): A scaling factor for the accretion rate.
    - bin_subsamples (int): The number of subsampled points for each bin in the grain size distribution.
    Set > 1 for small number of bins
    - factor_clumping (bool): Whether to include the clumping factor in the calculation.

    Returns:
    - list: A list of accretion rates for each particle in the particles object.
    """

    # Get the physical properties of each particle needed to calculate rates
    npart = particles.npart
    nH = particles.get_property('nH')
    rho = particles.get_property('density')
    temp = particles.get_property('temperature')
    M = particles.get_property('mach_number')
    b = 0.5 # turbulence mode ratio assumed to be constant in sims
    dust_bin_numbers = particles.get_property('grain_bin_num')
    dust_bin_slopes = particles.get_property('grain_bin_slope')

    # Global bin properties
    bin_num = particles.sp.Flag_GrainSizeBins
    # All grain sizes need to be in units of cm
    a_edges = particles.sp.Grain_Bin_Edges * config.um_to_cm 
    a_centers = particles.sp.Grain_Bin_Centers * config.um_to_cm 

    # The rate change in grain size for each bin for each gas particle
    dadt = np.zeros([npart,bin_num])
    # The rate change in mass for each bin for each gas particle
    dMbin_dt = np.zeros([npart,bin_num])

    # Need to step though each 
    species = ['silicates' , 'carbonaceous', 'iron']
    for i,spec in enumerate(species):
        spec_bin_number = dust_bin_numbers[:,i]
        spec_bin_slope = dust_bin_slopes[:,i]

        # Physical properties of dust species needed for calculations
        spec_props = dust_species_properties(spec)
        rho_c = spec_props['rho_c']

        # Determine clumping factor due to subresolved gas-dust clumping using assumed Mach number
        if factor_clumping:
            temp_clump_factor = 1/(1+b*b*M*M)
            eff_clump_factor = (1+b*b*M*M)
        else:
            temp_clump_factor = np.ones(npart)
            eff_clump_factor = np.ones(npart)

        # Sputtering starts to become efficient above 10^5 K
        temp_mask = temp*temp_clump_factor > T_cutoff
        logt = np.log10(temp*temp_clump_factor)
        # Determine sputtering erosion rate (um yr^-1 cm^3)
        if spec == 'silicates':
            Y_sput = np.power(10,-226.95 + 127.94*logt - 29.920*np.power(logt,2) + 3.5354*np.power(logt,3) - 0.21055*np.power(logt,4) + 0.0050362*np.power(logt,5));
        elif spec == 'carbonaceous':
            Y_sput = np.power(10,-226.85 + 133.44*logt - 32.572*np.power(logt,2) + 4.0057*np.power(logt,3) - 0.24747*np.power(logt,4) + 0.0061212*np.power(logt,5));
        elif spec == 'iron':
            Y_sput = np.power(10,-156.88 +  82.110*logt - 18.238*np.power(logt,2) + 2.0692*np.power(logt,3) - 0.11933*np.power(logt,4) + 0.0027788*np.power(logt,5));

        dadt[temp_mask] = (- scaling_factor * eff_clump_factor * nH * Y_sput * config.um_to_cm / 1E-9)[temp_mask,np.newaxis] # change to cm/Gyr


        # Change in mass 
        # For simplicity assume all grains in a bin have the same size as the bin center
        for j in range(bin_num):
            # Assume all grains in a bin have the same size as the bin center
            if bin_subsamples==1: dMbin_dt[:,j] += dadt[:,j] * 4 * np.pi * rho_c * np.power(a_centers[j],2) * spec_bin_number[:,j] # g/Gyr
            # Subsample each bin into M linearly spaced points. Using bin number and slope, find number of grains in subsample
            # and assume they have a single size equal to the subsample center
            else:
                # For subsampling, we need to sum over all the subsampled points in the bin
                a_edges_in_bin = np.linspace(a_edges[j], a_edges[j+1], bin_subsamples+1)
                size_diff_in_bin = a_edges_in_bin[1:] - a_edges_in_bin[:-1]
                a_centers_in_bin = (a_edges_in_bin[1:] + a_edges_in_bin[:-1])/2
                for k in range(bin_subsamples):
                    number_in_subsample = (spec_bin_number[:,j]/(a_edges[j+1]-a_edges[j]) + spec_bin_slope[:,j]*a_centers[j]) * (size_diff_in_bin[k]) + spec_bin_slope[:,j]*(np.square(a_edges_in_bin[k+1])-np.square(a_edges_in_bin[k]))/2
                    dMbin_dt[:,j] += dadt[:,j] * 4 * np.pi * rho_c * np.power(a_centers_in_bin[k],2) * number_in_subsample # g/Gyr

    # Convert to more useful units Msol/yr
    dMbin_dt *= config.grams_to_Msolar / 1E9
    dM_total = np.sum(dMbin_dt,axis=1) # total change in mass for each gas particle
    return dM_total


def get_dust_shattering_rate(particles:Particle, 
                             scaling_factor:float=1.0, 
                             bin_subsamples:int=1,
                             factor_clumping:bool=True):
    """
    Determines the rate of the given grain process for gas particles based on their properties.
    Parameters:
    - particles (Particle): The particle object containing properties of the gas particles.
    - T_cutoff (float): The temperature cutoff for accretion in Kelvin.
    - scaling_factor (float): A scaling factor for the accretion rate.
    - bin_subsamples (int): The number of subsampled points for each bin in the grain size distribution.
    Set > 1 for small number of bins
    - factor_clumping (bool): Whether to include the clumping factor in the calculation.

    Returns:
    - list: A list of accretion rates for each particle in the particles object.
    """


def get_grain_size_distribution(gas: Particle,
                                species: str ='silicates', 
                                mask: list|None = None, 
                                points_per_bin: int = 1, 
                                std_percentiles: list = [16, 84],
                                weight: str = 'gas'):
    """
    Calculates the normalized grain size probability distribution (dn/da and dm/dloga) of a dust species from gas particles. 
    Gives the mean and standard deviation of the distribution for all particles. 
    Note this is determined by calculating the normalized distributions for all particles and then calculating the percentiles
    with the dust species masses as weights.

    Parameters
    ----------
    gas: Particle
        Gas particles to determine grain size distribution for.
    species: str
        Name of species you want size distribution for. (silicates, carbonaceous, or iron)
    mask : list
        Boolean array to mask particles. Set to None for all particles.
    points_per_bin : int, optional
        Number of data points you want in each grain size bin. If 1, will use the center of each bin.
        Note this uses the bin slopes, so this won't be pretty to look at.
    std_percentiles : list, optional
        Percentiles of the standard deviation you want.
    weight: str
        Property to weight the grain size distribution by. Default is 'gas', which uses the gas mass.
        Can also use 'dust' to weight by the dust species mass.

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
    if mask is None: mask = np.ones(gas.npart,dtype=bool)
    num_part = len(gas.get_property('M_gas')[mask])
    bin_edges = gas.sp.Grain_Bin_Edges
    bin_centers = gas.sp.Grain_Bin_Centers
    num_bins = gas.sp.Flag_GrainSizeBins
    if species == 'silicates': spec_ind = 0
    elif species == 'carbonaceous': spec_ind = 1
    elif species == 'iron': spec_ind = 2
    else: assert 0, "Dust species %s not supported"%species; return;

    bin_nums = gas.get_property('grain_bin_num')[mask,spec_ind]
    bin_slopes = gas.get_property('grain_bin_slope')[mask,spec_ind]
    total_species_mass = gas.get_property('M_gas')[mask]*gas.get_property('dust_spec')[mask,spec_ind]
    gas_masses = gas.get_property('M_gas')[mask]

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
        dmdloga_vals[~no_dust,i*points_per_bin:(i+1)*points_per_bin] = (4/3*np.pi*rho_c*np.power(x_points,4)*(bin_num[~no_dust]/(bin_edges[i+1]-bin_edges[i])+bin_slope[~no_dust]*(x_points-bin_centers[i])))/total_M[~no_dust]/config.Msolar_to_g

    # Determine percentile distribution values from all of the the particles
    # Weight each particle by their the total dust species mass
    if weight == 'dust':
        weights = total_species_mass
    else:
        weights = gas_masses
    percentile_dnda = np.zeros([len(percentiles),points_per_bin*num_bins])
    percentile_dmdloga = np.zeros([len(percentiles),points_per_bin*num_bins])
    # Get percentiles for each point in each bin
    for i in range(len(grain_size_vals)):
        percentile_dnda[:,i] = weighted_percentile(dnda_vals[:,i], percentiles=percentiles, weights=weights, ignore_invalid=True)
        percentile_dmdloga[:,i] = weighted_percentile(dmdloga_vals[:,i], percentiles=percentiles, weights=weights, ignore_invalid=True)

    return grain_size_vals, percentile_dnda, percentile_dmdloga



def get_dust_shattering_and_coagulation_rate(particles:Particle, 
                             scaling_factor:float=1.0,
                             factor_clumping:bool=True,
                             small_large_cutoff:float=0.01
                             ):
    """
    Determines the mass change rate of small dust grains due to shattering and coagulation for the given gas particles.
    The grain size cutoff for large vs small grains is set by the small_large_cutoff parameter.

    Parameters:
    - particles (Particle): The particle object containing properties of the gas particles.
    - scaling_factor (float): A scaling factor for the accretion rate.
    - factor_clumping (bool): Whether to include the clumping factor in the calculation.
    - small_large_cutoff (float): The cutoff grain size (microns) for small and large grains 
    used to determine mass change between the two.

    Returns:
    - list: A list of accretion rates for each particle in the particles object.
    """

    # Get the physical properties of each particle needed to calculate rates
    npart = particles.npart
    nH = particles.get_property('nH')
    rho = particles.get_property('density')
    temp = particles.get_property('temperature')
    M = particles.get_property('mach_number')
    b = 0.5 # turbulence mode ratio assumed to be constant in sims
    sigma = np.sqrt(np.log(1+b*b*M*M))
    dust_bin_numbers = particles.get_property('grain_bin_num')
    dust_bin_slopes = particles.get_property('grain_bin_slope')
    # Use this to get cell volume (not calculated directly due to possible float overflow)
    mass_grams = particles.get_property('mass')*config.Msolar_to_g


    # Global bin properties
    bin_num = particles.sp.Flag_GrainSizeBins
    # All grain sizes need to be in units of cm
    a_edges = particles.sp.Grain_Bin_Edges * config.um_to_cm 
    a_centers = particles.sp.Grain_Bin_Centers * config.um_to_cm 

    # The rate change in mass for each bin for each gas particle
    shat_dMbin_dt = np.zeros([npart,bin_num])
    coag_dMbin_dt = np.zeros([npart,bin_num])


    # Need to step though each 
    species = ['silicates' , 'carbonaceous', 'iron']
    for s,spec in enumerate(species):
        spec_bin_number = dust_bin_numbers[:,s]
        spec_bin_slope = dust_bin_slopes[:,s]

        # Physical properties of dust species needed for calculations
        spec_props = dust_species_properties(spec)
        rho_c = spec_props['rho_c']
        nH_max = spec_props['nH_max']

        P1 = spec_props['P1']
        v_shat = spec_props['v_shat']
        poisson = spec_props['poisson']
        youngs = spec_props['youngs']
        gamma = spec_props['gamma']

        # Determine clumping factor due to subresolved gas-dust clumping using assumed Mach number
        if factor_clumping:
            temp_clump_factor = 1/(np.exp(sigma*sigma)/2 * (1 + erf((3/2*sigma*sigma + np.log(nH_max/nH)) / (np.sqrt(2)*sigma))))
            eff_clump_factor = np.exp(sigma*sigma)/2 * erfc((3/2*sigma*sigma-np.log(nH_max/nH)) / (np.sqrt(2)*sigma))
        else:
            temp_clump_factor = np.ones(npart)
            eff_clump_factor = np.ones(npart)


        for i in range(bin_num):
            ai_upper = a_edges[i+1]
            ai_lower = a_edges[i]
            ai_center = (ai_upper + ai_lower)/2
            mi_acenter = 4*np.pi/3*rho_c*np.power(ai_center,3)

            Ni = spec_bin_number[:,i]
            si = spec_bin_slope[:,i]

            coag_removal_term = np.zeros(npart)
            coag_injection_term = np.zeros(npart)
            shat_removal_term = np.zeros(npart)
            shat_injection_term = np.zeros(npart)
            print('i:',i)
            for j in range(bin_num):
                print('j:',j)
                aj_upper = a_edges[j+1]
                aj_lower = a_edges[j]
                aj_center = (aj_upper + aj_lower)/2

                Nj = spec_bin_number[:,j]
                sj = spec_bin_slope[:,j]

                #int_I_ij = ((2*np.power(ai_lower,2) + 2*ai_lower*ai_upper + 2*np.power(ai_upper,2) + 3*ai_lower*(aj_lower + aj_upper) + 3*ai_upper*(aj_lower + aj_upper) + 2*(np.power(aj_lower,2) + aj_lower*aj_upper + np.power(aj_upper,2)))*spec_bin_number[:,i]*spec_bin_number[:,j])/6.
                int_I_ij = shattering_coagulation_polynomial(Ni, Nj, si, sj, ai_lower, ai_upper, ai_center, aj_lower, aj_upper, aj_center)
                
                vijrel = grain_relative_velocity(ai_center, aj_center, rho_c, gas_particles = particles, fixed_impact_angle=True)
                v_coag = v_coagulation(ai_center, aj_center, rho_c, poisson, youngs, gamma)
                # Sometimes v_coag can go above v_shat (mainly for metallic iron)
                if v_coag > v_shat: v_coag = v_shat

                shat_mask = (vijrel > v_shat) | (vijrel <= v_coag)
                shat_removal_term[shat_mask] += scaling_factor * vijrel[shat_mask] * mi_acenter * int_I_ij[shat_mask]
                coag_mask = (vijrel <= v_coag)
                coag_removal_term[coag_mask] += scaling_factor * vijrel[coag_mask] * mi_acenter * int_I_ij[coag_mask]

                for k in range(bin_num):
                    print('k:',k)
                    ak_upper = a_edges[k+1]
                    ak_lower = a_edges[k]
                    ak_center = (ak_upper + ak_lower)/2
                    Nk = spec_bin_number[:,k]
                    sk = spec_bin_slope[:,k]

                    #int_I_kj = ((2*np.power(aj_lower,2) + 2*aj_lower*aj_upper + 2*np.power(aj_upper,2) + 3*aj_lower*(ak_lower + ak_upper) + 3*aj_upper*(ak_lower + ak_upper) + 2*(np.power(ak_lower,2) + ak_lower*ak_upper + np.power(ak_upper,2)))*spec_bin_number[j]*spec_bin_number[k])/6.
                    int_I_kj = shattering_coagulation_polynomial(Nj, Nk, sj, sk, aj_lower, aj_upper, aj_center, ak_lower, ak_upper, ak_center)
                    vkjrel = grain_relative_velocity(ak_center, aj_center, rho_c, gas_particles = particles, fixed_impact_angle=True)
                    v_coag = v_coagulation(ak_center, aj_center, rho_c, poisson, youngs, gamma)
                    # Sometimes v_coag can go above v_shat (mainly for metallic iron)
                    if v_coag > v_shat: v_coag = v_shat

                    shatter_mask = vkjrel > v_shat
                    mshat_kj = m_shatter(ai_lower, ai_upper, ak_center, aj_center, vkjrel, P1, rho_c, v_shat)
                    shat_injection_term[shatter_mask] += scaling_factor * vkjrel[shatter_mask] * mshat_kj[shatter_mask] * int_I_kj[shatter_mask]

                    coag_mask = vkjrel <= v_coag
                    mcoag_kj = m_coagulation(ai_lower, ai_upper, ak_center, aj_center, vkjrel, v_coag, rho_c)
                    coag_injection_term[coag_mask] += scaling_factor * vkjrel[coag_mask] * mcoag_kj[coag_mask] * int_I_kj[coag_mask]

            # Volume of gas cell in cm^3
            shat_dMbin_dt[:,i] = eff_clump_factor / (mass_grams/rho) * np.pi * (-shat_removal_term + shat_injection_term) # g/sec
            coag_dMbin_dt[:,i] = eff_clump_factor / (mass_grams/rho) * np.pi * (-coag_removal_term + coag_injection_term) # g/sec

 
    # Convert to more useful units Msol/yr
    shat_dMbin_dt *= config.grams_to_Msolar / config.sec_to_yr
    coag_dMbin_dt *= config.grams_to_Msolar / config.sec_to_yr
    # Determine change in mass for snall grain bins
    small_grain_bins = a_centers<small_large_cutoff
    shat_dM_total = np.sum(shat_dMbin_dt[:,small_grain_bins],axis=1) # total change in mass for small grains for each gas particle
    coag_dM_total = np.sum(coag_dMbin_dt[:,small_grain_bins],axis=1) # total change in mass for small grains for each gas particle
    return shat_dM_total, coag_dM_total



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

    

def calculate_extinction_curve(gas: Particle, 
                               species: str = 'silicates', 
                               mask: list|None = None, 
                               std_percentiles: list = [16, 84],
                               bin_subsamples: int = 1):
    """
    Calculates the median and percentile extinction curve normalized by the extinction 
    in the visible band (A_lambda / A_V) from the gas cell grain size distributions. 
    Can specify only contributions from a given species (silicates, carbonaceous, or iron) 
    or the total extinction from all species. The median and percentiles are calculated as 
    such. A normalized extinction curve is calculated for each gas cell given its 
    grain size distribution for each dust species. The median and percentiles are 
    then calculated from all of the gas cells weighted by their total dust mass.

    Parameters
    ----------
    gas : Particle
       Particle data to determine extinction curve from.
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
    if mask is None: mask = np.ones(gas.npart,dtype=bool)
    num_part = len(gas.get_property('M_gas')[mask])
    bin_nums = gas.get_property('grain_bin_num')[mask]
    bin_slopes = gas.get_property('grain_bin_slope')[mask]
    gas_masses = gas.get_property('M_gas')[mask]
    dust_masses = gas.get_property('M_dust')[mask]
    bin_centers = gas.sp.Grain_Bin_Centers
    bin_edges = gas.sp.Grain_Bin_Edges
    num_bins = gas.sp.Flag_GrainSizeBins

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
                grain_size_wave_vals = np.zeros([N_wave_bins,2])
                for l,grain_size_center in enumerate(grain_centers_in_bin):
                    grain_size_wave_vals[:,0] = grain_size_center
                    grain_size_wave_vals[:,1] = unique_wavelengths

                    N_in_bin = dnda_in_bin[:,l]*size_diff_in_bin[l]  # Assume dnda is constant value with a = a at bin center for each bin
                    A_lambda_spec += (grain_size_center*grain_size_center) * Qext([grain_size_wave_vals])[0][np.newaxis,:] * N_in_bin[:,np.newaxis]
                    
            # Calculate A_V for all species since we normalize by total A_V and want to know the relative contributions of each species
            for l,grain_size_center in enumerate(grain_centers_in_bin):
                N_in_bin = dnda_in_bin[:,l]*size_diff_in_bin[l]  # Assume dnda is constant value with a = a at bin center for each bin
                A_V_spec += (grain_size_center*grain_size_center) * Qext([grain_size_center,lambda_V])[0] * N_in_bin


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





def calculate_idealized_extinction_curve(amin:float = 1E-3, 
                                         amax:float = 1,
                                         sil_to_carbon_ratio:float = 2.0,
                                         MRN_slope:float = -3.5,):
    """
    Calculates the extinction curve normalized by the extinction in the
    visible band (A_lambda / A_V) for an idealized MRN grain population. 

    Parameters
    ----------
    amin : float
        Minimum grain size in micron.
    amax : float
        Maximum grain size in micron.
    sil_to_carbon_ratio : float
        Ratio of silicate to carbonaceous dust mass. Default is 2.0.
    MRN_slope : float
        Slope of the MRN grain size distribution. Default is -3.5.

    Returns
    -------
    wavelength_points: list
        Wavelength data points in micron.
    A_lambda_points : list
        A_lambda/A_V values at corresponding wavelength points.
    """	    

    N_wave_bins = 500 # Number of wavelength bins for interpolation
    N_size_bins = 100
    MRN_slope = -3.5
    sil_to_carbon_ratio = 2.0 # Assuming a 2:1 silicate to carbonaceous dust mass ratio

    lambda_V = 0.5470 # V band wavelength in microns
    # Need wavelengths for A_lambda values. Assuming all dust species tables have the same wavelengths
    # Make sure this is the same order as appears in the first subtable
    # WARNING: If using numpy.unique the wavelengths order in the table is not preserved
    optical_property = get_dust_optical_properties('silicates')
    unique_wavelengths = optical_property['w(micron)'].values[optical_property['radius(micron)']==np.min(optical_property['radius(micron)'])] # use the smallest grain radius table to get the corresponding Qext wavelengths
    # Extend the wavelength grid for interpolation of Qext
    unique_wavelengths = np.logspace(np.log10(np.min(unique_wavelengths)), np.log10(np.max(unique_wavelengths)), N_wave_bins)

    # Assuming MRN size distribution with 2:1 silicate to carbonaceous dust mass ratio
    # We use a 1st order numerical grain size distribution


    dust_species = ['silicates', 'carbonaceous']
    spec_indices = [0,1]
    num_species = len(dust_species)
    optical_properties = [get_dust_optical_properties('silicates'),
                          get_dust_optical_properties('carbonaceous')]
    


    # Determine grain size and dn/da values for each dust species
    bin_size = np.power(10,np.log10(amax/amin)/N_size_bins)
    bin_edges = np.zeros(N_size_bins+1)
    bin_centers = np.zeros(N_size_bins)
    for i in range(N_size_bins+1):
        bin_edges[i] = pow(bin_size,i)*amin
    for i in range(N_size_bins):
        bin_centers[i] = (bin_edges[i+1] + bin_edges[i])/2.
    dNda_vals = np.power(bin_centers, MRN_slope) # dN/da ~ a^(-3.5) grain surface density (since we are normalizing A_lambda by A_V dont need to normalize this)

        

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
    A_lambda_total = np.zeros(N_wave_bins)
    A_V_total = 0 # Extinction in V band (5470 Angstrom)
    for i,spec_ind in enumerate(spec_indices):
        # Calculate the dust species A_lambda and A_V 
        A_lambda_spec = np.zeros(N_wave_bins)
        A_V_spec =  0 # Extinction in V band (5470 Angstrom) for one species
        Qext = spec_Qext[i]
        if spec_ind == 0: # silicates
            rho_c_sil = dust_species_properties('silicates')['rho_c'] # Bulk density of silicates [g/cm^3]
            rho_c_carb = dust_species_properties('carbonaceous')['rho_c'] # Bulk density of carbonaceous dust [g/cm^3]
            rho_c_sil = dust_species_properties('silicates')['rho_c'] # Bulk density of silicates [g/cm^3]
            spec_dNda = dNda_vals * sil_to_carbon_ratio / (rho_c_sil/rho_c_carb) # 
        else: # carbonaceous
            spec_dNda = dNda_vals

        # We are approximating the integral Qext(a,lambda) * dn/da(a) da from a_min to a_max 
        # as a sum over grain bins Qext(a_i,center,lambda) * dn/da(a_i,center) * (a_i,upper - a_i,lower)
        for j in range(N_size_bins):
            bin_upper = bin_edges[j+1]
            bin_lower = bin_edges[j]
            size_diff_in_bin = bin_upper-bin_lower
            bin_center = bin_centers[j]
    
            dNda = spec_dNda[j] # Assume dN/da is constant value with a = a at bin center for each bin

            grain_size_wave_vals = np.zeros([N_wave_bins,2])
            grain_size_wave_vals[:,0] = bin_center
            grain_size_wave_vals[:,1] = unique_wavelengths

            A_lambda_spec += (bin_center*bin_center) * Qext([grain_size_wave_vals])[0] * size_diff_in_bin * dNda
                    
            # Calculate A_V for all species since we normalize by total A_V and want to know the relative contributions of each species
            A_V_spec += (bin_center*bin_center) * Qext([bin_center,lambda_V])[0] * size_diff_in_bin * dNda
                


        A_lambda_total += A_lambda_spec
        A_V_total += A_V_spec

    # Normalize by A_V for each particle
    A_lambda_norm = A_lambda_total/A_V_total



    return unique_wavelengths, A_lambda_norm



def calculate_idealized_Av(NH:float = 1E21,
                           DTG:float = 0.014*0.5,
                           amin:float = 1E-3,
                           amax:float = 1.0,
                           sil_to_carbon_ratio:float = 2.0,
                           MRN_slope:float = -3.5):
    """
    Calculates the visible band extinction Av (5470 angstrom) for an idealized sight line 
    assuming an MRN grain size distribution. Can specify the sight line surface density and
    dust-to-gas ratio.

    Parameters
    ----------
    NH : float, optional
        Sight line surface density of hydrogen in cm^-2. Default is 1E21.
    DTG : float, optional
        Dust-to-gas mass ratio. Default is 0.014*0.5, assuming solar metallicity with 50% of metals in dust.
    amin : float, optional
        Minimum grain size in microns. Default is 1E-3.
    amax : float, optional
        Maximum grain size in microns. Default is 1.0.
    sil_to_carbon_ratio : float, optional
        Ratio of silicate to carbonaceous dust. Default is 2.0.
    MRN_slope : float, optional
        Slope of the MRN grain size distribution. Default is -3.5.

    Returns
    -------
    Av : float
        Total extinction in the V band (5470 Angstrom).
    """	    

    lambda_V = 0.5470 # V band wavelength in microns

    N_size_bins=100
    # Load snapshot gas particle data and grain size bin data
    amin*=config.um_to_cm
    amax*=config.um_to_cm
    bin_size = np.power(10,np.log10(amax/amin)/N_size_bins)
    bin_edges = np.zeros(N_size_bins+1)
    bin_centers = np.zeros(N_size_bins)
    for i in range(N_size_bins+1):
        bin_edges[i] = pow(bin_size,i)*amin
    for i in range(N_size_bins):
        bin_centers[i] = (bin_edges[i+1] + bin_edges[i])/2.

    # Determine mass surface densities for silicates and carbonaceous dust
    # Then determine the number surface density of grains in each size bin for each assuming an MRN size distribution
    sigma_dust = DTG * 1.4 * NH * config.PROTONMASS # Dust mass surface density in g/cm^2, assuming hydrogen is ~70% of total gas mass
    sil_surface_density = sigma_dust * sil_to_carbon_ratio / (1 + sil_to_carbon_ratio) # Silicate dust surface density in g/cm^2
    carb_surface_density = sigma_dust / (1 + sil_to_carbon_ratio) # Carbonaceous dust surface density in g/cm^2
    spec_props = dust_species_properties('silicates')
    sil_rho_c = spec_props['rho_c'] # Bulk density of silicates [g/cm^3]
    spec_props = dust_species_properties('carbonaceous')
    carb_rho_c = spec_props['rho_c'] # Bulk density of carbonaceous [g/cm^3]    

    sil_N_bin = np.zeros(N_size_bins)
    carb_N_bin = np.zeros(N_size_bins)    

    # Determine normalization constant for grain size distribution given total mass of dust species
    sil_C_norm = (sil_surface_density) * (12 + 3 * MRN_slope) / (
        4 * np.pi * sil_rho_c * (np.power(amax, 4 + MRN_slope) - np.power(amin, 4 + MRN_slope)))
    carb_C_norm = (carb_surface_density) * (12 + 3 * MRN_slope) / (
        4 * np.pi * carb_rho_c * (np.power(amax, 4 + MRN_slope) - np.power(amin, 4 + MRN_slope)))

    for k in range(N_size_bins):
        alower = bin_edges[k]
        aupper = bin_edges[k + 1]

        # Calculate number in bin
        sil_N_bin[k] = sil_C_norm / (MRN_slope + 1) * (np.power(aupper, MRN_slope + 1) - np.power(alower, MRN_slope + 1))
        carb_N_bin[k] = carb_C_norm / (MRN_slope + 1) * (np.power(aupper, MRN_slope + 1) - np.power(alower, MRN_slope + 1))

    N_in_bins = np.array([sil_N_bin, carb_N_bin])



    dust_species = ['silicates', 'carbonaceous']
    spec_indices = [0,1]
    num_species = len(dust_species)
    optical_properties = [get_dust_optical_properties('silicates'),
                          get_dust_optical_properties('carbonaceous')]
        

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


    A_V_total = 0 # Total extinction in V band (5470 Angstrom)
    for i,spec_ind in enumerate(spec_indices):
        A_V_spec = 0
        Qext = spec_Qext[i]
        spec_N_in_bin = N_in_bins[i]

        # We are approximating the integral Qext(a,lambda) * dn/da(a) da from a_min to a_max 
        # as a sum over grain bins Qext(a_i,center,lambda) * dn/da(a_i,center) * (a_i,upper - a_i,lower)
        for j in range(N_size_bins):
            bin_center = bin_centers[j]
            SigmaNdust_in_bin = spec_N_in_bin[j] 
            A_V_spec += (2.5*np.log10(np.exp(1))*np.pi*bin_center*bin_center) * Qext([bin_centers[j]*config.cm_to_um,lambda_V])[0] * SigmaNdust_in_bin 

        A_V_total += A_V_spec

    return A_V_total



def calculate_Av(gas: Particle, 
                 species: str = 'silicates', 
                 mask: list|None = None, 
                 bin_subsamples: int = 1):
    """
    Calculates the visible band extinction Av (5470 angstrom) for gas cells in the snapshot 
    given their grain size distributions. Can specify only contributions from a 
    given species (silicates, carbonaceous, or iron) or  the total extinction from all 
    species.

    Parameters
    ----------
    gas : Particle
        Gas particle data to calculate Av.
    species: str
        Species you want to extinction curve for. (silicates, carbonaceous, or all). 
        Note this is still normalized by the total Av from all species.
    mask : ndarray
        Boolean array to mask particles. Set to None for all particles.
    bin_subsamples : int
        Number of dn/da subsamples from each grain size bin to be used for calculating extinction. 
        Default of 1 means the grain size distribution at only the centers of each bin are used 
        to calculate the extinction curves. If set to N>1, the grain size distribution at N points 
        is used. This is useful when you have a small number of grain size bins 
        (i.e. your bins cover a large range in grain sizes).


    Returns
    -------
    Av : list
        Av values for each gas cell in snapshot.
    """	    

    lambda_V = 0.5470 # V band wavelength in microns

    if mask is None: mask = np.ones(gas.npart,dtype=bool)
    num_part = len(gas.get_property('M_gas')[mask])
    bin_nums = gas.get_property('grain_bin_num')[mask]
    bin_slopes = gas.get_property('grain_bin_slope')[mask] / config.um_to_cm**2
    hsml = np.asarray(gas.get_property('size')[mask],dtype=np.float64) * config.kpc_to_cm
    bin_centers = gas.sp.Grain_Bin_Centers * config.um_to_cm
    bin_edges = gas.sp.Grain_Bin_Edges * config.um_to_cm
    num_bins = gas.sp.Flag_GrainSizeBins         


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
        table_grain_radii = optical_property['radius(micron)'].values * config.um_to_cm # Convert to cm for RGI interpolation
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


    A_V_total = np.zeros(num_part) # Total extinction in V band (5470 Angstrom)
    for i,spec_ind in enumerate(spec_indices):
        # Calculate the dust species A_lambda and A_V 
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
                    
            # Calculate A_V only for species we are not excluding
            if spec_ind not in exclude_spec_ind:
                for l,grain_size_center in enumerate(grain_centers_in_bin):
                    N_in_bin = dnda_in_bin[:,l]*size_diff_in_bin[l]  # Assume dnda is constant value with a = a at bin center for each bin
                    # Assume all dust grains in the cell are in a cylindrical slab with a radius equal to the smoothing length
                    SigmaNdust_in_bin = N_in_bin / (np.pi * hsml * hsml) # Grain surface number density [um^-2]
                    A_V_spec += (2.5*np.log10(np.exp(1))*np.pi*grain_size_center*grain_size_center) * Qext([grain_size_center,lambda_V])[0] * SigmaNdust_in_bin 

        A_V_total += A_V_spec

    return A_V_total


def shattering_coagulation_polynomial(Ni, Nj, si, sj, ail, aiu, aic, ajl, aju, ajc):
    """
    Calculate the interaction rate between grains in bins i and j.
    
    Parameters:
    - Ni: Number of grains in bin i.`
    - Nj: Number of grains in bin j.
    - si: Slope of bin i.
    - sj: Slope of bin j.
    - ail: Lower edge of bin i.
    - aiu: Upper edge of bin i.
    - aic: Center of bin i.
    - ajl: Lower edge of bin j.
    - aju: Upper edge of bin j.
    - ajc: Center of bin j.

    Returns:
    - Iij: Interaction rate between grains in bin i and bin j.
    """

    # Interaction rate between grains of bin_i and bin_j. An ugly polynomial but it's analytically solvable 
    Iij = (12*(2*aiu*aiu + 3*aiu*(ajl + aju) + 2*(ajl*ajl + ajl*aju + aju*aju))*Ni*Nj + 
     6*aiu*(-2*aic*(2*aiu*aiu + 3*aiu*(ajl + aju) + 2*(ajl*ajl + ajl*aju + aju*aju)) + 
        aiu*(3*aiu*aiu + 4*aiu*(ajl + aju) + 2*(ajl*ajl + ajl*aju + aju*aju)))*Nj*si + 
     6*(-3*ajl*ajl*ajl*ajl + 2*aiu*aiu*(2*ajc - ajl - aju)*(ajl - aju) + 3*aju*aju*aju*aju + 4*ajc*(ajl*ajl*ajl - aju*aju*aju) + 
        aiu*(-4*ajl*ajl*ajl + 4*aju*aju*aju + 6*ajc*(ajl - aju)*(ajl + aju)))*Ni*sj + 
     aiu*(-6*aic*(ajl*(4*aiu*aiu*ajc - 2*aiu*(aiu - 3*ajc)*ajl + 4*(-aiu + ajc)*ajl*ajl - 3*ajl*ajl*ajl) - 
           4*aiu*aiu*ajc*aju + 2*aiu*(aiu - 3*ajc)*aju*aju + 4*(aiu - ajc)*aju*aju*aju + 3*aju*aju*aju*aju) + 
        aiu*(-9*ajl*ajl*ajl*ajl + 9*aiu*aiu*(2*ajc - ajl - aju)*(ajl - aju) + 9*aju*aju*aju*aju + 
           12*ajc*(ajl*ajl*ajl - aju*aju*aju) + 8*aiu*(-2*ajl*ajl*ajl + 2*aju*aju*aju + 3*ajc*(ajl - aju)*(ajl + aju))))*si*sj
      - 9*ail*ail*ail*ail*si*(2*Nj + (2*ajc - ajl - aju)*(ajl - aju)*sj) + 
     4*ail*ail*ail*si*(6*(aic - ajl - aju)*Nj + (ajl - aju)*
         (6*aic*ajc - 3*aic*(ajl + aju) - 6*ajc*(ajl + aju) + 4*(ajl*ajl + ajl*aju + aju*aju))*sj) + 
     6*ail*(ajl*(6*Ni*Nj + 4*aic*aju*Nj*si) - 3*aic*ajl*ajl*ajl*ajl*si*sj - 4*ajl*ajl*ajl*(Ni - aic*ajc*si)*sj + 
        2*aiu*Ni*(2*Nj + (2*ajc - ajl - aju)*(ajl - aju)*sj) + ajl*ajl*(4*aic*Nj*si + 6*ajc*Ni*sj) + 
        aju*(6*Ni*Nj + 4*aic*aju*Nj*si + aju*(-6*ajc*Ni + 4*aju*Ni - 4*aic*ajc*aju*si + 3*aic*aju*aju*si)*sj)) + 
     3*ail*ail*(8*Ni*Nj + 4*(2*ajc - ajl - aju)*(ajl - aju)*Ni*sj + 
        si*(-4*ajl*ajl*Nj - 4*ajl*aju*Nj - 4*ajc*ajl*ajl*ajl*sj + 3*ajl*ajl*ajl*ajl*sj + 
           aju*aju*(-4*Nj + (4*ajc - 3*aju)*aju*sj) + 
           4*aic*(3*ajl*Nj + 3*ajc*ajl*ajl*sj - 2*ajl*ajl*ajl*sj + aju*(3*Nj + aju*(-3*ajc + 2*aju)*sj)))))/72.;
    
    if len(Iij) > 1: Iij[Iij<0] = 0.0; # Set negative values to zero
    return Iij



def get_mass_of_dust_that_can_shatter_or_coagulate(particles:Particle):
    """
    Determines the dust mass that can shatter or coagulate for the given gas particles. This determines whether
    there are which grain size bins in a gas particle can shatter or coagulate and adds up their dust mass.

    Parameters:
    - particles (Particle): The particle object containing properties of the gas particles.
    Returns:
    - shat_Mdust (list): Mass of dust that can shatter in each gas particle.
    - coag_Mdust (list): Mass of dust that can coagulate in each gas particle.
    """

    # Get the physical properties of each particle needed to calculate rates
    npart = particles.npart
    nH = particles.get_property('nH')
    rho = particles.get_property('density')
    temp = particles.get_property('temperature')
    dust_bin_numbers = particles.get_property('grain_bin_num')
    dust_bin_slopes = particles.get_property('grain_bin_slope')
    dust_bin_masses = particles.get_property('grain_bin_mass')*config.grams_to_Msolar


    # Global bin properties
    bin_num = particles.sp.Flag_GrainSizeBins
    # All grain sizes need to be in units of cm
    a_edges = particles.sp.Grain_Bin_Edges * config.um_to_cm 
    a_centers = particles.sp.Grain_Bin_Centers * config.um_to_cm 

    shat_Mdust = np.zeros([npart])
    coag_Mdust = np.zeros([npart])

    # Need to step though each 
    species = ['silicates' , 'carbonaceous', 'iron']
    for s,spec in enumerate(species):
        spec_bin_number = dust_bin_numbers[:,s]
        spec_bin_slope = dust_bin_slopes[:,s]
        spec_bin_mass = dust_bin_masses[:,s]

        # Physical properties of dust species needed for calculations
        spec_props = dust_species_properties(spec)
        rho_c = spec_props['rho_c']
        nH_max = spec_props['nH_max']

        P1 = spec_props['P1']
        v_shat = spec_props['v_shat']
        poisson = spec_props['poisson']
        youngs = spec_props['youngs']
        gamma = spec_props['gamma']
        for i in range(bin_num):
            ai_upper = a_edges[i+1]
            ai_lower = a_edges[i]
            ai_center = (ai_upper + ai_lower)/2

            Ni = spec_bin_number[:,i]
            si = spec_bin_slope[:,i]
            Mi = spec_bin_mass[:,i]

            shat_mask = np.zeros(npart, dtype=bool)
            coag_mask = np.zeros(npart, dtype=bool)

            for j in range(bin_num):
                aj_upper = a_edges[j+1]
                aj_lower = a_edges[j]
                aj_center = (aj_upper + aj_lower)/2

                vijrel = grain_relative_velocity(ai_center, aj_center, rho_c, gas_particles = particles, fixed_impact_angle=True)
                v_coag = v_coagulation(ai_center, aj_center, rho_c, poisson, youngs, gamma)
                # Sometimes v_coag can go above v_shat (mainly for metallic iron)
                if v_coag > v_shat: v_coag = v_shat

                shat_mask = (shat_mask) | (vijrel > v_shat)
                coag_mask = (coag_mask) | (vijrel <= v_coag)
            
            shat_Mdust[shat_mask] += Mi[shat_mask]
            coag_Mdust[coag_mask] += Mi[coag_mask]


    return shat_Mdust, coag_Mdust
