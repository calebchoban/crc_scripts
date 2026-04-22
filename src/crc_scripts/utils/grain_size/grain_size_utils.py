
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
from scipy.interpolate import CubicSpline
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

def SNe_dnda(a, rho_c=1, a_norm=0.1*config.um_to_cm, sigma_a=0.2, a_cut = 0.1*config.um_to_cm, gamma=3.5, a_min = 0.5E-3*config.um_to_cm):
    # Need to determine normalization constants so function is continuous
    total_mass = 1;

    C1_norm = total_mass*3*(gamma-4)*a_cut*np.power(a_min,gamma)*np.exp(-(np.power(np.log(a_cut/a_norm),2)/(2*sigma_a*sigma_a))) / \
    (2*np.pi*rho_c*(2*np.power(a_cut,gamma)*np.power(a_min,4) - 2*np.power(a_cut,4)*np.power(a_min,gamma) + \
    np.power(a_norm,3)*a_cut*np.power(a_min,gamma)*np.sqrt(2*np.pi)*(gamma-4)*sigma_a*np.exp((9*np.power(sigma_a,4) + \
    np.power(np.log(a_cut/a_norm),2))/(2*sigma_a*sigma_a))*erfc((np.log(a_cut/a_norm)-3*sigma_a*sigma_a)/(np.sqrt(2)*sigma_a))));

    C2_norm = C1_norm*np.power(a_cut,gamma-1)*np.exp(-(np.power(np.log(a_cut/a_norm),2)/(2*sigma_a*sigma_a)));

    dnda = np.piecewise(a, [a >= a_cut, a < a_cut], [lambda x: C1_norm/x * np.exp(-np.power(np.log(x/a_norm),2) / (2*sigma_a*sigma_a)), lambda x: C2_norm*np.power(x,-gamma)])
    return dnda

def SNe_dmdloga(a, rho_c=1, a_norm=0.1*config.um_to_cm, sigma_a=0.2, a_cut = 0.1*config.um_to_cm, gamma=3.5, a_min = 0.5E-3*config.um_to_cm):
    return 4/3*np.pi * rho_c * np.power(a,4)*SNe_dnda(a,rho_c, a_norm, sigma_a, a_cut, gamma, a_min)

def AGB_dnda(a, a_norm=0.1*config.um_to_cm, sigma_a=0.47):
    dnda = 1/a**5 * np.exp(-np.power(np.log(a/a_norm),2) / (2*sigma_a*sigma_a))
    return dnda

def AGB_dmdloga(a, rho_c=1, a_norm=0.1*config.um_to_cm, sigma_a=0.47):
    return 4/3*np.pi * rho_c * np.power(a,4)*AGB_dnda(a,a_norm, sigma_a)


# Astrodust size distribution from Hensley & Draine 2023
# Returns only astrodust or PAHs component
def astrodust_dnda(a,PAHs=False):
    # PAH parameters
    B1=7.52E-7; B2=8.09E-10;
    a01 = 4*config.angstrom_to_um; a02 = 30*config.angstrom_to_um; sigma=0.4;
    # Astrodust parameters
    BAd = 3.31E-10; a0_Ad = 63.8*config.angstrom_to_um;
    sigma_Ad = 0.353; A0 = 2.97E-5;
    A1=-3.4;A2=-0.807;A3=0.157;A4=7.96E-3;A5=-1.68E-3;

    if PAHs:
        dnda = B1/a * np.exp(-(np.log(a/a01)**2)/(2*sigma**2)) + B2/a * np.exp(-(np.log(a/a02)**2)/(2*sigma**2))
    else:
        dnda = BAd / a * np.exp(-(np.log(a/a0_Ad)**2)/(2*sigma_Ad**2)) + A0/a * np.exp( A1*(np.power(np.log(a*config.um_to_angstrom),1)) + A2*(np.power(np.log(a*config.um_to_angstrom),2)) + A3*(np.power(np.log(a*config.um_to_angstrom),3)) + A4*(np.power(np.log(a*config.um_to_angstrom),4)) + A5*(np.power(np.log(a*config.um_to_angstrom),5)))

    return dnda

def astrodust_dmdloga(a,PAHs=False):
    if PAHs: rho_c=2 # g/cm^3
    else: rho_c=2.74
    rho_c /= config.cm_to_um**3 # g/cm^3 to g/um^3 since grain radii are in um
    return 4/3*np.pi * rho_c * np.power(a,4)*astrodust_dnda(a,PAHs=PAHs)


def DL07_dnda(a, species='silicates'):
    # PAH values from Drain+Li 2007
    a01 = 4*config.angstrom_to_um; a02 = 20*config.angstrom_to_um;
    sigma1 = 0.4; sigma2 = 0.55;
    amin=3.5*config.angstrom_to_um;
    rho_c = 2.24 / config.cm_to_um**3; # g/um^3
    m_c = 1.9944733E-23; # g
    bc=6E-5 # total C abundance per H nucleus
    b1 = 0.75*bc; b2 = 0.25*bc
    # Values for Rv=3.1 and bC=6E5 from Weingartner+Draine 2001
    # large graphite
    a_cs=0.1 # micron
    alpha_g = -1.54; beta_g = -0.165;
    a_tg = 0.0107; a_cg = 0.428; Cg=9.99E-12;
    alpha_s = -2.21; beta_s = 0.300; a_ts = 0.164; Cs=1.00E-13
    if species == 'carbonaceous':
        aM1 = a01*np.exp(3*sigma1**2)
        x1 = np.log(aM1/amin)/(np.sqrt(2)*sigma1)
        n01 = 3/np.power(2*np.pi,3/2) * (np.exp(4.5*sigma1**2)/1+erf(x1)) * (m_c / (rho_c * aM1**3*sigma1)) * b1
        aM2 = a02*np.exp(3*sigma2**2)
        x2 = np.log(aM2/amin)/(np.sqrt(2)*sigma2)
        n02 = 3/np.power(2*np.pi,3/2) * (np.exp(4.5*sigma2**2)/1+erf(x2)) * (m_c / (rho_c * aM2**3*sigma2)) * b2

        dnda_PAH = n01/a * np.exp(-np.power(np.log(a/a01),2)/(2*sigma1**2)) + n02/a * np.exp(-np.power(np.log(a/a02),2)/(2*sigma2**2))

        if beta_g >=0: F_term = 1+beta_g*a/a_tg
        else: F_term = 1/(1-beta_g*a/a_tg)
        # exp_term = np.ones(len(a))
        # exp_term[a>a_tg] = np.exp(-np.power((a-a_tg)/a_cg,3))
        exp_term = np.piecewise(a, [a <= a_tg, a > a_tg], [lambda x: 1, lambda x: np.exp(-np.power((x-a_tg)/a_cg,3))])
        dnda = dnda_PAH + Cg/a * np.power(a/a_tg,alpha_g) * F_term * exp_term
    else:
        if beta_s >=0: F_term = 1+beta_s*a/a_ts
        else: F_term = 1/(1-beta_s*a/a_ts)
        # exp_term = np.ones(len(a))
        # exp_term[a>a_ts] = np.exp(-np.power((a-a_ts)/a_cs,3))
        exp_term = np.piecewise(a, [a <= a_ts, a > a_ts], [lambda x: 1, lambda x: np.exp(-np.power((x-a_ts)/a_cs,3))])
        dnda = Cs/a * np.power(a/a_ts,alpha_s) * F_term * exp_term
    
    return dnda

def DL07_dmdloga(a,species='silicates'):
    if species=='carbonaceous': rho_c=2.24 # g/cm^3
    else: rho_c=2.74
    rho_c /= config.cm_to_um**3 # g/cm^3 to g/um^3 since grain radii are in um
    return 4/3*np.pi * rho_c * np.power(a,4)*DL07_dnda(a,species=species)



def get_grain_bin_info_assuming_MRN(particle: Particle,
                                    assume_depletion: bool = False):
    """
    Calculates the grain bin numbers and slopes assuming an MRN grain size distribution for the given gas particles.
    Parameters:
    - particle (Particle): The particle object containing properties of the gas particles.
    - assume_depletion (bool): Whether to assume a fixed depletion pattern for the dust species based on total metallicity. If False, uses tracked dust species abundances used.
    Returns:
    - tuple: A tuple containing two numpy arrays:
        - grain_bin_numbers: A 3D array of shape (# of particles, # of dust species, # of grain size bins) representing the number of grains in each bin for each dust species for each gas particle.
        - grain_bin_slopes: A 3D array of shape (# of particles, # of dust species, # of grain size bins) representing the slope of the grain size distribution in each bin for each dust species for each gas particle.
    """
        
    dust_species = particle.sp.dust_species
    spec_indicies = particle.sp.dust_species_indices
    num_part = particle.npart
    num_bins = particle.sp.Flag_GrainSizeBins
    a_max = particle.sp.Grain_Size_Max
    a_min = particle.sp.Grain_Size_Min
    a_edges = particle.sp.Grain_Bin_Edges
    a_centers = particle.sp.Grain_Bin_Centers
    gas_mass = particle.get_property('M_gas')
    metallicity = particle.get_property('Z_all')
    dust_spec = particle.get_property('dust_spec').astype(np.float64) # Force double precision here for grain bin mass calculations
    dust_bin_numbers = np.zeros(np.shape(particle.get_property('grain_bin_num')))
    dust_bin_slopes = np.zeros(np.shape(particle.get_property('grain_bin_slope')))

    # Instead of using tracked dust species abundances we are calculating them assuming a set depletion pattern
    if assume_depletion:
        # Assuming Si and Fe have the same depletion, Mg and O depletion set by Si depletion + assumed silicate stoichiometry, carb determined by silicate to carbonaceous dust mass ratio
        sil_iron_depl = 0.5; sil_to_carb_ratio = 2;
        for i, spec in enumerate(dust_species):
            spec_indx = spec_indicies[i]
            # Silicate dust
            if spec=='silicates':
                sil_elem_key = particle.sp.Silicates_Element_Key
                sil_elem_num = particle.sp.Silicates_Element_Number
                # Get ratio of total silicate mass per mass of Si in silicates
                silicate_atomic_weight = np.sum(sil_elem_num * config.ATOMIC_MASS[sil_elem_key])
                Si_atomic_weight = config.ATOMIC_MASS[7]
                Si_dust_metallicity = sil_iron_depl*metallicity[:,7]
                print(silicate_atomic_weight/Si_atomic_weight)
                dust_spec[:,spec_indx] = Si_dust_metallicity * silicate_atomic_weight/Si_atomic_weight
            elif spec=='carbonaceous':
                dust_spec[:,spec_indx] = dust_spec[:,spec_indicies[dust_species == 'silicates']]/sil_to_carb_ratio
            elif spec=='iron':
                dust_spec[:,spec_indx] = sil_iron_depl*metallicity[:,10]
            else:
                dust_spec[:,spec_indx] = 0
    
    # Assume MRN powerlaw size distribution
    powerlaw = -3.5; 
    for i, spec in enumerate(dust_species):
        spec_indx = spec_indicies[i]
        spec_props = config.dust_species_properties(spec)
        bulk_dens = spec_props['rho_c']/(config.cm_to_um**3) # g/cm^3 to g/um^3 since grain radii are in um
        # Determine normalization constant for grain size distribution given total mass of dust species
        C_norm = (dust_spec[:,spec_indx]*gas_mass*config.Msolar_to_g)*(12+3*powerlaw) / (4 * np.pi * bulk_dens * (np.power(a_max,4+powerlaw)-np.power(a_min,4+powerlaw)));
        for j in range(num_bins):
            alower = a_edges[j]; aupper = a_edges[j+1];
            a_center = a_centers[j]
            mass_in_bin = 4*np.pi*bulk_dens/(3*(4+powerlaw))*C_norm*(np.power(aupper,4+powerlaw)-np.power(alower,4+powerlaw));
            number_in_bin = C_norm/(powerlaw+1)*(np.power(aupper,powerlaw+1) - np.power(alower,powerlaw+1));
            # Calculate slope in bin that has dust
            no_dust = (mass_in_bin <= 0) | (number_in_bin <= 0)
            spec_bin_slopes = (3*mass_in_bin/(4*np.pi*bulk_dens)-number_in_bin/(4*(aupper-alower))*(np.power(aupper,4)-np.power(alower,4))) / ((np.power(aupper,5)-np.power(alower,5))/5-a_center/4*(np.power(aupper,4)-np.power(alower,4)));
            spec_bin_slopes[no_dust] = 0

            dust_bin_numbers[:,i,j] = number_in_bin
            dust_bin_slopes[:,i,j] = spec_bin_slopes

    return dust_bin_numbers, dust_bin_slopes


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

    supported_species = ['silicates', 'carbonaceous', 'iron']

    species = particle.sp.dust_species
    spec_indices = particle.sp.dust_species_indices

    # Calculate grain bin mass from numbers and slopes
    grain_bin_slopes = np.zeros((particle.npart, snap.Flag_DustSpecies, snap.Flag_GrainSizeBins),dtype='double')
    for i,spec in enumerate(species):
        if spec not in supported_species:
             raise ValueError(f"Dust species {spec} not supported. Supported species are {supported_species}. Species in snap are {species}.")  
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
                            factor_clumping:bool=True,
                            Coulomb_factor:bool=True,
                            assume_MRN_dust:bool=False):
    """
    Determines the mass rate of dust growth from gas-dust accretion for the given gas particles.
    Parameters:
    - particles (Particle): The particle object containing properties of the gas particles.
    - T_cutoff (float): The temperature cutoff for accretion in Kelvin.
    - scaling_factor (float): A scaling factor for the accretion rate.
    - bin_subsamples(int): The number of subsampled points for each bin in the grain size distribution.
    Set > 1 for small number of bins
    - factor_clumping (bool): Whether to include the clumping factor in the calculation.
    - Coulomb_factor (bool): Whether to include the Coulomb enhancement factor in the calculation.
    - assume_MRN_dust (bool): Whether to assume an MRN grain size distribution for the dust instead of what is tracked in snapshot.

    Returns:
    - list: A list of accretion rates for each particle in the particles object.
    """

    # Get the physical properties of each particle needed to calculate rates
    npart = particles.npart
    nH = particles.get_property('nH')
    rho = particles.get_property('density')
    temp = particles.get_property('temperature')
    M = particles.get_property('mach_number')
    HII_delaytime = particles.get_property('HII_delaytime')
    b = 0.5 # turbulence mode ratio assumed to be constant in sims
    sigma = np.sqrt(np.log(1+b*b*M*M))
    metallicity = particles.get_property('Z_all')

    # Global bin properties
    bin_num = particles.sp.Flag_GrainSizeBins
    # All grain sizes need to be in units of cm
    a_edges = particles.sp.Grain_Bin_Edges * config.um_to_cm 
    a_centers = particles.sp.Grain_Bin_Centers * config.um_to_cm 

    if assume_MRN_dust:
        dust_metallicity = particles.get_property('dust_Z')
        dust_bin_numbers, dust_bin_slopes = get_grain_bin_info_assuming_MRN(particles)
    else:
        dust_metallicity = particles.get_property('dust_Z')
        dust_bin_numbers = particles.get_property('grain_bin_num')
        dust_bin_slopes = particles.get_property('grain_bin_slope')

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
        temp_clump_factor = np.ones(npart)
        eff_clump_factor = np.ones(npart)
        fdense = np.zeros(npart)
        if factor_clumping:
            # Only clumping factor when there is a nonzero mach number
            mask = sigma != 0
            nonzero_sigma = sigma[mask]
            temp_clump_factor[mask] = 1/(np.exp(nonzero_sigma*nonzero_sigma)/2 * (1 + erf((3/2*nonzero_sigma*nonzero_sigma + np.log(nH_max/nH[mask])) / (np.sqrt(2)*nonzero_sigma))))
            eff_clump_factor[mask] = np.exp(nonzero_sigma*nonzero_sigma)/2 * erfc((3/2*nonzero_sigma*nonzero_sigma-np.log(nH_max/nH[mask])) / (np.sqrt(2)*nonzero_sigma))

            # Dense gas fraction used for calculating the effective Coulomb enhancement factor
            # In dense molecular gas all gas-phase metals are neutral so no Coulomb enhancement
            nH_dense = 1E3
            fdense[mask] = 1/2+1/2*erf((nonzero_sigma*nonzero_sigma/2 - np.log(nH_dense/nH[mask]))/(np.sqrt(2)*nonzero_sigma));

        # Simple power law prescription for Coulomb enhancement in each grain size bin
        if Coulomb_factor:
            Coulomb_enhancement = np.ones(len(a_centers))
            a_mid = 0.01*config.um_to_cm; a_min = 0.001*config.um_to_cm
            if spec == 'silicates': 
                D_small=10; D_large=0.5;
            elif spec == 'carbonaceous': 
                D_small=3; D_large=0;
            elif spec == 'iron': 
                D_small=20; D_large=1;
            else: 
                D_small=1; D_large=1;

            Coulomb_enhancement[a_centers<=a_min] = D_small
            Coulomb_enhancement[(a_min<=a_centers) & (a_centers<=a_mid)] = ((D_large-D_small)/np.log10(a_mid/a_min)) * np.log10(a_centers[(a_min<=a_centers) & (a_centers<=a_mid)]/a_min) + D_small
            Coulomb_enhancement[a_centers>a_mid] = D_large

            Coulomb_enhancement = (1-fdense[:,np.newaxis])*Coulomb_enhancement[np.newaxis,:] + fdense[:,np.newaxis]
        else:
            Coulomb_enhancement = np.ones([npart,bin_num])
        
        # Accretion occurs below a critical temperature
        temp_mask = (temp*temp_clump_factor <= T_cutoff)
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



def get_photodestruction_rate(particles:Particle, 
                             max_grain_size:float=5E-3,
                             scaling_factor:float=1.0, 
                             bin_subsamples:int=1):
    """
    Determines the mass rate of dust destruction from thermal sputtering for the given gas particles.
    Parameters:
    - particles (Particle): The particle object containing properties of the gas particles.
    - max_grain_size (float): Max grain size for photodestruction in microns.
    - scaling_factor (float): A scaling factor for the photodestruction rate.
    - bin_subsamples (int): The number of subsampled points for each bin in the grain size distribution.
    Set > 1 for small number of bins

    Returns:
    - list: A list of accretion rates for each particle in the particles object.
    """

    # Largest grain size photodestroyed (5 nm) and typical photodestruction timescale (5 Myr)
    # Note we only assume grains smaller than the max size are destroyed
    a_pd = max_grain_size* config.um_to_cm; tau_pd = 5E-3; 

    # Get the physical properties of each particle needed to calculate rates
    npart = particles.npart
    HII_delaytime = particles.get_property('HII_delaytime')

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

        # Sputtering starts to become efficient above 10^5 K
        HII_mask = HII_delaytime != 0
        # Determine photodestruction erosion rate (um yr^-1 cm^3)
        dadt[HII_mask] = (- scaling_factor * a_pd / tau_pd)  # change to cm/Gyr

        # Change in mass 
        # For simplicity assume all grains in a bin have the same size as the bin center
        for j in range(bin_num):
            # Assume all grains in a bin have the same size as the bin center
            if bin_subsamples==1: 
                if a_centers[j] < a_pd:
                    dMbin_dt[:,j] += dadt[:,j] * 4 * np.pi * rho_c * np.power(a_centers[j],2) * spec_bin_number[:,j] # g/Gyr
            # Subsample each bin into M linearly spaced points. Using bin number and slope, find number of grains in subsample
            # and assume they have a single size equal to the subsample center
            else:
                # For subsampling, we need to sum over all the subsampled points in the bin
                a_edges_in_bin = np.linspace(a_edges[j], a_edges[j+1], bin_subsamples+1)
                size_diff_in_bin = a_edges_in_bin[1:] - a_edges_in_bin[:-1]
                a_centers_in_bin = (a_edges_in_bin[1:] + a_edges_in_bin[:-1])/2
                for k in range(bin_subsamples):
                    if a_centers_in_bin[k] < a_pd:
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
                                dust_species: str = 'all',
                                mask: list|None = None, 
                                points_per_bin: int = 1, 
                                std_percentiles: list = [16, 84],
                                weight: str = 'gas'):
    """
    Calculates the median grain size probability distribution (dn/da and dm/dloga) for the specified dust 
    species across all gas particles, normalized by each particles total dust mass. 
    Gives the mean and standard deviation of the distribution for all particles. 
    Note this is determined by calculating the normalized distributions for all particles and then calculating the percentiles
    with the dust masses as weights.

    Parameters
    ----------
    gas: Particle
        Gas particles to determine grain size distribution for.
    dust_species: str
        Dust species you want the size distribution for. Options are ['all','silicates','carbonaceous','iron']
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

    total_bin_nums = gas.get_property('grain_bin_num')[mask]
    species_masses = gas.get_property('M_spec')[mask]
    dust_masses = gas.get_property('M_dust')[mask]
    gas_masses = gas.get_property('M_gas')[mask]

    grain_size_vals = np.zeros(points_per_bin*num_bins)
    dnda_vals = np.zeros([num_part,points_per_bin*num_bins], dtype=np.float64)
    dmdloga_vals = np.zeros([num_part,points_per_bin*num_bins], dtype=np.float64)

    snap_species = gas.sp.dust_species
    spec_indices = gas.sp.dust_species_indices

    supported_species = ['silicates','carbonaceous','iron']

    total_dust_N = np.zeros(num_part)
    total_dust_M = np.zeros(num_part)
    total_dust_M = dust_masses

    if dust_species in supported_species and dust_species in snap_species:
        desired_species = [dust_species]
    elif dust_species == 'all':
        desired_species = snap_species
    else:
        raise ValueError(f"Dust species {dust_species} not supported. Supported species are {supported_species} and 'all'. Species in snap are {snap_species}.")

    for j,spec in enumerate(desired_species):
        spec_ind = spec_indices[snap_species.index(spec)]
        bin_nums = gas.get_property('grain_bin_num')[mask,spec_ind]
        bin_slopes = gas.get_property('grain_bin_slope')[mask,spec_ind]

        total_dust_N += np.sum(total_bin_nums[:,spec_ind],axis=1)
        #total_dust_M += species_masses[:,spec_ind]
        # internal density for given dust species
        # Physical properties of dust species needed for calculations
        spec_props = dust_species_properties(spec)
        rho_c = spec_props['rho_c']/(config.cm_to_um**3) # g/cm^3 to g/um^3 since grain radii are in um
        
        # Need to normalize the distributions by total number and total mass, since we are only considering their shapes
        no_spec_dust = (species_masses[:,spec_ind] == 0)

        # Determine grain size, dn/da, and dm/dloga values for points in each bin
        for i in range(num_bins):
            bin_num = bin_nums[:,i,np.newaxis]; # Add extra dimension for numpy math below
            bin_slope = bin_slopes[:,i,np.newaxis]; 
            # If one point per bin, set it to the center of the bin
            if points_per_bin == 1: x_points = np.array([bin_centers[i]])
            else: x_points = np.logspace(np.log10(bin_edges[i]*1.02),np.log10(bin_edges[i+1]*0.98),points_per_bin) # shave off the very edges of each bin since they can be near zero
            grain_size_vals[i*points_per_bin:(i+1)*points_per_bin] = x_points

            dnda_vals[no_spec_dust,i*points_per_bin:(i+1)*points_per_bin] += 0
            dmdloga_vals[no_spec_dust,i*points_per_bin:(i+1)*points_per_bin] += 0
            dnda_vals[~no_spec_dust,i*points_per_bin:(i+1)*points_per_bin] += (bin_num[~no_spec_dust]/(bin_edges[i+1]-bin_edges[i])+bin_slope[~no_spec_dust]*(x_points-bin_centers[i]))
            dmdloga_vals[~no_spec_dust,i*points_per_bin:(i+1)*points_per_bin] += (4/3*np.pi*rho_c*np.power(x_points,4)*(bin_num[~no_spec_dust]/(bin_edges[i+1]-bin_edges[i])+bin_slope[~no_spec_dust]*(x_points-bin_centers[i])))
        
    
    # Normalize the distributions
    has_dust = total_dust_M>0
    dnda_vals[has_dust] /= total_dust_N[has_dust][:,np.newaxis]
    dmdloga_vals[has_dust] /= (total_dust_M[has_dust][:,np.newaxis]*config.Msolar_to_g)

    # Determine percentile distribution values from all of the the particles
    # Weight each particle by their the total dust species mass
    if weight == 'dust':
        weights = total_dust_M
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
                             factor_clumping:bool=True,):
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
            #temp_clump_factor = 1/(np.exp(sigma*sigma)/2 * (1 + erf((3/2*sigma*sigma + np.log(nH_max/nH)) / (np.sqrt(2)*sigma))))
            eff_clump_factor = np.exp(sigma*sigma)/2 * erfc((3/2*sigma*sigma-np.log(nH_max/nH)) / (np.sqrt(2)*sigma))
            eff_clump_factor[sigma==0] = 1
        else:
            #temp_clump_factor = np.ones(npart)
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
            for j in range(bin_num):
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

            # Note the Volume of gas cell in cm^3 is large so need to convert to higher floating point precision
            V_cell = (np.asarray(mass_grams,dtype=np.float64)/rho)
            shat_dMbin_dt[:,i] = eff_clump_factor / V_cell * np.pi * (-shat_removal_term + shat_injection_term) # g/sec
            coag_dMbin_dt[:,i] = eff_clump_factor / V_cell * np.pi * (-coag_removal_term + coag_injection_term) # g/sec

 
    # Convert to more useful units Msol/yr
    shat_dMbin_dt *= config.grams_to_Msolar / config.sec_to_yr
    coag_dMbin_dt *= config.grams_to_Msolar / config.sec_to_yr

    return shat_dMbin_dt, coag_dMbin_dt



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
                               dust_species: str = 'silicates', 
                               mask: list|None = None, 
                               weights: list|None= None,
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
    dust_species: str
        Species you want to extinction curve for. (silicates, carbonaceous, or all). 
        Note this is still normalized by the total A_V from all species.
    mask : ndarray
        Boolean array to mask particles. Set to None for all particles.
    weights : ndarray
        Array of weights used for determing percentiles. If None all particles will be weighted by their dust mass.
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
    bin_max = gas.sp.Grain_Size_Max
    bin_min = gas.sp.Grain_Size_Min



    supported_species = ['silicates', 'carbonaceous']
    snap_species = gas.sp.dust_species
    spec_indices = gas.sp.dust_species_indices

    if dust_species in snap_species:
        desired_species = [dust_species]
    elif dust_species == 'all':
        desired_species = snap_species
    else:
        raise ValueError(f"Dust species {dust_species} not supported. Supported species are {supported_species} and 'all'. Species in snap are {snap_species}.")

    num_species = len(snap_species)

    # Determine grain size and dn/da values for each dust species
    grain_size_vals = np.zeros(bin_subsamples*num_bins)
    dnda_vals = np.zeros([num_part,num_species,bin_subsamples*num_bins])

    # Determine dn/da values for points in each bin for each dust species
    for i, spec in enumerate(snap_species):
        spec_index = spec_indices[i]
        spec_bin_nums = bin_nums[:,spec_index]
        spec_bin_slopes = bin_slopes[:,spec_index]
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
    for i, spec in enumerate(snap_species):
        # Load in Q extinction data for the given species
        if spec not in supported_species:
            print(f"WARNING: Dust species {spec} has no stored optical properties. Will default to silicate properties.")
            optical_property = get_dust_optical_properties('silicates')
        else:
            optical_property = get_dust_optical_properties(spec)

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
        Qext = RGI((unique_table_radii,unique_table_wavelengths), Qext_matrix, method='cubic', bounds_error=True) 
        spec_Qext += [Qext]

    table_radii_min = np.min(unique_table_radii)
    table_radii_max = np.max(unique_table_radii)
    if bin_min<table_radii_min or bin_max>table_radii_max:
        print("WARNING: The simulated grain sizes are beyond the range of sizes supported by the dust grain optical properties.\n Will truncate grain sizes beyond supported range.")

    # Precompute grain centers, size diffs, and dnda column indices across all bins.
    # These are species-independent (depend only on bin structure and subsamples).
    _all_grain_centers = []
    _all_size_diffs = []
    _all_dnda_cols = []
    for j in range(num_bins):
        bin_upper = bin_edges[j+1]
        bin_lower = bin_edges[j]
        in_bin_mask = (grain_size_vals >= bin_lower) & (grain_size_vals < bin_upper)
        grain_sizes_in_bin = grain_size_vals[in_bin_mask]
        in_bin_indices = np.where(in_bin_mask)[0]
        if len(grain_sizes_in_bin) == 1:
            _all_grain_centers.append(grain_sizes_in_bin[0])
            _all_size_diffs.append(bin_upper - bin_lower)
            _all_dnda_cols.append(in_bin_indices[0])
        else:
            _all_grain_centers.extend((grain_sizes_in_bin[1:] + grain_sizes_in_bin[:-1]) / 2)
            _all_size_diffs.extend(grain_sizes_in_bin[1:] - grain_sizes_in_bin[:-1])
            _all_dnda_cols.extend(in_bin_indices[:-1])
    all_grain_centers = np.array(_all_grain_centers)
    all_size_diffs = np.array(_all_size_diffs)
    all_dnda_cols = np.array(_all_dnda_cols, dtype=int)
    total_centers = len(all_grain_centers)
    # Clamp grain centers to the supported table range once for all species
    clamped_centers = np.clip(all_grain_centers, table_radii_min, table_radii_max)

    # Calculate the extinction curve for each particle.
    # We approximate integral Qext(a,lambda) * dn/da(a) da as a sum over grain centers
    # Qext(a_i,center,lambda) * dn/da(a_i,center) * (a_i,upper - a_i,lower).
    # Instead of looping over each center and calling RGI separately, we batch all
    # (grain_center, wavelength) query points into a single RGI call per species,
    # then use a matmul to accumulate the weighted sum over grain centers.
    A_lambda_total = np.zeros([num_part, N_wave_bins])
    A_V_total = np.zeros(num_part) # Extinction in V band (5470 Angstrom)
    for i, spec in enumerate(snap_species):
        spec_index = spec_indices[i]
        Qext = spec_Qext[spec_index]
        spec_dnda = dnda_vals[:, spec_index, :]

        # N_in_bin_all[p, c] = dn/da(p, a_c) * da_c  shape: [num_part, total_centers]
        N_in_bin_all = spec_dnda[:, all_dnda_cols] * all_size_diffs[np.newaxis, :]

        # Calculate A_lambda only for desired species
        if spec in desired_species:
            # Single batched RGI call over all grain centers × wavelengths
            query_pts_lambda = np.column_stack([np.repeat(clamped_centers, N_wave_bins),
                                                np.tile(unique_wavelengths, total_centers)])
            Qext_all = Qext(query_pts_lambda).reshape(total_centers, N_wave_bins)  # [total_centers, N_wave_bins]
            # sum_c: a_c^2 * Qext(a_c, lambda) * N_in_bin[p, c]  →  matmul
            A_lambda_total += N_in_bin_all @ ((all_grain_centers ** 2)[:, np.newaxis] * Qext_all)

        # Always calculate A_V for all species since we normalize by total A_V
        query_pts_V = np.column_stack([clamped_centers, np.full(total_centers, lambda_V)])
        Qext_V_all = Qext(query_pts_V)  # [total_centers]
        A_V_total += N_in_bin_all @ (all_grain_centers ** 2 * Qext_V_all)

    # Normalize by A_V for each particle
    A_lambda_norm = A_lambda_total/A_V_total[:,np.newaxis] 

    # Calculate the percentiles for each wavelength point across all particles
    percentile_A_lambda = np.zeros([len(percentiles),N_wave_bins])
    for i in range(N_wave_bins):
        if weights is None:
            weights = dust_masses # Weight each particle extinction by their the total dust mass
        A_lambda_vals = A_lambda_norm[:,i]
        percentile_A_lambda[:,i] = weighted_percentile(A_lambda_vals, percentiles=percentiles, weights=weights, ignore_invalid=True)


    return unique_wavelengths, percentile_A_lambda





def calculate_idealized_extinction_curve(a_min:float = 1E-3, 
                                         amax:float = 1,
                                         sil_to_carbon_ratio:float = 2.0,
                                         MRN_slope:float = -3.5,):
    """
    Calculates the extinction curve normalized by the extinction in the
    visible band (A_lambda / A_V) for an idealized MRN grain population. 

    Parameters
    ----------
    a_min : float
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
    bin_size = np.power(10,np.log10(amax/a_min)/N_size_bins)
    bin_edges = np.zeros(N_size_bins+1)
    bin_centers = np.zeros(N_size_bins)
    for i in range(N_size_bins+1):
        bin_edges[i] = pow(bin_size,i)*a_min
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


def calculate_extinction_parameters(wavelengths:list,
                                    A_lambda:list):
    """
    Calculates slope and bump strength parameters for the provided 
    extinction curve following Salim & Narayanan (2020). 

    Parameters
    ----------
    wavelengths: list
        Wavelength data points in micron.
    A_lambda : list
        A_lambda values at corresponding wavelength points.

    Returns
    -------
    slope: float
        UV slope.
    bump: flat
        2175 A bump strength.
    """	  
    
    lambda_V = 0.5470 # V band wavelength in microns
    A_lambda_func = CubicSpline(wavelengths,A_lambda)
    # Slope is ratio of A at 1500 angstrom and AV
    slope = A_lambda_func(0.150)/A_lambda_func(lambda_V)

    A_2175_0 = 0.33*A_lambda_func(0.150)+0.67*A_lambda_func(0.3)
    A_bump = A_lambda_func(0.2175) - A_2175_0
    bump = A_bump / A_lambda_func(0.2175)

    return slope, bump



def calculate_idealized_Av(NH:float|list = 1E21,
                           DTG:float = 0.014*0.5,
                           a_min:float = 1E-3,
                           amax:float = 1.0,
                           sil_to_carbon_ratio:float = 2.0,
                           MRN_slope:float = -3.5):
    """
    Calculates the visible band extinction Av (5470 angstrom) for an idealized sight line 
    assuming an MRN grain size distribution. Can specify the sight line surface density and
    dust-to-gas ratio.

    Parameters
    ----------
    NH : float or list, optional
        Sight line surface density of hydrogen in cm^-2. Default is 1E21.
    DTG : float, optional
        Dust-to-gas mass ratio. Default is 0.014*0.5, assuming solar metallicity with 50% of metals in dust.
    a_min : float, optional
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

    # Since AV scales with NH can calculate AV for one NH and then determine the rest from scaling
    idealized_NH = 1E21

    lambda_V = 0.5470 # V band wavelength in microns

    N_size_bins=100
    # Load snapshot gas particle data and grain size bin data
    a_min*=config.um_to_cm
    amax*=config.um_to_cm
    bin_size = np.power(10,np.log10(amax/a_min)/N_size_bins)
    bin_edges = np.zeros(N_size_bins+1)
    bin_centers = np.zeros(N_size_bins)
    for i in range(N_size_bins+1):
        bin_edges[i] = pow(bin_size,i)*a_min
    for i in range(N_size_bins):
        bin_centers[i] = (bin_edges[i+1] + bin_edges[i])/2.

    # Determine mass surface densities for silicates and carbonaceous dust
    # Then determine the number surface density of grains in each size bin for each assuming an MRN size distribution
    sigma_dust = DTG * 1.4 * idealized_NH * config.PROTONMASS # Dust mass surface density in g/cm^2, assuming hydrogen is ~70% of total gas mass
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
        4 * np.pi * sil_rho_c * (np.power(amax, 4 + MRN_slope) - np.power(a_min, 4 + MRN_slope)))
    carb_C_norm = (carb_surface_density) * (12 + 3 * MRN_slope) / (
        4 * np.pi * carb_rho_c * (np.power(amax, 4 + MRN_slope) - np.power(a_min, 4 + MRN_slope)))

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

    A_V_total = A_V_total * (NH/idealized_NH)

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
    bin_max = gas.sp.Grain_Size_Max
    bin_min = gas.sp.Grain_Size_Min



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
        Qext = RGI((unique_table_radii,unique_table_wavelengths), Qext_matrix, method='cubic', bounds_error=True) 
        spec_Qext += [Qext]

    table_radii_min = np.min(unique_table_radii)
    table_radii_max = np.max(unique_table_radii)
    if bin_min<table_radii_min or bin_max>table_radii_max:
        print("WARNING: The simulated grain sizes are beyond the range of sizes supported by the dust grain optical properties.\n Will truncate grain sizes beyond supported range.")


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
                    # Assume grain sizes beyond supported range for optical properties have the same optical properties as the edges of the range
                    if grain_size_center < table_radii_min: Qext_vals = Qext([table_radii_min,lambda_V])[0]
                    elif grain_size_center > table_radii_max: Qext_vals = Qext([table_radii_max,lambda_V])[0]
                    else: Qext_vals = Qext([grain_size_center,lambda_V])[0]
                    A_V_spec += (2.5*np.log10(np.exp(1))*np.pi*grain_size_center*grain_size_center) * Qext_vals * SigmaNdust_in_bin 

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
