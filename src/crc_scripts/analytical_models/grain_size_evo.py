import numpy as np
from scipy.integrate import quad, dblquad, odeint
from scipy.interpolate import interp1d
from scipy.special import erfc,erf
from .. import config
from ..utils.math_utils import weighted_percentile


def MRN_dnda(a):
    return np.power(a,-3.5)

def MRN_dmda(a, rho_c=1):
    return 4/3*np.pi * rho_c * np.power(a,3)*np.power(a,-3.5)

def MRN_dmda_update(a,da):
    return 4/3*np.pi * 1 * np.power(a+da,3)*np.power(a,-3.5)

def lognorm_dnda(a, a_norm=0.1*config.um_to_cm, sigma_a=0.6):
    return 1/a * np.exp(-np.power(np.log(a/a_norm),2) / (2*sigma_a*sigma_a))

def ISM_phase_properties(ISM_phase):
    """
    Return the properties of the given interstellar medium (ISM) phase.

    Parameters:
    - ISM_phase (str): The ISM phase. Valid options are 'HIM', 'WIM', 'CNM', and 'MC'.

    Returns:
    - ISM_phase_props (dict): A dictionary containing the properties of the ISM phase. The dictionary has the following keys:
        - 'nH' (float): Number density of hydrogen atoms in the ISM phase.
        - 'rho' (float): Mass density of the ISM phase.
        - 'temp' (float): Temperature of the ISM phase.
        - 'M' (float): Mach number of the ISM phase.

    Raises:
    - AssertionError: If an invalid ISM phase is given.

    """

    assert ISM_phase in ['HIM','WIM','CNM','MC'], "Invalid ISM phase given"

    if ISM_phase == 'HIM':
        nH = 0.001
        rho = nH * config.PROTONMASS; 
        temp = 1E6
        M=1
    elif ISM_phase == 'WIM':
        nH = 0.1
        rho = nH * config.PROTONMASS; 
        temp = 1E4
        M=3.3
    elif ISM_phase == 'CNM':
        nH = 30
        rho = nH * config.PROTONMASS; 
        temp = 100
        M=3.3    
    elif ISM_phase == 'MC':
        nH = 1E4
        rho = nH * config.PROTONMASS; 
        temp = 10
        M=3.3
    else:
        nH = 1
        rho = 1 * config.PROTONMASS; 
        temp = 1000
        M=3.3


    ISM_phase_props = {'nH':nH, 'rho':rho, 'temp':temp, 'M':M}
    return ISM_phase_props


def dust_species_properties(species):
    """
    Returns the physical properties of a given dust species.

    Parameters:
    - species (str): The type of dust species. Supported values are 'silicates', 'carbonaceous', and 'iron'.

    Returns:
    - spec_props (dict): A dictionary containing the physical properties of the dust species. The dictionary includes the following keys:
        - dust_atomic_weight (float): The atomic weight of the dust species.
        - key_mass (float): The mass of the key atom in the dust species.
        - key_abundance (float): The abundance of the key atom in the dust species.
        - key_num_atoms (int): The number of key atoms in the dust species.
        - P1 (float): The critical shock pressure of the dust species.
        - v_shat (float): The shattering velocity threshold of the dust species.
        - poisson (float): The Poisson ratio of the dust species.
        - youngs (float): The Young's modulus of the dust species.
        - gamma (float): The surface energy of the dust species.
        - rho_c (float): The density of the dust material.

    Raises:
    - AssertionError: If the given species is not supported.

    """

    # Physical properties of dust species needed for calculations
    if species == 'silicates':
        dust_atomic_weight = config.SIL_ATOMIC_WEIGHT
        key_mass = config.ATOMIC_MASS[7] # Assume Si
        key_abundance = config.A09_ABUNDANCES[7]
        key_num_atoms = 1
        P1 = 3E11 # critical shock pressure (dyn cm^-2)
        v_shat = 2.7E5 # shattering velocity threshold (cm/s)
        poisson = 0.17; # poisson ratio 
        youngs = 5.4E11 # youngs modulus (dyn cm^-2) 
        gamma = 25 # surface energy (rgc cm^-2)
        rho_c = 3.13 # density of dust material (g cm^-3)
        nH_max = 1E4
    elif species == 'carbonaceous':
        dust_atomic_weight = config.ATOMIC_MASS[2]
        key_mass = dust_atomic_weight
        key_abundance = config.A09_ABUNDANCES[2]
        key_num_atoms = 1
        P1 = 4E10 # critical shock pressure
        v_shat = 1.2E5 # shattering velocity threshold
        poisson = 0.32; # poisson ratio
        youngs = 1E11 # youngs modulus (dyn cm^-2)
        gamma = 75 # surface energy (rgc cm^-2)
        rho_c = 2.25 # density of dust material (g cm^-3)
        nH_max = 1E3
    elif species == 'iron':
        dust_atomic_weight = config.ATOMIC_MASS[10]
        key_mass = dust_atomic_weight
        key_abundance = config.A09_ABUNDANCES[10]
        key_num_atoms = 1
        P1 = 5.5E10 # critical shock pressure
        v_shat = 2.2E5 # shattering velocity threshold
        poisson = 0.27; # poisson ratio
        youngs = 2.1E12 # youngs modulus (dyn cm^-2) 
        gamma = 3000 # surface energy (rgc cm^-2)
        rho_c = 7.86 # density of dust material (g cm^-3) 
        nH_max = 1E4
    else:
        assert 1, "Species type not supported"


    spec_props = {
    "dust_atomic_weight": dust_atomic_weight,
    "key_mass": key_mass,
    "key_abundance": key_abundance,
    "key_num_atoms": key_num_atoms,
    "P1": P1,
    "v_shat": v_shat,
    "poisson": poisson,
    "youngs": youngs,
    "gamma": gamma,
    "rho_c": rho_c,
    "nH_max": nH_max
    }

    return spec_props


def change_in_grain_distribution_from_acc_sput(dt_Gyr, amin=1E-3,amax=1E0,bin_num=1000, init_dnda = MRN_dnda, depl_frac=0.1, ISM_phase='CNM', species='silicates', subcycle_constraints='min_bin'):
    """
    Calculate the change in an MRN grain size distribution based on a constant da/dt. 
    Useful for testing predictions for gas-dust accretion and sputtering. 

    Parameters:
    - da (float): The constant change in grain size.
    - amin (float): The minimum grain size (microns).
    - amax (float): The maximum grain size (microns).
    - bin_num (int): The number of bins for grain size distribution.
    - depl_frac (float): The initial depletion fraction.

    Returns:
    - interp1d: The updated dnda distribution normalized to one.
    """

    

    # Get the physical properties of the ISM phase
    ISM_phase_props = ISM_phase_properties(ISM_phase)
    nH = ISM_phase_props['nH']
    rho = ISM_phase_props['rho']
    temp = ISM_phase_props['temp']
    M = ISM_phase_props['M']

    # Physical properties of dust species needed for calculations
    spec_props = dust_species_properties(species)
    dust_atomic_weight = spec_props['dust_atomic_weight']
    key_mass = spec_props['key_mass']
    key_abundance = spec_props['key_abundance']
    key_num_atoms = spec_props['key_num_atoms'] 
    rho_c = spec_props['rho_c']
    nH_max = spec_props['nH_max']

    # number abundance of key element factoring in depletion into dust
    key_num_dens = rho * key_abundance * (1 - depl_frac) / (key_mass*config.PROTONMASS)

    M_cell = 7100*config.Msolar_to_g;
    # dust to gas mass ratio for given species for given depletion used for normalization of initial size distribution
    DTG_spec = key_abundance*depl_frac*dust_atomic_weight / key_mass


    # Accretion occurs below 300 K
    if temp <= 300:
        dadt_ref = 1.91249E-4 # reference change in grain size in cm/Gyr assuming purely hard-sphere type encounters
        Coulomb_enhancement = 1

        # Determine clumping factor due to subresolved gas-dust clumping using assumed Mach number
        b = 0.5
        sigma = np.sqrt(np.log(1+b*b*M*M))
        temp_clump_factor = 1/(np.exp(sigma*sigma)/2 * (1 + erf((3/2*sigma*sigma + np.log(nH_max/nH)) / (np.sqrt(2)*sigma))))
        eff_clump_factor = np.exp(sigma*sigma)/2 * erfc((3/2*sigma*sigma-np.log(nH_max/nH)) / (np.sqrt(2)*sigma))
        print("Clumping factor", eff_clump_factor)
        dadt = dadt_ref * (dust_atomic_weight / (key_num_atoms * np.sqrt(key_mass))) * key_num_dens * np.sqrt(temp * temp_clump_factor) / rho_c * Coulomb_enhancement * eff_clump_factor; # change in cm/Gyr
    # Sputtering starts to become efficient above 10^5 K
    elif temp > 1E4:
        b = 0.5
        eff_clump_factor = (1+b*b*M*M)
        logt = np.log10(temp)
        # Determine sputtering erosion rate (um yr^-1 cm^3)
        if species == 'silicates':
            Y_sput = np.power(10,-226.95 + 127.94*logt - 29.920*np.power(logt,2) + 3.5354*np.power(logt,3) - 0.21055*np.power(logt,4) + 0.0050362*np.power(logt,5));
        elif species == 'carbonaceous':
            Y_sput = np.power(10,-226.85 + 133.44*logt - 32.572*np.power(logt,2) + 4.0057*np.power(logt,3) - 0.24747*np.power(logt,4) + 0.0061212*np.power(logt,5));
        elif species == 'iron':
            Y_sput = np.power(10,-156.88 +  82.110*logt - 18.238*np.power(logt,2) + 2.0692*np.power(logt,3) - 0.11933*np.power(logt,4) + 0.0027788*np.power(logt,5));

        dadt = -eff_clump_factor * nH * Y_sput * config.um_to_cm / 1E-9; # change to cm/Gyr
    else:
        dadt = 0

    print("Predicted dadt (um/Gyr):",dadt*config.cm_to_um)

    amin_cm = amin*config.um_to_cm; amax_cm= amax*config.um_to_cm
    a_vals = np.logspace(np.log10(amin_cm),np.log10(amax_cm),bin_num+1)
    init_dmda = lambda a: init_dnda(a) * 4*np.pi/3*rho_c * np.power(a,3)
    init_norm = DTG_spec * M_cell / quad(init_dmda,amin,amax)[0]

    init_M = np.zeros(len(a_vals)-1)
    init_N = np.zeros(len(a_vals)-1)
    final_M = np.zeros(len(a_vals)-1)
    final_N = np.zeros(len(a_vals)-1)

    for i in range(len(a_vals)-1):
        ai_upper = a_vals[i+1]
        ai_lower = a_vals[i]
        init_N[i] = quad(init_dnda,ai_lower,ai_upper)[0]*init_norm
        init_M[i] = quad(init_dmda,ai_lower,ai_upper)[0]*init_norm
        final_N[i]= init_N[i]
        final_M[i]= init_M[i]

    initial_total_N = np.sum(init_N)
    initial_total_M = np.sum(init_M)
    init_depl_frac=depl_frac

    total_cycles = 0
    n_cycle = 0


    # Determine if timestep subcycling is needed
    # Assume no subycylcing as default
    total_cycles = 1; 
    dt_subcycle = dt_Gyr;     
    if subcycle_constraints == 'min_bin':
        epsilon_cycle = 0.3
        a1_width = a_vals[1]-a_vals[0]
        print("Min bin width used for subcyling",a1_width*config.cm_to_um)
        dt_acc = epsilon_cycle*a1_width/dadt;
        if (dt_acc < dt_Gyr):
            total_cycles = np.ceil(np.abs(dt_Gyr/dt_acc)); 
            dt_subcycle = dt_Gyr/total_cycles;
    elif subcycle_constraints == 'size':
        epsilon_cycle = 0.0005
        atotal_width = amax_cm-amin_cm
        dt_acc = epsilon_cycle*atotal_width/dadt;
        if (dt_acc < dt_Gyr):
            total_cycles = np.ceil(np.abs(dt_Gyr/dt_acc)); 
            dt_subcycle = dt_Gyr/total_cycles;

    
    print("Total cycles needed", total_cycles, "Subcycle timestep", dt_subcycle)
    # This completes one full cycle to check if subcycling is needed. If not it's done, else it will loop over the needed number of subcycles
    while (n_cycle<total_cycles):
        if depl_frac >=1: break # All elements in dust so nothing to do
        # Need to recalculate dadt for each new cycle since key abundance will change
        if n_cycle != 0:
            # Need to recalculate dada for accretion since it depends on the key element abundance which changes as the dust grows
            if temp <= 300:
                key_num_dens = rho * key_abundance * (1 - depl_frac) / (key_mass*config.PROTONMASS)
                dadt = dadt_ref * (dust_atomic_weight / (key_num_atoms * np.sqrt(key_mass))) * key_num_dens * np.sqrt(temp * temp_clump_factor) / rho_c * Coulomb_enhancement * eff_clump_factor; # change in cm/Gyr
        da = dadt*dt_subcycle        
        if n_cycle != 0:
            for i in range(len(final_N)):
                init_N[i] = final_N[i]
                init_M[i] = final_M[i]

        N_update = np.zeros(len(a_vals)-1)
        M_update = np.zeros(len(a_vals)-1)

        # Loop over all bins to determine the change in grain mass and number
        for j in range(bin_num):
            aj_upper = a_vals[j+1]; aj_lower = a_vals[j];
            # Determine how many grains in bin i move to bin j given da
            for i in range(bin_num):
                ai_upper = a_vals[i+1]; ai_lower = a_vals[i]
                intersect = np.array([np.max([aj_lower-da, ai_lower]), np.min([aj_upper-da, ai_upper])])
                
                # No intersection between bin j and bin i grains given da/dt
                if intersect[0]>intersect[1] or intersect[0]>amax_cm or intersect[1]<amin_cm: 
                    continue
                # Grains in bin i will move into bin j
                else:
                    # Nothing beyond min/max grain size
                    intersect[0]=np.max([intersect[0],amin_cm])
                    intersect[1]=np.min([intersect[1],amax_cm])

                    N_update[j] += init_N[i]/(ai_upper-ai_lower)*(intersect[1]-intersect[0])
                    M_update[j] += 4*np.pi/3*rho_c * init_N[i]/(4*(ai_upper-ai_lower))*(np.power(intersect[1]+da,4)-np.power(intersect[0]+da,4))
        
        total_new_M = np.sum(M_update)
        total_init_M = np.sum(init_M)

        if total_new_M/total_init_M > 1/depl_frac:
            limit_fac = (1/depl_frac)/(total_new_M/np.sum(init_M))
            M_update *= limit_fac
            N_update *= limit_fac
            print('Mass of grains after update will exceed available mass. Rescaling all masses to avoid this.')
            print("Change in grains factoring in depletion fraction", depl_frac)   
            depl_frac = 1; # All elements are now in dust
        # Only need to update depletion factor
        else:
            depl_frac = total_new_M / total_init_M * depl_frac
        
        final_M = M_update
        final_N = N_update
        n_cycle+=1


    final_total_N = np.sum(final_N)
    final_total_M = np.sum(final_M)

    print("Change in total grain number",final_total_N/initial_total_N)
    print("Change in total grain mass",final_total_M/initial_total_M)
    print("Initial and final depletion fraction", init_depl_frac, depl_frac)
    # Final grain size distribution normalized back to total mass and total number
    a_bins_widths = (a_vals[1:]-a_vals[:-1])*config.cm_to_um
    a_centers = (a_vals[1:]+a_vals[:-1])/2
    dni_da = final_N / a_bins_widths / final_total_N
    final_dnda = interp1d(a_centers*config.cm_to_um,dni_da)
    dmi_da = final_M / a_bins_widths / final_total_M
    final_dmda = interp1d(a_centers*config.cm_to_um,dmi_da)
    return a_centers*config.cm_to_um, final_dnda, final_dmda




def change_in_grain_distribution_from_shat_coag(dt, amin=1E-3, amax=1E0, bin_num=5, init_dnda = MRN_dnda, depl_frac=0.1, subcycle_constraints='both', ISM_phase='simple', species='silicates'):
    """
    Calculates the change in grain size distribution due to shattering and coagulation over given time step.
    Parameters:
    - dt (float): Time step in Gyr.
    - amin (float): Minimum grain size (um).
    - amax (float): Maximum grain size (um).
    - bin_num (int): Number of bins for grain size distribution.
    - init_dnda (function): Initial grain size distribution function.
    - subcycle_constraints (str): Subcycling constraints, options are 'mass', 'number', or 'both'.
    - ISM_phase (str): Interstellar medium phase, options are 'WIM', 'MC', or 'simple'.
    - species (str): Dust species, options are 'silicates', 'carbonaceous', or 'iron'.
    Returns:
    - a_centers (ndarray): Centers of the grain size bins (um).
    - final_dnda (function): Final grain size distribution function.
    - final_dmda (function): Final grain mass distribution function.
    """

    amin*=config.um_to_cm; amax*=config.um_to_cm
    dt_sec = dt * config.Gyr_to_sec
    # used for subcycling timesteps
    M_epsilon_cycle = 0.1 
    N_epsilon_cycle = 0.1

    ISM_phase_props = ISM_phase_properties(ISM_phase)
    nH = ISM_phase_props['nH']
    rho = ISM_phase_props['rho']
    temp = ISM_phase_props['temp']
    M = ISM_phase_props['M']
    
    # Typical mass resolution (Msol) of FIRE simulations
    M_cell = 7100*config.Msolar_to_g;
    V_cell = M_cell/rho

    # Physical properties of dust species needed for calculations
    spec_props = dust_species_properties(species)
    dust_atomic_weight = spec_props['dust_atomic_weight']
    key_mass = spec_props['key_mass']
    key_abundance = spec_props['key_abundance']
    key_num_atoms = spec_props['key_num_atoms'] 
    P1 = spec_props['P1']
    v_shat = spec_props['v_shat']
    poisson = spec_props['poisson']
    youngs = spec_props['youngs']
    gamma = spec_props['gamma']
    rho_c = spec_props['rho_c']

    # Get the total mass of the dust species given the initial depletion
    total_mass = M_cell * key_abundance * depl_frac * dust_atomic_weight / (key_num_atoms * key_mass)

    # Clumping factor which enhances dust-dust interactions
    b = 0.5
    eff_clumping_factor = 1+b*b*M*M


    a_vals = np.logspace(np.log10(amin),np.log10(amax),bin_num+1)
    a_centers = (a_vals[1:]+a_vals[:-1])/2

    init_dmda = lambda a: init_dnda(a) * 4*np.pi/3*rho_c * np.power(a,3)
    init_norm = total_mass / quad(init_dmda,amin,amax)[0]

    init_M = np.zeros(len(a_vals)-1)
    init_N = np.zeros(len(a_vals)-1)
    final_M = np.zeros(len(a_vals)-1)
    final_N = np.zeros(len(a_vals)-1)

    for i in range(len(a_vals)-1):
        ai_upper = a_vals[i+1]
        ai_lower = a_vals[i]
        init_N[i] = quad(init_dnda,ai_lower,ai_upper)[0]*init_norm
        init_M[i] = quad(init_dmda,ai_lower,ai_upper)[0]*init_norm
        final_N[i]= init_N[i]
        final_M[i]= init_M[i]

    initial_total_N = np.sum(init_N)
    initial_total_M = np.sum(init_M)

    total_cycles = 0
    n_cycle = 0
    dt_cycle= dt_sec
    total_dt = 0

    # This completes one full cycle to check if subcycling is needed. If not it's done, else it will loop over the needed number of subcycles
    while (n_cycle<=total_cycles):
        dM_dt = np.zeros(len(a_vals)-1)
        for i in range(len(a_vals)-1):
            ai_upper = a_vals[i+1]
            ai_lower = a_vals[i]
            if n_cycle != 0:
                init_N[i] = final_N[i]
                init_M[i] = final_M[i]

        for i in range(len(a_vals)-1):
            ai_upper = a_vals[i+1]
            ai_lower = a_vals[i]
            ai_center = (ai_upper + ai_lower)/2
            # grains can shatter to below the minimum size. Need to include them in the mshat calc to conserve mass
            if i==0: ai_lower=0 
            mi_acenter = 4*np.pi/3*rho_c*np.power(ai_center,3)


            removal_term = 0
            injection_term = 0
            for j in range(len(a_vals)-1):
                aj_upper = a_vals[j+1]
                aj_lower = a_vals[j]
                aj_center = (aj_upper + aj_lower)/2
                # grains can shatter to below the minimum size. Need to include them in the mshat calc to conserve mass
                if j==0: aj_lower=0 
                

                int_I_ij = ((2*np.power(ai_lower,2) + 2*ai_lower*ai_upper + 2*np.power(ai_upper,2) + 3*ai_lower*(aj_lower + aj_upper) + 3*ai_upper*(aj_lower + aj_upper) + 2*(np.power(aj_lower,2) + aj_lower*aj_upper + np.power(aj_upper,2)))*init_N[i]*init_N[j])/6.

                
                vijrel = grain_relative_velocity(ai_center, aj_center, rho_c, ISM_phase)
                v_coag = v_coagulation(ai_center, aj_center, rho_c, poisson, youngs, gamma)
                if vijrel > v_shat or vijrel < v_coag:
                    removal_term += vijrel * mi_acenter * int_I_ij

                for k in range(len(a_vals)-1):
                    ak_upper = a_vals[k+1]
                    ak_lower = a_vals[k]
                    ak_center = (ak_upper + ak_lower)/2
                    # grains can shatter to below the minimum size. Need to include them in the mshat calc to conserve mass
                    if k==0: ak_lower=0

                    int_I_kj = ((2*np.power(aj_lower,2) + 2*aj_lower*aj_upper + 2*np.power(aj_upper,2) + 3*aj_lower*(ak_lower + ak_upper) + 3*aj_upper*(ak_lower + ak_upper) + 2*(np.power(ak_lower,2) + ak_lower*ak_upper + np.power(ak_upper,2)))*init_N[j]*init_N[k])/6.
                    vkjrel = grain_relative_velocity(ak_center, aj_center, rho_c, ISM_phase)
                    v_coag = v_coagulation(ak_center, aj_center, rho_c, poisson, youngs, gamma)
                    if vkjrel > v_shat:
                        mshat_kj = m_shatter(ai_lower, ai_upper, ak_center, aj_center, vkjrel, P1, rho_c, v_shat)
                        injection_term += vkjrel * mshat_kj * int_I_kj
                    elif vkjrel < v_coag:
                        mcoag_kj = m_coagulation(ai_lower, ai_upper, ak_center, aj_center, vkjrel, v_coag, rho_c)
                        injection_term += vkjrel * mcoag_kj * int_I_kj

            dM_dt[i] = eff_clumping_factor * np.pi * (-removal_term + injection_term)


        dM = dM_dt*dt_cycle/V_cell
        # Deal with moving more mass than available in a bin
        dM[dM < -init_M] = -init_M[dM < -init_M]
        dN = dM/(4*np.pi/3*rho_c * np.power(a_centers,3))



        # The total mass moving between bins in the given timestep
        moved_dMdt = np.abs(np.sum(-dM[dM<0] / dt_cycle))
        # The total reduction in grain number in the given timestep 
        # Only consider reduction since we are targeting coagulation with this timestep restritction
        moved_dNdt = np.abs(np.sum(-dN[dM<0] / dt_cycle))
        if n_cycle == 0 and subcycle_constraints is not None:
            # Determine if we need to subcycle based on mass transfer criteria
            # If we don't then update the bin masses and be done
            if subcycle_constraints == 'mass': tau_coll = M_epsilon_cycle * np.sum(init_M)/moved_dMdt
            elif subcycle_constraints == 'number': 
                tau_coll = N_epsilon_cycle * np.sum(init_N)/moved_dNdt
            else: tau_coll = np.min([M_epsilon_cycle*np.sum(init_M)/moved_dMdt, N_epsilon_cycle*np.sum(init_N)/moved_dNdt])
            if tau_coll < dt_cycle:
                total_cycles = np.ceil(dt_sec/tau_coll)
                dt_cycle = dt_sec/total_cycles;
                print("Need to subcycle for %i cycle"%total_cycles)
                print("\t Number fraction changed", np.sum(-dN[dN<0])/np.sum(init_N))
                print("\t Mass fraction change", np.sum(-dM[dM<0])/np.sum(init_M))

                print("\t Change if we did not subcycle")
                print("\t Mass fraction change in given timestep", np.sum(init_M+dM)/np.sum(init_M))
                print("\t Number fraction change in given timestep", np.sum(init_N+dN)/np.sum(init_N))
            else:    
                final_M = init_M + dM
                final_N = final_M/(4*np.pi/3*rho_c * np.power(a_centers,3))
                total_dt+=dt_cycle
        else:
            final_M = init_M + dM
            final_N = final_M/(4*np.pi/3*rho_c * np.power(a_centers,3))
            total_dt+=dt_cycle

        n_cycle+=1           

    final_total_N = np.sum(final_N)
    final_total_M = np.sum(final_M)

    print("Final changes")
    print("Mass fraction moved in given timestep", final_total_M/initial_total_M)
    print("Number fraction changed in given timestep", final_total_N/initial_total_N)

    # Final grain size distribution normalized back to total mass and total number
    a_bins_widths = (a_vals[1:]-a_vals[:-1])*config.cm_to_um
    dni_da = final_N / a_bins_widths / final_total_N
    final_dnda = interp1d(a_centers*config.cm_to_um,dni_da)
    dmi_da = final_M / a_bins_widths / final_total_M
    final_dmda = interp1d(a_centers*config.cm_to_um,dmi_da)
    return a_centers*config.cm_to_um, final_dnda, final_dmda
    





def grain_relative_velocity(a1, a2, rho_c, ISM_phase):
    """
    Calculate the relative velocity between two grains.

    Parameters:
    - a1 (float): Size of the first grain in centimeters.
    - a2 (float): Size of the second grain in centimeters.
    - vrel_case (str): Case determing environment for calculating the relative velocity. Possible values are 'simple' and 'WIM'.
    - rho_c (float, optional): Density of dust material in grams per cubic centimeter. Default is 1.

    Returns:
    - vrel (float): Relative velocity between the two grains in centimeters per second.
    """

    # Assume a simple case of only large grains moving fast relative to each other.
    if ISM_phase == 'simple':
        if a1>0.1*config.um_to_cm and a2>0.1*config.um_to_cm: vrel = 3E5 # cm/s
        else: vrel = 0
        return vrel
    else:
        ISM_phase_props = ISM_phase_properties(ISM_phase)
        nH = ISM_phase_props['nH']
        rho = ISM_phase_props['rho']
        temp = ISM_phase_props['temp']
        M = ISM_phase_props['M']

    # For consitencany in calculation we assume the impact angle is always 90 degrees
    # cos_imp_angle = 2*(np.random.rand()-0.5)
    cos_imp_angle=0

    vgr1 = 0.96E5 * np.power(M,1.5) * np.power(a1/0.1E-4,0.5) * np.power(temp/1E4,0.25) * \
            np.power(rho/(1*config.H_MASS),-0.25)*np.power(rho_c/3.5,0.5) # cm/s
    vgr2 = 0.96E5 * np.power(M,1.5) * np.power(a2/0.1E-4,0.5) * np.power(temp/1E4,0.25) * \
            np.power(rho/(1*config.H_MASS),-0.25) * np.power(rho_c/3.5,0.5) #cm/s
    v12rel = np.sqrt(vgr1*vgr1 + vgr2*vgr2 - 2*vgr1*vgr2*cos_imp_angle) # cm/s
    return v12rel
    


def m_shatter(alower, aupper, a1, a2, vrel, P1, rho_c, vshat):
    """
    Calculate the mass of shattered fragments from dust grain 1 given a collision with dust grain 2.

    Parameters:
    alower (float): Lower limit of the size range for integration (cm).
    aupper (float): Upper limit of the size range for integration (cm).
    a1 (float): Size of the impacted particle 1 (cm).
    a2 (float): Size of colliding particle 2 (cm).
    vrel (float): Relative velocity between the particles (cm/s).
    P1 (float): Critical shock pressure of dust grain (dyn cm^-3).
    rho_c (float): Density of the particle (g cm^-3).
    vshat (float): Shattering velocity above which shattering occurs (cm/s).

    Returns:
    float: The mass of shattered fragments.

    """


    if (vrel<vshat): return 0
    ma1 = 4*np.pi/3 * rho_c * np.power(a1,3)
    ma2 = 4*np.pi/3 * rho_c * np.power(a2,3)

    QDstar = P1/(2*rho_c)
    Eimp = 1/2 * ma1 * ma2 / (ma1 + ma2) * vrel*vrel
    phi = Eimp / (ma1 * QDstar)
    mej = phi / (1 + phi) * ma1
    aej = np.power(mej / (4 * np.pi /3 * rho_c),1/3)
    arem = np.power(np.power(a1,3) - np.power(aej,3),1/3)
    mrem = ma1-mej

    # Min/max sizes of shattered fragments and normalization constant for size fragment distribution asuming dn_frag/da = C_frag a^-3.3
    afrag_max = np.power(0.02*mej/ma1,1/3) * a1
    afrag_min = 0.01*afrag_max

    C_frag = mej / ((10/7*4/3*np.pi*rho_c)*(np.power(afrag_max,0.7) - np.power(afrag_min,0.7)))


    a_int_lower = alower
    a_int_upper = aupper

    m_shat=0
    # The mass of the remnant after ma1 shattering
    if arem > a_int_lower and arem < a_int_upper:
        m_shat += mrem
    if a_int_lower < afrag_max and a_int_upper > afrag_min:
        if a_int_lower < afrag_min and a_int_upper > afrag_min: a_int_lower = afrag_min
        if a_int_upper > afrag_max and a_int_lower < afrag_max: a_int_upper = afrag_max
        # The mass of fragments shattered from ma1
        m_shat += 4/3*np.pi*rho_c * C_frag * 10/7 * (np.power(a_int_upper,0.7) - np.power(a_int_lower,0.7))

    return m_shat


def v_coagulation(a1, a2, rho_c, poisson, youngs, gamma):
    """
    Calculate the coagulation velocity between two colliding grains.
    Parameters:
    a1 (float): Radius of the first grain (cm).
    a2 (float): Radius of the second grain (cm).
    rho_c (float): Density of the grains (g cm^-3).
    poisson (float): Poisson's ratio of the grains.
    youngs (float): Young's modulus of the grains.
    gamma (float): Surface tension coefficient of grains.
    Returns:
    float: Coagulation velocity between the two colliding grains.
    """
    
    poisson1 = poisson;
    youngs1 = youngs;
    # assume colliding grains are same species
    poisson2=poisson1; youngs2=youngs1

    R12 = a1*a2/(a1+a2) # reduced radius of colliding grains
    Estar = 1 / (np.square(1-poisson1)/youngs1 + np.square(1-poisson2)/youngs2) # reduced elastic modules 
    v_coag = 21.4 * np.sqrt((np.power(a1,3)+np.power(a2,3)) / np.power(a1+a2,3)) * np.power(gamma,5/6) / (np.power(Estar,1/3) * np.power(R12,5/6) * np.sqrt(rho_c))

    return v_coag


def m_coagulation(alower, aupper, a1, a2, vrel, vcoag, rho_c):
    """
    Calculate the coagulation mass for two particles.

    Parameters:
    alower (float): Lower limit of particle size range (cm).
    aupper (float): Upper limit of particle size range (cm).
    a1 (float): Size of particle 1 (cm).
    a2 (float): Size of particle 2 (cm).
    vrel (float): Relative velocity between particles (cm/s).
    vcoag (float): Coagulation velocity threshold (cm/s).
    rho_c (float): Density of particles (g cm^-3).

    Returns:
    float: Coagulation mass if coagulation occurs, 0 otherwise.
    """

    if (vrel > vcoag): return 0
    ma1 = 4*np.pi/3 * rho_c * np.power(a1,3)
    ma2 = 4*np.pi/3 * rho_c * np.power(a2,3)

    a_coag = np.power((ma1+ma2) / (4*np.pi/3 * rho_c),1/3)
    if alower < a_coag and a_coag < aupper:
        m_coag= (ma1+ma2)/2
    else:
        m_coag = 0

    return m_coag



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
