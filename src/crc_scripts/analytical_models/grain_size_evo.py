import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d,CubicSpline
from scipy.special import erfc,erf
from astropy.table import Table
import os
from .. import config
from ..config import dust_species_properties


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
        M=2
    elif ISM_phase == 'CNM':
        nH = 30
        rho = nH * config.PROTONMASS; 
        temp = 100
        M=3   
    elif ISM_phase == 'MC':
        nH = 1E4
        rho = nH * config.PROTONMASS; 
        temp = 10
        M=10
    else:
        nH = 1
        rho = 1 * config.PROTONMASS;
        temp = 1000
        M=3.3


    ISM_phase_props = {'nH':nH, 'rho':rho, 'temp':temp, 'M':M}
    return ISM_phase_props


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

    M_cell = config.FIRE_GAS_PARTICLE_MASS*config.Msolar_to_g;
    # dust to gas mass ratio for given species for given depletion used for normalization of initial size distribution
    DTG_spec = key_abundance*depl_frac*dust_atomic_weight / key_mass


    amin_cm = amin*config.um_to_cm; amax_cm = amax*config.um_to_cm
    a_vals = np.logspace(np.log10(amin_cm),np.log10(amax_cm),bin_num+1)
    a_centers = (a_vals[1:]+a_vals[:-1])/2
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


    # Get dadt for the smallest grain size bin
    # Accretion occurs below 300 K
    if temp <= 300:
        dadt_ref = 1.91249E-4 # reference change in grain size in cm/Gyr assuming purely hard-sphere type encounters
        # Determine clumping factor due to subresolved gas-dust clumping using assumed Mach number
        b = 0.5
        sigma = np.sqrt(np.log(1+b*b*M*M))
        temp_clump_factor = 1/(np.exp(sigma*sigma)/2 * (1 + erf((3/2*sigma*sigma + np.log(nH_max/nH)) / (np.sqrt(2)*sigma))))
        eff_clump_factor = np.exp(sigma*sigma)/2 * erfc((3/2*sigma*sigma-np.log(nH_max/nH)) / (np.sqrt(2)*sigma))

        # Determine Coulomb enhancement factor
        if species == 'silicates':
            b0=1.96617;b1=-0.910511;b2=0.0150985;b3=-0.906869;b4=0.580115;b5=-0.102265;
        elif species == 'carbonaceous':
            b0=1.76851;b1=-1.4336;b2=-0.344758;b3=0.420086;b4=-0.641419;b5=-0.585337;
        elif species == 'iron':
            b0=2.14226;b1=-0.910511;b2=0.0150985;b3=-0.906869;b4=0.580115;b5=-0.102265;
        else:
            b1=0;b2=0;b3=0;b4=0;b5=0;
        nH_dense = 1E3
        fdense = 1/2+1/2*erf((sigma*sigma/2 - np.log(nH_dense/nH))/(np.sqrt(2)*sigma));

        # Fit to Weingartner & Draine 2001 Coulomb enhancement factor
        # Not using this since it is very high for small grains
        log_a_nano = np.log10(a_centers*config.cm_to_nm); # convert to nm
        Coulomb_enhancement = np.power(10,b0 + b1*log_a_nano + b2*log_a_nano*log_a_nano + b3*log_a_nano*log_a_nano*log_a_nano + b4*log_a_nano*log_a_nano*log_a_nano*log_a_nano + b5*log_a_nano*log_a_nano*log_a_nano*log_a_nano*log_a_nano)


        # New simple prescription for Coulomb enhancement
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

        Coulomb_enhancement = (1-fdense)*Coulomb_enhancement + fdense
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

        dadt = np.full(len(a_centers),-eff_clump_factor * nH * Y_sput * config.um_to_cm / 1E-9); # change to cm/Gyr
    else:
        dadt = np.zeros(len(a_centers))

    print("Predicted dadt (um/Gyr):",dadt*config.cm_to_um)


    # Determine if timestep subcycling is needed
    # Assume no subycycling as default
    total_cycles = 1; 
    dt_subcycle = dt_Gyr; 
    dadt_0 = dadt[0] # grain size change in smallest bin   
    if subcycle_constraints == 'min_bin':
        epsilon_cycle = 1
        a1_width = a_vals[1]-a_vals[0]
        print("Min bin width used for subcyling",a1_width*config.cm_to_um)
        dt_acc = epsilon_cycle*a1_width/dadt_0;
        if (dt_acc < dt_Gyr):
            total_cycles = np.ceil(np.abs(dt_Gyr/dt_acc)); 
            dt_subcycle = dt_Gyr/total_cycles;
    elif subcycle_constraints == 'size':
        epsilon_cycle = 0.0005
        atotal_width = amax_cm-amin_cm
        dt_acc = epsilon_cycle*atotal_width/dadt_0;
        if (dt_acc < dt_Gyr):
            total_cycles = np.ceil(np.abs(dt_Gyr/dt_acc)); 
            dt_subcycle = dt_Gyr/total_cycles;

    
    print("Total cycles needed", total_cycles, "Subcycle timestep", dt_subcycle)
    # This completes one full cycle to check if subcycling is needed. If not it's done, else it will loop over the needed number of subcycles
    n_cycle = 0
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
                dai = da[i]
                intersect = np.array([np.max([aj_lower-dai, ai_lower]), np.min([aj_upper-dai, ai_upper])])
                
                # No intersection between bin j and bin i grains given da/dt
                if intersect[0]>intersect[1] or intersect[0]>amax_cm or intersect[1]<amin_cm: 
                    continue
                # Grains in bin i will move into bin j
                else:
                    # Nothing beyond min/max grain size
                    intersect[0]=np.max([intersect[0],amin_cm])
                    intersect[1]=np.min([intersect[1],amax_cm])

                    N_update[j] += init_N[i]/(ai_upper-ai_lower)*(intersect[1]-intersect[0])
                    M_update[j] += 4*np.pi/3*rho_c * init_N[i]/(4*(ai_upper-ai_lower))*(np.power(intersect[1]+dai,4)-np.power(intersect[0]+dai,4))
        
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
    a_centers = (a_vals[1:]+a_vals[:-1])/2 *config.cm_to_um
    dni_da = final_N / a_bins_widths / final_total_N
    final_dnda = interp1d(a_centers,dni_da)
    dmi_da = final_M / a_bins_widths / final_total_M
    final_dmda = interp1d(a_centers,dmi_da)
    return a_centers, final_dnda, final_dmda




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
    M_cell = config.FIRE_GAS_PARTICLE_MASS*config.Msolar_to_g;
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
            mi_acenter = 4*np.pi/3*rho_c*np.power(ai_center,3)

            removal_term = 0
            injection_term = 0
            for j in range(len(a_vals)-1):
                aj_upper = a_vals[j+1]
                aj_lower = a_vals[j]
                aj_center = (aj_upper + aj_lower)/2

                int_I_ij = ((2*np.power(ai_lower,2) + 2*ai_lower*ai_upper + 2*np.power(ai_upper,2) + 3*ai_lower*(aj_lower + aj_upper) + 3*ai_upper*(aj_lower + aj_upper) + 2*(np.power(aj_lower,2) + aj_lower*aj_upper + np.power(aj_upper,2)))*init_N[i]*init_N[j])/6.
                
                vijrel = grain_relative_velocity(ai_center, aj_center, rho_c, ISM_phase)
                v_coag = v_coagulation(ai_center, aj_center, rho_c, poisson, youngs, gamma)
                # Sometimes v_coag can go above v_shat (mainly for metallic iron)
                
                if v_coag > v_shat: v_coag = v_shat
                if vijrel > v_shat or vijrel <= v_coag:
                    removal_term += vijrel * mi_acenter * int_I_ij

                for k in range(len(a_vals)-1):
                    ak_upper = a_vals[k+1]
                    ak_lower = a_vals[k]
                    ak_center = (ak_upper + ak_lower)/2

                    int_I_kj = ((2*np.power(aj_lower,2) + 2*aj_lower*aj_upper + 2*np.power(aj_upper,2) + 3*aj_lower*(ak_lower + ak_upper) + 3*aj_upper*(ak_lower + ak_upper) + 2*(np.power(ak_lower,2) + ak_lower*ak_upper + np.power(ak_upper,2)))*init_N[j]*init_N[k])/6.
                    vkjrel = grain_relative_velocity(ak_center, aj_center, rho_c, ISM_phase)
                    v_coag = v_coagulation(ak_center, aj_center, rho_c, poisson, youngs, gamma)
                    # Sometimes v_coag can go above v_shat (mainly for metallic iron)
                    if v_coag > v_shat: v_coag = v_shat
                    if vkjrel > v_shat:
                        mshat_kj = m_shatter(ai_lower, ai_upper, ak_center, aj_center, vkjrel, P1, rho_c, v_shat)
                        injection_term += vkjrel * mshat_kj * int_I_kj
                    elif vkjrel <= v_coag:
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
        if n_cycle == 0 and subcycle_constraints is not None and moved_dMdt != 0:
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
    





def grain_relative_velocity(a1, a2, rho_c, ISM_phase, scheme='HC23'):
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


    # Scheme from Hirashita & Aoyama 2019
    if scheme == 'HA19':
        # For consitencany in calculation we assume the impact angle is always 90 degrees
        # cos_imp_angle = 2*(np.random.rand()-0.5)
        cos_imp_angle=0

        # These are the velocities of indivdiual grains
        M=3.3
        vgr1 = 0.96E5 * np.power(M,1.5) * np.power(a1/0.1E-4,0.5) * np.power(temp/1E4,0.25) * \
                np.power(rho/(1*config.H_MASS),-0.25)*np.power(rho_c/3.5,0.5) # cm/s
        vgr2 = 0.96E5 * np.power(M,1.5) * np.power(a2/0.1E-4,0.5) * np.power(temp/1E4,0.25) * \
                np.power(rho/(1*config.H_MASS),-0.25) * np.power(rho_c/3.5,0.5) #cm/s
        v12rel = np.sqrt(vgr1*vgr1 + vgr2*vgr2 - 2*vgr1*vgr2*cos_imp_angle) # cm/s
    # Scheme from Hirashita & Chen 2023
    elif scheme == 'HC23':
        # For consitencany in calculation we assume the impact angle is always 90 degrees
        # cos_imp_angle = 2*(np.random.rand()-0.5)
        cos_imp_angle=0

        # Account for sub resolution clumping
        b = 0.5
        nH_rms =np.sqrt(1+b*b*M*M)*nH
        vgr1 = 10*0.32E5 * (M/3) * np.power(a1/1E-4,0.5) * np.power(temp/100,0.25) * \
                np.power(nH_rms/1E3,-0.25)*np.power(rho_c/3.5,0.5)
        vgr2 = 10*0.32E5 * (M/3) * np.power(a2/1E-4,0.5) * np.power(temp/100,0.25) * \
                np.power(nH_rms/1E3,-0.25)*np.power(rho_c/3.5,0.5)
        v12rel = np.sqrt(vgr1*vgr1 + vgr2*vgr2 - 2*vgr1*vgr2*cos_imp_angle) # cm/s
    # Scheme from Li+ 2019
    elif scheme == 'Li21':
        # These are velocity dispersions of grain sizes
        sigma_gr1 = 0.054*1E5 * np.power(M,2) * (a1/0.1E-4) * np.power(rho/(1*config.H_MASS),-0.5) * np.power(rho_c/2.4,0.5) # cm/s
        sigma_gr2 = 0.054*1E5 * np.power(M,2) * (a2/0.1E-4) * np.power(rho/(1*config.H_MASS),-0.5) * np.power(rho_c/2.4,0.5) #cm/s

        sigma_gr1 = 0.96E5 * np.power(M,1.5) * np.power(a1/0.1E-4,0.5) * np.power(temp/1E4,0.25) * \
                np.power(rho/(1*config.H_MASS),-0.25)*np.power(rho_c/3.5,0.5) # cm/s
        sigma_gr2 = 0.96E5 * np.power(M,1.5) * np.power(a2/0.1E-4,0.5) * np.power(temp/1E4,0.25) * \
                np.power(rho/(1*config.H_MASS),-0.25) * np.power(rho_c/3.5,0.5) #cm/s

        # Randomly sample each grains x,y, and z velocity components from a Gaussian normal distribution
        vgr1_x = np.random.normal(0,sigma_gr1*sigma_gr1/3)
        vgr1_y = np.random.normal(0,sigma_gr1*sigma_gr1/3)
        vgr1_z = np.random.normal(0,sigma_gr1*sigma_gr1/3)
        vgr2_x = np.random.normal(0,sigma_gr2*sigma_gr2/3)
        vgr2_y = np.random.normal(0,sigma_gr2*sigma_gr2/3)
        vgr2_z = np.random.normal(0,sigma_gr2*sigma_gr2/3)

        v12rel = np.sqrt((vgr1_x-vgr2_x)*(vgr1_x-vgr2_x) + (vgr1_y-vgr2_y)*(vgr1_y-vgr2_y) + (vgr1_z-vgr2_z)*(vgr1_z-vgr2_z))
    else:
        assert 0, "Scheme not supported"



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
    float: Coagulation velocity between the two colliding grains (cm/s).
    """
    
    poisson1 = poisson;
    youngs1 = youngs;
    # assume colliding grains are same species
    poisson2=poisson1; youngs2=youngs1

    R12 = a1*a2/(a1+a2) # reduced radius of colliding grains
    Estar = 1 / (np.square(1-poisson1)/youngs1 + np.square(1-poisson2)/youngs2) # reduced elastic modules 
    v_coag = 2.14 * np.sqrt((np.power(a1,3)+np.power(a2,3)) / np.power(a1+a2,3)) * np.power(gamma,5/6) / (np.power(Estar,1/3) * np.power(R12,5/6) * np.sqrt(rho_c))

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


class SNe_Dust_Processing(object):
    """
    Class to handle the dust processing by supernovae in the ISM.
    Methods:
    - __init__(dnda, a_limit, bins=500): Initializes the SNe_Dust_Processing object.
    - dnda_SNe_sputtering_approximation(dnda, a, delta_sput=0.1, a_sput=0.1): Returns the resulting dnda after a sputtering approximation step given an initial dnda.
    - dnda_SNe_shattering_approximation(dnda, a, delta_shat=0.1, a_shat=0.1, a_frag_max=0.1): Returns the resulting dnda after a shattering step given an initial dnda.
    - get_SNe_processed_dnda(approx='shat_sput_shat', **kwargs): Returns the processed dnda based on the given approximation scheme.
    - shat_sput_shat(delta_sput=0.1, delta_shat=0.1, a_sput=0.05, a_shat=0.05, a_frag_max=0.05): Returns the processed dnda using the 'shat_sput_shat' approximation scheme.
    - sput_only(delta_sput=0.1, a_sput=0.05): Returns the processed dnda using the 'sput' approximation scheme.
    """


    def __init__(self, dnda, a_limit, bins=500):
        """
        Parameters:
        - dnda (function): Initial grain size distribution function.
        - a_limit (ndarray): Grain size limits.
        - bins (int): Number of bins for grain size distribution
        """

        self.init_dnda = dnda
        # Renorm grain size distribution to one for ease comparison
        total_N = quad(self.init_dnda, a_limit[0], a_limit[1])[0]
        self.init_dnda = lambda a: dnda(a)/total_N
        self.init_dmda = lambda a: np.power(a,3) * dnda(a)/total_N
        self.amin = a_limit[0]
        self.amax = a_limit[1]
        self.a_values = np.logspace(np.log10(self.amin),np.log10(self.amax),bins)

    # Returns the resulting dnda after a sputtering approximation step given an initial dnda
    def dnda_SNe_sputtering_approximation(self, dnda, a, delta_sput=0.1, a_sput=0.1):
        """
        Calculates the approximate grain size evolution for the SNe sputtering approximation. All size units in microns.

        Parameters:
        - dnda (function): The function that calculates the initial grain size distribution.
        - a (float): The grain size.
        - delta_sput (float, optional): The sputtering parameter. Defaults to 0.1.
        - a_sput (float, optional): The characteristic grain size for sputtering. Defaults to 0.1.

        Returns:
        - float: The grain size function after sputtering.
        """
        def eff_sput(a, delta_sput, a_sput):
            return (1 - np.exp(-(delta_sput / (a/a_sput))))
        return dnda(a)*(1-eff_sput(a,delta_sput,a_sput))

    # Returns the resulting dnda after a shattering step given an initial dnda
    def dnda_SNe_shattering_approximation(self, dnda, a, delta_shat=0.1, a_shat=0.1, a_frag_max=0.1):
        """
        Calculate the grain size distribution after Sne shattering approximation. All size units in microns.

        Parameters:
        - dnda (function): The function that represents the initial grain size distribution.
        - a (float): The grain size.
        - delta_shat (float, optional): The shattering efficiency parameter. Default is 0.1.
        - a_shat (float, optional): The characteristic grain size for shattering. Default is 0.1.
        - a_frag_max (float, optional): The maximum grain size for fragmentation. Default is 0.1.

        Returns:
        - ndarray: The grain size distribution after shattering.
        """
        def eff_shat(a, delta_shat, a_shat):
            return (1 - np.exp(-(delta_shat * (a/a_shat))))
        def dmda_shattered(a, dnda):
            return eff_shat(a, delta_shat, a_shat) * dnda(a) * a * a * a
        def dnda_fragments(a, C_frag):
            return np.piecewise(a, [a <= a_frag_max, a > a_frag_max], [lambda a: C_frag * np.power(a,-3.3), 0])

        M_shattered = quad(dmda_shattered, self.amin, self.amax, args=(dnda))[0]
        C_frag = M_shattered * 7 / (10 * (np.power(a_frag_max,0.7) - np.power(self.amin,0.7)))
        return dnda_fragments(a, C_frag) + dnda(a)*(1-eff_shat(a, delta_shat, a_shat))
    

    def get_SNe_processed_dnda(self, approx='shat_sput_shat', species=None, **kwargs):
        """
        Calculate the SNe processed grain size distribution.

        Parameters:
        - approx (str): The approximation scheme to use. Supported values are 'sput_shat_sput', 'sput', and 'nozawa'.
        - **kwargs: Additional keyword arguments specific to the chosen approximation scheme.

        Returns:
        - ndarray: The processed grain size distribution.

        Raises:
        - AssertionError: If the given approximation scheme is not supported.
        """
        assert approx in ['shat_sput_shat','sput','nozawa'], "Given approximation scheme not supported"

        # If species given then override parameters 
        if species is not None and approx != 'nozawa':
            spec_props = dust_species_properties(species)
            kwargs['delta_sput'] = spec_props['delta_sput']
            kwargs['delta_shat'] = spec_props['delta_shat']
            kwargs['a_sput'] = 0.05
            kwargs['a_shat'] = 0.05
            kwargs['a_frag_max'] = 0.05

        if approx == 'shat_sput_shat':
            return self.shat_sput_shat(**kwargs)
        elif approx == 'sput':
            return self.sput_only(**kwargs)
        elif approx == 'nozawa':
            return self.Nozawa_prescription(**kwargs)


    def shat_sput_shat(self, delta_sput=0.1, delta_shat=0.1, a_sput=0.05, a_shat=0.05, a_frag_max=0.05):
        """
        Calculates the destruction fraction, final size distribution, and final mass distribution
        for an approximate shatter, sputter, shatter scheme. All size units in microns.
        Parameters:
        - delta_sput (float): The sputtering parameter.
        - delta_shat (float): The shattering parameter.
        - a_sput (float): The sputtering efficiency parameter.
        - a_shat (float): The shattering efficiency parameter.
        - a_frag_max (float): The maximum fragmentation efficiency parameter.
        Returns:
        - dest_frac (float): The destruction fraction, which represents the fraction of the initial mass that is lost.
        - final_dnda (CubicSpline): The final size distribution.
        - final_dmda (CubicSpline): The final mass distribution.
        """

        initial_mass = quad(self.init_dmda, self.amin, self.amax)[0]
        dnda_shat1 = self.dnda_SNe_shattering_approximation(self.init_dnda,self.a_values,delta_shat=delta_shat,a_shat=a_shat,a_frag_max=a_frag_max)
        dnda_shat1 = CubicSpline(self.a_values,dnda_shat1); 
        
        dnda_sput1 = self.dnda_SNe_sputtering_approximation(dnda_shat1,self.a_values,delta_sput=delta_sput,a_sput=a_sput)
        dnda_sput1 = CubicSpline(self.a_values,dnda_sput1); 
        
        dnda_shat2 = self.dnda_SNe_shattering_approximation(dnda_sput1,self.a_values,delta_shat=delta_shat,a_shat=a_shat,a_frag_max=a_frag_max)
        dnda_shat2 = CubicSpline(self.a_values,dnda_shat2); 
        
        final_dnda = dnda_shat2
        total_N = quad(final_dnda, self.amin, self.amax)[0]
        final_dmda = CubicSpline(self.a_values, np.power(self.a_values,3) * final_dnda(self.a_values))
        total_M = quad(final_dmda, self.amin, self.amax)[0]
        dest_frac = 1-total_M/initial_mass
        # # Now renorm dnda and dmda to one
        # final_dnda = CubicSpline(self.a_values,final_dnda(self.a_values)/total_N); 
        # final_dmda = CubicSpline(self.a_values,final_dmda(self.a_values)/total_M); 
    
        return dest_frac, final_dnda, final_dmda

    def sput_only(self, delta_sput=0.1, a_sput=0.05):
        """
        Calculate the approximate sputtering-only evolution of grain size distribution. All size units in microns.
        Parameters:
        - delta_sput (float): The sputtering efficiency parameter.
        - a_sput (float): The sputtering yield parameter.
        Returns:
        - dest_frac (float): The fraction of mass lost due to sputtering.
        - final_dnda (CubicSpline): The final grain size distribution.
        - final_dmda (CubicSpline): The final grain mass distribution.
        """

        initial_mass = quad(self.init_dmda, self.amin, self.amax)[0]

        dnda_sput1 = self.dnda_SNe_sputtering_approximation(self.init_dnda,self.a_values,delta_sput=delta_sput,a_sput=a_sput)
        dnda_sput1 = CubicSpline(self.a_values,dnda_sput1); 

        final_dnda = dnda_sput1
        total_N = quad(final_dnda, self.amin, self.amax)[0]
        final_dmda = CubicSpline(self.a_values, np.power(self.a_values,3) * final_dnda(self.a_values))
        total_M = quad(final_dmda, self.amin, self.amax)[0]
        dest_frac = 1-total_M/initial_mass

        # # Now renorm dnda and dmda to one
        # final_dnda = CubicSpline(self.a_values,final_dnda(self.a_values)/total_N); 
        # final_dmda = CubicSpline(self.a_values,final_dmda(self.a_values)/total_M); 

        return dest_frac, final_dnda, final_dmda
    
    def Nozawa_prescription(self, smooth=True):

        initial_mass = quad(self.init_dmda, self.amin, self.amax)[0]

        # Load the Nozawa data tables for nH = 1 cm^-3
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, 'Nozawa_nH1.dat')
        Nozawa_table = Table.read(file_path,format='ascii')
        # Convert all sizes to microns
        ac_init = np.unique(Nozawa_table['a_init (cm)'].data)*1E4
        a1 = (Nozawa_table['a_init (cm)'].data)*1E4
        a2 = (Nozawa_table['a_final (cm)'].data)*1E4
        frac_i_to_f = Nozawa_table['Mg2SiO4'].data
        # Slight rounding errors in the Nozawa table means you can miss the size limit boundaries so extended them slightly
        ac_init = ac_init[(ac_init>=self.amin*0.9) & (ac_init<=self.amax*1.1)]
        ac_final = np.copy(ac_init)
        a_binwidth = 0.1 # dex

        
        # Check how many grains in each in bin move to other bins
        after_N_bin = np.zeros(len(ac_init))
        for i,ac in enumerate(ac_init):
            if i==0:
                bin_limits = [self.amin,np.power(10,np.log10(self.amin)+(a_binwidth/2))]
            elif i == len(ac_init)-1:
                bin_limits = [np.power(10,np.log10(self.amax)-(a_binwidth/2)),self.amax]
            else:
                bin_limits = [np.power(10,np.log10(ac)-(a_binwidth/2)),np.power(10,np.log10(ac)+(a_binwidth/2))]

            N_bin = quad(self.init_dnda,bin_limits[0],bin_limits[1])[0]
            total_frac = 0
            for j,af in enumerate(ac_final):
                if af > ac: continue
                frac = frac_i_to_f[(a1==ac) & (a2==af)][0]
                after_N_bin[j] += frac * N_bin
                total_frac+=frac
        
        after_dnda = np.zeros(len(after_N_bin))
        for i,ac in enumerate(ac_init):
            if i==0:
                bin_limits = [self.amin,np.power(10,np.log10(self.amin)+(a_binwidth/2))]
            elif i == len(ac_init)-1:
                bin_limits = [np.power(10,np.log10(self.amax)-(a_binwidth/2)),self.amax]
            else:
                bin_limits = [np.power(10,np.log10(ac)-(a_binwidth/2)),np.power(10,np.log10(ac)+(a_binwidth/2))]
        
            after_dnda[i] = after_N_bin[i]/(bin_limits[1]-bin_limits[0])
        
        
        # Need to smooth out dnda since this can have sharp dips for whatever reason
        if smooth:
            for i,ac in enumerate(ac_init):
                if i==0 or i ==len(ac_init)-1:
                    continue
                
                if after_dnda[i] < after_dnda[i-1] and after_dnda[i] < after_dnda[i+1]:
                    after_dnda[i] = np.power(10,(np.log10(after_dnda[i-1]) + np.log10(after_dnda[i+1]))/2)

        dnda_Nozawa = CubicSpline(ac_init,after_dnda)
        dmda_Nozawa = CubicSpline(ac_init,after_dnda*np.power(ac_init,3))
        f_nozawa = 1-quad(dmda_Nozawa, self.amin, self.amax)[0]/initial_mass

        return f_nozawa, dnda_Nozawa, dmda_Nozawa