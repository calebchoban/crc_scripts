#Phil's routine!!!

import numpy as np

def gas_mu(P):
    num_e = P['ne']
    XH = P['z'][:,1];
    yhelium = XH/4.*(1-XH);
    return (1.+4.*yhelium)/(1.+yhelium+num_e);


def gas_temperature(P, keV=0):
    ## returns gas particles temperature in Kelvin

    g_gamma= 5.0/3.0
    g_minus_1= g_gamma-1.0
    PROTONMASS = 1.6726e-24
    BoltzMann_ergs= 1.3806e-16
    UnitMass_in_g= 1.989e43 # 1.0e10 solar masses
    UnitEnergy_in_cgs= 1.989e53
    UnitVelocity_in_cm_per_s = 1e5
    # note gadget units of energy/mass = 1e10 ergs/g,
    # this comes into the formula below

    u = P['u']

    mu = gas_mu(P);
    MeanWeight= mu*PROTONMASS
    Temp= MeanWeight/BoltzMann_ergs * g_minus_1 * u * UnitVelocity_in_cm_per_s**2

    # do we want units of keV?  (0.001 factor converts from eV to keV)
    if (keV==1):
        BoltzMann_keV = 8.617e-8;
        Temp *= BoltzMann_keV;

    return Temp


def gas_cs_effective_eos(u, q_eos=1.0):
    ## returns gas sound speed in km/s (for old effective gadget equation-of-state)
    ## u is in GADGET units = (kpc/Gyr)^2 = 0.957*(km/s)^2 
    ## actually no, time unit in gadget isnt Gyr, exactly (velocity unit *is* km/s
    g_gamma = 5./3.
    u_min = 100. ## 10^4 K -- *roughly* 10 km/s, though exact conversion depends on mean molecular weight, etc.
    cs_gas = np.sqrt(g_gamma*(g_gamma-1.) * (q_eos*u + (1.-q_eos)*u_min))
    return cs_gas


def gas_xray_brems(mass_in_gadget_units, u_in_gadget_units, rho_in_gadget_units, num_e, num_h):
    ## returns gas x-ray bremstrahhlung luminosity (x-ray line cooling is separate)
    protonmass = 1.6726e-24;
    brem_normalization= 1.2e-24;
    m = mass_in_gadget_units * 1.989e43; ## convert to g
    u = u_in_gadget_units;

    MeanWeight = gas_mu(num_e) * protonmass;
    keV = gas_temperature(u, num_e, keV=1);
    density = rho_in_gadget_units * 6.76991e-22 ## convert to g/cm3

    xray_lum = brem_normalization * (m/MeanWeight) * (density/MeanWeight) * \
        np.sqrt(keV) * num_e * (1.-num_h);

    return xray_lum
