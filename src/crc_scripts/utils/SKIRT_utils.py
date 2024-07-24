from astropy.io import ascii
import numpy as np

from .math_utils import quick_redshift_to_distance
from .stellar_hsml_utils import get_particle_hsml
from ..io.gizmo import load_halo
from .. import config

# Houses functions for reducing snapshot data for SKIRT and creating plots from SKIRT outputs


def create_SKIRT_input_files(snap_dir, snap_num, output_dir, importDust=True, file_prefix=None, Rvir_cut=None):
    '''
    This function extracts and reduces the star and gas particle data from the specified simulation snapshot 
    into SKIRT input file. When extracting dust data it only uses the dust-to-metals (D/Z) ratio.

    Parameters
    ----------
    snap_dir : string
        Name of snapshot directory
    snap_num : int
        Snapshot number
    output_dir : string
        Name of directory where SKIRT outputs will be written to
    importDust : boolean
        Sets whether dust information will be extracted from simulation.
    file_prefix : string, optional
        Prefix for star.dat and gas.dat files.
    Rvir_cut : float, optional
        Radial cutoff for considered particles in units of Rvir 

    Returns
    -------
    None
    '''
    # This loads the galactic halo from the snapshot
    halo = load_halo(snap_dir, snap_num, mode='AHF')
    # This orientates the halo so that the galactic disk is face-on
    print("Orientating halo")
    halo.set_orientation()
    if Rvir_cut is not None:
        halo.set_zoom(rout = Rvir_cut)

    if not halo.sp.Flag_DustSpecies and importDust:
        print("WARNING: importDust set to True but this snapshot does not have dust. Set importDust to False.")

    # Load data for star particles (ptype = 4)
    star = halo.loadpart(4)
    # x,y,x coordinates
    coords = star.get_property('position')
    x, y, z = coords[:,0], coords[:,1], coords[:,2]
    # Compute the star softening lengths. This takes some time.
    print("Calculating star particle smoothing lengths...")
    h = get_particle_hsml(x, y, z)
    # mass, metallicity, and age
    m, Z, t = star.get_property('M'), star.get_property('Z_all')[:,0], 1e9*star.get_property('age')

    filename = output_dir+"/"+file_prefix+"star.dat"
    f = open(filename, 'w')
    # Write header for star file
    header =    '# star.dat \n' + \
                '# Column 1: position x (pc)\n' + \
                '# Column 2: position y (pc)\n' + \
                '# Column 3: position z (pc)\n' + \
                '# Column 4: smoothing length (pc)\n' + \
                '# Column 5: mass (Msun)\n' + \
                '# Column 6: metallicity (1)\n' + \
                '# Column 7: age (yr)\n'
    f.write(header)
    # Step through each star particle and write its data
    for i in range(star.npart):
        line = "%.2f %.2f %.2f %.2f %.3e %.3e %.3e\n" %(1e3*x[i],1e3*y[i],1e3*z[i],1e3*h[i],m[i],Z[i],t[i])
        f.write(line)
    f.close()

    print("Star data written to " + filename)

    # Load gas particle data (ptype = 0)
    gas = halo.loadpart(0)
    # x,y,x coordinates
    coords = gas.get_property('position')
    x, y, z = coords[:,0], coords[:,1], coords[:,2]
    # If the snapshots include dust amounts, give those to SKIRT and set D/Z to 1
    # Else just assume a constant D/Z everywhere.
    if importDust:
        # smoothing length, dust mass, and temperature
        h, m, T = gas.get_property('size'), gas.get_property('M_dust'), gas.get_property('temperature')
    else:
        # smoothing length, gas mass, metallicity, and temperature
        h, m, Z, T = gas.get_property('size'), gas.get_property('M'), gas.get_property('Z_all')[:,0], gas.get_property('temperature')

    filename = output_dir+"/"+file_prefix+"gas.dat"
    f = open(filename, 'w')
    # Make header for gas/dust. Needs to be in this specific order
    header =   '# gas.dat \n' + \
               '# Column 1: position x (pc)\n' + \
               '# Column 2: position y (pc)\n' + \
               '# Column 3: position z (pc)\n' + \
               '# Column 4: smoothing length (pc)\n'
    if importDust:
        header += '# Column 5: dust mass (Msun)\n' + \
                  '# Column 6: temperature (K)\n'
    else:
        header += '# Column 5: mass (Msun)\n' + \
                  '# Column 6: metallicity (1)\n' + \
                  '# Column 7: temperature (K)\n'
    f.write(header)

    if importDust:
        for i in range(gas.npart):
            line = "%.2f %.2f %.2f %.3e %.3e %.3e\n" %(1e3*x[i],1e3*y[i],1e3*z[i],1e3*h[i],m[i],T[i])
            f.write(line)
    else:
        for i in range(gas.npart):
            line = "%.2f %.2f %.2f %.3e %.3e %.3e %.3e\n" %(1e3*x[i],1e3*y[i],1e3*z[i],1e3*h[i],m[i],Z[i],T[i])
            f.write(line)
    f.close()

    print("Gas/Dust data written to " + filename)


def get_SKIRT_SED_data(dirc, inst_file, distance=10E6, redshift=0):
    '''
    This function extracts and reduces the star and gas particle data from the specified simulation snapshot 
    into SKIRT input file. When extracting dust data it only uses the dust-to-metals (D/Z) ratio.

    Parameters
    ----------
    dirc : string
        Name of instrument directory
    inst_file : string
        Name of instrument SED file
    distance : double, optional
        Instrument distance (set in SKIRT) in units of pc
    redshfit : double, optional
        Redshift of observation for determining instrument distance. Override distance argument

    Returns
    -------
    sed_data: dict
        Dictionary with wavelength and flux information for each source component
    '''
    # These are the typical columns for the SKIRT simulations I run
    if redshift <=0:
        camera_dist = distance*config.pc_to_m
    else:
        # Need to get luminosity distance
        camera_dist = quick_redshift_to_distance(redshift)*config.pc_to_m
    flux_to_L = 4*np.pi*np.power(camera_dist,2)/config.L_solar # Flux to Solar Luminosity
    
    sed_data={}
    # SKIRT names for these columns
    column_names = ['wavelength','total','transparent', 'direct_primary','scattered_primary', 'direct_secondary','scattered_secondary','transparent_secondary']
    # Our names
    key_names = ['wavelength','total','trasparent stars', 'direct stars','scattered stars', 'direct dust','scatter dust','transparent dust']
    data = np.loadtxt(dirc+inst_file).T
    for col in range(np.shape(data)[0]):
        sed_data[key_names[col]] = data[col,:]
        if key_names[col] != 'wavelength':
            sed_data[key_names[col]]*=flux_to_L

    return sed_data

