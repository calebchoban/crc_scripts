from ..math_utils import quick_cosmological_calc
from ..stellar_hsml_utils import get_particle_hsml
from ...io.gizmo import load_halo
from ... import config

from astropy.nddata import block_reduce
import numpy as np
import importlib.util




def create_SKIRT_particle_files(snap_dir, snap_num, output_dir, import_dust=False, import_gas_hsml=True, use_halo_file=False, file_prefix=None):
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
    import_dust : boolean
        Sets whether dust mass will be extracted from simulation. Otherwise a constant D/Z ratio is assumed by SKIRT.
    import_gas_hsml : boolean
        Sets whether gas smoothing length will be extracted from simulation. This is used by SKIRT when building an octotree grid for the dust medium.
        Set to False if using the Voronoi grid since each particle position is used to make a Voronoi tesselation.
    use_halo_file : string, optional
        Use halo files (i.e. AHF) to initially center halo. Only really needed for high-z or subhalos.
    file_prefix : string, optional
        Prefix for star.dat and gas.dat files.
    '''
    # This loads the galactic halo from the snapshot
    mode='AHF' if use_halo_file else None
    halo = load_halo(snap_dir, snap_num, mode=mode)
    # This orientates the halo so that the galactic disk is face-on
    print("Orientating halo")
    halo.set_orientation()

    if not halo.sp.Flag_DustSpecies and import_dust:
        print("WARNING: import_dust set to True but this snapshot does not have dust. Set import_dust to False.")

    # Load data for star particles (ptype = 4)
    p4 = halo.loadpart(4)
    # x,y,x coordinates
    coords = p4.get_property('position')
    x, y, z = coords[:,0], coords[:,1], coords[:,2]
    # Compute the star softening lengths. This takes some time.
    print("Calculating star particle smoothing lengths...")
    h = get_particle_hsml(x, y, z)

    spam_spec = importlib.util.find_spec("gizmo_analysis")
    has_gizmo_analysis = spam_spec is not None

    if has_gizmo_analysis:
        # mass at formation, metallicity, and age
        m, Z, t = p4.get_property('M_form'), p4.get_property('Z_all')[:,0], 1e9*p4.get_property('age')
    else:
        print("WARNING: gizmo_analysis not installed. This module is needed to determine the zero-age stellar mass which SKIRT expects for star particles.")
        print("This will instead use the current star mass, which can underestimate the mass by <=30%.")
        print("You can pip install gizmo_analysis @ https://git@bitbucket.org/awetzel/gizmo_analysis.git")
        # current mass , metallicity, and age
        m, Z, t = p4.get_property('M'), p4.get_property('Z_all')[:,0], 1e9*p4.get_property('age')
    
    f = open(output_dir+"/"+file_prefix+"star.dat", 'w')
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
    for i in range(p4.npart):
        line = "%.2f %.2f %.2f %.2f %.3e %.3e %.3e\n" %(1e3*x[i],1e3*y[i],1e3*z[i],1e3*h[i],m[i],Z[i],t[i])
        f.write(line)
    f.close()

    print("Star data written to star.dat...")

    # Load gas particle data (ptype = 0)
    p0 = halo.loadpart(0)
    # x,y,x coordinates
    coords = p0.get_property('position')
    x, y, z = coords[:,0], coords[:,1], coords[:,2]
    # If the snapshots include dust amounts, give those to SKIRT and set D/Z to 1
    # Else just assume a constant D/Z everywhere.
    if import_dust:
        # smoothing length, dust mass, and temperature
        h, m, T = p0.get_property('size'), p0.get_property('M_dust'), p0.get_property('temperature')
    else:
        # smoothing length, gas mass, metallicity, and temperature
        h, m, Z, T = p0.get_property('size'), p0.get_property('M'), p0.get_property('Z_all')[:,0], p0.get_property('temperature')

    f = open(output_dir+"/"+file_prefix+"gas.dat", 'w')
    # Make header for gas/dust with data columns matching specified properties
    # Position is always needed, smoothing length depends on if you are using a grid or a voronoi tessellation
    # Mass depends on if you are import a dust mass tracked in the simulation or assuming a dust mass using
    # the gas mass and metallicity and an assumed dust-to-metals ratio such that Mdust = Mgas * Z * D/Z.
    # Temperature is used as a cut off for dust mass. Anything above a set temperature is assumed to have no dust.
    header =   '# gas.dat \n' + \
               '# Column 1: position x (pc)\n' + \
               '# Column 2: position y (pc)\n' + \
               '# Column 3: position z (pc)\n'
    if import_dust and import_gas_hsml:
        header += '# Column 4: smoothing length (pc)\n' + \
                  '# Column 5: dust mass (Msun)\n' + \
                  '# Column 6: temperature (K)\n'
    elif import_dust:
        header += '# Column 4: dust mass (Msun)\n' + \
                  '# Column 5: temperature (K)\n'
    elif import_gas_hsml:
        header += '# Column 4: smoothing length (pc)\n' + \
                  '# Column 5: mass (Msun)\n' + \
                  '# Column 6: metallicity (1)\n' + \
                  '# Column 7: temperature (K)\n'
    else:
        header += '# Column 4: mass (Msun)\n' + \
                  '# Column 5: metallicity (1)\n' + \
                  '# Column 6: temperature (K)\n'   

    f.write(header)

    if import_dust and import_gas_hsml:
        for i in range(p0.npart):
            line = "%.2f %.2f %.2f %.3e %.3e %.3e\n" %(1e3*x[i],1e3*y[i],1e3*z[i],1e3*h[i],m[i],T[i])
            f.write(line)
    elif import_dust:
        for i in range(p0.npart):
            line = "%.2f %.2f %.2f %.3e %.3e\n" %(1e3*x[i],1e3*y[i],1e3*z[i],m[i],T[i])
            f.write(line)       
    elif import_gas_hsml:
        for i in range(p0.npart):
            line = "%.2f %.2f %.2f %.3e %.3e %.3e %.3e\n" %(1e3*x[i],1e3*y[i],1e3*z[i],1e3*h[i],m[i],Z[i],T[i])
            f.write(line)     
    else:
        for i in range(p0.npart):
            line = "%.2f %.2f %.2f %.3e %.3e %.3e\n" %(1e3*x[i],1e3*y[i],1e3*z[i],m[i],Z[i],T[i])
            f.write(line)
    f.close()

    print("Gas/Dust data written to gas.dat...")





"""
############################
WARNING: This is deprecated code that will be deleted. Most functionality is being transferred to the instrument.py module.
############################
"""

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

    if redshift <=0:
        camera_dist = distance*config.pc_to_m
    else:
        # Need to get luminosity distance
        camera_dist = quick_cosmological_calc(redshift, 'luminosity_distance')*config.pc_to_m
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




def downsample_image_data(image, image_res, desired_res, image_property='surface_brightness'):
    '''
    Given a high-resolution instrument image and a desired lower resolution, downsample the image resolution to a resolution closest to the desired resolution given by dividing by whole numbers. 
    If the two resolutions are not perfectly divisible (i.e remainder not 0) then the edge pixels of the instrument image will be trimmed.

    Parameters
    ----------
    image : ndarray (N,N)
        NxN pixel images to be downsampled.
    image_res : float
        Resolution of image in arcsec.
    desired_res: float
        Desired image resolution in arcsec. 
    image_property: optional, str
        Units of image. Only surface_brightness or flux supported. This determines whether the downsampled pixels are meaned or summed respectively.

    Returns
    -------
    downsampled_image: ndarray (M,M)
        MxM pixel image that have been downsampled to the desired resolution.
    new_resolution: float
        New resolution of image
    '''

    
    downsample_factor = int(np.round(desired_res/image_res))
    new_resolution = image_res*downsample_factor
    if downsample_factor < 2:
        print("Desired resolution is either higher than the given image or >0.5 of given image so nothing to downsample.")
        return image, image_res

    if image_property == 'surface_brightness':
        reduce_func = np.mean
    elif image_property == 'flux':
        reduce_func = np.sum
    else:
        raise ValueError("Invalid image property %s."%image_property)

    # image_pixels = np.shape(image)[0]
    # reduced_pixels = int(image_pixels / downsample_factor)
    # excess_pixels = image_pixels%reduced_pixels
    # fov_rescale = 1.-excess_pixels/image_pixels

    # # Trim excess pixels from each side of the image instead of letting block_reduce remove all 
    # # excess from the ends of the image array, which can offset the image center
    # if excess_pixels != 0:
    #     lefttop_trim = int(np.ceil(excess_pixels/2))
    #     rightbottom_trim = int(np.floor(excess_pixels/2))
    #     image = image[lefttop_trim:image_pixels-rightbottom_trim+1,lefttop_trim:image_pixels-rightbottom_trim+1]

    downsampled_image = block_reduce(image, downsample_factor, func = reduce_func) 
    
    return downsampled_image, new_resolution

