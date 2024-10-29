from ..math_utils import quick_cosmological_calc
from ..stellar_hsml_utils import get_particle_hsml
from ...io.gizmo import load_halo
from ... import config
from ...figure import Figure

from astropy.io import fits
from astropy.convolution import convolve_fft,Gaussian2DKernel
from astropy import units as u
from astropy.nddata import block_reduce
import numpy as np

try:   
    import webbpsf
except:
    print("Need to install webbpsf and download its corresponding files to use SKIRT_utils. Follow instructions here \n https://webbpsf.readthedocs.io/en/latest/index.html \n")

# Houses functions for reducing snapshot data for SKIRT and creating instrument data products from SKIRT outputs


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


def get_SKIRT_image_data(dirc, inst_file, verbose=True):
    '''
    This function extracts SKIRT instrument image data from the given FITS data cube. Usually these mock instruments have MxNxN data cube representing NxN pixels images across M wavelengths.

    Parameters
    ----------
    dirc : string
        Name of instrument directory
    inst_file : string
        Name of instrument FITS file
    verbose : bool, optional
        Print out some FITS files data for checking.

    Returns
    -------
    wavelengths : list
        Wavelengths of each image.
    images : ndarray
        MxNxN pixel images.
    pixel_res_kpc : float
        Resolution of each pixel in kpc.
    pixel_res_arcsec : float
        Resolution of each pixel in arcseconds.
    '''
    hdul = fits.open(dirc+inst_file)
    wavelengths = hdul[1].data['GRID_POINTS']
    pixel_res_arcsec = hdul[0].header['CDELT1']
    pixel_units = hdul[0].header['CUNIT1'] # arcsec usually
    angular_distance = hdul[0].header['DISTANGD']
    distance_units = hdul[0].header['DISTUNIT'] # Mpc ususally
    pixel_res_kpc = angular_distance*1E3*config.rad_per_arcsec*pixel_res_arcsec

    if verbose:
        print("FITS files info")
        print(hdul.info())
        print(hdul[0].header)
        print("Pixel resolution", pixel_res_arcsec, "arsec",pixel_res_kpc, "kpc")
        print("There are %i photometric images at various wavelengths"%len(wavelengths))

    images = hdul[0].data
    hdul.close()

    # Possiblty add option to zoom in on image.
    # pixel_zoom = int(images[0].shape[0]//2*zoom)
    # middle_pixel = images[0].shape[0]//2
    # upper_pixel = int(middle_pixel+pixel_zoom)
    # lower_pixel = int(middle_pixel-pixel_zoom)
    # new_size = int(upper_pixel-lower_pixel)
    # zoomed_images = np.zeros([np.shape(images)[0],new_size,new_size])
    # for i, image in enumerate(images):
    #     zoomed_images[i] = image[lower_pixel:upper_pixel,lower_pixel:upper_pixel]

    return wavelengths,images,pixel_res_kpc,pixel_res_arcsec


def create_RGB_log_image_data(sed_data, dynamic_mag_range=3, max_bright_frac=1, max_brightness=None, allow_oversaturation=False, create_rgb_hist=False, only_full_rgb_pixels=True):
    '''
    This function creates a 3-color band image data array using a log scale. 

    Parameters
    ----------
    sed_data : ndarray (3,N,N)
        3 NxN pixel images for 3-color bands.
    dynamic_mag_range : float, optional
        The range between the min and max pixel brightness considered.
    max_bright_frac : float, optional
        The fraction of the pixel with the maximum brightness to set the max for the brightess range.
    max_brightness : float, optional
        Set the maximum brightness for dynamic range. Overrides max_bright_frac.
    allow_oversaturation : bool, optional
        Allow pixels to be oversaturated to white when they are above the maximum brightness. Otherwise keeps true color following Lupton+04.
    create_rgb_hist : bool, optional
        Output a histogram for the RGB color bands. Usefule when testing dynamic ranges and max brightness.
    only_full_rgb_pixels : bool, optional
        Consider only pixels with nonzero data for all 3 color bands.

    Returns
    -------
    RGB_image: ndarray (N,N,3)
        Array with RGB values for each pixel.
    '''

    image_shape = np.shape(sed_data)[1:]
    R = np.zeros(image_shape); G = np.zeros(image_shape); B = np.zeros(image_shape);

    # If any pixels do not have detections for each rgb image remove them 
    if only_full_rgb_pixels:
        mask = ~np.all(sed_data>0,axis=0) & np.any(sed_data>0,axis=0)
        sed_data[:,mask] = 0

    if create_rgb_hist:
        fig = Figure(1)
        labels = ['r','g','b']
        # Set minimum to be right above 
        x_lim = [np.min(sed_data[sed_data>0]),np.max(sed_data)]
        fig.set_axis(0, 'Brightness', 'CDF',y_lim=[0.001,1],y_label='CDF',y_log=False,x_lim=x_lim,x_label='Brightness $(W/m^2/arcsec^2)$',x_log=True)
        for i,image in enumerate(sed_data):
            data = image.flatten()
            data = data[data>x_lim[0]] # ignore zero pixels
            fig.plot_1Dhistogram(0,data,bin_lims=None, bin_nums=100, bin_log=True, label=labels[i], density=True, color = labels[i],cumulative=True)
        I = ((sed_data[0]+sed_data[1]+sed_data[2])/3).flatten()
        I = I[I>x_lim[0]]
        fig.plot_1Dhistogram(0,I,bin_lims=None, bin_nums=100, bin_log=True, label='(r+b+g)/3', density=True, color = 'black', cumulative=True)
        if max_brightness is None:
            I_max = np.max(I)*max_bright_frac
        else:
            I_max = max_brightness
        fig.plot_shaded_region(0, [np.power(10,np.log10(I_max)-dynamic_mag_range),I_max], 0, 1, alpha=0.2)
        fig.set_all_legends() 
        fig.save('./rgb_cdf.png')


    # Since our synthetic images have no noise need to avoid zeros
    sed_data[sed_data<=0] = np.min(sed_data[sed_data>0])

    # Raw rgb data
    r = sed_data[0]
    g = sed_data[1]
    b = sed_data[2]



    # Follow the Lupton+04 coloring scheme so pixels above max brightness retain their true color and are not oversaturated to white
    if not allow_oversaturation:
        scale_func = np.zeros(image_shape)
        I = (r+g+b)/3
        if max_brightness is None:
            I_max = np.max(I)*max_bright_frac
        else:
            I_max = max_brightness
        I_min = np.power(10,np.log10(I_max)-dynamic_mag_range)

        in_range = (I > I_min) & (I <= I_max)
        scale_func[I>I_max] = 1
        scale_func[in_range] = (np.log10(I[in_range])-np.log10(I_min))/(np.log10(I_max)-np.log10(I_min))

        R = r * scale_func / I
        G = g * scale_func / I
        B = b * scale_func / I 

        max_RGB = np.max([R,G,B],axis=0)

        overstat = max_RGB > 1
        R[overstat] /= max_RGB[overstat]
        G[overstat] /= max_RGB[overstat]
        B[overstat] /= max_RGB[overstat]
        RGB_image = np.transpose(np.array([R,G,B]),(1,2,0))
    
    # Oversaturate pixels to white
    else:
        if max_brightness is None:
            I_max = np.max(np.array([r,g,b]))*max_bright_frac
        else:
            I_max = max_brightness
        I_min = np.power(10,np.log10(I_max)-dynamic_mag_range)
        print('min I',I_min)
        print('max I',I_max)
        RGB = np.array([R,G,B])
        for i,I in enumerate([r,g,b]):
            scale_func = np.zeros(image_shape)
            in_range = (I > I_min) & (I <= I_max)
            scale_func[I>I_max] = 1
            scale_func[in_range] = (np.log10(I[in_range])-np.log10(I_min))/(np.log10(I_max)-np.log10(I_min))
            RGB[i] = scale_func

        RGB_image = np.transpose(RGB,(1,2,0))        


    return RGB_image


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


def convolve_images_w_PSF(images, instrument, filter_names, instrument_res: float = None, downsample_image: bool=True,
    oversample_factor: int = 4, extension_name: str = "DET_SAMP", generic_telescope_res: float = None, ):
    '''
    Convolves the given images with PSFs for the selected telescope and filters.

    Parameters
    ----------
    images : ndarray (M,N,N)
        M NxN pixel images to be convolved.
    instrument : string
        The telescope instrument you want to get filter PSFs. Only JWST MIRI and NIRCam supported right now.
    instrument_res : float
        Resolution of image in arcsec.
    filter_names : list or string
        Either List of M filters or 1 filter to convolve each image over.
    downsample_image : optional, boolean
        Downsample image to a resolution close to the telescope resolution. If you ar using multiple telescopes with different resolutions
        this will only use the lowest resolution. Different color bands with different resolutions are not supported...yet.
    
    Returns
    -------
    convolved_images: ndarray (M,N,N)
        M NxN pixel images that have been convolved over selected filters.
    '''


    NIRCam_filters = ["F115W","F150W","F200W","F277W","F356W","F444W"]
    MIRI_filters = ["F560W","F770W","F1000W","F1130W","F1280W","F1500W","F1800W","F2100W","F2550W"]

    indiv_filter = True if isinstance(filter_names,list) else False

    if instrument not in [
        "JWST",
        "Generic"
    ]:
        raise ValueError("Invalid instrument name %s."%instrument)
    
    if extension_name not in ["DET_SAMP", "OVERSAMP", "DET_DIST", "OVERDIST"]:
        raise ValueError("Invalid extension name %s."%extension_name)
    
    # First get resolution to downsample all images to
    if downsample_image:
        downsample_res = 0
        for i in range(len(images)):
            if instrument == 'Generic': 
                if generic_telescope_res > downsample_res: downsample_res = generic_telescope_res
            else:
                if indiv_filter:
                    filter_name = filter_names[i]
                else:
                    filter_name = filter_names

                if filter_name in NIRCam_filters:
                    NIRCam_res = 0.06291 # arcsec / pixel
                    NIRCam_res = 0.031 # arcsec / pixel
                    if NIRCam_res > downsample_res: downsample_res = NIRCam_res
                elif filter_name in MIRI_filters:
                    MIRI_res = 0.1109 # arcsec / pixel
                    if MIRI_res > downsample_res: downsample_res = MIRI_res
                else:
                    raise ValueError("Invalid filter name %s."%filter_name)
        print(f'Downsampled resolution set by telescope resolution of {downsample_res:.4g} arcsec / pixel\n')

    # Now colvolve each image with its corresponding psf and downsample if needed
    convolved_images = []
    for i in range(len(images)):
        image = images[i]
        if instrument == 'Generic':
            psf = get_gaussian_model(
                telescope_res=generic_telescope_res,
                oversample_factor=oversample_factor,
                instrument_res=instrument_res)
        else:
            if indiv_filter:
                filter_name = filter_names[i]
            else:
                filter_name = filter_names

            if filter_name in NIRCam_filters: psf_func = get_nircam_webbpsf_model
            elif filter_name in MIRI_filters: psf_func = get_miri_webbpsf_model
            else: raise ValueError("Invalid filter name %s."%filter_name)
        
            # Our instrument can have arbitrary resolution. 
            # We need the number of pixels the telescope would actually have over the given FOV
            if downsample_image:
                image, new_resolution = downsample_image_data(image, instrument_res, downsample_res, image_property='surface_brightness')
                print(f"Downsampled image has resolution {new_resolution} arcsec / pixel compared to desired telescope resolution of  {downsample_res:.4g} arcsec / pixel.\n")

            instrument_npixels = len(image)

            psf = psf_func(
                filter_name,
                instrument_npixels=instrument_npixels,
                oversample_factor=oversample_factor,
                extension_name=extension_name,
            )    

        convolved_images += [convolve_fft(image, psf)]

    return np.array(convolved_images)


def get_nircam_webbpsf_model(
    filter_name: str,
    instrument_res: float = None,
    instrument_npixels: int = 101,
    oversample_factor: int = 4,
    extension_name: str = "DET_SAMP",
    outputfilepath: str=None):

    if filter_name not in [
        "F115W",
        "F150W",
        "F200W",
        "F277W",
        "F365W",
        "F444W",
    ]:
        raise ValueError("Invalid filter name %s."%filter_name)
    
    # Calculate npixels for the telescope given the npixels and resolution of the instrument given
    # Else we assume the instruemtn and telescope have the same resolution
    if instrument_res is not None:
        # Have to run this once to get the pixel scale for the instrument
        nircam = webbpsf.NIRCam()
        nircam.filter = filter_name
        psf_hdulist = nircam.calc_psf()
        telescope_res = psf_hdulist[extension_name].header["PIXELSCL"]
        telescope_npixels = int(instrument_npixels * instrument_res / telescope_res)
    else:
        telescope_npixels = instrument_npixels

    # Create NIRCam PSF model
    nircam = webbpsf.NIRCam()
    nircam.filter = filter_name
    psf_hdulist = nircam.calc_psf(
        fov_pixels=telescope_npixels,
        oversample=oversample_factor,
        outfile=outputfilepath,
    )

    psf = psf_hdulist[extension_name].data
    pixel_scale = psf_hdulist[extension_name].header["PIXELSCL"]

    psf /= np.sum(psf)  # normalize PSF

    print(f"PSF model for filter {filter_name} has pixel scale {pixel_scale:.4g} arcsec / pixel.\n")

    return psf


def get_miri_webbpsf_model(
    filter_name: str,
    instrument_res: float = None,
    instrument_npixels: int = 101,
    oversample_factor: int = 4,
    extension_name: str = "DET_SAMP",
    outputfilepath: str=None):

    if filter_name not in [
        "F560W",
        "F770W",
        "F1000W",
        "F1130W",
        "F1280W",
        "F1500W",
        "F1800W",
        "F2100W",
        "F2550W",
    ]:
        raise ValueError("Invalid filter name %s."%filter_name)

    # Calculate npixels for the telescope given the npixels and resolution of the instrument given
    # Else we assume the instruemtn and telescope have the same resolution
    if instrument_res is not None:
        # Have to run this once to get the pixel scale for the instrument
        miri = webbpsf.MIRI()
        miri.filter = filter_name
        psf_hdulist = miri.calc_psf()
        telescope_res = psf_hdulist[extension_name].header["PIXELSCL"]
        telescope_npixels = int(instrument_npixels * instrument_res / telescope_res)
    else:
        telescope_npixels = instrument_npixels

    # Create NIRCam PSF model
    miri = webbpsf.MIRI()
    miri.filter = filter_name
    psf_hdulist = miri.calc_psf(
        fov_pixels=telescope_npixels,
        oversample=oversample_factor,
        outfile=outputfilepath,
    )

    psf = psf_hdulist[extension_name].data
    pixel_scale = psf_hdulist[extension_name].header["PIXELSCL"]

    psf /= np.sum(psf)  # normalize PSF

    print(f"PSF model for filter {filter_name} has pixel scale {pixel_scale:.4g} arcsec / pixel.\n")


    return psf


def get_gaussian_model(
    telescope_res: float = 1,
    oversample_factor: int = 4,
    instrument_res: float = None):


    # Telescope resolution in arcsecond
    telescope_resolution = telescope_res
    # calculate the sigma in pixels.
    if instrument_res is None: instrument_res=telescope_res
    sigma = telescope_resolution/instrument_res

    psf = Gaussian2DKernel(sigma, factor = oversample_factor)

    print(f"PSF model for filter GENERIC has pixel scale {telescope_res:.4g} arcsec / pixel.\n")


    return psf
