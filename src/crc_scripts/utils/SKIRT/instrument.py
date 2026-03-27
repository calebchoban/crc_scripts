from astropy import units as u
from astropy import constants as const
from astropy.io import fits
from astropy.visualization import make_lupton_rgb, make_rgb, LogStretch, LuptonAsinhStretch, AsinhStretch, ManualInterval
from astropy.convolution import convolve_fft,Gaussian2DKernel
from astropy.nddata import block_reduce
from astropy.table import QTable
from scipy import integrate
from scipy.optimize import curve_fit
import numpy as np

from ... import config
from ...figure import Figure, Projection



try:   
    import stpsf
except:
    print("Need to install stpsf and download its corresponding files to use JWST and Roman PSFs. \n If you don't generic Gaussian PSFs will be used. \n Follow instructions here \n https://stpsf.readthedocs.io/en/latest/installation.html#installation \n")


# These are the broadbands and corresponding pivot wavelengths taken from the SKIRT documentation
# https://skirt.ugent.be/skirt9/class_broad_band.html
# Useful for determining which broadbands are in a SKIRT FITS file.
filter_names = np.array([
    "2MASS_2MASS_J", "2MASS_2MASS_H", "2MASS_2MASS_KS", "ALMA_ALMA_10", "ALMA_ALMA_9",
    "ALMA_ALMA_8", "ALMA_ALMA_7", "ALMA_ALMA_6", "ALMA_ALMA_5", "ALMA_ALMA_4",
    "ALMA_ALMA_3", "EUCLID_VIS_VIS", "EUCLID_NISP_Y", "EUCLID_NISP_J", "EUCLID_NISP_H",
    "GALEX_GALEX_FUV", "GALEX_GALEX_NUV", "GENERIC_JOHNSON_U", "GENERIC_JOHNSON_B",
    "GENERIC_JOHNSON_V", "GENERIC_JOHNSON_R", "GENERIC_JOHNSON_I", "GENERIC_JOHNSON_J",
    "GENERIC_JOHNSON_M", "HERSCHEL_PACS_70", "HERSCHEL_PACS_100", "HERSCHEL_PACS_160",
    "HERSCHEL_SPIRE_250", "HERSCHEL_SPIRE_350", "HERSCHEL_SPIRE_500", "IRAS_IRAS_12",
    "IRAS_IRAS_25", "IRAS_IRAS_60", "IRAS_IRAS_100", "JCMT_SCUBA2_450", "JCMT_SCUBA2_850",
    "PLANCK_HFI_857", "PLANCK_HFI_545", "PLANCK_HFI_353", "PLANCK_HFI_217", "PLANCK_HFI_143",
    "PLANCK_HFI_100", "PLANCK_LFI_70", "PLANCK_LFI_44", "PLANCK_LFI_30", "RUBIN_LSST_U",
    "RUBIN_LSST_G", "RUBIN_LSST_R", "RUBIN_LSST_I", "RUBIN_LSST_Z", "RUBIN_LSST_Y",
    "SLOAN_SDSS_U", "SLOAN_SDSS_G", "SLOAN_SDSS_R", "SLOAN_SDSS_I", "SLOAN_SDSS_Z",
    "SPITZER_IRAC_I1", "SPITZER_IRAC_I2", "SPITZER_IRAC_I3", "SPITZER_IRAC_I4",
    "SPITZER_MIPS_24", "SPITZER_MIPS_70", "SPITZER_MIPS_160", "SWIFT_UVOT_UVW2",
    "SWIFT_UVOT_UVM2", "SWIFT_UVOT_UVW1", "SWIFT_UVOT_U", "SWIFT_UVOT_B", "SWIFT_UVOT_V",
    "TNG_OIG_U", "TNG_OIG_B", "TNG_OIG_V", "TNG_OIG_R", "TNG_NICS_J", "TNG_NICS_H",
    "TNG_NICS_K", "UKIRT_UKIDSS_Z", "UKIRT_UKIDSS_Y", "UKIRT_UKIDSS_J", "UKIRT_UKIDSS_H",
    "UKIRT_UKIDSS_K", "WISE_WISE_W1", "WISE_WISE_W2", "WISE_WISE_W3", "WISE_WISE_W4",
    "JWST_NIRCAM_F070W", "JWST_NIRCAM_F090W", "JWST_NIRCAM_F115W", "JWST_NIRCAM_F140M",
    "JWST_NIRCAM_F150W", "JWST_NIRCAM_F162M", "JWST_NIRCAM_F164N", "JWST_NIRCAM_F150W2",
    "JWST_NIRCAM_F182M", "JWST_NIRCAM_F187N", "JWST_NIRCAM_F200W", "JWST_NIRCAM_F210M",
    "JWST_NIRCAM_F212N", "JWST_NIRCAM_F250M", "JWST_NIRCAM_F277W", "JWST_NIRCAM_F300M",
    "JWST_NIRCAM_F322W2", "JWST_NIRCAM_F323N", "JWST_NIRCAM_F335M", "JWST_NIRCAM_F356W",
    "JWST_NIRCAM_F360M", "JWST_NIRCAM_F405N", "JWST_NIRCAM_F410M", "JWST_NIRCAM_F430M",
    "JWST_NIRCAM_F444W", "JWST_NIRCAM_F460M", "JWST_NIRCAM_F466N", "JWST_NIRCAM_F470N",
    "JWST_NIRCAM_F480M", "JWST_MIRI_F560W", "JWST_MIRI_F770W", "JWST_MIRI_F1000W",
    "JWST_MIRI_F1130W", "JWST_MIRI_F1280W", "JWST_MIRI_F1500W", "JWST_MIRI_F1800W",
    "JWST_MIRI_F2100W", "JWST_MIRI_F2550W"
])
# Array of corresponding pivot wavelengths
filter_wavelengths = np.array([
    1.2393, 1.6494, 2.1638, 349.89, 456.2, 689.59, 937.98, 1244.4, 1616, 2100.2, 3043.4,
    0.71032, 1.0808, 1.3644, 1.7696, 0.15351, 0.23008, 0.35236, 0.44146, 0.55223, 0.68967,
    0.87374, 1.2429, 5.0114, 70.77, 100.8, 161.89, 252.55, 354.27, 515.36, 11.4, 23.605,
    60.344, 101.05, 449.3, 853.81, 352.42, 545.55, 839.3, 1367.6, 2130.7, 3001.1, 4303,
    6845.9, 10674, 0.368, 0.47823, 0.62178, 0.75323, 0.86851, 0.97301, 0.35565, 0.47024,
    0.61755, 0.74899, 0.89467, 3.5508, 4.496, 5.7245, 7.8842, 23.759, 71.987, 156.43,
    0.20551, 0.22462, 0.25804, 0.34628, 0.43496, 0.54254, 0.37335, 0.43975, 0.53727,
    0.63917, 1.2758, 1.6265, 2.2016, 0.88263, 1.0314, 1.2501, 1.6354, 2.2058, 3.3897,
    4.6406, 12.568, 22.314,0.7039, 0.9022, 1.154, 1.405, 1.501, 1.627, 1.645, 1.659, 
    1.845, 1.874, 1.989, 2.095, 2.121, 2.503, 2.762, 2.989, 3.232, 3.237, 3.362, 3.568, 
    3.624, 4.052, 4.082, 4.281, 4.404, 4.630, 4.654, 4.708, 4.818, 5.635, 7.639, 9.953, 
    11.31, 12.81, 15.06, 17.98, 20.80, 25.36
]) * u.micron


class SKIRT_Instrument(object):
    '''
    This class is used to extract images from SKIRT instruments. It is not fully implemented yet.
    '''

    def __init__(self, dirc, inst_file, verbose=True):
        self.dirc = dirc
        self.inst_file = inst_file
        self.verbose = verbose

        hdul = fits.open(self.dirc+self.inst_file)
        self.images = hdul[0].data * u.Unit(hdul[0].header['BUNIT'])
        self.pivot_wavelengths = (hdul[1].data['GRID_POINTS'] * u.Unit(hdul[0].header['CUNIT3'])).to('micron')
        self.pixel_res_angle = (hdul[0].header['CDELT1'] * u.Unit(hdul[0].header['CUNIT1'])).to('arcsec') # arcsec
        self.num_pixels = np.asarray([hdul[0].header['NAXIS1'],hdul[0].header['NAXIS2']])
        self.fov_angle = self.num_pixels * self.pixel_res_angle.to('arcsec')
        try:
            self.angular_distance = (hdul[0].header['DISTANGD'] * (u.Unit(hdul[0].header['DISTUNIT']))/u.Unit('rad')).to('kpc/rad') # kpc
            self.pixel_res_physical = self.angular_distance.to('kpc/rad') * self.pixel_res_angle.to('rad')
            self.fov_physical = self.num_pixels * self.pixel_res_physical.to('kpc')
        except KeyError:
            print("Not given angular distance in FITS file. Cannot determine physical pixel resolution and FOV. Set to None.")
            self.angular_distance = None
            self.pixel_res_physical = None
            self.fov_physical = None
        
    
        # Determine what filters corresponds to the pivot wavelengths in the FITS file
        # Note you can have custom filters so this will let you know if now known filter matches.
        self.filters = ["N/A" for i in range(len(self.pivot_wavelengths))]
        self.unknown_filter = 0
        for i,wavelength in enumerate(self.pivot_wavelengths):
            idx = np.nanargmin(np.abs(wavelength - filter_wavelengths))
            if np.abs(wavelength - filter_wavelengths[idx])/filter_wavelengths[idx] > 0.001: # Needs to be small since commonly used filers can have slightly different pivot wavelengths for different telescopes
                self.unknown_filter += 1
            else:
                self.filters[i] = filter_names[idx]

        self.filters = np.array(self.filters)

        if self.verbose:
            print("FITS files info")
            print(hdul.info())
            print(hdul[0].header)
            print("Pixel brightness in units of", self.images.unit)
            print("Pixel resolution", self.pixel_res_angle, self.pixel_res_physical)
            print("Image FOV",self.fov_physical,self.fov_angle)
            print("There are %i photometric images"%len(self.pivot_wavelengths))
            print(f"{'Filter Name':<20} {'Pivot Wavelength':<15}")
            print("-" * 50)
            for i,wavelength in enumerate(self.pivot_wavelengths):
                print(f"{self.filters[i]:<20} {wavelength:<15.5f}")
            if self.unknown_filter:
                print("WARNING: %i filters in the FITS file do not match any known SKIRT filter. Check the SKIRT .ski file to see if they are custom filters."%self.unknown_filter)

        hdul.close()


    def get_filter_image(self, filter, psf=None, brightness_units=None, downsample_resolution=None, downsample_factor=1, sum_downsample=False):
        '''
        This function extracts the image data for the specified filter.

        Parameters
        ----------
        filter : string
            Name of filter to extract from FITS file.
        psf : Telescope_PSF, optional
            PSF you want to convolve the image with. Default is None.
        brightness_units : string, optional
            Surface brightness units wanted (wavelength, frequency, neutral). Default is None which uses units in FITS file.
        downsample_resolution : double, optional
            Resolution in arsec to downsample the image to. Default is None.
        downsample_factor : int, optional
            Integer factor you want to downsample the image by. Default is 1.
        sum_downsample : bool, optional
            Whether to sum the pixel values when downsampling instead of taking the average. Default is False since images are typically in surface brightnesses units which cannot be summed since they are per unit area. This is mainly used when downsampling images with reliability statistics.
        Returns
        -------
        image : ndarray (N,N)
            NxN pixel image for the specified filter.
        '''


        if filter not in self.filters:
            print("Filter %s not found in FITS file. Returning None."%filter)
            return None

        idx = np.where(self.filters == filter)
        image = np.copy(self.images[idx][0])

        # Determine surface brightness units of images
        image_units = None
        if image.unit.is_equivalent(u.MJy / u.sr):
            image_units = 'frequency'
        elif image.unit.is_equivalent(u.erg / u.cm**2 / u.s / u.micron / u.sr):
            image_units = 'wavelength'
        elif image.unit.is_equivalent(u.erg / u.cm**2 / u.s / u.sr):
            image_units = 'neutral'
        else:
            print("WARNING: Unknown brightness units. Output image will have no units.")
            image_units = None
        

        if self.verbose:
            print("%s image brightness is per %s with units %s"%(filter,image_units, image.unit))

        # Convert to different units if requested
        if brightness_units is not None and image_units != brightness_units:
            wavelength = self.get_filter_wavelength(filter)
            if image_units != 'neutral':
                if brightness_units == 'wavelength':    
                    # Convert to F_lambda units
                    image = image.to(u.erg / u.cm**2 / u.s / u.angstrom / u.sr,
                    equivalencies=u.spectral_density(wavelength))
                elif brightness_units == 'frequency':
                    # Convert to F_nu units
                    image = image.to(u.erg / u.cm**2 / u.s / u.Hz/ u.sr,
                    equivalencies=u.spectral_density(wavelength))
                elif brightness_units == 'neutral':
                    # Convert to neutral (lambda F_lamda = nu F_nu) units
                    image = image.to(u.erg / u.cm**2 / u.s / u.micron / u.sr,
                    equivalencies=u.spectral_density(wavelength))*wavelength.to('micron')
            else:
                # Convert to F_lambda units
                if brightness_units == 'wavelength':
                    image = image / wavelength.to('micron')
                # Convert to F_nu units
                if brightness_units == 'frequency':
                    image = image / wavelength.to('Hz', equivalencies=u.spectral())

            if self.verbose:
                print("Image brightness units converted to %s."%image.unit)


        # Convolve with the instrument PSF. This is the effects of the telescope mirrors, 
        # reflectors, etc. before light hits the detectors. You always want to apply the PSF
        # first and then downsample to the true resolution of your instrument.
        if psf is not None:
            image = convolve_fft(image, psf)
        
        # Down sample instrument resolution to a lower resolution
        if downsample_resolution is not None or downsample_factor > 1:
            if downsample_resolution is not None:
                desired_resolution = downsample_resolution * u.arcsec
                downsample_factor = int(np.round(desired_resolution/self.pixel_res_angle))
            new_resolution = self.pixel_res_angle*downsample_factor
            if downsample_factor < 2:
                print("Desired resolution is either higher than the given image or >0.5 of given image so nothing to downsample.")
            else:
                print("Downsampling image from %s to %s arcsec."%(self.pixel_res_angle.to('arcsec'), new_resolution.to('arcsec')))

            if sum_downsample:
                reduce_func = np.sum # reducing resolution means new larger pixels have the sum of the smaller pixels
            else:
                reduce_func = np.mean # reducing resolution means new larger pixels have the average brigtness of the smaller pixels
            image = block_reduce(image, downsample_factor, func = reduce_func) 
        else:
            if psf is not None:
                print("WARNING: You convolved your image with a PSF but didnt downsample.\n If your image is already at " \
                "the resolution of your telescope this will make your image have lower resolution than a real image!\n " \
                "For a realistic image you should convolve the PSF with a higher resolution image and then downsample. ")


        return image
    

    def cut_image_to_fov(self, image, fov):
        """
        Parameters
        ----------
        image : ndarray (N,N)
            NxN pixel image you want to cut to a specified fov.
        fov : list
            Set the x,y FOV of the image in kpc as a single value of specific x and y values.
        Returns
        -------
        cut_image : ndarray (M,L)
            MxL image cut out of the original image to the desired FOV

        """
        
        fov = np.asarray(fov)*u.kiloparsec
        if np.any(fov>self.fov_physical):
            print("Given FOV %f kpc is larger than instrument FOV %f kpc!"%(fov, self.fov_physical))
            return
        else:
            print(fov,self.fov_physical)
            cut_pix = (np.floor(self.num_pixels * fov / self.fov_physical).value/2).astype(np.int32)
            middle_pix = np.floor(self.num_pixels/2).astype(np.int32)
            cut_image = image[middle_pix[0]-cut_pix[0]:middle_pix[0]+cut_pix[0],middle_pix[0]-cut_pix[1]:middle_pix[1]+cut_pix[1]]
            return cut_image
        
    
    def get_filter_wavelength(self, filter):
        '''
        This function extracts the pivot wavelength for the specified filter.

        Parameters
        ----------
        filter : string
            Name of filter to extract from FITS file.

        Returns
        -------
        wavelength : float
            Pivot wavelength for the specified filter.
        '''
        if filter not in self.filters:
            print("Filter %s not found in FITS file. Returning None."%filter)
            return None

        idx = np.where(self.filters == filter)
        return self.pivot_wavelengths[idx]
    

    def make_log_lupton_RGB_image(self, rgb_filters, psf=None, max_percentile = 100, max_frac = 1, min_percentile = 0, min_frac = 0, stretch = 1000, label=None, output_name=None, **kwargs):
        """
        Generate a log-scaled RGB image using Lupton's RGB algorithm.
        Parameters:
        - rgb_filters (list): List of three filter names to use for the red, green, and blue channels.
        - max_percentile (int or list): Maximum percentile value(s) to use for each channel. If an integer is provided, it will be used for all channels. Overrides max_frac.
        - max_frac (float): Fraction of maximum pixel brightness for max limit of scaling. Overridden by max_percentile.
        - stretch (int): Stretch factor for the log scaling.
        - verbose (bool): Flag indicating whether to print verbose output.
        - label (str): Label for the image.
        - output_name (str): Output file name for the image.
        Returns:
        - None
        """

        if psf is not None:
            if isinstance(psf, list):
                r_frame = self.get_filter_image(rgb_filters[0], psf=psf[0], **kwargs)
                g_frame = self.get_filter_image(rgb_filters[1], psf=psf[1], **kwargs)
                b_frame = self.get_filter_image(rgb_filters[2], psf=psf[2], **kwargs)
        else:
            r_frame = self.get_filter_image(rgb_filters[0], psf=psf, **kwargs)
            g_frame = self.get_filter_image(rgb_filters[1], psf=psf, **kwargs)
            b_frame = self.get_filter_image(rgb_filters[2], psf=psf, **kwargs)

        if max_percentile is not None:
            if not isinstance(max_percentile, list):
                max_percentile = [max_percentile]*3
            maximums = [np.percentile(r_frame[r_frame>0].value,max_percentile[0]),np.percentile(g_frame[g_frame>0].value,max_percentile[1]),np.percentile(b_frame[b_frame>0].value,max_percentile[2])]
        else:
            if not isinstance(max_frac, list):
                max_frac = [max_frac]*3
            maximums = [np.max(r_frame.value)*max_frac[0],np.max(g_frame.value)*max_frac[1],np.max(b_frame.value)*max_frac[2]]
        if min_percentile is not None:
            if not isinstance(min_percentile, list):
                min_percentile = [min_percentile]*3
            minimums = [np.percentile(r_frame[r_frame>0].value,min_percentile[0]),np.percentile(g_frame[g_frame>0].value,min_percentile[1]),np.percentile(b_frame[b_frame>0].value,min_percentile[2])]
        else:
            if not isinstance(min_frac, list):
                min_frac = [min_frac]*3
            minimums = [np.max(r_frame.value)*min_frac[0],np.max(g_frame.value)*min_frac[1],np.max(b_frame.value)*min_frac[2]]

        
        intervals = [ManualInterval(vmin=minimums[0], vmax=maximums[0]),ManualInterval(vmin=minimums[1], vmax=maximums[1]),ManualInterval(vmin=minimums[2], vmax=maximums[2])]

        if self.verbose:
            print("RGB maximum values", maximums)
            print("RGB minimum values", minimums)

        RGB_image = make_lupton_rgb(r_frame,g_frame,b_frame, interval=intervals,
                   stretch_object=LogStretch(a=stretch))
        


        img = Projection(1)
        img.set_image_axis(0)
        img.plot_image(0, RGB_image, fov_kpc=self.fov_physical.value, fov_arcsec=self.fov_angle.value, label = label)
        
        if output_name is not None:
            img.save(output_name)
    

    def make_lupton_RGB_image(self, rgb_filters, psf = None, stretch = 0.5, Q = 8, min_percentile=0, max_percentile=100, label=None, output_name=None, **kwargs):
        """
        Generate an RGB image using Lupton's RGB algorithm which keeps the true color of each pixel.
        Parameters:
        - rgb_filters (list): List of three filter names to use for the red, green, and blue channels.
        - max_frac (float): Fraction of maximum pixel brightness for max limit of scaling.
        - stretch (float): Stretch factor for arcsinh. Sets where you want the color to be linear. Typically want this to be the brightness point at which the cumulative distribution function of pixel brightness rapidly increases.
        - Q (float): Q parameter for arcsinh. Sets how bright the brightest pixels are. Smaller makes the brightest pixels brighter.
        - label (str): Label for the image.
        - output_name (str): Output file name for the image.
        Returns:
        - None
        """
        if psf is not None:
            if isinstance(psf, list):
                r_frame = self.get_filter_image(rgb_filters[0], psf=psf[0], **kwargs)
                g_frame = self.get_filter_image(rgb_filters[1], psf=psf[1], **kwargs)
                b_frame = self.get_filter_image(rgb_filters[2], psf=psf[2], **kwargs)
        else:
            r_frame = self.get_filter_image(rgb_filters[0], psf=psf, **kwargs)
            g_frame = self.get_filter_image(rgb_filters[1], psf=psf, **kwargs)
            b_frame = self.get_filter_image(rgb_filters[2], psf=psf, **kwargs)

        if not isinstance(max_percentile, list):
            max_percentile = [max_percentile]*3
        maximums = [np.percentile(r_frame[r_frame>0].value,max_percentile[0]),np.percentile(g_frame[g_frame>0].value,max_percentile[1]),np.percentile(b_frame[b_frame>0].value,max_percentile[2])]
        if not isinstance(min_percentile, list):
            min_percentile = [min_percentile]*3
        minimums = [np.percentile(r_frame[r_frame>0].value,min_percentile[0]),np.percentile(g_frame[g_frame>0].value,min_percentile[1]),np.percentile(b_frame[b_frame>0].value,min_percentile[2])]

        intervals = [ManualInterval(vmin=minimums[0], vmax=maximums[0]),ManualInterval(vmin=minimums[1], vmax=maximums[1]),ManualInterval(vmin=minimums[2], vmax=maximums[2])]

        if self.verbose:
            print("RGB maximum values", maximums)
            print("RGB minimum values", minimums)

        RGB_image = make_lupton_rgb(
            r_frame,g_frame,b_frame,
            stretch=stretch,
            Q=Q,
            interval=intervals,
        )

        img = Projection(1)
        img.set_image_axis(0)

        img.plot_image(0, RGB_image, fov_kpc=self.fov_physical.value, fov_arcsec=self.fov_angle.value, label = label)

        if output_name is not None:
            img.save(output_name)


    def make_lupton_RGB_image_grid(self, rgb_filters, psf=None, Q_lims=[1,10], stretch_lims=[0.1,1], bins=4, output_name=None, **kwargs):
        '''
        This function creates a set 3-color band across the Q and stretch limits given. Useful for determining what Q and stretch to use. 

        Parameters
        ----------
        rgb_filters : list of strings
            List of 3 filters to use for the RGB image.
        Q_lims : list of floats, optional
            The range between the min and max Q parameter binned in linear space.
        stretch_lims : list of floats, optional
            The range between the min and max stretch parameter binned in log space.
        bins : int, optional
            Number of bins between Q and stretch limits.
        output_name : string, optional
            Name of output file for the image. If None, the image will not be saved.

        Returns
        -------
        None
        '''

        if psf is not None:
            if isinstance(psf, list):
                r_frame = self.get_filter_image(rgb_filters[0], psf=psf[0], **kwargs)
                g_frame = self.get_filter_image(rgb_filters[1], psf=psf[1], **kwargs)
                b_frame = self.get_filter_image(rgb_filters[2], psf=psf[2], **kwargs)
        else:
            r_frame = self.get_filter_image(rgb_filters[0], psf=psf, **kwargs)
            g_frame = self.get_filter_image(rgb_filters[1], psf=psf, **kwargs)
            b_frame = self.get_filter_image(rgb_filters[2], psf=psf, **kwargs)

        stretchs=np.logspace(np.log10(stretch_lims[0]), np.log10(stretch_lims[1]), bins)
        Qs=np.linspace(Q_lims[0], Q_lims[1], bins)

        N=bins*bins
        nrows = bins
        img = Projection(N, nrows=nrows)
        for i in range(N):
            img.set_image_axis(i)

        for i,Q in enumerate(Qs):
            for j,stretch in enumerate(stretchs):
                # Make the RGB image
                RGB_image = make_lupton_rgb(
                    r_frame,g_frame,b_frame,
                    stretch=stretch,
                    Q=Q,
                )

                label='Q=%1.1f, stretch=%1.1e'%(Q,stretch)
                img.plot_image(i*bins+j, RGB_image, label = label)
        
        if output_name is not None:
            img.save(output_name)

        return


    def make_RGB_image(self, rgb_filters, psf = None, stretch = 0.5, Q = 8, min_percentile=0, max_percentile=100, label=None, output_name=None, trim_monocolor=True, scale_bar='both', **kwargs):
        """
        Generate an RGB image. This will not keep the true color of pixels above the maximum specified brightness, but can be used to emphasize certain colors unlike the Lupton scheme.
        Parameters:
        - rgb_filters (list): List of three filter names to use for the red, green, and blue channels.
        - max_frac (float): Fraction of maximum pixel brightness for max limit of scaling.
        - stretch (float): Stretch factor for arcsinh. Sets where you want the color to be linear. Typically want this to be the brightness point at which the cumulative distribution function of pixel brightness rapidly increases.
        - Q (float): Q parameter for arcsinh. Sets how bright the brightest pixels are. Smaller makes the brightest pixels brighter.
        - label (str): Label for the image.
        - output_name (str): Output file name for the image.
        - trim_monocolor (bool): If True, will trim the RGB image to remove pixels that are monochromatic (i.e. only one non-zero color). This is useful for high-resolution images where some pixels may not recieve photons for all bands .
        Returns:
        - scale_bar (str): Set to 'physical', 'angle', or 'both' for a kpc, arcminute, or both scale bars
        - None
        """
        if psf is not None:
            if isinstance(psf, list):
                r_frame = self.get_filter_image(rgb_filters[0], psf=psf[0], **kwargs)
                g_frame = self.get_filter_image(rgb_filters[1], psf=psf[1], **kwargs)
                b_frame = self.get_filter_image(rgb_filters[2], psf=psf[2], **kwargs)
        else:
            r_frame = self.get_filter_image(rgb_filters[0], psf=psf, **kwargs)
            g_frame = self.get_filter_image(rgb_filters[1], psf=psf, **kwargs)
            b_frame = self.get_filter_image(rgb_filters[2], psf=psf, **kwargs)

        # Create a combined mask for monochromatic pixels
        monochromatic_mask = (
            ((r_frame > 0) & (g_frame == 0) & (b_frame == 0)) |  # Only r_frame is nonzero
            ((r_frame == 0) & (g_frame > 0) & (b_frame == 0)) |  # Only g_frame is nonzero
            ((r_frame == 0) & (g_frame == 0) & (b_frame > 0))    # Only b_frame is nonzero
        )

        # Apply the mask to trim monochromatic pixels if trim_monocolor is True
        if trim_monocolor:
            r_frame[monochromatic_mask] = 0
            g_frame[monochromatic_mask] = 0
            b_frame[monochromatic_mask] = 0

        if not isinstance(max_percentile, list):
            max_percentile = [max_percentile]*3
        maximums = [np.percentile(r_frame[r_frame>0].value,max_percentile[0]),np.percentile(g_frame[g_frame>0].value,max_percentile[1]),np.percentile(b_frame[b_frame>0].value,max_percentile[2])]
        if not isinstance(min_percentile, list):
            min_percentile = [min_percentile]*3
        minimums = [np.percentile(r_frame[r_frame>0].value,min_percentile[0]),np.percentile(g_frame[g_frame>0].value,min_percentile[1]),np.percentile(b_frame[b_frame>0].value,min_percentile[2])]

        intervals = [ManualInterval(vmin=minimums[0], vmax=maximums[0]),ManualInterval(vmin=minimums[1], vmax=maximums[1]),ManualInterval(vmin=minimums[2], vmax=maximums[2])]

        if self.verbose:
            print("RGB maximum values", maximums)
            print("RGB minimum values", minimums)

        stretch=LuptonAsinhStretch(stretch=stretch, Q=Q)

        RGB_image = make_rgb(
            r_frame,g_frame,b_frame,
            stretch=stretch,
            interval=intervals,
        )

        img = Projection(1)
        img.set_image_axis(0)

        fov_kpc=None; fov_arcsec=None
        if scale_bar == 'both': 
            fov_kpc=self.fov_physical.value; fov_arcsec=self.fov_angle.value;
        elif scale_bar == 'physical': fov_kpc=self.fov_physical.value
        elif scale_bar == 'angle': fov_arcsec=self.fov_angle.value
        img.plot_image(0, RGB_image, fov_kpc=fov_kpc, fov_arcsec=fov_arcsec, label = label)

        if output_name is not None:
            img.save(output_name)


    def make_rgb_hist(self, rgb_filters, psf = None, output_name = None, **kwargs):
        '''
        This function creates a histogram of the pixel brightness for each frame in an RGB image for the specified filters.

        Parameters
        ----------
        rgb_filters : list
            List of three filter names to use for the red, green, and blue channels.   
        output_name : string, optional
            Name of output file for the histogram. If None, the histogram will not be saved.
        **kwargs : keyword arguments
            Additional keyword arguments to pass to the get_filter_image method.
        Returns
        -------
        None
        '''

        if psf is not None:
            if isinstance(psf, list):
                r_frame = self.get_filter_image(rgb_filters[0], psf=psf[0], **kwargs)
                g_frame = self.get_filter_image(rgb_filters[1], psf=psf[1], **kwargs)
                b_frame = self.get_filter_image(rgb_filters[2], psf=psf[2], **kwargs)
        else:
            r_frame = self.get_filter_image(rgb_filters[0], psf=psf, **kwargs)
            g_frame = self.get_filter_image(rgb_filters[1], psf=psf, **kwargs)
            b_frame = self.get_filter_image(rgb_filters[2], psf=psf, **kwargs)
        rgb_image = np.array([r_frame, g_frame, b_frame])
        
        fig = Figure(1)
        labels = ['r','g','b']
        
        # Set minimum to be right above 
        x_lim = [np.min(rgb_image[rgb_image>0]),np.max(rgb_image)]
        fig.set_axis(0, 'Brightness', 'CDF',y_lim=[0.001,1],y_label='CDF',y_log=False,x_lim=x_lim,x_label=f'Brightness ({r_frame.unit:latex})',x_log=True)
        for i,image in enumerate(rgb_image):
            data = image.flatten()
            data = data[data>x_lim[0]] # ignore zero pixels
            fig.plot_1Dhistogram(0,data,bin_lims=None, bin_nums=100, bin_log=True, label=labels[i], density=True, color = labels[i],cumulative=True)
        fig.set_all_legends() 
        if output_name is not None:
            fig.save(output_name)







class Telescope_PSF(object):
    '''
    This class is used to create a telescope PSF to be convlved with given SKIRT instrument. It is not fully implemented yet.
    '''

    def __init__(self,
                 instrument: SKIRT_Instrument,
                 filter: str,  
                 verbose: bool = True):
    
        self.instrument = instrument
        self.filter = filter
        self.verbose = verbose

        self.telescope = 'GENERIC'
        if 'NIRCAM' in filter:
            self.telescope = 'NIRCAM'
        elif 'MIRI' in filter:
            self.telescope = 'MIRI'
        else:
            if self.verbose:
                print("WARNING: No telescope found for filter %s. Using GENERIC telescope."%filter)

        
    def get_psf(self,   
                telescope_res: float = 0.1, # in arcsec
                oversample_factor: int = 4, # oversample factor for the PSF
                use_instrument_res: bool = True, # Use the instrument resolution to determine the PSF. If False will assume instrument and telescope have the same resolution
                ):
        
        telescope_res = telescope_res * u.arcsec

        if self.telescope == 'GENERIC':
            self.get_gaussian_psf(telescope_res=telescope_res,
                                  oversample_factor=oversample_factor,
                                  use_instrument_res=use_instrument_res)
        elif self.telescope == 'NIRCAM':
            self.get_NIRCam_psf(oversample_factor=oversample_factor)
        elif self.telescope == 'MIRI':
            self.get_MIRI_psf(oversample_factor=oversample_factor)
        else:
            print("Something went wrong. Telescope %s not supported."%self.telescope)
            return None
        
        return self.psf


    def get_gaussian_psf(self,
                        telescope_res: float = 0.1,
                        oversample_factor: int = 4,
                        use_instrument_res: bool = True,):
        
        # Telescope resolution in arcsecond
        telescope_resolution = telescope_res
        # SKIRT instrument can have any resolution. 
        # The relative resolution of the instrument and the actual telescope will determine the Gaussian sigma
        if self.instrument is not None and use_instrument_res:
            instrument_res = self.instrument.pixel_res_angle.to('arcsec') # in arcsec
        else: instrument_res=telescope_res # If no SKIRT instrument given assume same resolution as telescope
        # calculate the sigma in pixels.
        sigma = telescope_resolution/instrument_res

        psf = Gaussian2DKernel(sigma, factor = oversample_factor)
        self.psf = psf

        if self.verbose:
            print(f"PSF model for filter GENERIC has pixel scale {telescope_res:.4g} arcsec / pixel.\n")


    def get_NIRCam_psf(self,
                       oversample_factor: int = 4,
                       use_instrument_res: bool = True,):

        # Only need filter number (i.e. F070W) and not the full name (i.e. JWST_NIRCAM_F070W)
        filter_name = self.filter.split('_')[-1]
        # Name of extension in the FITS file
        extension_name= "DET_SAMP" # Other choices are ["DET_SAMP", "OVERSAMP", "DET_DIST", "OVERDIST"]
        # Calculate npixels for the telescope given the npixels and resolution of the instrument given
        # Else we assume the instrument and telescope have the same resolution
        if self.instrument is not None and use_instrument_res:
            # Have to run this once to get the pixel scale for the instrument
            nircam = stpsf.NIRCam()
            nircam.filter = filter_name
            psf_hdulist = nircam.calc_psf()
            telescope_res = psf_hdulist[extension_name].header["PIXELSCL"] * u.arcsec
            instrument_npixels = self.instrument.num_pixels[0]
            instrument_res = self.instrument.pixel_res_angle.to('arcsec')
            telescope_npixels = int(instrument_npixels * instrument_res / telescope_res)
            if self.verbose:
                print(f"JWST NIRCam filter {filter_name} has resolution of {telescope_res} arcsec.\n")
                print(f"SKIRT instrument has resolution of {instrument_res} arcsec.\n")
        else:
            # If no SKIRT instrument given assume same resolution as telescope
            telescope_npixels = instrument_npixels

        # Create NIRCam PSF model

        if self.verbose:
            print(f"Creating PSF model for JWST NIRCam filter {filter_name} with {telescope_npixels} pixels and {oversample_factor} oversampling.\n")
        nircam = stpsf.NIRCam()
        nircam.filter = filter_name
        psf_hdulist = nircam.calc_psf(
            fov_pixels=telescope_npixels,
            oversample=oversample_factor,
            outfile=None,
        )

        psf = psf_hdulist[extension_name].data
        pixel_scale = psf_hdulist[extension_name].header["PIXELSCL"]

        psf /= np.sum(psf)  # normalize PSF

        if self.verbose:
            print(f"PSF model for filter {filter_name} has pixel scale {pixel_scale:.4g} arcsec / pixel.\n")
        
        self.psf = psf


    def get_MIRI_psf(self,
                    oversample_factor: int = 4,
                    use_instrument_res: bool = True,):

        # Only need filter number (i.e. F770W) and not the full name (i.e. JWST_MIRI_F770W)
        filter_name = self.filter.split('_')[-1]

        # Name of extension in the FITS file
        extension_name= "DET_SAMP" # Other choices are ["DET_SAMP", "OVERSAMP", "DET_DIST", "OVERDIST"]
        # Calculate npixels for the telescope given the npixels and resolution of the instrument given
        # Else we assume the instrument and telescope have the same resolution
        if self.instrument is not None and use_instrument_res:
            # Have to run this once to get the pixel scale for the instrument
            miri = stpsf.MIRI()
            miri.filter = filter_name
            psf_hdulist = miri.calc_psf()
            telescope_res = psf_hdulist[extension_name].header["PIXELSCL"] * u.arcsec
            instrument_npixels = self.instrument.num_pixels[0]
            instrument_res = self.instrument.pixel_res_angle.to('arcsec')
            telescope_npixels = int(instrument_npixels * instrument_res / telescope_res)
            if self.verbose:
                print(f"JWST MIRI filter {filter_name} has resolution of {telescope_res} arcsec.\n")
                print(f"SKIRT instrument has resolution of {instrument_res} arcsec.\n")
        else:
            # If no SKIRT instrument given assume same resolution as telescope
            telescope_npixels = instrument_npixels

        # Create NIRCam PSF model

        if self.verbose:
            print(f"Creating PSF model for JWST MIRI filter {filter_name} with {telescope_npixels} pixels and {oversample_factor} oversampling.\n")
        miri = stpsf.MIRI()
        miri.filter = filter_name
        psf_hdulist = miri.calc_psf(
            fov_pixels=telescope_npixels,
            oversample=oversample_factor,
            outfile=None,
        )

        psf = psf_hdulist[extension_name].data
        pixel_scale = psf_hdulist[extension_name].header["PIXELSCL"]

        psf /= np.sum(psf)  # normalize PSF

        if self.verbose:
            print(f"PSF model for MIRI filter {filter_name} has pixel scale {pixel_scale:.4g} arcsec / pixel.\n")
        
        self.psf = psf



class SKIRT_SED:
    """
    A class to read and parse SKIRT SED (Spectral Energy Distribution) files.
    
    This class can handle SED files with the following format:
    - First line contains distance information: "# SED at inclination X deg, azimuth Y deg, distance Z Unit"
    - Subsequent comment lines describe columns: "# column N: description; variable_name (unit)"
    - Followed by numerical data in columns
    
    Example usage:
    -------------
    # Create SED reader object
    sed = SKIRT_SED('path/to/sed_file.txt')
    
    # Read the data 
    sed.read_data()
    
    # Access metadata
    print(f"Distance: {sed.distance}")
    print(f"Columns: {sed.columns}")
    
    # Access wavelength and flux data
    wavelength = sed.wavelength  # First column with units
    total_flux = sed.total_flux  # Second column with units
    
    # Access specific columns by index or name
    transparent_flux = sed.get_column(2)  # Third column (0-indexed)
    direct_primary = sed.get_column_by_name('direct primary flux')
    
    # Print summary
    print(sed)
    """
    
    def __init__(self, file_path):
        """
        Initialize the SKIRT_SED with the file path.

        Parameters:
        file_path (str): Path to the data file.
        """
        self.file_path = file_path
        self.data = None
        self.distance = None  # To store the distance as an astropy Quantity
        self.columns = []
        self.column_units = []

    def read_data(self):
        """
        Reads the metadata and numerical data from the file, and assigns units to the columns.
        """
        try:
            with open(self.file_path, "r") as file:
                for line in file:
                    if line.startswith("#"):
                        self._extract_metadata(line)

            # Read the numerical data
            data = np.loadtxt(self.file_path)

            # Create an astropy Table and assign units
            self.data = QTable(data, names=self.columns, units=self.column_units)

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.file_path}")

    def _extract_metadata(self, line):
        """
        Extracts metadata (distance, column names, and units) from a comment line.

        Parameters:
        line (str): The comment line containing metadata.
        """
        if "distance" in line.lower() or "redshift" in line.lower():
            if "distance" in line.lower():
                # Extract the distance with units from format: "distance 10 Mpc" or "luminosity distance 10 Mpc"
                parts = line.split("distance")
                if len(parts) > 1:
                    distance_part = parts[1].strip()
                    # Remove any trailing characters like commas
                    if "," in distance_part:
                        distance_part = distance_part.split(",")[0]
                    dist_parts = distance_part.split()
                    if len(dist_parts) >= 2:
                        value = float(dist_parts[0])
                        unit = dist_parts[1]
                        self.distance = value * u.Unit(unit)
            if "redshift" in line.lower():
                # Extract redshift and convert to distance using astropy cosmology
                parts = line.split("redshift")
                if len(parts) > 1:
                    redshift_part = parts[1].strip()
                    if "," in redshift_part:
                        redshift_part = redshift_part.split(",")[0]
                    self.redshift = float(redshift_part)
            else:
                self.redshift = 0 # Default to zero if no redshift information is provided
        elif "column" in line.lower():
            # Extract column names and their units from format: "column 1: wavelength; lambda (micron)"
            parts = line.split(":")
            if len(parts) >= 2:
                column_info = ":".join(parts[1:]).strip()  # Rejoin in case there are multiple colons
                if ";" in column_info:
                    description, variable_unit = column_info.split(";", 1)
                    description = description.strip()
                    variable_unit = variable_unit.strip()
                    
                    # Extract unit from format "variable_name (unit)"
                    if "(" in variable_unit and ")" in variable_unit:
                        start_paren = variable_unit.rfind("(")
                        end_paren = variable_unit.rfind(")")
                        unit_str = variable_unit[start_paren+1:end_paren]
                        variable_name = variable_unit[:start_paren].strip()
                    else:
                        # Fallback if no parentheses found
                        unit_str = "dimensionless"
                        variable_name = variable_unit
                    
                    self.columns.append(description)
                    try:
                        self.column_units.append(u.Unit(unit_str))
                    except ValueError:
                        # Handle unknown units by treating as dimensionless
                        print(f"Warning: Unknown unit '{unit_str}', treating as dimensionless")
                        self.column_units.append(u.dimensionless_unscaled)
    
    @property
    def wavelength(self):
        """Access the wavelength column."""
        if self.data is None:
            raise ValueError("No data loaded. Call read_data() first.")
        return self.data[self.columns[0]]  # First column is wavelength
    
    @property
    def rest_wavelength(self):
        """Access the rest wavelength column, if redshift information is available."""
        if self.data is None:
            raise ValueError("No data loaded. Call read_data() first.")
        return self.wavelength / (1 + self.redshift)
    
    @property
    def total_flux(self):
        """Access the total flux column."""
        if self.data is None:
            raise ValueError("No data loaded. Call read_data() first.")
        return self.data[self.columns[1]]  # Second column is total flux

    @property
    def transparent_flux(self):
        """Access the transparent flux column."""
        if self.data is None:
            raise ValueError("No data loaded. Call read_data() first.")
        return self.data[self.columns[2]]  # Third column is transparent flux

    def get_column(self, column_index, spectral_density_units='frequency', convert_to_luminosity=False, restframe=False):
        """
        Get a specific column by index.
        
        Parameters:
        column_index (int): Index of the column to retrieve (0-based)
        spectral_density_units (str): What spectral density units to convert to if the column is a flux column. Options are 'frequency','wavelength', or 'neutral'. Only applicable for flux columns.
        convert_to_luminosity (bool): Whether to convert the column from flux to luminosity units using the distance. Only applicable for flux columns.
        restframe (bool): Whether to return the column in the rest frame. Only applicable for wavelength columns.
        
        Returns:
        astropy.table.Column: The requested column with units
        """
        if self.data is None:
            raise ValueError("No data loaded. Call read_data() first.")
        if column_index >= len(self.columns):
            raise IndexError(f"Column index {column_index} out of range. Available columns: 0-{len(self.columns)-1}")
        
        column_data = self.data[self.columns[column_index]]
        if 'flux' in self.columns[column_index].lower():
            column_data = self._convert_spectral_density_units(column_data, self.wavelength, convert_to_luminosity=convert_to_luminosity, spectral_density_units=spectral_density_units)
        if restframe and 'wavelength' in self.columns[column_index].lower():
            column_data = column_data / (1 + self.redshift)

        return column_data


    def get_column_by_name(self, column_name, spectral_density_units='frequency', convert_to_luminosity=False, restframe=False):
        """
        Get a column by its description name.
        
        Parameters:
        column_name (str): Name/description of the column
        spectral_density_units (str): What spectral density units to convert to if the column is a flux column. Options are 'frequency','wavelength', or 'neutral'. Only applicable for flux columns.
        convert_to_luminosity (bool): Whether to convert the column from flux to luminosity units using the distance. Only applicable for flux columns.
        restframe (bool): Whether to return the column in the rest frame. Only applicable for wavelength columns.

        Returns:
        astropy.table.Column: The requested column with units
        """
        if self.data is None:
            raise ValueError("No data loaded. Call read_data() first.")
        for column in self.columns:
                if column_name.lower() in column.lower():
                    column_name = column  # Use the full column name if a match is found
                    break
        if column_name not in self.columns:
            raise KeyError(f"Column '{column_name}' not found. Available columns: {self.columns}")
        
        column_data = self.data[column_name]
        if 'flux' in column_name.lower():
            column_data=self._convert_spectral_density_units(column_data, self.wavelength, convert_to_luminosity=convert_to_luminosity, spectral_density_units=spectral_density_units)
        if restframe and 'wavelength' in column_name.lower():
            column_data = column_data / (1 + self.redshift)
        return column_data


    def _convert_spectral_density_units(self, column_data, wavelengths, convert_to_luminosity=False, spectral_density_units='frequency'):
        """
        Convert the units of a column to the specified spectral density units.
        
        Parameters:
        column_data (astropy.table.Column): The column to convert
        wavelengths (astropy.table.Column): The wavelength column to use for the conversion
        convert_to_luminosity (bool): Whether to convert the column from flux to luminosity units using the distance. Only applicable for flux columns.
        spectral_density_units (str): What spectral density units to convert to if the column is a flux column. Options are 'frequency','wavelength', or 'neutral'. Only applicable for flux columns.

        Returns:
        astropy.table.Column: The converted column with units
        """
        if convert_to_luminosity:
            if self.distance is None:
                raise ValueError("Distance information not found. Cannot convert to luminosity.")
            
            flux_to_luminosity_factor = 4 * np.pi * self.distance**2
            neutral_factor = 1
            if spectral_density_units == 'frequency':
                lum_units = u.L_sun / u.Hz
                flux_to_luminosity_factor *= 1/(1 + self.redshift) 
            elif spectral_density_units == 'wavelength':
                lum_units = u.L_sun / u.micron
                flux_to_luminosity_factor *= (1 + self.redshift) 
            elif spectral_density_units == 'neutral':
                lum_units = u.L_sun / u.micron
                neutral_factor *= wavelengths.to('micron')
            column_data = (column_data * flux_to_luminosity_factor).to(lum_units,equivalencies=u.spectral_density(wavelengths))*neutral_factor
        else:
            unit_conversion_factor = 1
            if spectral_density_units == 'frequency':
                flux_units = u.Jy
            elif spectral_density_units == 'wavelength':
                flux_units = u.erg / u.cm**2 / u.s / u.micron
            elif spectral_density_units == 'neutral':
                flux_units = u.erg / u.cm**2 / u.s / u.micron
                unit_conversion_factor *= wavelengths.to('micron')
            column_data = (column_data).to(flux_units,equivalencies=u.spectral_density(wavelengths))*unit_conversion_factor

        return column_data
    
    def __repr__(self):
        """String representation of the SED object."""
        if self.data is None:
            return f"SKIRT_SED(file_path='{self.file_path}', data_loaded=False)"
        
        info = f"SKIRT_SED(file_path='{self.file_path}'\n"
        info += f"  Distance: {self.distance}\n"
        info += f"  {len(self.data)} data points\n"
        info += f"  Columns: {len(self.columns)}\n"
        for i, (col, unit) in enumerate(zip(self.columns, self.column_units)):
            info += f"    {i+1}. {col} ({unit})\n"
        info += ")"
        return info

    def calculate_MUV(self, UV_wavelength=0.15*u.micron, no_dust=False):
        """
        Calculate the absolute ultraviolet magnitude (MUV) at a specific wavelength.

        Parameters:
        UV_wavelength (astropy.units.Quantity): The wavelength at which to calculate the UV magnitude (e.g., 0.15*u.micron).
        no_dust (bool): If True, will use the transparent flux instead of the total flux to calculate MUV, effectively giving the intrinsic UV magnitude without dust attenuation.

        Returns:
        astropy.units.Quantity: The ultraviolet magnitude at the specified wavelength.
        """
        # Ensure data is loaded
        if self.data is None:
            raise ValueError("SED data not loaded. Call sed.read_data() first.")

        # Get the L_nu at the specified UV wavelength
        wavelength = self.rest_wavelength
        if no_dust:
            total_lum_nu = self.get_column_by_name('transparent flux', 
                                                  spectral_density_units='frequency', 
                                                  convert_to_luminosity=True)
        else:
            total_lum_nu = self.get_column_by_name('total flux', 
                                                  spectral_density_units='frequency', 
                                                  convert_to_luminosity=True)
        mUV_0 = -32.36 # Used for AB magnitude system. Calculated as -2.5*log10(3631 Jy in Lsol/Hz at 10 pc)
        L_nu = np.interp(UV_wavelength.to(wavelength.unit).value, wavelength.value, total_lum_nu.value) * total_lum_nu.unit
        MUV = -2.5 * np.log10(L_nu.to(u.L_sun / u.Hz, equivalencies=u.spectral_density(wavelength)).value) + mUV_0

        return MUV


    def calculate_LIR(self, wavelength_range=(8*u.micron, 1000*u.micron)):
        """
        Calculate the total infrared luminosity (LIR) by integrating the SED over a specified wavelength range.

        Parameters:
        wavelength_range (tuple): A tuple specifying the lower and upper bounds of the wavelength range for integration, with astropy units (e.g., (8*u.micron, 1000*u.micron)).

        Returns:
        astropy.units.Quantity: The total infrared luminosity in units of solar luminosities (L_sun).
        """
        # Ensure data is loaded
        if self.data is None:
            raise ValueError("SED data not loaded. Call sed.read_data() first.")
        
        # Get wavelength and total flux columns

        wavelength = self.rest_wavelength
        total_flux = self.get_column_by_name('total flux', 
                                             spectral_density_units='wavelength', 
                                             convert_to_luminosity=True)  # Convert to L_lambda units
        
        # Create a mask for the specified wavelength range
        mask = (wavelength >= wavelength_range[0]) & (wavelength <= wavelength_range[1])
        
        # Integrate the flux over the specified wavelength range using trapezoidal rule
        LIR = integrate.trapezoid(total_flux[mask], x=wavelength[mask])
        
        return LIR.to(u.L_sun)
    
    def calculate_LUV(self, UV_wavelength=0.16*u.micron):
        """
        Calculate the ultraviolet luminosity (LUV) at a specific wavelength.

        Parameters:
        UV_wavelength (astropy.units.Quantity): The wavelength at which to calculate the UV luminosity (e.g., 0.16*u.micron).

        Returns:
        astropy.units.Quantity: The ultraviolet luminosity at the specified wavelength in units of solar luminosities (L_sun).
        """
        # Ensure data is loaded
        if self.data is None:
            raise ValueError("SED data not loaded. Call sed.read_data() first.")
        
        # Get wavelength and total flux columns
        wavelength = self.rest_wavelength
        total_flux = self.get_column_by_name('total flux', 
                                             spectral_density_units='wavelength', 
                                             convert_to_luminosity=True)  # Convert to L_lambda units
        
        # Interpolate the total flux at the specified UV wavelength
        LUV = UV_wavelength.to(wavelength.unit) * (np.interp(UV_wavelength.to(wavelength.unit).value, wavelength.value, total_flux.value) * total_flux.unit)
        
        return LUV.to(u.L_sun)


    def calculate_attenuation(self):
        """
        Calculate the attenuation (A_lambda) across all wavelengths by comparing the total flux to the transparent flux.

        Returns:
        astropy.table.Column: The attenuation A_lambda in magnitudes for each wavelength.
        """
        # Ensure data is loaded
        if self.data is None:
            raise ValueError("SED data not loaded. Call sed.read_data() first.")
        
        # Get wavelength, total flux, and transparent flux columns
        total_flux = self.get_column_by_name('total flux', 
                                             spectral_density_units='wavelength', 
                                             convert_to_luminosity=False)  # Keep in flux units
        transparent_flux = self.get_column_by_name('transparent flux', 
                                                  spectral_density_units='wavelength', 
                                                  convert_to_luminosity=False)  # Keep in flux units
        
        # Calculate A_lambda using the formula: A_lambda = 2.5 * log10(transparent_flux / total_flux)
        A_lambda = 2.5 * np.log10(transparent_flux / total_flux)
        
        return A_lambda

    def calculate_beta_UV(self, column_name='total'):
        """
        Calculate the UV spectral slope (beta) with a simple linear fit to the SED.

        Parameters:
        UV_wavelength_range (tuple): A tuple specifying the lower and upper bounds of the UV wavelength range for fitting, with astropy units (e.g., (0.13*u.micron, 0.3*u.micron)).

        Returns:
        float: The calculated UV spectral slope beta.
        """
        # Ensure data is loaded
        if self.data is None:
            raise ValueError("SED data not loaded. Call sed.read_data() first.")
        
        # Get wavelength and total flux columns
        wavelength = self.rest_wavelength
        total_flux = self.get_column_by_name(column_name, 
                                             spectral_density_units='wavelength', 
                                             convert_to_luminosity=False)  # Keep in flux units
        
        wavelength_1230A = 0.123*u.micron
        wavelength_3200A = 0.32*u.micron
        flux_1230A = np.interp(wavelength_1230A.to(wavelength.unit).value, wavelength.value, total_flux.value) * total_flux.unit
        flux_3200A = np.interp(wavelength_3200A.to(wavelength.unit).value, wavelength.value, total_flux.value) * total_flux.unit

        beta_UV = np.log10(flux_1230A/flux_3200A) / np.log10(wavelength_1230A/wavelength_3200A)

        return beta_UV



    def calculate_Td_peak(self):
        """
        Estimate the dust temperature from the peak wavelength of the far-infrared portion of the SED.
        
        Returns:
        astropy.units.Quantity: The estimated dust temperature in Kelvin.
        """
        # Ensure data is loaded
        if self.data is None:
            raise ValueError("SED data not loaded. Call sed.read_data() first.")
        
        # Get wavelength and total flux columns
        wavelength = self.rest_wavelength
        total_flux = self.get_column_by_name('total', 
                                             spectral_density_units='frequency', 
                                             convert_to_luminosity=False)  # Keep in flux units
        
        # Select the far-infrared portion of the SED (e.g., > 20 microns)
        FIR_mask = wavelength > 20*u.micron
        wavelength_FIR = wavelength[FIR_mask]
        flux_FIR = total_flux[FIR_mask]

        # Estimate T_dust from the peak wavelength of the FIR SED using Wien's displacement law
        peak_index = np.argmax(flux_FIR)
        peak_wavelength = wavelength_FIR[peak_index]
        T_dust_estimated = const.b_wien / peak_wavelength.to(u.m)

        return T_dust_estimated
