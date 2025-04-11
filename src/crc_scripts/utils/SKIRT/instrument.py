from astropy.io import fits
from astropy.visualization import make_lupton_rgb, LogStretch, ManualInterval
from astropy.convolution import convolve_fft,Gaussian2DKernel
from astropy import units as u
import numpy as np


from ..math_utils import quick_cosmological_calc
from ... import config
from ...figure import Figure, Projection



try:   
    import webbpsf
except:
    print("Need to install webbpsf and download its corresponding files to use JWST PSFs. \n If you don't generic Gaussian PSFs will be used. \n Follow instructions here \n https://webbpsf.readthedocs.io/en/latest/index.html \n")


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
        self.angular_distance = (hdul[0].header['DISTANGD'] * (u.Unit(hdul[0].header['DISTUNIT']))/u.Unit('rad')).to('kpc/rad') # kpc
        self.pixel_res_physical = self.angular_distance.to('kpc/rad') * self.pixel_res_angle.to('rad')
        self.num_pixels = hdul[0].header['NAXIS1']
        self.fov_physical = self.num_pixels * self.pixel_res_physical.to('kpc')
        self.fov_angle = self.num_pixels * self.pixel_res_angle.to('arcsec')

        # Determine what filters corresponds to the pivot wavelengths in the FITS file
        # Note you can have custom filters so this will let you know if now known filter matches.
        self.filters = ["N/A"]*len(self.pivot_wavelengths)
        self.unknown_filter = 0
        for i,wavelength in enumerate(self.pivot_wavelengths):
            idx = np.nanargmin(np.abs(wavelength - filter_wavelengths))
            if np.abs(wavelength - filter_wavelengths[idx])/filter_wavelengths[idx] > 0.0001: # Needs to be very small since commonly used filers can have slightly different pivot wavelengths for different telescopes
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
            print("Filter Name \t\t Pivot Wavelength (microns)")
            for i,wavelength in enumerate(self.pivot_wavelengths):
                print(self.filters[i], "\t\t", self.pivot_wavelengths[i])
            if self.unknown_filter:
                print("WARNING: %i filters in the FITS file do not match any known SKIRT filter. Check the SKIRT .ski file to see if they are custom filters."%self.unknown_filter)

        hdul.close()


    def get_filter_image(self, filter, psf=None, brightness_units=None):
        '''
        This function extracts the image data for the specified filter.

        Parameters
        ----------
        filter : string
            Name of filter to extract from FITS file.
        psf : Telescope_PSF, optional
            PSF you want to convolve the image with. Default is None.

        Returns
        -------
        image : ndarray (N,N)
            NxN pixel image for the specified filter.
        '''
        if filter not in self.filters:
            print("Filter %s not found in FITS file. Returning None."%filter)
            return None

        idx = np.where(self.filters == filter)
        image = self.images[idx][0]

        # Determine surface brightness units of images
        image_units = None
        if image.unit.is_equivalent(u.MJy / u.sr):
            image_units = 'frequency'
        elif image.unit.is_equivalent(u.erg / u.cm**2 / u.s / u.micron / u.sr):
            image_units = 'wavelength'
        elif image.unit.is_equivalent(u.erg / u.cm**2 / u.s / u.sr):
            image_units = 'neutral'
        else:
            print("WARNING: Unknown brightness units. Cannot covert units.")
            return None
        

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

        if psf is not None:
            image = convolve_fft(image, psf)


        return image
        
    
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
    

    def make_lupton_RGB_image(self, rgb_filters, psf = None, stretch = 0.5, Q = 8, minimum=0, label=None, output_name=None, **kwargs):
        """
        Generate a log-scaled RGB image using Lupton's RGB algorithm.
        Parameters:
        - rgb_filters (list): List of three filter names to use for the red, green, and blue channels.
        - max_frac (float): Fraction of maximum pixel brightness for max limit of scaling.
        - stretch (int): Stretch factor for the log scaling.
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

        RGB_image = make_lupton_rgb(
            r_frame,g_frame,b_frame,
            stretch=stretch,
            Q=Q,
            minimum=minimum,
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
            The range between the min and max Q parameter.
        stretch_lims : list of floats, optional
            The range between the min and max stretch parameter.
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

        stretchs=np.linspace(stretch_lims[0], stretch_lims[1], bins)
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

                label='Q=%1.1f, stretch=%1.1f'%(Q,stretch)
                img.plot_image(i*bins+j, RGB_image, label = label)
        
        if output_name is not None:
            img.save(output_name)

        return


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
                ):
        
        telescope_res = telescope_res * u.arcsec

        if self.telescope == 'GENERIC':
            self.get_gaussian_psf(telescope_res=telescope_res,
                                  oversample_factor=oversample_factor,)
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
                        oversample_factor: int = 4,):
        
        # Telescope resolution in arcsecond
        telescope_resolution = telescope_res
        # SKIRT instrument can have any resolution. 
        # The relative resolution of the instrument and the actual telescope will determine the Gaussian sigma
        if not self.instrument is None:
            instrument_res = self.instrument.pixel_res_angle.to('arcsec') # in arcsec
        else: instrument_res=telescope_res # If no SKIRT instrument given assume same resolution as telescope
        # calculate the sigma in pixels.
        sigma = telescope_resolution/instrument_res

        psf = Gaussian2DKernel(sigma, factor = oversample_factor)
        self.psf = psf

        if self.verbose:
            print(f"PSF model for filter GENERIC has pixel scale {telescope_res:.4g} arcsec / pixel.\n")


    def get_NIRCam_psf(self,
                       oversample_factor: int = 4,):

        # Only need filter number (i.e. F070W) and not the full name (i.e. JWST_NIRCAM_F070W)
        filter_name = self.filter.split('_')[-1]
        # Name of extension in the FITS file
        extension_name= "DET_SAMP" # Other choices are ["DET_SAMP", "OVERSAMP", "DET_DIST", "OVERDIST"]
        # Calculate npixels for the telescope given the npixels and resolution of the instrument given
        # Else we assume the instrument and telescope have the same resolution
        if not self.instrument is None:
            # Have to run this once to get the pixel scale for the instrument
            nircam = webbpsf.NIRCam()
            nircam.filter = filter_name
            psf_hdulist = nircam.calc_psf()
            telescope_res = psf_hdulist[extension_name].header["PIXELSCL"]
            instrument_npixels = self.instrument.num_pixels
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
        nircam = webbpsf.NIRCam()
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
                    oversample_factor: int = 4,):

        # Only need filter number (i.e. F770W) and not the full name (i.e. JWST_MIRI_F770W)
        filter_name = self.filter.split('_')[-1]

        # Name of extension in the FITS file
        extension_name= "DET_SAMP" # Other choices are ["DET_SAMP", "OVERSAMP", "DET_DIST", "OVERDIST"]
        # Calculate npixels for the telescope given the npixels and resolution of the instrument given
        # Else we assume the instrument and telescope have the same resolution
        if not self.instrument is None:
            # Have to run this once to get the pixel scale for the instrument
            miri = webbpsf.MIRI()
            miri.filter = filter_name
            psf_hdulist = miri.calc_psf()
            telescope_res = psf_hdulist[extension_name].header["PIXELSCL"]
            instrument_npixels = self.instrument.num_pixels
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
        miri = webbpsf.MIRI()
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



