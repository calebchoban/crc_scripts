import os
import h5py
import numpy as np
from . import math_utils
from ..io.galaxy import Halo

def check_snap_exist(sdir, snum):
    """
    Check if the snapshot with the given number exists in the given directory and returns the number of files
    it comprises if it exists.

    Parameters
    ----------
    sdir : string
        Directory of snapshot
    snum : int
        Number of snapshot

    Returns
    -------
    nsnap : int
        Number of files that make up snapshot or 0 if no snapshot exists

    """
    
    # single file case
    snapfile = sdir + "/snapshot_%03d.hdf5" %snum
    if (os.path.isfile(snapfile)): return 1
    
    # multiple files
    snapfile = os.path.normpath(sdir + "/snapdir_%03d/snapshot_%03d.0.hdf5" %(snum,snum))
    print(snapfile)
    if (os.path.isfile(snapfile)):
        f = h5py.File(snapfile, 'r')
        nsnap = f['Header'].attrs['NumFilesPerSnapshot']
    else:
        print("Snapshot",sdir + "/snapshot_%03d.hdf5" %snum,"doesn't exist.")
        return 0
    
    for i in np.arange(1,nsnap,1):
        snapfile = sdir + "/snapdir_%03d/snapshot_%03d.%d.hdf5" %(snum,snum,i)
        if (not os.path.isfile(snapfile)):
            print("Snapshot is not complete.")
            return 0
    
    return nsnap


def get_snap_file_name(sdir, snum, nsnap, i):
    """
    Get name of file for snapshot. If snapshot is multiple files it will return the name of the specified subfile.
    Parameters
    ----------
    sdir : string
        Directory of snapshot
    snum : int
        Number of snapshot
    nsnap : int
        Number of subfiles making up snapshot
    i : int
        Number of subfile

    Returns
    -------
    snapfile : string
        Full name of snapshot in specified directory.
        
    """
    if (nsnap==1):
        snapfile = sdir + "/snapshot_%03d.hdf5" %snum
    else:
        snapfile = sdir + "/snapdir_%03d/snapshot_%03d.%d.hdf5" %(snum,snum,i)

    return snapfile


    # get star formation history using all stars in Rvir
def get_SFH(sp, dt=0.01, cum=0, rout=1.0, kpc=0):
    """ 
    Returns the archeological star formation history from the given Snapshot/Halo object.

    Parameters
    ----------
    sp : Snapshot/Halo
        Snapshot/Halo object you want the SFH for.
    dt : double, optional
        Time steps for SFH data in Gyr
    cum : boolean, optional
        Give cumulative star formation history (i.e. stellar mass)
    rout : double, optional
        If sp is a Halo object, this sets the maximum radius from the halo center for considered star particles. 
        This is a fraction of the virial radius if kpc=False, else its the physical radius in kpc if kpc=True.
    kpc: boolean, optional
        If sp is a Halo object, set to True of you want rout to be a physical radius in units of kpc.

    Returns
    -------
    time : ndarray
        Time data for SFH.
    sfr :
        Star formation rate data for SFH.     
    """

    part = sp.loadpart(4)
    if (part.k==-1): 
        print("Particle data in snapshot is not loaded. Need to load it first.")
        return None, None
    
    if isinstance(sp, Halo):
        p, sft, m = part.get_property('position'), part.get_property('sft'), part.get_property('M_form')
        r = np.sqrt((p[:,0])**2+(p[:,1])**2+(p[:,2])**2)
        rmax = rout*sp.rvir if kpc==0 else rout
        time, sfr = math_utils.SFH(sft[r<rmax], m[r<rmax], sp, dt=dt, cum=cum)
    else:
        sft, m = part.get_property('sft'), part.get_property('M_form')
        time, sfr = math_utils.SFH(sft, m, sp, dt=dt, cum=cum)

    return time, sfr


# Calculate the stellar scale radius
def calc_stellar_scale_r(sp, guess=[1E4,1E2,0.5,3], bounds=(0, [1E6,1E6,5,10]), radius_max=10, output_fit=False,
                        foutname='stellar_bulge+disk_fit.png', bulge_profile='de_vauc', no_exp=False):
    """
    Estimate stellar scale radius by fitting a Sersic or De Vauc profile with out without an added exponential

    Parameters
    ----------
    sp: Snapshot/Halo
        Snapshot/Halo you want to determine the stellar scale radius for.
    guess : array, optional
        Initial guess for central density of sersic profile, central density of exponential disk, 
        sersic scale length, disk scale length, and sersic index.
    bounds : tuple, optional
        Bounds for above values.
    radius_max : double, optional
        The maximum radius (in kpc) for fitting Sersic profile. 
    output_fit : boolean, optional
        Creates a plot of the fit and the data.
    foutname : string, optional
        Name of output plot file.
    bulge_profile : string, optional
        Set to either a 'sersic' or de_vauc' bulge profile
    no_exp : boolean, optional
        Adds an exponential component for the disk


    Returns
    -------
    disk_l : 
        Stellar scale radius.
    """

    stars = sp.part[4]
    stars.load()
    stars.orientate(sp.center_position,sp.center_velocity,sp.principal_axes_vectors)
    star_mass = stars.get_property('M')
    r_bins = np.linspace(0,radius_max,int(np.floor(radius_max/.1)))
    r_vals = np.array([(r_bins[i+1]+r_bins[i])/2. for i in range(len(r_bins)-1)])
    rmag = np.sqrt(np.sum(np.power(stars.p[:,:2],2),axis=1))
    sigma_star = np.zeros(len(r_vals))
    for i in range(len(r_vals)):
        rmin = r_bins[i]; rmax = r_bins[i+1]
        mass = np.sum(star_mass[(rmag>rmin) & (rmag<=rmax)])
        area = np.pi*(rmax**2 - rmin**2)
        sigma_star[i] = mass/(area*1E6) # M_sol/pc^2


    # Fit a sersic/bulge+exponential/disk profile to the disk stellar surface density
    fit_params,_ = math_utils.fit_bulge_and_disk(r_vals, sigma_star, guess=guess, bounds=bounds, bulge_profile=bulge_profile, no_exp=no_exp)
    if not no_exp:
        coeff1 = fit_params[0]; coeff2 = fit_params[1]; sersic_l=fit_params[2]; disk_l=fit_params[3];
        if bulge_profile == 'sersic':
            sersic_index = fit_params[4]
        else:
            sersic_index = 4
    else:
        coeff1 = fit_params[0]; sersic_l = fit_params[1];sersic_index = fit_params[2];
        coeff2 = 0; disk_l=0

    print("Results for bulge+disk fit to disk galaxy...\n \
            Sersic Coefficient = %e M_solar/pc^2 \n \
            Disk Coefficient = %e M_solar/pc^2 \n \
            Sersic scale length = %e kpc \n \
            Disk scale length = %e kpc \n \
            Sersic index = %e "%(coeff1, coeff2, sersic_l,disk_l, sersic_index))
    if output_fit:
        plt.figure()
        plt.scatter(r_vals, sigma_star,c='xkcd:black')
        x_vals = np.linspace(0,radius_max,100)
        plt.plot(x_vals, coeff1*np.exp(-np.power(x_vals/sersic_l,1./sersic_index)), label='Sersic Profile')
        plt.plot(x_vals, coeff2*np.exp(-x_vals/disk_l), label='Exponential Disk')
        plt.plot(x_vals, coeff1*np.exp(-np.power(x_vals/sersic_l,1./sersic_index))+coeff2*np.exp(-x_vals/disk_l), label='Total')
        plt.ylim([np.min(sigma_star),np.max(sigma_star)])
        plt.yscale('log')
        plt.ylabel(r'$\Sigma_{\rm star} \; (M_{\odot}/{\rm pc}^2$')
        plt.xlabel("Radius (kpc)")
        plt.legend()
        plt.savefig(foutname)
        plt.close()

    return disk_l