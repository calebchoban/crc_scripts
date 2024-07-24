import numpy as np
import matplotlib.pyplot as plt

from ..utils import math_utils
from ..utils import coordinate_utils

# This is a class that manages a galaxy/halo in a given
# snapshot, for either cosmological or isolated simulations.

class Halo(object):
    """Halo object that loads and stores data for a given galactic halo in a given snapshot."""

    def __init__(self, sp, id=0):
        """
        Construct a Halo instance.

        Parameters
        ----------
        sp : Snapshot
            Snapshot object representing snapshot from which galactic halo is taken.
        id: int, optional
            ID of the galactic halo you want taken from AHF files. Default is the most massive halo. 
            If not AHF files exists this won't do anything.

        Returns
        -------
        Halo
            Halo instance created from given Snapshot object and for a given galactic halo.
        """

        # the halo must be taken from a snapshot
        self.sp = sp
        self.k = -1 if sp.k==-1 else 0 # no snapshot, no halo
        self.id = id
        self.cosmological = sp.cosmological

        # these are linked to its parent snapshot
        self.gas = sp.gas
        self.DM = sp.DM
        self.disk = sp.disk
        self.bulge = sp.bulge
        self.star = sp.star
        self.BH = sp.BH
        self.part = sp.part

        # Set later if you want to zoom in
        self.zoom = False
        self.calc_center = False

        self.center_position = None
        self.center_velocity = None
        self.principal_axes_vectors = None
        self.principal_axes_ratios = None

        return


    # Set zoom-in region if you don't want all data in rvir
    def set_zoom(self, rout=1.0, kpc=False):
        """
        Set the radius of zoom-in region around the halo center. This will determine the extent of the region you want to get particle data for. Default is the virial radius.

        Parameters
        ----------
        rout : double, optional
            The radius of the zoom in region. This is a fraction of the virial radius if kpc=False, else its the physical radius in kpc if kpc=True.
        kpc: boolean, optional
            Set to True of you want rout to be a physical radius in units of kpc.

        Returns
        -------
        None
        """

        # How far out do you want to load data, default is rvir
        self.rout = rout
        self.outkpc = kpc
        self.zoom = True

        return

    def set_orientation(self, ptype=4, mass_radius_max = 100, velocity_radius_max=15, radius_max=10, age_limits=[0,1]):
        """
        Set the orientation (center position velocity and priciple axes) of the halo. The default values work well for stable galaxies,
        but may need to be tweaked for chaotic events like mergers or at high-z.

        Parameters
        ----------
        ptype : int, optional
            The particle type you want to use to determine the orientation. For stable galaxies, stars (ptype=4) are best since they lie 
            well within the galaxy.
        mass_radius_max : double, optional
            The maximum radius (in kpc) from the halo center you want to consider particles for determining center position.
        velocity_radius_max : double, optional
            The maximum radius (in kpc) from the halo center you want to consider particle for determining center velocity.          
        velocity_radius_max : double, optional
            The maximum radius (in kpc) from the halo center you want to consider particle for determining center velocity.             
        radius_max : double, optional
            The maximum radius (in kpc) from the halo center you want to consider particle for determining principle vectors. 
        age_limits : (2,), array
            If using star particles, the age limits [Gyr] for what stars to consider when determining principle vectors. Young stars are best.

        Returns
        -------
        None
        """

        self.assign_center(ptype, mass_radius_max, velocity_radius_max)
        self.assign_principal_axes(ptype, radius_max, age_limits)

        return


    def load(self, mode='AHF'):
        """
        Loads halo data based on either AHF halo file data or center of gas particles in snapshot.

        Parameters
        ----------
        mode : string, optional
            Set to 'AHF' to load AHF halo file information if available. Else, the center of gas particles 
            and snapshot volume is used to define the extent of the halo.

        Returns
        -------
        None
        """

        if (self.k!=0): return # already loaded

        sp = self.sp
 
        # cosmological, use AHF
        if mode=='AHF' and self.cosmological:
            AHF = sp.loadAHF()
            # catalog exists
            if (AHF.k==1):
                
                self.k = 1
                self.time = sp.time
                self.redshift = sp.redshift
                self.catalog = 'AHF'
                self.id = AHF.ID[self.id]
                self.host = AHF.hostHalo[self.id]
                self.npart = AHF.npart[self.id]
                self.ngas = AHF.n_gas[self.id]
                self.nstar = AHF.n_star[self.id]
                self.mvir = AHF.Mvir[self.id]
                self.mgas = AHF.M_gas[self.id]
                self.mstar = AHF.M_star[self.id]
                self.xc = AHF.Xc[self.id]
                self.yc = AHF.Yc[self.id]
                self.zc = AHF.Zc[self.id]
                self.rvir = AHF.Rvir[self.id]
                self.rmax = AHF.Rmax[self.id]
                self.vmax = AHF.Vmax[self.id]
                self.fhi = AHF.fMhires[self.id]
                self.Lhat = AHF.Lhat[:,self.id]
            # no catalog so just chose center of snapshot box
            else:
                print("AHF option chosen for halo but AHF catalog does not exist!")
                print("Defaulting to center of snapshot box.")
                self.k = 0 # so you can re-load it
                self.redshift = sp.redshift
                self.xc = sp.boxsize/2.
                self.yc = sp.boxsize/2.
                self.zc = sp.boxsize/2.
                self.rvir = sp.boxsize*2.
                self.Lhat = np.array([0,0,1.])
        # Default for non-cosmological and no AHF to center of star mass
        else:
            print("Centering galaxy based on average position of gas")
            self.k = 1
            self.time = sp.time
            self.redshift = sp.redshift
            # Get center from average of gas particles
            part = self.sp.loadpart(0)
            self.xc = np.average(part.get_property('position')[:,0],weights=part.get_property('mass'))
            self.yc = np.average(part.get_property('position')[:,1],weights=part.get_property('mass'))
            self.zc = np.average(part.get_property('position')[:,2],weights=part.get_property('mass'))
            self.rvir = sp.boxsize*2.
            self.Lhat = np.array([0,0,1.])

        self.center_position = [self.xc,self.yc,self.zc]
    
        return


    def get_rmax(self):
        """ Returns the radius of the halo """ 
        return self.rvir


    def get_half_mass_radius(self, ptype=4, within_radius=None, rvir_frac=1.0, geometry='spherical'):
        """
        Determines the half mass radius for a given particle type within the halo.

        Parameters
        ----------
        ptype : int, optional
            Particle type you want the half mass radius for. Default is stars.
        within_radius : double, optional
            Maximum physical radius [kpc] you want to considered particles when calculating the half mass radius. 
            Use this or rvir_frac to set the radius. This takes precedence over rvir_frac.
        rvir_frac : double, optional
            Maximum radius as a fraction of the virial radius you want to considered particles when calculating the half mass radius. 
            Use this or within_radius to set the radius.
        geometry : string, optional
            The geometry you want to use given the maximum radius. Supports 'spherical','cylindrical', and 'scale_height'

        Returns
        -------
        half_mass_radius: double
            The half mass radius for the given particle type.

        """

        within_radius = self.rvir*rvir_frac if within_radius is None else within_radius

        part = self.loadpart(ptype)
        coords = part.get_property('coords')
        masses = part.get_property('M')

        edges = np.linspace(0, within_radius, 5000, endpoint=True)

        if geometry in ['cylindrical', 'scale_height']:
            radii = np.sum(coords[:, :2] ** 2, axis=1) ** 0.5
        elif geometry == 'spherical':
            radii = np.sum(coords ** 2, axis=1) ** 0.5

        within_mask = radii <= within_radius

        ## let's co-opt this method to calculate a scale height as well
        if geometry == 'scale_height':
            ## take the z-component
            radii = np.abs(coords[:, -1])
            edges = np.linspace(0, 10 * within_radius, 5000, endpoint=True)

        h, edges = np.histogram(
            radii[within_mask],
            bins=edges,
            weights=masses[within_mask])
        edges = edges[1:]
        h /= 1.0 * np.sum(h)
        cdf = np.cumsum(h)

        # Find closest edge to the middle of the cdf
        argmin = np.argmin((cdf - 0.5) ** 2)
        half_mass_radius = edges[argmin]
        print("Ptype %i half mass radius: %e kpc"%(ptype, half_mass_radius))

        return half_mass_radius


    # load all particles in the halo/galaxy centered on halo center
    def loadpart(self, ptype):
        """
        Load particle data within the Halo object. This data will be orientated to be centered on halo center.

        Parameters
        ----------
        ptype : int, optional
            Particle type you want to load.

        Returns
        -------
        None
            
        """

        part = self.part[ptype]
        # If the particles have previously been loaded and orientated we are done here
        if not part.k or not part.orientated:
            part.load()
            part.orientate(self.center_position,self.center_velocity,self.principal_axes_vectors)
            if self.zoom:
                rmax = self.rout*self.rvir if not self.outkpc else self.rout
            else:
                rmax = self.rvir
            in_halo = np.sum(np.power(part.get_property('position'),2),axis=1) <= np.power(rmax,2.)
            if np.all(in_halo==False):
                print("WARNING: No particle of ptype %i in the zoom in region when loading particle data."%ptype)
            part.mask(in_halo)

        return part


    # calculate center position and velocity
    def assign_center(self, ptype=4, mass_radius_max = None, velocity_radius_max=15):
        """
        Assign the center position and velocity of the halo using the given particle information.

        Parameters
        ----------
        ptype : int, optional
            The particle type you want to use to determine the center. For stable galaxies, stars (ptype=4) are best since they lie 
            well within the galaxy.
        mass_radius_max : double, optional
            The maximum radius (in kpc) from the halo center you want to consider particles for determining center position.
        velocity_radius_max : double, optional
            The maximum radius (in kpc) from the halo center you want to consider particle for determining center velocity.          

        Returns
        -------
        None
            
        """

        if self.calc_center: return
        if mass_radius_max is None:
            mass_radius_max=self.rvir
        print('assigning center of galaxy:')
        part = self.part[ptype]
        part.load()
        if len(part.get_property('mass')) < 0:
            print("""There are no particles of ptype %i to use for assigning center of galaxy. You should use another
                  type of particle such as gas or dark matter."""%ptype)
            return

        # calculate center position
        self.center_position = coordinate_utils.get_center_position_zoom(part.get_property('position'), part.get_property('mass'), self.sp.boxsize,
            center_position=np.array([self.xc,self.yc,self.zc]), distance_max=mass_radius_max)
        self.xc = self.center_position[0]
        self.yc = self.center_position[1]
        self.zc = self.center_position[2]

        print('  center position [kpc] = {:.3f}, {:.3f}, {:.3f}'.format(
            self.center_position[0], self.center_position[1], self.center_position[2]))

        # calculate center velocity
        self.center_velocity = coordinate_utils.get_center_velocity(
            part.get_property('velocity'), part.get_property('mass'), part.get_property('position'), self.center_position, velocity_radius_max, self.sp.boxsize)
        self.vx = self.center_velocity[0]
        self.vy = self.center_velocity[1]
        self.vz = self.center_velocity[2]

        print('  center velocity [km/s] = {:.1f}, {:.1f}, {:.1f}'.format(
            self.center_velocity[0], self.center_velocity[1], self.center_velocity[2]))

        self.calc_center = True
        return


    def assign_principal_axes(self, ptype=4, radius_max=10, age_limits=[0,1]):
        """
        Assign principal axes (rotation vectors defined by moment of inertia tensor) to galaxy.

        Parameters
        ----------
        ptype : int, optional
            The particle type you want to use to determine the orientation. For stable galaxies, stars (ptype=4) are best since they lie 
            well within the galaxy.         
        radius_max : double, optional
            The maximum radius (in kpc) from the halo center you want to consider particle for determining principle vectors. 
        age_limits : (2,), array
            If using star particles, the age limits [Gyr] for what stars to consider when determining principle vectors. Young stars are best.

        Returns
        -------
        None
        """
        
        part = self.part[ptype] # particle types to use to compute MOI tensor and principal axes
        part.load()
        if len(part.get_property('mass')) < 0:
            print("""There are no particles of ptype %i to use for assigning principle axes of galaxy. You should use another
                  type of particle such as gas or dark matter."""%ptype)
            return


        if self.sp.npart[ptype] <=0:
            print('! catalog not contain ptype %i particles, so cannot assign principal axes'%ptype)
            return

        print('assigning principal axes:')
        print('  using ptype {} particles at radius < {} kpc'.format(ptype, radius_max))
        print('  using ptype {} particles with age = {} Gyr'.format(ptype, age_limits))

        # get particles within age limits
        if age_limits is not None and len(age_limits):
            ages = part.get_property('age')
            part_indices = np.where(
                (ages >= min(age_limits)) * (ages < max(age_limits)))[0]
        else:
            part_indices = np.arange(len(part.get_property('mass')))

        # store galaxy center
        center_position = self.center_position
        center_velocity = self.center_velocity

        # compute radii wrt galaxy center [kpc physical]
        radius_vectors = coordinate_utils.get_distances(
            part.get_property('position')[part_indices], center_position, self.sp.boxsize)

        # keep only particles within radius_max
        radius2s = np.sum(radius_vectors ** 2, 1)
        masks = (radius2s < radius_max ** 2)

        radius_vectors = radius_vectors[masks]
        part_indices = part_indices[masks]
        if (len(radius_vectors) <=0):
            print("WARNING: No particles of pytpe",ptype," within max_radius ",radius_max," and within age limits ",age_limits, "found when determining principle axes!")
            return 

        # compute rotation vectors for principal axes (defined via moment of inertia tensor)
        rotation_vectors, _eigen_values, axes_ratios = coordinate_utils.get_principal_axes(
            radius_vectors, part.get_property('mass')[part_indices], print_results=False)

        # test if need to flip principal axis to ensure that v_phi is defined as moving
        # clockwise as seen from + Z (silly Galactocentric convention)
        velocity_vectors = coordinate_utils.get_velocity_differences(part.get_property('velocity')[part_indices], center_velocity)
        velocity_vectors_rot = coordinate_utils.get_coordinates_rotated(velocity_vectors, rotation_vectors)
        radius_vectors_rot = coordinate_utils.get_coordinates_rotated(radius_vectors, rotation_vectors)
        velocity_vectors_cyl = coordinate_utils.get_velocities_in_coordinate_system(
            velocity_vectors_rot, radius_vectors_rot, 'cartesian', 'cylindrical')
        if np.median(velocity_vectors_cyl[:, 2]) > 0:
            rotation_vectors[0] *= -1  # flip v_phi
        else:
            rotation_vectors[0] *= -1  # consistency for m12f
            rotation_vectors[1] *= -1

        # store in particle catalog
        self.principal_axes_vectors = rotation_vectors
        self.principal_axes_ratios = axes_ratios


        print('  axis ratios:  min/maj = {:.3f}, min/med = {:.3f}, med/maj = {:.3f}'.format(
              axes_ratios[0], axes_ratios[1], axes_ratios[2]))

        return


# THE FUNCTIONS BELOW ARE OLD AND NEED TO BE UPDATE. USE AT YOUR OWN RISK

class Disk(Halo):

    def __init__(self, sp, id=0, rmax=20, height=5):
        super(Disk, self).__init__(sp, id=id)
        # specify the dimensions of the disk
        self.rmax = rmax
        self.height = height

        self.center_position = None
        self.center_velocity = None
        self.principal_axes_vectors = None
        self.principal_axes_ratios = None


        return


    # load basic info to the class
    def load(self, mode='AHF'):

        if (self.k!=0): return # already loaded

        sp = self.sp
        
        # non-cosmological snapshots
        if sp.cosmological==0 or mode!='AHF':

            self.k = 1
            self.time = sp.time
            # Get center from average of gas particles
            part = self.sp.loadpart(0)
            self.xc = np.average(part.get_property('position')[:,0],weights=part.get_property('mass'))
            self.yc = np.average(part.get_property('position')[:,1],weights=part.get_property('mass'))
            self.zc = np.average(part.get_property('position')[:,2],weights=part.get_property('mass'))
            self.rvir = sp.boxsize*2.
            self.Lhat = np.array([0,0,1.])
        
            return

        # cosmological, use AHF
        if (mode=='AHF'):
            
            AHF = sp.loadAHF()
    
            # catalog exists
            if (AHF.k==1):
                
                self.k = 1
                self.time = sp.time
                self.redshift = sp.redshift
                self.catalog = 'AHF'
                self.id = AHF.ID[self.id]
                self.host = AHF.hostHalo[self.id]
                self.npart = AHF.npart[self.id]
                self.ngas = AHF.n_gas[self.id]
                self.nstar = AHF.n_star[self.id]
                self.mvir = AHF.Mvir[self.id]
                self.mgas = AHF.M_gas[self.id]
                self.mstar = AHF.M_star[self.id]
                self.xc = AHF.Xc[self.id]
                self.yc = AHF.Yc[self.id]
                self.zc = AHF.Zc[self.id]
                self.rvir = AHF.Rvir[self.id]
                self.vmax = AHF.Vmax[self.id]
                self.fhi = AHF.fMhires[self.id]
                self.Lhat = AHF.Lhat[:,self.id]
    
        return


    # calculate the center and principle axis of disk from particles instead of just using AHF values
    def set_disk(self, ptype=4, mass_radius_max=100, velocity_radius_max=15, radius_max=10, age_limits=[0,1]):
        # ptype: int
        #   particle types to use to compute disk center and axis
        # velocity_radius_max: float
        #   compute average velocity using particles within this radius [kpc]
        # distance_max : float
        #   maximum radius to select particles [kpc physical]
        # age_limits : float
        #   min and max limits of age to select star particles [Gyr]


        self.assign_center(ptype, mass_radius_max, velocity_radius_max)
        self.assign_principal_axes(ptype, radius_max, age_limits)

        return


    # load all particles in the galactic disk
    def loadpart(self, ptype):

        part = self.part[ptype]

        part.load()
        if part.npart>0:
            part.orientate(self.center_position,self.center_velocity,self.principal_axes_vectors)
            zmag = part.get_property('position')[:,2]
            smag = np.sqrt(np.sum(np.power(part.get_property('position')[:,:2],2),axis=1))
            in_disk = np.logical_and(np.abs(zmag) <= self.height, smag <= self.rmax)
            part.mask(in_disk)

        return part


    # Returns the maximum radius
    def get_rmax(self):
        return self.rmax