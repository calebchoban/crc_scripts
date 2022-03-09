import numpy as np
from . import utils
from . import coordinate

# This is a class that manages a galaxy/halo in a given
# snapshot, for either cosmological or isolated simulations.

class Halo(object):

    def __init__(self, sp, id=0):

        # the halo must be taken from a snapshot
        self.sp = sp
        self.k = -1 if sp.k==-1 else 0 # no snapshot, no halo
        self.id = id
        self.cosmological = sp.cosmological

        # these are linked to its parent snapshot
        self.header = sp.header
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

        return


    # Set zoom-in region if you don't want all data in rvir
    def set_zoom(self, rout=1.0, kpc=False, ptype=4):

        # How far out do you want to load data, default is rvir
        self.rout = rout
        self.outkpc = kpc
        self.zoom_ptype = ptype
        self.zoom = True

        return


    # load basic info to the class
    def load(self, mode='AHF'):

        if (self.k!=0): return # already loaded

        sp = self.sp

        # non-cosmological snapshots
        if (sp.cosmological==0):

            self.k = 1
            self.time = sp.time
            self.xc = sp.boxsize/2.
            self.yc = sp.boxsize/2.
            self.zc = sp.boxsize/2.
            self.rvir = sp.boxsize*2.
            self.Lhat = np.array([0,0,1.])

            return

        # cosmological, use AHF
        if mode=='AHF':
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
                self.Lhat = AHF.Lhat[self.id]

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


        # Default to chose center of snapshot box
        else:

            self.k = 0 # so you can re-load it
            self.redshift = sp.redshift
            self.xc = sp.boxsize/2.
            self.yc = sp.boxsize/2.
            self.zc = sp.boxsize/2.
            self.rvir = sp.boxsize*2.
            self.Lhat = np.array([0,0,1.])
    
        return


    # Returns the maximum radius
    def get_rmax(self):
        return self.rvir


    # load all particles in the halo/galaxy centered on halo center
    def loadpart(self, ptype):

        part = self.part[ptype]

        part.load()
        if self.zoom:
            rmax = self.rout*self.rvir if not self.outkpc else self.rout
            self.calculate_center(self.zoom_ptype)
        else:
            rmax = self.rvir
        xc=self.xc; yc=self.yc; zc=self.zc
        part.orientate([xc,yc,zc])
        in_halo = np.sum(np.power(part.p,2),axis=1) <= np.power(rmax,2.)
        part.mask(in_halo)

        return part


    def loadheader(self):

        header=self.header
        header.load()

        return header

    
    # load AHF particles for the halo
    def loadAHFpart(self):
    
        AHF = self.sp.loadAHF()
        PartID, PType = AHF.loadpart(id=self.id)
    
        return PartID, PType


    # get star formation history using all stars in Rvir
    def get_SFH(self, dt=0.01, cum=0, cen=None, rout=1.0, kpc=0):

        self.load()
        if (self.k==-1): return 0., 0.

        try:
            xc, yc, zc = cen[0], cen[1], cen[2]
        except (TypeError,IndexError):
            xc, yc, zc = self.xc, self.yc, self.zc

        part = self.loadpart(4)
        if (part.k==-1): return 0., 0.

        p, sft, m = part.p, part.sft, part.m
        r = np.sqrt((p[:,0])**2+(p[:,1])**2+(p[:,2])**2)
        rmax = rout*self.rvir if kpc==0 else rout
        t, sfr = utils.SFH(sft[r<rmax], m[r<rmax], dt=dt, cum=cum, sp=self.sp)

        return t, sfr


    def calculate_center(self, ptype=4, velocity_radius_max=15):

        if self.calc_center: return
        part = self.part[ptype]
        part.load()

        # calculate center position
        center_position = coordinate.get_center_position_zoom(part.p, part.m, self.sp.boxsize,
            center_position=np.array([self.xc,self.yc,self.zc]))
        self.xc = center_position[0]
        self.yc = center_position[1]
        self.zc = center_position[2]

        print('  center position [kpc] = {:.3f}, {:.3f}, {:.3f}'.format(
            center_position[0], center_position[1], center_position[2]))

        # calculate center velocity
        center_velocity = coordinate.get_center_velocity(
            part.v, part.m, part.p, center_position, velocity_radius_max, self.sp.boxsize)
        self.vx = center_velocity[0]
        self.vy = center_velocity[1]
        self.vz = center_velocity[2]

        print('  center velocity [km/s] = {:.1f}, {:.1f}, {:.1f}'.format(
            center_velocity[0], center_velocity[1], center_velocity[2]))

        self.calc_center = True
        return


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
            self.xc = np.average(part.p[:,0],weights=part.m)
            self.yc = np.average(part.p[:,1],weights=part.m)
            self.zc = np.average(part.p[:,2],weights=part.m)
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
                self.Lhat = AHF.Lhat[self.id]
    
        return


    # calculate the center and principle axis of disk from particles instead of just using AHF values
    def set_disk(self, ptype=4, velocity_radius_max=15, radius_max=10, age_limits=[0,1]):
        # ptype: int
        #   particle types to use to compute disk center and axis
        # velocity_radius_max: float
        #   compute average velocity using particles within this radius [kpc]
        # distance_max : float
        #   maximum radius to select particles [kpc physical]
        # age_limits : float
        #   min and max limits of age to select star particles [Gyr]


        self.assign_center(ptype, velocity_radius_max)
        self.assign_principal_axes(ptype, radius_max, age_limits)

        return

    # calculate center position and velocity
    def assign_center(self, ptype=4, velocity_radius_max=15):

        print('assigning center of galaxy:')
        part = self.part[ptype]
        part.load()
        if len(part.m) < 0:
            print("""There are no particles of ptype %i to use for assigning center of galaxy. You should use another
                  type of particle such as gas or dark matter."""%ptype)
            return

        # calculate center position
        self.center_position = coordinate.get_center_position_zoom(part.p, part.m, self.sp.boxsize,
            center_position=np.array([self.xc,self.yc,self.zc]))
        self.xc = self.center_position[0]
        self.yc = self.center_position[1]
        self.zc = self.center_position[2]

        print('  center position [kpc] = {:.3f}, {:.3f}, {:.3f}'.format(
            self.center_position[0], self.center_position[1], self.center_position[2]))

        # calculate center velocity
        self.center_velocity = coordinate.get_center_velocity(
            part.v, part.m, part.p, self.center_position, velocity_radius_max, self.sp.boxsize)
        self.vx = self.center_velocity[0]
        self.vy = self.center_velocity[1]
        self.vz = self.center_velocity[2]

        print('  center velocity [km/s] = {:.1f}, {:.1f}, {:.1f}'.format(
            self.center_velocity[0], self.center_velocity[1], self.center_velocity[2]))

        return


    def assign_principal_axes(self, ptype=4, radius_max=10, age_limits=[0,1]):
        '''
        Assign principal axes (rotation vectors defined by moment of inertia tensor) to galaxy.

        Parameters
        ----------
        part : dictionary class : catalog of particles
        distance_max : float : maximum radius to select particles [kpc physical]
        age_limits : float : min and max limits of age to select star particles [Gyr]
        '''
        part = self.part[ptype] # particle types to use to compute MOI tensor and principal axes
        part.load()
        if len(part.m) < 0:
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
            ages = part.age
            part_indices = np.where(
                (ages >= min(age_limits)) * (ages < max(age_limits)))[0]
        else:
            part_indices = np.arange(len(part.m))

        # store galaxy center
        center_position = self.center_position
        center_velocity = self.center_velocity

        # compute radii wrt galaxy center [kpc physical]
        radius_vectors = coordinate.get_distances(
            part.p[part_indices], center_position, self.sp.boxsize, self.sp.scale_factor)

        # keep only particles within radius_max
        radius2s = np.sum(radius_vectors ** 2, 1)
        masks = (radius2s < radius_max ** 2)

        radius_vectors = radius_vectors[masks]
        part_indices = part_indices[masks]

        # compute rotation vectors for principal axes (defined via moment of inertia tensor)
        rotation_vectors, _eigen_values, axes_ratios = coordinate.get_principal_axes(
            radius_vectors, part.m[part_indices], print_results=False)

        # test if need to flip principal axis to ensure that v_phi is defined as moving
        # clockwise as seen from + Z (silly Galactocentric convention)
        velocity_vectors = coordinate.get_velocity_differences(part.v[part_indices], center_velocity)
        velocity_vectors_rot = coordinate.get_coordinates_rotated(velocity_vectors, rotation_vectors)
        radius_vectors_rot = coordinate.get_coordinates_rotated(radius_vectors, rotation_vectors)
        velocity_vectors_cyl = coordinate.get_velocities_in_coordinate_system(
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



    # load all particles in the galactic disk
    def loadpart(self, ptype):

        part = self.part[ptype]

        part.load()
        if part.npart>0:
            part.orientate(self.center_position,self.center_velocity,self.principal_axes_vectors)
            zmag = part.p[:,2]
            smag = np.sqrt(np.sum(np.power(part.p[:,:2],2),axis=1))
            in_disk = np.logical_and(np.abs(zmag) <= self.height, smag <= self.rmax)
            part.mask(in_disk)

        return part


    # Returns the maximum radius
    def get_rmax(self):
        return self.rmax


    # Calculate the stellar scale radius
    def calc_stellar_scale_r(self, guess=[1E3,2], bounds=(0, [1E6, 10]), radius_max=None):
        # guess : initial guess for central density and scale length
        # bounds : Bounds for possible central density (M_sol/pc^2) and scale length (kpc) values
        # radius_max : maximum disk radius for fit

        if radius_max is None:
            radius_max = self.rmax

        stars = self.part[4]
        stars.load()
        stars.orientate(self.center_position,self.center_velocity,self.principal_axes_vectors)
        star_mass = stars.get_property('M')
        r_bins = np.linspace(0,radius_max,int(np.floor(radius_max/0.25)))
        r_vals = np.array([(r_bins[i+1]+r_bins[i])/2. for i in range(len(r_bins)-1)])
        rmag = np.sqrt(np.sum(np.power(stars.p[:,:2],2),axis=1))
        sigma_star = np.zeros(len(r_vals))
        for i in range(len(r_vals)):
            rmin = r_bins[i]; rmax = r_bins[i+1]
            mass = np.sum(star_mass[(rmag>rmin) & (rmag<=rmax)])
            area = np.pi*(rmax**2 - rmin**2)
            sigma_star[i] = mass/(area*1E6) # M_sol/pc^2

        fit_params,_ = utils.fit_exponential(r_vals, sigma_star, guess=guess, bounds=bounds)
        coeff = fit_params[0]; scale_r=fit_params[1];
        print("Calculated stellar disk scale length: Scale Radius = %e kpc, \
              Central Surface Density = %e M_solar/pc^2"%(scale_r, coeff))
        import matplotlib.pyplot as plt
        plt.scatter(r_vals, sigma_star)
        x_vals = np.linspace(0,radius_max,100)
        plt.plot(x_vals, coeff*np.exp(-x_vals/scale_r))
        plt.yscale('log')
        plt.savefig('stellar_scale_test.png')

        return scale_r
