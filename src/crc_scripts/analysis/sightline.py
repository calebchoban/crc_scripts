import os
import numpy as np

import pickle

from shutil import copyfile
import h5py

from ..math_utils import weighted_percentile
from .. import config
from .. import coordinate_utils

# Need to add a bunch of derived fields for yt
import yt

def _He_gas_number_density(field, data):
    return (data["gas", "He_nuclei_density"]*(data["gas", "He_metallicity"]-data["gas", "He_dust_metallicity"])/data["gas", "He_metallicity"])
def _He_dust_metallicity(field, data):
    return (data["PartType0", "DustMetallicity"][:,1])
def _C_gas_number_density(field, data):
    return (data["gas", "C_nuclei_density"]*(data["gas", "C_metallicity"]-data["gas", "C_dust_metallicity"])/data["gas", "C_metallicity"])
def _C_dust_metallicity(field, data):
    return (data["PartType0", "DustMetallicity"][:,2])
def _N_gas_number_density(field, data):
    return (data["gas", "N_nuclei_density"]*(data["gas", "N_metallicity"]-data["gas", "N_dust_metallicity"])/data["gas", "N_metallicity"])
def _N_dust_metallicity(field, data):
    return (data["PartType0", "DustMetallicity"][:,3])
def _O_gas_number_density(field, data):
    return (data["gas", "O_nuclei_density"]*(data["gas", "O_metallicity"]-data["gas", "O_dust_metallicity"])/data["gas", "O_metallicity"])
def _O_dust_metallicity(field, data):
    return (data["PartType0", "DustMetallicity"][:,4])
def _Ne_gas_number_density(field, data):
    return (data["gas", "Ne_nuclei_density"]*(data["gas", "Ne_metallicity"]-data["gas", "Ne_dust_metallicity"])/data["gas", "Ne_metallicity"])
def _Ne_dust_metallicity(field, data):
    return (data["PartType0", "DustMetallicity"][:,5])
def _Mg_gas_number_density(field, data):
    return (data["gas", "Mg_nuclei_density"]*(data["gas", "Mg_metallicity"]-data["gas", "Mg_dust_metallicity"])/data["gas", "Mg_metallicity"])
def _Mg_dust_metallicity(field, data):
    return (data["PartType0", "DustMetallicity"][:,6])
def _Si_gas_number_density(field, data):
    return (data["gas", "Si_nuclei_density"]*(data["gas", "Si_metallicity"]-data["gas", "Si_dust_metallicity"])/data["gas", "Si_metallicity"])
def _Si_dust_metallicity(field, data):
    return (data["PartType0", "DustMetallicity"][:,7])
def _S_gas_number_density(field, data):
    return (data["gas", "S_nuclei_density"]*(data["gas", "S_metallicity"]-data["gas", "S_dust_metallicity"])/data["gas", "S_metallicity"])
def _S_dust_metallicity(field, data):
    return (data["PartType0", "DustMetallicity"][:,8])
def _Ca_gas_number_density(field, data):
    return (data["gas", "Ca_nuclei_density"]*(data["gas", "Ca_metallicity"]-data["gas", "Ca_dust_metallicity"])/data["gas", "Ca_metallicity"])
def _Ca_dust_metallicity(field, data):
    return (data["PartType0", "DustMetallicity"][:,9])
def _Fe_gas_number_density(field, data):
    return (data["gas", "Fe_nuclei_density"]*(data["gas", "Fe_metallicity"]-data["gas", "Fe_dust_metallicity"])/data["gas", "Fe_metallicity"])
def _Fe_dust_metallicity(field, data):
    return (data["PartType0", "DustMetallicity"][:,10])

yt.add_field(name=("gas", "He_dust_metallicity"),function=_He_dust_metallicity,sampling_type="local",units="",force_override=True)
yt.add_field(name=("gas", "He_gas_number_density"),function=_He_gas_number_density,sampling_type="local",units="cm**-3",force_override=True)
yt.add_field(name=("gas", "C_dust_metallicity"),function=_C_dust_metallicity,sampling_type="local",units="",force_override=True)
yt.add_field(name=("gas", "C_gas_number_density"),function=_C_gas_number_density,sampling_type="local",units="cm**-3",force_override=True)
yt.add_field(name=("gas", "N_dust_metallicity"),function=_N_dust_metallicity,sampling_type="local",units="",force_override=True)
yt.add_field(name=("gas", "N_gas_number_density"),function=_N_gas_number_density,sampling_type="local",units="cm**-3",force_override=True)
yt.add_field(name=("gas", "O_dust_metallicity"),function=_O_dust_metallicity,sampling_type="local",units="",force_override=True)
yt.add_field(name=("gas", "O_gas_number_density"),function=_O_gas_number_density,sampling_type="local",units="cm**-3",force_override=True)
yt.add_field(name=("gas", "Ne_dust_metallicity"),function=_Ne_dust_metallicity,sampling_type="local",units="",force_override=True)
yt.add_field(name=("gas", "Ne_gas_number_density"),function=_Ne_gas_number_density,sampling_type="local",units="cm**-3",force_override=True)
yt.add_field(name=("gas", "Mg_dust_metallicity"),function=_Mg_dust_metallicity,sampling_type="local",units="",force_override=True)
yt.add_field(name=("gas", "Mg_gas_number_density"),function=_Mg_gas_number_density,sampling_type="local",units="cm**-3",force_override=True)
yt.add_field(name=("gas", "Si_dust_metallicity"),function=_Si_dust_metallicity,sampling_type="local",units="",force_override=True)
yt.add_field(name=("gas", "Si_gas_number_density"),function=_Si_gas_number_density,sampling_type="local",units="cm**-3",force_override=True)
yt.add_field(name=("gas", "S_dust_metallicity"),function=_S_dust_metallicity,sampling_type="local",units="",force_override=True)
yt.add_field(name=("gas", "S_gas_number_density"),function=_S_gas_number_density,sampling_type="local",units="cm**-3",force_override=True)
yt.add_field(name=("gas", "Ca_dust_metallicity"),function=_Ca_dust_metallicity,sampling_type="local",units="",force_override=True)
yt.add_field(name=("gas", "Ca_gas_number_density"),function=_Ca_gas_number_density,sampling_type="local",units="cm**-3",force_override=True)
yt.add_field(name=("gas", "Fe_dust_metallicity"),function=_Fe_dust_metallicity,sampling_type="local",units="",force_override=True)
yt.add_field(name=("gas", "Fe_gas_number_density"),function=_Fe_gas_number_density,sampling_type="local",units="cm**-3",force_override=True)


# This is a class that creates mock sight lines from a given Snapshot using yt
class Sight_Lines(object):

    def __init__(self, sdir, snum, center, normal_vecs, cosmological=1, periodic_bound_fix=False, dirc='./', name=None):

        self.sdir = sdir
        self.snap_num = snum
        self.cosmological = cosmological
        self.dirc = dirc
        # In case the sim was non-cosmological and used periodic BC which causes
        # galaxy to be split between the 4 corners of the box
        self.pb_fix = False
        if periodic_bound_fix and cosmological==0:
            self.pb_fix=True


        # Get the basename of the directory the snapshots are stored in
        self.basename = os.path.basename(os.path.dirname(os.path.normpath(sdir)))
        # Name for saving object
        if name == None:
            self.name = self.basename+'_sightlines_snap_'+str(self.snap_num)+'.pickle'
        else:
            self.name = name

        if self.pb_fix:
            # First need to load in and fix periodic coordinate_utilss since non-cosmo sims with
            # Periodic coordinate_utilss have funky coordinate_utilss
            self.snap_name = 'snapshot_'+str(self.snap_num)+'.hdf5'
            copyfile(self.sdir+self.snap_name, './fixed_'+self.snap_name)
            self.snap_name='fixed_'+self.snap_name

            f = h5py.File('./'+self.snap_name, 'r+')
            grp = f['PartType0']
            old_p = grp['coordinates']
            new_p = old_p[...].copy()
            boxsize = f['Header'].attrs['BoxSize']
            mask1 = new_p > boxsize/2; mask2 = new_p <= boxsize/2
            new_p[mask1] -= boxsize/2; new_p[mask2] += boxsize/2;
            old_p[...]=new_p
            f.close()

            full_dir = './'+self.snap_name
        else:
            full_dir = sdir + "/snapshot_%03d.hdf5" % snum
            snapfile = sdir + "/snapshot_%03d.hdf5" % snum
            # multiple files
            if not (os.path.isfile(snapfile)):
                print(snapfile)
                full_dir = sdir + "/snapdir_%03d" % snum
                snapfile = sdir + "/snapdir_%03d/snapshot_%03d.0.hdf5" % (snum, snum)
                if not (os.path.isfile(snapfile)):
                    print("Snapshot does not exist.")
                    return 0


        # Load data with yt
        self.ds = yt.load(full_dir)
        self.center = self.ds.arr(center,'kpc')
        self.normal_vectors = normal_vecs
        # will use this so we dont have to load the entire snapshot for each ray
        self.data_source = self.ds.sphere(self.center, (30, "kpc"))


        # Check if data has already been saved and if so load that instead
        if os.path.isfile(self.dirc+self.name):
            with open(self.dirc+self.name, 'rb') as handle:
                self.sightline_data = pickle.load(handle)
            print("Preexisting file %s already exists and is loaded!"%self.name)
            self.k = 1
        else:
            print("Preexisting file %s does not exist, so you need to load in the data."%self.name)
            self.k = 0




        return


    # Creates random start and end points and distance for sight line given start galactic radius and center coordinate_utilss of galaxy
    def ray_points(self, center, radius, dist_lims):
        solar_r = radius
        solar_theta = np.random.random()*2*np.pi
        start = np.array([solar_r*np.cos(solar_theta), solar_r*np.sin(solar_theta),0])+center
        # Random sight line distance of 0.1-2kpc
        distance = dist_lims[0]+np.random.random()*dist_lims[1]
        theta = np.random.random()*2*np.pi
        end =  np.array([distance*np.cos(theta), distance*np.sin(theta),0])+start

        return start,end,distance

    # Creates random start and end points for ray in disk within range of radii, parallel with the disk and a random length chosen
    # from a uniform distribution from dist_lims. Current values align with typical lengths from Jenkins09.
    def ray_points_in_disk(self, radius_lims=[8,8], dist_lims=[0.1,1.9]):
        solar_theta = np.random.random() * 2 * np.pi
        # Random radius from limits
        radius = (radius_lims[1]-radius_lims[0])*np.random.random()+radius_lims[0]
        start = np.array([radius * np.cos(solar_theta), radius * np.sin(solar_theta), 0])
        # Random sight line distance
        distance = dist_lims[0]+np.random.random()*dist_lims[1]
        theta = np.random.random() * 2 * np.pi
        end = np.array([distance * np.cos(theta), distance * np.sin(theta), 0]) + start
        return start, end, distance

    # Chooses a random start point in the disk within range of radii> Then finds all young stars within a given range of distances
    # and creates rays to these stars.
    def ray_points_in_disk_to_stars(self, num_starts=10, radius_lims=[8,8], dist_lims=[0.1,1.9], age_max=0.01, max_rays=10):
        total = num_starts*max_rays
        starts = self.ds.arr(np.zeros((total,3)),'kpc')
        ends = self.ds.arr(np.zeros((total,3)),'kpc')
        distances = self.ds.arr(np.zeros(total),'kpc')
        young_star_ages = self.ds.arr(np.zeros(total),'Gyr')

        for i in range(num_starts):
            # First determine randomized starting point of ray
            solar_theta = np.random.random() * 2 * np.pi
            radius = (radius_lims[1] - radius_lims[0]) * np.random.random() + radius_lims[0]
            start = np.array([radius * np.cos(solar_theta), radius * np.sin(solar_theta), 0])
            start = self.ds.arr(start, 'kpc')
            start = coordinate_utils.orientated_coords(start, self.center, self.normal_vectors[2])

            # Now find all young stars from the starting point with the distance limits given
            sphere = self.ds.sphere(start, (dist_lims[1], "kpc"))
            star_age = sphere['PartType4', 'age'].in_units('Gyr')
            star_coords = sphere['PartType4', 'Coordinates']
            radius = np.sqrt(np.sum((star_coords - start) ** 2, axis=1))
            young_stars = (star_age < self.ds.arr(age_max, 'Gyr')) & (radius > self.ds.arr(dist_lims[0], 'kpc'))
            young_star_coords = star_coords[young_stars].in_units('kpc')
            young_star_age = star_age[young_stars]
            end = young_star_coords

            # Only keep as many end points as specified
            if len(end) > max_rays:
                # Keep a random subset
                random_idx = np.random.choice(len(end), size=max_rays, replace=False)
                end = end[random_idx]
                young_star_age = young_star_age[random_idx]

            distance = np.sqrt(np.sum((end-start)**2,axis=1))

            starts[i*max_rays:(i+1)*max_rays] = np.full((max_rays,3),start)
            ends[i * max_rays:(i + 1) * max_rays] = end
            distances[i * max_rays:(i + 1) * max_rays] = distance
            young_star_ages[i * max_rays:(i + 1) * max_rays] = young_star_age

        return starts, ends, distances, young_star_ages

    # Creates random start and end points for ray through galaxy face-on.
    # Sightlines are perpendicular to the galaxy at a random radius, up to rmax, and theta.
    # Distance determines starting and ending height of sightline from the galactic center. So starting and end points are distance/2 from the galaxy
    def ray_points_faceon(self, rmax=5, distance=20, rotation_angles=[0.,0.,0.]):
        theta = np.random.random() * 2 * np.pi
        r = np.random.random() * rmax
        start = np.array([r * np.cos(theta), r * np.sin(theta), distance / 2.])
        end = np.array([r * np.cos(theta), r * np.sin(theta), -distance / 2.])
        return start, end, distance

    # Creates ray through galaxy face-on to young stars within the galaxy.
    def ray_points_faceon_to_young_star(self, rmax=5, age_max=0.1,distance=20, observer_angles=[[0,0,1]]):
        sphere = self.ds.sphere(self.center, (rmax, "kpc"))
        star_age = sphere['PartType4', 'age'].in_units('Gyr')
        young_stars = star_age < self.ds.arr(age_max,'Gyr')
        star_coords = sphere['PartType4', 'Coordinates']
        young_star_coords = star_coords[young_stars]
        end = young_star_coords.in_units('kpc')
        start = np.empty((0,3))
        # Iterate through observer angles (i.e. starting positions)
        for observer_angle in np.array(observer_angles):
            initial_norm = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
            RM = np.linalg.solve(initial_norm, self.normal_vectors)
            observer_norm_vecs = observer_angle.dot(RM)
            start = np.append(start,end.in_units('kpc').v+observer_norm_vecs*distance,axis=0)
        end = self.ds.arr(np.tile(end,(np.shape(observer_angles)[0],1)),'kpc')
        start =  self.ds.arr(start,'kpc')
        distances = self.ds.arr(np.tile(distance,np.shape(end)[0]), 'kpc')
        star_age = self.ds.arr(np.tile(star_age[young_stars].v,len(observer_angles)))

        return start, end, distances, star_age

    # Create N number of sightlines
    def create_random_sightlines(self, sightline_type='in_disk', N=100, append=False, overwrite=False,
                          radius_lims=[8,8], dist_lims=[0.1,1.9], rmax=5, distance=20, rotation_angles=[0.,0.,0.]):
        if self.k and (not overwrite and not append):
            print("The sightline data was already loaded so nothing to do here.")
            print("If you want to append more sightlines or overwrite the data, use the append or overwrite arguments.")
            return


        NH_all = np.zeros(N)
        NH_neutral = np.zeros(N)
        NH2 = np.zeros(N)
        NX_gas=np.zeros((N,len(config.ELEMENTS)))
        NX_dust=np.zeros((N,len(config.ELEMENTS)))
        distances = np.zeros(N)
        points = np.zeros((N,2,3))
        angles = np.zeros((N, 3))

        # Make a ray
        for i in range(N):
            if i%10==0: print(i)
            if sightline_type == 'in_disk':
                start, end, distance = self.ray_points_in_disk(radius_lims=radius_lims, dist_lims=dist_lims)
                start = self.ds.arr(start, 'kpc'); end = self.ds.arr(end, 'kpc'); distance = self.ds.arr(distance, 'kpc')
                start_coord = coordinate_utils.orientated_coords(start, self.center, self.normal_vectors[2])
                end_coord = coordinate_utils.orientated_coords(end, self.center, self.normal_vectors[2])
            elif sightline_type == 'face_on':
                start, end, distance = self.ray_points_faceon(rmax=rmax, distance=distance)
                start = self.ds.arr(start, 'kpc'); end = self.ds.arr(end, 'kpc'); distance = self.ds.arr(distance, 'kpc')
                # Rays will default to face-on (aligned with given z-axis), but you can also give x,y,z, rotation angles to rotate rays
                initial_norm = np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=float)
                new_normal = coordinate_utils.get_coordinate_utilss_rotated(initial_norm, rotation_angles=rotation_angles)
                # Get matrix that will transform x,y,z coordinate_utils basis to the galaxy's coordinate_utils basis
                RM = np.linalg.solve(new_normal, self.normal_vectors)
                start_coord = start.dot(RM) + self.center
                end_coord = end.dot(RM) + self.center
                # start_coord = RM.dot(start) + self.center
                # end_coord = RM.dot(end) + self.center
                # If rotation angle is given, rotation z-axis normal vector in case you want the sightlines to be at an angle with face-on
                # previous_normal = coordinate_utils.get_coordinate_utilss_rotated(np.array([0., 0., 1.]), rotation_angles=rotation_angles)
                # start_coord = coordinate_utils.orientated_coords(start, self.center, self.normal_vectors[2],
                #                                            previous_normal=previous_normal)
                # end_coord = coordinate_utils.orientated_coords(end, self.center, self.normal_vectors[2],
                #                                            previous_normal=previous_normal)
            else:
                print("Given sightline_type is not supported. Must be in_disk or face_on.")
                return

            ray = self.ds.ray(start_coord, end_coord, data_source=self.data_source)
            points[i] = np.array([start, end])
            distances[i] = distance.v
            NH_neutral[i] = np.sum(ray["gas", "H_p0_number_density"] * ray['dts'].v * distance)
            NH_all[i] = np.sum(ray["gas", "H_nuclei_density"] * ray['dts'].v * distance)
            NH2[i] = np.sum((ray["gas", "H_p0_number_density"]*ray['PartType0', 'MolecularMassFraction'])/2.*ray['dts'].v*distance)
            angles[i] = rotation_angles

            for j in range(1, len(config.ELEMENTS)):
                elem = config.ELEMENTS[j]
                NX_gas[i, j] = np.sum(ray["gas", elem + "_gas_number_density"] * ray['dts'].v * distance)
                NX_dust[i, j] = np.sum((ray["gas", elem + "_nuclei_density"] - ray["gas", elem + "_gas_number_density"]) * ray['dts'].v * distance)

            # Now add them all up for total Z
            NX_gas[i,0] = np.sum(NX_gas[i,1:])
            NX_dust[i,0] = np.sum(NX_dust[i,1:])

        depl_X = NX_gas/(NX_gas+NX_dust)
        new_sightline_data = {'NH_neutral': NH_neutral, 'NH2': NH2, 'NH': NH_all, 'NX_gas': NX_gas, 'NX_dust': NX_dust, 'depl_X': depl_X,
                              'points': points, 'distance': distances,'rotation_angle': angles, 'sightline_type': [sightline_type]*N}

        if overwrite or not self.k:
            self.sightline_data = new_sightline_data
        elif append:
            for key in self.sightline_data:
                if key=='sightline_type':
                    self.sightline_data[key] = self.sightline_data[key]+new_sightline_data[key]
                else:
                    self.sightline_data[key] = np.append(self.sightline_data[key],new_sightline_data[key],axis=0)

        pickle.dump(self.sightline_data, open(self.name, "wb" ))

        # Delete extra snap if pb_fix work around used
        if self.pb_fix:
            os.remove(self.snap_name)

        self.k=1

        return



    def create_determined_sightlines(self, sightline_type='face_on_young_star', save_interval=10, max_num=10000, ray_args={}):

        # Don't need to do anything if data has already been loaded
        if  self.k and not np.any(self.sightline_data['NH'] <= 0.):
            print("All sightlines have already been calculated so nothing to load.")
            return

        if sightline_type == 'face_on_young_star':
            starts, ends, distances, ages = self.ray_points_faceon_to_young_star(**ray_args)
            N = len(ages)
            if 'observer_angles' in ray_args:
                num_angles = np.shape(ray_args[ 'observer_angles'])[0]
                indiv_N = N/num_angles
                print("There are %i young star particles in the galaxy to make face-on sightlines from. "%indiv_N)
                print("%i sight lines will be made for each star particle based on the provided observer angles." % num_angles)
            else:
                print("There are %i young star particles in the galaxy to make face-on sightlines from."%N)
                print("One sight line from the z-axis will be made for each star particle.")
            if N > max_num:
                print("We will randomly select %i of them to make sightlines from."%max_num)
            else:
                print("max_num=%i is greater than the possible number of sightlines so we will truncate."%max_num)
        elif sightline_type == 'in_disk_young_star':
            num_starts = max_num//ray_args['max_rays']
            starts, ends, distances, ages = self.ray_points_in_disk_to_stars(num_starts=num_starts,**ray_args)
            N = max_num
        else:
            print("Given sightline_type is not supported.")
            return

        # Start from scratch if not loaded
        if not self.k:
            self.sightline_data = {'NH_neutral': np.zeros(N), 'NH2': np.zeros(N), 'NH': np.zeros(N), 'NX_gas': np.zeros((N, len(config.ELEMENTS))),
                                   'NX_dust': np.zeros((N, len(config.ELEMENTS))), 'depl_X': np.zeros((N, len(config.ELEMENTS))),
                                   'points': np.zeros((N, 2, 3)), 'distance': np.zeros(N), 'rotation_angle': np.zeros((N, 3)),
                                   'age': ages}

        # Make a ray
        for i in range(N):
            # Check if data is partially loaded. If so skip ahead
            if self.sightline_data['NH_neutral'][i] != 0:
                continue

            if i%save_interval==0:
                pickle.dump(self.sightline_data, open(self.dirc+self.name, "wb"))
                print(i)


            start_coord = starts[i]
            end_coord = ends[i]

            ray = self.ds.ray(start_coord, end_coord, data_source=self.data_source)
            self.sightline_data['points'][i] = np.array([start_coord.in_units('kpc').v, end_coord.in_units('kpc').v])
            distance = distances[i]
            self.sightline_data['distance'][i] = distance.in_units('kpc').v
            self.sightline_data['NH_neutral'][i] = np.sum(ray["gas", "H_p0_number_density"] * ray['dts'].v * distance)
            self.sightline_data['NH'][i] = np.sum(ray["gas", "H_nuclei_density"] * ray['dts'].v * distance)
            self.sightline_data['NH2'][i] = np.sum((ray["gas", "H_p0_number_density"]*ray['PartType0', 'MolecularMassFraction'])/2.*ray['dts'].v*distance)
            #self.sightline_data['rotation_angles'][i] = rotation_angles

            for j in range(1, len(config.ELEMENTS)):
                elem = config.ELEMENTS[j]
                self.sightline_data['NX_gas'][i, j] = np.sum(ray["gas", elem + "_gas_number_density"] * ray['dts'].v * distance)
                self.sightline_data['NX_dust'][i, j] = np.sum((ray["gas", elem + "_nuclei_density"] - ray["gas", elem + "_gas_number_density"]) * ray['dts'].v * distance)

            # Now add them all up for total Z
            self.sightline_data['NX_gas'][i,0] = np.sum(self.sightline_data['NX_gas'][i,1:])
            self.sightline_data['NX_dust'][i,0] = np.sum(self.sightline_data['NX_dust'][i,1:])
            self.sightline_data['depl_X'][i] = self.sightline_data['NX_gas'][i]/(self.sightline_data['NX_gas'][i]+self.sightline_data['NX_dust'][i])


        pickle.dump(self.sightline_data, open(self.dirc+self.name, "wb" ))

        self.k=1

        return