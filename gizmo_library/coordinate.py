'''
Utility functions for positions and velocities.

@author: Andrew Wetzel <arwetzel@gmail.com>
'''

import numpy as np
from . import config

#===================================================================================================
# coordinate transformation
#===================================================================================================
def get_positions_in_coordinate_system(
    position_vectors, system_from='cartesian', system_to='cylindrical'):
    '''
    Convert input 3-D position vectors from (cartesian, cylindrical, spherical) to
    (cartesian, cylindrical, spherical):
        cartesian : x, y, z
        cylindrical : R (along major axes, absolute/unsigned), Z (along minor axis, signed),
            angle phi [0, 2 * pi)
        spherical : r (absolute/unsigned), angle theta [0, pi), angle phi [0, 2 * pi)

    Parameters
    ----------
    position_vectors : array (object number x 3) : position[s]/distance[s] wrt a center

    Returns
    -------
    positions_new : array (object number x 3) : position[s]/distance[s] in new coordiante system
    '''
    assert system_from in ('cartesian', 'cylindrical', 'spherical')
    assert system_to in ('cartesian', 'cylindrical', 'spherical')

    if system_from == system_to:
        return position_vectors

    position_vectors = np.asarray(position_vectors)
    if np.ndim(position_vectors) == 1:
        position_vectors = np.asarray([position_vectors])

    assert np.shape(position_vectors)[1] == 3

    positions_new = np.zeros(position_vectors.shape, dtype=position_vectors.dtype)

    if system_from == 'cartesian':
        if system_to == 'cylindrical':
            # R = sqrt(x^2 + y^2)
            positions_new[:, 0] = np.sqrt(np.sum(position_vectors[:, [0, 1]] ** 2, 1))
            # Z = z
            positions_new[:, 1] = position_vectors[:, 2]
            # phi = arctan(y / x)
            positions_new[:, 2] = np.arctan2(position_vectors[:, 1], position_vectors[:, 0])
            positions_new[:, 2][positions_new[:, 2] < 0] += 2 * np.pi  # convert to [0, 2 * pi)
        elif system_to == 'spherical':
            # r = sqrt(x^2 + y^2 + z^2)
            positions_new[:, 0] = np.sqrt(np.sum(position_vectors ** 2, 1))
            # theta = arccos(z / r)
            positions_new[:, 1] = np.arccos(position_vectors[:, 2] / positions_new[:, 0])
            # phi = arctan(y / x)
            positions_new[:, 2] = np.arctan2(position_vectors[:, 1], position_vectors[:, 0])
            positions_new[:, 2][positions_new[:, 2] < 0] += 2 * np.pi  # convert to [0, 2 * pi)

    elif system_from == 'cylindrical':
        if system_to == 'cartesian':
            # x = R * cos(phi)
            positions_new[:, 0] = position_vectors[:, 0] * np.cos(position_vectors[:, 2])
            # y = R * sin(phi)
            positions_new[:, 1] = position_vectors[:, 0] * np.sin(position_vectors[:, 2])
            # z = Z
            positions_new[:, 2] = position_vectors[:, 1]
        elif system_to == 'spherical':
            # r = sqrt(R^2 + Z^2)
            positions_new[:, 0] = np.sqrt(position_vectors[:, 0] ** 2 + position_vectors[:, 1] ** 2)
            # theta = arctan(R / Z)
            positions_new[:, 1] = np.arctan2(position_vectors[:, 0], position_vectors[:, 1])
            # phi = phi
            positions_new[:, 2] = position_vectors[:, 2]

    elif system_from == 'spherical':
        if system_to == 'cartesian':
            # x = r * sin(theta) * cos(phi)
            positions_new[:, 0] = (position_vectors[:, 0] * np.sin(position_vectors[:, 1]) *
                                   np.cos(position_vectors[:, 2]))
            # y = r * sin(theta) * sin(phi)
            positions_new[:, 1] = (position_vectors[:, 0] * np.sin(position_vectors[:, 1]) *
                                   np.sin(position_vectors[:, 2]))
            # z = r * cos(theta)
            positions_new[:, 2] = position_vectors[:, 0] * np.cos(position_vectors[:, 1])
        elif system_to == 'cylindrical':
            # R = r * sin(theta)
            positions_new[:, 0] = position_vectors[:, 0] * np.sin(position_vectors[:, 1])
            # Z = r * cos(theta)
            positions_new[:, 1] = position_vectors[:, 0] * np.cos(position_vectors[:, 1])
            # phi = phi
            positions_new[:, 2] = position_vectors[:, 2]

    # if only one position vector, return as 1-D array
    if len(positions_new) == 1:
        positions_new = positions_new[0]

    return positions_new


def get_velocities_in_coordinate_system(
    velocity_vectors, position_vectors, system_from='cartesian', system_to='cylindrical'):
    '''
    Convert input 3-D velocity vectors from (cartesian, cylindrical, spherical) to
    (cartesian, cylindrical, spherical).
        cartesian : velocity along x, y, z
        cylindrical : velocity along R (major axes), Z (minor axis), angle phi
        spherical : velocity along r, angle theta, angle phi

    Parameters
    ----------
    velocity_vectors : array (object number x 3) : velocity[s] wrt a center
    position_vectors : array (object number x 3) : position[s]/distance[s] wrt a center

    Returns
    -------
    velocity_vectors_new : array (object number x 3) : velocity[s] in new coordiante system
    '''
    assert system_from in ('cartesian', 'cylindrical', 'spherical')
    assert system_to in ('cartesian', 'cylindrical', 'spherical')

    if system_from == system_to:
        return velocity_vectors

    velocity_vectors = np.asarray(velocity_vectors)
    if np.ndim(velocity_vectors) == 1:
        velocity_vectors = np.asarray([velocity_vectors])

    position_vectors = np.asarray(position_vectors)
    if np.ndim(position_vectors) == 1:
        position_vectors = np.asarray([position_vectors])

    assert np.shape(velocity_vectors)[1] == 3 and np.shape(position_vectors)[1] == 3

    velocities_new = np.zeros(velocity_vectors.shape, dtype=velocity_vectors.dtype)

    if system_from == 'cartesian':
        # convert position vectors
        # R = {x,y}
        R = position_vectors[:, [0, 1]]
        R_norm = np.zeros(R.shape, position_vectors.dtype)
        # R_total = sqrt(x^2 + y^2)
        R_total = np.sqrt(np.sum(R ** 2, 1))
        masks = np.where(R_total > 0)[0]
        # need to do this way
        R_norm[masks] = np.transpose(R[masks].transpose() / R_total[masks])

        if system_to == 'cylindrical':
            # v_R = dot(v_{x,y}, R_norm)
            velocities_new[:, 0] = np.sum(velocity_vectors[:, [0, 1]] * R_norm, 1)
            # v_Z = v_z
            velocities_new[:, 1] = velocity_vectors[:, 2]
            # v_phi = cross(R_norm, v_{x,y})
            velocities_new[:, 2] = np.cross(R_norm, velocity_vectors[:, [0, 1]])
        elif system_to == 'spherical':
            # convert position vectors
            position_vectors_norm = np.zeros(position_vectors.shape, position_vectors.dtype)
            position_vectors_total = np.sqrt(np.sum(position_vectors ** 2, 1))
            masks = np.where(position_vectors_total > 0)[0]
            # need to do this way
            position_vectors_norm[masks] = np.transpose(
                position_vectors[masks].transpose() / position_vectors_total[masks])

            # v_r = dot(v, r)
            velocities_new[:, 0] = np.sum(velocity_vectors * position_vectors_norm, 1)
            # v_theta
            a = np.transpose([R_norm[:, 0] * position_vectors_norm[:, 2],
                              R_norm[:, 1] * position_vectors_norm[:, 2],
                              -R_total / position_vectors_total])
            velocities_new[:, 1] = np.sum(velocity_vectors * a, 1)
            # v_phi = cross(R_norm, v_{x,y})
            velocities_new[:, 2] = np.cross(R_norm, velocity_vectors[:, [0, 1]])

    elif system_from == 'cylindrical':
        raise ValueError('not yet support conversion from {} to {}'.format(system_from, system_to))

    elif system_from == 'spherical':
        raise ValueError('not yet support conversion from {} to {}'.format(system_from, system_to))

    # if only one velocity vector, return as 1-D array
    if len(velocities_new) == 1:
        velocities_new = velocities_new[0]

    return velocities_new


#===================================================================================================
# rotation of position or velocity
#===================================================================================================
def get_coordinates_rotated(coordinate_vectors, rotation_vectors=None, rotation_angles=None):
    '''
    Get 3-D coordinate[s] (distance or velocity vector[s]) that are rotated by input rotation
    vectors or input rotation angles.
    If rotation_vectors, need to input vectors that are orthogonal.
    If rotation_angles, rotate by rotation_angles[0] about x-axis, then by rotation_angles[1] about
    y-axis, then by rotation_angles[2] about z-axis.

    Parameters
    ----------
    coordinate_vectors : array : coordinate[s] (distance[s] or velocity[s]) wrt a center of rotation
        (object number x dimension number)
    rotation_vectors : array : *orthogonal* rotation vectors (such as max, med, min eigen-vectors)
    rotation_angles : array : rotation angles about x-axis, y-axis, z-axis [radians]

    Returns
    -------
    coordinate[s] (distance[s] or velocity[s]) in rotated basis :
        array (object number x dimension number)
    '''
    if rotation_vectors is not None:
        # sanity check - ensure input rotation vectors are orthogonal
        tolerance = 1e-6
        if (np.abs(np.dot(rotation_vectors[0], rotation_vectors[1])) > tolerance or
                np.abs(np.dot(rotation_vectors[0], rotation_vectors[2])) > tolerance or
                np.abs(np.dot(rotation_vectors[1], rotation_vectors[2])) > tolerance):
            raise ValueError('input rotation_vectors are not orthogonal')

    elif rotation_angles is not None:
        m11 = np.cos(rotation_angles[1]) * np.cos(rotation_angles[2])
        m12 = (np.cos(rotation_angles[0]) * np.sin(rotation_angles[2]) +
               np.sin(rotation_angles[0]) * np.sin(rotation_angles[1]) * np.cos(rotation_angles[2]))
        m13 = (np.sin(rotation_angles[0]) * np.sin(rotation_angles[2]) -
               np.cos(rotation_angles[0]) * np.sin(rotation_angles[1]) * np.cos(rotation_angles[2]))
        m21 = -np.cos(rotation_angles[1]) * np.sin(rotation_angles[2])
        m22 = (np.cos(rotation_angles[0]) * np.cos(rotation_angles[2]) -
               np.sin(rotation_angles[0]) * np.sin(rotation_angles[1]) * np.sin(rotation_angles[2]))
        m23 = (np.sin(rotation_angles[0]) * np.cos(rotation_angles[2]) +
               np.cos(rotation_angles[0]) * np.sin(rotation_angles[1]) * np.sin(rotation_angles[2]))
        m31 = np.sin(rotation_angles[1])
        m32 = -np.sin(rotation_angles[0]) * np.cos(rotation_angles[1])
        m33 = np.cos(rotation_angles[0]) * np.cos(rotation_angles[1])

        rotation_vectors = np.array([
            [m11, m12, m13],
            [m21, m22, m23],
            [m31, m32, m33]],
            dtype=coordinate_vectors.dtype
        )

    else:
        raise ValueError('need to input either rotation angles or rotation vectors')

    # have to do this way
    coordinate_vectors_rotated = np.asarray(
        np.dot(coordinate_vectors, rotation_vectors.transpose()), dtype=coordinate_vectors.dtype)

    return coordinate_vectors_rotated


def get_principal_axes(position_vectors, weights=None, print_results=True):
    '''
    Compute principal axes of input position_vectors (which should be wrt a center),
    defined via the moment of inertia tensor.
    Get reverse-sorted eigen-vectors, eigen-values, and axis ratios of these principal axes.

    Parameters
    ----------
    position_vectors : array (object number x dimension number) :
        position[s]/distance[s] wrt a center
    weights : array : weight for each position (usually mass) - if None, assume all have same weight
    print_results : boolean : whether to print axis ratios

    Returns
    -------
    eigen_vectors : array : max, med, min eigen-vectors
    eigen_values : array : max, med, min eigen-values
    axis_ratios : array : ratios of principal axes
    '''
    if weights is None or not len(weights):
        weights = 1
    else:
        weights = weights / np.median(weights)

    if position_vectors.shape[1] == 3:
        # 3-D
        xx = np.sum(weights * position_vectors[:, 0] ** 2)
        yy = np.sum(weights * position_vectors[:, 1] ** 2)
        zz = np.sum(weights * position_vectors[:, 2] ** 2)
        xy = yx = np.sum(weights * position_vectors[:, 0] * position_vectors[:, 1])
        xz = zx = np.sum(weights * position_vectors[:, 0] * position_vectors[:, 2])
        yz = zy = np.sum(weights * position_vectors[:, 1] * position_vectors[:, 2])

        moi_tensor = [[xx, xy, xz],
                      [yx, yy, yz],
                      [zx, zy, zz]]

    elif position_vectors.shape[1] == 2:
        # 2-D
        xx = np.sum(weights * position_vectors[:, 0] ** 2)
        yy = np.sum(weights * position_vectors[:, 1] ** 2)
        xy = yx = np.sum(weights * position_vectors[:, 0] * position_vectors[:, 1])

        moi_tensor = [[xx, xy],
                      [yx, yy]]

    eigen_values, eigen_vectors = np.linalg.eig(moi_tensor)

    # order eigen-vectors by eigen-values, from largest to smallest
    eigen_indices_sorted = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[eigen_indices_sorted]
    eigen_values /= eigen_values.max()  # renormalize to 1
    # make eigen_vectors[0] corresponds to vector of eigen_values[0]
    eigen_vectors = eigen_vectors.transpose()[eigen_indices_sorted]

    if position_vectors.shape[1] == 3:
        axis_ratios = np.sqrt(
            [eigen_values[2] / eigen_values[0],
             eigen_values[2] / eigen_values[1],
             eigen_values[1] / eigen_values[0]]
        )

        if print_results:
            print('* principal axes:  min/maj = {:.3f}, min/med = {:.3f}, med/maj = {:.3f}'.format(
                  axis_ratios[0], axis_ratios[1], axis_ratios[2]))

    elif position_vectors.shape[1] == 2:
        axis_ratios = eigen_values[1] / eigen_values[0]

        if print_results:
            print('* principal axes:  min/maj = {:.3f}'.format(axis_ratios))

    return eigen_vectors, eigen_values, axis_ratios


#===================================================================================================
# position distances
#===================================================================================================
def get_positions_periodic(positions, periodic_length=None):
    '''
    Get position in range [0, periodic_length).

    Parameters
    ----------
    positions : float or array
    periodic_length : float : periodicity length (if none, return array as is)
    '''
    if periodic_length is None:
        return positions

    if np.isscalar(positions):
        if positions >= periodic_length:
            positions -= periodic_length
        elif positions < 0:
            positions += periodic_length
    else:
        positions[positions >= periodic_length] -= periodic_length
        positions[positions < 0] += periodic_length

    return positions


def get_position_differences(position_difs, periodic_length=None):
    '''
    Get distance / separation vector, in range [-periodic_length/2, periodic_length/2).

    Parameters
    ----------
    position_difs : array : position difference[s]
    periodic_length : float : periodicity length (if none, return array as is)
    '''
    if not periodic_length:
        return position_difs
    else:
        if np.isscalar(periodic_length) and periodic_length <= 1:
            print('! got unusual periodic_length = {}'.format(periodic_length))

    if np.isscalar(position_difs):
        if position_difs >= 0.5 * periodic_length:
            position_difs -= periodic_length
        elif position_difs < -0.5 * periodic_length:
            position_difs += periodic_length
    else:
        position_difs[position_difs >= 0.5 * periodic_length] -= periodic_length
        position_difs[position_difs < -0.5 * periodic_length] += periodic_length

    return position_difs


def get_distances(
    positions_1=None, positions_2=None, periodic_length=None, scalefactor=None,
    total_distance=False):
    '''
    Get vector or total/scalar distance[s] between input position vectors.

    Parameters
    ----------
    positions_1 : array : position[s]
    positions_2 : array : position[s]
    periodic_length : float : periodic length (if none, not use periodic)
    scalefactor : float or array : expansion scale-factor (to convert comoving to physical)
    total : boolean : whether to compute total/scalar (instead of vector) distance

    Returns
    -------
    distances : array (object number x dimension number, or object number) :
        vector or total/scalar distance[s]
    '''
    if not isinstance(positions_1, np.ndarray):
        positions_1 = np.array(positions_1)
    if not isinstance(positions_2, np.ndarray):
        positions_2 = np.array(positions_2)

    if len(positions_1.shape) == 1 and len(positions_2.shape) == 1:
        shape_pos = 0
    else:
        shape_pos = 1

    distances = get_position_differences(positions_1 - positions_2, periodic_length)

    if total_distance:
        distances = np.sqrt(np.sum(distances ** 2, shape_pos))

    if scalefactor is not None:
        if scalefactor > 1 or scalefactor <= 0:
            print('! got unusual scalefactor = {}'.format(scalefactor))
        distances *= scalefactor

    return distances


def get_distances_angular(positions_1=None, positions_2=None, sphere_angle=360):
    '''
    Get angular separation[s] between input positions, valid for small separations.

    Parameters
    ----------
    positions_1, positions_2 : arrays : positions in [RA, dec]
    sphere_angle : float : angular size of sphere 360 [degrees], 2 * pi [radians]

    Returns
    -------
    angular distances : array (object number x angular dimension number)
    '''
    if sphere_angle == 360:
        angle_scale = constant.radian_per_degree
    elif sphere_angle == 2 * np.pi:
        angle_scale = 1
    else:
        raise ValueError('angle of sphere = {} does not make sense'.format(sphere_angle))

    if np.ndim(positions_1) == 1 and positions_1.size == 2:
        ras_1, decs_1 = positions_1[0], positions_1[1]
    else:
        ras_1, decs_1 = positions_1[:, 0], positions_1[:, 1]

    if np.ndim(positions_2) == 1 and positions_2.size == 2:
        ras_2, decs_2 = positions_2[0], positions_2[1]
    else:
        ras_2, decs_2 = positions_2[:, 0], positions_2[:, 1]

    return np.sqrt((get_position_differences(ras_1 - ras_2, sphere_angle) *
                    np.cos(angle_scale * 0.5 * (decs_1 + decs_2))) ** 2 + (decs_1 - decs_2) ** 2)


#===================================================================================================
# velocity conversion
#===================================================================================================
def get_velocity_differences(
    velocity_vectors_1=None, velocity_vectors_2=None,
    position_vectors_1=None, position_vectors_2=None, periodic_length=None,
    scalefactor=None, hubble_time=None,
    total_velocity=False):
    '''
    Get relative velocity[s] [km / s] between input velocity vectors.
    If input positions as well, add Hubble flow to velocities.

    Parameters
    ----------
    velocity_vectors_1 : array : velocity[s] (object number x dimension number) [km / s]
    velocity_vectors_2 : array : velocity[s] (object number x dimension number) [km / s]
    position_vectors_1 : array : position[s] associated with velocity_vector_1
        (object number x dimension number) [kpc comoving]
    position_vectors_2 : array : position[s] associated with velocity_vector_2
        (object number x dimension number) [kpc comoving]
    periodic_length : float : periodicity length [kpc comoving]
    scalefactor : float : expansion scale-factor
    hubble_time : float : 1 / H(z) [Gyr]
    total_velocity : boolean : whether to compute total/scalar (instead of vector) velocity

    Returns
    -------
    velocity_difs  : array (object number x dimension number, or object number) :
        velocity differences [km / s]
    '''
    if np.ndim(velocity_vectors_1) == 1 and np.ndim(velocity_vectors_1) == 1:
        dimension_shape = 0
    else:
        dimension_shape = 1

    velocity_difs = velocity_vectors_1 - velocity_vectors_2  # [km / s]

    if position_vectors_1 is not None and position_vectors_2 is not None:
        # add hubble flow: dr/dt = a * dx/dt + da/dt * x = a(t) * dx/dt + r * H(t)
        # [kpc / Gyr]
        vels_hubble = (scalefactor / hubble_time *
                       get_distances(position_vectors_1, position_vectors_2, periodic_length))
        vels_hubble *= config.km_per_kpc / config.sec_per_Gyr  # [km / s]
        velocity_difs += vels_hubble

    if total_velocity:
        velocity_difs = np.sqrt(np.sum(velocity_difs ** 2, dimension_shape))

    return velocity_difs


#===================================================================================================
# center of mass: position and velocity
#===================================================================================================
def get_center_position_zoom(
    positions, weights=None, periodic_length=None, position_number_min=32, center_position=None,
    distance_max=np.Inf):
    '''
    Get position of center of mass, using iterative zoom-in.

    Parameters
    ----------
    positions : array (particle number x dimension number) : position[s]
    weights : array : weight for each position (usually mass) - if None, assume all have same weight
    periodic_length : float : periodic box length
    position_number_min : int : minimum number of positions within distance to keep zooming in
    center_position : array : initial center position to use
    distance_max : float : maximum distance to consider initially

    Returns
    -------
    center_position : array : position vector of center of mass
    '''
    distance_bins = np.array([
        np.Inf, 1000, 700, 500, 300, 200, 150, 100,
        70, 50, 30, 20, 15, 10,
        7, 5, 3, 2, 1.5, 1,
        0.7, 0.5, 0.3, 0.2, 0.15, 0.1,
        0.07, 0.05, 0.03, 0.02, 0.015, 0.01,
        0.007, 0.005, 0.003, 0.002, 0.0015, 0.001,
    ])
    distance_bins = distance_bins[distance_bins <= distance_max]

    if weights is not None:
        assert positions.shape[0] == weights.size
        # normalizing weights by median seems to improve numerical stability
        weights = np.asarray(weights) / np.median(weights)

    if center_position is None or not len(center_position):
        center_position = np.zeros(positions.shape[1], positions.dtype)
    else:
        center_position = np.array(center_position, positions.dtype)

    if positions.shape[0] > 2147483647:
        idtype = np.int64
    else:
        idtype = np.int32
    part_indices = np.arange(positions.shape[0], dtype=idtype)

    for dist_i, dist_max in enumerate(distance_bins):
        ## direct method ----------
        distance2s = get_position_differences(
            positions[part_indices] - center_position, periodic_length) ** 2
        distance2s = np.sum(distance2s, 1)

        # get particles within distance max
        masks = (distance2s < dist_max ** 2)
        part_indices_dist = part_indices[masks]

        # store particles slightly beyond distance max for next interation
        masks = (distance2s < (1.5 * dist_max) ** 2)
        part_indices = part_indices[masks]

        """
        ## kd-tree method ----------
        if dist_i == 0:
            # does not handle periodic boundaries, but should be ok for zoom-in
            KDTree = spatial.cKDTree(positions)
            particle_number_max = positions.shape[0]

        distances, indices = KDTree.query(
            center_position, particle_number_max, distance_upper_bound=dist_max)

        masks = (distances < dist_max)
        part_indices_dist = indices[masks]
        particle_number_max = part_indices_dist.size
        """

        # check whether reached minimum total number of particles within distance
        # but force at least one loop over distance bins to get *a* center
        if part_indices_dist.size <= position_number_min and dist_i > 0:
            return center_position

        if weights is None:
            weights_use = weights
        else:
            weights_use = weights[part_indices_dist]

        # ensure that np.average uses 64-bit internally for accuracy, but returns as input dtype
        center_position = np.average(
            positions[part_indices_dist].astype(np.float64), 0, weights_use).astype(positions.dtype)

    return center_position


def get_center_velocity(
    velocities, weights=None, positions=None, center_position=None, distance_max=20,
    periodic_length=None):
    '''
    Get velocity of center of mass.
    If no input masses, assume all masses are the same.

    Parameters
    ----------
    velocities : array (particle number x 3) : velocity[s]
    weights : array : weight for each position (usually mass) - if None, assume all have same weight
    positions : array : positions (particle number x dimension number), if want to select by this
    center_position : array : center position, if want to select by this
    distance_max : float : maximum position difference from center to use particles
    periodic_length : float : periodic box length

    Returns
    -------
    center_velocity : array : velocity vector of center of mass
    '''
    masks = np.full(velocities.shape[0], True)

    # ensure that use only finite values
    for dimen_i in range(velocities.shape[1]):
        masks *= np.isfinite(velocities[:, dimen_i])

    if positions is not None and center_position is not None and len(center_position):
        assert velocities.shape == positions.shape
        distance2s = np.sum(
            get_position_differences(positions - center_position, periodic_length) ** 2, 1)
        masks *= (distance2s < distance_max ** 2)

    if weights is not None:
        assert velocities.shape[0] == weights.size
        # normalizing weights by median seems to improve numerical stability
        weights = weights[masks] / np.median(weights[masks])

    # ensure that np.average uses 64-bit internally for accuracy, but returns as input dtype
    return np.average(velocities[masks].astype(np.float64), 0, weights).astype(velocities.dtype)



def rotation_matrix_from_vectors(vec1, vec2):
    '''
    Find the rotation matrix that aligns vec1 to vec2

    Parameters
    ----------
    vec1: A 3d "source" vector
    vec2: A 3d "destination" vector

    Returns
    -------
    mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    '''
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def orientated_coords(coords, center, normal_vector, previous_normal=[0,0,1.]):
    '''
    Orientates coordinates to a new origin and aligns z axis with given normal vector.

    Parameters
    ----------
    coords: coordinates to be orientated
    center: new origin for given coordinates
    normal_vector: new direction of normal axis
    previous_normal: previous normal axis (assumed to be z-axis)

    Returns
    -------
    new_coords: New coordinates that have been orientated
    '''
    RM = rotation_matrix_from_vectors(previous_normal,normal_vector)
    # rotate coordinates so z-axis is aligned with given normal vector
    new_coords = RM.dot(coords)
    # Now move to new center
    new_coords=new_coords+center
    return new_coords

def normalize_vec(vector):
    '''
    Normalizes a given vector,

    Parameters
    ----------
    vector: vector to be normalized
    Returns
    -------
    norm_vec: normalized vector
    '''

    magnitude = np.sqrt(np.sum(vector**2,axis=0))
    print(magnitude)
    norm_vec = vector/magnitude
    return norm_vec