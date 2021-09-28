import math as m

import numpy as np
from sklearn.decomposition import SparsePCA

from . import basic_functions as bf


def point_in_hypercube(vector, hc_min, hc_max):
    """
    Test whether vector is inside the hypercube bounds defined by hc_min and hc_max.
    Returns True if inside.
    """
    outside = np.ones(vector.shape[0])
    for i, val in enumerate(vector):
        if hc_min[i] <= val <= hc_max[i]:
            # flip bit for inside
            outside[i] = 0
    return not np.any(outside)


def point_in_pcylinder(vector, center, radius, compact_dims, compact_ranges):
    """
    Test whether a vector is inside the p-cylinder boundary defined by
    center, radius, compact_dims, and compact_ranges.
    Returns a bool 'inside' and a float 'distance' which is 0.0 if the vector
    is inside the boundary, otherwise the distance from the boundary.
    """
    # Must account for cases where q=0 (all compact dimensions) or q=p (none)
    all_compact = np.all(compact_dims)
    none_compact = np.all(np.logical_not(compact_dims))
    normal_inside = False
    sphere_inside = False
    centered_vector = vector - center
    sphere_components = centered_vector[np.logical_not(compact_dims)]
    distance_from_sphere_center = np.linalg.norm(sphere_components)
    distance_from_sphere_surface =  distance_from_sphere_center - radius
    if (distance_from_sphere_surface < 0) and (not all_compact):
        sphere_inside = True
    normal_components = centered_vector[compact_dims]
    normal_min = compact_ranges[0, compact_dims]
    normal_max = compact_ranges[1, compact_dims]
    normal_distances = np.zeros(normal_min.shape[0])
    # This loop does not execute if there are no compact components
    for i, val in enumerate(normal_components):
        if normal_min[i] <= val <= normal_max[i]:
            normal_distances[i] = 0.0
        else:
            if val < normal_min[i]:
                distance = normal_min[i] - val
            else:
                distance = val - normal_max[i]
            normal_distances[i] = distance
    # True if all compact distances are zero AND if shape is not [0,]
    normal_inside = (not np.any(normal_distances)) and (not none_compact)
    # Can only execute if there is at least one compact dimension
    if not normal_inside:
        normal_distance = np.min(normal_distances[np.where(normal_distances > 0)[0]])
    else:
        normal_distance = 0.0
    # Case where there is no spherical component
    if all_compact:
        return normal_inside, normal_distance
    # Case where there is no compact component
    elif none_compact:
        return sphere_inside, max(0.0, distance_from_sphere_surface)
    # More typical mixed case
    else:
        # Point is inside both geometries
        if sphere_inside and normal_inside:
            return True, 0.0
        # Point is outside one or both of the geometries
        else:
            if sphere_inside:
                distance_outside_sphere = 0.0
            else:
                distance_outside_sphere = distance_from_sphere_surface
            # either of these might be zero but normally not both
            distance_from_pcylinder_surface = max(normal_distance, distance_outside_sphere)
            # legacy from debugging
            #x1 = [normal_distance, distance_outside_sphere]
            #x2 = [d for d in x1 if d > 0.0]
            #if len(x2) > 0:
            #    distance_from_pcylinder_surface = min(x2)
            #else:
            #    print('Something broke', x1, vector, center, radius, compact_dims, compact_ranges)
            #    print('Something broke', normal_distances) 
            return False, distance_from_pcylinder_surface


class PSphere:
    """
    The PSphere class contains cluster data and the attributes of the p-sphere
    and p-cylinder containing the data. By default it calculates both shapes.
    The p-sphere shape is just represented as a radius and a center.
    The p-cylinder shape is represented as a radius and center for the
    dimensions retained by SparsePCA, and as a set of upper and lower bounds
    for the compact dimensions. The arrays compact_dims and compact_ranges are
    used to keep track of which dimensions are compact.

    Attributes:
    ----------
    used: bool, the p-sphere is used in the domain model (i.e., the p-sphere hull)
    redundant: bool, the p-sphere contains no unique data
    label: int, the label of the cluster used to generate the p-sphere
    pdim: int, the number of dimensions in the dataset
    ndata: int, the number of data points in the cluster
    local_data: the subset of data belonging to the cluster, expressed in local
        coordinates (the origin is the cluster center)
    center: float, the cluster center
    radius: float, the radius of the p-sphere
    volume: float, the volume of the p-sphere
    localdim: int, the dimensionality of the local manifold as determined by PCA
    pczero: [float], first principal component vector of local_data
    pcylinder: boolean, the p-cylinder shape is intialized
    qsphere_radius: float, the radius of the q-sphere part of a p-cylinder
    qsphere_volume: float, the volume of the q-sphere part of a p-cylinder
    sparse_basis: [[float]], array of localdim x p sparse PCA components
    compact_dims: [bool], p flags inidicating which dimensions are compact
    compact_ranges: [[float]]: 2 x p values indicating the lower [0] and
        upper [1] bounds of the compact dimensions
    compact_vol: float, the hypercuboid volume of the compact dimensions
    pcylinder_vol: float, qsphere_vol x compact_vol

    Methods:
    -------
    set_used(x: bool) : setter for used attribute
    set_redundant(x: bool) : setter for redundant attribute
    distance(vector: [float]) : returns the distance from the p-sphere boundary.
        This is a basic function, you should usually use contains() or 
        pcylinder_contains() to obtain distances.
    make_localdim() : runs PCA, then sets the localdim and pczero attributes
    contains(vector: [float]) : returns a tuple (bool, float). The bool is
        True if the vector is inside the p-sphere, the float is the same as
        the value returned by distance(vector)
    make_sparse_basis() : runs sparse PCA on the data to identify compact
        dimensions. Sets several attributes needed for make_pcylinder()
    make_pcylinder() : completes the calculation of p-cylinder attributes.
    pcylinder_contains(vector) : returns a tuple (bool, float) with the same
        meaning as contains(), but using the pcylinder boundary.

    Note:
    ----
    A single PSphere object contains the dataset of a cluster in its local
    coordinates. However, distance(), contains() and pcylinder_contains() take
    vectors expressed in the original coordinates.
    """

    def __init__(self, data, center, label, pdim, use_pcylinders=True):
        self.used = True
        self.redundant = False
        self.center = center
        self.label = label
        self.pdim = pdim
        self.ndata = data.shape[0]
        self.local_data = bf.get_translated_vectors(data, center)
        self.radius = bf.radius_from_centered_data(self.local_data)
        self.volume = bf.psphere_volume(self.radius, pdim)
        # variables that are only calculated when requested by a method call
        self.localdim = 0
        self.pczero = np.zeros(pdim, dtype="float64")
        self.sparse_basis = None
        self.compact_dims = None
        # compact ranges expressed in local coordinates
        self.compact_ranges = None
        self.pcylinder = False
        self.pcylinder_vol = 0.0
        self.qsphere_radius = 0.0
        self.qsphere_vol = 0.0
        self.compact_vol = 0.0
        self.make_localdim()
        if use_pcylinders:
            self.make_sparse_basis()
            self.make_pcylinder()
        return

    def set_used(self, state=True):
        """
        Setter for used attribute.
        """
        self.used = state
        return

    def set_redundant(self, state=True):
        """
        Setter for redundant attribute.
        """
        self.redundant = state
        return

    def distance(self, new_vector):
        """
        Calculates the distance from a new row vector to the edge of the sphere.
        Returns 0. if the vector is inside.
        """
        dist_from_center = np.linalg.norm(new_vector - self.center)
        return max(dist_from_center - self.radius, 0.0)

    def make_localdim(self, pct_threshold=0.9):
        """
        Uses PCA to calculate the local dimensionality of the points in the p-sphere.

        Arguments:
        ---------
        pct_threshold : float, The fraction of variance that must be explained by
        the first Q principal components to conclude that the local manifold
        has dimension Q.
        """
        if self.localdim == 0:
            self.localdim, self.pczero = bf.cluster_local_dimensions(
                self.local_data, pct_threshold
            )
        return

    def contains(self, new_vector):
        """
        Boolean test whether a new row vector is inside the p-sphere.
        """
        # new_vector should have shape (p,) either a 1d nparray or a 1d slice
        # of a 2d nparray. However, it must be in the form of an nparray.
        inside = False
        dist = 0.0
        if new_vector.shape[0] != self.pdim:
            print("PSphere.contains(): vector to test has incompatible dimension")
            return inside, dist
        else:
            dist = self.distance(new_vector)
            if dist == 0.0:
                inside = True
        return inside, dist

    def make_sparse_basis(self, n_components=0):
        """
        Run SparsePCA to find a number of principal components equal to n_components
        or to localdim (if n_components==0). Sets the attribute compact_dims, an array
        of boolean flags indicating which dimensions are unused in the sparse basis.

        Example: for a 6D p-sphere with localdim==2, SparsePCA tries to find 2 sparse
        principal components. If the vectors are [1,0,2,1,0,0] and [0,0,1,3,2,0],
        then the value of compact_dims will be [False, True, False, False, False, True].
        Hence, the method make_pcylinder() will calculate the dimensions of a 4D
        "q-sphere" for the non-compact dimensions and a 2D "hypercuboid" for the
        compact dimensions with indices 1 and 5.
        """
        # n_components = 0 means to use the local dimensionality
        if n_components > self.pdim:
            print("PSphere.make_sparse_basis(): n_components > p is impossible")
            return
        # only one data point in cluster implies no p-cylinder model (sphere volume, r=EPS)
        if self.local_data.shape[0] == 1:
            self.compact_dims = [False] * self.local_data.shape[1]
            return
        # uninitialized
        if self.localdim == 0:
            self.make_localdim()
        if n_components == 0:
            nc = self.localdim
        else:
            nc = n_components
        spca = SparsePCA(n_components=nc)
        spca.fit(self.local_data)
        self.sparse_basis = spca.components_
        # spca.components_ has dimensions (n_components,p)
        # look for columns of spca.components_ where all elements are zero
        self.compact_dims = [
            not np.any(spca.components_[:, i]) for i in range(spca.components_.shape[1])
        ]
        return

    def make_pcylinder(self, min_compact_width=0.01):
        """
        Calculate the dimensions of the p-cylinder that bounds the data. The compact
        dimensions identified by make_sparse_basis() are represented as a hypercuboid,
        while the other dimensions are represented as a "q-sphere" (hypersphere with
        dimension q < p). Setter for pcylinder attributes.

        Arguments:
        ---------
        min_compact_width: float, the minimum width of a compact dimension as a
        percentage of the p-sphere radius (not the q-sphere radius). This must be
        non-zero to prevent the hypercylinder volume from collapsing to zero if
        a dimension has no variation.
        """
        # uninitialized
        if self.compact_dims is None:
            print(
                "PSphere.make_pcylinder(): you must call make_sparse_basis() before calling this method."
            )
            return
        if not any(self.compact_dims):
            # print('PSphere.make_pcylinder(): cluster', self.label, 'has no compact dimensions')
            self.pcylinder_vol = self.volume
            return
        self.pcylinder = True
        EPS = 1.0e-7
        compact_vol = 1.0
        sphere_qdim = self.pdim
        self.compact_ranges = np.zeros((2, self.pdim))
        # get local data min and max values in the compact dimensions
        for idim, compact in enumerate(self.compact_dims):
            if compact:
                dim_min = np.min(self.local_data[:, idim])
                dim_max = np.max(self.local_data[:, idim])
                # if the width of a compact dimension is zero, change it to 0.01*radius
                if dim_min == dim_max:
                    dim_min -= self.radius * min_compact_width * 0.5
                    dim_max += self.radius * min_compact_width * 0.5
                self.compact_ranges[0, idim] = dim_min - EPS
                self.compact_ranges[1, idim] = dim_max + EPS
                compact_vol *= dim_max - dim_min
                sphere_qdim -= 1
        if sphere_qdim >= 1:
            qsphere_data = self.local_data[:, np.logical_not(self.compact_dims)]
            self.qsphere_radius = np.max(np.linalg.norm(qsphere_data, axis=1)) + EPS
            self.qsphere_vol = bf.psphere_volume(self.qsphere_radius, sphere_qdim)
        else:
            self.qsphere_vol = 1.0
        self.compact_vol = compact_vol
        self.pcylinder_vol = self.qsphere_vol * compact_vol
        return

    def pcylinder_contains(self, new_vector):
        """
        Calculates the distance between a new row vector and the boundary of
        a p-cylinder.

        Returns:
        -------
        inside: bool, the vector is inside the boundary.
        distance: flloat, the distance from the vector to the boundary. If zero,
        the vector is inside.
        """
        # new_vector should have shape (p,) either a 1d nparray or a 1d slice
        # of a 2d nparray. However, it must be in the form of an nparray.
        inside = False
        dist = 0.0
        if new_vector.shape[0] != self.pdim:
            print(
                "PSphere.pcylinder_contains(): vector to test has incompatible dimension"
            )
            return inside, dist
        # running the method make_pcylinder() on a sphere with no compact dimensions should
        # not prevent us from using this method to test whether a point is inside.
        if not self.pcylinder:
            return self.contains(new_vector)
        else:
            inside, dist = point_in_pcylinder(
                new_vector,
                self.center,
                self.qsphere_radius,
                self.compact_dims,
                self.compact_ranges,
            )
            return inside, dist


class PSphereHull:
    """
    The PSphereHull class contains a collection of PSphere objects. While the
    PSphere class can be used on its own, the expectation is that the user will
    initialize PSphereHull with the full dataset and its cluster labels and centers,
    so that the constructor can build the full set of PSpheres.

    Attributes:
    ----------
    pdim: int, the number of dimensions in the dataset
    n_spheres: int, the number of clusters and the number of p-spheres
    spheres: [PSphere], a list of p-spheres
    hc_min: [float], the lower bounds of each dimension for the dataset
    hc_max: [float], the upper bounds of each dimension for the dataset
    hc_volume: float, the hypercuboid volume of the dataset
    localdims: [int], the local dimensionalities of the p-spheres
    pcylinders: bool, True means that all PSphere objects have calculated p-cylinders
    naive_vratio: float, sum of p-sphere volumes divided by hc_volume
    sampled_vratio: float, fraction of randomly sampled points inside any p-sphere
    sampled_vratio_sigma: float, sqrt(N) uncertainty on sampled_vratio
    naive_pcylinder_vratio: float, sum of p-cylinder volumes divided by hc_volume
    sampled_pcylinder_vratio: float, fraction of randomly sampled points
        inside any p-cylinder
    sampled_pcylinder_vratio_sigma: float, sqrt(N) uncertainty on
        sampled_pcylinder_vratio

    Methods:
    -------
    make_local_dimensions(pct_threshold: float) : recalculates the localdim attribute
        for all p-spheres and the localdims attribute for the PSphereHull. Note
        that it allows the user to enter a different value for pct_threshold
    contains(new_data: ndarray) : tests a collection of row vectors
        for membership in the PSphereHull, returns a list of booleans.
    find_containing_spheres(vector: ndarray) : tests a row vector for
        membership in every PSphere, returns the labels of containing p-spheres
        as a tuple.
    make_volume_ratios(n_samples: int) : calculates all attributes with the suffix
        vratio. If n_samples=0, it skips the sampled_*_vratio attributes.
    psphere_distances(new_data: ndarray) : tests a collection of row vectors for
        membership in the PSphereHull, returns a n x k ndarray of distances of
        each row vector from the k PSpheres in the hull.
    make_pcylinders() : calculates the p-cylinder dimensions for each p-sphere if
        it hasn't already been done.
    mask_spheres(cids: [int]) : sets the used attribute to False for the spheres
        indicated in cids.
    use_spheres(cids: [int]) : sets the used attribute to True for the spheres
        indicated in cids.
    sphere_is_redundant(cid: int) : for each data point in the cluster cid
        find all its containing p-spheres. Returns a tuple (True/False, n_unique).
    flag_largest_redundant_sphere(max_unique: int): 
        Test all p-spheres currently used by the hull for unique points using 
        sphere_is_redundant(), identify the one with the highest volume having
        no more unique points than the argument.
    flag_redundant_spheres(max_unique: int) : Iteratively run the helper function
        flag_largest_redundant_sphere() on the collection of p-spheres, masking
        one at a time until all remaining p-spheres have unique data. 
    apply_standard_pipeline() : functions to run automatically when the PSphereHull
        is initialized. Set use_pcylinders=True and compute_all=True to calculate
        p-cylinders and mask redundant p-cylinders.
    """

    def __init__(self, data, labels, centers, compute_all=False, use_pcylinders=True):
        if centers.shape[1] != data.shape[1]:
            print(
                "PSphereHull: initialization failed, data and centers have different dimension p"
            )
            return
        elif (set(labels) - {-1}) != set(range(centers.shape[0])):
            print("PSphereHull: initialization failed, label set != range(n_centers)")
            return
        else:
            EPS = 1.0e-7
            self.pdim = centers.shape[1]
            self.n_spheres = centers.shape[0]
            self.spheres = []
            for k in range(centers.shape[0]):
                rows = np.where(labels == k)[0]
                this_data = data[rows, :]
                this_sphere = PSphere(
                    this_data, centers[k], k, self.pdim, use_pcylinders
                )
                self.spheres.append(this_sphere)
            self.hc_min = np.min(data, axis=0) - EPS
            self.hc_max = np.max(data, axis=0) + EPS
            self.hc_volume = np.prod(self.hc_max - self.hc_min)
            self.localdims = [s.localdim for s in self.spheres]
            # p-cylinders exist, but some methods can still use the p-spheres
            self.pcylinders = use_pcylinders
            self.naive_vratio = -1.0
            self.sampled_vratio = -1.0
            self.sampled_vratio_sigma = -1.0
            self.naive_pcylinder_vratio = -1.0
            self.sampled_pcylinder_vratio = -1.0
            self.sampled_pcylinder_vratio_sigma = -1.0
            self.apply_standard_pipeline(compute_all, use_pcylinders)
        return

    # calculate local dimensionality of data within a cluster using PCA method
    # this method permits the user to recalculate the values with a different
    # threshold if desired, without recreating the hull.
    def make_local_dimensions(self, pct_threshold=0.9):
        """
        Calculate and set the PSphere.localdim and PSphereHull.localdims attributes.

        Arguments:
        ---------
        pct_threshold : float, The fraction of variance that must be explained by
        the first Q principal components to conclude that the local manifold
        has dimension Q.
        """
        self.localdims = []
        for sphere in self.spheres:
            sphere.make_localdim(pct_threshold=pct_threshold)
            self.localdims.append(sphere.localdim)
        return

    def contains(self, new_data, use_pcylinders=True):
        """
        For each row vector in new_data, check whether it belongs to any unmasked
        p-sphere of the hull.

        Arguments:
        ---------
        new_data: [[float]], n x p matrix representing a set of row vectors.

        Returns:
        -------
        in_any_psphere: [bool], list of flags indicating which data are contained
        in the hull.
        """
        if use_pcylinders and not self.pcylinders:
            print(
                "PSphereHull.contains(): you must run make_pcylinders() before calling this\
             method with use_pcylinders=True"
            )
            return
        # protect against case of new_data being a single 1d vector
        if new_data.ndim == 1:
            print(
                "PSphereHull.contains(): expected (n,p) 2d ndarray, reshaping vector to (1,p)"
            )
            p = len(new_data)
            data = new_data.reshape((1, p))
        else:
            data = new_data
        in_any_psphere = np.zeros(data.shape[0], dtype="bool")
        for ivec in range(data.shape[0]):
            # assign False if the vector is outside the data hypercube
            if not point_in_hypercube(data[ivec, :], self.hc_min, self.hc_max):
                in_any_psphere[ivec] = False
            # otherwise test p-spheres and stop if one contains the vector
            else:
                inside = False
                isph = 0
                while (not inside) and (isph < self.n_spheres):
                    if self.spheres[isph].used:
                        if use_pcylinders:
                            inside, dist = self.spheres[isph].pcylinder_contains(
                                data[ivec, :]
                            )
                        else:
                            inside, dist = self.spheres[isph].contains(data[ivec, :])
                    isph += 1
                in_any_psphere[ivec] = inside
        return in_any_psphere

    def find_containing_spheres(self, vector, use_pcylinders=True):
        """
        For a single new data point, check its membership for all unmasked p-spheres
        in the hull.
        Arguments:
        ---------
        vector: [float], a row vector representing the new data point.

        Returns:
        -------
        containing_spheres: (int), tuple of the containing p-sphere labels.
        """
        # vector is a single 1d vector
        containing_spheres = []
        if use_pcylinders and not self.pcylinders:
            print(
                "PSphereHull.find_containing_spheres(): you must run make_pcylinders() before calling this\
             method with use_pcylinders=True"
            )
            return containing_spheres
#        if not point_in_hypercube(vector, self.hc_min, self.hc_max):
#            print("PSPhereHull.find_containing_spheres(): point outside hypercube.")
#            return containing_spheres
        for sphere in [s for s in self.spheres if s.used]:
            inside = False
            if use_pcylinders:
                inside, dist = sphere.pcylinder_contains(vector)
            else:
                inside, dist = sphere.contains(vector)
            if inside:
                containing_spheres.append(sphere.label)
        return tuple(containing_spheres)

    def make_volume_ratios(self, n_sample=0, use_pcylinders=True):
        """
        Calculate and set the *_vratio attributes of the hull.

        Arguments:
        ---------
        n_sample: int, the number of randomly generated points to estimate the hull
        hypervolume directly. If 0, only the 'naive' hypervolumes will be calculated,
        so the result does not take into account overlapping p-spheres.
        """
        if use_pcylinders and not self.pcylinders:
            return
        # naive because it doesn't take into account overlapping spheres
        if use_pcylinders:
            volumes = [sphere.pcylinder_vol for sphere in self.spheres if sphere.used]
            self.naive_pcylinder_vratio = np.sum(volumes) / self.hc_volume
        else:
            volumes = [sphere.volume for sphere in self.spheres if sphere.used]
            self.naive_vratio = np.sum(volumes) / self.hc_volume
        # estimate the volume ratio by random sampling with uncertainty
        if n_sample > 0:
            # Generate random samples and rescale to dataset hypercube
            sample = np.random.random_sample((n_sample, self.pdim))
            for ip in range(self.pdim):
                scale = self.hc_max[ip] - self.hc_min[ip]
                sample[:, ip] = (sample[:, ip] * scale) + self.hc_min[ip]
            # Test each sample for membership in the hull
            n_inside = len(np.where(self.contains(sample, use_pcylinders))[0])
            if n_inside > 0:
                if use_pcylinders:
                    self.sampled_pcylinder_vratio = n_inside / n_sample
                    self.sampled_pcylinder_vratio_sigma = m.sqrt(n_inside) / n_sample
                else:
                    self.sampled_vratio = n_inside / n_sample
                    self.sampled_vratio_sigma = m.sqrt(n_inside) / n_sample
            else:
                # signals no sample points in volume
                if use_pcylinders:
                    self.sampled_pcylinder_vratio = 0.0
                    self.sampled_pcylinder_vratio_sigma = 0.0
                else:
                    self.sampled_vratio = 0.0
                    self.sampled_vratio_sigma = 0.0
        return

    # return an array of distances from all psphere surfaces (0 if in a psphere, never negative)
    # computationally more expensive than contains, but it can be used for a wider variety of
    # purposes: for example, counting the number of containing pspheres and calculating the
    # minimum distance.
    def psphere_distances(self, new_data, use_pcylinders=True):
        """
        Calculate a 2D array of distances from a new set of data points to all
        unmasked p-spheres in the hull.

        Arguments:
        ---------
        new_data: [[float]], n x p matrix of row vectors representing n new data.

        Returns:
        distances [[float]], n x k matrix of distances from each row vector to the
        k unmasked p-spheres in the hull. A value of 0 means that the data point is
        inside the p-sphere.
        -------
        """
        distances = np.zeros(new_data.shape[0] * self.n_spheres, dtype="float").reshape(
            new_data.shape[0], self.n_spheres
        )
        for ivec in range(new_data.shape[0]):
            for isphere, sphere in enumerate(self.spheres):
                if not use_pcylinders:
                    distances[ivec, isphere] = sphere.distance(new_data[ivec, :])
                else:
                    inside, dist = sphere.pcylinder_contains(new_data[ivec, :])
                    distances[ivec, isphere] = dist
        return distances

    def make_pcylinders(self, n_components=0, min_compact_width=0.01):
        """
        Set for all p-spheres in the hull, call the methods to create a sparse
        local basis (using SparsePCA) and define the dimensions of the
        p-cylinder.

        Arguments:
        ---------
        n_components: int, a key argument for SparsePCA. The default value
        zero tells sphere.make_sparse_basis to set this value to the local
        dimensionality of the sphere. However, we leave open the possibility
        to ask SparsePCA to calculate the same number of sparse principal
        components for all p-spheres.
        min_compact_width: float, defines the minimum size of a compact
        dimension as a percentage of the p-sphere radius.
        """
        if self.pcylinders:
            return
        for sphere in self.spheres:
            self.pcylinders = True
            sphere.make_sparse_basis(n_components=n_components)
            sphere.make_pcylinder(min_compact_width=min_compact_width)
        return

    def mask_spheres(self, cluster_indices):
        """
        Set the 'used' attribute to False for a list of p-spheres.
        """
        for cid in cluster_indices:
            self.spheres[cid].set_used(False)
        return

    def use_spheres(self, cluster_indices):
        """
        Set the 'used' attribute to True for a list of p-spheres.
        """
        for cid in cluster_indices:
            self.spheres[cid].set_used(True)
        return

    def sphere_is_redundant(self, cid, use_pcylinders=True):
        """
        Helper function for flag_redundant_spheres()
        Test each data point in a p-sphere for membership in all other p-spheres.
        Note that most spheres do not have any unique points.

        Returns:
        -------
        success : bool, the function did not abort early
        
        n_unique: int, The number of data points contained in no other p-spheres.
        """
        if use_pcylinders and not self.pcylinders:
            print("PSphereHull.sphere_is_redundant(): call make_pcylinders first")
            return False, 0
        if cid not in [s.label for s in self.spheres if s.used]:
            print("PSphereHull.sphere_is_redundant(): cid absent or masked")
            return False, 0
        sphere = self.spheres[cid]
        # loop over data in sphere
        n_unique = 0
        for vector in sphere.local_data:
            ctup = self.find_containing_spheres(vector + sphere.center, use_pcylinders)
            # if the only p-sphere containing this point is itself
            if (len(ctup) == 1):
                n_unique += 1
        if n_unique == 0:
            return True, n_unique
        else:
            return False, n_unique
        
    def flag_largest_redundant_sphere(self, max_unique=0, use_pcylinders=True):
        """
        Helper function for flag_redundant_spheres().
        This function assumes that some of the spheres tested might already be masked.
        This function is a bit time-consuming! It scales as n * k 
        """
        # order from largest to smallest
        used_spheres = [s.label for s in self.spheres if s.used]
        if use_pcylinders:
            used_volumes = [self.spheres[i].pcylinder_vol for i in used_spheres]
            size_order = np.array(used_volumes).argsort()[::-1]
        else:
            used_volumes = [self.spheres[i].volume for i in used_spheres]
            size_order = np.array(used_volumes).argsort()[::-1]
        sphere_candidates = []
        sphere_nunique = []
        for idx in size_order:
            sphere_label = used_spheres[idx]
            success, n_unique = self.sphere_is_redundant(sphere_label, use_pcylinders)
            if n_unique <= max_unique:
                sphere_candidates.append(sphere_label)
                sphere_nunique.append(n_unique)
        #print('flag_largest_redundant_sphere, no. candidates:', len(sphere_candidates))
        if len(sphere_candidates) > 0:
            return True, sphere_candidates[0]
        else:
            return False, -1

    def flag_redundant_spheres(self, max_unique=0, use_pcylinders=True, reset_masks=True, apply_masks=True):
        """
        The goal of this function is to calculate the number of unique points in each p-sphere,
        then mask the single sparsest sphere with zero unique points. The process is repeated
        iteratively until all remaining p-spheres have unique points.
        
        It scales as n * k^2 : each call to flag_largest_redundant_sphere scalas as n * k, and
        removes one sphere from the hull. We cannot remove them all at once, however, because
        masking one sphere can change the number of unique points for all the others.
        
        Arguments:
        ---------
        max_unique: int, A p-sphere is flagged as redundant if the number
        of unique points in the p-sphere is less than or equal to this value.
        use_pcylinders: bool, when testing for unique points, use the p-cylinder geometry.
        reset_masks: bool, set all p-spheres back to 'Used' before running the function.
        apply_masks: bool, apply mask to all p-spheres identified as redundant. (If False, the
        function still returns a list of redundant p-spheres)

        Returns:
        -------
        spheres_list [int], a list of p-sphere IDs with no unique data points.
        """
        if use_pcylinders and not self.pcylinders:
            print("PSphereHull.flag_redundant_spheres(): call make_pcylinders first")
            return []
        if max_unique < 0:
            print("PSphereHull.flag_redundant_spheres(): max_unique must be >= 0")
            return []
        # reset used flags if asked
        if reset_masks:
            self.use_spheres([s.label for s in self.spheres])
        spheres_to_mask = []
        success = True
        # mask spheres one by one
        while success:
            success, cid = self.flag_largest_redundant_sphere(max_unique, use_pcylinders)
            if success:
                self.mask_spheres([cid])
                spheres_to_mask.append(cid)
        # reset masks if asked
        if not apply_masks:
            self.use_spheres([s.label for s in self.spheres])
        return spheres_to_mask

    def apply_standard_pipeline(self, compute_all, pc):
        self.make_local_dimensions()
        # p-sphere volume ratios
        if compute_all:
            self.make_volume_ratios(n_sample=1000, use_pcylinders=False)
        # p-cylinder volume ratios
        if pc:
            self.make_pcylinders()
        # p-cylinder volume ratios
        if compute_all and pc:
            self.make_volume_ratios(n_sample=1000, use_pcylinders=pc)
        # redundancy mask with p-spheres or p-cylinders
        if compute_all:
            self.flag_redundant_spheres()

# Functions below this line are deprecated.
                      
    def sphere_is_redundant_old(self, cid, use_pcylinders=True):
        """
        UNUSED, because this approach is too complex and supports a workflow
        that tends to flag small, compact p-spheres as redundant because they are
        entirely overlapped by a large, sparse p-sphere. We want to prefer keeping
        the small, compact p-spheres.
        
        Test each data point in a p-sphere for membership in all other p-spheres.
        If any data point is unique, return False (the p-sphere is not redundant).
        If all the data points are contained in other spheres, return True. If
        any of the other spheres contain all the data points, then also return
        a list of their IDs.
        Returns:
        -------
        is_redundant: bool, The p-sphere labeled cid is redundant, because all
        of its data points are also found in other p-spheres.
        superset_list: [int], list of p-spheres that are supersets of cid,
        meaning they contain all the data points of cid.
        """
        if use_pcylinders and not self.pcylinders:
            print("PSphereHull.sphere_is_redundant(): call make_pcylinders first")
            return False
        if cid not in [s.label for s in self.spheres]:
            print("PSphereHull.sphere_is_redundant(): cid not in sphere labels")
            return False
        sphere = self.spheres[cid]
        # initial set is the list of other spheres containing first data point
        vector = sphere.local_data[0, :] + sphere.center
        ctup = self.find_containing_spheres(vector, use_pcylinders)
        supersets = set(ctup)
        irow = 1
        # terminate loop early if we find a point whose supersets = {cid}
        while (ctup != tuple([cid])) and (irow < sphere.ndata):
            vector = sphere.local_data[irow, :] + sphere.center
            ctup = self.find_containing_spheres(vector, use_pcylinders)
            # set intersect_update operator removes any p-spheres that are
            # not a superset of the cluster data.
            supersets &= set(ctup)
            irow += 1
        # exit condition 1 is ctup == tuple(cid), so no superset is possible
        if ctup == tuple([cid]):
            return False, []
        # exit condition 2 is we ran out of data without ever finding a
        # unique point. Then there are two possibilities:
        else:
            # the cluster has one or more supersets
            if len(supersets) > 1:
                superset_list = list(supersets)
                if cid in superset_list:
                    superset_list.remove(cid)
                return True, superset_list
            # the cluster is redundant but has no supersets
            else:
                return True, []

    def flag_redundant_spheres_old(self, use_pcylinders=True, apply_mask=True):
        """
        UNUSED because this workflow masks small, compact p-spheres overlapped
        by large, sparse p-spheres. (It calls sphere_is_redundant_old)
        
        For each p-sphere in the hull, call sphere_is_redundant() to see if it
        has any data points that are unique to itself. If redundant, then set
        PSphere.redundant=True. If asked, also apply the flag PSphere.used=False.
        The spheres are tested and masked from smallest to largest (in terms of
        the number of data points).

        Note: A simple example is that it is possible for a point to belong
        to only two spheres, but masking one of the spheres would then make
        the other sphere non-redundant. In such cases, if apply_mask=True,
        we do not want to mask both spheres. However, both spheres
        are still flagged as redundant.

        Returns:
        -------
        spheres_list [int], a list of p-sphere IDs with no unique data points.
        """
        # reset used flags
        self.use_spheres([s.label for s in self.spheres])
        size_order = np.array([s.label for s in self.spheres]).argsort()
        spheres_to_mask = []
        for idx in size_order:
            sphere_id = self.spheres[idx].label
            redundant, supersets = self.sphere_is_redundant_old(sphere_id, use_pcylinders)
            if redundant:
                spheres_to_mask.append(sphere_id)
                self.spheres[sphere_id].set_redundant(True)
                if apply_mask:
                    self.mask_spheres([sphere_id])
        return spheres_to_mask
