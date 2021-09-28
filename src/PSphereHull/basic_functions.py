import math as m

import numpy as np
from sklearn.decomposition import PCA

"""
This module contains low-level functions to calculate volumes and detect
the dimensionality of the local manifold in a cluster by PCA analysis.
"""


def psphere_volume(radius, pdim):
    """
    Purpose
    -------
    Calculate the volume of a perfect p-sphere with the given radius and
    dimensionality. Standard mathematical formula.

    Arguments
    ---------
    radius : sphere radius
    pdim : dimensionality of the p-sphere, e.g. 3 for a normal sphere

    Returns
    -------
    float : The p-sphere volume (arbitrary units)
    """
    coeff = (m.pi ** (pdim / 2)) / m.gamma(pdim / 2 + 1)
    return coeff * (radius ** pdim)


def cluster_hcvolume(vector_set):
    """
    Purpose
    -------
    Calculate the volume of the hypercuboid defined by the (min,max) range
    of each feature in a dataset. Meant to be used on the small datasets of
    individual clusters, and only for estimating their relative volumes.

    Notes
    -----
    The volume is not permitted to be zero just because one feature has zero
    range. Each dimension is increased by a margin equal to either 1%
    of the side or 0.01, whichever is larger. (The data are expected
    to be scaled to sigma, so typical side lengths of the whole dataset
    should be on the order of 1-10)
    Another, perhaps more sensible option would be to set the minimum side
    length to 0.01 of the range in that feature for the whole dataset, but
    that global information is not available from the perspective of a single
    cluster.

    Arguments
    ---------
    vector_set : 2D nparray, the result of running get_translated_vectors()

    Returns
    -------
    float : The hypercuboid volume (arbitrary units)

    """
    EPS = 1.0e-7
    hc_min = np.min(vector_set, axis=0) - EPS
    hc_max = np.max(vector_set, axis=0) + EPS
    # no side lengths are zero but some may be single-precision EPS
    deltas = hc_max - hc_min
    # large side lengths increase by 1%, all sides increase by at least 0.01.
    margins = np.array([max(0.01 * d, 0.01) for d in deltas])
    return np.prod(deltas + margins)


# For a set of cluster data, find the sphere that wraps its data
def radius_from_centered_data(centered_data):
    """
    Purpose
    -------
    Calculate the maximum distance of any vector from the cluster center.
    If centered_data contains a single point the radius is EPS.

    Arguments
    ---------
    centered_data : 2D nparray, the result of running get_translated_vectors()

    Returns
    -------
    float : The largest absolute value of the centered vector set

    """
    EPS = 1.0e-7
    max_dist = np.max(np.linalg.norm(centered_data, axis=1)) + EPS
    return max_dist


def get_translated_vectors(vector_set, new_origin):
    """
    Purpose
    -------
    Transform a set of data vectors (dimensions n,p) to the new frame of
    reference defined by new_origin (dimension p).

    Arguments
    ---------
    vector_set : 2D nparray
    new_origin : 1D nparray

    Returns
    -------
    The translated vector set (dimensions n,p)

    """
    translated_vector_set = np.zeros(vector_set.shape, dtype=np.float64)
    for i in range(vector_set.shape[0]):
        translated_vector_set[i] = vector_set[i] - new_origin
    return translated_vector_set


def cluster_is_linear(vector_set, center, pct_threshold):
    """
    Purpose
    -------
    Test if the points in a cluster are arranged approximately on a linear
    manifold, using the PCA method. Returns False if the vector set contains
    a single data point.

    Arguments
    ---------
    vector_set : 2D nparray containing the data points in the cluster
    center : center of the cluster to test
    pct_threshold : float
        The fraction of variance that must be explained by the first
        Q principal components to conclude that the local manifold has
        dimension Q.

    Returns
    -------
    bool : the points in the cluster are approximately linear
    direction : the first PCA component
    """
    cluster_vectors = get_translated_vectors(vector_set, center)
    if cluster_vectors.shape[0] > 1:
        pca_model = PCA(n_components=1)
        pca_model.fit(cluster_vectors)
        explained_variance = pca_model.explained_variance_ratio_[0]
        if explained_variance < pct_threshold:
            return False
        else:
            return True
    else:
        return False


# For each cluster, find the number of principal components that explain pct_threshold of variance.
def cluster_local_dimensions(centered_data, pct_threshold=0.9):
    """
    Purpose
    -------
    Use PCA to estimate the dimensionality of the local manifold within
    the cluster: i.e., the number of principal components needed to explain
    pct_threshold of the variance. Returns 0 if the cluster contains a single
    data point.

    Arguments
    ---------
    centered_data : 2D nparray containing the data in the cluster, translated
    to a coordinate system where the origin is the cluster center.
    pct_threshold : float
        The fraction of variance that must be explained by the first
        Q principal components to conclude that the local manifold has
        dimension Q.

    Returns
    -------
    dim : int
        the dimension Q is the number of principal components needed
    pc0 : nparray containing the first principal component
    """
    dim = 0
    pca = PCA()
    if centered_data.shape[0] > 1:
        pca.fit(centered_data)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        dim = np.where(cumulative_variance >= pct_threshold)[0][0] + 1
        pczero = pca.components_[0, :]
        return dim, pczero
    else:
        return dim, centered_data[0, :]


# Functions below this line are deprecated


def intrinsic_dimensions_from_data_clusters(data, labels, centers, pct_threshold=0.9):
    """
    ***UNUSED in V3, all clusters are tested one at a time
    """
    # option: return basis vectors of manifold
    # another version of this function can do the same with sparse PCA
    dims = np.zeros(centers.shape[0], dtype="int")
    pczero = np.zeros(centers.shape[0] * centers.shape[1], dtype="float").reshape(
        (centers.shape[0], centers.shape[1])
    )
    pca = PCA()
    for k in range(centers.shape[0]):
        cluster_vectors = data[np.where(labels == k)[0]]
        # translate reference frame to cluster center
        centered_vectors = get_translated_vectors(cluster_vectors, centers[k])
        pca.fit(centered_vectors)
        # intrinsic dimensionality is the number of components explaining pct_threshold variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        dims[k] = np.where(cumulative_variance >= pct_threshold)[0][0] + 1
        pczero[k, :] = pca.components_[0, :]
    return dims, pczero


def linearity_from_data_clusters(data, labels, centers, pct_threshold=0.9):
    """
    ***UNUSED in V3, all clusters are tested one at a time
    """
    nclus = centers.shape[0]
    linear_flag = np.zeros(nclus, dtype="bool")
    for k in range(nclus):
        data_rows = np.where(labels == k)[0]
        linear_flag[k] = cluster_is_linear(
            data[data_rows, :], centers[k, :], pct_threshold
        )
    return linear_flag


def radii_from_data_clusters(data, labels, centers):
    """
    ***UNUSED in V3, all clusters are tested one at a time.
    """
    EPS = 1.0e-7
    nclus = centers.shape[0]
    sph_radius = np.zeros(nclus, dtype="float")
    for k in range(nclus):
        data_rows = np.where(labels == k)[0]
        data_vectors = get_translated_vectors(data[data_rows, :], centers[k, :])
        max_dist = np.max(np.linalg.norm(data_vectors, axis=1))
        sph_radius[k] = max_dist + EPS
    return sph_radius


def get_cosine_sims(vector_set):
    """
    Purpose
    -------
    ***UNUSED in V3, but left in for future testing if wanted***

    Calculate cosine similarity between each pair of vectors in a set of
    data vectors (dimensions n,p). Normally this is applied to the vectors
    belonging to a single cluster.

    Arguments
    ---------
    vector_set : 2D nparray

    Returns
    -------
    [float] cosine similarities in range [-1,1]
    """
    nvec = vector_set.shape[0]
    csim_list = []
    norms = np.linalg.norm(vector_set, axis=1)
    for i in range(nvec - 1):
        for j in range(i + 1, nvec):
            csim = np.dot(vector_set[i, :], vector_set[j, :]) / (norms[i] * norms[j])
            csim_list.append(csim)
    return np.array(csim_list)


def test_linearity_by_cosines(vector_set, pct_threshold, degree_threshold):
    """
    Purpose
    -------
    *** UNUSED IN V3, but left in for future testing if wanted ***

    Test the linearity of a data vector set (dimensions n,p) using the cosine
    similarities between each pair of vectors. The criteria for linearity is
    given by pct_threshold and degree_threshold as described in Arguments.
    Normally this is applied to the vectors belonging to a single cluster.

    The problem with this method is that it requires two parameters while
    the PCA method requires only one.

    Arguments
    ---------
    vector_set : 2D nparray
    pct_threshold : float
        The percentage of cosine similarities sufficiently close to +/-1,
        corresponding to a range of +/- degree_threshold around theta = 0
        or theta = 180
    degree_threshold : float
        Expresses the desired range in cosine_similarity around theta = 0
        or theta = 180. (More intuitive in terms of geometry than a window
        around cosine = +/-1)

    Returns
    -------
    bool :
        True if pct_threshold of vector pairs fulfil the criterion.
    """
    cosine_threshold = m.cos(degree_threshold * m.pi / 180)
    vector_norms = np.linalg.norm(vector_set, axis=1)
    outer_vec_idx = np.where(vector_norms > 0.5 * np.max(vector_norms))[0]
    if outer_vec_idx.shape[0] < 3:
        print("test_linearity_by_cosines: not enough outer vectors")
        return False
    else:
        outer_vectors = vector_set[outer_vec_idx, :]
        cos_sims = get_cosine_sims(outer_vectors)
        n_low = len(np.where(cos_sims > cosine_threshold)[0])
        n_high = len(np.where(cos_sims < -cosine_threshold)[0])
        # We must find both low (same side pairs) and high (opposite side pairs)
        if (n_low == 0) or (n_high == 0):
            return False
        else:
            # Most of the pairs have angles close to pi or zero
            pct_approx_linear = (n_low + n_high) / len(cos_sims)
            if pct_approx_linear < pct_threshold:
                return False
            else:
                return True


# def cluster_is_linear_unused (vector_set, centroid, method, pct_threshold, degree_threshold):
#    """
#    Purpose
#    -------
#    Test if the point in a cluster are arranged approximately on a linear
#    manifold, by either the cosine similarity or the PCA method.
#
#    Arguments
#    ---------
#    vector_set : 2D nparray containing the data points in the cluster
#    centroid : center of the cluster to test
#    method : str
#        'csim' (default) see test_linearity_by_cosines()
#        'pca' see test_linearity_by_pca()
#    pct_threshold : float
#        Major parameter used by both methods, can be interpreted as either
#        the fraction of vector pairs with cosine similarity close to 1 or
#        the fraction of variance explained by the first principal component.
#    degree_threshold : float
#        Expresses the desired range in cosine_similarity around theta = 0
#        or theta = 180. (More intuitive in terms of geometry than a window
#        around cosine = +/-1)
#
#    Returns
#    -------
#    bool : the points in the cluster are approximately linear
#    """
#    cluster_vectors = get_translated_vectors(vector_set, centroid)
#    if method == 'csim':
#        return test_linearity_by_cosines(cluster_vectors, pct_threshold, degree_threshold)
#    elif method == 'pca':
#        return test_linearity_by_pca(cluster_vectors, pct_threshold)
#
# def test_linearity_by_pca(vector_set, pct_threshold):
#    """
#    Purpose
#    -------
#    Test the linearity of a data vector set (dimensions n,p) by running
#    principal component analysis on the data and comparing the explained
#    variance of the first component to pct_threshold.
#
#    Use this function if you only need a flag, if you also want the first
#    principal component then use cluster_local_dimensions()
#
#    Arguments
#    ---------
#    vector_set : 2D nparray
#    pct_threshold : float
#
#    Returns
#    -------
#    bool :
#        True if the first principal component fulfils the criterion.
#    """
#    pca_model = PCA(n_components=1)
#    pca_model.fit(vector_set)
#    explained_variance = pca_model.explained_variance_ratio_[0]
#    if explained_variance < pct_threshold:
#        return False
#    else:
#        return True
