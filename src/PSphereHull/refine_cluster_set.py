""" 
The module refine_cluster_set contains high-level functions to visualize and
modify a specific clustering solution, before using those clusters to
define the domain as a collection of p-cylinders or p-boxes. Most of the
methods take the sample data, cluster labels, and cluster centers as inputs,
and return new versions of the labels and centers as outputs. Any clustering
method can be used to generate the initial list of labels and centers.

User functions:
---------------
flag_outlier(data, labels, centers, cid) : Finds the data point in cluster
cid that is farthest from the center and replaces its label with -1.
fuse_clusters(data, labels, centers, cid1, cid2) : Merges clusters cid1
and cid2 by relabeling all the data points in cid2 with cid1 and 
recalculating the cluster center.
split_cluster(data, labels, centers, cid) : Runs KMeans(2) on the data in
cluster cid1, relabels the points in one subcluster, and updates the list
of cluster centers.
refine_linear_clusters(data, labels, centers, min_size, pct_threshold) :
calculates the local dimensionality of each cluster using PCA, and run
split_cluster() on any cluster whose points are approximately linear
(i.e., the first principal component explains at least pct_threshold 
of the variance). This function recursively checks any newly generated
clusters with size at least min_size.
describe_cluster_pspheres(data, labels, centers, use_pcylinders) : Generates
a pandas dataframe with volume and density statistics on the p-spheres, 
p-cylinders, or hypercubes that bound the clusters.
plot_cluster_pspheres(df, pdim, arg1, arg2, filename)) : Generates a
matplotlib log-log plot of two columns in the dataframe df made by
describe_cluster_pshperes, with markers colored and scaled automatically.
pcylinder_patch(xidx, yidx, center, radius, compact_dims, compact_ranges, 
color) : generates a matplotlib artist patch for a 2D cross-section of a
p-cylinder, which can be either a circle or a rectangle depending on the
dimensions xidx and yidx of the cross-section.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import cluster

from . import basic_functions as bf
from . import psphere_hull as psh


def flag_outlier(data, labels, centers, cid):
    """
    Purpose
    -------
    Given any individual cluster, flag the point farthest from the center
    by changing its label to -1. This point will be ignored when creating
    the p-sphere hull.

    Arguments
    ---------
    data : nparray
        the complete original dataset with dimensions (n,p)
    labels : nparray
        the original cluster labels with dimension n
    centers : nparray
        the original cluster centers with dimensions (k,p)
    cid : int
        the label value indicating which cluster to split
    """
    cluster_idx = np.where(labels == cid)[0]
    vector_set = data[cluster_idx, :]
    centered_vectors = bf.get_translated_vectors(vector_set, centers[cid, :])
    radii = np.linalg.norm(centered_vectors, axis=1)
    outlier_idx = cluster_idx[np.argmax(radii)]
    # copy the labels nparray to return a new version
    new_labels = np.array(labels)
    new_labels[outlier_idx] = -1
    return new_labels


def fuse_clusters(data, labels, centers, cid1, cid2):
    """
    Purpose
    -------
    Given any two clusters, merge them into a single cluster by relabeling
    the data in cid2 with the cluster label of cid1. The cluster center
    is recalculated.
    Note that the centers argument might be a list or might be an nparray.
    Note also that split_cluster and fuse_clusters are not inverse operations!

    Arguments
    ---------
    data : nparray
        the complete original dataset with dimensions (n,p)
    labels : nparray
        the original cluster labels with dimension n
    centers : nparray
        the original cluster centers with dimensions (k,p)
    cid1 : int
        the label value indicating the first cluster to fuse
    cid2 : int
        the label value indicating the second cluster to fuse
    """

    success = False
    if (cid1 not in labels) or (cid2 not in labels):
        print("......fuse_cluster: one or both cluster ids are absent")
        return labels, centers, success
    cluster_idx1 = np.where(labels == cid1)[0]
    cluster_idx2 = np.where(labels == cid2)[0]
    new_cluster_idx = np.concatenate((cluster_idx1, cluster_idx2))
    # collect all points in both clusters, recalculate center
    vector_set = data[new_cluster_idx, :]
    fused_center = np.mean(vector_set, axis=0)
    # copy the arrays so we can modify them without modifying the originals
    new_labels = np.copy(labels)
    new_centers = list(centers)
    # relabel data in cluster 2
    new_labels[cluster_idx2] = cid1
    new_centers[cid1] = fused_center
    new_centers[cid2] = np.array([np.nan, np.nan, np.nan])
    if type(centers) == np.ndarray:
        new_centers = np.array(new_centers)
    success = True
    return new_labels, new_centers, success


def split_cluster(data, labels, centers, cid, min_size=0, verbose=False):
    """
    Purpose
    -------
    Given any cluster, attempt to subdivide it into two
    clusters with KMeans(2). If a split is possible without
    reducing the number of data points in a cluster below p+1, it updates
    labels and centers to represent the new set of clusters. (Specifically,
    some of the data in a split cluster are relabeled with a new cluster ID,
    a new center is appended to the list, and the old center is updated.)

    Note that the centers argument might be a list or might be an nparray.

    Arguments
    ---------
    data : nparray
        the complete original dataset with dimensions (n,p)
    labels : nparray
        the original cluster labels with dimension n
    centers : nparray
        the original cluster centers with dimensions (k,p)
    cid : int
        the label value indicating which cluster to split
    min_size : int
        the default value 0 signals that each subcluster must have at least p+1
        points as usual; otherwise the minimum size is the argument.
    """

    success = False
    if cid not in labels:
        print("......split_cluster: cluster id out of range")
        return labels, centers, success
    cluster_idx = np.where(labels == cid)[0]
    cluster_size = len(cluster_idx)
    if min_size >= cluster_size:
        print("......split_cluster: min_size is larger than cluster to split")
        return labels, centers, success
    elif min_size <= 0:
        cluster_size_floor = data.shape[1] + 1
    else:
        cluster_size_floor = min_size
    n_clusters = len(set(labels) - {-1})
    vector_set = data[cluster_idx, :]
    # copy the arrays so we can modify them without modifying the originals
    new_labels = np.copy(labels)
    new_centers = list(centers)
    km = cluster.KMeans(2)
    km.fit(vector_set)
    # elements of cluster_idx array corresponding to k=1
    cluster_1_idx = np.where(km.labels_ == 1)[0]
    cluster_1_size = len(cluster_1_idx)
    # one or both of the subclusters has fewer than p+1 members
    if (cluster_1_size < cluster_size_floor) or (
        cluster_size - cluster_1_size < cluster_size_floor
    ):
        if verbose:
            print("......cluster", cid, "not split because child size < min_size")
        return labels, centers, success
    else:
        # elements of original labels array corresponding to k=1
        success = True
        new_labels[cluster_idx[cluster_1_idx]] = n_clusters
        new_centers[cid] = km.cluster_centers_[0]
        new_centers.append(km.cluster_centers_[1])
        if type(centers) == np.ndarray:
            new_centers = np.array(new_centers)
        if verbose:
            print("......cluster", cid, "split successfully")
        return new_labels, new_centers, success


def split_linear_cluster(
    data, labels, centers, cid, min_size=0, pct_threshold=0.9, verbose=False
):
    """
    Purpose
    -------
    Given a cluster, test if it is linear, and if so attempt to subdivide
    it into two clusters withe KMeans(2). If a split is possible without
    reducing the number of data points in a cluster below p+1, it updates
    labels and centers to represent the new set of clusters. (Specifically,
    some of the data in a split cluster are relabeled with a new cluster ID,
    a new center is appended to the list, and the old center is updated.)

    Arguments
    ---------
    data : nparray
        the complete original dataset with dimensions (n,p)
    labels : nparray
        the original cluster labels with dimension n
    centers : nparray
        the original cluster centers with dimensions (k,p)
    cid : int
        the label value indicating which cluster to split
    min_size : int
        the default value 0 signals that each subcluster must have at least p+1
        points as usual; otherwise the minimum size is the argument.
    pct_threshold : float
        The fraction of variance that must be explained by the first
        Q principal components to conclude that the local manifold has
        dimension Q.

    Returns
    -------
    labels : nparray
        The set of data labels obtained after splitting cluster cid
        and creating a new cluster with label k. For example, if cid=0
        then some of data with label 0 will be assigned the new label k.
    centers : nparray
        The new set of cluster centers obtained after splitting cluster cid.
    success : bool
        True if the function succeeded in splitting cluster cid.
    """
    success = False
    if min_size <= 0:
        cluster_size_floor = data.shape[1] + 1
    else:
        cluster_size_floor = min_size
    if cid not in labels:
        if verbose:
            print("split_cluster: cluster id is not among labels")
        return labels, centers, success
    else:
        cluster_idx = np.where(labels == cid)[0]
        vector_set = data[cluster_idx, :]
        if bf.cluster_is_linear(vector_set, centers[cid], pct_threshold):
            return split_cluster(
                data, labels, centers, cid, cluster_size_floor, verbose
            )
        else:
            return labels, centers, success


def split_all_linear_clusters(
    data, labels, centers, fixed=None, min_size=0, pct_threshold=0.9, verbose=False
):
    """
    Purpose
    -------
    This function simply looks at each cluster in the current list, of clusters
    and calls split_linear_cluster() for each one without the "fixed" flag.
    It returns the updated data labels, cluster centers, and list of flags.

    Arguments
    ---------
    data : nparray
        the complete dataset with dimensions (n,p), which was originally
        organized into k clusters.
    labels : nparray
        the current set of cluster labels with dimension n, taking values
        in [0..k'-1]
    centers : [nparray]
        the current list of cluster centers (nparray p-vectors), with length k'
    fixed : [int]
        the current list of cluster ids indicating which ones have already been
        tested with split_cluster and left alone, with length k'
    min_size : [int]
        Minimum size of a new cluster after splitting.
    pct_threshold : float
        The fraction of variance that must be explained by the first
        Q principal components to conclude that the local manifold has
        dimension Q.

    Returns
    -------
    labels : nparray
        The new set of data labels
    centers : [nparray]
        The new set of cluster centers
    fixed : [int]
        The new set of cluster ids indicating which ones do not need
        to be checked again.
    """
    if fixed is None:
        fixed = []
    # original number of clusters
    n_clusters = len(centers)
    # Transform the original array to a list so that split_cluster can append
    # a new p-vector if successful.
    clusters_to_test = [i for i in range(n_clusters) if i not in fixed]
    if verbose:
        print("Running split_all_linear_clusters, clusters to test:", clusters_to_test)
    if len(clusters_to_test) > 0:
        for k in clusters_to_test:
            labels, centers, split = split_linear_cluster(
                data, labels, centers, k, min_size, pct_threshold, verbose
            )
            if not split:
                 fixed.append(k)
            else:
                if verbose:
                    print("...created new cluster", k, "with center", centers[-1])
    return labels, centers, fixed


def refine_linear_clusters(
    data, labels, centers, min_size=2, pct_threshold=0.9, verbose=False
):
    """
    Purpose
    -------
    This function is exposed to users. The expectation is that the user has
    already done data exploration, cleaning, and scaling, and has performed
    an initial clustering of the data with a relatively large number
    of clusters, in order to characterize the data's domain as a collection
    of p-spherical subdomains.

    The goal of the function is to subdivide all clusters whose local manifold
    is approximately linear into smaller clusters. A set of p-spherical
    subdomains along a 1-D manifold has a much smaller hypervolume than a
    a single p-sphere around the cluster. Specifically, in the idealized
    case the hypervolume goes down by a huge factor: 1/M^(p-1).

    The function calls split_linear_clusters() several times, updating the list
    of data labels and cluster centers each time, as well as keeping track of
    which clusters could not be split (because their data are not linear or
    they are too small). It continues until no more can be split.

    In short, the main value brought by this function is to look at a standard
    clustering solution, and replace the large p-sphere implied by a large,
    linear cluster with a set of small p-spheres along the line. The data
    domain for the original cluster can then be represented as a "string of
    beads", instead of a large hypersphere with wasted space.

    I could have written this with recursion, but I think the logic would be
    harder to follow.
    
    NOTE: This function was written before the library supported p-cylinders.
    In most cases I expect that the p-cylinders treat this use case better than
    a string of p-spheres. However, this function would probably
    perform better on a linear cluster not aligned with a feature axis.

    Arguments
    ---------
    data : nparray
        the complete original dataset with dimensions (n,p)
    labels : nparray
        the original cluster labels with dimension n, taking values in [0..k-1]
    centers : nparray
        the original cluster centers with dimensions (k,p)
    pct_threshold : float
        The fraction of variance that must be explained by the first
        Q principal components to conclude that the local manifold has
        dimension Q.

    Returns
    -------
    labels : nparray
        The final of data labels.
    centers : nparray
        The final set of cluster centers.
    """
    # Checks for corrupt data or arguments
    if data.shape[0] != len(labels):
        print("refine_linear_clusters: data and labels have incompatible dimensions")
        return labels, centers
    if (centers.shape[0] - 1) != np.max(labels):
        print("refine_linear_clusters: number of centroids is incompatible with label set")
        return labels, centers
    if pct_threshold < 0 or pct_threshold > 1:
        print("refine_linear_clusters: pct_threshold must be between 0 and 1")
        return labels, centers
    # Start processing
    fixed = []
    # change to list of p-vectors for processing
    cc = list(centers)
    ll = np.copy(labels)
    continue_splitting = True
    while continue_splitting:
        ll, cc, fixed = split_all_linear_clusters(
            data, ll, cc, fixed, min_size, pct_threshold, verbose
        )
        # If outliers were removed before running this function, label -1 might be in the set.
        if set(ll) - set([-1]) == set(fixed):
            continue_splitting = False
    # return centers to original format
    new_centers = np.array(cc).reshape((len(cc), centers.shape[1]))
    return ll, new_centers


def describe_cluster_pspheres(data, labels, centers, use_pcylinders=True):
    """
    Purpose
    -------
    Generates a pandas dataframe with statistics on the clusters:
        ndata = number of data points in cluster
        localdim = dimensionality of local manifold in cluster by PCA analysis
        radius = p-sphere radius
        ps_density = ndata divided by ps_volume
        ps_volume = p-sphere or p-cylinder volume
        ps_vratio = ps_volume divided by dataset volume
        hc_density = ndata divided by hc_volume
        hc_volume = hypercuboid volume of the cluster data
        hc_vratio = hc_volume divided by the dataset volume
        ps_to_hc = ps_volume divided by hc_volume

    Arguments
    ---------
    data : ndarray, dimensions nxp
        the dataset
    labels : ndarray, dimension n
        the cluster index associated with each data vector, -1 indicates a
        flagged outlier rather than a cluster
    centers : ndarray, dimension kxp
        cluster centers as found by an algorithm or after splitting/fusing
    use_pcylinders : bool
        If True, this function will use the p-cylinder volume to calculate
        columns ps_density, ps_volume, ps_vratio, and ps_to_hc.
    """
    # remove any flagged outliers
    clusters = sorted(list(set(labels) - {-1}))
    full_hypercube_volume = bf.cluster_hcvolume(data)
    # intialize column arrays
    ndata = []
    radii = []
    psvolumes = []
    hcvolumes = []
    ps_to_hc = []
    psvolume_ratios = []
    hcvolume_ratios = []
    localdims = []
    psdensities = []
    hcdensities = []
    # fill column arrays
    if use_pcylinders:
        print(
            "Note: ps_density, ps_volume, ps_vratio, and ps_to_hc are using p-cylinder volumes"
        )
    for k in clusters:
        cluster_data = data[np.where(labels == k)[0], :]
        cluster_psphere = psh.PSphere(cluster_data, centers[k], k, data.shape[1])
        ndata.append(cluster_psphere.ndata)
        radii.append(cluster_psphere.radius)
        psvolume = cluster_psphere.pcylinder_vol
        if not use_pcylinders:
            psvolume = cluster_psphere.volume
        psvolumes.append(psvolume)
        psvolume_ratios.append(psvolume / full_hypercube_volume)
        hcvolume = bf.cluster_hcvolume(cluster_data)
        hcvolumes.append(hcvolume)
        hcvolume_ratios.append(hcvolume / full_hypercube_volume)
        ps_to_hc.append(psvolume / hcvolume)
        localdims.append(cluster_psphere.localdim)
        psdensities.append(cluster_psphere.ndata / psvolume)
        hcdensities.append(cluster_psphere.ndata / hcvolume)
    # build dataframe
    df_data = {"cluster": clusters, "ndata": ndata}
    df = pd.DataFrame(df_data).set_index("cluster")
    df["localdim"] = localdims
    df["radius"] = radii
    df["ps_volume"] = psvolumes
    df["ps_vratio"] = psvolume_ratios
    df["ps_density"] = psdensities
    df["hc_volume"] = hcvolumes
    df["hc_vratio"] = hcvolume_ratios
    df["hc_density"] = hcdensities
    df["ps_to_hc"] = ps_to_hc
    return df


def plot_cluster_pspheres(df, pdim, arg1, arg2, plot_isolates=False, filename=""):
    """
    Purpose
    -------
    Generates a log10-log10 plot of cluster volumes and densities.
    The color of the marker represents the local dimensionality of the points
    in the cluster : red = 1, orange = 2, yellow = 3, green = 4, blue = 5,
    violet = 6, grey = any higher value.
    The area of the marker is proportional to the number of data points in the
    cluster.

    Arguments
    ---------
    df : pandas dataframe
        the output of describe_cluster_pspheres()
    pdim : int
        The number of dimensions in the dataset. This value is only
        used to generate the color map for the local dimensionality.
    arg1 : str
        the name of the column in df to use for the horizontal axis
    arg2 : str
        the name of the column in df to use for the vertical axis
    plot_isolates : bool
        include clusters with one data point in the plot
    filename : str
        local filename to save the plot (optional)
    """
    if pdim < 1:
        print("plot_cluster_spheres: pdim must be at least 1")
        return
    if (arg1 not in df.columns.tolist()) or (arg2 not in df.columns.tolist()):
        print("plot_cluster_spheres: arg1 and arg2 must be in dataframe columns")
        print("   the choices are:", df.columns.tolist())
        return
    if plot_isolates:
        view = df
    else:
        view = df[df.ndata > 1]
    dims = list(range(0, pdim + 1))
    # clusters with just 1 point (localdim == 0) are also displayed as red
    colors = ["red", "red", "orange", "yellow", "green", "blue", "purple"]
    if pdim <= 6:
        cols_to_use = colors[:pdim+1]
    else:
        cols_to_use = colors + ["grey"] * (pdim - 6)
    dim_map = dict(zip(dims, cols_to_use))
    dims = [dim_map[i] for i in view.localdim]
    # largest cluster has a size of 900 (30x30 pixels).
    # Marker area is proportional to ndata
    size_ratio = 900 / max(view.ndata.values)
    sizes = np.round(view.ndata.values * size_ratio)
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(111)
    ax1.scatter(np.log10(view[arg1]), np.log10(view[arg2]), s=sizes, c=dims, alpha=0.5)
    for ic in range(view.shape[0]):
        ax1.text(np.log10(view[arg1].values[ic]), np.log10(view[arg2].values[ic]), str(ic))
    ax1.set_xlabel("log10(" + arg1 + ")")
    ax1.set_ylabel("log10(" + arg2 + ")")
    ax1.set_title("Relative cluster sizes, colors RROYGBV <=> localdims 0123456")
    plt.show()
    if filename != "":
        fig.tight_layout()
        fig.savefig(filename)
    return


def pcylinder_patch(xidx, yidx, center, radius, compact_dims, compact_ranges, color):
    """
    Purpose
    -------
    Creates a matplotlib patch to display a 2D cross-section of a p-cylinder.
    The patch will be either a rectangle or a circle.
    Note that center, radius, compact_dims, and compact_ranges are all attributes
    of the PSphere object.
    However, compact_dims and compact_ranges are not generated when a PSphere is
    initialized, you must use other methods to define the p-cylinder first.

    Arguments
    ---------
    xidx : int
        the horizontal dimension of the cross-secion
    yidx : int
        the vertical dimension of the cross-secion
    center : nparray, dtype=float, dimension p
        the coordinates of the cluster center (dimension p)
    radius : float
        the radius of the p-cylinder's spherical component
    compact_dims : nparray, dtype=bool, dimension p
        array of p flags, True indicates the dimension is compact
    compact_ranges : nparray, dtype=float, dimension 2xp
        2xp array of floats, indicating the minimum (row 0) and maximum
        (row 1) limits of the compact dimensions.

    Returns
    -------
    patch : matplotlib Patch object
    rectangle : bool
        True if the patch is a rectangle, False if the patch is a circle.
    """
    rectangle = True
    if compact_dims[xidx] and compact_dims[yidx]:
        ll_corner = (
            compact_ranges[0, xidx] + center[xidx],
            compact_ranges[0, yidx] + center[yidx],
        )
        dx = compact_ranges[1, xidx] - compact_ranges[0, xidx]
        dy = compact_ranges[1, yidx] - compact_ranges[0, yidx]
        patch = plt.Rectangle(ll_corner, dx, dy, fill=False, color=color)
    elif (not compact_dims[xidx]) and (not compact_dims[yidx]):
        patch = plt.Circle(
            (center[xidx], center[yidx]), radius, color=color, fill=False
        )
        rectangle = False
    elif compact_dims[xidx] and (not compact_dims[yidx]):
        ll_corner = (compact_ranges[0, xidx] + center[xidx], center[yidx] - radius)
        dx = compact_ranges[1, xidx] - compact_ranges[0, xidx]
        dy = 2.0 * radius
        patch = plt.Rectangle(ll_corner, dx, dy, fill=False, color=color)
    else:
        ll_corner = (center[xidx] - radius, compact_ranges[0, yidx] + center[yidx])
        dx = 2.0 * radius
        dy = compact_ranges[1, yidx] - compact_ranges[0, yidx]
        patch = plt.Rectangle(ll_corner, dx, dy, fill=False, color=color)
    return patch, rectangle
