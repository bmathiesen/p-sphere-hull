import pathlib
import unittest

import numpy as np

from PSphereHull import basic_functions as bf
from PSphereHull import refine_cluster_set as rcs


class TestSplitFuseCluster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_path = pathlib.Path(__file__).parents[1] / "data"

        cls.fused_center = np.array([0.0218388, -0.01216467, -0.01744412])
        cls.new_center_2 = np.array([0.39221039,  0.08710292, -0.19463418])
        cls.new_center_0 = np.array([0.48302155, -0.10281553,  0.14590736])
        cls.data = np.loadtxt(data_path / "toydata_2clusters.csv", delimiter=";")
        cls.centers = np.loadtxt(
            data_path / "toydata_2clusters_centers.csv", delimiter=";"
        )
        cls.labels = np.loadtxt(
            data_path / "toydata_2clusters_labels.csv", delimiter=";"
        )

    def test_fuse_cluster(self):
        new_labels, new_centers, success = rcs.fuse_clusters(
            self.data, self.labels, self.centers, 0, 1
        )
        self.assertTrue(np.allclose(new_centers[0], self.fused_center))

    def test_split_cluster(self):
        # The function uses KMeans(2) to split cluster 0, but it is random
        # whether the subclusters are labeled 0 and 2 or vice versa.
        np.random.seed(42)
        new_labels, new_centers, success = rcs.split_cluster(
            self.data, self.labels, self.centers, 0
        )
        self.assertTrue(np.allclose(new_centers[0], self.new_center_0))
        self.assertTrue(np.allclose(new_centers[1], self.centers[1]))
        self.assertTrue(np.allclose(new_centers[2], self.new_center_2))


class TestRecursiveSplit(unittest.TestCase):
    pass


class TestSinglePCylinder(unittest.TestCase):
    """
    Data contains 512 uniformly distributed points in a volume defined
    by a unit 3-sphere centered on the origin (columns 0..2) and a
    box of width 0.05 centered on the origin (columns 3..5). Therefore
    the natural shape of the data is a single p-cylinder with 3 compact
    dimensions.
    """

    @classmethod
    def setUpClass(cls):
        data_path = pathlib.Path(__file__).parents[1] / "data"
        cls.data = np.loadtxt(data_path / "unit_pcylinder_6D.csv", delimiter=";")
        cls.centers = np.zeros(3)
        cls.labels = np.zeros(cls.data.shape[0])

    # local dimensionality of the cluster according to PCA
    def test_psphere_local_dimensions(self):
        localdim, pc0 = bf.cluster_local_dimensions(self.data)
        self.assertEqual(localdim, 3, "Should be 3")

    def test_hypercube_volume(self):
        volume = bf.cluster_hcvolume(self.data)
        self.assertAlmostEqual(
            volume, 0.00019300584213838, 10, "Should be ~0.0001930058"
        )

    def test_psphere_radius_volume(self):
        radius = bf.radius_from_centered_data(self.data)
        volume = bf.psphere_volume(radius, 6)
        self.assertAlmostEqual(radius, 0.5008192322, 6, "Should be ~0.5008192")
        self.assertAlmostEqual(volume, 0.0815425626, 6, "Should be ~0.08154256")


if __name__ == "__main__":
    unittest.main()
