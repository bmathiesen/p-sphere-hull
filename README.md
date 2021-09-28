# P-Sphere Hull

Welcome to the P-Sphere Hull project!  This git contains a python library that will help you model the training
dataset of an AI project as a collection p-dimensional spheres or cylinders, in the original feature space,
and quickly test whether new data vectors lie inside or outside their boundaries.  In this way, an AI application
can detect data outside its training domain, preventing risky predictions in the extrapolation regime 
from reaching end users.

## Introduction

The library supports the following workflow to model the domain of a p-dimensional dataset:

1. Cluster the dataset with a high value of k (30-100), using any method appropriate to its global topology. Gaussian Mixtures often
work well. The goal is to produce a collection of subdomains that are compact and convex.
2. Optionally, use the functions defined in the `refine_cluster_set` module to modify the initial clustering solution. For example,
you can split a high-volume cluster into two subclusters or remove an outlier that is increasing the cluster radius.
3. Construct a `PSphereHull` object based on the training dataset and final clustering solution (data labels and cluster centers).
This creates a `PSphere` object for each cluster. By default, compact features are identified automatically, and the
p-sphere is replaced by a p-cylinder aligned with the feature axes.
4. The `PSphereHull` manages the full collection of p-spheres or p-cylinders. It contains methods to test new data for membership
in the collection, identify and flag redundant spheres, and several other useful functions. It can be deployed in parallel with 
an AI application to screen incoming data for extrapolation risk before making a prediction.

## Folders

* `/data` : toy datasets for unit tests, and one OpenML dataset used by the demonstrator notebook.
* `/examples` : Jupyter notebooks to explain the process of modeling a training dataset and building the p-sphere hull.
* `/src/PSphereHull` : Python package that you can import into your own projects.
* `/tests`: Python unit tests for the package.

## Install directly via pip

```bash
python -m pip install git+https://code.tessella.com/scm/frar/pspherehull.git#egg=PSphereHull
```

### Preparation

1. Check-out code
2. Create virtual environment
3. Install package in editable mode with development dependencies

```bash
python -m pip install -e .[dev]
```

### Run unit tests

In virtual environment:

```bash
python -m pytest
```

### Examples

Examples also require the following to be installed:

* juypter
* ipython
* ipywidgets
* watermark
