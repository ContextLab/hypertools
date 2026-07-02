#!/usr/bin/env python
import warnings
from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    AgglomerativeClustering,
    Birch,
    FeatureAgglomeration,
    SpectralClustering,
    HDBSCAN,
)
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.decomposition import LatentDirichletAllocation, NMF
import numpy as np
from .._shared.helpers import *
from .format_data import format_data as formatter

# dictionary of models
models = {
    "KMeans": KMeans,
    "MiniBatchKMeans": MiniBatchKMeans,
    "AgglomerativeClustering": AgglomerativeClustering,
    "FeatureAgglomeration": FeatureAgglomeration,
    "Birch": Birch,
    "SpectralClustering": SpectralClustering,
    # sklearn's built-in HDBSCAN (>=1.3) replaces the unmaintained external
    # hdbscan package, which required a SyntaxWarning filter to import cleanly
    "HDBSCAN": HDBSCAN,
}

# mixture / soft-clustering models: instead of a hard label per observation,
# these return an (n_samples, n_components) matrix of membership proportions
mixture_models = {
    "GaussianMixture": GaussianMixture,
    "BayesianGaussianMixture": BayesianGaussianMixture,
    "LatentDirichletAllocation": LatentDirichletAllocation,
    "NMF": NMF,
}


def cluster(x, cluster="KMeans", n_clusters=3, format_data=True):
    """
    Performs clustering analysis and returns a list of cluster labels

    Parameters
    ----------
    x : A Numpy array, Pandas Dataframe or list of arrays/dfs
        The data to be clustered.  You can pass a single array/df or a list.
        If a list is passed, the arrays will be stacked and the clustering
        will be performed across all lists (i.e. not within each list).

    cluster : str or dict
        Model to use to discover clusters.  Supported algorithms are: KMeans,
        MiniBatchKMeans, AgglomerativeClustering, Birch, FeatureAgglomeration,
        SpectralClustering and HDBSCAN (default: KMeans), plus the mixture
        (soft-clustering) models GaussianMixture, BayesianGaussianMixture,
        LatentDirichletAllocation and NMF. Can be passed as a
        string, but for finer control of the model parameters, pass as a
        dictionary, e.g. reduce={'model' : 'KMeans', 'params' : {'max_iter' : 100}}.
        See scikit-learn specific model docs for details on parameters supported for
        each model. Note: LatentDirichletAllocation and NMF require
        non-negative data.

    n_clusters : int
        Number of clusters to discover. Not required for HDBSCAN. For mixture
        models this sets the number of components.

    format_data : bool
        Whether or not to first call the format_data function (default: True).

    Returns
    ----------
    cluster_labels : list or numpy.ndarray
        For hard-clustering models, a list of cluster labels (one per
        observation). For mixture models, an (n_samples, n_components) array
        of membership proportions whose rows sum to 1.

    """

    if cluster == None:
        return x

    if format_data:
        x = formatter(x, ppca=True)

    # resolve the model name and any custom params
    if isinstance(cluster, str):
        model_name = cluster
        model_params = None
    elif type(cluster) is dict:
        model_name = cluster["model"]
        model_params = cluster["params"]
    else:
        raise ValueError("cluster must be a string or a dict with 'model' "
                         "and 'params' keys")

    stacked = np.vstack(x)

    # mixture models: fit, then return soft membership proportions
    if model_name in mixture_models:
        if model_params is None:
            model_params = {"n_components": n_clusters}
        model = mixture_models[model_name](**model_params)
        if model_name in ("GaussianMixture", "BayesianGaussianMixture"):
            model.fit(stacked)
            proportions = model.predict_proba(stacked)
        else:
            # LDA / NMF: transform gives per-component loadings; normalize
            # rows so they are interpretable as membership proportions
            loadings = model.fit_transform(stacked)
            row_sums = loadings.sum(axis=1, keepdims=True)
            proportions = loadings / np.where(row_sums == 0, 1, row_sums)
        return proportions

    # hard-clustering models: return a list of labels
    model = models[model_name]
    if model_params is None:
        if model_name != "HDBSCAN":
            model_params = {"n_clusters": n_clusters}
        else:
            model_params = {}

    model = model(**model_params)
    model.fit(stacked)
    return list(model.labels_)
