#!/usr/bin/env python
import inspect
import warnings
from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    AgglomerativeClustering,
    Birch,
    FeatureAgglomeration,
    SpectralClustering,
    HDBSCAN,
    MeanShift,
    DBSCAN,
    OPTICS,
    AffinityPropagation,
)
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.decomposition import LatentDirichletAllocation, NMF
import numpy as np
from .._shared.helpers import *
from ..tools.format_data import format_data as formatter

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
    # density/bandwidth-based clusterers: these discover the number of
    # clusters themselves and have no n_clusters parameter (see the
    # signature-based exemption below)
    "MeanShift": MeanShift,
    "DBSCAN": DBSCAN,
    "OPTICS": OPTICS,
    "AffinityPropagation": AffinityPropagation,
}

# mixture / soft-clustering models: instead of a hard label per observation,
# these return an (n_samples, n_components) matrix of membership proportions
mixture_models = {
    "GaussianMixture": GaussianMixture,
    "BayesianGaussianMixture": BayesianGaussianMixture,
    "LatentDirichletAllocation": LatentDirichletAllocation,
    "NMF": NMF,
}


def normalize_membership_rows(loadings):
    """Normalize each row of a loadings matrix to sum to 1.

    Used to turn LDA/NMF per-component loadings into membership
    proportions. Shared with `hypertools.reduce.common.Reducer` (GH #174)
    so both `hyp.cluster` and `hyp.reduce` use the exact same
    normalization logic for these models.

    Parameters
    ----------
    loadings : numpy.ndarray
        An (n_samples, n_components) array of non-negative loadings.

    Returns
    -------
    numpy.ndarray
        `loadings` with each row divided by its sum (rows that sum to zero
        are left unchanged, to avoid division by zero).
    """
    row_sums = loadings.sum(axis=1, keepdims=True)
    return loadings / np.where(row_sums == 0, 1, row_sums)


def mixture_proportions(model_name, model, stacked):
    """Fit a mixture/soft-clustering model and return membership proportions.

    Shared by `hyp.cluster` (this module) and
    `hypertools.reduce.common.Reducer` (GH #174), so `hyp.reduce(x,
    reduce='GaussianMixture', ndims=3)` returns exactly the same style of
    proportions `hyp.cluster` does, via the SAME code path.

    Parameters
    ----------
    model_name : str
        One of `mixture_models`'s keys ('GaussianMixture',
        'BayesianGaussianMixture', 'LatentDirichletAllocation', 'NMF').
    model : object
        An unfitted instance of the corresponding scikit-learn model.
    stacked : numpy.ndarray
        A single (row-concatenated) 2D array to fit and transform.

    Returns
    -------
    numpy.ndarray
        An (n_samples, n_components) array of membership proportions; rows
        sum to 1 (except all-zero rows, left as-is).
    """
    if model_name in ("GaussianMixture", "BayesianGaussianMixture"):
        model.fit(stacked)
        return model.predict_proba(stacked)
    # LDA / NMF: transform gives per-component loadings; normalize rows so
    # they are interpretable as membership proportions
    loadings = model.fit_transform(stacked)
    return normalize_membership_rows(loadings)


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
        SpectralClustering, HDBSCAN, MeanShift, DBSCAN, OPTICS and
        AffinityPropagation (default: KMeans), plus the mixture
        (soft-clustering) models GaussianMixture, BayesianGaussianMixture,
        LatentDirichletAllocation and NMF. Can be passed as a
        string, but for finer control of the model parameters, pass as a
        dictionary, e.g. reduce={'model' : 'KMeans', 'params' : {'max_iter' : 100}}.
        A model class (e.g. sklearn.cluster.MeanShift) may also be passed
        directly as the 'model' value. See scikit-learn specific model docs
        for details on parameters supported for each model. Note:
        LatentDirichletAllocation and NMF require non-negative data.

    n_clusters : int
        Number of clusters to discover. Not used for models that discover
        the number of clusters automatically (HDBSCAN, MeanShift, DBSCAN,
        OPTICS, AffinityPropagation). For mixture models this sets the
        number of components.

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

    # resolve the model spec (name string or sklearn-style class) and params
    if isinstance(cluster, str):
        model_name = cluster
        model_params = None
    elif isinstance(cluster, dict):
        model_name = cluster["model"]
        model_params = dict(cluster.get("params", {})) or None
        if "n_clusters" in cluster:
            # top-level convenience: cluster={'model': ..., 'n_clusters': k}
            n_clusters = cluster["n_clusters"]
            model_params = model_params or {}
    else:
        raise ValueError("cluster must be a string or a dict with a 'model' "
                         "key (plus optional 'params' / 'n_clusters')")

    # model classes are accepted anywhere a name string is: resolve to the
    # registry name when known, otherwise use the class directly
    custom_cls = None
    if not isinstance(model_name, str):
        custom_cls = model_name
        model_name = getattr(model_name, "__name__", str(model_name))

    stacked = np.vstack(x)

    # mixture models: fit, then return soft membership proportions
    if model_name in mixture_models:
        if model_params is None:
            model_params = {}
        model_params.setdefault("n_components", n_clusters)
        model_cls = custom_cls or mixture_models[model_name]
        model = model_cls(**model_params)
        return mixture_proportions(model_name, model, stacked)

    # hard-clustering models: return a list of labels
    model = custom_cls or models[model_name]
    if model_params is None:
        model_params = {}
    # only inject n_clusters if the resolved model actually accepts it --
    # density/bandwidth clusterers (HDBSCAN, DBSCAN, MeanShift, OPTICS,
    # AffinityPropagation) discover the number of clusters themselves and
    # have no such __init__ parameter
    if "n_clusters" in inspect.signature(model).parameters:
        model_params.setdefault("n_clusters", n_clusters)

    model = model(**model_params)
    model.fit(stacked)
    return list(model.labels_)
