#!/usr/bin/env python
"""hyp.cluster dispatcher: resolve a cluster spec and fit_transform it.

Follows the same stack-once-fit-once recipe as `hyp.reduce`
(`hypertools.reduce.reduce`): the input is stacked into one 2D array, a
single `Clusterer` is fit on it (so cluster assignments are comparable
across input datasets), and hard-clustering models return a list of labels
while mixture/soft-clustering models return an (n_samples, n_components)
membership-proportion matrix (GH #174).
"""
import inspect
import warnings

import numpy as np

from .common import Clusterer, CLUSTERERS, MIXTURES, mixture_proportions, normalize_membership_rows
from .._shared.helpers import *
from ..tools.format_data import format_data as formatter

# backward-compatible aliases: `hypertools.cluster.cluster.models` and
# `.mixture_models` were the pre-1.0 registries (plain dicts of name ->
# class). `hypertools.reduce.common` and `hypertools.core.pipeline` still
# import these names from here and must keep working unchanged.
models = CLUSTERERS
mixture_models = MIXTURES


def _resolve_cluster_spec(cluster, n_clusters):
    """Resolve a `cluster=` spec into an unfitted `Clusterer`.

    Accepts the full model-spec grammar: a registry name (string), a bare
    (uninstantiated) scikit-learn-style class, an already-constructed
    instance, the canonical dict spec `{'model': ..., 'args': [...],
    'kwargs': {...}}`, or the LEGACY dict spec `{'model': ..., 'params':
    {...}}` (accepted for backward compatibility, but emits a
    `DeprecationWarning`).

    The `n_clusters=` convenience is preserved exactly as the pre-1.0 API
    behaved: it is injected into the constructor only when the resolved
    model's `__init__` signature accepts an `n_clusters` parameter
    (density/bandwidth clusterers that discover their own cluster count --
    HDBSCAN, MeanShift, DBSCAN, OPTICS, AffinityPropagation -- are left
    alone); mixture models always get `n_components` instead.

    Parameters
    ----------
    cluster : str, class, instance, or dict
        The cluster spec (see above).
    n_clusters : int
        Number of clusters/components, used as described above.

    Returns
    -------
    Clusterer
        An unfitted `Clusterer` wrapping the resolved model.
    """
    if isinstance(cluster, dict):
        try:
            model_name = cluster["model"]
        except KeyError:
            raise ValueError("If passing a dictionary, pass the model as the "
                             "value of the 'model' key and a dictionary of "
                             "custom parameters as the value of the 'kwargs' "
                             "key (the legacy 'params' key is also accepted).")
        if "args" in cluster or "kwargs" in cluster:
            # canonical 1.0 dict spec: {'model': ..., 'args': [...], 'kwargs': {...}}
            model_params = dict(cluster.get("kwargs", {}))
        elif "params" in cluster:
            # LEGACY form (dev-1.0/fork): accepted for backward
            # compatibility, but deprecated in favor of the canonical
            # {'model', 'args', 'kwargs'} triple above.
            warnings.warn(
                "{'model': ..., 'params': {...}} is deprecated; use "
                "{'model': ..., 'args': [...], 'kwargs': {...}} instead",
                DeprecationWarning, stacklevel=3)
            model_params = dict(cluster["params"])
        else:
            # e.g. {'model': ..., 'n_clusters': k} with no params/kwargs at all
            model_params = {}
        if "n_clusters" in cluster:
            # top-level convenience: cluster={'model': ..., 'n_clusters': k}
            n_clusters = cluster["n_clusters"]
    else:
        model_name = cluster
        model_params = {}

    # bare classes and already-constructed instances are accepted anywhere
    # a name string is
    if not isinstance(model_name, str):
        if not inspect.isclass(model_name):
            # already-constructed instance: params ignored, used as-is
            # (mirrors hypertools.reduce.common.Reducer)
            return Clusterer(model_name)
        model_cls = model_name
        registry_name = getattr(model_name, "__name__", str(model_name))
    elif model_name in MIXTURES:
        model_cls = MIXTURES[model_name]
        registry_name = model_name
    elif model_name in CLUSTERERS:
        model_cls = CLUSTERERS[model_name]
        registry_name = model_name
    else:
        raise ValueError(
            f"unknown cluster model {model_name!r}; supported names: "
            f"{', '.join(sorted(list(CLUSTERERS) + list(MIXTURES)))} (or "
            f"pass a scikit-learn style instance directly)")

    if registry_name in MIXTURES:
        model_params.setdefault("n_components", n_clusters)
    elif "n_clusters" in inspect.signature(model_cls).parameters:
        # only inject n_clusters if the resolved model actually accepts it --
        # density/bandwidth clusterers (HDBSCAN, DBSCAN, MeanShift, OPTICS,
        # AffinityPropagation) discover the number of clusters themselves
        # and have no such __init__ parameter
        model_params.setdefault("n_clusters", n_clusters)

    return Clusterer(model_cls, params=model_params)


def cluster(x, cluster="KMeans", n_clusters=3, return_model=False,
           manip=None, normalize=None, reduce=None, ndims=None, align=None,
           format_data=True):
    """
    Performs clustering analysis and returns a list of cluster labels

    Parameters
    ----------
    x : A Numpy array, Pandas Dataframe or list of arrays/dfs
        The data to be clustered.  You can pass a single array/df or a list.
        If a list is passed, the arrays will be stacked and the clustering
        will be performed across all lists (i.e. not within each list).

    cluster : str, class, instance, dict, or fitted Clusterer
        Model to use to discover clusters.  Supported algorithms are: KMeans,
        MiniBatchKMeans, AgglomerativeClustering, Birch, FeatureAgglomeration,
        SpectralClustering, HDBSCAN, MeanShift, DBSCAN, OPTICS and
        AffinityPropagation (default: KMeans), plus the mixture
        (soft-clustering) models GaussianMixture, BayesianGaussianMixture,
        LatentDirichletAllocation and NMF. Can be passed as a string, a bare
        (uninstantiated) scikit-learn-style class, an already-constructed
        instance, the canonical dict spec `{'model': ..., 'args': [...],
        'kwargs': {...}}`, or the LEGACY dict spec `{'model' : 'KMeans',
        'params' : {'max_iter' : 100}}` (accepted for backward
        compatibility, but emits a `DeprecationWarning`). A
        previously-fitted `Clusterer` (as returned by `return_model=True`)
        is applied via `.transform`/`.predict` instead of being refit. See
        scikit-learn specific model docs for details on parameters
        supported for each model. Note: LatentDirichletAllocation and NMF
        require non-negative data.

    n_clusters : int
        Number of clusters to discover. Not used for models that discover
        the number of clusters automatically (HDBSCAN, MeanShift, DBSCAN,
        OPTICS, AffinityPropagation). For mixture models this sets the
        number of components.

    return_model : bool
        If True, also return the fitted model: the fitted `Clusterer`
        wrapper when only the `cluster` stage ran, or a fitted
        `hypertools.Pipeline` when `manip=`/`normalize=`/`reduce=`/`align=`
        made multiple stages run (default: False).

    manip, normalize, reduce, align : model spec or None
        Cross-module stage kwargs (GH #138): when any of these is given,
        the other stages also run (via
        `hypertools.core.pipeline.build_pipeline`), in the canonical order
        `manip -> normalize -> reduce -> align -> cluster` (GH #153), with
        this function's own `cluster=`/`n_clusters=` slotted in at the
        cluster stage (default: None for all four, i.e. only `cluster`
        runs).

    ndims : int or None
        Passed through to the `reduce` stage (as `ndims=`) when `reduce=`
        is also given.

    format_data : bool
        Whether or not to first call the format_data function (default: True).

    Returns
    ----------
    cluster_labels : list or numpy.ndarray
        For hard-clustering models, a list of cluster labels (one per
        observation). For mixture models, an (n_samples, n_components) array
        of membership proportions whose rows sum to 1. If `return_model=True`,
        a `(cluster_labels, model)` tuple is returned instead.

    """
    # cross-module kwargs (#138): assemble and run a Pipeline (in canonical
    # order, #153) instead of the single-stage path below whenever another
    # stage is requested. Lazy import avoids a cluster<->core.pipeline cycle
    # (core.pipeline itself lazily imports cluster.cluster).
    if any(stage is not None for stage in (manip, normalize, reduce, align)):
        from ..core.pipeline import build_pipeline
        # bake n_clusters into the cluster-stage spec up front: build_pipeline
        # has no n_clusters= kwarg of its own (mirrors how it threads ndims=
        # through to the reduce stage), so the resolved (unfitted) Clusterer
        # is what gets passed to the cluster stage instead of the raw spec
        cluster_spec = _resolve_cluster_spec(cluster, n_clusters) if cluster is not None else None
        pipeline = build_pipeline(manip=manip, normalize=normalize,
                                   reduce=reduce, ndims=ndims,
                                   align=align, cluster=cluster_spec)
        result = pipeline.fit_transform(x)
        return (result, pipeline) if return_model else result

    if cluster is None:
        return (x, None) if return_model else x

    if format_data:
        x = formatter(x, ppca=True)

    stacked = np.vstack(x)

    # an already-fitted Clusterer (returned from an earlier
    # return_model=True call, or built above for the cross-module pipeline
    # path) is reused via `transform`, never refit; an unfitted one is
    # fit_transform'd directly, skipping spec re-resolution
    if isinstance(cluster, Clusterer):
        if cluster.is_fitted:
            result = cluster.transform(stacked)
        else:
            result = cluster.fit_transform(stacked)
        return (result, cluster) if return_model else result

    clusterer = _resolve_cluster_spec(cluster, n_clusters)
    result = clusterer.fit_transform(stacked)
    return (result, clusterer) if return_model else result
