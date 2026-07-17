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


def _resolve_cluster_spec(cluster, n_clusters, random_state=None,
                          n_clusters_explicit=False):
    """Resolve a `cluster=` spec into an unfitted `Clusterer`.

    Accepts the full model-spec grammar: a registry name (string), a bare
    (uninstantiated) scikit-learn-style class, an already-constructed
    instance, the canonical dict spec `{'model': ..., 'args': [...],
    'kwargs': {...}}` (both keys optional; positional `'args'` are bound
    to the constructor's parameters by position, with `'kwargs'` winning
    on a conflict -- final wave item 3), or the LEGACY dict spec
    `{'model': ..., 'params': {...}}` (accepted for backward
    compatibility, but emits a `DeprecationWarning`).

    The `n_clusters=` convenience is preserved exactly as the pre-1.0 API
    behaved: it is injected into the constructor only when the resolved
    model's `__init__` signature accepts an `n_clusters` parameter
    (density/bandwidth clusterers that discover their own cluster count --
    HDBSCAN, MeanShift, DBSCAN, OPTICS, AffinityPropagation -- are left
    alone); mixture models always get `n_components` instead. A cluster
    count carried by the spec itself (an already-constructed instance's
    own setting, a dict spec's kwargs, or a dict's top-level 'n_clusters')
    always wins over the `n_clusters=` argument; when
    `n_clusters_explicit` is True and the two conflict, a `UserWarning`
    names the ignored value (F13-cluster-008/-009).

    Parameters
    ----------
    cluster : str, class, instance, or dict
        The cluster spec (see above).
    n_clusters : int
        Number of clusters/components, used as described above.
    random_state : int, RandomState, or None
        Seed injected into the constructor when the model accepts one and
        the spec did not set it (default: None).
    n_clusters_explicit : bool
        Whether `n_clusters` was explicitly provided by the caller (rather
        than being the dispatcher default). Only explicit values trigger
        conflict warnings (default: False).

    Returns
    -------
    Clusterer
        An unfitted `Clusterer` wrapping the resolved model.
    """
    # already-resolved wrapper: pass an existing Clusterer through UNWRAPPED
    # (idempotent). Wrapping it again in a fresh Clusterer double-wraps, and
    # Clusterer.fit_transform then calls `.fit`/`.labels_` on the inner
    # Clusterer, which has no such attributes (QC 2026-07: this crashed when a
    # fitted Clusterer was reused alongside cross-module reduce=/manip=).
    if isinstance(cluster, Clusterer):
        return cluster

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
            if "params" in cluster:
                # both the canonical and the legacy parameter keys were
                # given: the canonical 'args'/'kwargs' win, but say so
                # instead of silently dropping 'params' (F13-cluster-008)
                warnings.warn(
                    "cluster spec contains both the canonical "
                    "'args'/'kwargs' keys and the legacy 'params' key; "
                    "ignoring 'params' and using 'args'/'kwargs'",
                    UserWarning, stacklevel=3)
            model_params = dict(cluster.get("kwargs", {}))
            model_args = list(cluster.get("args", []))
        elif "params" in cluster:
            # LEGACY form (dev-1.0/fork): accepted for backward
            # compatibility, but deprecated in favor of the canonical
            # {'model', 'args', 'kwargs'} triple above.
            warnings.warn(
                "{'model': ..., 'params': {...}} is deprecated; use "
                "{'model': ..., 'args': [...], 'kwargs': {...}} instead",
                DeprecationWarning, stacklevel=3)
            model_params = dict(cluster["params"])
            model_args = []
        else:
            # e.g. {'model': ..., 'n_clusters': k} with no params/kwargs at all
            model_params = {}
            model_args = []
        if "n_clusters" in cluster:
            # top-level convenience: cluster={'model': ..., 'n_clusters': k}
            if n_clusters_explicit and cluster["n_clusters"] != n_clusters:
                warnings.warn(
                    f"n_clusters={n_clusters} conflicts with the cluster "
                    f"spec's own 'n_clusters' entry "
                    f"({cluster['n_clusters']}); using the spec's value",
                    UserWarning, stacklevel=3)
            n_clusters = cluster["n_clusters"]
    else:
        model_name = cluster
        model_params = {}
        model_args = []

    # bare classes and already-constructed instances are accepted anywhere
    # a name string is
    if not isinstance(model_name, str):
        if not inspect.isclass(model_name):
            # already-constructed instance: params ignored, used as-is
            # (mirrors hypertools.reduce.common.Reducer)
            if not (hasattr(model_name, 'fit')
                    or hasattr(model_name, 'fit_predict')):
                # e.g. cluster=42: fail here with the accepted spec forms
                # instead of a downstream AttributeError (F13-cluster-011)
                raise ValueError(
                    f"invalid cluster model {model_name!r} (type "
                    f"{type(model_name).__name__}): it has no fit or "
                    f"fit_predict method. Pass one of the supported model "
                    f"names "
                    f"({', '.join(sorted(list(CLUSTERERS) + list(MIXTURES)))}), "
                    f"a scikit-learn style clusterer class or instance, a "
                    f"dict spec like {{'model': 'KMeans', 'kwargs': "
                    f"{{...}}}}, or a fitted Clusterer.")
            if model_params or model_args:
                # an already-constructed instance inside a dict spec cannot
                # absorb the spec's 'args'/'kwargs'; warn instead of
                # silently dropping them (final wave item 4, matching the
                # instance + top-level n_clusters warning below)
                dropped = [k for k, v in (("'args'", model_args),
                                          ("'kwargs'", model_params)) if v]
                warnings.warn(
                    f"the cluster spec's 'model' is an already-constructed "
                    f"{type(model_name).__name__} instance (used as-is), so "
                    f"the spec's {' and '.join(dropped)} entries are "
                    "ignored; configure the instance directly, or pass the "
                    "class (or its name) to apply constructor parameters",
                    UserWarning, stacklevel=3)
            if n_clusters_explicit:
                # the instance's own configuration always wins; say so when
                # it visibly conflicts with n_clusters= (F13-cluster-008)
                inst_k = getattr(model_name, 'n_clusters',
                                 getattr(model_name, 'n_components', None))
                if inst_k is not None and inst_k != n_clusters:
                    warnings.warn(
                        f"cluster= is an already-constructed "
                        f"{type(model_name).__name__} instance (used as-is), "
                        f"so n_clusters={n_clusters} is ignored in favor of "
                        f"the instance's own setting ({inst_k}); configure "
                        f"the instance directly to change it",
                        UserWarning, stacklevel=3)
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

    if model_args:
        # honor a dict spec's positional 'args' (final wave item 3: they
        # used to be silently DISCARDED, so cluster={'model': 'KMeans',
        # 'args': [5]} quietly clustered with the default n_clusters=3).
        # Bind each positional value to its parameter NAME so it
        # participates in the documented precedence rules below
        # (spec kwargs win over 'args' on a conflict, and a spec-carried
        # cluster count wins over the n_clusters= argument).
        try:
            positional = [
                p for p in inspect.signature(model_cls).parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                              inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        except (TypeError, ValueError):
            positional = []
        if len(model_args) > len(positional):
            raise TypeError(
                f"the cluster spec's 'args' entry has {len(model_args)} "
                f"value(s) ({model_args!r}) but {registry_name} accepts at "
                f"most {len(positional)} positional argument(s); pass the "
                "extra parameters by name in the spec's 'kwargs' instead.")
        for value, p in zip(model_args, positional):
            if p.name in model_params:
                warnings.warn(
                    f"the cluster spec sets {p.name!r} both positionally "
                    f"(in 'args': {value!r}) and by name (in 'kwargs': "
                    f"{model_params[p.name]!r}); using the 'kwargs' value",
                    UserWarning, stacklevel=3)
            else:
                model_params[p.name] = value

    if registry_name in MIXTURES:
        if (n_clusters_explicit and "n_components" in model_params
                and model_params["n_components"] != n_clusters):
            warnings.warn(
                f"n_clusters={n_clusters} conflicts with the cluster spec's "
                f"n_components={model_params['n_components']}; using the "
                f"spec's value", UserWarning, stacklevel=3)
        model_params.setdefault("n_components", n_clusters)
    elif "n_clusters" in inspect.signature(model_cls).parameters:
        # only inject n_clusters if the resolved model actually accepts it --
        # density/bandwidth clusterers (HDBSCAN, DBSCAN, MeanShift, OPTICS,
        # AffinityPropagation) discover the number of clusters themselves
        # and have no such __init__ parameter
        if (n_clusters_explicit and "n_clusters" in model_params
                and model_params["n_clusters"] != n_clusters):
            warnings.warn(
                f"n_clusters={n_clusters} conflicts with the cluster spec's "
                f"n_clusters={model_params['n_clusters']}; using the spec's "
                f"value", UserWarning, stacklevel=3)
        model_params.setdefault("n_clusters", n_clusters)

    # silence sklearn >= 1.8's FutureWarning about HDBSCAN's changing `copy`
    # default by pinning today's effective value (False) explicitly
    # (F13-cluster-014). Results are identical either way: `copy` only
    # controls whether the input array may be modified in place, and
    # cluster() always fits on a freshly stacked array.
    if (registry_name == "HDBSCAN"
            and "copy" in inspect.signature(model_cls).parameters):
        model_params.setdefault("copy", False)

    # reproducibility (QC 2026-07): inject a top-level random_state when the
    # model accepts it and the user did not set it (KMeans, GaussianMixture,
    # SpectralClustering, ...); density clusterers without one are left alone.
    if (random_state is not None and 'random_state' not in model_params
            and 'random_state' in inspect.signature(model_cls).parameters):
        model_params['random_state'] = random_state

    return Clusterer(model_cls, params=model_params)


def cluster(x, cluster="KMeans", n_clusters=None, return_model=False,
           manip=None, normalize=None, reduce=None, ndims=None, align=None,
           format_data=True, random_state=None, model=None):
    """
    Performs clustering analysis and returns a list of cluster labels

    Parameters
    ----------
    x : A Numpy array, Pandas Dataframe or list/tuple of arrays/dfs
        The data to be clustered.  You can pass a single array/df or a
        list (a tuple of datasets is treated exactly like a list).
        If a list is passed, the arrays will be stacked and the clustering
        will be performed across all lists (i.e. not within each list).
        All datasets in a list must have the same number of columns (the
        stacked data shares one feature space); reduce or align them to a
        common dimensionality first if they differ. `None` raises a
        `TypeError`.

    cluster : str, class, instance, dict, fitted Clusterer, False, or None
        Model to use to discover clusters.  Supported algorithms are: KMeans,
        MiniBatchKMeans, AgglomerativeClustering, Birch, FeatureAgglomeration,
        SpectralClustering, HDBSCAN, MeanShift, DBSCAN, OPTICS and
        AffinityPropagation (default: KMeans), plus the mixture
        (soft-clustering) models GaussianMixture, BayesianGaussianMixture,
        LatentDirichletAllocation and NMF. Can be passed as a string, a bare
        (uninstantiated) scikit-learn-style class, an already-constructed
        instance, the canonical dict spec `{'model': ..., 'args': [...],
        'kwargs': {...}}` (both `'args'` and `'kwargs'` are OPTIONAL;
        positional `'args'` are bound to the model's constructor
        parameters by position -- e.g. `{'model': 'KMeans', 'args': [5]}`
        asks for 5 clusters -- with `'kwargs'` winning over `'args'` on a
        conflict, and a spec-carried cluster count winning over
        `n_clusters=`), or the LEGACY dict spec `{'model' : 'KMeans',
        'params' : {'max_iter' : 100}}` (accepted for backward
        compatibility, but emits a `DeprecationWarning`). A
        previously-fitted `Clusterer` (as returned by `return_model=True`)
        is applied via `.transform`/`.predict` instead of being refit;
        no-predict models (e.g. AgglomerativeClustering) can only recover
        their fit-time labels this way -- reusing them on different data
        warns or raises rather than silently mislabeling. `None` or `False`
        skips clustering entirely and returns the input unchanged. See
        scikit-learn specific model docs for details on parameters
        supported for each model. Note: LatentDirichletAllocation and NMF
        require non-negative data, and FeatureAgglomeration clusters
        features (columns), not observations -- it returns one label per
        column of the input (with a `UserWarning`), not one per row.

    n_clusters : int or None
        Number of clusters to discover (default: None, which means 3). Not
        used for models that discover the number of clusters automatically
        (HDBSCAN, MeanShift, DBSCAN, OPTICS, AffinityPropagation). For
        mixture models this sets the number of components. If the cluster
        spec itself carries a cluster count (an already-constructed
        instance's own setting, or `n_clusters`/`n_components` in a dict
        spec's kwargs), the spec's value wins and a `UserWarning` notes the
        conflict.

    return_model : bool
        If True, also return the fitted model: the fitted `Clusterer`
        wrapper when only the `cluster` stage ran, or a fitted
        `hypertools.Pipeline` when `manip=`/`normalize=`/`reduce=`/`align=`
        made multiple stages run (default: False).

    manip, normalize, reduce, align : model spec, False, or None
        Cross-module stage kwargs (GH #138): when any of these is given,
        the other stages also run (via
        `hypertools.core.pipeline.build_pipeline`), in the canonical order
        `manip -> normalize -> reduce -> align -> cluster` (GH #153), with
        this function's own `cluster=`/`n_clusters=` slotted in at the
        cluster stage (default: None for all four, i.e. only `cluster`
        runs). `False` skips a stage, exactly like None.

    ndims : int or None
        Passed through to the `reduce` stage (as `ndims=`) when `reduce=`
        is also given. Without `reduce=` it has no effect, and a
        `UserWarning` says so (the pre-1.0 `cluster(ndims=...)` shortcut
        that reduced before clustering was removed).

    format_data : bool
        Whether or not to first call the format_data function (default: True).

    random_state : int, RandomState, or None
        Seed for reproducibility. Injected into the clustering model's
        constructor when it accepts a `random_state` (KMeans,
        SpectralClustering, GaussianMixture, ...) and the spec did not set
        one itself; density clusterers without a `random_state` parameter,
        and already-constructed instances you pass in, are left alone
        (default: None).

    model : same forms as `cluster`, or None
        Alias for `cluster=`, so the own-stage model spec can be spelled
        `model=` here exactly as in `hyp.manip`/`hyp.impute`/`hyp.predict`/
        `hyp.align` (release-1.0 audit: the sibling APIs used two different
        kwarg conventions). Pass only one of `cluster=`/`model=`; passing
        both (with different values) raises `ValueError` (default: None).

    Returns
    -------
    cluster_labels : list or numpy.ndarray
        For hard-clustering models, a list of cluster labels (one per
        observation; FeatureAgglomeration instead returns one label per
        COLUMN of the input -- it clusters features). For mixture models,
        an (n_samples, n_components) array of membership proportions whose
        rows sum to 1. If `return_model=True`, a `(cluster_labels, model)`
        tuple is returned instead.

    Examples
    --------
    >>> import numpy as np
    >>> import hypertools as hyp
    >>> rng = np.random.default_rng(0)
    >>> x = np.vstack([rng.standard_normal((20, 4)),
    ...                rng.standard_normal((20, 4)) + 10.0])
    >>> labels = hyp.cluster(x, n_clusters=2, random_state=0)
    >>> len(labels), len(set(labels))
    (40, 2)

    """
    from ..core.shared import require_data
    # None always raises the unified dispatcher TypeError, and a tuple of
    # datasets is accepted exactly like a list (2026-07 release audit,
    # final wave items 9/15)
    require_data(x, 'cluster')
    if isinstance(x, tuple):
        x = list(x)

    # model= is an alias for cluster= (release-1.0 audit,
    # D05-gallery-data-text-020: manip/impute/predict/align spell their
    # own-stage spec `model=`, and hyp.cluster(x, model='KMeans') used to
    # die with a bare TypeError naming neither kwarg).
    if model is not None:
        if cluster != 'KMeans' and cluster is not model:
            raise ValueError(
                "cannot pass both cluster= and model=; they are aliases "
                "for the same model spec -- pass just one (e.g. "
                "cluster='HDBSCAN' or model='HDBSCAN').")
        cluster = model

    # False is an explicit "skip this stage", for every stage kwarg, exactly
    # like None (F13-cluster-007 contract) -- normalize it up front so
    # plot()/analyze() can thread cluster=False (and friends) through
    if cluster is False:
        cluster = None
    manip = None if manip is False else manip
    normalize = None if normalize is False else normalize
    reduce = None if reduce is False else reduce
    align = None if align is False else align

    # n_clusters=None (the signature default) means 3; only an explicit
    # value participates in spec-conflict warnings (F13-cluster-008/-009)
    n_clusters_explicit = n_clusters is not None
    if n_clusters is None:
        n_clusters = 3

    # a whole already-fitted Pipeline handed back as cluster= (e.g. the model
    # from an earlier cross-module return_model=True call) is reused as-is via
    # .transform, BEFORE the cross-module branch below -- otherwise it would be
    # wrapped in a fresh Clusterer whose fit_transform reads `.labels_` off the
    # Pipeline and crashes (QC 2026-07). Redundant stage kwargs are warned +
    # ignored (the Pipeline already encodes them).
    from ..core.shared import is_reused_pipeline
    if is_reused_pipeline(cluster, {'manip': manip, 'normalize': normalize,
                                    'reduce': reduce, 'align': align}, 'cluster'):
        result = cluster.transform(x)
        return (result, cluster) if return_model else result

    # the pre-1.0 cluster(ndims=...) shortcut (reduce, then cluster) was
    # removed: in 1.0, ndims= is only a passthrough to an explicitly
    # requested reduce stage. Say so instead of silently no-oping the
    # argument for migrating 0.x users (F13-cluster-019).
    if ndims is not None and reduce is None:
        warnings.warn(
            f"cluster()'s ndims= is a passthrough to the reduce stage and "
            f"has no effect unless reduce= is also given; ignoring "
            f"ndims={ndims}. Pass e.g. reduce='IncrementalPCA' to reduce "
            f"before clustering.", UserWarning)

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
        cluster_spec = (_resolve_cluster_spec(cluster, n_clusters, random_state,
                                              n_clusters_explicit)
                        if cluster is not None else None)
        pipeline = build_pipeline(manip=manip, normalize=normalize,
                                   reduce=reduce, ndims=ndims,
                                   align=align, cluster=cluster_spec)
        result = pipeline.fit_transform(x)
        return (result, pipeline) if return_model else result

    if cluster is None:
        return (x, None) if return_model else x

    if format_data:
        x = formatter(x, ppca=True)

    # give ragged lists a real error instead of numpy's raw concatenation
    # message (F13-cluster-012): the stack-once-fit-once recipe needs every
    # dataset in one shared feature space
    if isinstance(x, list) and len(x) > 1:
        widths = [np.atleast_2d(np.asarray(xi)).shape[1] for xi in x]
        if len(set(widths)) > 1:
            raise ValueError(
                f"cannot cluster a list of datasets with different numbers "
                f"of columns (got column counts {widths}): the datasets are "
                f"stacked and clustered in one shared feature space. Reduce "
                f"or align them to a common dimensionality first, e.g. "
                f"cluster(x, reduce='IncrementalPCA', ndims=k) or "
                f"cluster(x, align='HyperAlign').")

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

    clusterer = _resolve_cluster_spec(cluster, n_clusters, random_state,
                                      n_clusters_explicit)
    result = clusterer.fit_transform(stacked)
    return (result, clusterer) if return_model else result
