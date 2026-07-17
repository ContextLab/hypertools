#!/usr/bin/env python
"""hyp.analyze: the classic manip -> normalize -> reduce -> align -> cluster
pipeline dispatcher (GH #138 cross-module kwargs, GH #227 `pipeline=` reuse).
"""

import warnings

from ..reduce.reduce import reduce as reducer
from .align import align as aligner
from .normalize import normalize as normalizer


_STAGE_KWARG_NAMES = ('manip', 'normalize', 'reduce', 'align', 'cluster')


def _impute_format(data, impute):
    """Run `format_data` with the user's `impute=` override BEFORE any stage
    runs, so the chosen imputer (rather than the PPCA default buried inside
    each stage's own `format_data` pass) fills any missing values. Preserves
    the caller's single-dataset vs list-of-datasets shape."""
    from .format_data import format_data as formatter

    formatted = formatter(data, ppca=True, impute=impute)
    if isinstance(data, (list, tuple)):
        return formatted
    return formatted[0]


def analyze(data, manip=None, normalize=None, reduce=None, ndims=None, align=None,
           cluster=None, pipeline=None, return_model=False, internal=False, impute=None,
           random_state=None):
    """
    Wrapper function for manip -> normalize -> reduce -> align -> cluster
    transformations (the canonical 1.0 pipeline order, GH #153): each
    requested stage is applied to the previous stage's output, in that
    order (e.g. `normalize=` output feeds `reduce=`, whose output feeds
    `align=`).

    Parameters
    ----------
    data : numpy array, pandas df, or list of arrays/dfs
        The data to analyze. Each dataset must be 2-D (observations x
        features); 1-D vectors are treated as single-feature columns.

    manip : model spec, False, or None
        Cross-module stage kwarg (GH #138): a `hypertools.manip` spec (a
        registry name, dict spec, class/instance, or a `list` chaining
        several -- see `hypertools.manip.manip.manip`), applied FIRST (the
        `manip` stage runs before `normalize`/`reduce`/`align`/`cluster` in
        the canonical order). `False` or `None` (default) skips this stage.

    normalize : str or False or None
        If set to 'across', the columns of the input data will be z-scored
        across lists. That is, the z-scores will be computed with
        respect to column n across all arrays passed in the list. If set
        to 'within', the columns will be z-scored within each list that is
        passed. If set to 'row', each row of the input data will be z-scored.
        If set to False or None (default), the input data will be returned
        with no z-scoring.

    reduce : str, dict, class, instance, fitted Reducer, False, or None
        Decomposition/manifold learning model to use, or `False`/`None`
        (default) to SKIP dimensionality reduction entirely (in which case
        `ndims=` has no effect -- see `ndims` below). Models supported:
        PCA, IncrementalPCA, SparsePCA,
        MiniBatchSparsePCA, KernelPCA, FastICA, FactorAnalysis, TruncatedSVD,
        DictionaryLearning, MiniBatchDictionaryLearning, TSNE, Isomap,
        SpectralEmbedding, LocallyLinearEmbedding, MDS, and UMAP; the mixture
        models GaussianMixture, BayesianGaussianMixture,
        LatentDirichletAllocation and NMF (GH #174); and the torch autoencoders
        Autoencoder, DeepAutoencoder, SparseAutoencoder,
        ConvolutionalAutoencoder, SequenceAutoencoder and
        VariationalAutoencoder (GH #162, `pip install "hypertools[torch]"`).
        Can be passed as a string, or for finer control as a dictionary, e.g.
        reduce={'model': 'PCA', 'kwargs': {'whiten': True}}. See scikit-learn
        model docs for details on parameters supported for each model.

    ndims : int
        Number of dimensions to reduce to. Only takes effect when `reduce=`
        is also given: if `reduce` is left at its default of `None` (or is
        `False`), no reduction runs and `ndims=` is ignored (a `UserWarning`
        is emitted so the request does not silently no-op).

    align : str, dict, False, or None
        Alignment model to bring a list of datasets into a shared space. If
        str, 'hyper' (hyperalignment) or 'SRM' (shared response model). You
        can also pass a dictionary for finer control, where 'model' specifies
        the model and 'kwargs' holds its parameters, e.g.
        align={'model': 'HyperAlign', 'kwargs': {'n_iter': 10}}. If False or
        None, no alignment is applied (default: None).

    cluster : model spec, False, or None
        Cross-module stage kwarg (GH #138): a `hypertools.cluster` spec (a
        registry name, dict spec, or class/instance -- see
        `hypertools.cluster.cluster.cluster`), applied LAST (after `align`,
        the canonical order). `analyze` still returns the TRANSFORMED DATA
        (not cluster labels) when `cluster=` is given; the cluster labels
        themselves are retrievable from the fitted `hypertools.Pipeline`'s
        `'cluster'` step (`model.named_steps['cluster']`) when
        `return_model=True` is also passed -- pass the RETURNED transformed
        data back through that step's `.transform` to recover the labels
        (this works for every clusterer, including hard clusterers such as
        DBSCAN / AgglomerativeClustering that have no out-of-sample
        `predict`). For a list of datasets, `.transform` returns one flat
        label sequence over the row-concatenated data; to get labels split
        PER dataset, split that flat sequence by each dataset's row count,
        e.g. ``np.split(labels, np.cumsum([len(d) for d in data])[:-1])``
        (`hypertools.cluster` likewise returns one flat label sequence for
        a list input, because the datasets are row-stacked before
        clustering). `False` or
        `None` (default) skips this stage.

    pipeline : hypertools.Pipeline or None
        A previously-FITTED `Pipeline` (e.g. from an earlier
        `analyze(..., return_model=True)` call) to apply to `data` via
        `.transform` -- reusing its learned parameters rather than
        re-fitting them (GH #227). Mutually exclusive with
        `manip=`/`normalize=`/`reduce=`/`align=`/`cluster=` (all must be
        left at their default of `None`) -- passing both raises
        `ValueError` naming the conflicting kwarg(s). `internal=`/`impute=`
        are still honored (`internal=True` still guarantees a list is
        returned, even for a single-dataset `data`; `impute=` overrides the
        PPCA missing-data fill exactly as on the fitting paths). `ndims=`
        is IGNORED on this path (with a `UserWarning`): a fitted Pipeline
        applies its `reduce` stage exactly as fitted -- re-fit with
        `analyze(..., reduce=..., ndims=..., return_model=True)` to change
        the dimensionality. Reusing a pipeline whose last step is
        `'cluster'` returns the TRANSFORMED DATA (matching the `cluster=`
        contract above, not the labels); recover the labels via
        `pipeline.named_steps['cluster'].transform(returned_data)`
        (default: None).

    return_model : bool
        If True, also return the fitted model: a fitted `hypertools.Pipeline`
        covering whichever stages ran (default: False). Using ONLY the
        legacy `normalize=`/`reduce=`/`align=` kwargs (no `manip=`/
        `cluster=`/`pipeline=`) with `return_model=False` (the default) runs
        the exact same code path `analyze` has always used, so every
        existing caller (`hyp.plot`, `hyp.load`, pre-1.0 scripts) is
        byte-identical. `return_model=True`, or passing `manip=`/`cluster=`,
        routes through `hypertools.core.pipeline.build_pipeline` instead
        (needed to hand back a genuinely fit-once-reusable `Pipeline`);
        `impute=` is honored either way, but `internal=` is only otherwise
        meaningful for the legacy path -- see `pipeline=` above for how it
        is handled on that path.

    internal : bool
        (Internal use, e.g. by `hyp.plot`) if True, always return a list
        even when the input was a single dataset (default: False).

    impute : str, dict, class, class instance or None
        Overrides the default PPCA missing-data fill (applied at the
        `format_data` stage, before any pipeline stage runs) with a
        different `hypertools.impute` model, e.g. 'Kalman', 'KNNImputer'.
        Honored on every path -- with or without `normalize=`, and with
        `pipeline=` (default: None, i.e. PPCA -- byte-compatible with
        pre-1.0 behavior).

    random_state : int, numpy.random RandomState/Generator, or None
        Seed (or seeded generator) threaded through to the `reduce=` and
        `cluster=` stages so stochastic models (e.g. TSNE, MDS, KMeans)
        give reproducible results across calls (default: None).

    Returns
    -------
    analyzed_data : list of numpy arrays (or a single array)
        The processed data: for a LIST of datasets, a list with one entry
        per dataset. For a SINGLE dataset, `normalize=`/`reduce=`-only
        calls return a single array, while combinations that include
        `align=` return a list of length 1 (alignment always operates on --
        and returns -- a list); `manip=`-only calls return the
        manipulator's own output type (typically pandas DataFrames). Pass
        `internal=True` to guarantee a list regardless of input shape. If
        `return_model=True`, an `(analyzed_data, model)` tuple is returned
        instead.

    Examples
    --------
    >>> import numpy as np
    >>> import hypertools as hyp
    >>> x = np.cumsum(np.random.default_rng(0).standard_normal((40, 5)),
    ...               axis=0)
    >>> analyzed = hyp.analyze(x, normalize='within', reduce='PCA', ndims=3)
    >>> analyzed.shape
    (40, 3)

    """
    stage_kwargs = {'manip': manip, 'normalize': normalize, 'reduce': reduce,
                     'align': align, 'cluster': cluster}

    if pipeline is not None:
        if not hasattr(pipeline, 'transform'):
            raise TypeError(
                "pipeline= expects a fitted hypertools.Pipeline (e.g. the "
                "model returned by analyze(..., return_model=True)); got "
                f"{type(pipeline).__name__}: {pipeline!r}. To specify "
                "pipeline stages by name, use the manip=/normalize=/reduce=/"
                "align=/cluster= kwargs instead."
            )
        conflicting = sorted(name for name, value in stage_kwargs.items()
                             if value is not None)
        if conflicting:
            raise ValueError(
                "pipeline= is mutually exclusive with the stage kwarg(s) "
                f"{', '.join(conflicting)} (a fitted Pipeline already "
                "encodes which stages run and their fitted parameters); "
                "pass pipeline= alone."
            )
        if ndims is not None:
            warnings.warn(
                f"ndims={ndims!r} is ignored when pipeline= is given: a "
                "fitted Pipeline applies its reduce stage exactly as fitted. "
                "Re-fit with analyze(..., reduce=..., ndims=..., "
                "return_model=True) to change the dimensionality."
            )
        if impute is not None:
            # honor impute= on the reuse path too (it was previously
            # silently ignored here, falling back to PPCA -- QC 2026-07)
            data = _impute_format(data, impute)
        steps = getattr(pipeline, 'steps', None)
        if steps and steps[-1][0] == 'cluster' and getattr(pipeline, 'is_fitted', False):
            # `analyze` returns the TRANSFORMED DATA, never cluster labels
            # (see the cluster=/pipeline= docs above) -- but a cluster-bearing
            # Pipeline's own .transform ends AT the cluster step and would
            # hand back the labels (QC 2026-07: the fit-then-reuse workflow
            # silently flipped from (n, ndims) data to a 1-D label sequence).
            # Apply the fitted non-cluster steps only; the labels stay
            # recoverable via pipeline.named_steps['cluster'].transform(...).
            from ..core.pipeline import _step_transform
            result = data
            for _name, step in steps[:-1]:
                result = _step_transform(step, result)
        else:
            result = pipeline.transform(data)
        if internal and not isinstance(result, list):
            result = [result]
        return (result, pipeline) if return_model else result

    # False disables a stage, exactly like None (the documented contract for
    # align= -- 'If False or None, no alignment is applied' -- and how
    # normalize=False has always behaved; QC 2026-07: align=False previously
    # crashed with 'unknown model: False'). align=True (the retired pre-1.0
    # boolean) gets the curated removal message on EVERY path, not just the
    # legacy chain.
    if align is True:
        raise ValueError("align=True was removed in hypertools 1.0; specify the "
                         "algorithm instead, e.g. align='hyper' or align='SRM'.")
    if manip is False:
        manip = None
    if reduce is False:
        reduce = None
    if align is False:
        align = None
    if cluster is False:
        cluster = None

    if ndims is not None and reduce is None:
        warnings.warn(
            f"ndims={ndims!r} was passed but reduce= is None/False, so NO "
            "dimensionality reduction will be performed; also pass e.g. "
            f"reduce='IncrementalPCA' to reduce to {ndims!r} dimensions."
        )

    if impute is not None:
        # honor impute= wherever format_data imputes (QC 2026-07: it was
        # previously threaded only through the normalize stage, so
        # reduce-only calls silently fell back to PPCA): fill missing values
        # with the requested model up front, before any stage runs.
        data = _impute_format(data, impute)

    if return_model or manip is not None or cluster is not None:
        # return_model=True (any combination), or the NEW manip=/cluster=
        # cross-module kwargs: assemble and run a Pipeline (in canonical
        # order, GH #153) via hypertools.core.pipeline.build_pipeline, so
        # return_model=True hands back a genuinely fit-once-reusable
        # Pipeline (see hypertools.core.pipeline._DispatchStep).
        from ..core.pipeline import build_pipeline
        pipe = build_pipeline(manip=manip, normalize=normalize, reduce=reduce,
                              ndims=ndims, align=align, cluster=cluster,
                              random_state=random_state)
        if cluster is None:
            result = pipe.fit_transform(data)
        else:
            # analyze returns the TRANSFORMED DATA, not cluster labels, even
            # when cluster= is given -- the labels live in the fitted
            # 'cluster' step of the returned Pipeline (see this function's
            # docstring). Fit the steps one at a time (exactly what
            # Pipeline.fit_transform does), CAPTURING the pre-cluster data
            # along the way: re-deriving it afterwards via .transform breaks
            # for embedding models with no transform method (TSNE / MDS /
            # SpectralEmbedding -- QC 2026-07: these crashed with
            # "'TSNE' object has no attribute 'transform'" after all the
            # fitting work was done).
            result = data
            for name, step in pipe.steps:
                if name == 'cluster':
                    step.fit(result)
                else:
                    result = step.fit_transform(result)
            # every step is now fitted; mark the Pipeline itself fitted the
            # same way Pipeline.fit_transform does, so it is reusable via
            # .transform / pipeline= (there is no public "mark fitted" API)
            pipe._is_fitted = True
        if internal and not isinstance(result, list):
            result = [result]
        return (result, pipe) if return_model else result

    # legacy path (normalize=/reduce=/ndims=/align=/impute=/internal= only,
    # return_model=False): the EXACT SAME normalize -> reduce -> align chain
    # analyze() has always run, byte-identical for every existing caller
    # (hyp.plot, hyp.load, and every pre-1.0 script/test).
    return aligner(reducer(normalizer(data, normalize=normalize, internal=internal,
                                      impute=impute),
                   reduce=reduce, ndims=ndims, internal=internal,
                   random_state=random_state), align=align)
