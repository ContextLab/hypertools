#!/usr/bin/env python
"""hyp.analyze: the classic manip -> normalize -> reduce -> align -> cluster
pipeline dispatcher (GH #138 cross-module kwargs, GH #227 `pipeline=` reuse).
"""

from ..reduce.reduce import reduce as reducer
from .align import align as aligner
from .normalize import normalize as normalizer


_STAGE_KWARG_NAMES = ('manip', 'normalize', 'reduce', 'align', 'cluster')


def analyze(data, manip=None, normalize=None, reduce=None, ndims=None, align=None,
           cluster=None, pipeline=None, return_model=False, internal=False, impute=None):
    """
    Wrapper function for manip -> normalize -> reduce -> align -> cluster
    transformations (the canonical 1.0 pipeline order, GH #153).

    Parameters
    ----------
    data : numpy array, pandas df, or list of arrays/dfs
        The data to analyze

    manip : model spec or None
        Cross-module stage kwarg (GH #138): a `hypertools.manip` spec (a
        registry name, dict spec, class/instance, or a `list` chaining
        several -- see `hypertools.manip.manip.manip`), applied FIRST (the
        `manip` stage runs before `normalize`/`reduce`/`align`/`cluster` in
        the canonical order). `None` (default) skips this stage.

    normalize : str or False or None
        If set to 'across', the columns of the input data will be z-scored
        across lists (default). That is, the z-scores will be computed with
        with respect to column n across all arrays passed in the list. If set
        to 'within', the columns will be z-scored within each list that is
        passed. If set to 'row', each row of the input data will be z-scored.
        If set to False, the input data will be returned with no z-scoring.

    reduce : str, dict, class, instance, or fitted Reducer
        Decomposition/manifold learning model to use (default:
        'IncrementalPCA'). Models supported: PCA, IncrementalPCA, SparsePCA,
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
        Number of dimensions to reduce

    align : str, dict, False, or None
        Alignment model to bring a list of datasets into a shared space. If
        str, 'hyper' (hyperalignment) or 'SRM' (shared response model). You
        can also pass a dictionary for finer control, where 'model' specifies
        the model and 'kwargs' holds its parameters, e.g.
        align={'model': 'HyperAlign', 'kwargs': {'n_iter': 10}}. If False or
        None, no alignment is applied (default: None).

    cluster : model spec or None
        Cross-module stage kwarg (GH #138): a `hypertools.cluster` spec (a
        registry name, dict spec, or class/instance -- see
        `hypertools.cluster.cluster.cluster`), applied LAST (after `align`,
        the canonical order). `analyze` still returns the TRANSFORMED DATA
        (not cluster labels) when `cluster=` is given; the cluster labels
        themselves are retrievable from the fitted `hypertools.Pipeline`'s
        `'cluster'` step (`model.named_steps['cluster']`) when
        `return_model=True` is also passed -- pass the SAME data back
        through that step's `.transform` to recover the labels, or call
        `hypertools.cluster.cluster.cluster` directly for labels alongside
        the transformed data in one call. `None` (default) skips this
        stage.

    pipeline : hypertools.Pipeline or None
        A previously-FITTED `Pipeline` (e.g. from an earlier
        `analyze(..., return_model=True)` call) to apply to `data` via
        `.transform` -- reusing its learned parameters rather than
        re-fitting them (GH #227). Mutually exclusive with
        `manip=`/`normalize=`/`reduce=`/`align=`/`cluster=` (all must be
        left at their default of `None`) -- passing both raises
        `ValueError` naming the conflicting kwarg(s). `ndims=`/`internal=`/
        `impute=` are still honored (`internal=True` still guarantees a
        list is returned, even for a single-dataset `data`) (default: None).

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
        `format_data` stage, when normalization triggers it) with a
        different `hypertools.impute` model, e.g. 'Kalman', 'KNNImputer'
        (default: None, i.e. PPCA -- byte-compatible with pre-1.0 behavior).

    Returns
    ----------
    analyzed_data : list of numpy arrays
        The processed data. If `return_model=True`, an `(analyzed_data,
        model)` tuple is returned instead.

    """
    stage_kwargs = {'manip': manip, 'normalize': normalize, 'reduce': reduce,
                     'align': align, 'cluster': cluster}

    if pipeline is not None:
        conflicting = sorted(name for name, value in stage_kwargs.items()
                             if value is not None)
        if conflicting:
            raise ValueError(
                "pipeline= is mutually exclusive with the stage kwarg(s) "
                f"{', '.join(conflicting)} (a fitted Pipeline already "
                "encodes which stages run and their fitted parameters); "
                "pass pipeline= alone."
            )
        result = pipeline.transform(data)
        if internal and not isinstance(result, list):
            result = [result]
        return (result, pipeline) if return_model else result

    if return_model or manip is not None or cluster is not None:
        # return_model=True (any combination), or the NEW manip=/cluster=
        # cross-module kwargs: assemble and run a Pipeline (in canonical
        # order, GH #153) via hypertools.core.pipeline.build_pipeline, so
        # return_model=True hands back a genuinely fit-once-reusable
        # Pipeline (see hypertools.core.pipeline._DispatchStep).
        from ..core.pipeline import build_pipeline
        if impute is not None and normalize not in (False, None):
            # thread impute= through the same way the legacy chain below
            # does (impute at format time, BEFORE any pipeline stage runs):
            # build_pipeline's normalize stage calls
            # hypertools.tools.normalize.normalize() with no impute=, so
            # without this it always falls back to PPCA regardless of what
            # impute= was passed here. Gated on `normalize not in (False,
            # None)` to match normalize()'s own legacy gating -- format_data
            # (and therefore impute=) only runs there when normalization is
            # actually requested.
            from .format_data import format_data as formatter
            data = formatter(data, ppca=True, impute=impute)
        pipe = build_pipeline(manip=manip, normalize=normalize, reduce=reduce,
                              ndims=ndims, align=align, cluster=cluster)
        result = pipe.fit_transform(data)
        if internal and not isinstance(result, list):
            result = [result]
        return (result, pipe) if return_model else result

    # legacy path (normalize=/reduce=/ndims=/align=/impute=/internal= only,
    # return_model=False): the EXACT SAME normalize -> reduce -> align chain
    # analyze() has always run, byte-identical for every existing caller
    # (hyp.plot, hyp.load, and every pre-1.0 script/test).
    return aligner(reducer(normalizer(data, normalize=normalize, internal=internal,
                                      impute=impute),
                   reduce=reduce, ndims=ndims, internal=internal), align=align)
