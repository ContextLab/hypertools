#!/usr/bin/env python

import inspect
import warnings
import numpy as np
from .common import (Reducer, models, REDUCERS, AUTOENCODER_NAMES,
                     resolve_reducer)
from .._shared.helpers import *
from ..tools.format_data import format_data as formatter


def _resolve_model(model_name):
    """Look up a reduction model by name.

    Covers `models` (the classic decomposition/manifold reducers) plus the
    mixture/soft-clustering models (`GaussianMixture`,
    `BayesianGaussianMixture`, `LatentDirichletAllocation`, `NMF` -- GH
    #174) via `REDUCERS`. UMAP is resolved lazily because importing umap
    triggers numba JIT compilation that adds seconds to `import hypertools`
    even when UMAP is never used.
    """
    return resolve_reducer(model_name)


# main function
def reduce(x, reduce='IncrementalPCA', ndims=None, return_model=False,
           manip=None, normalize=None, align=None, cluster=None,
           internal=False, format_data=True, random_state=None):
    """
    Reduces dimensionality of an array, or list of arrays

    Parameters
    ----------
    x : Numpy array, Pandas DataFrame, text (list of strings), or list of
        arrays/DataFrames
        The data to reduce. Lists are stacked and reduced in one SHARED
        space (a single model fit on the row-concatenated data), so all
        datasets in a list must have the same number of columns.

    reduce : str, dict, class, instance, fitted Reducer, False, or None
        Decomposition/manifold learning model to use.  Models supported: PCA,
        IncrementalPCA, SparsePCA, MiniBatchSparsePCA, KernelPCA, FastICA,
        FactorAnalysis, TruncatedSVD, DictionaryLearning, MiniBatchDictionaryLearning,
        TSNE, Isomap, SpectralEmbedding, LocallyLinearEmbedding, MDS and UMAP,
        plus the mixture (soft-clustering) models GaussianMixture,
        BayesianGaussianMixture, LatentDirichletAllocation and NMF -- for
        these, the returned array holds (n_samples, ndims) membership
        proportions rather than a projection (GH #174). Also supports the
        six torch-backed autoencoder reducers (GH #162,
        `hypertools.reduce.autoencoders`): Autoencoder, DeepAutoencoder,
        SparseAutoencoder, ConvolutionalAutoencoder, SequenceAutoencoder,
        and VariationalAutoencoder -- these require the optional `torch`
        dependency (`pip install "hypertools[torch]"`); resolving one of
        these names without `torch` installed raises a friendly
        `ImportError`. Can be passed as a
        string, a bare (uninstantiated) scikit-learn-style class, an
        already-constructed instance, the canonical dict spec
        `{'model': ..., 'args': [...], 'kwargs': {...}}`, or the LEGACY
        dict spec `{'model' : 'PCA', 'params' : {'whiten' : True}}`
        (accepted for backward compatibility, but emits a
        `DeprecationWarning`). A previously-fitted `Reducer` (as returned
        by `return_model=True`) is applied via `.transform` instead of
        being refit; models without an out-of-sample transform (TSNE, MDS,
        SpectralEmbedding) cannot embed new data this way, so reusing their
        fitted `Reducer` raises `NotImplementedError` explaining the refit.
        `None` or `False` skips the reduction entirely and returns the
        input unchanged. See scikit-learn specific model docs for details
        on parameters supported for each model.

    ndims : int or None
        Number of dimensions to reduce to. If None (the default), or if
        every dataset already has <= ndims columns, no model is fit and
        the (formatted) input is returned unchanged -- reduce() never
        expands or rotates data at full dimensionality.

    return_model : bool
        If True, also return the fitted model: the fitted `Reducer` wrapper
        when only the `reduce` stage ran, or a fitted `hypertools.Pipeline`
        when `manip=`/`normalize=`/`align=`/`cluster=` made multiple stages
        run (default: False).

    manip, normalize, align, cluster : model spec or None
        Cross-module stage kwargs (GH #138): when any of these is given,
        the other stages also run (via
        `hypertools.core.pipeline.build_pipeline`), in the canonical order
        `manip -> normalize -> reduce -> align -> cluster` (GH #153), with
        this function's own `reduce=`/`ndims=` slotted in at the reduce
        stage (default: None for all four, i.e. only `reduce` runs).

    internal : bool
        (Internal use, e.g. by `hyp.plot`/`hyp.analyze`) if True, always
        return a list even when the input was a single dataset (default:
        False).

    format_data : bool
        Whether or not to first call the format_data function (default: True).

    random_state : int, RandomState, or None
        Seed for reproducibility. Injected into the reduction model's
        constructor when it accepts a `random_state` (UMAP, TSNE, MDS,
        FastICA, the mixture models, ...); ignored for deterministic models
        (PCA, IncrementalPCA, ...) and for an already-constructed model
        instance you pass in (configure that yourself). An explicit
        `random_state` in a dict spec's `kwargs` takes precedence
        (default: None).

    Returns
    ----------
    x_reduced : Numpy array or list of arrays
        The reduced data with ndims dimensionality is returned. A list is
        returned when the input is a list of two or more datasets (or when
        `internal=True`); a single dataset -- even inside a one-element
        list -- comes back as a bare array. If `return_model=True`, an
        `(x_reduced, model)` tuple is returned instead.

    """
    # validate ndims up front (QC 2026-07): a non-int silently hit a
    # `TypeError: '<=' not supported between int and str`, and ndims<=0
    # silently reduced to 0/negative columns. bool is an int subclass but is
    # never a valid dimension count.
    if ndims is not None:
        if isinstance(ndims, bool) or not isinstance(ndims, (int, np.integer)):
            raise ValueError(
                f"ndims must be a positive integer or None; got {ndims!r}")
        if ndims < 1:
            raise ValueError(f"ndims must be >= 1; got {ndims}")

    # False is an explicit "skip this stage", for every stage kwarg, exactly
    # like None (release-audit contract) -- normalize it up front so
    # plot()/analyze() can thread reduce=False (and friends) through
    if reduce is False:
        reduce = None
    manip = None if manip is False else manip
    normalize = None if normalize is False else normalize
    align = None if align is False else align
    cluster = None if cluster is False else cluster

    # a whole already-fitted Pipeline handed back as reduce= (e.g. the model
    # from an earlier cross-module return_model=True call) is reused as-is via
    # .transform, BEFORE the cross-module branch below -- otherwise it would be
    # wrapped in a fresh Reducer and crash (QC 2026-07). Any redundant stage
    # kwargs are warned about + ignored (the Pipeline already encodes them).
    from ..core.shared import is_reused_pipeline
    if is_reused_pipeline(reduce, {'manip': manip, 'normalize': normalize,
                                   'align': align, 'cluster': cluster}, 'reduce'):
        result = reduce.transform(x)
        return (result, reduce) if return_model else result

    # cross-module kwargs (#138): assemble and run a Pipeline (in canonical
    # order, #153) instead of the single-stage path below whenever another
    # stage is requested. Lazy import avoids a reduce<->core.pipeline cycle
    # (core.pipeline itself lazily imports reduce.reduce).
    if any(stage is not None for stage in (manip, normalize, align, cluster)):
        from ..core.pipeline import build_pipeline
        pipeline = build_pipeline(manip=manip, normalize=normalize,
                                   reduce=reduce, ndims=ndims,
                                   align=align, cluster=cluster)
        result = pipeline.fit_transform(x)
        return (result, pipeline) if return_model else result

    # if model is None (or False, normalized above), just return data
    if reduce is None:
        return (x, None) if return_model else x

    # set by the canonical dict-spec branch below when a TSNE instance is
    # constructed without a user-supplied perplexity (see the small-dataset
    # clamp further down, F11-reduce-describe-002)
    tsne_perplexity_unset = False

    # an already-fitted Reducer (returned from an earlier
    # return_model=True call) is reused via `transform`, never refit
    if isinstance(reduce, Reducer) and reduce.is_fitted:
        fitted_n_components = getattr(reduce.model_, 'n_components', None)
        if (ndims is not None) and (fitted_n_components is not None) and (ndims != fitted_n_components):
            warnings.warn('Unequal values passed to dims and n_components. Using the already-fitted model.')
        if format_data:
            x = formatter(x, ppca=True)
        x_reduced, fitted = reduce_list(x, None, reuse=reduce)
        result = x_reduced if (internal or len(x_reduced) > 1) else x_reduced[0]
        return (result, fitted) if return_model else result

    elif isinstance(reduce, str):  # Remove np.string_ check as it's deprecated in NumPy 2.0
        model_name = reduce
        model_params = {
            'n_components': ndims
        }

    elif isinstance(reduce, dict):
        if 'args' in reduce or 'kwargs' in reduce:
            # canonical 1.0 dict spec: {'model': ..., 'args': [...], 'kwargs': {...}}
            try:
                c_model = reduce['model']
            except KeyError:
                raise ValueError(
                    "invalid reduce dict spec: pass the model as the value "
                    "of the 'model' key, with optional constructor arguments "
                    "under 'args' (positional) and 'kwargs' (keyword), e.g. "
                    "{'model': 'PCA', 'kwargs': {'whiten': True}} (the "
                    "legacy 'params' key is also accepted).")
            c_args = list(reduce.get('args', []))
            c_kwargs = dict(reduce.get('kwargs', {}))
            # remember whether the user left TSNE's perplexity at its
            # sklearn default: reduce() clamps it for small datasets below
            # (F11-reduce-describe-002), but must never touch a value the
            # user set themselves
            tsne_perplexity_unset = (
                (c_model == 'TSNE'
                 or getattr(c_model, '__name__', None) == 'TSNE'
                 or type(c_model).__name__ == 'TSNE')
                and 'perplexity' not in c_kwargs)
            if isinstance(c_model, str):
                c_model = _resolve_model(c_model)
            # inject n_components=ndims when the model accepts it and the user
            # did not set it themselves. Without this, the canonical dict form
            # pre-built the instance with n_components=None (its default) and
            # `ndims` was silently ignored -- e.g. reduce={'model':'PCA',
            # 'kwargs':{'whiten':True}}, ndims=2 returned the FULL-dim data
            # (QC 2026-07). The bare-string and legacy 'params' forms already
            # applied ndims; this brings the documented dict form in line.
            if (ndims is not None and 'n_components' not in c_kwargs
                    and inspect.isclass(c_model)
                    and 'n_components' in inspect.signature(c_model).parameters):
                c_kwargs['n_components'] = ndims
            # same for a top-level random_state (QC 2026-07 reproducibility): the
            # dict spec constructs its instance here, so inject before that
            if (random_state is not None and 'random_state' not in c_kwargs
                    and inspect.isclass(c_model)
                    and 'random_state' in inspect.signature(c_model).parameters):
                c_kwargs['random_state'] = random_state
            # construct immediately; the resulting instance flows through
            # the same already-constructed-instance handling below as a
            # bare instance passed directly as `reduce=`
            model_name = c_model(*c_args, **c_kwargs)
            model_params = {
                'n_components': ndims
            }
        else:
            try:
                model_name = reduce['model']
                model_params = reduce['params']
            except KeyError:
                raise ValueError(
                    "invalid reduce dict spec: pass the model as the value "
                    "of the 'model' key, with optional constructor arguments "
                    "under 'args' (positional) and 'kwargs' (keyword), e.g. "
                    "{'model': 'PCA', 'kwargs': {'whiten': True}} (the "
                    "legacy 'params' key is also accepted).")
            # LEGACY form (dev-1.0/fork): accepted for backward
            # compatibility, but deprecated in favor of the canonical
            # {'model', 'args', 'kwargs'} triple above.
            warnings.warn(
                "{'model': ..., 'params': {...}} is deprecated; use "
                "{'model': ..., 'args': [...], 'kwargs': {...}} instead",
                DeprecationWarning, stacklevel=2)

    else:
        # handle other possibilities below: a bare (uninstantiated) custom
        # model class, or an already-constructed custom model instance
        model_name = reduce
        model_params = {
            'n_components': ndims
        }

    supported_names = sorted(list(REDUCERS) + ['UMAP'] + list(AUTOENCODER_NAMES))
    # if the model passed is a string, make sure it's one of the supported options
    if isinstance(model_name, str):  # Remove np.string_ check as it's deprecated in NumPy 2.0
        try:
            model = _resolve_model(model_name)
        except KeyError:
            # name the offending value and, for near-misses like 'umap',
            # suggest the correctly-cased registry name
            # (F11-reduce-describe-005)
            match = next((name for name in supported_names
                          if name.lower() == model_name.lower()), None)
            hint = f" (did you mean {match!r}?)" if match else ""
            raise ValueError(
                f"unknown reduce model {model_name!r}{hint}; supported "
                f"names: {', '.join(supported_names)}. A scikit-learn style "
                f"class or instance can also be passed directly.") from None
    # otherwise check any custom object for necessary methods
    else:
        model = model_name
        if not (hasattr(model, 'fit_transform')
                # mixture-style estimators (GaussianMixture, etc.) have no
                # fit_transform; fit + predict_proba is an equally valid
                # reducer-like interface (GH #174)
                or (hasattr(model, 'fit') and hasattr(model, 'predict_proba'))):
            raise ValueError(
                f"invalid reduce model {model!r} (type "
                f"{type(model).__name__}): a reduce spec must be a supported "
                f"model name, a dict spec, a fitted Reducer, or a "
                f"scikit-learn style class/instance with fit_transform (or "
                f"fit + predict_proba) and n_components. Supported names: "
                f"{', '.join(supported_names)}.")
        # a bare class won't have n_components until it's constructed;
        # only already-constructed instances are expected to have it
        if not inspect.isclass(model) and not hasattr(model, 'n_components'):
            raise ValueError(
                f"invalid reduce model {model!r} (type "
                f"{type(model).__name__}): the instance has no n_components "
                f"attribute; construct it with an n_components (e.g. "
                f"PCA(n_components=3)) or pass the bare class together with "
                f"ndims=.")

    # an already-constructed instance is used as-is: it's already configured,
    # so we must not re-construct it or clobber its params below
    model_is_instance = not inspect.isclass(model)

    # check for multiple values from n_components & ndims args
    if model_is_instance:
        instance_n_components = getattr(model, 'n_components', None)
        if (ndims is not None) and (instance_n_components is not None) and (ndims != instance_n_components):
            warnings.warn('Unequal values passed to dims and n_components. Using the already-configured model instance.')
        model_params['n_components'] = instance_n_components if instance_n_components is not None else ndims
    elif 'n_components' in model_params:
        if (ndims is None) or (ndims == model_params['n_components']):
            pass
        else:
            warnings.warn('Unequal values passed to dims and n_components. Using ndims parameter.')
            model_params['n_components'] = ndims
    else:
        model_params['n_components'] = ndims

    # convert to common format
    if format_data:
        x = formatter(x, ppca=True)

    # if ndims/n_components is not passed or all data is < ndims-dimensional, just return it
    # (unwrap a single-dataset input to a bare array, matching the reduced-data
    # path below and the fitted-model reuse path above -- QC 2026-07: this early
    # return used to hand back a 1-element LIST for a single array, so the return
    # type flipped between ndarray and list depending on ndims).
    if model_params['n_components'] is None or all([i.shape[1] <= model_params['n_components'] for i in x]):
        result = x if (internal or len(x) > 1) else x[0]
        return (result, None) if return_model else result

    # Handle empty arrays and type conversion
    if isinstance(x, list):
        _require_equal_columns(x)
    stacked_x = np.vstack([np.asarray(arr, dtype=np.float64) for arr in x])

    if stacked_x.shape[0] == 1:
        warnings.warn('Cannot reduce the dimensionality of a single row of'
                      ' data. Return zeros length of ndims')
        result = [np.zeros((1, model_params['n_components']), dtype=np.float64)]
        result = result if (internal or len(x) > 1) else result[0]
        return (result, None) if return_model else result

    elif stacked_x.shape[0] < model_params['n_components']:
            warnings.warn('The number of rows in your data is less than ndims.'
                          ' The data will be reduced to the number of rows.')
            model_params['n_components'] = stacked_x.shape[0]
            if model_is_instance:
                model.n_components = stacked_x.shape[0]

    # reproducibility (QC 2026-07): inject a top-level `random_state` into the
    # model's constructor kwargs when the model accepts it and the user did not
    # set it themselves -- so `hyp.reduce(x, reduce='UMAP', random_state=1)`
    # gives repeatable embeddings. Already-constructed instances are left as the
    # user configured them.
    if (random_state is not None and not model_is_instance
            and 'random_state' not in model_params
            and 'random_state' in inspect.signature(model).parameters):
        model_params['random_state'] = random_state

    # sklearn TSNE's default perplexity (30) requires n_samples > 30, so
    # small datasets crashed on a parameter the user never set
    # (F11-reduce-describe-002). When hypertools constructed (or is about to
    # construct) the TSNE itself and the user left perplexity unset, clamp
    # it to a workable value; user-supplied perplexities are never touched.
    n_rows = stacked_x.shape[0]
    if n_rows <= 30:
        clamped_perplexity = max(1.0, (n_rows - 1) / 3.0)
        tsne_warning = (
            f"TSNE's default perplexity (30) must be less than the number "
            f"of observations ({n_rows}); using "
            f"perplexity={clamped_perplexity:g} instead. Set it yourself "
            "with reduce={'model': 'TSNE', 'kwargs': {'perplexity': ...}}.")
        if (not model_is_instance
                and getattr(model, '__name__', None) == 'TSNE'
                and 'perplexity' not in model_params):
            warnings.warn(tsne_warning, UserWarning)
            model_params['perplexity'] = clamped_perplexity
        elif (model_is_instance and tsne_perplexity_unset
              and type(model).__name__ == 'TSNE'
              and getattr(model, 'perplexity', 0) >= n_rows):
            # the canonical dict spec constructed this instance above,
            # before the data size was known
            warnings.warn(tsne_warning, UserWarning)
            model.perplexity = clamped_perplexity

    # pin MDS's changing sklearn defaults (n_init: 4 -> 1 in sklearn 1.9;
    # init: 'random' -> 'classical_mds' in 1.10) to today's values so
    # default MDS results stay stable across sklearn upgrades and default
    # use is FutureWarning-free (F11-reduce-describe-016)
    if not model_is_instance and getattr(model, '__name__', None) == 'MDS':
        mds_params = inspect.signature(model).parameters
        if 'n_init' in mds_params:
            model_params.setdefault('n_init', 4)
        if 'init' in mds_params:
            model_params.setdefault('init', 'random')

    # initialize model: bare classes are constructed with model_params;
    # already-configured instances are used as-is
    if not model_is_instance:
        model = model(**model_params)

    # reduce data
    x_reduced, fitted = reduce_list(x, model)

    # return data
    if internal or len(x_reduced) > 1:
        result = x_reduced
    else:
        result = x_reduced[0]
    return (result, fitted) if return_model else result


# sub functions
def _require_equal_columns(x):
    """Raise a hypertools-level error for ragged dataset lists.

    The stack-once-fit-once recipe (`reduce_list`) row-concatenates every
    dataset and fits ONE model, which requires a shared feature space;
    without this check, ragged lists died inside `numpy.vstack` with a raw
    concatenation error that never mentioned datasets or columns
    (F11-reduce-describe-007).
    """
    widths = [np.atleast_2d(np.asarray(xi)).shape[1] for xi in x]
    if len(set(widths)) > 1:
        raise ValueError(
            f"cannot reduce a list of datasets with different numbers of "
            f"columns (got column counts {widths}): the datasets are "
            f"stacked and fit in one SHARED space, which requires the same "
            f"columns in every dataset. Bring them to a common set of "
            f"columns first (e.g. hyp.align, or pad/trim the features).")


def reduce_list(x, model, reuse=None):
    """Helper function to reduce a list of arrays.

    Stacks `x` (row-concatenated) into one array, fits (or reuses an
    already-fitted `Reducer` on) that single stacked array -- so the
    reduction is comparable across datasets -- and splits the result back
    into a list matching `x`'s per-dataset row counts.

    Parameters
    ----------
    x : list of numpy.ndarray
        Datasets to stack, reduce, and split back apart.

    model : object or None
        An already-configured (unfitted) scikit-learn-style reducer
        instance to fit. Ignored when `reuse` is given.

    reuse : Reducer or None
        An already-fitted `Reducer` to `transform` `x` with instead of
        fitting a new model (the `return_model=True` reuse path -- an
        already-fitted model is applied via `.transform`, never refit).

    Returns
    -------
    (list of numpy.ndarray, Reducer)
        The reduced (or reused-transform) pieces, split back to match `x`'s
        structure, and the (newly fitted, or reused) `Reducer` wrapper.
    """
    # Ensure all arrays are float64 for consistent handling
    x = [np.asarray(arr, dtype=np.float64) for arr in x]
    _require_equal_columns(x)
    split = np.cumsum([len(xi) for xi in x])[:-1]
    stacked = np.vstack(x)

    # Handle potential NaN values
    if np.any(np.isnan(stacked)):
        warnings.warn('NaN values detected in input data. These may affect the reduction results.')

    if reuse is not None:
        fitted = reuse
        transformed = np.asarray(fitted.transform(stacked))
    else:
        fitted = Reducer(model)
        transformed = np.asarray(fitted.fit_transform(stacked))

    x_r = np.vsplit(transformed, split)
    if len(x) > 1:
        return [xi for xi in x_r], fitted
    else:
        return [x_r[0]], fitted
