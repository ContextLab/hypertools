#!/usr/bin/env python

import inspect
import warnings
import numpy as np
from .common import Reducer, models, REDUCERS, resolve_reducer
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
           internal=False, format_data=True):
    """
    Reduces dimensionality of an array, or list of arrays

    Parameters
    ----------
    x : Numpy array or list of arrays
        Dimensionality reduction using PCA is performed on this array.

    reduce : str, dict, class, instance, or fitted Reducer
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
        being refit. See scikit-learn specific model docs for details on
        parameters supported for each model.

    ndims : int
        Number of dimensions to reduce

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

    Returns
    ----------
    x_reduced : Numpy array or list of arrays
        The reduced data with ndims dimensionality is returned.  If the input
        is a list, a list is returned. If `return_model=True`, an
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

    # if model is None, just return data
    if reduce is None:
        return (x, None) if return_model else x

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
                raise ValueError('If passing a dictionary, pass the model as the value of the "model" key and a \
                dictionary of custom params as the value of the "params" key.')
            c_args = list(reduce.get('args', []))
            c_kwargs = dict(reduce.get('kwargs', {}))
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
                raise ValueError('If passing a dictionary, pass the model as the value of the "model" key and a \
                dictionary of custom params as the value of the "params" key.')
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

    try:
        # if the model passed is a string, make sure it's one of the supported options
        if isinstance(model_name, str):  # Remove np.string_ check as it's deprecated in NumPy 2.0
            model = _resolve_model(model_name)
        # otherwise check any custom object for necessary methods
        else:
            model = model_name
            if not hasattr(model, 'fit_transform'):
                # mixture-style estimators (GaussianMixture, etc.) have no
                # fit_transform; fit + predict_proba is an equally valid
                # reducer-like interface (GH #174)
                if not (hasattr(model, 'fit') and hasattr(model, 'predict_proba')):
                    raise AttributeError
            # a bare class won't have n_components until it's constructed;
            # only already-constructed instances are expected to have it
            if not inspect.isclass(model):
                getattr(model, 'n_components')
    except (KeyError, AttributeError):
        raise ValueError('reduce must be one of the supported options or support n_components and fit_transform \
         methods. See http://hypertools.readthedocs.io/en/latest/hypertools.tools.reduce.html#hypertools.tools.reduce \
         for supported models')

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
