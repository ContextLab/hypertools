import warnings
import six
import sys
import numpy as np
import pandas as pd
from ppca import PPCA
from .data.format import format_data
from .core.configurator import get_default_options
from .align.align import srm, hyper, procrustes

from umap import UMAP
import sklearn.decomposition as decomposition
import sklearn.manifold as manifold
import sklearn.feature_extraction.text as text
import sklearn.cluster as cluster
import sklearn.mixture as mixture


# import all model-like classes within a sklearn-like module; return a list of model names
def import_sklearn_models(module):
    models = [d for d in dir(module) if hasattr(eval(f'module.{d}'), 'fit_transform')]
    for m in models:
        exec(f'from {module.__name__} import {m}', globals())
    return models


align_models = ['srm', 'hyper', 'procrustes']

reduce_models = ['UMAP']
reduce_models.extend(import_sklearn_models(decomposition))
reduce_models.extend(import_sklearn_models(manifold))

cluster_models = import_sklearn_models(cluster)

mixture_models = import_sklearn_models(mixture)
mixture_models.extend(['LatentDirichletAllocation', 'NMF'])

text_vectorizers = import_sklearn_models(text)

# source: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html
interpolation_models = ['linear', 'time', 'index', 'pad', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'spline',
                        'barycentric', 'polynomial']

defaults = get_default_options()


# make a function work for either a single object or a list of objects by calling the function on each element
def list_generalizer(f):
    @functools.wraps(f)
    def wrapped(data, **kwargs):
        if type(data) == list:
            return [f(d, **kwargs) for d in data]
        else:
            return f(data, **kwargs)

    return wrapped


# coerce the data passed into the function into a pandas dataframe or a list of dataframes
@list_generalizer
def funnel(f):
    @functools.wraps(f)
    def wrapped(data, **kwargs):
        return f(format_data(data, **kwargs), **kwargs)

    return wrapped


# helper function for filling in missing data (not a decorator)
@funnel
def fill_missing(data, **kwargs):
    if 'interp_kwargs' in kwargs.keys():
        interp_kwargs = kwargs.pop('interp_kwargs', None)
    else:
        interp_kwargs = {}

    if len(interp_kwargs) == 0:
        return data

    if ('apply_ppca' in interp_kwargs.keys()) and interp_kwargs['apply_ppca']:
        covariance_model = PPCA()
        covariance_model.fit(data.values)
        data.values = covariance_model.transform()
    interp_kwargs.pop('apply_ppca', None)

    if len(interp_kwargs) == 0:
        return data
    else:
        return data.interpolate(**interp_kwargs)


# fill in missing data by applying PPCA (to fill in isolated missing entries within a row) and then by interpolating
def interpolate(f):
    @functools.wraps(f)
    def wrapped(data, **kwargs):
        return f(fill_missing(data, **kwargs), **kwargs)

    return wrapped


# intercept data passed to a function by stacking (or unstacking) the dataframes, applying the given function,
# and then inverting the stack/unstack operation (unless return_override is True)
def stack_handler(apply_stacked=False, return_override=False):
    # noinspection PyUnusedLocal
    @interpolate
    def format_interp_stack_extract(data, keys=None, **kwargs):
        stacked_data = pandas_stack(data, keys=keys)
        vals = stacked_data.values
        return vals, stacked_data

    def decorator(f):
        @functools.wraps(f)
        def wrapped(data, **kwargs):
            def returner(x, rmodel=None, rreturn_model=False):
                if rreturn_model:
                    return rmodel, x
                else:
                    return x

            if 'keys' not in kwargs.keys():
                kwargs['keys'] = None

            if 'stack' not in kwargs.keys():
                kwargs['stack'] = False

            return_model = (not return_override) and ('return_model' in kwargs.keys()) and kwargs['return_model']
            if not return_model:
                kwargs.pop('return_model', None)

            keys = kwargs.pop('keys', None)
            stack = kwargs.pop('stack', None)

            vals, stacked_data = format_interp_stack_extract(data, keys=keys, **kwargs)
            unstacked_data = pandas_unstack(stacked_data)

            # ignore sklearn warnings...this should be written more responsibly :)
            warnings.simplefilter('ignore')

            if apply_stacked:
                transformed = f(stacked_data, **kwargs)
                if return_override:
                    return transformed

                if return_model:
                    model, transformed = transformed
                else:
                    model = None

                transformed = pd.DataFrame(data=transformed, index=stacked_data.index,
                                           columns=np.arange(transformed.shape[1]))
                if stack:
                    return returner(transformed, rmodel=model, rreturn_model=return_model)
                else:
                    return returner(pandas_unstack(transformed), rmodel=model, rreturn_model=return_model)
            else:
                transformed = f([x.values for x in unstacked_data], **kwargs)
                if return_override:
                    return transformed

                if return_model:
                    model, transformed = transformed
                else:
                    model = None

                if stack:
                    return returner(pd.DataFrame(data=np.vstack(transformed), index=stacked_data.index), rmodel=model,
                                    rreturn_model=return_model)
                else:
                    return returner(
                        [pd.DataFrame(data=v, index=unstacked_data[i].index) for i, v in enumerate(transformed)],
                        rmodel=model, rreturn_model=return_model)

        return wrapped

    return decorator


# ensure that the named algorithm is contained within the list of approved algorithms of the given type
def module_checker(modules=None, alg_list=None):
    if modules is None:
        modules = []
    if alg_list is None:
        alg_list = []

    def decorator(f):
        @functools.wraps(f)
        def wrapped(data, **kwargs):
            if 'algorithm' not in kwargs.keys():
                algorithm = defaults[f.__name__]['algorithm']
            else:
                algorithm = kwargs.pop('algorithm', None)

            if is_text(algorithm):
                # security check to prevent executing arbitrary code
                verified = False
                if len(alg_list) > 0:
                    assert any([algorithm in eval(f'{a}_models') for a in alg_list]), f'Unknown {f.__name__} ' \
                                                                                      f'algorithm: {algorithm}'
                    verified = True
                if not verified:
                    assert algorithm in eval(f'{f.__name__}_models'), f'Unknown {f.__name__} algorithm: {algorithm}'
                algorithm = eval(algorithm)

            # make sure a function from the appropriate module is being passed
            if len(modules) > 0:
                assert any([m in algorithm.__module__ for m in modules]), f'Unknown {f.__name__} ' \
                                                                          f'algorithm: {algorithm.__name__}'

            kwargs['algorithm'] = algorithm
            return f(data, **kwargs)

        return wrapped

    return decorator


# unstack the data, apply the given function, then re-stack if needed
@stack_handler(apply_stacked=False)
def unstack_apply(data, **kwargs):
    assert 'algorithm' in kwargs.keys(), 'must specify algorithm'
    return algorithm(data, **kwargs)


# stack the data, apply the given function, then unstack if needed
@stack_handler(apply_stacked=True)
def stack_apply(data, **kwargs):
    assert 'algorithm' in kwargs.keys(), 'must specify algorithm'
    return algorithm(data, **kwargs)


# add in default keyword arguments (and values) specified in config.ini based on the function name
def apply_defaults(f):
    if f.__name__ in defaults.keys():
        default_args = defaults[f.__name__]
    else:
        default_args = {}

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        for k, v in kwargs:
            default_args[k] = v
        return f(*args, **default_args)

    return wrapped
