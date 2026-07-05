"""SimpleImputer / KNNImputer / IterativeImputer wrappers (scikit-learn).

Each wraps the corresponding sklearn imputer's fit/transform split: `fitter`
fits the sklearn imputer on the (stacked) data, `transformer` calls its
`.transform` and splices the result back in ONLY where the input was
missing, so every non-missing entry passes through byte-identical --
sklearn's imputers operate over the whole array, but we only trust their
output at the NaN positions.

`IterativeImputer` is experimental in scikit-learn and requires explicitly
opting in via `from sklearn.experimental import enable_iterative_imputer`
before `IterativeImputer` becomes importable from `sklearn.impute`; done
lazily inside `_iterative_fitter` so importing `hypertools.impute` never
triggers sklearn's experimental-API warning unless this imputer is used.
"""
import numpy as np
import pandas as pd

from .common import Imputer


def _splice(x, filled):
    mask = np.isnan(x)
    out = x.copy()
    out[mask] = filled[mask]
    return out


def transformer(data, **kwargs):
    imputer = kwargs['imputer']
    x = data.to_numpy(dtype=float)
    filled = imputer.transform(x)
    out = _splice(x, filled)
    return pd.DataFrame(out, index=data.index, columns=data.columns)


def _simple_fitter(data, **kwargs):
    from sklearn.impute import SimpleImputer as _SimpleImputer

    strategy = kwargs.get('strategy', 'mean')
    fill_value = kwargs.get('fill_value', None)
    imputer = _SimpleImputer(strategy=strategy, fill_value=fill_value)
    imputer.fit(data.to_numpy(dtype=float))
    return {'imputer': imputer}


def _knn_fitter(data, **kwargs):
    from sklearn.impute import KNNImputer as _KNNImputer

    n_neighbors = kwargs.get('n_neighbors', 5)
    weights = kwargs.get('weights', 'uniform')
    imputer = _KNNImputer(n_neighbors=n_neighbors, weights=weights)
    imputer.fit(data.to_numpy(dtype=float))
    return {'imputer': imputer}


def _iterative_fitter(data, **kwargs):
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer as _IterativeImputer

    max_iter = kwargs.get('max_iter', 10)
    random_state = kwargs.get('random_state', None)
    imputer = _IterativeImputer(max_iter=max_iter, random_state=random_state)
    imputer.fit(data.to_numpy(dtype=float))
    return {'imputer': imputer}


class SimpleImputer(Imputer):
    """Fill scattered NaNs with a per-column statistic.

    Wraps `sklearn.impute.SimpleImputer`; only missing entries are replaced.

    Parameters
    ----------
    strategy : str
        One of 'mean', 'median', 'most_frequent', 'constant' (default: 'mean').
    fill_value : scalar or None
        Used when `strategy='constant'` (default: None).
    """

    def __init__(self, strategy='mean', fill_value=None, **kwargs):
        required = ['imputer']
        super().__init__(strategy=strategy, fill_value=fill_value, fitter=_simple_fitter,
                          transformer=transformer, data=None, required=required, **kwargs)
        self.strategy = strategy
        self.fill_value = fill_value
        self.fitter = _simple_fitter
        self.transformer = transformer
        self.data = None
        self.required = required


class KNNImputer(Imputer):
    """Fill scattered NaNs via k-nearest-neighbor averaging.

    Wraps `sklearn.impute.KNNImputer`; only missing entries are replaced.

    Parameters
    ----------
    n_neighbors : int
        Number of neighboring samples to use (default: 5).
    weights : str
        'uniform' or 'distance' (default: 'uniform').
    """

    def __init__(self, n_neighbors=5, weights='uniform', **kwargs):
        required = ['imputer']
        super().__init__(n_neighbors=n_neighbors, weights=weights, fitter=_knn_fitter,
                          transformer=transformer, data=None, required=required, **kwargs)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.fitter = _knn_fitter
        self.transformer = transformer
        self.data = None
        self.required = required


class IterativeImputer(Imputer):
    """Fill scattered NaNs via round-robin multivariate regression (MICE-style).

    Wraps `sklearn.impute.IterativeImputer` (an experimental sklearn
    estimator); only missing entries are replaced.

    Parameters
    ----------
    max_iter : int
        Maximum number of imputation rounds (default: 10).
    random_state : int or None
        Seed for reproducibility (default: None).
    """

    def __init__(self, max_iter=10, random_state=None, **kwargs):
        required = ['imputer']
        super().__init__(max_iter=max_iter, random_state=random_state, fitter=_iterative_fitter,
                          transformer=transformer, data=None, required=required, **kwargs)
        self.max_iter = max_iter
        self.random_state = random_state
        self.fitter = _iterative_fitter
        self.transformer = transformer
        self.data = None
        self.required = required
