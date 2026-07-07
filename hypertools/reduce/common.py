"""Base class for hypertools reducers (scikit-learn compatible).

A `Reducer` wraps a scikit-learn-style dimensionality-reduction/manifold-
learning model class or instance, mirroring
`hypertools.manip.common.Manipulator`'s already-constructed-vs-bare-class
handling, but specialized to `hypertools.reduce.reduce`'s stack-once-fit-once
recipe: `fit_transform`/`transform` operate on an already-stacked 2D array
(row-concatenated across datasets), not a list.

Mixture / soft-clustering models (`GaussianMixture`, `BayesianGaussianMixture`,
`LatentDirichletAllocation`, `NMF`) are special-cased to return
`(n_samples, n_components)` membership-proportion matrices -- REUSING
`hypertools.cluster.cluster.mixture_proportions`/`normalize_membership_rows`
(the exact logic `hyp.cluster` uses for these models) rather than
reimplementing it -- so `hyp.reduce(x, reduce='GaussianMixture', ndims=3)`
returns properly normalized membership proportions (GH #174).
"""
import inspect

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.decomposition import (
    PCA, FastICA, IncrementalPCA, KernelPCA, FactorAnalysis, TruncatedSVD,
    SparsePCA, MiniBatchSparsePCA, DictionaryLearning, MiniBatchDictionaryLearning,
)
from sklearn.manifold import TSNE, MDS, SpectralEmbedding, LocallyLinearEmbedding, Isomap

from ..cluster.common import MIXTURES as mixture_models
from ..cluster.common import mixture_proportions, normalize_membership_rows


#: name -> class registry for the decomposition/manifold-learning reducers
#: `hyp.reduce`'s `reduce=` string spec has always supported. Kept identical
#: (same keys, same classes) to the pre-1.0 `models` dict that used to live
#: in `hypertools.reduce.reduce` -- `hypertools.core.model._build_registry`
#: and `hypertools.core.pipeline._resolve_step_class` both import it (via
#: `hypertools.reduce.reduce.models`, re-exported there for backward
#: compatibility) and must keep working unchanged.
models = {
    'PCA': PCA,
    'IncrementalPCA': IncrementalPCA,
    'SparsePCA': SparsePCA,
    'MiniBatchSparsePCA': MiniBatchSparsePCA,
    'KernelPCA': KernelPCA,
    'FastICA': FastICA,
    'FactorAnalysis': FactorAnalysis,
    'TruncatedSVD': TruncatedSVD,
    'DictionaryLearning': DictionaryLearning,
    'MiniBatchDictionaryLearning': MiniBatchDictionaryLearning,
    'TSNE': TSNE,
    'Isomap': Isomap,
    'SpectralEmbedding': SpectralEmbedding,
    'LocallyLinearEmbedding': LocallyLinearEmbedding,
    'MDS': MDS,
}

#: `models` plus the mixture/soft-clustering models (GH #174) -- this is the
#: registry `reduce=`'s string spec resolves against. `'UMAP'` is
#: deliberately excluded (resolved lazily by `resolve_reducer`; importing
#: `umap` eagerly triggers numba JIT compilation that adds seconds to
#: `import hypertools` even when UMAP is never used).
REDUCERS = {**models, **mixture_models}

#: the six torch-backed autoencoder reducers (GH #162,
#: `hypertools.reduce.autoencoders`) -- deliberately excluded from
#: `REDUCERS` and resolved lazily by `resolve_reducer` (mirroring
#: `'UMAP'`), so `import hypertools` never requires `torch` to be
#: installed. `torch` ships as the optional `[torch]` extra.
AUTOENCODER_NAMES = (
    'Autoencoder', 'DeepAutoencoder', 'SparseAutoencoder',
    'ConvolutionalAutoencoder', 'SequenceAutoencoder',
    'VariationalAutoencoder',
)


def resolve_reducer(name):
    """Resolve a registered reducer name to its class.

    Parameters
    ----------
    name : str
        A key of `REDUCERS` (any of `models`' or `mixture_models`' names),
        `'UMAP'`, or one of `AUTOENCODER_NAMES` (all resolved lazily via a
        local import -- see `REDUCERS`/`AUTOENCODER_NAMES`).

    Returns
    -------
    class
        The resolved scikit-learn-style reducer class.

    Raises
    ------
    KeyError
        If `name` is not a recognized reducer name.
    ImportError
        If `name` is one of `AUTOENCODER_NAMES` and `torch` is not
        installed.
    """
    if name == 'UMAP':
        from umap import UMAP
        return UMAP
    if name in AUTOENCODER_NAMES:
        try:
            from . import autoencoders
        except ImportError as e:
            raise ImportError(
                f'{name} requires torch, which is not installed; install '
                'it with pip install "hypertools[torch]"'
            ) from e
        return getattr(autoencoders, name)
    return REDUCERS[name]


class Reducer(BaseEstimator):
    """Wrap a scikit-learn-style dimensionality-reduction model.

    Parameters
    ----------
    model : class or instance
        A scikit-learn-style reducer class (constructed with `params` on
        `fit_transform`) or an already-constructed instance (used as-is;
        `params` is ignored).
    params : dict or None
        Constructor keyword arguments, used only when `model` is a bare
        (uninstantiated) class (default: None).

    Attributes
    ----------
    model_ : object or None
        The fitted underlying scikit-learn(-style) model, set after
        `fit_transform` runs (default: None, i.e. not yet fitted).
    """

    def __init__(self, model, params=None):
        self.model = model
        self.params = dict(params) if params else {}
        self.model_ = None

    @property
    def is_fitted(self):
        """Whether `fit_transform` has already been run.

        Lets a fitted `Reducer` returned from an earlier
        `hypertools.reduce.reduce.reduce(..., return_model=True)` call be
        passed back in as `reduce=` on NEW data and reuse its learned
        parameters via `transform`, without re-fitting.
        """
        return self.model_ is not None

    @staticmethod
    def _is_mixture(model):
        name = model.__name__ if inspect.isclass(model) else type(model).__name__
        return name in mixture_models

    def fit_transform(self, stacked):
        """Fit the underlying model on an already-stacked 2D array.

        Parameters
        ----------
        stacked : numpy.ndarray
            A single (row-concatenated) 2D array.

        Returns
        -------
        numpy.ndarray
            The fitted model's projection of `stacked`, or -- for mixture
            models (GH #174) -- an `(n_samples, n_components)` array of
            membership proportions.
        """
        model = self.model(**self.params) if inspect.isclass(self.model) else self.model
        if self._is_mixture(model):
            result = mixture_proportions(type(model).__name__, model, stacked)
        else:
            result = model.fit_transform(stacked)
        self.model_ = model
        return result

    def transform(self, stacked):
        """Apply the already-fitted model to new (already-stacked) data,
        without re-fitting.

        Parameters
        ----------
        stacked : numpy.ndarray
            A single (row-concatenated) 2D array.

        Returns
        -------
        numpy.ndarray
            The fitted model's projection of `stacked`, or -- for mixture
            models -- membership proportions.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If `fit_transform` has not been called yet.
        """
        if self.model_ is None:
            raise NotFittedError('must fit reducer before transforming data')
        model = self.model_
        if self._is_mixture(model):
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(stacked)
            loadings = model.transform(stacked)
            return normalize_membership_rows(loadings)
        return model.transform(stacked)
