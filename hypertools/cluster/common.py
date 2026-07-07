"""Base class for hypertools clusterers (scikit-learn compatible).

A `Clusterer` wraps a scikit-learn-style clustering model class or instance,
mirroring `hypertools.reduce.common.Reducer`'s already-constructed-vs-bare-
class handling, but specialized to `hypertools.cluster.cluster.cluster`'s
stack-once-fit-once recipe: `fit_transform`/`transform` operate on an
already-stacked 2D array (row-concatenated across datasets), not a list.

Hard-clustering models (KMeans, MiniBatchKMeans, AgglomerativeClustering,
Birch, FeatureAgglomeration, SpectralClustering, HDBSCAN, MeanShift, DBSCAN,
OPTICS, AffinityPropagation) return a list of per-observation cluster
labels. Mixture / soft-clustering models (GaussianMixture,
BayesianGaussianMixture, LatentDirichletAllocation, NMF) return an
`(n_samples, n_components)` membership-proportion matrix instead -- this is
the exact logic `hypertools.reduce.common.Reducer` reuses (via
`mixture_proportions`/`normalize_membership_rows`, re-exported here) so
`hyp.reduce(x, reduce='GaussianMixture', ndims=3)` and
`hyp.cluster(x, cluster='GaussianMixture')` return proportions computed by
the SAME code path (GH #174).
"""
import inspect

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
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


#: name -> class registry for the hard-clustering models `hyp.cluster`'s
#: `cluster=` string spec has always supported. `hypertools.core.model`
#: (via `hypertools.cluster.cluster.models`, re-exported there for backward
#: compatibility) and `hypertools.core.pipeline` both import this and must
#: keep working unchanged.
CLUSTERERS = {
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
    # signature-based exemption in `hypertools.cluster.cluster`)
    "MeanShift": MeanShift,
    "DBSCAN": DBSCAN,
    "OPTICS": OPTICS,
    "AffinityPropagation": AffinityPropagation,
}

#: mixture / soft-clustering models: instead of a hard label per
#: observation, these return an (n_samples, n_components) matrix of
#: membership proportions. Shared with `hypertools.reduce.common.REDUCERS`
#: (GH #174).
MIXTURES = {
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

    Shared by `hyp.cluster` (this module, via `Clusterer`) and
    `hypertools.reduce.common.Reducer` (GH #174), so `hyp.reduce(x,
    reduce='GaussianMixture', ndims=3)` returns exactly the same style of
    proportions `hyp.cluster` does, via the SAME code path.

    Parameters
    ----------
    model_name : str
        One of `MIXTURES`'s keys ('GaussianMixture',
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


class Clusterer(BaseEstimator):
    """Wrap a scikit-learn-style clustering model.

    Parameters
    ----------
    model : class or instance
        A scikit-learn-style clusterer class (constructed with `params` on
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

        Lets a fitted `Clusterer` returned from an earlier
        `hypertools.cluster.cluster.cluster(..., return_model=True)` call
        be passed back in as `cluster=` on NEW data and reuse its learned
        parameters via `transform`, without re-fitting.
        """
        return self.model_ is not None

    @staticmethod
    def _is_mixture(model):
        name = model.__name__ if inspect.isclass(model) else type(model).__name__
        return name in MIXTURES

    def fit_transform(self, stacked):
        """Fit the underlying model on an already-stacked 2D array.

        Parameters
        ----------
        stacked : numpy.ndarray
            A single (row-concatenated) 2D array.

        Returns
        -------
        list or numpy.ndarray
            A list of per-observation cluster labels for hard-clustering
            models, or -- for mixture models (GH #174) -- an
            `(n_samples, n_components)` array of membership proportions.
        """
        model = self.model(**self.params) if inspect.isclass(self.model) else self.model
        if self._is_mixture(model):
            result = mixture_proportions(type(model).__name__, model, stacked)
        else:
            model.fit(stacked)
            result = list(model.labels_)
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
        list or numpy.ndarray
            A list of per-observation cluster labels, or -- for mixture
            models -- membership proportions.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If `fit_transform` has not been called yet.
        NotImplementedError
            If the fitted model is a hard-clustering model with no
            out-of-sample `predict` (e.g. `AgglomerativeClustering`,
            `SpectralClustering`, `DBSCAN`, `OPTICS`, `HDBSCAN` -- these
            only support `fit_predict` on the data they were fit on).
        """
        if self.model_ is None:
            raise NotFittedError('must fit clusterer before transforming data')
        model = self.model_
        if self._is_mixture(model):
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(stacked)
            loadings = model.transform(stacked)
            return normalize_membership_rows(loadings)
        if hasattr(model, 'predict'):
            return list(model.predict(stacked))
        raise NotImplementedError(
            f"{type(model).__name__} has no out-of-sample prediction (no "
            f"predict method); cannot reuse a fitted {type(model).__name__} "
            f"clusterer on new data without refitting")
