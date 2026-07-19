"""Base class for hypertools clusterers (scikit-learn compatible).

A `Clusterer` wraps a scikit-learn-style clustering model class or instance,
mirroring `hypertools.reduce.common.Reducer`'s already-constructed-vs-bare-
class handling, but specialized to `hypertools.cluster.cluster.cluster`'s
stack-once-fit-once recipe: `fit_transform`/`transform` operate on an
already-stacked 2D array (row-concatenated across datasets), not a list.

Hard-clustering models (KMeans, MiniBatchKMeans, AgglomerativeClustering,
Birch, SpectralClustering, HDBSCAN, MeanShift, DBSCAN, OPTICS,
AffinityPropagation) return a list of per-observation cluster labels.
FeatureAgglomeration is the one exception: it transposes the clustering
problem and groups FEATURES (columns), so it returns one label per column
of the input, not one per row (a `UserWarning` says so whenever it is fit
-- see F13-cluster-001). Mixture / soft-clustering models (GaussianMixture,
BayesianGaussianMixture, LatentDirichletAllocation, NMF) return an
`(n_samples, n_components)` membership-proportion matrix instead -- this is
the exact logic `hypertools.reduce.common.Reducer` reuses (via
`mixture_proportions`/`normalize_membership_rows`, re-exported here) so
`hyp.reduce(x, reduce='GaussianMixture', ndims=3)` and
`hyp.cluster(x, cluster='GaussianMixture')` return proportions computed by
the SAME code path (GH #174).
"""
import inspect
import warnings

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
        # Store constructor args VERBATIM (no copy/normalization) so
        # sklearn's `get_params`/`set_params`/`clone` round-trip correctly;
        # copying params here broke `clone()` with a RuntimeError (QC 2026-07).
        # `None` is normalized to `{}` at the point of use in `fit_transform`.
        self.model = model
        self.params = params
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
            models (per-FEATURE labels for FeatureAgglomeration, which
            clusters columns -- a `UserWarning` is emitted), or -- for
            mixture models (GH #174) -- an `(n_samples, n_components)`
            array of membership proportions.

        Raises
        ------
        ValueError
            If the wrapped model is not a scikit-learn-style clusterer
            (no `fit`/`fit_predict`, or no way to extract labels).
        """
        model = self.model(**(self.params or {})) if inspect.isclass(self.model) else self.model
        if not (hasattr(model, 'fit') or hasattr(model, 'fit_predict')):
            raise ValueError(
                f"invalid cluster model {model!r} (type "
                f"{type(model).__name__}): it has no fit or fit_predict "
                f"method. Pass one of the supported model names "
                f"({', '.join(sorted(list(CLUSTERERS) + list(MIXTURES)))}), a "
                f"scikit-learn style clusterer class or instance, or a dict "
                f"spec like {{'model': 'KMeans', 'kwargs': {{...}}}}.")
        if self._is_mixture(model):
            result = mixture_proportions(type(model).__name__, model, stacked)
        elif isinstance(model, FeatureAgglomeration):
            # FeatureAgglomeration transposes the clustering problem: it
            # groups COLUMNS, so labels_ has one entry per feature, not one
            # per observation (F13-cluster-001) -- warn so the wrong-length
            # label list is never a silent surprise.
            warnings.warn(
                "FeatureAgglomeration clusters features (columns), not "
                "observations: the result has one label per column of the "
                f"input ({np.asarray(stacked).shape[1]} labels), not one per "
                "row. To cluster observations, use e.g. cluster='KMeans'.",
                UserWarning)
            model.fit(stacked)
            result = list(model.labels_)
        elif hasattr(model, 'fit_predict'):
            # covers every sklearn clusterer (ClusterMixin.fit_predict is
            # fit(X).labels_) plus fit_predict-only estimators such as an
            # sklearn Pipeline ending in a clusterer, which fits fine but
            # never exposes labels_ (F13-cluster-011)
            result = list(model.fit_predict(stacked))
        else:
            model.fit(stacked)
            labels = getattr(model, 'labels_', None)
            if labels is None:
                raise ValueError(
                    f"cluster model {type(model).__name__} was fit "
                    f"successfully but exposes neither a labels_ attribute "
                    f"nor a fit_predict method, so no cluster labels can be "
                    f"extracted; pass a scikit-learn style clusterer instead.")
            result = list(labels)
        # fingerprint the fit-time data (shape plus per-column moments) so
        # `transform`'s label-recovery fallback can tell "the same data the
        # model was fit on" apart from unrelated data that merely has the
        # same number of rows (F13-cluster-006)
        arr = np.asarray(stacked)
        self._fit_shape = tuple(arr.shape)
        if arr.ndim == 2 and arr.size:
            self._fit_col_means = arr.mean(axis=0)
            self._fit_col_stds = arr.std(axis=0)
        else:
            self._fit_col_means = None
            self._fit_col_stds = None
        self.model_ = model
        return result

    def _matches_fit_data(self, arr):
        """Best-effort check that `arr` is the data `fit_transform` saw.

        Compares the fit-time fingerprint (shape plus per-column means and
        standard deviations, recorded by `fit_transform`) against `arr`.
        Small floating-point jitter (e.g. an upstream `PCA.transform` vs
        `fit_transform` round-trip) passes; genuinely different data with
        the same shape does not. Returns True when no fingerprint was
        recorded (e.g. a `Clusterer` unpickled from an older hypertools).
        """
        fit_shape = getattr(self, '_fit_shape', None)
        if fit_shape is None:
            return True
        arr = np.asarray(arr)
        if tuple(arr.shape) != fit_shape:
            return False
        if self._fit_col_means is None:
            return True
        return (np.allclose(arr.mean(axis=0), self._fit_col_means,
                            rtol=1e-5, atol=1e-8, equal_nan=True)
                and np.allclose(arr.std(axis=0), self._fit_col_stds,
                                rtol=1e-5, atol=1e-8, equal_nan=True))

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
            only support `fit_predict` on the data they were fit on) and
            the input's row count differs from the fit-time row count
            (column count, for `FeatureAgglomeration`).

        Warns
        -----
        UserWarning
            When a no-predict model's stored fit-time labels are returned
            for data that has the fit-time row count but does not match the
            fit-time data fingerprint (F13-cluster-006): the input was NOT
            re-clustered, and the warning says so.
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
        # Hard clusterers (AgglomerativeClustering, SpectralClustering, DBSCAN,
        # OPTICS, HDBSCAN) have no out-of-sample predict. The documented
        # analyze(cluster=...) label-RECOVERY path re-applies this fitted step to
        # the SAME data it was fit on -- so when `stacked` matches the fit-time
        # labels, return those stored labels (QC 2026-07 red-team: this used to
        # raise NotImplementedError, so the documented
        # `named_steps['cluster'].transform(data)` recovery broke for 3 of the 5
        # hard clusterers). Genuinely NEW data (a different row count) still
        # cannot be labeled without refitting, so that case still raises; data
        # with the SAME row count that does not match the fit-time fingerprint
        # gets the stored labels back WITH a UserWarning (F13-cluster-006).
        arr = np.asarray(stacked)
        name = type(model).__name__
        labels = getattr(model, 'labels_', None)
        if labels is not None and isinstance(model, FeatureAgglomeration):
            # per-FEATURE labels (F13-cluster-001): the stored labels apply
            # to the fit-time COLUMNS, so match on the column count, not rows
            if arr.ndim == 2 and arr.shape[1] == len(labels):
                if not self._matches_fit_data(arr):
                    warnings.warn(
                        "returning the fit-time per-feature cluster labels: "
                        "FeatureAgglomeration has no out-of-sample "
                        "prediction, so the data passed here (which does not "
                        "match the data the model was fit on) was NOT "
                        "re-clustered. If this is genuinely new data, refit "
                        "instead (e.g. hypertools.cluster(x, "
                        "cluster='FeatureAgglomeration')).", UserWarning)
                return list(labels)
            raise NotImplementedError(
                f"FeatureAgglomeration clusters features (columns), not "
                f"observations, and has no out-of-sample prediction: a "
                f"fitted FeatureAgglomeration can only return its fit-time "
                f"labels for data with the same {len(labels)} columns it was "
                f"fit on (got "
                f"{arr.shape[1] if arr.ndim == 2 else 'a non-2D input'}). "
                f"Refit on the new data instead.")
        if labels is not None and arr.shape[0] == len(labels):
            if not self._matches_fit_data(arr):
                warnings.warn(
                    f"returning the fit-time cluster labels: {name} has no "
                    f"out-of-sample prediction (no predict method), so the "
                    f"data passed here (which does not match the data the "
                    f"model was fit on) was NOT re-clustered. If this is "
                    f"genuinely new data, refit instead (e.g. "
                    f"hypertools.cluster(x, cluster='{name}')).", UserWarning)
            return list(labels)
        raise NotImplementedError(
            f"{name} has no out-of-sample prediction (no predict method); "
            f"cannot reuse a fitted {name} clusterer on new data (got "
            f"{arr.shape[0]} rows vs "
            f"{len(labels) if labels is not None else 'an unknown number of'} "
            f"fit-time observations) without refitting")
