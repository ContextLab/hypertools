# Vendored from the pca-magic project. See license in-repo.

import os

import numpy as np
from scipy.linalg import orth


class PPCA(object):
    """Probabilistic PCA via an EM algorithm, with support for missing data.

    Vendored from the `pca-magic` project. Unlike standard PCA, `fit`
    tolerates NaN entries in the input (imputing them internally during
    the EM iterations) and drops any column with fewer than `min_obs`
    non-missing observations. After fitting, `self.C` holds the
    (orthonormalized, variance-ordered) component loadings and
    `self.var_exp` holds the cumulative fraction of variance explained.
    """

    def __init__(self):

        self.raw = None
        self.data = None
        self.C = None
        self.means = None
        self.stds = None

    def _standardize(self, X):

        if self.means is None or self.stds is None:
            raise RuntimeError("Fit model first")

        return (X - self.means) / self.stds

    def fit(self, data, d=None, tol=1e-4, min_obs=10, verbose=False):
        """Fit the probabilistic PCA model on `data` via EM, tolerating NaNs.

        Parameters
        ----------
        data : numpy.ndarray
            2D array (observations x features), possibly containing NaN
            (missing) and/or infinite entries. Infinite entries are
            clipped to the maximum finite value in `data`. Modified
            in-place (stored as `self.raw`).
        d : int or None, optional
            Number of latent components to fit. Defaults to the number
            of (valid) columns in `data`.
        tol : float, optional
            Relative-change convergence tolerance on the EM objective
            (default: 1e-4).
        min_obs : int, optional
            Columns with fewer than this many non-missing observations
            are dropped before fitting (default: 10).
        verbose : bool, optional
            If True, print the convergence diagnostic each iteration.

        Notes
        -----
        Fits `self.C` (component loadings, orthonormalized and sorted by
        descending eigenvalue), `self.data` (the standardized, missing-
        imputed data used for fitting), `self.eig_vals`, `self.means`,
        `self.stds`, and (via `_calc_var`) `self.var_exp`.
        """
        self.raw = data
        self.raw[np.isinf(self.raw)] = np.max(self.raw[np.isfinite(self.raw)])

        valid_series = np.sum(~np.isnan(self.raw), axis=0) >= min_obs
        # remember which original columns were kept (>= min_obs observations),
        # so callers can map the fitted (kept-column) reconstruction back to the
        # full input width (QC 2026-07: the PPCA imputer needs this to preserve
        # the input's column count).
        self.valid_series = valid_series

        data = self.raw[:, valid_series].copy()
        N = data.shape[0]
        D = data.shape[1]

        self.means = np.nanmean(data, axis=0)
        self.stds = np.nanstd(data, axis=0)

        data = self._standardize(data)
        observed = ~np.isnan(data)
        missing = np.sum(~observed)
        data[~observed] = 0

        # initial

        if d is None:
            d = data.shape[1]

        if self.C is None:
            C = np.random.randn(D, d)
        else:
            C = self.C
        CC = np.dot(C.T, C)
        X = np.dot(np.dot(data, C), np.linalg.inv(CC))
        recon = np.dot(X, C.T)
        recon[~observed] = 0
        ss = np.sum((recon - data)**2)/(N*D - missing)

        v0 = np.inf
        counter = 0

        while True:

            Sx = np.linalg.inv(np.eye(d) + CC/ss)

            # e-step
            ss0 = ss
            if missing > 0:
                proj = np.dot(X, C.T)
                data[~observed] = proj[~observed]
            X = np.dot(np.dot(data, C), Sx) / ss

            # m-step
            XX = np.dot(X.T, X)
            C = np.dot(np.dot(data.T, X), np.linalg.pinv(XX + N*Sx))
            CC = np.dot(C.T, C)
            recon = np.dot(X, C.T)
            recon[~observed] = 0
            ss = (np.sum((recon-data)**2) + N*np.sum(CC*Sx) + missing*ss0)/(N*D)

            # calc diff for convergence
            det = np.log(np.linalg.det(Sx))
            if np.isinf(det):
                det = abs(np.linalg.slogdet(Sx)[1])
            v1 = N*(D*np.log(ss) + np.trace(Sx) - det) \
                + np.trace(XX) - missing*np.log(ss0)
            diff = abs(v1/v0 - 1)
            if verbose:
                print(diff)
            if (diff < tol) and (counter > 5):
                break

            counter += 1
            v0 = v1


        C = orth(C)
        vals, vecs = np.linalg.eig(np.cov(np.dot(data, C).T))
        order = np.flipud(np.argsort(vals))
        vecs = vecs[:, order]
        vals = vals[order]

        C = np.dot(C, vecs)

        # attach objects to class
        self.C = C
        self.data = data
        self.eig_vals = vals
        self._calc_var()

    def transform(self, data=None):
        """Project `data` onto the fitted principal components.

        Parameters
        ----------
        data : numpy.ndarray or None, optional
            Data to project. If `None` (default), projects the
            (standardized, missing-imputed) data stored from `fit`.

        Returns
        -------
        numpy.ndarray
            `data @ self.C` (or `self.data @ self.C` when `data` is None).

        Raises
        ------
        RuntimeError
            If `fit` has not been called yet (`self.C` is None).
        """
        if self.C is None:
            raise RuntimeError('Fit the data model first.')
        if data is None:
            return np.dot(self.data, self.C)
        return np.dot(data, self.C)

    def _calc_var(self):

        if self.data is None:
            raise RuntimeError('Fit the data model first.')

        data = self.data.T

        # variance calc
        var = np.nanvar(data, axis=1)
        total_var = var.sum()
        self.var_exp = self.eig_vals.cumsum() / total_var

    def save(self, fpath):
        """Save the fitted component loadings (`self.C`) to `fpath` via `numpy.save`.

        Parameters
        ----------
        fpath : str
            Destination path (`.npy` extension appended by `numpy.save`
            if not already present).
        """
        np.save(fpath, self.C)

    def load(self, fpath):
        """Load component loadings from `fpath` into `self.C` via `numpy.load`.

        Parameters
        ----------
        fpath : str
            Path to a `.npy` file previously written by `save`.

        Raises
        ------
        AssertionError
            If `fpath` does not exist.
        """
        assert os.path.isfile(fpath)

        self.C = np.load(fpath)
