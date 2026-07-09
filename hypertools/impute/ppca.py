"""PPCA imputer wrapping `hypertools.external.ppca.PPCA`.

Cannot reconstruct rows with NO observed features at all -- there is nothing
to project from -- so those rows are left as NaN and a warning is raised.
This is the GH #169 gap that `hypertools.impute.kalman.Kalman` closes (its
state propagates across time, so it can fill fully-missing rows given
neighboring observations).

Unlike the other imputers in this package, PPCA does not merely replace
missing entries: `hypertools.external.ppca.PPCA.transform` returns a full
reconstruction (the fitted-and-standardized data rotated into a PCA basis),
so non-missing entries are NOT guaranteed to be preserved exactly. This
matches the pre-existing `hypertools.tools.format_data` behavior (see its
"Inexact solution computed with PPCA" warning) -- PPCA is kept as the
lossy, approximate default for backwards compatibility; use
SimpleImputer/KNNImputer/IterativeImputer/Kalman for exact preservation of
non-missing values.
"""
import warnings

import numpy as np
import pandas as pd

from .common import Imputer
from ..external.ppca import PPCA as _PPCAModel


def fitter(data, **kwargs):
    """Fit `hypertools.external.ppca.PPCA` on `data`, warning about fully-missing rows.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to fit on; NaN entries are treated as missing.
    **kwargs
        `d` : int or None, number of latent dimensions. `min_obs` : int,
        minimum non-missing observations per column (default: 10). `tol`
        : float, EM convergence tolerance (default: 1e-4).

    Returns
    -------
    dict
        `{'ppca': <fitted external.ppca.PPCA instance>}`.
    """
    d = kwargs.get('d', None)
    min_obs = kwargs.get('min_obs', 10)
    tol = kwargs.get('tol', 1e-4)

    x = data.to_numpy(dtype=float)
    all_missing_rows = np.all(np.isnan(x), axis=1)
    if all_missing_rows.any():
        warnings.warn(
            f"PPCA cannot fill {int(all_missing_rows.sum())} row(s) with no "
            "observed features at all; those rows will remain NaN. Use "
            "model='Kalman' (hypertools.impute.kalman.Kalman) to fill "
            "fully-missing rows too (see GH #169).")

    m = _PPCAModel()
    m.fit(data=x.copy(), d=d, tol=tol, min_obs=min_obs)
    return {'ppca': m}


def transformer(data, **kwargs):
    """Reconstruct `data` via the fitted PPCA model's PCA-basis projection.

    On the original fit-time data, returns the EM-refined reconstruction
    (byte-identical to the legacy `format_data.fill_missing` behavior).
    On new data (the `return_model=True` reuse path), standardizes with
    the fitted mean/std, zero-fills NaNs, and does a single-shot
    projection through the fitted rotation (an approximation, since PPCA
    has no clean way to reuse learned parameters without re-running EM).
    Rows that were entirely missing in `data` are set back to NaN in the
    output (PPCA cannot reconstruct them at all).

    Parameters
    ----------
    data : pandas.DataFrame
        Data to reconstruct/impute.
    **kwargs
        `ppca` : the fitted `external.ppca.PPCA` instance from `fitter`.
        `_is_original_fit_data` : bool, whether `data` is the same data
        `fitter` was called on (default: True).

    Returns
    -------
    pandas.DataFrame
        The reconstructed/imputed data, indexed like `data` (columns
        fall back to a default integer range if PPCA dropped columns at
        fit time).

    Raises
    ------
    ValueError
        If `data` has a different number of columns than the imputer was
        fit on, on the new-data (non-original) path.
    """
    m = kwargs['ppca']
    is_original = kwargs.get('_is_original_fit_data', True)
    original = data.to_numpy(dtype=float)
    # which original columns PPCA kept (>= min_obs observations). Dropped
    # columns cannot be imputed and keep their original values.
    valid = getattr(m, 'valid_series', None)
    if valid is None:
        valid = np.ones(original.shape[1], dtype=bool)

    if is_original:
        # Reconstruct the EM-imputed data in the ORIGINAL feature space:
        # m.data is the standardized, EM-filled (kept-column) data, so
        # un-standardizing it recovers the imputed values. QC 2026-07: this
        # used to return m.transform() = m.data @ m.C, i.e. the LATENT PCA
        # scores -- which ROTATED the observed values (for full-rank data the
        # "imputed" output differed from the input by several units) and
        # returned FEWER columns for rank-deficient data. Neither is the
        # imputed data.
        recon_kept = np.asarray(m.data, dtype=float) * m.stds + m.means
    else:
        # Reuse path (return_model=True round-trip): standardize the KEPT
        # columns with the fitted mean/std, zero-fill NaNs, then project to the
        # latent space and back to reconstruct in feature space.
        if original[:, valid].shape[1] != len(m.means):
            raise ValueError(
                'PPCA.transform on new data requires the same columns the '
                'imputer was fit on (columns dropped for having fewer than '
                'min_obs valid observations at fit time cannot be reused)')
        standardized = (original[:, valid] - m.means) / m.stds
        standardized = np.where(np.isnan(standardized), 0.0, standardized)
        recon_kept = ((standardized @ m.C) @ m.C.T) * m.stds + m.means

    # SPLICE: keep every observed (non-NaN) value exactly and fill ONLY the
    # missing positions with the reconstruction -- preserving the input shape
    # and matching the documented "fills missing values in place" contract and
    # the sklearn imputers' behavior.
    recon_full = np.full_like(original, np.nan)
    recon_full[:, valid] = recon_kept
    out = original.copy()
    fillable = np.isnan(original) & ~np.isnan(recon_full)
    out[fillable] = recon_full[fillable]

    # Columns PPCA DROPPED (fewer than min_obs observations) are not modeled, so
    # their missing entries are still NaN after the splice. Leaving them NaN
    # regressed the primary path: hyp.reduce/hyp.plot feed PPCA's output straight
    # into PCA, which raised "Input X contains NaN" on sparse-column data (QC
    # 2026-07 red-team). Fill each still-missing position with its column's
    # observed mean (0.0 for a column with no observations at all) so the imputed
    # matrix is dense -- exactly as the pre-splice reconstruction was. Observed
    # values are untouched (only originally-NaN positions are filled). Rows that
    # are entirely missing are re-masked to NaN below (documented limitation).
    still_missing = np.isnan(out)
    if still_missing.any():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN col -> nan
            col_means = np.nanmean(original, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        out[still_missing] = np.broadcast_to(col_means, original.shape)[still_missing]

    # rows that were entirely missing cannot be reconstructed at all
    all_missing_rows = np.all(np.isnan(original), axis=1)
    if all_missing_rows.any():
        out[all_missing_rows, :] = np.nan
    return pd.DataFrame(out, index=data.index, columns=data.columns)


class PPCA(Imputer):
    """PPCA imputer: fills missing values via probabilistic PCA.

    Parameters
    ----------
    d : int or None
        Number of latent dimensions (default: None, meaning full rank --
        same as the number of fit columns).
    min_obs : int
        Columns with fewer than `min_obs` non-missing observations are
        dropped before fitting (default: 10).
    tol : float
        EM convergence tolerance (default: 1e-4).
    """

    def __init__(self, d=None, min_obs=10, tol=1e-4, **kwargs):
        required = ['ppca']
        super().__init__(d=d, min_obs=min_obs, tol=tol, fitter=fitter, transformer=transformer,
                          data=None, required=required, **kwargs)
        self.d = d
        self.min_obs = min_obs
        self.tol = tol
        self.fitter = fitter
        self.transformer = transformer
        self.data = None
        self.required = required
