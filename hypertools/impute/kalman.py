"""Kalman-smoothing imputer (pykalman).

EM-fits an independent univariate linear-Gaussian state-space model to each
COLUMN (``em_vars`` includes the transition matrix, so the per-column
dynamics are genuinely estimated), then Rauch-Tung-Striebel-smooths each
column's masked series and splices the smoothed values back in ONLY where
the input was missing -- every non-missing entry passes through unchanged.

Why per-column? pykalman treats an observation vector with ANY masked entry
as ENTIRELY missing, so a joint (all-columns) state-space model -- the
pre-fix design -- never conditioned on partially-observed rows at all. For
wide data (>= ~50 columns at ~10% missingness, where nearly every row has
at least one NaN) the joint EM therefore learned nothing and silently
filled every missing value with exactly 0.0 (QC 2026-07 red-team
D05-gallery-data-text-001). Per-column smoothing conditions on every
observed value of that column, interpolates through time exactly as the
"from the neighboring (observed) timepoints" contract describes, and its
recovery quality is independent of the number of columns (measured r ~0.996
at 5, 20, 50, and 100 columns on latent-sine data where the joint model
degenerated to zero-fills at >= 50).

Unlike `hypertools.impute.ppca.PPCA`, which cannot reconstruct rows with NO
observed features at all, the Kalman smoother's state propagates across
time, so it CAN fill rows where every feature is missing, as long as
neighboring rows have some observations -- this closes the GH #169 gap.
A column with NO observed values at all carries no information; its missing
entries are filled with 0.0 (matching the sklearn imputers'
``keep_empty_features`` behavior).

`pykalman` is a core hypertools dependency (shared with
`hypertools.predict.kalman`), so the `Kalman` imputer works out of the box;
imported lazily so `hypertools.impute` stays importable even where the core
deps were stripped, raising a friendly `ImportError` only then.
"""
import numpy as np
import pandas as pd

from .common import Imputer
from ..core.shared import import_kalman_filter

_EM_VARS = ['transition_matrices', 'transition_covariance',
            'observation_covariance', 'initial_state_mean',
            'initial_state_covariance']


def fitter(data, **kwargs):
    """EM-fit one univariate Kalman filter per column of `data`.

    Each column is mean-centered and fit with
    ``KalmanFilter(n_dim_obs=1, n_dim_state=1).em(..., em_vars=[...])``
    (missing entries masked via `numpy.ma.masked_invalid`); the EM
    estimates the column's transition dynamics and noise covariances.
    Columns with no observed values cannot be fit and are recorded as
    `None` (their missing entries are later filled with 0.0).

    Parameters
    ----------
    data : pandas.DataFrame
        Data to fit on; NaN entries are masked (treated as missing).
    **kwargs
        `n_iter` : int, number of EM iterations per column (default: 5).

    Returns
    -------
    dict
        `{'kfs': [<fitted pykalman.KalmanFilter> or None, one per
        column], 'col_means': <observed mean per column (0.0 for
        all-missing columns)>}`.

    Raises
    ------
    ValueError
        If `data` has fewer than 2 rows (timepoints) -- Kalman smoothing
        interpolates through time, so a single row carries no temporal
        information.
    """
    kalman_filter_cls = import_kalman_filter('imputer')
    n_iter = kwargs.get('n_iter', 5)

    x = data.to_numpy(dtype=float)
    n, d = x.shape
    # a single row used to crash deep inside pykalman with a raw broadcast
    # error (QC 2026-07 red-team F17-impute-009).
    if n < 2:
        raise ValueError(
            f'Kalman imputation needs at least 2 rows (timepoints) to '
            f'interpolate through time; got {n}. Use '
            "model='SimpleImputer' or model='KNNImputer' for single-row data.")

    kfs = []
    col_means = np.zeros(d)
    for j in range(d):
        col = x[:, j]
        observed = ~np.isnan(col)
        if not observed.any():
            kfs.append(None)  # nothing to learn from; transformer fills 0.0
            continue
        mu = col[observed].mean()
        col_means[j] = mu
        masked = np.ma.masked_invalid(col - mu)
        kf = kalman_filter_cls(n_dim_obs=1, n_dim_state=1).em(
            masked, n_iter=n_iter, em_vars=_EM_VARS)
        kfs.append(kf)

    return {'kfs': kfs, 'col_means': col_means}


def transformer(data, **kwargs):
    """Fill missing entries of `data` with per-column Kalman-smoothed estimates.

    Smooths each column's masked (mean-centered) series with that
    column's fitted Kalman filter, then splices the smoothed values back
    in ONLY where `data` was originally missing (NaN) -- every
    non-missing entry passes through unchanged. Columns whose filter
    could not be fit (no observed values at fit time) fall back to their
    fit-time mean (0.0 when nothing was ever observed).

    Parameters
    ----------
    data : pandas.DataFrame
        Data to impute; must have the same number of columns the imputer
        was fit on.
    **kwargs
        `kfs`, `col_means` : the fitted per-column filters and means from
        `fitter`.

    Returns
    -------
    pandas.DataFrame
        `data` with missing entries replaced by smoothed estimates,
        same index/columns as `data`.

    Raises
    ------
    ValueError
        If `data` has a different number of columns than the imputer was
        fit on.
    """
    kfs = kwargs['kfs']
    col_means = kwargs['col_means']

    x = data.to_numpy(dtype=float)
    if x.shape[1] != len(kfs):
        raise ValueError(
            f'this Kalman imputer was fit on {len(kfs)} column(s) but the '
            f'data to impute has {x.shape[1]}; reuse a fitted imputer only '
            'on data with the same columns it was fit on.')

    out = x.copy()
    for j, kf in enumerate(kfs):
        col = x[:, j]
        mask = np.isnan(col)
        if not mask.any():
            continue
        if kf is None:
            out[mask, j] = col_means[j]
            continue
        masked = np.ma.masked_invalid(col - col_means[j])
        smoothed, _ = kf.smooth(masked)
        out[mask, j] = np.asarray(smoothed)[mask, 0] + col_means[j]

    return pd.DataFrame(out, index=data.index, columns=data.columns)


class Kalman(Imputer):
    """Kalman-smoothing imputer: EM-fit an independent univariate
    linear-Gaussian state-space model per column, then replace ONLY the
    missing entries with each column's smoothed (temporally-interpolated)
    estimates.

    Unlike PPCA, this fills rows where EVERY feature is missing (each
    column's smoother propagates its state across time from neighboring
    observed rows) -- the GH #169 gap. See the module docstring for why the
    model is per-column rather than joint (the joint design silently
    zero-filled wide data), and for the all-missing-column 0.0 fill.

    Parameters
    ----------
    n_iter : int
        Number of EM iterations per column (default: 5).
    """

    def __init__(self, n_iter=5):
        required = ['kfs', 'col_means']
        super().__init__(n_iter=n_iter, fitter=fitter, transformer=transformer, data=None,
                          required=required)
        self.n_iter = n_iter
        self.fitter = fitter
        self.transformer = transformer
        self.data = None
        self.required = required
