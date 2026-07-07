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

    if is_original:
        # Byte-identical to the legacy `format_data.fill_missing` behavior:
        # `m.transform()` with no args projects the EM-refined, fit-time
        # filled data (`m.data`) through the learned rotation `m.C`.
        filled = m.transform()
    else:
        # Reuse path (`return_model=True` round-trip): no re-fitting, so
        # there is no EM refinement available for NEW data. Best-effort
        # single-shot projection using the LEARNED mean/std/rotation:
        # standardize with the fitted `means`/`stds`, zero-fill any NaNs
        # (mirroring PPCA's own initial E-step guess before EM iterates),
        # then project through the fitted `C`. This is an approximation --
        # PPCA has no clean way to reuse learned parameters without
        # re-running EM -- and is documented as such.
        x_new = data.to_numpy(dtype=float)
        assert x_new.shape[1] == len(m.means), ValueError(
            'PPCA.transform on new data requires the same number of '
            'columns the imputer was fit on (columns dropped for having '
            'fewer than min_obs valid observations at fit time cannot be '
            'reused)')
        standardized = (x_new - m.means) / m.stds
        standardized = np.where(np.isnan(standardized), 0.0, standardized)
        filled = m.transform(data=standardized)

    filled = np.asarray(filled, dtype=float).copy()

    all_missing_rows = np.all(np.isnan(data.to_numpy(dtype=float)), axis=1)
    if all_missing_rows.any():
        filled[all_missing_rows, :] = np.nan

    # PPCA drops columns with too few observations (`min_obs`) before
    # fitting, so the reconstruction can have fewer columns than the input
    # in that (rare) edge case; fall back to a default column index rather
    # than mismatching names.
    columns = data.columns if filled.shape[1] == data.shape[1] else range(filled.shape[1])
    return pd.DataFrame(filled, index=data.index, columns=columns)


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
