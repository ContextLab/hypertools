"""PPCA imputer wrapping `hypertools.external.ppca.PPCA`.

Missing entries are filled with the PPCA reconstruction and SPLICED into the
original data: every observed (non-NaN) value is preserved exactly, and only
the missing positions are filled (verified byte-identical preservation --
the same contract as the SimpleImputer/KNNImputer/IterativeImputer/Kalman
imputers).

Cannot reconstruct rows with NO observed features at all -- there is nothing
to project from -- so those rows are left as NaN and a warning is raised
(they are also excluded from the EM fit, where they used to poison the
convergence objective with NaN). This is the GH #169 gap that
`hypertools.impute.kalman.Kalman` closes (its state propagates across time,
so it can fill fully-missing rows given neighboring observations).

Zero-variance (constant) and all-NaN columns are likewise excluded from the
EM fit (standardizing a constant column divides by zero, which used to
poison the EM objective for EVERY column -- collapsing the whole
reconstruction to column-mean fills, warning about non-convergence, or
intermittently crashing with a LinAlgError; QC 2026-07 red-team
F17-impute-004). Their missing entries are filled with the column's observed
mean (the constant itself; 0.0 for a column with no observations at all).

The default number of latent dimensions is ONE FEWER than the number of
usable columns: a full-rank PPCA (the pre-fix default) can reproduce any
data exactly, so its EM never has to learn cross-column structure and its
"imputations" were numerically indistinguishable from plain column-mean
fills (QC 2026-07 red-team F17-impute-002: recovery r=0.13 at full rank vs
r=0.98 at any d below it, on rank-3 benchmark data). Pass ``d=`` explicitly
to control the latent dimensionality.
"""
import warnings

import numpy as np
import pandas as pd

from .common import Imputer
from ..external.ppca import PPCA as _PPCAModel


def fitter(data, **kwargs):
    """Fit `hypertools.external.ppca.PPCA` on the usable part of `data`.

    Rows with no observed features (warned about) and columns that are
    unusable for the EM -- fewer than `min_obs` observations or zero
    observed variance -- are excluded from the fit (see the module
    docstring); the transformer maps the reconstruction back to the full
    input shape.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to fit on; NaN entries are treated as missing.
    **kwargs
        `d` : int or None, number of latent dimensions (None, the
        default, uses one fewer than the number of usable columns; must
        satisfy ``2 <= d <= n_usable_columns``). `min_obs` : int, minimum
        non-missing observations per column (default: 10). `tol` :
        float, EM convergence tolerance (default: 1e-4). `random_state` :
        int or None, seed for the EM's random initialization (default:
        None -- nondeterministic, drawn from the global numpy RNG; pass
        an int for reproducible imputations).

    Returns
    -------
    dict
        `{'ppca': <fitted external.ppca.PPCA instance>, 'fit_rows':
        <boolean row mask used for fitting>, 'keep_cols': <boolean
        column mask used for fitting>}`.

    Raises
    ------
    ValueError
        If fewer than 2 usable columns remain, or `d` is invalid.
    """
    d = kwargs.get('d', None)
    min_obs = kwargs.get('min_obs', 10)
    tol = kwargs.get('tol', 1e-4)
    random_state = kwargs.get('random_state', None)

    x = data.to_numpy(dtype=float)
    all_missing_rows = np.all(np.isnan(x), axis=1)
    if all_missing_rows.any():
        warnings.warn(
            f"PPCA cannot fill {int(all_missing_rows.sum())} row(s) with no "
            "observed features at all; those rows will remain NaN. Use "
            "model='Kalman' (hypertools.impute.kalman.Kalman) to fill "
            "fully-missing rows too (see GH #169).")
    fit_rows = ~all_missing_rows

    obs_counts = (~np.isnan(x)).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN col -> nan
        col_stds = np.nanstd(x, axis=0)
    keep_cols = (obs_counts >= min_obs) & (col_stds > 0)
    n_usable = int(keep_cols.sum())
    if n_usable < 2:
        raise ValueError(
            f'PPCA needs at least 2 columns with >= min_obs ({min_obs}) '
            f'non-missing observations and non-zero variance to model '
            f'cross-column structure; got {n_usable}. Use model="Kalman", '
            '"SimpleImputer", or "KNNImputer" for single-column (or very '
            'sparse/constant) data.')

    if d is None:
        # one fewer than the number of usable columns: full rank degenerates
        # to column-mean fills (see the module docstring); never below 2 (the
        # vendored implementation cannot fit a single latent dimension).
        d = max(2, n_usable - 1)
    else:
        if (not isinstance(d, (int, np.integer))) or isinstance(d, bool) \
                or d < 2 or d > n_usable:
            raise ValueError(
                f'd (number of latent dimensions) must be an integer with '
                f'2 <= d <= the number of usable columns ({n_usable}); got '
                f'{d!r}.')
        d = int(d)

    m = _PPCAModel()
    fit_data = x[np.ix_(fit_rows, keep_cols)].copy()
    if random_state is None:
        m.fit(data=fit_data, d=d, tol=tol, min_obs=min_obs)
    else:
        # the vendored PPCA initializes its EM from the GLOBAL numpy RNG with
        # no seed parameter (QC 2026-07 red-team F17-impute-003): seed it
        # temporarily, restoring the caller's RNG state afterwards.
        state = np.random.get_state()
        try:
            np.random.seed(random_state)
            m.fit(data=fit_data, d=d, tol=tol, min_obs=min_obs)
        finally:
            np.random.set_state(state)
    return {'ppca': m, 'fit_rows': fit_rows, 'keep_cols': keep_cols}


def transformer(data, **kwargs):
    """Fill `data`'s missing entries from the fitted PPCA reconstruction.

    On the original fit-time data, un-standardizes the EM-refined
    reconstruction. On new data (the `return_model=True` reuse path),
    standardizes with the fitted mean/std, zero-fills NaNs, and does a
    single-shot projection through the fitted rotation (an approximation,
    since PPCA has no clean way to reuse learned parameters without
    re-running EM). Either way the reconstruction is SPLICED into `data`:
    observed (non-NaN) values are preserved exactly and only missing
    positions are filled. Missing entries in columns the fit excluded
    (too sparse or zero-variance) are filled with the column's observed
    mean (0.0 if nothing was observed); rows that were entirely missing
    are set back to NaN (PPCA cannot reconstruct them at all).

    Parameters
    ----------
    data : pandas.DataFrame
        Data to impute; must have the same number of columns the imputer
        was fit on.
    **kwargs
        `ppca` : the fitted `external.ppca.PPCA` instance from `fitter`.
        `fit_rows`, `keep_cols` : the fit-time row/column masks from
        `fitter`. `_is_original_fit_data` : bool, whether `data` is the
        same data `fitter` was called on (default: True).

    Returns
    -------
    pandas.DataFrame
        The imputed data, indexed/columned like `data`.

    Raises
    ------
    ValueError
        If `data` has a different number of columns than the imputer was
        fit on (checked up front on the reuse path; this used to be
        unreachable dead code behind an IndexError -- QC 2026-07 red-team
        F17-impute-005).
    """
    m = kwargs['ppca']
    is_original = kwargs.get('_is_original_fit_data', True)
    original = data.to_numpy(dtype=float)
    keep_cols = np.asarray(kwargs['keep_cols'], dtype=bool)
    fit_rows = np.asarray(kwargs['fit_rows'], dtype=bool)

    if original.shape[1] != len(keep_cols):
        raise ValueError(
            f'PPCA.transform on new data requires the same columns the '
            f'imputer was fit on: got {original.shape[1]} column(s), but the '
            f'imputer was fit on {len(keep_cols)}.')

    # which fit-time columns the external model kept (all of `keep_cols` by
    # construction -- kept defensively in case min_obs filtering inside the
    # external model drops more).
    valid_sub = getattr(m, 'valid_series', None)
    if valid_sub is None:
        valid_sub = np.ones(int(keep_cols.sum()), dtype=bool)
    full_valid = np.zeros(len(keep_cols), dtype=bool)
    full_valid[np.flatnonzero(keep_cols)[valid_sub]] = True

    recon_full = np.full_like(original, np.nan)
    if is_original:
        # Reconstruct the EM-imputed data in the ORIGINAL feature space:
        # m.data is the standardized, EM-filled (kept-column, fit-row) data,
        # so un-standardizing it recovers the imputed values. QC 2026-07:
        # this used to return m.transform() = m.data @ m.C, i.e. the LATENT
        # PCA scores -- which ROTATED the observed values and returned FEWER
        # columns for rank-deficient data. Neither is the imputed data.
        recon_kept = np.asarray(m.data, dtype=float) * m.stds + m.means
        recon_full[np.ix_(fit_rows, full_valid)] = recon_kept
    else:
        # Reuse path (return_model=True round-trip): standardize the KEPT
        # columns with the fitted mean/std, zero-fill NaNs, then project to
        # the latent space and back to reconstruct in feature space.
        standardized = (original[:, full_valid] - m.means) / m.stds
        standardized = np.where(np.isnan(standardized), 0.0, standardized)
        recon_full[:, full_valid] = ((standardized @ m.C) @ m.C.T) * m.stds + m.means

    # SPLICE: keep every observed (non-NaN) value exactly and fill ONLY the
    # missing positions with the reconstruction -- preserving the input shape
    # and matching the documented "fills missing values in place" contract and
    # the sklearn imputers' behavior.
    out = original.copy()
    fillable = np.isnan(original) & ~np.isnan(recon_full)
    out[fillable] = recon_full[fillable]

    # Columns excluded from the fit (too sparse or zero-variance) are not
    # modeled, so their missing entries are still NaN after the splice.
    # Leaving them NaN regressed the primary path: hyp.reduce/hyp.plot feed
    # PPCA's output straight into PCA, which raised "Input X contains NaN" on
    # sparse-column data (QC 2026-07 red-team). Fill each still-missing
    # position with its column's observed mean (0.0 for a column with no
    # observations at all) so the imputed matrix is dense. Observed values
    # are untouched (only originally-NaN positions are filled). Rows that
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
    """PPCA imputer: fills missing values via probabilistic PCA (observed
    values are preserved exactly; see the module docstring).

    Parameters
    ----------
    d : int or None
        Number of latent dimensions (default: None, meaning one fewer
        than the number of usable columns -- full rank degenerates to
        column-mean fills; see the module docstring). Must satisfy
        ``2 <= d <= n_usable_columns``.
    min_obs : int
        Columns with fewer than `min_obs` non-missing observations are
        excluded from the fit (default: 10); their missing entries are
        filled with the column's observed mean.
    tol : float
        EM convergence tolerance (default: 1e-4).
    random_state : int or None
        Seed for the EM's random initialization (default: None --
        nondeterministic; pass an int for reproducible imputations).
    """

    def __init__(self, d=None, min_obs=10, tol=1e-4, random_state=None):
        required = ['ppca', 'fit_rows', 'keep_cols']
        super().__init__(d=d, min_obs=min_obs, tol=tol, random_state=random_state,
                          fitter=fitter, transformer=transformer,
                          data=None, required=required)
        self.d = d
        self.min_obs = min_obs
        self.tol = tol
        self.random_state = random_state
        self.fitter = fitter
        self.transformer = transformer
        self.data = None
        self.required = required
