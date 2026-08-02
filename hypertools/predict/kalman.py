"""Kalman-filter forecaster (pykalman).

Builds a delay-embedded linear-Gaussian state-space model: the state vector
stacks the most recent `lags` observations, the transition matrix is a
companion matrix whose top block is estimated from the data by least squares
(regressing each observation on the `lags` observations before it), and the
observation matrix reads the newest block of the state. The noise
covariances and initial state are then refined with pykalman's EM
(``em_vars=['transition_covariance', 'observation_covariance',
'initial_state_mean', 'initial_state_covariance']``), and the fitted filter
is extended forward with `filter_update` (no new observations) to produce a
`t`-step-ahead forecast.

Why not let EM fit the transition matrix too? pykalman's EM initializes the
transition matrix at identity and, from that starting point, converges to a
near-identity local optimum that forecasts a flat line for any dynamic
signal (QC 2026-07 red-team F16-predict-001: the pre-fix code never
estimated the transition matrix AT ALL -- pykalman's default ``em_vars``
only fits covariances -- so every forecast was the last filtered state
repeated). The least-squares companion estimate captures real dynamics
(oscillations, trends), and letting EM subsequently re-estimate it was
measured to DEGRADE forecasts back toward flat/anti-phase, so EM refines
only the covariances and initial state.

The least-squares estimate is, however, constrained to be non-explosive
before use. Forecasting applies the transition operator once per step, so
its spectral radius is raised to the power of the horizon; an unconstrained
fit routinely lands outside the unit circle (the regression has `lags * d`
predictors but only `n - lags` windows, so it is near-saturated and
ill-conditioned whenever those are comparable) and the forecast then
diverges as ``rho ** t`` -- measured at 1e7 times the data range on 40x3
random walks before the fix (audit
notes/audit/kalman_instability_2026-08-02.md). `_constrain_stability`
rescales the fitted coefficient blocks so that ``rho <= 1``, which admits
random walks, trends and undamped oscillations unchanged while excluding
explosive dynamics. Because genuinely explosive processes do exist, that
constraint is applied only when the regression is too weakly determined to
justify the explosive estimate; a strongly over-determined fit is trusted
as-is, so a real exponential is still followed exactly.

NaNs in the input are tolerated via `np.ma.masked_invalid`, which pykalman
treats as missing observations during both EM and filtering (note pykalman
treats a row with ANY masked entry as entirely missing); rows containing NaN
are likewise skipped when estimating the transition matrix.

`pykalman` is a core hypertools dependency (pure-python, lightweight), so the
`Kalman` forecaster works out of the box. It is still imported lazily (inside
the fitter) so `hypertools.predict` stays importable even in an environment
where the core deps were stripped, raising a friendly `ImportError` only then.
"""
import numpy as np
import pandas as pd

from .common import Forecaster
from ..core.shared import import_kalman_filter

#: soft cap on the delay-embedded state dimension (lags * n_features); the
#: automatic `lags` choice shrinks toward 1 for wide data so the state (and
#: the EM covariance estimates) stay tractable.
_MAX_STATE_DIM = 32

#: largest spectral radius allowed for the estimated transition operator.
#: The forecast applies that operator once per step, so anything above 1
#: diverges as ``rho ** t``; 1 itself admits random walks, linear trends and
#: undamped oscillations. See `_constrain_stability`.
_MAX_SPECTRAL_RADIUS = 1.0

#: how strongly over-determined the companion regression must be before an
#: explosive least-squares fit is believed rather than constrained: at least
#: this many regression rows per predictor. See `_constrain_stability`.
_TRUST_EXPLOSIVE_ROWS_PER_PARAM = 3.0


def _resolve_lags(lags, n, d):
    """The delay-embedding order: user-provided `lags`, or an automatic
    choice (up to 5, shrinking for wide data so lags * d <= _MAX_STATE_DIM,
    and always < the number of observations)."""
    if lags is None:
        lags = max(1, min(5, _MAX_STATE_DIM // max(d, 1)))
    else:
        if not isinstance(lags, (int, np.integer)) or isinstance(lags, bool) or lags < 1:
            raise ValueError(f'lags must be a positive integer; got {lags!r}')
        lags = int(lags)
    return max(1, min(lags, n - 1))


def _constrain_stability(A, lags, d, n_rows, rho_max=_MAX_SPECTRAL_RADIUS,
                         margin=_TRUST_EXPLOSIVE_ROWS_PER_PARAM):
    """Constrain a fitted companion matrix to the non-explosive region.

    Forecasting iterates the transition operator once per step (see
    `_roll_forward`), so a state that is not annihilated by `A` grows like
    ``rho(A) ** t``. Any estimate with ``rho > 1`` therefore diverges
    geometrically in the forecast horizon, and unconstrained least squares
    routinely returns one: the design matrix regresses each observation on
    ``lags * d`` predictors built from only ``n - lags`` windows, so when
    those two are comparable the fit is near-saturated, ill-conditioned, and
    yields large coefficients (measured `rho` up to 4.2, i.e. ``rho ** 12``
    of 3e7, on 40x3 random walks -- audit notes/audit/kalman_instability_2026-08-02.md).

    The estimate is pulled back by scaling the lag-`j` coefficient block by
    ``c ** j``. Writing the characteristic polynomial of the companion matrix
    as ``det(z**p I - sum_j A_j z**(p-j))`` and substituting ``A_j -> c**j A_j``
    factors out ``c ** (p*d)``, so every eigenvalue is scaled by exactly `c`
    -- this is the standard stationarity/stability shrinkage of the VAR
    literature, applied to the estimated dynamics. It is deliberately a
    constraint on the MODEL, not a clamp on the output: the returned operator
    is a self-consistent non-explosive linear system that pykalman filters and
    `applier` reuses on new data. It also leaves the companion shift block
    untouched, so the state stays an honest delay embedding.

    `rho_max` is 1 rather than something strictly smaller so that genuinely
    persistent dynamics survive intact: a random walk, a linear trend and an
    undamped oscillation all sit exactly at ``rho == 1``, and shrinking below
    that would reintroduce the flat/mean-reverting forecasts that motivated
    estimating the transition matrix by least squares in the first place (see
    the module docstring). Fits at or inside the unit circle are returned
    unchanged.

    Genuinely explosive processes do exist, though, and a blanket constraint
    would make this forecaster unable to follow one. The two cases are told
    apart by how well-determined the fit is: an explosive estimate is trusted
    only when the regression is over-determined by at least `margin` rows per
    predictor. Measured separation is wide -- every blow-up observed on random
    walks had ``n_rows / (lags * d)`` between 0.87 and 1.13, while an
    exactly-exponential series is recovered from a ratio of 4.0 -- so a
    genuine exponential is followed exactly (forecast error 0.0) while the
    near-saturated artifacts are pulled back.
    """
    if lags < 1:
        return A
    eigenvalues = np.linalg.eigvals(A)
    rho = float(np.abs(eigenvalues).max()) if eigenvalues.size else 0.0
    if not np.isfinite(rho) or rho <= rho_max:
        return A
    if n_rows >= margin * lags * d:
        return A  # strongly over-determined: the explosive fit is real

    c = rho_max / rho
    for j in range(1, lags + 1):
        A[:d, (j - 1) * d:j * d] *= c ** j
    return A


def _companion_transition(x, lags):
    """Estimate the delay-embedded transition matrix by least squares.

    Regresses each observation on the `lags` observations before it
    (rows containing NaN are skipped) and embeds the coefficients as the
    top block of a companion matrix. Falls back to a random-walk
    companion (top block = [I 0 ... 0]) if fewer than 2 complete
    regression rows exist.

    The fitted coefficients are then constrained to the non-explosive
    region (spectral radius <= 1) by `_constrain_stability`, because the
    forecast applies this operator once per step and an explosive estimate
    diverges as ``rho ** t``.
    """
    n, d = x.shape
    state_dim = lags * d

    rows, targets = [], []
    for i in range(lags, n):
        window = x[i - lags:i][::-1].reshape(-1)  # newest observation first
        target = x[i]
        if np.isnan(window).any() or np.isnan(target).any():
            continue
        rows.append(window)
        targets.append(target)

    A = np.zeros((state_dim, state_dim))
    if len(rows) >= 2:
        coef, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(targets), rcond=None)
        A[:d, :] = coef.T  # (d, lags*d) block of fitted AR coefficients
    else:
        A[:d, :d] = np.eye(d)  # too much missing data: random-walk dynamics
    if lags > 1:
        A[d:, :-d] = np.eye((lags - 1) * d)  # shift older observations down

    A = _constrain_stability(A, lags, d, len(rows))

    H = np.zeros((d, state_dim))
    H[:, :d] = np.eye(d)  # observe the newest block of the state
    return A, H


def fitter(data, **kwargs):
    """Fit a delay-embedded linear-Gaussian Kalman filter on `data`.

    The transition matrix is a companion matrix whose top block is
    estimated from `data` by least squares (see `_companion_transition`);
    pykalman's EM then refines the noise covariances and initial state
    (``em_vars`` excludes the transition/observation matrices -- see the
    module docstring for why). NaN entries are masked (treated as missing)
    via `numpy.ma.masked_invalid`.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to fit on.
    **kwargs
        `n_iter` : int, number of EM iterations (default: 5).
        `lags` : int or None, delay-embedding order (default: None, an
        automatic choice; see `Kalman`).

    Returns
    -------
    dict
        `{'kf': <fitted pykalman.KalmanFilter>, 'mean': <final filtered
        state mean>, 'cov': <final filtered state covariance>,
        'n_features': <number of observed columns>}`.
    """
    kalman_filter_cls = import_kalman_filter('forecaster')
    n_iter = kwargs.get('n_iter', 5)
    lags = kwargs.get('lags', None)

    x = data.to_numpy(dtype=float)
    n, d = x.shape
    lags = _resolve_lags(lags, n, d)

    A, H = _companion_transition(x, lags)
    masked = np.ma.masked_invalid(x)

    kf = kalman_filter_cls(n_dim_obs=d, n_dim_state=lags * d,
                           transition_matrices=A, observation_matrices=H)
    kf = kf.em(masked, n_iter=n_iter,
               em_vars=['transition_covariance', 'observation_covariance',
                        'initial_state_mean', 'initial_state_covariance'])
    means, covs = kf.filter(masked)

    return {'kf': kf, 'mean': means[-1], 'cov': covs[-1], 'n_features': d}


def _roll_forward(kf, mean, cov, n_steps, n_features):
    """Iterate `filter_update` with no new observations, reading the
    forecast for each step off the observed (newest) block of the state."""
    rows = []
    for _ in range(n_steps):
        mean, cov = kf.filter_update(mean, cov)
        rows.append(np.asarray(mean)[:n_features])
    return np.asarray(rows)


def forecaster(data, n_steps, future_index, **kwargs):
    """Forecast `n_steps` ahead by iterating `filter_update` with no new observations.

    Parameters
    ----------
    data : pandas.DataFrame
        The (fit-time) data; only its column names/order are used.
    n_steps : int
        Number of steps to forecast ahead.
    future_index : pandas.Index
        Index to assign to the forecasted rows.
    **kwargs
        `kf`, `mean`, `cov`, `n_features` : the fitted filter and final
        filtered state from `fitter`.

    Returns
    -------
    pandas.DataFrame
        Forecasted observations (the newest block of the delay-embedded
        state), indexed by `future_index`, columns matching `data`.
    """
    kf = kwargs['kf']
    mean, cov = kwargs['mean'], kwargs['cov']
    n_features = kwargs.get('n_features', data.shape[1])

    rows = _roll_forward(kf, mean, cov, n_steps, n_features)
    return pd.DataFrame(rows, index=future_index, columns=data.columns)


def applier(fitted_params, new_data, t):
    """`predict_new` path: filter the NEW series with the LEARNED
    transition/observation model (no EM -- `kf` is reused unchanged),
    then iterate `filter_update` forward to forecast beyond it."""
    from .common import resolve_t

    kf = fitted_params['kf']
    n_features = fitted_params.get('n_features', new_data.shape[1])
    n_steps, future_index = resolve_t(new_data, t)
    if n_steps <= 0:
        return new_data.loc[future_index]

    x = np.ma.masked_invalid(new_data.to_numpy(dtype=float))
    means, covs = kf.filter(x)
    mean, cov = means[-1], covs[-1]

    rows = _roll_forward(kf, mean, cov, n_steps, n_features)
    return pd.DataFrame(rows, index=future_index, columns=new_data.columns)


class Kalman(Forecaster):
    """Kalman-filter forecaster over a delay-embedded state: estimate the
    transition dynamics from the data (least-squares companion matrix over
    the last `lags` observations, held to a spectral radius of at most 1
    unless the fit is strongly over-determined, so the forecast cannot
    diverge geometrically in the horizon on the strength of an
    under-determined fit), EM-refine the noise covariances, then iterate
    `filter_update` (no observations) to forecast forward.

    Parameters
    ----------
    n_iter : int
        Number of EM iterations used to fit the transition/observation
        noise covariances and the initial state (default: 5). The
        transition matrix itself is estimated by least squares and then
        held non-explosive unless strongly over-determined, NOT by EM (EM
        from pykalman's identity initialization converges to flat
        forecasts; see the module docstring).
    lags : int or None
        Delay-embedding order: how many trailing observations make up the
        state vector (default: None, which uses up to 5, shrinking for
        wide data so that ``lags * n_features <= 32``, and never more than
        ``n_observations - 1``).
    """

    def __init__(self, n_iter=5, lags=None):
        required = ['kf', 'mean', 'cov', 'n_features']
        super().__init__(n_iter=n_iter, lags=lags, fitter=fitter, forecaster=forecaster,
                          applier=applier, data=None, required=required)

        self.n_iter = n_iter
        self.lags = lags
        self.fitter = fitter
        self.forecaster = forecaster
        self.applier = applier
        self.data = None
        self.required = required
