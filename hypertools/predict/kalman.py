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

#: soft cap on the delay-embedded state dimension (lags * n_features); the
#: automatic `lags` choice shrinks toward 1 for wide data so the state (and
#: the EM covariance estimates) stay tractable.
_MAX_STATE_DIM = 32


def _import_kalman_filter():
    try:
        from pykalman import KalmanFilter
    except ImportError as e:
        raise ImportError(
            'pykalman is required for the Kalman forecaster. It is normally a '
            'core hypertools dependency; reinstall hypertools, or install it '
            'directly with `pip install pykalman`.'
        ) from e
    return KalmanFilter


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


def _companion_transition(x, lags):
    """Estimate the delay-embedded transition matrix by least squares.

    Regresses each observation on the `lags` observations before it
    (rows containing NaN are skipped) and embeds the coefficients as the
    top block of a companion matrix. Falls back to a random-walk
    companion (top block = [I 0 ... 0]) if fewer than 2 complete
    regression rows exist.
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
    kalman_filter_cls = _import_kalman_filter()
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
    the last `lags` observations), EM-refine the noise covariances, then
    iterate `filter_update` (no observations) to forecast forward.

    Parameters
    ----------
    n_iter : int
        Number of EM iterations used to fit the transition/observation
        noise covariances and the initial state (default: 5). The
        transition matrix itself is estimated by least squares, NOT by EM
        (EM from pykalman's identity initialization converges to flat
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
