"""Laplace forecaster (skaters).

`skaters.api.laplace(k=t)` is a FACTORY: it returns a closure
`f(y: float, state: dict | None) -> (list[Dist], state)`. Feeding the whole
series through the closure (one observation at a time, threading `state`
forward) leaves `f`'s last return value holding a `t`-length list of `Dist`
objects -- the forecast distributions for the next `t` steps -- whose
`.mean` (an attribute, not a method) gives the forecast mean.

Verified directly against the installed skaters==0.11.0: a single
`laplace(k=t)` call handles k up to at least 100 without truncating or
raising (checked k=5, 10, 50, 80, 100 against a 70-80-point series) --
`len(dists) == k` in every case. No chunking is needed in practice, but as a
defensive belt-and-suspenders measure (in case some other install/series
combination behaves differently), `forecaster` re-feeds forecast means back
into a fresh `laplace` call for any remaining steps if a single call ever
returns fewer than the requested horizon.

`skaters` ships via the optional `[predict]` extra; it is imported lazily
(inside the fitter) so `hypertools.predict` stays importable without it, and
a friendly `ImportError` is raised only when a `Laplace` forecaster is
actually fit.
"""
import numpy as np
import pandas as pd

from .common import Forecaster


def _import_laplace():
    try:
        from skaters.api import laplace
    except ImportError as e:
        raise ImportError(
            'skaters is required for the Laplace forecaster; install it with '
            'pip install "hypertools[predict]"'
        ) from e
    return laplace


def _forecast_column(x, n_steps):
    """Feed `x` through a fresh `laplace(k=n_steps)` closure and return the
    forecast means, chunking (by re-feeding forecast means) only if a single
    factory call ever returns fewer than `n_steps` distributions."""
    laplace = _import_laplace()

    means = []
    remaining = n_steps
    series = np.asarray(x, dtype=float)

    while remaining > 0:
        f = laplace(k=remaining)
        state = None
        dists = None
        for yt in series:
            dists, state = f(float(yt), state)

        chunk = [d.mean for d in dists]
        means.extend(chunk)
        remaining -= len(chunk)

        if len(chunk) == 0:
            raise RuntimeError('skaters.api.laplace returned no forecast steps')

        # Feed the forecast means back in as if observed, to seed the next chunk.
        series = np.concatenate([series, np.asarray(chunk, dtype=float)])

    return means[:n_steps]


def fitter(data, **kwargs):
    """Record each column's raw series for the `Laplace` forecaster.

    Nothing is actually pre-fit here: `skaters.api.laplace` is a
    stateless-online estimator driven by the full series at forecast
    time, so this just stores each column's series for `forecaster`.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to record.

    Returns
    -------
    dict
        `{'series': {col: <float numpy array>, ...}}`.
    """
    # Nothing to pre-fit: skaters' laplace is a stateless-online estimator
    # that is driven by the (full) series at forecast time. Store the raw
    # per-column series so `forecaster` can feed them through the closure.
    return {'series': {col: data[col].to_numpy(dtype=float) for col in data.columns}}


def forecaster(data, n_steps, future_index, **kwargs):
    """Forecast `n_steps` ahead per column via `skaters.api.laplace`'s online state loop.

    Feeds each column's full series through a fresh `laplace(k=n_steps)`
    closure (see `_forecast_column`) and takes each returned
    distribution's mean as the point forecast.

    Parameters
    ----------
    data : pandas.DataFrame
        The (fit-time) data; only its column names/order are used.
    n_steps : int
        Number of steps to forecast ahead.
    future_index : pandas.Index
        Index to assign to the forecasted rows.
    **kwargs
        `series` : per-column raw series from `fitter`.

    Returns
    -------
    pandas.DataFrame
        Forecasted values, indexed by `future_index`, columns matching `data`.
    """
    series = kwargs['series']

    columns = {}
    for col in data.columns:
        columns[col] = np.asarray(_forecast_column(series[col], n_steps))

    return pd.DataFrame(columns, index=future_index, columns=data.columns)


class Laplace(Forecaster):
    """Per-column Laplace forecaster (skaters).

    Feeds each column's full series through `skaters.api.laplace(k=t)`'s
    online state loop, then takes the mean of each returned forecast
    distribution.

    Reuse (`predict_new` / `return_model=True`) is conditioning-by-nature:
    `laplace` is a stateless-online estimator with no learned parameters to
    replay, so there is no custom `applier` here (it stays `None`). Passing
    a fitted `Laplace` back as `model=` on new data simply re-feeds the NEW
    series through a fresh `laplace(k=t)` closure -- "reuse" means
    conditioning on the new series, not replaying anything learned from the
    original fit.
    """

    def __init__(self, **kwargs):
        required = ['series']
        super().__init__(fitter=fitter, forecaster=forecaster, data=None,
                          required=required, **kwargs)

        self.fitter = fitter
        self.forecaster = forecaster
        self.data = None
        self.required = required
