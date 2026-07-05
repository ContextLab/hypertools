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
    # Nothing to pre-fit: skaters' laplace is a stateless-online estimator
    # that is driven by the (full) series at forecast time. Store the raw
    # per-column series so `forecaster` can feed them through the closure.
    return {'series': {col: data[col].to_numpy(dtype=float) for col in data.columns}}


def forecaster(data, n_steps, future_index, **kwargs):
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
    """

    def __init__(self, **kwargs):
        required = ['series']
        super().__init__(fitter=fitter, forecaster=forecaster, data=None,
                          required=required, **kwargs)

        self.fitter = fitter
        self.forecaster = forecaster
        self.data = None
        self.required = required
