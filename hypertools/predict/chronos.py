"""Chronos (HuggingFace foundation model) forecaster.

Uses Amazon's pretrained Chronos time-series foundation model
(`chronos-forecasting`, built on `torch`) to forecast `t` steps ahead per
column: each column is fed through `ChronosPipeline.predict` independently
(Chronos is a univariate per-series model), which returns a set of sampled
forecast trajectories; the median (0.5 quantile) across samples is taken as
the point forecast.

`chronos-forecasting` and `torch` ship via the optional `[predict-hf]`
extra; imported lazily (inside the fitter) so `hypertools.predict` stays
importable without them, and a friendly `ImportError` is raised only when a
`Chronos` forecaster is actually fit.
"""
import numpy as np
import pandas as pd

from .common import Forecaster


def _import_chronos():
    try:
        import torch
        from chronos import ChronosPipeline
    except ImportError as e:
        raise ImportError(
            'chronos-forecasting and torch are required for the Chronos '
            'forecaster; install them with pip install "hypertools[predict-hf]"'
        ) from e
    return torch, ChronosPipeline


def fitter(data, **kwargs):
    """Load the pretrained Chronos pipeline and record each column's raw series.

    There is no actual model fitting -- Chronos is a pretrained,
    context-conditioned foundation model -- so "fitting" just loads the
    pipeline once and stores each column's series for `forecaster`.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to record.
    **kwargs
        `model_name` : str, HuggingFace Hub id of the pretrained
        checkpoint (default: `'amazon/chronos-t5-tiny'`). `device_map` :
        str, passed to `ChronosPipeline.from_pretrained` (default: `'cpu'`).

    Returns
    -------
    dict
        `{'pipeline': <loaded ChronosPipeline>, 'series': {col: <float32
        numpy array>, ...}}`.
    """
    torch, chronos_pipeline_cls = _import_chronos()
    model_name = kwargs.get('model_name', 'amazon/chronos-t5-tiny')
    device_map = kwargs.get('device_map', 'cpu')

    pipeline = chronos_pipeline_cls.from_pretrained(model_name, device_map=device_map)
    series = {col: data[col].to_numpy(dtype=np.float32) for col in data.columns}

    return {'pipeline': pipeline, 'series': series}


def forecaster(data, n_steps, future_index, **kwargs):
    """Forecast `n_steps` ahead per column via the Chronos pipeline's median prediction.

    Each column's series is fed through `pipeline.predict` independently
    (Chronos is a univariate model), producing sampled forecast
    trajectories; the median (0.5 quantile) across samples is taken as
    the point forecast.

    Parameters
    ----------
    data : pandas.DataFrame
        The (fit-time) data; only its column names/order are used.
    n_steps : int
        Number of steps to forecast ahead.
    future_index : pandas.Index
        Index to assign to the forecasted rows.
    **kwargs
        `pipeline`, `series` : loaded pipeline and per-column series from `fitter`.

    Returns
    -------
    pandas.DataFrame
        Forecasted values, indexed by `future_index`, columns matching `data`.
    """
    torch, _ = _import_chronos()
    pipeline = kwargs['pipeline']
    series = kwargs['series']

    columns = {}
    for col in data.columns:
        x = torch.tensor(series[col], dtype=torch.float32)
        fc = pipeline.predict(x[None, :], prediction_length=n_steps)
        median = fc.quantile(0.5, dim=1)[0]  # (n_steps,)
        columns[col] = np.asarray(median)

    return pd.DataFrame(columns, index=future_index, columns=data.columns)


class Chronos(Forecaster):
    """HuggingFace Chronos (`chronos-forecasting`) time-series foundation
    model forecaster: per-column, context-conditioned. There is no reusable
    "fit" beyond loading the pretrained pipeline once and remembering each
    column's raw series.

    Reuse (`predict_new` / `return_model=True`) is conditioning-by-nature:
    Chronos has no learned per-series parameters to replay, so there is no
    custom `applier` here (it stays `None`). Passing a fitted `Chronos` back
    as `model=` on new data re-derives the fitted params from the NEW series
    (re-loading the pretrained pipeline) and forecasts from there -- "reuse"
    means conditioning on the new series, not replaying anything learned
    from the original fit.

    Parameters
    ----------
    model_name : str
        HuggingFace Hub id of the pretrained Chronos checkpoint
        (default: 'amazon/chronos-t5-tiny').
    device_map : str
        Passed through to `ChronosPipeline.from_pretrained` (default: 'cpu').
    """

    def __init__(self, model_name='amazon/chronos-t5-tiny', device_map='cpu', **kwargs):
        required = ['pipeline', 'series']
        super().__init__(model_name=model_name, device_map=device_map, fitter=fitter,
                          forecaster=forecaster, data=None, required=required, **kwargs)

        self.model_name = model_name
        self.device_map = device_map
        self.fitter = fitter
        self.forecaster = forecaster
        self.data = None
        self.required = required
