# -*- coding: utf-8 -*-
"""`hyp.predict(x, model=[...])`: several forecasters in one call (GH #285).

The LIST/mapping form of ``model=`` is the contract the later
``hyp.plot(..., predict=['Kalman', 'ARIMA', 'GP'])`` overlay consumes: a
dict keyed by model NAME, whose values are exactly what a single-model call
returns. Real forecasters on real (seeded) data throughout -- the whole
point is that each key holds that model's own forecast.
"""
import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.predict.common import Forecaster
from hypertools.predict.kalman import Kalman


def _series(n=60, ncols=2, seed=0):
    """A short, smooth, deterministic multivariate timeseries."""
    t = np.arange(n)
    rng = np.random.default_rng(seed)
    trend = 0.05 * t
    return pd.DataFrame({'a': trend + np.sin(t / 5.0) + 0.01 * rng.standard_normal(n),
                         'b': trend - np.sin(t / 5.0) + 0.01 * rng.standard_normal(n)}
                        ).iloc[:, :ncols]


def test_list_of_models_returns_one_forecast_per_model():
    df = _series()
    out = hyp.predict(df, model=['AutoRegressor', 'Kalman'], t=5)
    assert isinstance(out, dict)
    assert list(out) == ['AutoRegressor', 'Kalman']
    for forecast in out.values():
        assert isinstance(forecast, pd.DataFrame)
        assert forecast.shape == (5, 2)
    # each value is the SAME forecast the single-model call produces
    single = hyp.predict(df, model='AutoRegressor', t=5)
    assert np.allclose(out['AutoRegressor'].to_numpy(), single.to_numpy())
    # ...and the models really differ (the dict is not one forecast twice)
    assert not np.allclose(out['AutoRegressor'].to_numpy(),
                           out['Kalman'].to_numpy())


def test_tuple_of_models_behaves_like_a_list():
    df = _series()
    out = hyp.predict(df, model=('AutoRegressor', 'Kalman'), t=4)
    assert list(out) == ['AutoRegressor', 'Kalman']


def test_names_are_canonical_registry_spellings():
    # aliases and casing resolve to the registry name, so the plot legend
    # says 'GaussianProcess' whichever spelling the caller used
    df = _series(n=40)
    out = hyp.predict(df, model=['gp', 'kalman'], t=3)
    assert list(out) == ['GaussianProcess', 'Kalman']


def test_repeated_models_are_numbered():
    df = _series(n=40)
    out = hyp.predict(df, model=[{'model': 'AutoRegressor', 'kwargs': {'lags': 5}},
                                 {'model': 'AutoRegressor', 'kwargs': {'lags': 15}}],
                      t=3)
    assert list(out) == ['AutoRegressor', 'AutoRegressor (2)']
    # different lags -> genuinely different forecasts under the same name
    assert not np.allclose(out['AutoRegressor'].to_numpy(),
                           out['AutoRegressor (2)'].to_numpy())


def test_mapping_form_names_the_models():
    df = _series(n=40)
    out = hyp.predict(df, model={'short memory': {'model': 'AutoRegressor',
                                                 'kwargs': {'lags': 3}},
                                 'long memory': {'model': 'AutoRegressor',
                                                 'kwargs': {'lags': 20}}},
                      t=3)
    assert list(out) == ['short memory', 'long memory']


def test_class_and_instance_specs_are_named_by_type():
    df = _series(n=40)
    out = hyp.predict(df, model=[Kalman, Kalman()], t=3)
    assert list(out) == ['Kalman', 'Kalman (2)']


def test_single_dict_spec_is_still_one_model():
    # a dict carrying 'model'/'kwargs' is a SPEC, not a name->spec mapping
    df = _series(n=40)
    out = hyp.predict(df, model={'model': 'AutoRegressor', 'kwargs': {'lags': 5}},
                      t=3)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (3, 2)


def test_list_form_on_a_list_of_datasets():
    a, b = _series(n=40, seed=0), _series(n=40, seed=1)
    out = hyp.predict([a, b], model=['AutoRegressor', 'Kalman'], t=4)
    assert list(out) == ['AutoRegressor', 'Kalman']
    for forecasts in out.values():
        assert isinstance(forecasts, list) and len(forecasts) == 2
        assert all(f.shape == (4, 2) for f in forecasts)


def test_list_form_with_return_model_gives_parallel_dicts():
    df = _series(n=40)
    forecasts, models = hyp.predict(df, model=['AutoRegressor', 'Kalman'], t=3,
                                    return_model=True)
    assert list(forecasts) == list(models) == ['AutoRegressor', 'Kalman']
    assert all(isinstance(m, Forecaster) and m.is_fitted for m in models.values())
    # the returned models are reusable on new data, one per name
    reused = hyp.predict(_series(n=30, seed=2), model=models['Kalman'], t=2)
    assert reused.shape == (2, 2)


def test_kwargs_apply_to_every_model_in_the_collection():
    df = _series(n=40)
    out = hyp.predict(df, model=['AutoRegressor', 'AutoRegressor'], t=3, lags=4)
    single = hyp.predict(df, model='AutoRegressor', t=3, lags=4)
    assert np.allclose(out['AutoRegressor'].to_numpy(), single.to_numpy())


def test_empty_collection_raises():
    with pytest.raises(ValueError, match='model=\\[\\] is empty'):
        hyp.predict(_series(n=20), model=[], t=2)


def test_reserved_name_raises():
    with pytest.raises(ValueError, match='reserved'):
        hyp.predict(_series(n=20), model={'truth': 'Kalman'}, t=2)


def test_unknown_name_inside_a_collection_still_raises_the_usual_error():
    with pytest.raises(ValueError, match='unknown predict model'):
        hyp.predict(_series(n=20), model=['Kalman', 'Kalmann'], t=2)


def test_chronos_in_a_collection():
    pytest.importorskip('chronos')
    df = _series(n=40)
    out = hyp.predict(df, model={'tiny Chronos': {
        'model': 'Chronos',
        'kwargs': {'model_name': 'amazon/chronos-t5-tiny'}}}, t=3)
    assert list(out) == ['tiny Chronos']
    assert out['tiny Chronos'].shape == (3, 2)
