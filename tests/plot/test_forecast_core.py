"""The pure forecast helper behind animated `predict=` (no rendering here)."""

import numpy as np
import pandas as pd
import pytest

from hypertools.plot.forecast import forecast_from_history


def test_returns_none_below_min_history():
    assert forecast_from_history(np.zeros((1, 3)), 'Kalman', t=3) is None


def test_shape_is_t_plus_one_and_starts_at_the_origin():
    rng = np.random.default_rng(0)
    history = rng.normal(size=(40, 3)).cumsum(axis=0)
    out = forecast_from_history(history, 'Kalman', t=4)
    assert out.shape == (5, 3)
    assert np.allclose(out[0], 0.0), 'first row must be the anchor (zero displacement)'


# Kalman/ARIMA/GP all reproduce a unit ramp exactly (measured first forecast
# row = 30.0 = last_obs + 1, displacement steps [1, 1, 1]). Laplace does NOT
# (measured steps [1.0, 1.328, 1.909]), so it is deliberately excluded rather
# than the tolerance being loosened to hide the difference.
@pytest.mark.parametrize('model', ['Kalman', 'ARIMA'])
def test_displacement_is_anchored_on_the_last_observation(model):
    """`hyp.predict` returns t rows that are ALL future steps, so anchoring on
    f[0] would discard a step. Verified against a deterministic ramp."""
    if model == 'ARIMA':
        pytest.importorskip('statsmodels')
    ramp = np.tile(np.arange(30.0)[:, None], (1, 3))       # step of exactly 1.0
    out = forecast_from_history(ramp, model, t=3)
    steps = np.diff(out[:, 0])
    assert np.allclose(steps, 1.0, atol=0.25), (
        f'expected ~1.0 per step from a unit ramp; got {steps}')


def test_horizon_of_one_is_supported():
    """The maintainer wants next-day forecasts, i.e. t=1 RAW samples."""
    rng = np.random.default_rng(1)
    history = rng.normal(size=(40, 3)).cumsum(axis=0)
    out = forecast_from_history(history, 'Kalman', t=1)
    assert out.shape == (2, 3)


def test_history_must_be_two_dimensional():
    with pytest.raises(ValueError, match='2-D'):
        forecast_from_history(np.arange(10.0), 'Kalman', t=3)


def test_result_is_a_plain_ndarray_even_though_predict_returns_a_dataframe():
    """Undocumented in v1: hyp.predict hands back a pandas object, whose index
    would otherwise leak into the drawing code."""
    from hypertools.predict.predict import predict as _predict
    rng = np.random.default_rng(2)
    history = rng.normal(size=(30, 3)).cumsum(axis=0)
    assert isinstance(_predict(history, model='Kalman', t=3), pd.DataFrame)
    out = forecast_from_history(history, 'Kalman', t=3)
    assert type(out) is np.ndarray
    assert out.dtype == np.float64


def test_same_history_gives_the_same_forecast():
    """Memoization in Task 2 is only sound if this holds."""
    rng = np.random.default_rng(3)
    history = rng.normal(size=(50, 3)).cumsum(axis=0)
    a = forecast_from_history(history, 'Kalman', t=4)
    b = forecast_from_history(history.copy(), 'Kalman', t=4)
    assert np.allclose(a, b)
