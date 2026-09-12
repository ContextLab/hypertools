"""Release review 2026-09-09: integer clocks must not overflow or lose gaps."""
import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.predict.common import resolve_t
from hypertools.predict.time import infer_step, time_coordinates


MODELS = ['Kalman', 'ARIMA', 'GaussianProcess',
          {'model': 'AutoRegressor', 'kwargs': {'lags': 2}}]
TIMES = np.array([0, 1, 3, 6, 7, 9, 13, 15, 17, 20, 23, 24])


def frame(offset=0, dtype='int64'):
    """Identical signals on clocks differing only in epoch and storage type."""
    return pd.DataFrame(
        {'x': np.sin(TIMES), 'y': TIMES ** 2},
        index=pd.Index([offset + int(t) for t in TIMES], dtype=dtype))


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('dtype,offset', [
    ('uint64', 0), ('UInt64', 0),
    ('uint64', 2**63 + 10), ('UInt64', 2**63 + 10),
    ('int64', 2**62 + 10), ('Int64', 2**62 + 10),
])
def test_integer_clock_storage_and_epoch_do_not_change_forecasts(model, dtype, offset):
    expected, reference = hyp.predict(frame(), model=model, t=3, return_model=True)
    actual, fitted = hyp.predict(frame(offset, dtype), model=model, t=3,
                                 return_model=True)
    np.testing.assert_allclose(actual, expected)
    assert [int(t) - offset for t in actual.index] == expected.index.tolist()
    # Applying the learned model to a new epoch preserves its fitted time scale.
    expected_reuse = reference.predict_new(frame(100), 3)
    actual_reuse = fitted.predict_new(frame(offset + 100, dtype), 3)
    np.testing.assert_allclose(actual_reuse, expected_reuse)
    assert [int(t) - offset for t in actual_reuse.index] == expected_reuse.index.tolist()


@pytest.mark.parametrize('model', MODELS)
def test_unsigned_clock_backtesting_matches_signed_clock(model):
    expected_scores, expected = hyp.predict(frame(), model=model, holdout=3,
                                             return_forecasts=True)
    offset = 2**63 + 10
    actual_scores, actual = hyp.predict(frame(offset, 'uint64'), model=model,
                                         holdout=3, return_forecasts=True)
    pd.testing.assert_frame_equal(actual_scores, expected_scores)
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name])
        assert [int(t) - offset for t in actual[name].index] == expected[name].index.tolist()


def test_signed_clock_gaps_across_dtype_bounds_do_not_overflow():
    index = pd.Index([-2**63 + 1, -2**62, 2**62, 2**63 - 1])
    expected_gaps = [int(b) - int(a) for a, b in zip(index[:-1], index[1:])]
    assert infer_step(index) == np.median(expected_gaps)
    expected = [(int(t) - int(index[-1])) / 2**62 for t in index]
    np.testing.assert_allclose(time_coordinates(index, index[-1], 2**62), expected)


@pytest.mark.parametrize('dtype,limit', [('int64', 2**63 - 1), ('uint64', 2**64 - 1)])
def test_integer_forecast_horizon_cannot_wrap_into_the_past(dtype, limit):
    data = pd.DataFrame({'y': [1., 2., 3.]},
                        index=pd.Index([limit - 2, limit - 1, limit], dtype=dtype))
    count, future = resolve_t(data, 3)
    assert count == 3
    assert future.tolist() == [limit + 1, limit + 2, limit + 3]
