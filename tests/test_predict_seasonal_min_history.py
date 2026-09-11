"""ARIMA's minimum history counts its SEASONAL order (release review
2026-09-11, finding 3).

`ARIMA.min_history_for` only read ``order``, so a seasonal fit on a short
history fell through to statsmodels, which raised a bare ``IndexError``
(exactly ``d + D*s + 1`` rows) or ``numpy.linalg.LinAlgError`` instead of the
library's clear "shorter than the N observations ARIMA needs" message. The
failing lengths below were measured on statsmodels 0.14 with real fits.
"""
import numpy as np
import pytest

import hypertools as hyp
from hypertools.plot.forecast import model_min_history
from hypertools.predict.arima import ARIMA

# (order, seasonal_order, floor): floor = max(d + D*s + 2, p + P*s + q + Q*s + 1)
CASES = [((1, 1, 1), (1, 1, 1, 12), 27),
         ((1, 1, 0), (0, 1, 1, 7), 10),
         ((1, 0, 0), (0, 1, 0, 12), 14)]


def _series(n):
    return np.random.default_rng(0).normal(size=(n, 2)).cumsum(axis=0)


@pytest.mark.parametrize('order,seasonal,floor', CASES)
def test_seasonal_min_history_counts_the_seasonal_lags(order, seasonal, floor):
    assert ARIMA.min_history_for(order=order, seasonal_order=seasonal) == floor
    assert ARIMA(order=order, seasonal_order=seasonal).min_history == floor
    assert model_min_history({'model': 'ARIMA', 'kwargs': {
        'order': order, 'seasonal_order': seasonal}}) == floor
    # the non-seasonal default is unchanged
    assert ARIMA.min_history_for(order=order) == ARIMA.min_history_for(
        order=order, seasonal_order=(0, 0, 0, 0))


@pytest.mark.parametrize('order,seasonal,floor', CASES)
def test_short_seasonal_histories_get_the_clear_message(order, seasonal, floor):
    x = _series(60)
    for n in range(3, floor):
        with pytest.raises(ValueError, match=r'seasonal_order=') as caught:
            hyp.predict(x[:n], model='ARIMA', order=order,
                        seasonal_order=seasonal, t=3)
        assert f'{floor} observation' in str(caught.value)
    forecast = hyp.predict(x[:floor], model='ARIMA', order=order,
                           seasonal_order=seasonal, t=3)
    assert forecast.shape == (3, 2)
    assert np.isfinite(forecast.to_numpy()).all()


def test_sparse_seasonal_lags_count_their_highest_lag():
    # statsmodels' sparse form: seasonal AR lags 1 and 2 at period 4
    assert ARIMA.min_history_for(order=(0, 0, 0),
                                 seasonal_order=([1, 2], 0, 0, 4)) == 9
