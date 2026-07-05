# -*- coding: utf-8 -*-
"""
=============================
Forecasting timeseries with predict
=============================

The `predict` kwarg overlays a forecast on top of your plotted data: a
dashed, same-color tail extending `t` steps past the end of each dataset.
Under the hood this calls `hypertools.predict`, which supports several
forecasting models -- `'Kalman'` (a linear-Gaussian state-space filter,
used here), `'GaussianProcess'`, `'AutoRegressor'` (any sklearn regressor
run recursively), `'ARIMA'`, `'Laplace'`, and `'Chronos'` (a HuggingFace
time-series foundation model) -- selected via `model=` when calling
`hypertools.predict` directly. Calling `hyp.predict(data, model=...,
t=..., return_model=True)` also returns the fitted forecaster alongside
the forecast, so the same fitted model can be reused (without
re-estimating) on new data.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

# import
import numpy as np
import hypertools as hyp

# simulate three ~70-row random walks
np.random.seed(1234)
n_samples = 70
n_dims = 3
data = [np.cumsum(np.random.randn(n_samples, n_dims), axis=0) for _ in range(3)]

# plot, forecasting 30 steps ahead with a Kalman filter
hyp.plot(data, predict='Kalman', t=30,
         legend=['walk 1', 'walk 2', 'walk 3'])
