# Follow-up: `hyp.predict` forecasts depend on the units of the input

**Status:** open, NOT part of Plan 4. Raised by review round 12, finding 4.
**Filed as a GitHub issue:** not yet — needs the maintainer's go-ahead.

## What was measured

The weather study needed to know whether it could assert "rescale a city,
get the same verdict" end to end. It cannot, and the reason is a property
of the shipped forecasters rather than of the study.

Multiplying a 60x2 seasonal input series by 100 and dividing the one-step
forecast **change** back by 100 — an operation that would be exact for any
scale-equivariant model, and is exact to float precision for all six of the
study's trivial baselines:

| model | `n_iter` | relative change, column 0 | column 1 |
|-|-|-|-|
| Kalman | 1 | 0.012 | **2.03** |
| Kalman | 5 (default) | 0.003 | **0.41** |
| Kalman | 25 | 0.021 | **0.91** |
| Kalman | 100 | 0.084 | **0.62** |
| ARIMA | — | 0.0005 | **0.32** |

Reproduce with `scripts/weather_forecast_study.py`'s primitives:

```python
from scripts.market_representation_study import _predict_delta
a = _predict_delta(series, 'Kalman', 1)
b = _predict_delta(series * 100.0, 'Kalman', 1) / 100.0   # != a
```

## Why it happens

`hypertools/predict/kalman.py` estimates the transition matrix by least
squares (which *is* scale-equivariant) and then refines the noise
covariances and initial state with pykalman's EM, `n_iter=5` by default.
pykalman initialises those covariances at **identity**, which means
something completely different relative to data scaled by 100 — so EM
starts somewhere else and settles somewhere else. More iterations do not
converge the two runs together, so this is not "the default is too few".

This is ordinary behaviour for a fitted state-space model, and it is not
obviously a bug. The defect is that **nothing said so**: `hypertools.predict`
documented no scaling requirement at all.

## Done here

* A **Notes** section on `hypertools.predict`'s docstring
  (`hypertools/predict/predict.py`) stating the limitation with these
  measurements, and telling users to normalize heterogeneous features first.

## Still to decide

1. Should `predict()` offer explicit normalization (a `normalize=` stage of
   its own, or a documented `manip=` recipe) so the common case is one
   argument rather than a preprocessing step users have to know about?
2. Or scale-aware EM initialisation in `kalman.py` — initialise the
   covariances from the data's own variance instead of identity, which
   would make the model approximately scale-equivariant without changing
   its behaviour on already-normalized data?
3. Either way: a test that pins whichever guarantee is chosen. There is
   deliberately **no** test pinning the *current* behaviour, because a test
   asserting non-equivariance would lock the defect in.

## Why it does not invalidate the weather study

Every model and every baseline in that study is scored on the data as a
user would actually supply it, in its own units, and the comparison is
per-measure — so all competitors see the same series. The study's scoring
layer is separately asserted to be invariant to a city's units
(`tests/test_weather_forecast_study.py`).
