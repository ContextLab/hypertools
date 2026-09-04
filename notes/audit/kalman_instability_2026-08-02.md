# Kalman forecaster numerical instability — root cause and fix

**Date**: 2026-08-02
**Branch**: `dev-1.0`
**File**: `hypertools/predict/kalman.py`
**Status**: FIXED (tests added, full suite green)

---

## 1. Symptom

`hyp.predict(x, model='Kalman', t=12)` returned forecasts astronomically larger than
the data they were supposed to continue.

Reproducer (12 seeds x 36 history lengths = 432 fits on 40x3 drifting random walks):

```
19/432 fits exceed 100x the data range; worst = 10309425.2x
```

Concrete single case — `rng = np.random.default_rng(3)`,
`x = np.cumsum(rng.standard_normal((40,3)) + 0.6, 0)`, `hyp.predict(x[:20], model='Kalman', t=12)`:
max |forecast| ~= 1.8e5 against a data range of 28.06.

The blow-up was **not monotonic in history length**: k=20 and k=22 exploded while
k=15, k=30 and k=40 were fine. That non-monotonicity was the decisive clue.

---

## 2. Root cause

`_companion_transition` (`hypertools/predict/kalman.py`, the `np.linalg.lstsq` call) estimated
the delay-embedded VAR coefficient block by **unconstrained ordinary least squares**, with
nothing anywhere checking that the resulting transition operator was non-explosive.

`_roll_forward` then iterates `kf.filter_update(mean, cov)` with **no new observations**, which
for a linear-Gaussian state-space model is exactly the prediction step `mean <- A @ mean`. So the
forecast at step `t` is `A**t @ mean`, and its magnitude grows as **rho(A) ** t** where `rho` is the
spectral radius. Any estimate with `rho > 1` diverges geometrically in the forecast horizon.

Why the estimate lands outside the unit circle, and why non-monotonically:

- `_resolve_lags` picks `lags = 5` for `d = 3`, i.e. `state_dim = lags * d = 15` predictors.
- The design matrix has only `n - lags` rows.
- When `n - lags` approaches `lags * d` the regression is **near-saturated** (at `n = 20`:
  15 rows, 15 columns — exactly determined, i.e. interpolating). It is also intrinsically
  collinear, because consecutive lags of a random walk are nearly identical.
- Near-singular design => huge coefficients => spectral radius far above 1.

`_resolve_lags` enforces only `lags < n`; it never compares the number of observations to the
number of free parameters (`lags * d * d` = 45 coefficients at `d=3, lags=5`). Contrast
`hypertools/predict/autoreg.py` (the closest analogue, also a lagged-feature autoregression),
which *does* guard this — `if not n > lags: raise ValueError(...)` — and additionally defaults
to a **regularized** estimator (`Ridge`) rather than OLS. Kalman did neither.

### Eigenvalue / conditioning evidence

Seed 3, drift 0.6, 40x3 random walk (data range 28.06), varying history length `k`.
`rows` = regression rows (`n - lags`), `params` = predictors (`lags * d` = 15):

| k | rows | params | cond(design) | sigma_min | max abs coef | rho(A) | rho^12 |
|-|-|-|-|-|-|-|-|
| 15 | 10 | 15 | 5.52e+01 | 1.185 | 0.687 | 1.0513 | 1.8 |
| 18 | 13 | 15 | — | — | — | 1.1686 | 6.5 |
| 19 | 14 | 15 | 2.52e+02 | 0.368 | 1.711 | 1.6714 | 4.8e+02 |
| **20** | **15** | **15** | **1.32e+03** | **0.076** | **7.918** | **1.8172** | **1.3e+03** |
| 21 | 16 | 15 | 3.86e+02 | 0.282 | 1.495 | 1.1378 | 4.7 |
| **22** | **17** | **15** | **2.09e+02** | 0.560 | 2.386 | **2.1081** | **7.7e+03** |
| 25 | 20 | 15 | 1.19e+02 | 1.526 | 1.202 | 1.0660 | 2.2 |
| 30 | 25 | 15 | 1.19e+02 | 1.526 | 1.202 | 1.0340 | 1.5 |
| 40 | 35 | 15 | 1.38e+02 | 2.055 | 1.150 | 1.0593 | 2.0 |

The spike sits exactly where `rows ~ params`, and `sigma_min` collapses to 0.076 at k=20.
That is the whole mechanism: **ill-conditioned saturated regression -> explosive operator ->
operator raised to the power of the horizon.**

Across all 12 seeds, every history length that produced `rho**12 > 1e3` was in
k = 18..22 — the saturation band — with `rho` up to **4.157**:

```
(seed, k, rho): (0,20,2.204) (1,19,4.157) (1,20,2.036) (3,20,2.007) (3,22,1.986)
                (4,20,3.897) (5,18,2.215) (5,19,2.039) (5,22,2.101) (6,19,2.246)
                (7,19,3.777) (7,20,3.262) (8,20,2.210) (9,21,2.341) (10,20,2.361)
                (11,20,3.632)
```

---

## 3. Which forecasters are affected

Same sweep (12 seeds x history lengths 5..40, 40x3 drifting random walks, `t=12`),
fraction of fits whose max |forecast| exceeds 100x the data range:

| model | fits over 100x | fraction | worst ratio | notes |
|-|-|-|-|-|
| **Kalman** | **19 / 432** | **4.40%** | **1.03e+07** | the defect |
| GaussianProcess | 0 / 432 | 0.00% | 1.389 | clean |
| ARIMA | 0 / 432 | 0.00% | 1.404 | statsmodels 0.14.6 installed; clean |
| AutoRegressor | 0 / 360 | 0.00% | 2.948 | clean; 72 short-history fits correctly *raise* `ValueError` ("needs more than lags=10 observations") instead of fitting garbage |
| Laplace | 0 / 432 | 0.00% | 59.82 | under the bar, but the loosest of the clean models — worth watching |
| Chronos | 0 / 432 | 0.00% | 1.284 | clean (`chronos-forecasting` happens to be installed in this venv) |

`sklearn` 1.8.0 supplies GaussianProcess; `statsmodels` 0.14.6 supplies ARIMA.

**Only Kalman is affected.** AutoRegressor is the instructive comparison: same lagged-feature
autoregression idea, but it guards observations-vs-parameters and regularizes, so it never blows up.

---

## 4. The fix

Constrain the *estimated dynamics* to the non-explosive region — but only when the fit is not
well enough determined to justify them. New helper
`_constrain_stability(A, lags, d, n_rows, rho_max=_MAX_SPECTRAL_RADIUS, margin=_TRUST_EXPLOSIVE_ROWS_PER_PARAM)`
in `hypertools/predict/kalman.py`, called at the end of `_companion_transition`; new module
constants `_MAX_SPECTRAL_RADIUS = 1.0` and `_TRUST_EXPLOSIVE_ROWS_PER_PARAM = 3.0`.

Rule: if `rho(A) <= 1`, return unchanged (the common case). If `rho(A) > 1` **and** the regression
is over-determined by at least 3 rows per predictor, keep it — the explosive estimate is real. Only
otherwise pull it back, by scaling the lag-`j` coefficient block by `c ** j` with `c = 1 / rho`.

**Why that particular rescaling.** The companion matrix's characteristic polynomial is
`det(z**p I - sum_j A_j z**(p-j))`. Substituting `A_j -> c**j A_j` and `z = c*w` factors out
`c**(p*d)`, so every eigenvalue is scaled by exactly `c`. Verified numerically to 3e-15 across
`(d, lags)` in {(3,5), (2,4), (1,3), (4,2)}, with the companion **shift block left bit-identical** —
the state remains an honest delay embedding, which matters because `applier` reuses the same `kf`
on new data via `kf.filter`. A naive alternative (scaling all of `A` by `c`) would have scaled the
structural shift block too and destroyed the delay-embedding interpretation.

**This is modelling, not clipping.** Nothing touches the forecast output. What changes is the
estimated transition operator, which is pulled back into the stable region — the standard
stationarity/stability shrinkage of the VAR forecasting literature. The result is a
self-consistent non-explosive linear system that pykalman filters normally. Per the brief: bounded
output here genuinely does require constraining the model, because an explosive operator iterated
`t` times cannot produce bounded forecasts by any means other than constraining the operator (or
censoring the output, which would be clipping).

**Why `rho_max = 1` and not something smaller.** A random walk, a linear trend and an undamped
oscillation all sit *exactly* at `rho = 1`. The existing test fixture (`trend + sine`, n=70) fits
to `rho = 1.000000` — so the constraint is a **no-op on well-conditioned legitimate dynamics** and
engages only on explosive estimates. Shrinking below 1 would reintroduce the flat/mean-reverting
forecasts that motivated estimating the transition matrix by least squares in the first place
(the QC 2026-07 F16-predict-001 fix documented in the module docstring). Measured: `rho_max=1.0`
gives held-out MAE 5.31 vs `rho_max=0.99` at 5.51.

### Why the over-determination gate exists (the important part)

A **blanket** `rho <= 1` constraint was implemented first. It fixed the defect completely
(0/432, worst 1.2x) — and it was **rejected**, because measuring it against genuinely explosive
data showed it destroyed real capability:

| series | pre-fix Kalman | blanket `rho<=1` | gated (shipped) | best other model |
|-|-|-|-|-|
| noiseless `1.3**t`, 6-step MAE | 0.0 (exact) | **1018.3** | **0.0** | AutoRegressor 0.0 |
| noisy `1.15**t`, 6-step MAE | 1.75 | **20.8** | **1.75** | AutoRegressor 3.2 |

Under the blanket constraint Kalman became the **worst forecaster in the library** on growth data
(MAE 1018 against GaussianProcess 442, ARIMA 957, Laplace 0.5, AutoRegressor 0.0), having been the
**best** on the noisy version (1.75, beating AutoRegressor's 3.2). That is a real regression on
legitimate inputs and was not acceptable as the price of the fix.

The discriminator is **how well-determined the fit is**, and the measured separation is wide:

- every random-walk blow-up had `rows / params` between **0.87 and 1.13** (near-saturated: the
  regression interpolates, so its near-zero residual proves nothing);
- the exactly-exponential series is recovered from `rows / params = 4.0`.

`margin = 3` sits cleanly between them. Calibration sweep (simulated propagation, 432 fits):

| margin | fits over 100x | worst | explosive fits kept | `1.3**t` MAE | noisy `1.15**t` MAE |
|-|-|-|-|-|-|
| 2 | 0/432 | 5.181 | 71 | 0.000 | 1.76 |
| **3 (chosen)** | **0/432** | **3.767** | **0** | **0.000** | **1.76** |
| 4 | 0/432 | 3.767 | 0 | 0.000 | 1.76 |
| 5 | 0/432 | 3.767 | 0 | 1057.97 | 22.78 |

`margin=4` also works but the exponential case sits exactly at ratio 4.0, i.e. on the boundary —
`margin=3` keeps headroom on both sides. `margin=5` already flattens genuine exponentials.
Gating instead on the design-matrix condition number (`cond <= 50` or `<= 20`) was also tried and
**rejected**: a pure exponential's design matrix is itself highly collinear, so a condition gate
flattens exactly the case the gate exists to protect (MAE 1057.97).

### Candidate fixes measured before choosing

Held-out MAE = mean absolute error of the 12-step forecast against the actual continuation;
"naive" = repeating the last observation (2.921).

| candidate | fits over 100x | worst ratio | max rho | held-out MAE |
|-|-|-|-|-|
| baseline (no fix) | 19/432 | 1.03e+07 | 4.157 | 61452 |
| cap lags at 2 rows/param | 3/432 | 1.33e+07 | 4.346 | 44049 |
| cap lags at 3 rows/param | 3/432 | 1.33e+07 | 4.346 | 44049 |
| ridge 1e-3 | 0/432 | 60.85 | 1.510 | 5.92 |
| ridge 1e-2 | 0/432 | 10.08 | 1.313 | 4.40 |
| blanket stability rho<=1 | 0/432 | 3.767 | 1.000 | 5.31 |
| blanket stability rho<=0.99 | 0/432 | 3.697 | 0.990 | 5.51 |
| held-out-validated stability | 10/432 | 1.03e+07 | — | — |
| lags-cap + blanket stability | 0/432 | 1.225 | 1.000 | 3.91 |
| **gated stability (CHOSEN)** | **0/432** | **3.767** | **1.000\*** | **5.31** |

\* on this random-walk data the gate never trusts an explosive fit, so it coincides with the
blanket variant here; it diverges on genuinely explosive data (see above).

Notes on the rejected candidates:

- **Lags cap alone is falsified as a fix**: still 3/432 blow-ups, worst 1.3e7. Restricting the
  parameter count reduces how often the regression saturates but does nothing about the operator
  actually being explosive, which is the mechanism.
- **Ridge alone is insufficient and arbitrary**: it helps a lot but does not *guarantee*
  `rho <= 1` (max rho still 1.51 at 1e-3, 1.31 at 1e-2), leaves a 60x / 10x worst case, and
  requires tuning a penalty with no principled value.
- **Held-out validation of the explosive fit** (refit on a truncated window, roll forward, keep
  whichever operator predicts the held-out tail better) was prototyped and **falsified**: still
  10/432 blow-ups at worst 1.03e7, because a saturated regression also interpolates its truncated
  refit, so the holdout error looks deceptively small. 207 explosive operators were wrongly kept.
- **Lags cap + stability** was the most accurate combination on random walks, but the lags cap is a
  separable change that alters the documented `lags=None` auto-choice for all data, so it was
  deliberately not bundled. Recorded as a candidate follow-up (see section 6).

### Measured effect end-to-end

Original reproducer (432 fits):

```
before: 19/432 fits exceed 100x the data range; worst = 10309425.2x
after:   0/432 fits exceed 100x the data range; worst = 1.17x
```

Single case (seed 3, drift 0.6, `x[:20]`, `t=12`): max |forecast| **1.8e5 -> 8.019**, against a
data range of 28.06.

Wider stress through the real code path — `d` in {1,2,3,5}, 30 seeds, drift 0 and 0.5, history
lengths 6..45, **3360 fits**:

```
0/3360 over 100x, worst = 4.395x
```

(A simulation that propagated from raw observations rather than the EM-refined filtered state
predicted a single 148x leak at this breadth; it does not survive end-to-end, because the Kalman
filter's initial-state refinement damps the first propagation step.)

Held-out forecast accuracy on random walks post-fix: **MAE 4.902** (naive last-value 2.921),
median 4.200 vs 2.852 — roughly four orders of magnitude better than the pre-fix 61452, confirming
the constraint improves the model rather than damaging it. Kalman remains worse than naive on
*pure random-walk* data, which is expected (last-value is near-optimal for a unit-root process)
and is not the defect. Accuracy on genuinely explosive data is unchanged from pre-fix (MAE 0.0 and
1.75; see the table above).

---

## 5. Tests

Added to `tests/predict/test_kalman.py` (TDD: all six failed before the fix, all pass after;
no existing test was weakened):

| test | what it pins |
|-|-|
| `test_estimated_transition_is_non_explosive` | the real invariant: `rho(A) <= 1` at every history length 5..40 |
| `test_forecast_bounded_on_short_random_walk_history` | the exact reported case (seed 3, drift 0.6, `x[:20]`, `t=12`) stays under 10x the data range |
| `test_forecast_bounded_across_saturating_history_lengths[1-19 / 4-20 / 7-19 / 11-20]` | four independent seeds in the saturation band, each `rho > 3.6` pre-fix |
| `test_stability_constraint_preserves_delay_embedding_structure` | guards against a bad fix: the companion shift block and `H` must be untouched |
| `test_stability_constraint_is_inert_on_well_conditioned_dynamics` | guards against over-correction: trend+sine still fits `rho == 1` and the forecast does not flat-line |
| `test_genuinely_explosive_dynamics_are_still_followed` | pins the gate: `1.3**t` must keep `rho ~ 1.3` and be forecast to <5% relative error. **Fails under the blanket constraint** (rho forced to 1.0) |
| `test_noisy_exponential_growth_accuracy_is_retained` | same trade-off on noisy growth: MAE < 5 (pre-fix 1.75, blanket 20.8) |

Three of these are guards on *how* the bug may be fixed rather than detectors of the bug: the two
structural/inertness tests passed before the fix, and the two explosive-dynamics tests fail under
the blanket variant that was tried and rejected. They exist so a future simplification back to a
blanket `rho <= 1` clamp is caught.

Pre-fix run: `6 failed, 8 passed`. Post-fix `tests/predict/`: `80 passed`.

Full suite (`.venv/bin/python -m pytest -q`):

```
2834 passed, 13 skipped, 2 deselected, 1 warning in 578.98s (0:09:38)
```

(The single warning is emitted by `tests/plot/test_predict_animation.py` and belongs to concurrent
plot-side work, not to this change. `hypertools/plot/` was not touched.)

---

## 6. Follow-ups (not done here — separate changes)

1. **Observations-vs-parameters guard in `_resolve_lags`.** Kalman still happily fits 45 free
   coefficients from 45 target values when the user passes `lags=5` explicitly with `n=20`. The
   stability constraint now keeps the *forecast* bounded, but the fit itself is still an
   interpolation. Measured to improve held-out MAE from 5.31 to 3.91 when combined with the
   stability constraint. Changes the documented `lags=None` auto-choice, so it wants its own
   review. `autoreg.py` has the precedent (`ValueError` naming the fix).
2. **Regularized estimation.** Ridge shrinkage on the companion regression would attack the
   ill-conditioning at source rather than repairing its consequence, and `autoreg.py` already
   defaults to `Ridge`. Would need a principled penalty (e.g. scale-free, cross-validated).
3. **Laplace worst ratio 59.8x.** Under the 100x bar and not a defect by the criterion used here,
   but it is an order of magnitude looser than GaussianProcess/ARIMA/AutoRegressor. Worth a
   dedicated look.
4. No plot-side change is needed. `hypertools/plot/` was deliberately not touched (concurrent work).
