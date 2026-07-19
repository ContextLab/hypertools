# hyp.predict + hyp.impute Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `hyp.predict` module (timeseries forecasting: Kalman, ARIMA, Gaussian-process, any-sklearn-regressor autoregression, skaters.laplace, HuggingFace/Chronos) and a `hyp.impute` module (missing-data: PPCA, sklearn imputers, Kalman smoothing — resolves GH #169), both in the established module style, integrated into `hyp.plot`/`hyp.analyze`, with docs + two executed tutorials.

**Architecture:** Mirror `hypertools/manip/` exactly: a `BaseEstimator` child-per-file package with a `@dw.decorate.funnel` dispatcher resolved via `core.shared.unpack_model`. `predict` EXTENDS datasets (returns t new rows per dataset, same columns); `impute` fills NaNs in place (same shape). Plot integration appends one forecast trace per input dataset (same color, dashed, alpha), in the plotted (post normalize→reduce→align) space.

**Tech stack:** pykalman 0.11+, statsmodels (ARIMA), scikit-learn (GP, imputers, autoregression), skaters 0.11 (laplace), chronos-forecasting 2.x (HF), yfinance (docs only).

## Global Constraints

- Interpreter: ALWAYS `/Users/jmanning/hypertools/.venv/bin/python` (+ `MPLBACKEND=Agg`). Never bare python/pip/pytest.
- Branch `dev-1.0-refactor`; never push master. Commit trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- No mocks: every forecaster/imputer test runs the REAL algorithm on real (small) data. External fetches (yfinance, chronos weights, projectile dataset) are real; gate opt-in-extra tests with `pytest.importorskip`, never fake them.
- Follow the `manip/` style precisely: `common.py` base class copied-in-spirit from `hypertools/manip/common.py`; children one-per-file; dispatcher like `hypertools/manip/manip.py` (`@dw.decorate.funnel`, `unpack_model(model, valid=..., parent_class=...)`).
- GH #169 defines the `t` semantics (verbatim requirements below). #169 is the only related open issue (searched 2026-07-05; 0 comments on it).
- Keep the suite green at every commit. Local runs deselect the 6 kaleido tests (sandbox Chromium deadlock — see notes/session_2026-07-04_ci_green_legend.md).
- Wide legends / new plot elements must respect `_fit_right_legend` (measure saved pixels, not get_window_extent).

## Probed APIs (verified with real calls 2026-07-05 — use these exact patterns)

```python
# pykalman: smooth (imputation) + iterated forecast
from pykalman import KalmanFilter
kf = KalmanFilter(n_dim_obs=d, n_dim_state=d).em(X, n_iter=5)   # X: (n, d); np.ma.masked for NaNs
smoothed, _ = kf.smooth(X)                                       # imputation path
means, covs = kf.filter(X); m, c = means[-1], covs[-1]
for _ in range(t): m, c = kf.filter_update(m, c); forecast.append(m)

# statsmodels ARIMA (univariate -> per-column)
from statsmodels.tsa.arima.model import ARIMA
fc = ARIMA(x_col, order=(2, 1, 1)).fit().forecast(steps=t)       # -> (t,)

# sklearn GP on the time index (multivariate y supported natively)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
gp = GaussianProcessRegressor(RBF(10.0) + WhiteKernel(), normalize_y=True).fit(
    np.arange(n).reshape(-1, 1), X)
fc = gp.predict(np.arange(n, n + t).reshape(-1, 1))              # -> (t, d)

# skaters.laplace: FACTORY -> per-observation closure with explicit state (univariate -> per-column)
from skaters.api import laplace
f = laplace(k=t)                                    # k = horizon
state = None
for yt in x_col: dists, state = f(float(yt), state)  # feed the whole series
fc = [d.mean() for d in dists]                        # t-step forecast means

# chronos (HF forecasting; [predict-hf] extra) -- pin the tiny model
from chronos import ChronosPipeline           # package: chronos-forecasting
pipe = ChronosPipeline.from_pretrained('amazon/chronos-t5-tiny', device_map='cpu')
fc = pipe.predict(torch.tensor(x_col, dtype=torch.float32), prediction_length=t)
# -> (num_series, num_samples, t); take median over samples. Univariate -> per-column.

# yfinance (docs/tutorial only)
import yfinance as yf
df = yf.download(['AAPL','MSFT','NVDA'], period='2y', interval='1d', progress=False, auto_adjust=True)
```

## File Structure

- Create: `hypertools/predict/{__init__.py, common.py, kalman.py, gp.py, autoreg.py, arima.py, laplace.py, chronos.py, predict.py}`
- Create: `hypertools/impute/{__init__.py, common.py, ppca.py, sklearn_imputers.py, kalman.py, impute.py}`
- Modify: `hypertools/__init__.py` (export predict, impute), `hypertools/tools/analyze.py` (impute stage), `hypertools/tools/format_data.py` (route PPCA through hyp.impute), `hypertools/plot/plot.py` (+`predict=`, `t=`, `impute=`; forecast traces; return_model), `hypertools/plot/plotly_backend.py` (forecast trace parity), `pyproject.toml` (extras), `docs/api.rst`, `docs/tutorials.rst`, `docs/doc_requirements.txt` (+yfinance)
- Create: `examples/plot_predict.py`, `examples/plot_impute.py`, `docs/tutorials/stock_forecasting.ipynb`, `docs/tutorials/projectile_kalman.ipynb`
- Tests: `tests/predict/test_{common,kalman,gp,autoreg,arima,laplace,chronos,dispatcher}.py`, `tests/impute/test_{imputers,dispatcher,format_data_integration}.py`, `tests/plot/test_predict_integration.py`

---

### Task 1: predict package scaffolding — `Forecaster` base + `t` semantics

**Files:** create `hypertools/predict/__init__.py`, `hypertools/predict/common.py`; test `tests/predict/test_common.py`.

**Interfaces (produced):**
- `class Forecaster(BaseEstimator)` — mirrors `hypertools/manip/common.py`'s `Manipulator` but with a `(fitter, forecaster, required)` triple: `fit(data)` stores fitted params per dataset; `predict(t)` returns forecasts; `fit_predict(data, t)`. `data` may be a single DataFrame or a list (fit one model per dataset — GH #169: "fit ... to each numpy array/dataframe and store the results in a list of length len(data)"). Returns list-in→list-out, single-in→single-out.
- `resolve_t(data, t)` helper implementing GH #169 verbatim:
  - `t: int` → forecast t timesteps; "timestep duration is equal to the minimum non-zero difference between any pair of observations" (index-aware for time-indexed DataFrames; plain RangeIndex → step 1).
  - `t: datetime-like` + time-indexed data → number of steps up to that date (inferred step). If the date is IN THE PAST → return a negative count meaning truncate (no model); dispatcher slices the data instead of forecasting.
  - returns `(n_steps:int, future_index)` where `future_index` extends the input's index (DatetimeIndex extended by inferred freq; RangeIndex continued).
- Forecast outputs are DataFrames with `future_index` and the input's columns ("a new timeseries ... with the same dimensions as the input datasets").

**Steps:**
- [ ] Write failing tests: base fit/predict contract with a trivial inline fitter (e.g. constant-continuation), list-in/list-out, `resolve_t` for int, RangeIndex, DatetimeIndex (+future date, +past date → truncation), NotFittedError before fit.
- [ ] Implement `common.py`; run `MPLBACKEND=Agg .venv/bin/python -m pytest tests/predict/test_common.py -q` → green.
- [ ] Commit `feat(predict): Forecaster base + GH#169 t semantics`.

### Task 2: Kalman, GP, AutoReg forecasters

**Files:** create `hypertools/predict/{kalman.py, gp.py, autoreg.py}`; tests `tests/predict/test_{kalman,gp,autoreg}.py`.

- `Kalman(Forecaster)`: pykalman pattern above; `n_iter=5` default EM. Handles NaNs via `np.ma.masked_invalid` during em/filter. Guarded import: `pykalman` lives in the `[predict]` extra → raise `ImportError("pip install hypertools[predict]")` on use, and `pytest.importorskip('pykalman')` in tests.
- `GaussianProcess(Forecaster)`: sklearn pattern above; kwargs pass through (kernel, alpha, normalize_y default True). No new dep.
- `AutoRegressor(Forecaster)`: "any sklearn prediction algorithm, n timesteps ahead". Signature `AutoRegressor(model='Ridge', lags=10, **model_kwargs)`; `model` may be a string (resolved from sklearn via the registry style of `cluster/cluster.py`: explicit dict of common regressors — Ridge, Lasso, LinearRegression, RandomForestRegressor, GradientBoostingRegressor, SVR, KNeighborsRegressor), a class, or an instance. Build lagged-matrix X=(rows: windows of `lags` flattened features) → y=next row; recursive multi-step forecast feeding predictions back in. Multivariate directly (MultiOutput via native support or `sklearn.multioutput.MultiOutputRegressor` when needed).
- [ ] TDD each: synthetic linear trend + sine data; assert forecast shape `(t, d)`, index continuation, and sane direction (e.g., increasing trend keeps increasing within tolerance — loose bounds, these are real algorithms).
- [ ] Commit per forecaster or one commit `feat(predict): Kalman, GaussianProcess, AutoRegressor`.

### Task 3: ARIMA + Laplace forecasters

**Files:** create `hypertools/predict/{arima.py, laplace.py}`; tests.

- `ARIMA(Forecaster)`: statsmodels pattern; `order=(1,1,1)` default, per-column loop; `[predict]` extra guard.
- `Laplace(Forecaster)`: skaters pattern above (per-column, feed full series through the state loop, k=t). NOTE: skaters' `laplace(k=...)` caps/behaves per its own docs — if k>max supported, chunk by re-feeding forecasts (verify at implementation; probed k=5 works). `[predict]` extra guard.
- [ ] TDD as Task 2; commit `feat(predict): ARIMA + skaters Laplace forecasters`.

### Task 4: Chronos (HF) forecaster + dispatcher + export

**Files:** create `hypertools/predict/{chronos.py, predict.py}`; modify `hypertools/__init__.py`; tests `tests/predict/test_{chronos,dispatcher}.py`.

- `Chronos(Forecaster)`: pattern above; default `model_name='amazon/chronos-t5-tiny'` (~16 MB — real download in test), median over sample dim; `[predict-hf]` extra guard (`chronos-forecasting` + torch).
- `predict.py` dispatcher — copy `manip/manip.py`'s shape exactly:
  ```python
  FORECASTERS = [Kalman, GaussianProcess, AutoRegressor, ARIMA, Laplace, Chronos]

  @dw.decorate.funnel
  def predict(data, model='Kalman', t=10, **kwargs):
      resolved = unpack_model(model, valid=FORECASTERS, parent_class=Forecaster)
      ...
      return resolved.fit_predict(data, t=t)
  ```
  String names, dicts (`{'model': ..., 'params': ...}` AND fork `{'model','args','kwargs'}` forms — see `core.shared.unpack_model`), classes, and instances all resolve. Import children lazily inside the registry so missing extras only fail when that model is requested.
- **`return_model=True` flag (Jeremy's requirement, matching `core/model.py::apply_model`'s exact convention):** `predict(data, model=..., t=..., return_model=True)` returns `(forecasts, fitted_forecaster)`. The fitted instance can be passed back as `model=` on FUTURE calls with NEW data: the dispatcher detects an already-fitted `Forecaster` instance and calls a no-re-estimation path — `Forecaster.predict_new(data, t)` — that applies the LEARNED parameters to the new series and forecasts from its end. Per-child semantics: Kalman filters the new series with the learned transition/observation matrices (no EM); ARIMA uses `fitted_results.apply(new_series)`; AutoReg/GP predict with the already-fit estimator (GP conditions on original fit; AutoReg recursion seeds from the new series' tail); Laplace/Chronos are context-conditioned by nature — reuse == condition on the new series (document this in their docstrings). `Forecaster` base: add `is_fitted` property + `predict_new(data, t)` dispatching to a child `applier(fitted_params, new_data, t)` callable (triple becomes a quadruple: fitter, forecaster, applier, required; applier=None → falls back to conditioning-on-new-data behavior).
- `hypertools/__init__.py`: `from .predict.predict import predict` (note: this shadows any `hypertools.predict` subpackage attr, same known pattern as `align`/`cluster` — fine, documented).
- [ ] Dispatcher tests: every model name resolves (skip-gated per extra); dict/class/instance forms; `t` int + datetime; list-in/list-out; error message on unknown name lists supported names.
- [ ] return_model round-trip tests: `fc, fitted = predict(A, model='Kalman', t=5, return_model=True)`; then `predict(B, model=fitted, t=5)` succeeds on new data, produces B-continuation forecasts, and does NOT re-estimate (assert the learned params object is identical pre/post — e.g. same transition-matrix ndarray id/values). Same round-trip for GP and AutoRegressor (base-dep models).
- [ ] Chronos test: real `chronos-t5-tiny` download + forecast, `pytest.importorskip('chronos')`.
- [ ] Commit `feat(predict): Chronos forecaster + hyp.predict dispatcher`.

### Task 5: impute package (resolves GH #169's imputation half)

**Files:** create `hypertools/impute/*`; modify `hypertools/__init__.py`, `hypertools/tools/format_data.py`; tests `tests/impute/*`.

- `Imputer(Forecaster-style base)` in `common.py` (fit/transform/fit_transform; same-shape contract; per-dataset list handling).
- `PPCA(Imputer)`: clean interface over `hypertools/external/ppca.py` (what `format_data` calls today). Cannot fill all-NaN rows — document + warn (that's #169's motivating gap).
- `SklearnImputer` children by registry: `SimpleImputer`, `KNNImputer`, `IterativeImputer` (enable_iterative_imputer import dance). No new dep.
- `KalmanImputer(Imputer)`: pykalman `em(np.ma.masked_invalid(X)).smooth(...)`, replace ONLY the missing entries with smoothed values. MUST fill rows where ALL features are NaN (the #169 case) — regression-test exactly that: a dataset with 3 fully-NaN interior rows comes back finite everywhere, non-missing entries unchanged.
- `impute.py` dispatcher (funnel + unpack_model), default `model='PPCA'` (behavior-preserving).
- **`return_model=True`** here too (same `(result, fitted)` convention): a fitted Imputer passed back as `model=` transforms NEW data with the learned parameters (e.g. fitted PPCA components / KNN fit / Kalman matrices) without refitting — mirrors sklearn imputer fit/transform split naturally.
- `format_data.py`: route its PPCA fill through `hypertools.impute` (default PPCA → byte-identical behavior; existing missing-data tests must stay green unchanged).
- `hyp.impute` export.
- [ ] Commit `feat(impute): PPCA/sklearn/Kalman imputers + hyp.impute (GH #169)`.

### Task 6: plot() / analyze() integration

**Files:** modify `hypertools/tools/analyze.py`, `hypertools/plot/plot.py`, `hypertools/plot/plotly_backend.py`; tests `tests/plot/test_predict_integration.py`.

- `analyze(..., impute=None)`: when set (str/dict/class/instance), applied at the format_data stage in place of default PPCA. Threaded from `plot(impute=...)`.
- `plot(..., predict=None, t=10)`: after `analyze` produces `xform` (the plotted space — normalize→reduce→align), if `predict` is set: `forecasts = predictor(xform, model=predict, t=t)`; append one forecast trace PER input dataset — same color as its source dataset, `linestyle='--'`, `alpha=0.6`, `_nolegend_` (mirror the trail-artist precedent in matplotlib_backend). Forecast trace connects to the last observed point (prepend the final observed row so the dashed line is continuous). Same-dimension guarantee comes free (forecasts are in xform space).
- plotly backend: same appearance (`dash='dash'`, opacity 0.6, `showlegend=False`).
- `return_model=True` bundle gains `'predict': {'model': ..., 'params': ..., 'forecasts': [...]}`.
- Static plots only for v1 (animations + predict raises a clear `NotImplementedError` — forecast-in-animation is follow-up).
- [ ] Tests: `hyp.plot(data_list, predict='Kalman', t=15, show=False)` → axes contain `2*len(data_list)` line artists, forecast lines dashed + alpha, colors match source lines, forecast has 16 points (t + connector); legend unchanged (no duplicate entries — reuse the legend regression style in tests/test_animation_export.py); plotly parity test; `impute=` smoke test through plot.
- [ ] Commit `feat(plot,analyze): predict= and impute= integration`.

### Task 7: dependencies + CI

**Files:** modify `pyproject.toml`, `docs/doc_requirements.txt`.

- `[project.optional-dependencies]`: `predict = ["pykalman>=0.11", "statsmodels>=0.14", "skaters>=0.11"]`; `predict-hf = ["chronos-forecasting>=2.0"]`; add `predict` to the `dev` extra (CI exercises Kalman/ARIMA/Laplace on all platforms); `predict-hf` NOT in dev (tests importorskip; run locally).
- **yfinance is NOT a dependency anywhere** (Jeremy's call): the stock tutorial self-installs it in its first cell (`%pip install -q yfinance` guarded by an importable check). `docs/doc_requirements.txt` gains only the `[predict]` deps (gallery examples execute at doc-build; tutorials ship committed outputs via `nbsphinx_execute='never'`).
- [ ] Fresh-venv sanity: `pip install -e .[predict]` then `hyp.predict(np.random.rand(40,3), model='ARIMA', t=5)` works; base install (no extra) gives the friendly ImportError for Kalman/ARIMA/Laplace and WORKS for GP/AutoReg.
- [ ] Commit `deps: [predict] + [predict-hf] extras; yfinance for docs`.

### Task 8: gallery examples + API docs

**Files:** create `examples/plot_predict.py`, `examples/plot_impute.py`; modify `docs/api.rst`, `docs/tutorials.rst`.

- `plot_predict.py`: random-walk trio → `hyp.plot(data, predict='Kalman', t=30)` showing dashed forecast tails; text explains model options.
- `plot_impute.py`: take `weights_avg`, knock out 10% of entries + 3 full rows, show PPCA vs Kalman side by side (`ax=` panels), title the #169 all-NaN-row case.
- `api.rst`: add `hypertools.predict` and `hypertools.impute` autosummary sections (match existing sections' style).
- [ ] Both examples run standalone under Agg; doc build in Task 10 is the integration gate. Commit.

### Task 9: tutorials (executed notebooks, real data)

**Files:** create `docs/tutorials/stock_forecasting.ipynb`, `docs/tutorials/projectile_kalman.ipynb`; register in `docs/tutorials.rst`; re-run `scripts/add_colab_install_cell.py`.

- **stock_forecasting.ipynb**: FIRST CELL self-installs yfinance if missing (`import importlib.util; if importlib.util.find_spec('yfinance') is None: %pip install -q yfinance` pattern — it is not a hypertools dependency). Then yfinance download 3-5 tickers, 2y daily closes (real fetch at execution). Hold out the last 30 trading days; `hyp.predict` with ARIMA vs Kalman vs Laplace vs GP; report a real accuracy table (MAE + MAPE per model per ticker) against the held-out data — no cherry-picking, print what we get; `hyp.plot(..., predict=..., t=30)` visualization; discuss honestly that stock forecasting is hard (accuracy table will show it).
- **projectile_kalman.ipynb**: REAL projectile dataset. Primary candidate: NBA SportVU ball-tracking sample (github.com/linouk23/NBA-Player-Movements, JSON per game; ball has x,y,z at 25 Hz) — extract a shot arc. Fallbacks (verify at implementation, use the first that downloads cleanly): a physics-education video-tracked ball toss CSV. Demonstrate (a) `hyp.impute('Kalman')` filling artificially-dropped frames (incl. consecutive full rows), report RMSE vs ground truth, (b) `hyp.predict('Kalman', t=...)` forecasting the arc's remainder from the first half, plot forecast vs actual. Execute for committed outputs.
- [ ] Commit per notebook.

### Task 10: docs build + full suite + PR/notes

- [ ] `rm` new examples' `.md5`s (none yet — fresh), `MPLBACKEND=Agg PATH=".venv/bin:$PATH" make -C docs html` → succeeds; verify the 2 new gallery pages + 2 tutorial pages render (check images exist, non-trivial size).
- [ ] Full suite `MPLBACKEND=Agg .venv/bin/python -m pytest -q` (deselect the 6 kaleido tests locally) → green; push; CI 12 jobs green.
- [ ] `notes/issues-to-close-on-merge.md`: move #169 into "Bugs fixed on this branch" (both halves: hyp.predict + Kalman imputation).
- [ ] Update PR #272 body: new "predict + impute" section with evidence (test counts, tutorial links, gallery screenshots).

## Self-Review

1. **Spec coverage**: all 6 algorithms (sklearn-AR T2, kalman T2, GP T2, laplace T3, HF T4, ARIMA T3) ✓; callable directly (T4 dispatcher) ✓; plot+analyze integration like cluster/align (T6) ✓; new-timeseries-per-dataset same-dims plotting (T6) ✓; docs+tutorials (T8-9: yahoo-finance accuracy + kalman projectile) ✓; #169 imputation as own module w/ PPCA interface + sklearn (T5) ✓; issues read (only #169; 0 comments) ✓; deps handled (T7) ✓.
2. **Placeholder scan**: probed API snippets included verbatim; dataset fallback for projectile is an explicit verified-at-implementation decision point, not a TBD.
3. **Type consistency**: `Forecaster.fit_predict(data, t=)` used by dispatcher (T4) and plot (T6); `Imputer.fit_transform(data)` used by dispatcher + format_data (T5); forecast return = DataFrame w/ future_index (T1) consumed by plot traces (T6).
