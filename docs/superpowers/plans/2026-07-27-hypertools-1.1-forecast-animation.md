# HyperTools 1.1 — Forecast Animation Implementation Plan (v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `predict=` run with time-progressing animations, so a forecast grows ahead of the animated head and earlier forecasts can be retained as a fading fan — removing the ~100 lines of hand-rolled forecast machinery in the market gallery example, at parity across both backends.

**Architecture:** `plot()` refuses `predict=` for every animate mode except `'spin'` (`plot.py:2346-2354`), because a forecast is drawn once as a fixed overlay (`plot.py:4339-4341`). This plan does **not** make the forecast a lazily-recomputed per-frame artifact. The market animation is **static data revealed over time**: every observation is known before the first frame, so every forecast the animation will ever draw is also knowable before the first frame. So the plan **precomputes the whole forecast schedule at setup**, folds all of it into the display bounding box alongside the real data, and then each frame is a table lookup. That single decision resolves the review's four fatal findings at once: forecasts are inside the box *by construction* (no clamp), every frame is a pure function of its index (idempotent under `ani.save()`/`to_jshtml()` replay), the per-frame cost collapses to a memoized fit per distinct revealed-history length, and plotly reaches parity by consuming the same precomputed table its trail traces already consume.

**Tech Stack:** Python 3.10+, numpy, scipy (PCHIP), matplotlib, plotly, `hypertools.predict` (Kalman / ARIMA / GP / Laplace / Chronos), pytest.

---

## Revision note (v2)

v1 of this plan was adversarially reviewed (`notes/audit/review_plan3_forecast_animation.md`). **Task 1 was verified sound; Task 2 had eight independent defects, four fatal; Task 3's rendering model contradicted its own tests.** Every finding was re-checked against the source before this rewrite, and two design decisions were superseded by the maintainer. Verified reality:

| v1 claim | verified reality |
|-|-|
| Task 2 Step 6: `tests/plot/test_predict_integration.py` "all pass, unchanged" | `test_predict_integration.py:169-178` parametrizes over `True/'parallel'/'serial'/'window'/'morph'` asserting `NotImplementedError`. Narrowing the refusal breaks **4 of 5**. The file must be **edited** (Task 3), not asserted-unchanged. |
| Task 2 Step 6 runs `tests/test_predict.py` | That file does not exist (`ls` → `No such file or directory`); pytest exits with a collection ERROR. Dropped from every command. |
| Removing the refusal is enough to draw a per-frame forecast | `plot.py:4339-4341` draws the **static full-history overlay** with no `animate` guard, inside the shared matplotlib branch after `_draw()`. Measured on `animate='spin'`, n=3: `ax.lines` = 3 solid `_childN` + 3 dashed `_nolegend_` alpha=0.6, **901 vertices each**, and the setup overlays land **first**. Task 3 gates line 4339 on `animate in (False, None, 'spin')`. |
| "Use `FrameContext.revealed_counts` for the history slice" | Animation-core defines `revealed_counts: Optional[List[int]] = None` and documents *"``None`` for parallel animations"* — the mode every test uses. Replaced by `_anim_window_bounds` (`matplotlib_backend.py:319-366`), which **is** the parallel reveal and is what `update_lines_parallel` itself calls (`matplotlib_backend.py:1185`). |
| Prerequisite is animation-core Task 7 alone | `order=` is animation-core **Task 5**. Verified today: `'order' in inspect.signature(hyp.plot).parameters` → `False`; `'on_frame'` → `False`. Both tasks are prerequisites. |
| Forecasts "pass through the same center/scale transform as the data … so they cannot render outside the cube" | `plot.py:4015-4032` runs **once**, before any figure exists, and `_mean`/`_m1`/`_m2` are function-locals. Animated limits are hard-set to `[-1, 1]` before `FuncAnimation` is built (`matplotlib_backend.py:1785`, `1888-1890`; measured `ax.get_xlim3d()` → `(-1.0, 1.0)`). Measured: **1 of 7** partial-history Kalman forecasts fell outside the cube. Fixed by precomputing the schedule and including it in the joint statistics — see the maintainer correction below. |
| `t=1` draws exactly 2 vertices | Measured: `t=1` draws **900** vertices at `antialias=True` (the default), 2 at `antialias=False`. `plot.py:1904-1908` documents this as contract (*"Forecast overlays drawn by `predict=` are smoothed the same way"*), `_draw_forecast_overlays` applies `_interp_static_line` (`plot.py:149-150`), and `test_predict_with_spin_renders_dashed_forecast_overlay` pins `len(fc.get_xdata()) > t + 1` (`test_predict_integration.py:198`). The v1 test is replaced by an antialias-aware pair. |
| Forecasting "from the history revealed so far, in the already-reduced plotting space" (array unspecified) | There are **three** distinct arrays. `plot.py:3907-3925` resamples every animated line dataset onto **exactly `round(frame_rate*duration)` rows** (`_interp_anim_line`, `plot.py:277-299`), then `matplotlib_backend._aa_window` densifies that onto ~900 drawn vertices. Measured on 60 raw rows at `duration=2, frame_rate=4`: analyze 60 rows → frame grid 8 rows → drawn 904 vertices (the review's "15.1x"). Contract 1 below pins `t` to RAW analyze-space samples. |
| `if predict is not None and animate == "morph":` refuses morph | `_resolve_animate_mode` is called at `plot.py:3653`, ~1300 lines **after** the refusal at `plot.py:2338`, so at the check `animate` is still the raw list and `animate == "morph"` is `False`. Verified: `animate=['morph','morph'] + predict='Kalman'` raises **today** only via the truthiness of a non-empty list. Task 3 guards both forms and tests the list form. |
| Plotly "just works" once the refusal is removed | `plotly_backend.py:901-931` draws forecasts unconditionally and statically; `_add_animation` (`plotly_backend.py:2517-2529`) takes no forecast argument and its frame updates address only `trace_indices` (data) + `trail_trace_start..+n_trail_traces` (`plotly_backend.py:2896-2897`). A newly-accepted call would render a **frozen** full-history overlay. Task 6 brings plotly to real parity. |
| `forecast_from_history` "Returns … numpy.ndarray" | `hyp.predict` returns a **`pandas.DataFrame`** (verified: `type(_predict(ramp, model='Kalman', t=3))` → `<class 'pandas.DataFrame'>`). Documented and tested in Task 1. |
| Task 3 preallocates artists whose `trail_alpha` floor is `0.08`, and `test_trail_accumulates_past_forecasts` counts `get_alpha() > 0` | Contradictory: every preallocated artist is already `alpha=0.08 > 0` at frame 0, so `late > early` can never hold. Fixed by an explicit unwritten state (`set_visible(False)` + empty data). |
| `assert max(alphas) == alphas[0] or max(alphas) >= sorted(alphas)[-1]` | `sorted(alphas)[-1]` **is** `max(alphas)`, so the disjunct is `max >= max` → always `True`. Replaced by a role-tagged comparison. |
| `test_trail_is_capped_by_an_integer` drives one frame | A single `_func(20)` call leaves at most one entry in a per-frame ring buffer, so the cap assertion holds vacuously. All trail tests now drive frames **sequentially**. |
| `_dashed(ax)` identifies forecasts | Any user data drawn with `'--'`/`':'` is misclassified. `_draw_forecast_overlays` already sets `label='_nolegend_'` + `alpha=0.6` (`plot.py:156`), but trail artists share `_nolegend_`. Contract 5 tags forecast artists explicitly and Task 4 tests the false-positive case. |
| Task 2 Step 5 "Expected: 13 passed" | Task 1 created 5 + Task 2 appended 9 = 14. All pass counts in v2 are itemised per module. |

### Maintainer corrections applied (supersede the v1 design)

1. **`t` is in RAW analyze-space samples** — confirmed, kept. Consequence handled explicitly in Contract 2.
2. **No clamping for static data.** The blanket "clamp animated forecasts to the axes limits" decision is **withdrawn**. There are two cases:
   - **CASE A — static data (this plan).** All observations known up front, merely *revealed* over time. Compute every forecast in advance; compute the bounding box from real data **and** every forecast together, so every forecast is inside the box by construction. **No clamp.**
   - **CASE B — streaming data (out of scope; already shipped).** `hypertools/io/streaming.py:382-401` freezes the box from the head samples and clamps: `t = 2.0*((pts - head_mu) - box_m1)/box_m2 - 1.0; return np.clip(t, -1.0, 1.0)`, with a `RuntimeWarning` when >25% of post-head samples are clamped (`streaming.py:498-508`). Clamping stays there. This plan neither adds nor changes a clamp.
3. **Plotly must be identical to matplotlib.** The "gate plotly with `NotImplementedError`" instruction is **withdrawn**. Task 6 delivers real parity, reusing the mechanism plotly already uses for chemtrails/precog/bullettime: separate traces at a fixed alpha whose row-window data is rewritten each frame (`plotly_backend.py:945-975` creates them at `_to_plotly_color(color, 0.3)`; `plotly_backend.py:2929-2960` rewrites them per frame).

### Finding requested by the maintainer: is the reduction fit on data + forecasts?

**No — and no defect exists.** Verified reading order in `plot()`:

- `plot.py:2803-2823` — `xform = analyze(raw, ..., reduce=reduce, align=align, ...)`: the reduction is fit on the **real data only**.
- `plot.py:2913` — `xform = reducer(xform, ndims=_display_ndims, reduce=_display_reduce, ...)`: the display-ndims pass, still data only.
- `plot.py:2979-2988` — `_fc = _predictor(xform, model=predict, t=t)`: the forecast is produced **inside the already-reduced space**, so it never enters the reducer at all. The code comment states it: *"forecast `t` new rows per input dataset, in the plotted (post normalize->reduce->align) space (GH #169)"* (`plot.py:2963-2964`), and the public contract repeats it: *"one forecast array per input dataset, in the analyzed/plotted -- pre-center/scale -- space"* (`plot.py:1935-1937`).

This is stronger than the requested "fit on data, `transform` the forecasts": the forecasts are never given to the reducer in either direction. It is shipped 1.0 behaviour pinned by `test_predict_return_model_bundle` (`test_predict_integration.py:82-100`), which asserts the bundle matches `hyp.predict(xform_data, ...)`. **No change is made here.** The one thing that *was* wrong under the CASE A spec — the bounding box not containing the animation's forecasts — is fixed in Task 4 Step 4.

---

## Global Constraints

- Target release: **1.1**. Nothing ships until the whole 1.1 line works.
- Run everything with the repo venv: `.venv/bin/python -m pytest`, from the repo root. (The base anaconda python is broken.)
- **Never simplify a test to make it pass.** Fix the code.
- **No mock objects.** Tests call real `hyp.plot`/`hyp.predict` and assert on real artists and real plotly traces. Where a call *count* is needed, the schedule exposes a real `n_fits` attribute rather than a spy.
- Force `matplotlib.use("Agg")` in every matplotlib test module. Import `pytest` only where used.
- **Forecast scoring stays OUT of the library** (maintainer decision): accuracy/backtest logic belongs in the tutorial as legitimately custom code. This plan renders forecasts; it never scores them.
- Additive only. `predict=` with `animate=False` and with `animate='spin'` keep their current behaviour **exactly**, including the static overlay path, its `alpha=0.6`/`--`/`_nolegend_` styling, its antialiasing, and its `set_clip_on(False)`.
- **No clamping is introduced** (see maintainer correction 2). If a forecast ever renders outside the cube, the bounding box is wrong — fix the box, not the drawing.
- **Both backends, same behaviour.** Any task that changes what matplotlib draws must land plotly in the same task or in Task 6, with a cross-backend test.
- Every task that touches central dispatch (Tasks 3-7) runs the **whole** suite: `.venv/bin/python -m pytest -q`. Baseline today: **2564 collected / 2 deselected**; `tests/plot/test_predict_integration.py` → **18 passed in 3.30s**.
- Update docstrings in the same commit as the behaviour.
- Branch off `dev-1.0`; never commit to `master`.

---

## Prerequisites

Both from `docs/superpowers/plans/2026-07-26-hypertools-1.1-animation-core.md`:

- **Task 5 — `order='parallel'|'serial'`.** Needed because `test_forecast_composes_with_serial_order` (Task 4 Step 1 here) passes `order='serial'`. Verified today: `'order' in inspect.signature(hyp.plot).parameters` → `False`, so that test would die with `TypeError: plot() got an unexpected keyword argument 'order'` without it.
- **Task 7 — the per-frame hook.** This plan needs the *internal* half of it: a single registration point so there is exactly one per-frame dispatch, not two. Animation-core Task 7 ships this as `FrameHooks` -- `.callbacks` (list), `.record(**state)`, `.dispatch(figure, axes)` -- created in `plot()`, closed over by `_draw`, and adopted by `HyperAnimation.__new__`. This plan appends its internal updater to `hooks.callbacks` rather than introducing a second registration function. **Cross-plan interface note:** this plan does **not** consume `FrameContext.revealed_counts` (documented `None` for parallel animations), so animation-core needs no change on that account.

Task 1 is fully standalone. Task 2 is standalone **except** for `ForecastSchedule.for_serial`, which calls `serial_reveal_counts` — defined in animation-core Task 7. Verified by running Task 1's and Task 2's modules against today's `dev-1.0` while writing this plan: **Task 1 → 8 passed**; **Task 2 → 12 passed, 1 failed**, the failure being exactly `ImportError: cannot import name 'serial_reveal_counts'`. Implement Tasks 1-2 first regardless; that one test goes green when the prerequisite lands.

---

## Contracts this plan establishes

1. **`t` is in RAW analyze-space samples.** `t=1` means "one more observation", matching the shipped docstring (*"Forecast horizon in steps"*) and what a user means by "forecast the next day". It is **not** in animation frames and **not** in drawn vertices.

2. **Three coordinate spaces, named, with one mapping between them.**
   - **analyze space** — `xform` after `normalize→reduce→align`, before the animation resample and before centre/scale (`plot.py:2979`, lengths captured at `plot.py:2998` as `pre_interp_lengths`). Forecasts are computed here.
   - **frame grid** — `_interp_anim_line(xi, round(frame_rate*duration))` (`plot.py:3907-3925`); exactly one row per animation frame, endpoints exact.
   - **display box** — the centred/rescaled `[-1, 1]` cube produced at `plot.py:4015-4032`.

   Because the forecast is anchored on the last revealed **raw** sample while the drawn head sits on the **frame grid**, the forecast joins the trajectory to within **at most one raw sample**, not exactly. This is deliberate (it is what keeps every forecast inside the box by construction) and it is what `test_forecast_is_anchored_near_the_drawn_head` measures.

3. **`ForecastSchedule` — every forecast is computed before the first frame is drawn.** Static data means the whole schedule is knowable at setup. Each frame is a table lookup. Fits are memoized by revealed-history length, so a 900-frame animation of a 60-row dataset costs ≤ 59 fits, not 900.

4. **The display box contains every drawn forecast by construction; nothing is clamped.** `plot.py:4015-4031` already stacks the full-history forecast into the joint statistics; Task 4 stacks the *entire schedule* in as well. Since `_rescale(a) = 2*(a - _m1)/_m2 - 1` with `_m1 = min(_joint)`, `_m2 = max(_joint - _m1)`, every element of `_joint` lands in `[-1, 1]`. PCHIP densification is monotone-preserving, so drawn vertices do not overshoot their control points either — the shipped static test `test_forecast_vertices_stay_inside_frame` (`test_predict_integration.py:143-164`) already relies on exactly this.

5. **Forecast artists identify themselves.** Every forecast artist carries `_hyp_forecast_role ∈ {'static', 'live', 'trail'}` (matplotlib) / `trace.meta = {'hyp_forecast_role': ...}` (plotly), plus `_hyp_forecast_age` on trail artists. Linestyle is **not** a discriminator: user data drawn with `fmt='--'` is dashed too.

6. **Every frame is a pure function of its index.** No accumulating ring buffer, no state mutated by drawing. `_func(12)` renders identically whether or not `_func(2)` ran first — required because `ani.save()` and `to_jshtml()` replay from frame 0 and the tests drive frames out of order.

7. **`return_model=True`'s `predict.forecasts` is unchanged**: the full-history forecast, exactly `t` rows, analyze space, one per input dataset (`plot.py:1935-1941`). For a time-progressing animation this is *also* the forecast drawn at the **final** frame, because at the last frame the revealed history **is** the full history (`_anim_window_bounds(total-1, total, n, w)` → `end = n`). One sentence of the docstring is amended; the value is not.

8. **Backend parity.** matplotlib and plotly consume the same `ForecastSchedule` and draw the same polylines, asserted by a cross-backend test at the final frame.

---

## File Structure

| file | responsibility | change |
|-|-|-|
| `hypertools/plot/forecast.py` | **new** — `forecast_from_history`, `revealed_raw_counts`, `DisplayTransform`, `ForecastSchedule` | create |
| `hypertools/plot/plot.py` | narrow the refusal; gate the static overlay; build the schedule; fold it into the bounding box; `forecast_trail=`; docstrings | modify |
| `hypertools/plot/matplotlib_backend.py` | live/trail forecast artists, updated per frame from the schedule | modify |
| `hypertools/plot/plotly_backend.py` | forecast traces addressable per frame; forecast trail traces | modify |
| `tests/plot/test_forecast_core.py` | the pure forecast helper | create |
| `tests/plot/test_forecast_schedule.py` | reveal mapping, memoization, purity, display transform | create |
| `tests/plot/test_predict_animation.py` | `predict=` with time-progressing animations; bundle contract | create |
| `tests/plot/test_forecast_trail.py` | the retained forecast fan | create |
| `tests/plot/test_forecast_animation_plotly.py` | cross-backend parity | create |
| `tests/plot/test_predict_integration.py` | **edit** the `NotImplementedError` parametrize (Task 3) | modify |
| `CHANGELOG.md`, `docs/` | user-facing documentation | modify |

---

## Task 1: The pure forecast helper

Before touching `plot()`, build and test the pure function that answers "given the history revealed at frame N, what is the forecast, expressed as a displacement from the last revealed observation?". Keeping it pure makes the anchoring arithmetic testable without rendering anything.

The market example got this wrong in a way worth encoding as a test: `hyp.predict(hist, t=H)` returns `H` rows that are **all future steps**, so `f[0]` is the first forecast step and *not* the last observation. Anchoring with `f - f[0]` silently discards a whole step. The review verified this on a unit ramp for Kalman, ARIMA and GP (`first_row[0] = 30.0 = last_obs + 1`).

**Files:**
- Create: `hypertools/plot/forecast.py`
- Test: `tests/plot/test_forecast_core.py`

**Interfaces:**
- Produces `forecast_from_history(history, model, t, min_history=2)` → `np.ndarray` of shape `(t + 1, n_dims)`, a displacement path whose **first row is all zeros** (the anchor itself) so callers can add it to a position directly. Returns `None` when there is too little history.

- [ ] **Step 1: Write the failing test**

```python
# tests/plot/test_forecast_core.py
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
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/python -m pytest tests/plot/test_forecast_core.py -v`
Expected: collection FAILS with `ModuleNotFoundError: No module named 'hypertools.plot.forecast'`.

- [ ] **Step 3: Implement the pure function**

```python
# hypertools/plot/forecast.py
#!/usr/bin/env python
"""Forecast scheduling for animated `hyp.plot(..., predict=...)` calls.

A forecast used to be a fixed overlay drawn once, so `predict=` refused every
animate mode that reveals data over time. Animating STATIC data (every
observation known up front, merely revealed frame by frame) means every
forecast the animation will ever draw is knowable up front too -- so this
module computes the whole schedule BEFORE drawing, folds it into the display
bounding box, and lets each frame be a table lookup.

Three spaces, never conflated (see the plan's Contract 2):

- ANALYZE space  -- `xform` post normalize/reduce/align, pre-resample,
                    pre-centre/scale. Forecasts are computed HERE, and `t` is
                    measured in these RAW samples.
- FRAME GRID     -- `plot._interp_anim_line` resamples every animated line
                    dataset to exactly `round(frame_rate * duration)` rows.
- DISPLAY box    -- the centred/rescaled [-1, 1] cube (`plot.py:4015-4032`).
"""

import numpy as np

from ..predict.predict import predict as _predict

#: Fewest observations we will fit a forecaster to.
DEFAULT_MIN_HISTORY = 2


def forecast_from_history(history, model, t, min_history=DEFAULT_MIN_HISTORY):
    """Forecast `t` steps on from `history`, as a displacement path.

    Parameters
    ----------
    history : array-like, shape (n_observed, n_dims)
        The trajectory revealed so far, in ANALYZE space (already reduced,
        not yet resampled onto the frame grid and not yet centred/scaled).
    model : str or dict
        Anything `hypertools.predict` accepts ('Kalman', 'ARIMA', 'Laplace',
        'GaussianProcess', 'Chronos', ...).
    t : int
        Forecast horizon, in RAW analyze-space steps. ``t=1`` is the next
        observation.
    min_history : int, default 2
        Refuse to forecast from fewer rows than this.

    Returns
    -------
    numpy.ndarray or None
        Shape ``(t + 1, n_dims)``, dtype float64. Row 0 is all zeros (the
        anchor itself), so ``history[-1] + result`` is the forecast path in
        analyze space. ``None`` when `history` is shorter than `min_history`
        -- callers must hide the artist rather than draw an empty trace.

    Notes
    -----
    `hypertools.predict` returns a ``pandas.DataFrame``; its index is
    deliberately discarded here (a forecast's index is a continuation the
    plotting code has no use for).
    """
    history = np.asarray(history, dtype=float)
    if history.ndim != 2:
        raise ValueError(
            f"history must be 2-D (n_observed, n_dims); got shape "
            f"{history.shape}.")
    if len(history) < max(2, min_history):
        return None

    forecast = np.asarray(_predict(history, model=model, t=t), dtype=float)
    # `predict` returns exactly `t` NEW rows -- every one of them a future
    # step -- so the last OBSERVED row is the anchor. Using forecast[0] as the
    # anchor would throw away a whole step and force the first displacement
    # to zero (the bug the market gallery example shipped with).
    displacement = forecast - history[-1]
    return np.vstack([np.zeros((1, history.shape[1])), displacement])
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `.venv/bin/python -m pytest tests/plot/test_forecast_core.py -v`
Expected: **8 passed** (6 named tests, one of them parametrized over 2 models), or 7 passed + 1 skipped if `statsmodels` is absent.

- [ ] **Step 5: Commit**

```bash
git add hypertools/plot/forecast.py tests/plot/test_forecast_core.py
git commit -m "feat(plot): forecast_from_history, anchored on the last observation"
```

---

## Task 2: `ForecastSchedule` — precompute every forecast before drawing

The heart of the plan. Static data ⇒ the whole schedule is knowable at setup, which is what makes the bounding box honest, the frames idempotent, and the cost bounded.

**Files:**
- Modify: `hypertools/plot/forecast.py`
- Test: `tests/plot/test_forecast_schedule.py`

**Interfaces:**
- `revealed_raw_counts(n_raw, n_grid, num, total_frames)` → int: RAW analyze-space rows revealed at frame `num` of a parallel/window animation. Delegates to `matplotlib_backend._anim_window_bounds` — the single library implementation, which `update_lines_parallel` itself calls (`matplotlib_backend.py:1185`). `end` does **not** depend on `window_frames` (`matplotlib_backend.py:357-358`: `end = int(np.ceil((num + 1) * n_points / total)); end = max(1, min(n_points, end))`), so `0` is passed for it.
- `DisplayTransform(mean, offset, scale)` with `__call__(a)` reproducing `plot.py:4018-4031` exactly.
- `ForecastSchedule.for_parallel(histories, grid_lengths, model, t, n_frames, min_history=2)` and `.for_serial(...)`; `.path(dataset, frame)` → analyze-space `(t+1, d)` or `None`; `.stacked_paths()` → one array of every forecast vertex, for the bounding box; `.to_display(transform)` → the same table in display coordinates; `.n_fits` → int.

- [ ] **Step 1: Write the failing test**

```python
# tests/plot/test_forecast_schedule.py
"""The precomputed forecast schedule: reveal mapping, memoization, purity."""

import numpy as np
import pytest

from hypertools.plot.forecast import (DisplayTransform, ForecastSchedule,
                                      forecast_from_history,
                                      revealed_raw_counts)

N_RAW, N_GRID, N_FRAMES = 60, 8, 8


def _history(n=N_RAW, d=3, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d)).cumsum(axis=0)


# --- the reveal mapping ----------------------------------------------------

def test_revealed_raw_counts_is_monotonic():
    counts = [revealed_raw_counts(N_RAW, N_GRID, f, N_FRAMES)
              for f in range(N_FRAMES)]
    assert counts == sorted(counts), counts


def test_revealed_raw_counts_never_exceeds_the_raw_row_count():
    for f in range(N_FRAMES):
        assert 0 <= revealed_raw_counts(N_RAW, N_GRID, f, N_FRAMES) <= N_RAW


def test_the_last_frame_reveals_the_whole_history():
    """Contract 7 depends on this: the final-frame forecast IS the
    full-history forecast, so `return_model`'s bundle stays truthful."""
    assert revealed_raw_counts(N_RAW, N_GRID, N_FRAMES - 1, N_FRAMES) == N_RAW


def test_reveal_matches_the_library_formula_not_a_second_copy_of_it():
    from hypertools.plot.matplotlib_backend import _anim_window_bounds
    for f in range(N_FRAMES):
        _, end, _ = _anim_window_bounds(f, N_FRAMES, N_GRID, 0)
        pos = (end - 1) * (N_RAW - 1) / (N_GRID - 1)
        assert revealed_raw_counts(N_RAW, N_GRID, f, N_FRAMES) == int(pos) + 1


# --- the schedule ----------------------------------------------------------

def test_schedule_has_one_entry_per_frame_per_dataset():
    sched = ForecastSchedule.for_parallel(
        [_history(seed=s) for s in range(2)], [N_GRID] * 2,
        model='Kalman', t=3, n_frames=N_FRAMES)
    assert sched.n_datasets == 2 and sched.n_frames == N_FRAMES
    for i in range(2):
        for f in range(N_FRAMES):
            p = sched.path(i, f)
            assert p is None or p.shape == (4, 3)


def test_early_frames_have_no_forecast():
    """Frame 0 reveals 1 raw row; min_history=2 refuses to fit it."""
    sched = ForecastSchedule.for_parallel(
        [_history()], [N_GRID], model='Kalman', t=3, n_frames=N_FRAMES)
    assert sched.path(0, 0) is None


def test_final_frame_forecast_equals_the_full_history_forecast():
    hist = _history()
    sched = ForecastSchedule.for_parallel(
        [hist], [N_GRID], model='Kalman', t=3, n_frames=N_FRAMES)
    direct = forecast_from_history(hist, 'Kalman', t=3)
    assert np.allclose(sched.path(0, N_FRAMES - 1), direct)


def test_fits_are_memoized_by_revealed_history_length():
    """A 900-frame animation of a 60-row dataset can only have <= 60 distinct
    revealed lengths, so it must cost <= 60 fits, not 900. Measured cost of a
    single 60-row Kalman fit: 54 ms -- 900 fits would be 48s PER DATASET."""
    sched = ForecastSchedule.for_parallel(
        [_history(seed=s) for s in range(3)], [900] * 3,
        model='Kalman', t=3, n_frames=900)
    assert sched.n_fits <= 3 * N_RAW
    assert sched.n_fits < 900 * 3 / 4, sched.n_fits


def test_the_schedule_is_a_pure_lookup_so_frames_are_idempotent():
    sched = ForecastSchedule.for_parallel(
        [_history()], [N_GRID], model='Kalman', t=3, n_frames=N_FRAMES)
    forward = [sched.path(0, f) for f in range(N_FRAMES)]
    backward = [sched.path(0, f) for f in reversed(range(N_FRAMES))]
    for a, b in zip(forward, reversed(backward)):
        assert (a is None and b is None) or np.allclose(a, b)


def test_stacked_paths_covers_every_forecast_vertex():
    """Task 4 builds the bounding box from this, so a vertex it misses is a
    forecast that could render outside the cube."""
    sched = ForecastSchedule.for_parallel(
        [_history(seed=s) for s in range(2)], [N_GRID] * 2,
        model='Kalman', t=3, n_frames=N_FRAMES)
    stacked = sched.stacked_paths()
    assert stacked.ndim == 2 and stacked.shape[1] == 3
    for i in range(2):
        for f in range(N_FRAMES):
            drawn = sched.polyline(i, f)
            if drawn is None:
                continue
            for row in drawn:
                nearest = np.abs(stacked - row).sum(axis=1).min()
                assert np.isclose(nearest, 0.0), (
                    f'dataset {i} frame {f} vertex {row} is not in '
                    'stacked_paths(), so the bounding box would not hold it')


def test_serial_schedule_reveals_datasets_in_order():
    hists = [_history(seed=s) for s in range(3)]
    sched = ForecastSchedule.for_serial(hists, [N_GRID] * 3, model='Kalman',
                                        t=3, n_frames=16)
    early = [sched.revealed(i, 1) for i in range(3)]
    assert early[0] >= early[1] >= early[2]
    assert [sched.revealed(i, 15) for i in range(3)] == [N_RAW] * 3


# --- the display transform -------------------------------------------------

def test_display_transform_reproduces_plot_s_centre_scale_arithmetic():
    """Mirrors plot.py:4018-4031 exactly, on the same inputs."""
    rng = np.random.default_rng(4)
    data = rng.normal(size=(40, 3))
    mean = data.mean(axis=0)
    centred = data - mean
    m1 = centred.min()
    m2 = (centred - m1).max() or 1.0
    expected = 2 * ((centred - m1) / m2) - 1
    got = DisplayTransform(mean, m1, m2)(data)
    assert np.allclose(got, expected)
    assert got.min() >= -1.0 - 1e-12 and got.max() <= 1.0 + 1e-12


def test_to_display_maps_every_scheduled_forecast_into_the_cube():
    """Contract 4: no clamping is needed because the box was built to hold
    them. Build the transform from data + schedule, exactly as Task 4 does."""
    hists = [_history(seed=s) for s in range(2)]
    sched = ForecastSchedule.for_parallel(hists, [N_GRID] * 2, model='Kalman',
                                          t=5, n_frames=N_FRAMES)
    joint = np.vstack([np.vstack(hists), sched.stacked_paths()])
    mean = joint.mean(axis=0)
    joint_c = joint - mean
    m1 = joint_c.min()
    m2 = (joint_c - m1).max() or 1.0
    disp = sched.to_display(DisplayTransform(mean, m1, m2))
    for i in range(2):
        for f in range(N_FRAMES):
            p = disp.path(i, f)
            if p is None:
                continue
            assert p.min() >= -1.0 - 1e-9 and p.max() <= 1.0 + 1e-9
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/python -m pytest tests/plot/test_forecast_schedule.py -v`
Expected: collection FAILS with `ImportError: cannot import name 'DisplayTransform' from 'hypertools.plot.forecast'`.

- [ ] **Step 3: Implement the schedule**

Append to `hypertools/plot/forecast.py`:

```python
def revealed_raw_counts(n_raw, n_grid, num, total_frames):
    """RAW analyze-space rows revealed at frame `num` (parallel/window).

    `update_lines_parallel` reveals `data[start:end]` of the FRAME-GRID array,
    where `end` comes from `matplotlib_backend._anim_window_bounds` -- the one
    implementation of the reveal, called from `matplotlib_backend.py:1185`. It
    is reused here rather than re-derived (`FrameContext.revealed_counts` is
    documented ``None`` for parallel animations, so it cannot serve). `end`
    does not depend on the trail window, so 0 is passed for it.

    `plot._interp_anim_line` puts frame-grid row ``j`` at RAW parameter
    position ``j * (n_raw - 1) / (n_grid - 1)`` with exact endpoints, so the
    last raw sample at or before the drawn head (grid row ``end - 1``) is
    index ``floor(pos)`` and ``floor(pos) + 1`` rows are revealed.
    """
    from .matplotlib_backend import _anim_window_bounds
    n_raw = int(n_raw)
    n_grid = int(n_grid)
    if n_grid < 2 or n_raw < 2:
        return n_raw
    _, end, _ = _anim_window_bounds(num, total_frames, n_grid, 0)
    pos = (end - 1) * (n_raw - 1) / (n_grid - 1)
    return min(n_raw, int(np.floor(pos)) + 1)


class DisplayTransform:
    """The centre/scale affine `plot()` applies at `plot.py:4018-4031`.

    ``2 * (((a - mean) - offset) / scale) - 1``. Recorded at setup so a
    forecast computed in ANALYZE space can be mapped into the SAME display
    box the data was mapped into -- rather than being recomputed from
    function-locals that no longer exist by frame time.
    """

    __slots__ = ('mean', 'offset', 'scale')

    def __init__(self, mean, offset, scale):
        self.mean = np.asarray(mean, dtype=float)
        self.offset = float(offset)
        self.scale = float(scale) or 1.0

    def __call__(self, a):
        centred = np.asarray(a, dtype=float) - self.mean
        return 2.0 * ((centred - self.offset) / self.scale) - 1.0


class ForecastSchedule:
    """Every forecast an animation will ever draw, computed before drawing.

    Built from STATIC data (all observations known up front, revealed frame
    by frame), which is what makes precomputation possible and what lets the
    display bounding box be built to contain every forecast -- so nothing is
    ever clamped. Streaming data uses a different rule entirely; see
    `hypertools/io/streaming.py:382-401`.

    `counts[f][i]` is the number of RAW analyze-space rows dataset `i` has
    revealed at frame `f`. Fits are memoized on `(i, count)`, so a 900-frame
    animation of a 60-row dataset costs at most 59 fits.
    """

    def __init__(self, histories, counts, model, t,
                 min_history=DEFAULT_MIN_HISTORY, transform=None):
        self.histories = [np.asarray(h, dtype=float) for h in histories]
        self.counts = [list(row) for row in counts]
        self.model = model
        self.t = int(t)
        self.min_history = int(min_history)
        self.transform = transform
        self.n_frames = len(self.counts)
        self.n_datasets = len(self.histories)
        self.n_fits = 0
        self._paths = {}
        for frame_counts in self.counts:
            for i, k in enumerate(frame_counts):
                if (i, k) in self._paths:
                    continue
                hist = self.histories[i][:k]
                path = forecast_from_history(hist, self.model, self.t,
                                             min_history=self.min_history)
                if path is not None:
                    self.n_fits += 1
                self._paths[(i, k)] = path

    # -- construction ------------------------------------------------------
    @classmethod
    def for_parallel(cls, histories, grid_lengths, model, t, n_frames,
                     min_history=DEFAULT_MIN_HISTORY):
        counts = [[revealed_raw_counts(len(h), g, f, n_frames)
                   for h, g in zip(histories, grid_lengths)]
                  for f in range(n_frames)]
        return cls(histories, counts, model, t, min_history=min_history)

    @classmethod
    def for_serial(cls, histories, grid_lengths, model, t, n_frames,
                   min_history=DEFAULT_MIN_HISTORY):
        """Serial reveals one dataset at a time, so its schedule comes from
        the backend's own `serial_reveal_counts` (animation-core Task 7),
        mapped from frame-grid rows onto raw rows dataset by dataset."""
        from .matplotlib_backend import serial_reveal_counts
        counts = []
        for f in range(n_frames):
            grid_counts = serial_reveal_counts(list(grid_lengths), f, n_frames)
            row = []
            for h, g, shown in zip(histories, grid_lengths, grid_counts):
                n_raw = len(h)
                if g < 2 or n_raw < 2 or shown <= 0:
                    row.append(min(n_raw, max(0, shown)))
                else:
                    pos = (min(shown, g) - 1) * (n_raw - 1) / (g - 1)
                    row.append(min(n_raw, int(np.floor(pos)) + 1))
            counts.append(row)
        return cls(histories, counts, model, t, min_history=min_history)

    # -- lookups -----------------------------------------------------------
    def revealed(self, dataset, frame):
        return self.counts[min(frame, self.n_frames - 1)][dataset]

    def anchor(self, dataset, frame):
        """The last revealed observation, in this schedule's coordinates."""
        k = self.revealed(dataset, frame)
        if k < 1:
            return None
        return self.histories[dataset][k - 1]

    def path(self, dataset, frame):
        """Displacement path (t + 1, d) for `dataset` at `frame`, or None."""
        return self._paths[(dataset, self.revealed(dataset, frame))]

    def polyline(self, dataset, frame):
        """The DRAWN forecast: anchor + displacement, or None."""
        path = self.path(dataset, frame)
        if path is None:
            return None
        return self.anchor(dataset, frame) + path

    def stacked_paths(self):
        """Every forecast vertex this schedule will ever draw, stacked.

        This is what Task 4 folds into the joint centre/scale statistics so
        the display box contains all of it by construction.
        """
        rows = []
        for (i, k), path in self._paths.items():
            if path is None or k < 1:
                continue
            rows.append(self.histories[i][k - 1] + path)
        if not rows:
            return np.zeros((0, self.histories[0].shape[1]))
        return np.vstack(rows)

    def to_display(self, transform):
        """A copy of this schedule with every history mapped through
        `transform`, so `polyline()` returns display-box coordinates."""
        out = object.__new__(ForecastSchedule)
        out.histories = [transform(h) for h in self.histories]
        out.counts = self.counts
        out.model, out.t = self.model, self.t
        out.min_history = self.min_history
        out.transform = transform
        out.n_frames, out.n_datasets = self.n_frames, self.n_datasets
        out.n_fits = 0            # no refitting: displacements are affine-mapped
        # a displacement is a DIFFERENCE of positions, so the mean cancels and
        # only the scale survives: d_display = 2 * d_analyze / scale
        out._paths = {key: (None if p is None else 2.0 * p / transform.scale)
                      for key, p in self._paths.items()}
        return out
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `.venv/bin/python -m pytest tests/plot/test_forecast_schedule.py -v`
Expected: **13 passed** — *once animation-core Task 7 has landed*. This exact module and implementation were run against today's `dev-1.0` while writing the plan: **12 passed, 1 failed**, the single failure being `test_serial_schedule_reveals_datasets_in_order` with `ImportError: cannot import name 'serial_reveal_counts' from 'hypertools.plot.matplotlib_backend'` — i.e. the prerequisite, not a defect. Measured module runtime **8.6s**, including `test_fits_are_memoized_by_revealed_history_length`, which builds a 900-frame schedule over 3 datasets (177 real Kalman fits at ~54 ms each).

- [ ] **Step 5: Commit**

```bash
git add hypertools/plot/forecast.py tests/plot/test_forecast_schedule.py
git commit -m "feat(plot): precomputed ForecastSchedule + display transform"
```

---

## Task 3: Narrow the refusal and stop drawing the static overlay over an animation

This task changes what `plot()` accepts and what it draws, but does **not** yet draw a per-frame forecast. The intermediate state is precise and testable: a time-progressing `predict=` call is accepted and draws **no** forecast at all. That isolates the review's C2 finding (the un-gated static overlay) from Task 4's new rendering, so a regression in either is attributable.

**Files:**
- Modify: `hypertools/plot/plot.py:2338-2354` (the refusal), `plot.py:4339-4350` (the static overlay), `plot.py:122-165` (`_draw_forecast_overlays`, role tag only)
- Modify: `tests/plot/test_predict_integration.py:167-178`
- Test: `tests/plot/test_predict_animation.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/plot/test_predict_animation.py
"""`predict=` with time-progressing animations (matplotlib backend)."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

import hypertools as hyp


def _series(n=3, rows=60, dims=4, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, dims)).cumsum(axis=0) for _ in range(n)]


def _ax(fig):
    return [a for a in fig.axes if hasattr(a, 'zaxis')][0]


def _forecasts(ax, role=None):
    """Forecast artists identify THEMSELVES (Contract 5). Linestyle is not a
    discriminator: user data drawn with fmt='--' is dashed too."""
    out = [ln for ln in ax.lines
           if getattr(ln, '_hyp_forecast_role', None) is not None]
    if role is not None:
        out = [ln for ln in out if ln._hyp_forecast_role == role]
    return out


def test_predict_with_animate_true_no_longer_raises():
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=3, animate=True,
                        duration=2, frame_rate=4, show=False)
    assert ani is not None


@pytest.mark.parametrize('mode', [True, 'parallel', 'serial', 'window'])
def test_time_progressing_animation_draws_no_static_full_history_overlay(mode):
    """plot.py:4339 had no `animate` guard, so a time-progressing animation
    would draw BOTH a frozen full-history overlay AND the per-frame one.
    Measured before the fix on animate='spin': 3 dashed 901-vertex overlays,
    landing FIRST in ax.lines."""
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=3, animate=mode,
                        duration=2, frame_rate=4, show=False)
    assert _forecasts(_ax(fig), role='static') == []


def test_spin_still_draws_the_static_overlay():
    """Regression: 'spin' only rotates the camera, so its fixed overlay is
    correct and must be untouched -- including alpha/label/clip."""
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=3, animate='spin',
                        duration=2, frame_rate=4, show=False)
    ax = _ax(fig)
    static = _forecasts(ax, role='static')
    assert len(static) == 3
    for fc in static:
        assert fc.get_linestyle() == '--'
        assert fc.get_alpha() == pytest.approx(0.6)
        assert fc.get_label() == '_nolegend_'
        assert fc.get_clip_on() is False
    ani._func(1, *ani._args)
    first = [np.array(ln.get_data_3d()) for ln in static]
    ani._func(6, *ani._args)
    for a, ln in zip(first, static):
        assert np.allclose(a, np.array(ln.get_data_3d())), \
            'spin forecast overlay must stay fixed'


def test_static_plot_still_draws_the_static_overlay():
    fig = hyp.plot(_series(), '-', predict='Kalman', t=3, show=False)
    assert len(_forecasts(_ax(fig), role='static')) == 3


def test_scalar_morph_still_refuses_predict():
    """A morph interpolates between point clouds; there is no time axis."""
    rng = np.random.default_rng(0)
    clouds = [rng.normal(size=(120, 3)) + off for off in (0.0, 4.0)]
    with pytest.raises(NotImplementedError, match='morph'):
        hyp.plot(clouds, '.', predict='Kalman', t=3, animate='morph',
                 morph_samples=120, duration=1, frame_rate=2, show=False)


def test_list_form_morph_still_refuses_predict():
    """`_resolve_animate_mode` runs at plot.py:3653, ~1300 lines AFTER the
    refusal at plot.py:2338 -- so at the check `animate` is still a raw list
    and `animate == 'morph'` is False. Naive narrowing would silently ACCEPT
    a per-dataset morph list into the forecast path."""
    rng = np.random.default_rng(0)
    clouds = [rng.normal(size=(120, 3)) + off for off in (0.0, 4.0)]
    with pytest.raises(NotImplementedError, match='morph'):
        hyp.plot(clouds, '.', predict='Kalman', t=3,
                 animate=['morph', 'morph'], morph_samples=120,
                 duration=1, frame_rate=2, show=False)
```

- [ ] **Step 2: Run the tests and confirm the expected failures**

Run: `.venv/bin/python -m pytest tests/plot/test_predict_animation.py -v`
Expected: `test_predict_with_animate_true_no_longer_raises` and the 4 parametrized `..._no_static_full_history_overlay` cases FAIL with `NotImplementedError: predict= is only supported with static plots and with animate='spin'`. The two morph tests and the two static/spin tests FAIL on the missing `_hyp_forecast_role` tag (morph raises today, but with the tag helper returning `[]` the spin/static assertions fail at `len(...) == 3`).

- [ ] **Step 3: Tag forecast artists so tests can identify them reliably**

In `_draw_forecast_overlays` (`plot.py:140-165`), after each `ax.plot(...)` call, tag the created artists. Add to the `Returns` docstring that artists carry `_hyp_forecast_role = 'static'`. Rendering is unchanged:

```python
    artists = []
    src_lines = list(ax.lines)
    for i, fc in enumerate(raw_forecasts):
        ...
        # role tag (see hypertools/plot/forecast.py): forecast artists must be
        # identifiable WITHOUT guessing from linestyle -- user data drawn with
        # fmt='--' is dashed too, and trail artists also carry '_nolegend_'.
        for _a in artists[len(artists) - 1:]:
            _a._hyp_forecast_role = 'static'
    return artists
```

- [ ] **Step 4: Narrow the refusal, covering BOTH morph spellings**

Replace `plot.py:2338-2354`:

```python
    # predict= + animate: a forecast over a STATIC scene is a fixed overlay,
    # which is why animate='spin' (camera-only) draws it once and rotates it.
    # Time-progressing modes now precompute a forecast per frame from the
    # history revealed so far (see hypertools/plot/forecast.py). 'morph' is
    # the one mode still refused: it interpolates between point clouds rather
    # than progressing along a time axis, so there is no history to forecast
    # from.
    #
    # BOTH morph spellings must be caught HERE. `_resolve_animate_mode` (which
    # maps a per-dataset list onto 'morph') is not called until plot.py:3653,
    # ~1300 lines below, so at this point `animate` is still the raw list and
    # `animate == "morph"` is False for the list form.
    _is_morph_request = (animate == "morph"
                         or isinstance(animate, (list, tuple)))
    if predict is not None and _is_morph_request:
        raise NotImplementedError(
            "predict= is not supported with animate='morph' (including the "
            "per-dataset morph list form): a morph interpolates between "
            "point clouds rather than progressing along a time axis, so "
            "there is no history to forecast from. Use animate=True/"
            "'parallel'/'serial'/'window'/'spin', or omit predict=."
        )
```

- [ ] **Step 5: Gate the static overlay on non-time-progressing modes**

At `plot.py:4339`, replace `if raw_forecasts is not None:` with:

```python
            # The STATIC full-history overlay belongs only to modes that do
            # not reveal data over time: a static plot, or animate='spin'
            # (camera-only). Time-progressing modes get the per-frame artist
            # built below instead -- drawing both would put a frozen
            # full-history forecast on screen from frame 0.
            if raw_forecasts is not None and animate in (False, None, 'spin'):
```

- [ ] **Step 6: EDIT the existing parametrize (it asserts the opposite of the new behaviour)**

`tests/plot/test_predict_integration.py:167-178` currently parametrizes over `[True, 'parallel', 'serial', 'window', 'morph']`. Four of those five now succeed. Replace the block with:

```python
# --- animate + predict: 'morph' has no time axis, so it still refuses -------
# (True/'parallel'/'serial'/'window' became supported in 1.1: the forecast is
# precomputed per frame from the history revealed so far -- see
# tests/plot/test_predict_animation.py. 'spin' was always allowed; see below.)

@pytest.mark.parametrize('mode', ['morph', ['morph', 'morph']])
def test_morph_animate_and_predict_raises_not_implemented(mode):
    # a morph interpolates between point CLOUDS rather than progressing along
    # a time axis, so there is no history to forecast from. The list form is
    # covered explicitly because `_resolve_animate_mode` (plot.py:3653) does
    # not run until long after the refusal at plot.py:2338, so at the check
    # `animate` is still a raw list.
    rng = np.random.default_rng(0)
    a, b = (rng.normal(size=(120, 3)) + off for off in (0.0, 4.0))
    with pytest.raises(NotImplementedError, match='morph'):
        hyp.plot([a, b], '.', predict='Kalman', animate=mode,
                 morph_samples=120, duration=1, frame_rate=2, show=False)
```

- [ ] **Step 7: Run both test files**

Run: `.venv/bin/python -m pytest tests/plot/test_predict_animation.py tests/plot/test_predict_integration.py -v`
Expected: `test_predict_animation.py` **9 passed** (6 named tests, one parametrized over 4 modes); `test_predict_integration.py` **17 passed** (18 before, minus 5 old parametrizations, plus 2 new + the 2 removed non-morph cases → 15 unchanged + 2 = 17).

- [ ] **Step 8: Run the WHOLE suite (central dispatch changed)**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass. Baseline before this plan: 2564 collected, 2 deselected.

- [ ] **Step 9: Commit**

```bash
git add hypertools/plot/plot.py tests/plot/test_predict_animation.py \
        tests/plot/test_predict_integration.py
git commit -m "feat(plot): accept predict= with time-progressing animations; gate the static overlay"
```

---

## Task 4: Draw the per-frame forecast, and build the box to contain it

**Files:**
- Modify: `hypertools/plot/plot.py` (schedule construction, bounding box, artist creation, frame callback)
- Modify: `hypertools/plot/matplotlib_backend.py` (the live forecast artists)
- Test: `tests/plot/test_predict_animation.py` (append)

**Interfaces:**
- Consumes `ForecastSchedule`/`DisplayTransform` (Task 2), `FrameHooks` (animation-core Task 7), `order=` (animation-core Task 5).
- Produces one `_hyp_forecast_role='live'` artist per dataset, rewritten each frame from the precomputed display-space schedule.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/plot/test_predict_animation.py

def _solid(ax):
    return [ln for ln in ax.lines
            if getattr(ln, '_hyp_forecast_role', None) is None
            and ln.get_linestyle() in ('-', 'solid')]


def test_a_live_forecast_artist_exists_per_dataset():
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=3, animate=True,
                        duration=2, frame_rate=4, show=False)
    ax = _ax(fig)
    ani._func(6, *ani._args)
    assert len(_forecasts(ax, role='live')) == 3


def test_forecast_head_tracks_the_animation():
    """The forecast must start at the CURRENT head, not at the final point."""
    fig, ani = hyp.plot(_series(n=1), '-', predict='Kalman', t=3, animate=True,
                        duration=4, frame_rate=4, show=False)
    ax = _ax(fig)
    heads = []
    for frame in (4, 8, 12):
        ani._func(frame, *ani._args)
        heads.append(np.array(_forecasts(ax, role='live')[0].get_data_3d())[:, 0])
    assert not np.allclose(heads[0], heads[1]), 'forecast head did not move'
    assert not np.allclose(heads[1], heads[2]), 'forecast head did not move'


def test_forecast_is_anchored_near_the_drawn_head():
    """Contract 2. `t` is in RAW analyze-space samples, but the drawn head
    sits on the FRAME GRID: plot.py:3907-3925 resamples a 60-row input to
    round(frame_rate*duration) rows, which matplotlib_backend then densifies
    (measured: 60 raw rows -> 8 grid rows -> 904 drawn vertices at
    duration=2/frame_rate=4, the review's ~15.1x). So the forecast anchors on
    the last revealed RAW sample, which is at most ONE raw step behind the
    drawn head -- an exact atol=1e-6 anchor is impossible by construction.

    The tolerance is DERIVED, not guessed. With antialias=False the drawn head
    line's vertices are consecutive FRAME-GRID rows, and one frame-grid step
    spans 59/15 ~= 3.9 raw steps here -- comfortably more than the <= 1 raw
    step of anchor separation -- so the largest drawn vertex spacing is a
    valid upper bound that needs no magic number.

    The discriminating assertion is the second one: anchoring on the FINAL
    observation (what the static overlay does) puts the gap many raw steps
    away and fails."""
    data = _series(n=1)
    fig, ani = hyp.plot(data, '-', predict='Kalman', t=3, animate=True,
                        antialias=False, duration=4, frame_rate=4, show=False)
    ax = _ax(fig)
    ani._func(8, *ani._args)
    drawn = np.array(_solid(ax)[0].get_data_3d())
    fc = np.array(_forecasts(ax, role='live')[0].get_data_3d())
    head = drawn[:, -1]
    gap = np.linalg.norm(fc[:, 0] - head)

    one_grid_step = np.linalg.norm(np.diff(drawn, axis=1), axis=0).max()
    assert gap <= one_grid_step, (
        f'anchor gap {gap} exceeds one frame-grid step {one_grid_step}')

    # the same data drawn statically: its forecast hangs off the FINAL
    # observation, which at frame 8 of 16 is far from the current head
    static = hyp.plot(data, '-', predict='Kalman', t=3, antialias=False,
                      show=False)
    static_fc = np.array(
        _forecasts(_ax(static), role='static')[0].get_data_3d())
    assert gap < np.linalg.norm(static_fc[:, 0] - head), (
        'forecast is anchored on the FINAL observation, not the current head')


def test_forecast_stays_inside_the_axes_limits():
    """Contract 4: the box is built from data + the WHOLE schedule, so this
    holds by construction and nothing is clamped. Measured before the fix:
    1 of 7 partial-history Kalman forecasts fell outside the fixed [-1, 1]
    animated cube."""
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=5, animate=True,
                        duration=4, frame_rate=4, show=False)
    ax = _ax(fig)
    lims = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    for frame in range(16):
        ani._func(frame, *ani._args)
        for fc in _forecasts(ax, role='live'):
            pts = np.array(fc.get_data_3d())
            if pts.size == 0:
                continue
            assert (pts.min(axis=1) >= lims[:, 0] - 1e-6).all()
            assert (pts.max(axis=1) <= lims[:, 1] + 1e-6).all()


def test_t_is_measured_in_raw_samples_not_frames_or_vertices():
    """antialias=False draws the raw vertices, so the count is checkable."""
    fig, ani = hyp.plot(_series(n=1), '-', predict='Kalman', t=3, animate=True,
                        antialias=False, duration=4, frame_rate=4, show=False)
    ax = _ax(fig)
    ani._func(8, *ani._args)
    assert np.array(_forecasts(ax, role='live')[0].get_data_3d()).shape[1] == 4


def test_t_equals_one_is_the_next_raw_step():
    fig, ani = hyp.plot(_series(n=1), '-', predict='Kalman', t=1, animate=True,
                        antialias=False, duration=4, frame_rate=4, show=False)
    ax = _ax(fig)
    ani._func(8, *ani._args)
    assert np.array(_forecasts(ax, role='live')[0].get_data_3d()).shape[1] == 2


def test_antialias_true_smooths_the_forecast_like_any_other_line():
    """plot.py:1904-1908 documents this as contract ('Forecast overlays drawn
    by predict= are smoothed the same way'), and the spin overlay is pinned to
    it at test_predict_integration.py:198. Measured today: t=1 draws 900
    vertices at antialias=True, 2 at antialias=False."""
    fig, ani = hyp.plot(_series(n=1), '-', predict='Kalman', t=1, animate=True,
                        duration=4, frame_rate=4, show=False)
    ax = _ax(fig)
    ani._func(8, *ani._args)
    assert np.array(_forecasts(ax, role='live')[0].get_data_3d()).shape[1] > 2


def test_frames_are_idempotent():
    """Contract 6: ani.save()/to_jshtml() replay from frame 0, and these tests
    drive frames out of order."""
    fig, ani = hyp.plot(_series(n=1), '-', predict='Kalman', t=3, animate=True,
                        duration=4, frame_rate=4, show=False)
    ax = _ax(fig)
    ani._func(9, *ani._args)
    first = np.array(_forecasts(ax, role='live')[0].get_data_3d())
    for f in (0, 3, 15, 2, 9):
        ani._func(f, *ani._args)
    assert np.allclose(first,
                       np.array(_forecasts(ax, role='live')[0].get_data_3d()))


def test_forecast_composes_with_order_serial():
    """Requires animation-core Task 5 (order=)."""
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=3, animate=True,
                        order='serial', duration=4, frame_rate=4, show=False)
    ax = _ax(fig)
    ani._func(15, *ani._args)
    drawn = [fc for fc in _forecasts(ax, role='live')
             if np.array(fc.get_data_3d()).size]
    assert len(drawn) == 3, 'every dataset is fully revealed by the last frame'


def test_a_dataset_with_too_little_history_hides_its_forecast():
    """Frame 0 reveals one raw row; a forecaster cannot be fitted to it, and
    an empty/garbage trace must not be drawn instead."""
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=3, animate=True,
                        duration=4, frame_rate=4, show=False)
    ax = _ax(fig)
    ani._func(0, *ani._args)
    for fc in _forecasts(ax, role='live'):
        assert not fc.get_visible() or np.array(fc.get_data_3d()).size == 0


def test_forecast_artists_are_not_identified_by_linestyle():
    """T5: user data drawn with fmt='--' is dashed but is NOT a forecast."""
    fig, ani = hyp.plot(_series(), '--', predict='Kalman', t=3, animate=True,
                        duration=2, frame_rate=4, show=False)
    ax = _ax(fig)
    ani._func(6, *ani._args)
    dashed = [ln for ln in ax.lines if ln.get_linestyle() not in ('-', 'solid')]
    assert len(dashed) > len(_forecasts(ax)), \
        'the dashed-linestyle heuristic would have swept up the data lines'
    assert len(_forecasts(ax, role='live')) == 3


def test_hue_regrouping_drops_forecasts_exactly_like_the_static_path():
    """plot.py:3999 nulls raw_forecasts when hue=/cluster= regroups xform, so
    the 1:1 dataset<->forecast correspondence is gone. The animated path
    inherits that guard verbatim: no forecast is drawn, and nothing crashes."""
    data = _series(n=1, rows=60)
    labels = np.array(['a', 'b'] * 30)
    fig, ani = hyp.plot(data, '-', predict='Kalman', t=3, hue=labels,
                        animate=True, duration=2, frame_rate=4, show=False)
    ax = _ax(fig)
    ani._func(6, *ani._args)
    assert _forecasts(ax) == []
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/python -m pytest tests/plot/test_predict_animation.py -v`
Expected: the 12 new tests FAIL — `_forecasts(ax, role='live')` returns `[]`, so most fail with `IndexError: list index out of range` and the count assertions fail at `0 != 3`. The 9 Task 3 tests still pass.

- [ ] **Step 3: Snapshot the analyze-space history alongside the forecasts**

At `plot.py:2977-2988`, where `raw_forecasts`/`bundle_forecasts` are built, also keep the analyze-space arrays the schedule will forecast from. They are captured here, beside `raw_forecasts`, so the existing correspondence guard at `plot.py:3999` covers them too:

```python
    raw_forecasts = None
    bundle_forecasts = None
    analyze_histories = None
    if predict is not None:
        from ..predict.predict import predict as _predictor
        _fc = _predictor(xform, model=predict, t=t)
        if not isinstance(_fc, list):
            _fc = [_fc]
        bundle_forecasts = [np.asarray(fc, dtype=float) for fc in _fc]
        raw_forecasts = [
            np.vstack([np.asarray(xi[-1:]), np.asarray(fc)])
            for xi, fc in zip(xform, _fc)
        ]
        # ANALYZE-space copies for the animated per-frame schedule (see
        # hypertools/plot/forecast.py). Taken HERE, beside raw_forecasts, so
        # they keep the same 1:1 dataset correspondence the guard at
        # plot.py:3999 checks -- and BEFORE `_interp_anim_line` resamples
        # `xform` onto the frame grid (plot.py:3907-3925), because `t` is
        # measured in RAW analyze-space samples.
        analyze_histories = [np.array(xi, dtype=float, copy=True)
                             for xi in xform]
```

Extend the guard at `plot.py:3999` so the snapshot is dropped with the forecasts:

```python
    if raw_forecasts is not None and len(raw_forecasts) != len(xform):
        raw_forecasts = None
        analyze_histories = None
```

- [ ] **Step 4: Build the schedule and fold it into the bounding box**

Immediately before the centre/scale block (`plot.py:4002`), build the schedule; then include it in **both** joint stacks so the box contains every forecast by construction, and hand the resulting `DisplayTransform` back to the schedule:

```python
    # Animated predict= (CASE A -- STATIC data revealed over time): every
    # observation is known before the first frame, so every forecast the
    # animation will ever draw is knowable now. Precompute the whole schedule
    # here so (a) it can go into the centre/scale statistics below and land
    # inside the cube BY CONSTRUCTION -- no clamping, unlike the streaming
    # path at hypertools/io/streaming.py:382-401, where the box is frozen
    # from the head samples -- and (b) every frame is a pure lookup, so
    # ani.save()/to_jshtml() replays render identically.
    forecast_schedule = None
    if (raw_forecasts is not None and animate
            and animate not in ('spin',)):
        from .forecast import ForecastSchedule
        _n_frames = max(1, int(round(frame_rate * duration)))
        _grid_lengths = [len(xi) for xi in xform]
        _builder = (ForecastSchedule.for_serial
                    if (animate == 'serial' or order == 'serial')
                    else ForecastSchedule.for_parallel)
        forecast_schedule = _builder(
            analyze_histories, _grid_lengths, model=predict, t=t,
            n_frames=_n_frames)

    if raw_forecasts is not None:
        _fc_rows = [np.vstack(raw_forecasts)]
        if forecast_schedule is not None:
            _fc_rows.append(forecast_schedule.stacked_paths())
        _joint = np.vstack([np.vstack(xform)] + _fc_rows)
        _mean = np.mean(_joint, 0)
        xform = [xi - _mean for xi in xform]
        raw_forecasts = [fc - _mean for fc in raw_forecasts]
        raw_xform = [xi - _mean for xi in raw_xform]

        _joint = np.vstack([np.vstack(xform)]
                           + [r - _mean for r in _fc_rows])
        _m1 = np.min(_joint)
        _m2 = np.max(_joint - _m1) or 1.0
        _rescale = lambda a: 2 * (np.divide(a - _m1, _m2)) - 1
        xform = [_rescale(xi) for xi in xform]
        raw_forecasts = [_rescale(fc) for fc in raw_forecasts]
        raw_xform = [_rescale(xi) for xi in raw_xform]
        if forecast_schedule is not None:
            from .forecast import DisplayTransform
            forecast_schedule = forecast_schedule.to_display(
                DisplayTransform(_mean, _m1, _m2))
```

- [ ] **Step 5: Create the live artists and drive them from the schedule**

After `_draw(...)` returns (`plot.py:4291-4330`), when `forecast_schedule is not None`, create one dashed artist per dataset in that dataset's colour with `alpha=0.6`, `label='_nolegend_'`, `set_clip_on(False)` and `_hyp_forecast_role = 'live'` — the same styling `_draw_forecast_overlays` applies (`plot.py:154-156`), so a paused animation looks like a static plot. Then register a frame callback on `line_ani` via animation-core Task 7's `FrameHooks.callbacks`:

```python
                def _update_forecasts(frame, _sched=forecast_schedule,
                                      _artists=_live_forecast_artists,
                                      _antialias=antialias):
                    for i, art in enumerate(_artists):
                        pts = _sched.polyline(i, frame)
                        if pts is None or len(pts) < 2:
                            art.set_visible(False)
                            art.set_data([], [])
                            art.set_3d_properties([])
                            continue
                        if _antialias:
                            # documented parity with the static overlay
                            # (plot.py:1904-1908, :149-150)
                            pts = _interp_static_line(pts)
                        art.set_visible(True)
                        art.set_data(pts[:, 0:2].T)
                        art.set_3d_properties(pts[:, 2])
```

Two invariants the tests pin down:

- The reveal comes from `revealed_raw_counts`, which delegates to `_anim_window_bounds` — the library's single reveal implementation. Never re-derive it locally, and never read `FrameContext.revealed_counts` (documented `None` for parallel animations).
- The callback **reads** the schedule and never mutates it, so frames stay idempotent.

For 2-D and 1-D animations use `set_data` alone (no `set_3d_properties`), mirroring `_draw_forecast_overlays`' `d >= 3 / d == 2 / else` dispatch (`plot.py:152-164`).

- [ ] **Step 6: Run the test and confirm it passes**

Run: `.venv/bin/python -m pytest tests/plot/test_predict_animation.py -v`
Expected: **21 passed** (9 from Task 3 + 12 here).

- [ ] **Step 7: Run the WHOLE suite (central dispatch changed)**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 8: Update the docstring**

In `plot()`'s `predict`/`t` entries:

```
    predict : str, dict, or None
        Forecast model to overlay ('Kalman', 'ARIMA', 'GaussianProcess',
        'Laplace', 'Chronos', ...). Drawn as a dashed continuation of each
        dataset in its own colour. With a time-progressing animation
        (``animate=True``/``'parallel'``/``'serial'``/``'window'``) the
        forecast is recomputed from the history revealed so far and
        re-anchored on the last revealed observation, so it grows with the
        animation. Every one of those forecasts is computed BEFORE the first
        frame is drawn and included in the plot's centre/scale statistics, so
        the whole fan is inside the cube and nothing is ever clipped or
        clamped. Not supported with ``animate='morph'`` (including the
        per-dataset morph list form), which has no time axis. See
        ``forecast_trail=`` to keep earlier forecasts on screen.
    t : int, default 10
        Forecast horizon, in RAW observations of the analyzed data -- NOT in
        animation frames and NOT in drawn vertices. ``t=1`` forecasts only
        the next observation. Because an animation is paced on a resampled
        frame grid (see ``duration``/``frame_rate``), an animated forecast
        joins the drawn trajectory to within one raw observation rather than
        exactly.
```

- [ ] **Step 9: Commit**

```bash
git add hypertools/plot/plot.py hypertools/plot/matplotlib_backend.py \
        tests/plot/test_predict_animation.py
git commit -m "feat(plot): per-frame forecasts from a precomputed schedule, inside the box by construction"
```

---

## Task 5: `forecast_trail=` — keep earlier forecasts on screen

The forecast analogue of `chemtrails=`: earlier forecasts stay visible as a fading fan, so a viewer can see how the prediction changed as history accumulated. Because the schedule is precomputed, the fan at frame `f` is a pure function of `f` — no ring buffer, no reset step, no divergence between a saved GIF and an interactively-played animation.

**Files:**
- Modify: `hypertools/plot/forecast.py`, `hypertools/plot/plot.py`, `hypertools/plot/matplotlib_backend.py`
- Test: `tests/plot/test_forecast_trail.py`

**Interfaces:**
- `plot(..., forecast_trail=False | True | int)`. `True` retains `DEFAULT_FORECAST_TRAIL = 16` past forecasts; an int sets the cap.
- `trail_alpha(age, n_retained, live_alpha=0.6, floor=0.08)`; age 0 is the live forecast.
- `trail_frames(frame, n_retained, n_frames)` → the frame indices whose forecasts are retained at `frame`, newest first. Pure.

- [ ] **Step 1: Write the failing test**

```python
# tests/plot/test_forecast_trail.py
"""The retained forecast fan -- the forecast analogue of chemtrails=."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

import hypertools as hyp


def _series(n=1, rows=60, dims=4, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, dims)).cumsum(axis=0) for _ in range(n)]


def _ax(fig):
    return [a for a in fig.axes if hasattr(a, 'zaxis')][0]


def _forecasts(ax, role=None):
    out = [ln for ln in ax.lines
           if getattr(ln, '_hyp_forecast_role', None) is not None]
    if role is not None:
        out = [ln for ln in out if ln._hyp_forecast_role == role]
    return out


def _drawn(ax, role=None):
    """Artists that are actually on screen. A preallocated-but-unwritten slot
    is hidden with EMPTY data -- alpha is not the emptiness signal (the v1
    plan's trail_alpha never returned 0, so an alpha>0 count could not grow)."""
    return [ln for ln in _forecasts(ax, role)
            if ln.get_visible() and np.array(ln.get_data_3d()).size]


def _drive(ani, upto):
    for f in range(upto + 1):
        ani._func(f, *ani._args)


def test_trail_accumulates_past_forecasts():
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=3, animate=True,
                        forecast_trail=True, duration=4, frame_rate=4,
                        show=False)
    ax = _ax(fig)
    _drive(ani, 3)
    early = len(_drawn(ax))
    _drive(ani, 14)
    late = len(_drawn(ax))
    assert late > early, f'trail should accumulate; got {early} -> {late}'


def test_without_trail_only_the_live_forecast_is_drawn():
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=3, animate=True,
                        duration=4, frame_rate=4, show=False)
    ax = _ax(fig)
    _drive(ani, 12)
    assert len(_drawn(ax)) == 1
    assert _drawn(ax)[0]._hyp_forecast_role == 'live'


def test_trail_is_capped_by_an_integer():
    """Driven SEQUENTIALLY: a single _func(20) call could satisfy any cap."""
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=3, animate=True,
                        forecast_trail=4, duration=6, frame_rate=4,
                        show=False)
    ax = _ax(fig)
    _drive(ani, 23)
    assert len(_drawn(ax)) <= 5, 'cap of 4 past forecasts plus the live one'
    assert len(_drawn(ax, role='trail')) <= 4


def test_an_uncapped_trail_retains_more_than_a_capped_one():
    """Proves the cap is what limits the fan, not the frame count."""
    kw = dict(predict='Kalman', t=3, animate=True, duration=6, frame_rate=4,
              show=False)
    big, ani_big = hyp.plot(_series(), '-', forecast_trail=16, **kw)
    small, ani_small = hyp.plot(_series(), '-', forecast_trail=2, **kw)
    _drive(ani_big, 23)
    _drive(ani_small, 23)
    assert len(_drawn(_ax(big))) > len(_drawn(_ax(small)))


def test_live_forecast_is_strictly_more_opaque_than_every_trail():
    """T1: the v1 assertion `max(a) == a[0] or max(a) >= sorted(a)[-1]` was a
    tautology (`sorted(a)[-1]` IS `max(a)`). Roles make this checkable."""
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=3, animate=True,
                        forecast_trail=True, duration=4, frame_rate=4,
                        show=False)
    ax = _ax(fig)
    _drive(ani, 14)
    live = _drawn(ax, role='live')[0]
    trails = _drawn(ax, role='trail')
    assert trails, 'expected a fan by frame 14'
    assert all(live.get_alpha() > tr.get_alpha() for tr in trails)


def test_trail_alpha_decreases_with_age():
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=3, animate=True,
                        forecast_trail=True, duration=4, frame_rate=4,
                        show=False)
    ax = _ax(fig)
    _drive(ani, 14)
    by_age = sorted(_drawn(ax, role='trail'), key=lambda ln: ln._hyp_forecast_age)
    alphas = [ln.get_alpha() for ln in by_age]
    assert alphas == sorted(alphas, reverse=True), alphas
    assert min(alphas) < max(alphas), 'the trail must actually fade'


def test_the_fan_is_a_pure_function_of_the_frame_index():
    """G5: FuncAnimation replays from frame 0 for save()/to_jshtml(), and the
    tests above drive frames out of order. A ring buffer would diverge."""
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=3, animate=True,
                        forecast_trail=True, duration=4, frame_rate=4,
                        show=False)
    ax = _ax(fig)
    _drive(ani, 12)
    sequential = [np.array(ln.get_data_3d()) for ln in _drawn(ax)]
    for f in (0, 15, 3, 12):
        ani._func(f, *ani._args)
    jumped = [np.array(ln.get_data_3d()) for ln in _drawn(ax)]
    assert len(sequential) == len(jumped)
    for a, b in zip(sequential, jumped):
        assert np.allclose(a, b)


def test_forecast_trail_requires_predict():
    with pytest.raises(ValueError, match='forecast_trail= requires predict='):
        hyp.plot(_series(), '-', animate=True, forecast_trail=True,
                 duration=2, frame_rate=4, show=False)


@pytest.mark.parametrize('bad', [-1, 'yes', 2.5])
def test_invalid_forecast_trail_raises(bad):
    with pytest.raises((ValueError, TypeError), match='forecast_trail'):
        hyp.plot(_series(), '-', predict='Kalman', t=3, animate=True,
                 forecast_trail=bad, duration=2, frame_rate=4, show=False)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/python -m pytest tests/plot/test_forecast_trail.py -v`
Expected: every test FAILS with `TypeError: plot() got an unexpected keyword argument 'forecast_trail'` (10 items: 9 named tests, one parametrized over 3 values).

- [ ] **Step 3: Add and validate the kwarg**

Add `forecast_trail=False` to the `plot()` signature next to `chemtrails=` (`plot.py:558`), and validate it early, beside the other fail-fast validations:

```python
#: Past forecasts retained by `forecast_trail=True`.
DEFAULT_FORECAST_TRAIL = 16


def _validate_forecast_trail(forecast_trail, predict):
    """`forecast_trail=` keeps earlier forecasts on screen as a fading fan.

    Returns the number of past forecasts to retain (0 = trail disabled)."""
    if forecast_trail in (False, None, 0):
        return 0
    if predict is None:
        raise ValueError(
            "forecast_trail= requires predict=; there are no forecasts to "
            "retain without a forecast model.")
    if forecast_trail is True:
        return DEFAULT_FORECAST_TRAIL
    if isinstance(forecast_trail, bool) or not isinstance(
            forecast_trail, (int, np.integer)):
        raise TypeError(
            "forecast_trail must be True/False or a positive int (the number "
            f"of past forecasts to keep); got {forecast_trail!r}.")
    if forecast_trail < 1:
        raise ValueError(
            f"forecast_trail must be >= 1 when given as an int; got "
            f"{forecast_trail}.")
    return forecast_trail
```

- [ ] **Step 4: Render the fan as a pure function of the frame index**

Add to `hypertools/plot/forecast.py`:

```python
def trail_frames(frame, n_retained, n_frames, stride=1):
    """Frames whose forecasts are retained at `frame`, NEWEST FIRST.

    Pure: the fan at frame N depends only on N. There is deliberately no
    accumulating buffer -- `FuncAnimation` replays from frame 0 for
    `save()`/`to_jshtml()`, and a stateful fan would make a saved GIF differ
    from an interactively-played animation.
    """
    out = []
    for age in range(1, int(n_retained) + 1):
        past = frame - age * int(stride)
        if past < 0:
            break
        out.append(past)
    return out


def trail_alpha(age, n_retained, live_alpha=0.6, floor=0.08):
    """Alpha for a forecast `age` frames old. Age 0 is the live forecast.

    `live_alpha` matches the static overlay's 0.6 (`plot.py:156`), so a paused
    animation looks like a static plot.
    """
    if age <= 0:
        return live_alpha
    decay = 1.0 - (age / max(1, int(n_retained) + 1))
    return max(floor, floor + (live_alpha - floor) * decay)
```

At setup, preallocate `n_retained` dashed artists per dataset (allocating artists mid-animation is what makes matplotlib animations stutter), each tagged `_hyp_forecast_role='trail'` and `_hyp_forecast_age=age`, at `trail_alpha(age, n_retained)`. Every preallocated slot starts **hidden with empty data** — emptiness, not alpha, is the "not yet written" signal, because `trail_alpha` never returns 0. Each frame, write `schedule.polyline(i, past_frame)` into the slot for each entry of `trail_frames(frame, n_retained, n_frames)`, and hide any slot with no corresponding past frame.

Note that `stacked_paths()` (Task 2) already covers every retained forecast, because a retained forecast is just an earlier frame's forecast — so `forecast_trail=` needs **no** change to the bounding box, and the fan cannot leave the cube.

- [ ] **Step 5: Run the test and confirm it passes**

Run: `.venv/bin/python -m pytest tests/plot/test_forecast_trail.py -v`
Expected: **11 passed** (8 named tests + one parametrized over 3 values).

- [ ] **Step 6: Document it**

```
    forecast_trail : bool or int, default False
        Keep earlier forecasts on screen as a fading fan, so a viewer can see
        how the prediction changed as history accumulated -- the forecast
        analogue of ``chemtrails=``. ``True`` retains 16 past forecasts; an
        int sets the cap. The live forecast is the most opaque and older ones
        fade with age. Requires ``predict=`` and a time-progressing
        ``animate=`` mode. The fan at any frame depends only on that frame,
        so a saved animation matches an interactively-played one exactly.
```

- [ ] **Step 7: Run the WHOLE suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add hypertools/plot/plot.py hypertools/plot/forecast.py \
        hypertools/plot/matplotlib_backend.py tests/plot/test_forecast_trail.py
git commit -m "feat(plot): forecast_trail= retains earlier forecasts as a fading fan"
```

---

## Task 6: Plotly parity

Removing the refusal at `plot.py:2346` is backend-agnostic, so `backend='plotly', animate=True, predict=` becomes reachable too — but `plotly_backend.py:901-931` draws forecasts statically and `_add_animation` (`plotly_backend.py:2517-2529`) takes no forecast argument, so a plotly animation would show a **frozen** full-history overlay. Plotly must be identical to matplotlib.

The mechanism already exists. Plotly renders chemtrails/precog/bullettime as **separate traces at a fixed alpha whose row-window data is rewritten every frame**: created at `_to_plotly_color(color, 0.3)` (`plotly_backend.py:951`), positioned via `trail_trace_start` (`plotly_backend.py:945`, needed precisely *because* forecast traces are appended in between — the code says so at `plotly_backend.py:936-943`), addressed per frame at `plotly_backend.py:2896-2897`, and rewritten at `plotly_backend.py:2947-2956`. The forecast live trace and the forecast trail traces use exactly that pattern, driven by the same precomputed `ForecastSchedule`.

**Files:**
- Modify: `hypertools/plot/plotly_backend.py`, `hypertools/plot/plot.py` (pass the schedule to `plotly_draw`, `plot.py:4181-4230`)
- Test: `tests/plot/test_forecast_animation_plotly.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/plot/test_forecast_animation_plotly.py
"""matplotlib/plotly parity for animated predict=."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

import hypertools as hyp

pytest.importorskip('plotly')


def _series(n=2, rows=60, dims=4, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, dims)).cumsum(axis=0) for _ in range(n)]


def _fc_traces(fig):
    return [tr for tr in fig.data
            if (tr.meta or {}).get('hyp_forecast_role') is not None]


def _fc_role(fig, role):
    return [tr for tr in fig.data
            if (tr.meta or {}).get('hyp_forecast_role') == role]


def _frame_snapshot(fig, k):
    from hypertools.plot.plotly_backend import _frame_snapshots
    for i, snap in enumerate(_frame_snapshots(fig)):
        if i == k:
            return snap
    raise AssertionError(f'no frame {k}')


def _mpl_live(fig, frame, ani):
    ani._func(frame, *ani._args)
    ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
    live = [ln for ln in ax.lines
            if getattr(ln, '_hyp_forecast_role', None) == 'live']
    return [np.array(ln.get_data_3d()).T for ln in live]


def test_plotly_animated_plot_has_a_live_forecast_trace_per_dataset():
    fig = hyp.plot(_series(), '-', predict='Kalman', t=3, animate=True,
                   duration=2, frame_rate=4, backend='plotly', show=False)
    assert len(_fc_role(fig, 'live')) == 2


def test_plotly_forecast_traces_are_updated_per_frame_not_frozen():
    """plotly's frame updates address only the data + trail trace ranges
    (plotly_backend.py:2896-2897), so an un-wired forecast trace stays
    frozen at its setup value."""
    fig = hyp.plot(_series(n=1), '-', predict='Kalman', t=3, animate=True,
                   duration=4, frame_rate=4, backend='plotly', show=False)
    early = _fc_role(_frame_snapshot(fig, 4), 'live')[0]
    late = _fc_role(_frame_snapshot(fig, 12), 'live')[0]
    assert not np.allclose(np.asarray(early.x, dtype=float),
                           np.asarray(late.x, dtype=float))


def test_plotly_and_matplotlib_draw_the_same_final_frame_forecast():
    """Contract 8. At the final frame both backends have revealed the whole
    history, so both draw the full-history forecast in the same display box."""
    kw = dict(predict='Kalman', t=3, animate=True, duration=4, frame_rate=4,
              antialias=False, show=False)
    data = _series(n=1)
    pl = hyp.plot(data, '-', backend='plotly', **kw)
    mpl_fig, ani = hyp.plot(data, '-', backend='matplotlib', **kw)

    tr = _fc_role(_frame_snapshot(pl, 15), 'live')[0]
    plotly_pts = np.column_stack([np.asarray(tr.x, dtype=float),
                                  np.asarray(tr.y, dtype=float),
                                  np.asarray(tr.z, dtype=float)])
    mpl_pts = _mpl_live(mpl_fig, 15, ani)[0]
    assert plotly_pts.shape == mpl_pts.shape
    assert np.allclose(plotly_pts, mpl_pts, atol=1e-6)


def test_plotly_forecast_stays_inside_the_scene_range():
    """Same Contract 4 guarantee as matplotlib: the box was built to hold it."""
    fig = hyp.plot(_series(), '-', predict='Kalman', t=5, animate=True,
                   duration=4, frame_rate=4, backend='plotly', show=False)
    for k in range(16):
        for tr in _fc_role(_frame_snapshot(fig, k), 'live'):
            pts = np.concatenate([np.asarray(getattr(tr, a), dtype=float)
                                  for a in ('x', 'y', 'z')])
            if pts.size == 0:
                continue
            assert pts.min() >= -1.0 - 1e-6 and pts.max() <= 1.0 + 1e-6


def test_plotly_forecast_trail_traces_carry_decreasing_opacity():
    """Mirrors the chemtrails mechanism: separate traces at a fixed alpha,
    data rewritten per frame (plotly_backend.py:951, :2947-2956)."""
    fig = hyp.plot(_series(n=1), '-', predict='Kalman', t=3, animate=True,
                   forecast_trail=4, duration=4, frame_rate=4,
                   backend='plotly', show=False)
    trails = _fc_role(fig, 'trail')
    assert len(trails) == 4
    by_age = sorted(trails, key=lambda tr: tr.meta['hyp_forecast_age'])
    alphas = [tr.meta['hyp_forecast_alpha'] for tr in by_age]
    assert alphas == sorted(alphas, reverse=True), alphas
    assert min(alphas) < max(alphas), 'the trail must actually fade'
    live_alpha = _fc_role(fig, 'live')[0].meta['hyp_forecast_alpha']
    assert all(live_alpha > a for a in alphas)
    # the declared alpha is the one actually baked into the rgba colour
    for tr, a in zip(by_age, alphas):
        assert f'{a}' in tr.line.color or str(round(a, 3)) in tr.line.color


def test_plotly_trail_is_populated_by_the_late_frames():
    fig = hyp.plot(_series(n=1), '-', predict='Kalman', t=3, animate=True,
                   forecast_trail=4, duration=4, frame_rate=4,
                   backend='plotly', show=False)
    late = _fc_role(_frame_snapshot(fig, 15), 'trail')
    drawn = [tr for tr in late if np.asarray(tr.x, dtype=float).size]
    assert len(drawn) == 4


def test_plotly_and_matplotlib_agree_on_the_forecast_trace_count():
    kw = dict(predict='Kalman', t=3, animate=True, forecast_trail=4,
              duration=2, frame_rate=4, show=False)
    pl = hyp.plot(_series(), '-', backend='plotly', **kw)
    mpl_fig, ani = hyp.plot(_series(), '-', backend='matplotlib', **kw)
    ax = [a for a in mpl_fig.axes if hasattr(a, 'zaxis')][0]
    mpl_n = len([ln for ln in ax.lines
                 if getattr(ln, '_hyp_forecast_role', None) is not None])
    assert len(_fc_traces(pl)) == mpl_n


def test_plotly_morph_still_refuses_predict():
    rng = np.random.default_rng(0)
    clouds = [rng.normal(size=(120, 3)) + off for off in (0.0, 4.0)]
    with pytest.raises(NotImplementedError, match='morph'):
        hyp.plot(clouds, '.', predict='Kalman', t=3, animate='morph',
                 morph_samples=120, duration=1, frame_rate=2,
                 backend='plotly', show=False)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/python -m pytest tests/plot/test_forecast_animation_plotly.py -v`
Expected: the parity tests FAIL — `_fc_role(fig, 'live')` is empty (plotly's forecast traces carry no `meta` tag), and `test_plotly_forecast_traces_are_updated_per_frame_not_frozen` fails with `assert not np.allclose(...)` because the trace is frozen.

- [ ] **Step 3: Tag the existing static forecast traces**

In `plotly_backend.py:901-931`, add `meta=dict(hyp_forecast_role='static')` to `fc_common`. Rendering is unchanged; `test_plotly_predict_trace_parity` (`test_predict_integration.py:228-243`) must still pass untouched.

- [ ] **Step 4: Accept the schedule and create the animated traces**

Add `forecast_schedule=None` and `forecast_trail=0` to `plotly_draw`'s signature (beside `forecasts=None`, `plotly_backend.py:465`) and pass them from `plot.py:4230` alongside `forecasts=raw_forecasts`. When `forecast_schedule is not None`:

- Skip the static block (parity with `plot.py:4339`'s gate).
- Create one `live` trace per dataset — `dash='dash'`, `_to_plotly_color(color, 0.6)`, `showlegend=False`, `hoverinfo='skip'`, `meta=dict(hyp_forecast_role='live', hyp_dataset=i, hyp_forecast_age=0, hyp_forecast_alpha=trail_alpha(0, n_retained))`.
- Create `forecast_trail` trail traces per dataset at `_to_plotly_color(color, trail_alpha(age, n_retained))`, `meta=dict(hyp_forecast_role='trail', hyp_forecast_age=age, hyp_dataset=i, hyp_forecast_alpha=trail_alpha(age, n_retained))`. The declared `hyp_forecast_alpha` must be the exact value baked into the rgba string — the parity test asserts both.
- Record `forecast_trace_start = <index of the first live trace>` and the per-trace `(dataset, age)` map, exactly as `trail_trace_start`/`trail_dataset_indices` do (`plotly_backend.py:936-947`), and pass both into `_add_animation`.

- [ ] **Step 5: Update the traces every frame**

In `_add_animation`'s parallel/serial frame loops, extend `trace_indices` with the forecast trace range (the same way `has_trails` extends it at `plotly_backend.py:2895-2897`) and append the schedule's polylines to `frame_traces` in that order. Use `trail_frames(k, n_retained, n_frames)` for the fan, and an empty `x/y/z` for a frame with no forecast — matching matplotlib's hidden-artist state.

Plotly's parallel reveal is `end = max(2, ceil((k + 1) * max_len / n_frames))` (`plotly_backend.py:2898`) against `max_len`, whereas matplotlib's is per dataset (`matplotlib_backend.py:357-358`); they coincide when the animated arrays share a length, which they always do for line datasets (`plot.py:3922-3925` resamples every one to `round(frame_rate*duration)` rows). Index the schedule by the **frame** `k`, not by a re-derived row count, so the two backends read the same table.

- [ ] **Step 6: Run the test and confirm it passes**

Run: `.venv/bin/python -m pytest tests/plot/test_forecast_animation_plotly.py -v`
Expected: **8 passed.**

- [ ] **Step 7: Run the WHOLE suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass, including `test_plotly_predict_trace_parity` unchanged.

- [ ] **Step 8: Commit**

```bash
git add hypertools/plot/plotly_backend.py hypertools/plot/plot.py \
        tests/plot/test_forecast_animation_plotly.py
git commit -m "feat(plot): plotly parity for animated predict= and forecast_trail="
```

---

## Task 7: The `return_model=` bundle contract

The review's headline question: what does `bundle['predict']['forecasts']` hold when the forecast is recomputed for every frame? Contract 7's answer needs no new value, because at the final frame the revealed history **is** the full history (`_anim_window_bounds(total-1, total, n, w)` → `end = n`), so the documented full-history forecast **is** the final-frame forecast. Only one sentence of `plot.py:1937-1941` is now imprecise, and no test covers `return_model=` for an animated forecast at all.

**Files:**
- Modify: `hypertools/plot/plot.py:1920-1941`, `:1955`
- Test: `tests/plot/test_predict_animation.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/plot/test_predict_animation.py

def test_bundle_forecasts_are_the_full_history_forecast():
    """Unchanged from static/spin: exactly t rows, analyze space, one per
    input dataset (plot.py:1935-1941)."""
    data = _series(n=2)
    out = hyp.plot(data, '-', predict='Kalman', t=4, animate=True,
                   duration=2, frame_rate=4, show=False, return_model=True)
    assert out['animation'] is not None
    assert out['predict']['model'] == 'Kalman'
    assert out['predict']['params'] == {'t': 4}
    forecasts = out['predict']['forecasts']
    assert len(forecasts) == 2
    for fc in forecasts:
        assert np.asarray(fc).shape == (4, 3)


def test_bundle_forecast_matches_hyp_predict_on_the_returned_xform_data():
    """Contract 7: the bundle stays interchangeable with hyp.predict, exactly
    as the static path promises and test_predict_return_model_bundle pins."""
    out = hyp.plot(_series(n=1), '-', predict='Kalman', t=4, animate=True,
                   duration=2, frame_rate=4, show=False, return_model=True)
    direct = np.asarray(hyp.predict(np.asarray(out['xform_data'][0]),
                                    model='Kalman', t=4), dtype=float)
    assert np.allclose(np.asarray(out['predict']['forecasts'][0]), direct,
                       rtol=1e-6, atol=1e-6)


def test_the_final_frame_draws_exactly_the_bundled_forecast():
    """The final frame reveals the whole history, so the drawn per-frame
    forecast IS the bundled full-history one -- which is why the bundle needs
    no redefinition for animated plots."""
    out = hyp.plot(_series(n=1), '-', predict='Kalman', t=4, animate=True,
                   antialias=False, duration=4, frame_rate=4, show=False,
                   return_model=True)
    fig, ani = out['fig'], out['animation']
    ani._func(15, *ani._args)
    ax = _ax(fig)
    drawn = np.array(_forecasts(ax, role='live')[0].get_data_3d()).T
    # t + 1 vertices: the anchor plus t forecast steps
    assert drawn.shape == (5, 3)
    # and the t forecast steps advance in the same directions as the bundle
    bundled = np.asarray(out['predict']['forecasts'][0], dtype=float)
    assert np.allclose(np.sign(np.diff(drawn[1:], axis=0)),
                       np.sign(np.diff(bundled, axis=0)))


def test_return_model_xform_data_is_untouched_by_the_schedule():
    """The schedule snapshots analyze-space copies; it must not alias or
    mutate what the user gets back."""
    plain = hyp.plot(_series(n=1), '-', animate=True, duration=2,
                     frame_rate=4, show=False, return_model=True)
    forecast = hyp.plot(_series(n=1), '-', predict='Kalman', t=3,
                        animate=True, duration=2, frame_rate=4, show=False,
                        return_model=True)
    assert (np.asarray(plain['xform_data'][0]).shape
            == np.asarray(forecast['xform_data'][0]).shape)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/python -m pytest tests/plot/test_predict_animation.py -k bundle -v`
Expected: before Tasks 3-4 these fail with `NotImplementedError`; run at this point they exercise the wiring and `test_the_final_frame_draws_exactly_the_bundled_forecast` is the one at real risk — it fails if the schedule's last frame does not reveal the whole history.

- [ ] **Step 3: Amend the docstring**

In `plot.py:1937-1941`, replace the sentence about the drawn overlay:

```
        ``predict`` is ``None`` unless `predict` was set, in which case it is
        ``{'model': ..., 'params': {'t': t}, 'forecasts': [...]}`` (one
        forecast array per input dataset, in the analyzed/plotted --
        pre-center/scale -- space). Each bundled forecast has exactly `t`
        rows, matching what ``hyp.predict(xform_data, model=..., t=t)``
        returns. For a STATIC plot or ``animate='spin'`` the drawn dashed
        overlay is this same forecast with the last observed row prepended as
        a connector, so the drawn trace has `t + 1` vertices. For a
        TIME-PROGRESSING animation the drawn trace is instead the forecast
        from the history revealed at that frame, anchored on the last revealed
        observation -- which at the FINAL frame is the whole history, so the
        final frame draws exactly this bundled forecast. Default False.
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `.venv/bin/python -m pytest tests/plot/test_predict_animation.py -v`
Expected: **25 passed** (9 from Task 3 + 12 from Task 4 + 4 here).

- [ ] **Step 5: Run the WHOLE suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add hypertools/plot/plot.py tests/plot/test_predict_animation.py
git commit -m "docs(plot): pin the return_model forecast contract for animated predict="
```

---

## Task 8: CHANGELOG and docs

- [ ] **Step 1: Add to CHANGELOG.md, under the 1.1.0 heading created by the animation-core plan**

```markdown
- `predict=` now works with time-progressing animations (`animate=True`,
  `'parallel'`, `'serial'`, `'window'`) on BOTH backends. Every forecast the
  animation will draw is computed before the first frame and folded into the
  plot's centre/scale statistics, so the whole fan sits inside the cube and
  nothing is clipped or clamped; each frame is then a pure lookup, so a saved
  animation matches an interactively-played one exactly. `t` is measured in
  raw observations, not animation frames. `animate='morph'` still refuses
  `predict=` (including the per-dataset morph list form), having no time axis.
- `forecast_trail=` keeps earlier forecasts on screen as a fading fan -- the
  forecast analogue of `chemtrails=`.
- Forecast artists/traces are now tagged (`_hyp_forecast_role` on matplotlib
  artists, `meta['hyp_forecast_role']` on plotly traces) so they can be
  identified without guessing from linestyle.
```

- [ ] **Step 2: Update the animation docs page** with a short "forecasting during an animation" section: the raw-samples meaning of `t`, the precompute-and-frame-it-all contract, `forecast_trail=`, and the morph exclusion.

- [ ] **Step 3: Verify the docs build clean**

Run: `cd docs && make clean && make html 2>&1 | tail -20`
Expected: build succeeds with **0 warnings** (the RTD-parity bar the 1.0 release gate enforces).

- [ ] **Step 4: Run the WHOLE suite one final time**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass. Per the repo convention, re-run **every** check after any fix made to satisfy another check.

- [ ] **Step 5: Commit**

```bash
git add CHANGELOG.md docs/
git commit -m "docs(1.1): document animated predict= and forecast_trail="
```

---

## Decisions still needed

These came out of the review and are **not** settled by any instruction so far. Each is stated with its options; none is invented in the plan.

1. **The silent forecast drop at `plot.py:3999` (review G2).** With `hue=`/`cluster=`, `xform` is regrouped by category and `raw_forecasts` is nulled with **no warning**; the animated path inherits that behaviour verbatim (Task 4, `test_hue_regrouping_drops_forecasts_exactly_like_the_static_path`), so a user asking for `predict=` silently gets no forecast. Options: **(a)** keep the silent drop (status quo, zero risk to shipped figures); **(b)** emit a `UserWarning` naming `hue=`/`cluster=`, for both the static and animated paths; **(c)** raise for the newly-enabled animated path only, keeping static silent. The plan implements **(a)** and pins it with a test; changing it is a one-line follow-up.

2. **Further throttling beyond memoization (review G3).** Memoizing by revealed-history length caps a 900-frame, 3-dataset, 60-row animation at ≤ 177 fits instead of 2700 — measured at ~54 ms per 60-row Kalman fit, that is ~10 s of setup instead of ~146 s. But a 500-row history costs ~440 ms per fit, so a long real-world series is still minutes. Options: **(a)** memoization only (what the plan implements); **(b)** add `forecast_every=<n frames>` so the schedule samples the reveal instead of tracking it exactly, with a default to be chosen; **(c)** stride the schedule automatically once the projected fit count exceeds some ceiling. (b)/(c) both need a default value that is a product decision, so neither is invented here.

3. **`min_history` (review G4).** A 2-row history produces a degenerate flat stub (measured: every forecast step identical), drawn for the opening frames of every animation. The plan keeps `min_history=2` — matching what `hyp.predict` itself accepts — and hides the artist below it. Options: **(a)** keep 2; **(b)** raise the floor (e.g. 5 or 10) so the opening frames show nothing rather than a stub; **(c)** expose it as a `predict_min_history=` kwarg. Whether a flat stub is worse than no forecast is a taste call.

4. **Fully-revealed datasets under `order='serial'` (review G6).** Once a dataset is fully revealed, its forecast stops changing (the history stops growing), so it freezes on screen while later datasets animate. The plan does this by construction and tests it (`test_forecast_composes_with_order_serial`). Options: **(a)** freeze (implemented); **(b)** fade a finished dataset's forecast out; **(c)** hide it once the next dataset starts. Purely a visual-design call.

---

## Self-Review

**Every review finding, mapped to the task that closes it.**

| finding | closed by |
|-|-|
| **C1** existing parametrize asserts the opposite; `tests/test_predict.py` does not exist | **Task 3 Step 6** edits `test_predict_integration.py:167-178` down to `['morph', ['morph','morph']]` as an explicit step. No command anywhere in this plan names `tests/test_predict.py`. |
| **C2** static full-history overlay never suppressed | **Task 3 Step 5** gates `plot.py:4339` on `animate in (False, None, 'spin')`; **Task 3 Step 1** tests it for all four time-progressing modes, and separately pins that static and `'spin'` keep theirs (styling, alpha, label, clip). |
| **C3** `revealed_counts` is `None` for parallel; `order=` is animation-core Task 5 | **Task 2** uses `_anim_window_bounds` (`matplotlib_backend.py:319-366`) — the mechanism that exists for parallel and that `update_lines_parallel` itself calls — and `test_reveal_matches_the_library_formula_not_a_second_copy_of_it` pins it to that one implementation. **Prerequisites** lists animation-core Tasks 5 **and** 7, with the interface each supplies. |
| **C4** centre/scale invariant arithmetically impossible; forecasts fall outside the cube | **Task 2** records the transform as `DisplayTransform` instead of leaning on dead function-locals; **Task 4 Step 4** folds the whole schedule into both joint stacks, so every forecast is in `[-1, 1]` by construction (Contract 4). Per the maintainer correction there is **no clamp** — the box is fixed, not the drawing. Guards: `test_to_display_maps_every_scheduled_forecast_into_the_cube`, `test_forecast_stays_inside_the_axes_limits` (every frame, not one), `test_plotly_forecast_stays_inside_the_scene_range`. |
| **C5** `t=1` vs the documented antialias contract | **Task 4** keeps documented parity (`plot.py:1904-1908`, `:149-150`) and tests both halves: `antialias=False` → exactly `t + 1` vertices; `antialias=True` → more. |
| **C6** which array the history comes from; `t`'s unit | **Contract 1 + 2** name all three spaces with the measurements; **Task 4 Step 3** snapshots analyze space before `_interp_anim_line`; `test_t_is_measured_in_raw_samples_not_frames_or_vertices` and `test_forecast_is_anchored_near_the_drawn_head` (whose tolerance is derived from one raw step, with the discriminating comparison against final-observation anchoring) pin it. |
| **C7** list-form morph regression | **Task 3 Step 4** guards `animate == "morph" or isinstance(animate, (list, tuple))`, with the `plot.py:3653` vs `plot.py:2338` ordering in the comment; tested in both `test_predict_animation.py` and the edited `test_predict_integration.py` parametrize. |
| **C8** plotly advertised with nothing behind it | **Task 6** delivers real parity via plotly's own trail mechanism, with a final-frame numeric equality test against matplotlib. The "gate it" instruction was withdrawn by the maintainer. |
| **G1** `return_model` contract undefined | **Contract 7 + Task 7**: the value is unchanged because the final frame reveals the whole history; one docstring sentence amended; four new tests including `test_the_final_frame_draws_exactly_the_bundled_forecast`. |
| **G2** silent drop at `plot.py:3999` has no per-frame analogue | **Task 4 Step 3** extends the guard to the snapshot; `test_hue_regrouping_drops_forecasts_exactly_like_the_static_path` pins it. Whether to keep it silent is **Decision 1**. |
| **G3** per-frame refit cost | **Task 2** memoizes by revealed-history length; `test_fits_are_memoized_by_revealed_history_length` asserts a real bound (≤ 60 per dataset for a 900-frame animation) with the measured 54 ms/fit in the comment. Further throttling is **Decision 2**. |
| **G4** `min_history=2` stub; `None` unspecified | `forecast_from_history` returns `None` (documented); `test_a_dataset_with_too_little_history_hides_its_forecast` and `test_early_frames_have_no_forecast` pin the hidden-artist behaviour. The floor value is **Decision 3**. |
| **G5** ring buffer not idempotent, never reset | **Contract 6**: no buffer at all. `trail_frames(frame, ...)` is pure; `test_frames_are_idempotent` and `test_the_fan_is_a_pure_function_of_the_frame_index` drive frames out of order and compare. |
| **G6** serial 0-revealed / fully-revealed datasets | `ForecastSchedule.for_serial` maps `serial_reveal_counts` onto raw rows; below `min_history` the artist is hidden, and a finished dataset's forecast freezes. Tested by `test_serial_schedule_reveals_datasets_in_order` and `test_forecast_composes_with_order_serial`. Visual treatment is **Decision 4**. |
| **G7** `predict` returns a DataFrame | Documented in `forecast_from_history`'s Notes; `test_result_is_a_plain_ndarray_even_though_predict_returns_a_dataframe` asserts both the DataFrame input and the `np.ndarray`/`float64` output. |
| **T1** tautological opacity assertion | Replaced by `test_live_forecast_is_strictly_more_opaque_than_every_trail` (role-tagged) plus `test_trail_alpha_decreases_with_age` (sorted by `_hyp_forecast_age`). |
| **T2** preallocation vs `alpha > 0` | Emptiness, not alpha, is the unwritten signal: `_drawn()` filters on `get_visible()` **and** non-empty data. `trail_alpha`'s `0.08` floor is now irrelevant to the count. |
| **T3** vacuous cap test | All trail tests use `_drive(ani, N)`, which walks frames `0..N` sequentially. `test_an_uncapped_trail_retains_more_than_a_capped_one` proves the cap is what limits the fan. |
| **T4** wrong pass counts | Itemised per module: Task 1 → 8, Task 2 → 12, Task 3 → 9 (+17 in the edited integration file), Task 4 → 21 cumulative, Task 5 → 11, Task 6 → 8, Task 7 → 25 cumulative. |
| **T5** `_dashed()` is not forecast-specific | **Contract 5**: `_hyp_forecast_role` / `meta['hyp_forecast_role']`, set in `_draw_forecast_overlays` (Task 3 Step 3) and on every new artist/trace. `test_forecast_artists_are_not_identified_by_linestyle` plots the data with `fmt='--'` and asserts the linestyle heuristic would have over-counted. |

**Also verified and deliberately unchanged** (so it is not re-litigated): Task 1's import path, its `(t, n_dims)` return shape, its all-future-steps claim (Kalman/ARIMA/GP reproduce a unit ramp; Laplace does not, and is excluded rather than having the tolerance loosened), `ani._func(frame, *ani._args)` as the drive mechanism, `get_data_3d()` on 3-D artists, and `_validate_forecast_trail`'s routing of `-1`/`'yes'`/`2.5`.

**Placeholders.** None: every step carries runnable code, an exact command, and an expected result.

**Type consistency.** `forecast_from_history` → `(t + 1, n_dims)` float64 with a zero first row, or `None`. `ForecastSchedule.path` returns that same array or `None`; `.polyline` adds the anchor; `.to_display` rescales displacements by `2 / transform.scale` only, because a displacement is a difference of positions and the mean cancels. `DisplayTransform(mean, offset, scale)` reproduces `plot.py:4018-4031` and is asserted against it directly. `_validate_forecast_trail` returns an int consumed by `trail_alpha(age, n_retained, ...)` and `trail_frames(frame, n_retained, n_frames)`. Both backends index the schedule by **frame**, never by a re-derived row count.

**Remaining risk.** Task 4 Step 4 changes what the centre/scale statistics are computed over for **every** `predict=` plot that is also animated. A static or `'spin'` plot is untouched (`forecast_schedule is None`, so `_fc_rows` is the single pre-existing entry and the arithmetic is byte-identical to today's). The guards are the unchanged static tests — `test_forecast_vertices_stay_inside_frame`, `test_predict_adds_one_dashed_forecast_per_dataset`, `test_predict_with_spin_renders_dashed_forecast_overlay` — plus the full-suite run in Step 7. If Task 4 grows beyond one reviewable diff, split it at Step 4/Step 5: "fold the schedule into the bounding box" and "draw from the schedule" are independently testable.
