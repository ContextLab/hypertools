# Adversarial review — `2026-07-27-hypertools-1.1-forecast-animation.md`

Repo `/Users/jmanning/hypertools`, branch `dev-1.0`. All code run with `.venv/bin/python`
(Python 3.12.10, pytest 9.0.2). Baseline: `pytest tests/plot/test_predict_integration.py -q`
→ **18 passed in 3.10s**.

Verdict: **Task 1 is sound and verified. Task 2 is not implementable as written — eight
independent defects, four of them fatal. Task 3's rendering model contradicts its own tests.**

---

## CRITICAL FINDINGS

### C1. Task 2 Step 6 asserts existing tests pass unchanged; 4 of them are written to assert the exact opposite

`tests/plot/test_predict_integration.py:169-178`:

```python
@pytest.mark.parametrize('mode', [True, 'parallel', 'serial', 'window',
                                  'morph'])
def test_time_progressing_animate_and_predict_raises_not_implemented(mode):
    ...
    with pytest.raises(NotImplementedError):
        hyp.plot([a, b], predict='Kalman', animate=mode, show=False)
```

Task 2 Step 6 (plan line 333-336) says:

> Run: `.venv/bin/python -m pytest tests/plot/test_predict_integration.py tests/test_predict.py -q`
> Expected: **all pass, unchanged.** These cover the paths this task must not disturb.

After Task 2 Step 3, `mode` in `True`/`'parallel'`/`'serial'`/`'window'` no longer raises →
**4 of 5 parametrizations fail.** The File Structure table (plan line 39) labels this file
"modify (regression guard only)" but **no step in the plan ever edits it**, and Step 6
explicitly forbids the edit by demanding it pass unchanged.

Additionally the Step 6 command names a file that does not exist:

```
$ ls tests/test_predict.py
ls: tests/test_predict.py: No such file or directory
```

pytest exits with `ERROR: file or directory not found: tests/test_predict.py`. Task 2 Step 6
cannot be run as written.

### C2. The static full-history overlay is never suppressed — every Task 2 test that indexes `_dashed(ax)[0]` picks the wrong artist

`plot.py:4339-4341` sits inside the shared matplotlib branch, after `_draw()` (which builds
both static figures *and* animations), with **no `animate` guard**:

```python
            if raw_forecasts is not None:
                _forecast_artists = _draw_forecast_overlays(
                    ax, raw_forecasts, antialias=antialias)
```

`raw_forecasts` is built unconditionally at `plot.py:2977-2988` whenever `predict is not None`.
So removing the refusal at `plot.py:2346` — without gating line 4339 — makes every
time-progressing animation draw **both** a static full-history dashed overlay *and* the new
per-frame artist. The plan never mentions touching line 4339.

Measured today (`animate='spin'`, n=3, `predict='Kalman'`, t=3):

```
n ax.lines: 6
  line0: ls='-'  alpha=None label='_child0'      npts=1
  line1: ls='-'  alpha=None label='_child1'      npts=1
  line2: ls='-'  alpha=None label='_child2'      npts=1
  line3: ls='--' alpha=0.6  label='_nolegend_'   npts=901
  line4: ls='--' alpha=0.6  label='_nolegend_'   npts=901
  line5: ls='--' alpha=0.6  label='_nolegend_'   npts=901
```

Setup-time overlays land **first** in `ax.lines`. Consequences:

- `test_without_trail_only_the_live_forecast_is_visible` asserts `len(visible) == 1` → gets 2.
- `test_forecast_head_tracks_the_animation` reads `_dashed(ax)[0]` → the **static** overlay,
  which never moves → fails with its own message `'forecast head did not move'`.
- `test_forecast_starts_exactly_at_the_drawn_head` reads `_dashed(ax)[0]` → the static overlay
  is anchored on the *final* observation, not the frame-N head → fails at `atol=1e-6`.

### C3. `FrameContext.revealed_counts` is `None` for exactly the mode every Task 2 test uses

Task 2 Step 4 (plan line 325) states the invariant:

> Use `FrameContext.revealed_counts` for the history slice, never a locally re-derived reveal formula.

But the prerequisite plan `2026-07-26-hypertools-1.1-animation-core.md` defines it as:

```
    revealed_counts : list of int or None
        Number of points currently drawn for each dataset. ``None`` for
        parallel animations.
```
```python
    revealed_counts: Optional[List[int]] = None
```

Every Task 2 test uses `animate=True` — the parallel style (verified: updater is
`update_lines_parallel`). The mandated invariant is **unimplementable for the tested mode**.
Animation-core supplies only `serial_reveal_counts(lengths, num, total_frames)` — serial only.
Plan line 570's Self-Review claim, *"`FrameContext.revealed_counts` is consumed here exactly as
defined in the animation-core plan,"* is false.

Compounding: `order=` is animation-core **Task 4**, not Task 6, yet
`test_predict_composes_with_serial_order` passes `order='serial'`. Verified today:

```python
sig = inspect.signature(hyp.plot); 'order' in sig.parameters  # False
                                   'on_frame' in sig.parameters  # False
```

→ `TypeError: plot() got an unexpected keyword argument 'order'`. The stated prerequisite
(Task 6 only, plan line 26) is incomplete.

### C4. The center/scale invariant is arithmetically impossible after setup — this is the "state recorded along the way" defect

Global Constraint (plan line 20) and Task 2 Step 4 both require forecasts to "pass through the
same center/scale transform as the data (`plot.py:4002-4032`)" and to be "include[d] ... in the
center/scale statistics".

`plot.py:4015-4032` runs **once**, before any figure exists, and is **destructive**:

```python
    if raw_forecasts is not None:
        _joint = np.vstack([np.vstack(xform), np.vstack(raw_forecasts)])
        _mean = np.mean(_joint, 0)
        xform = [xi - _mean for xi in xform]
        ...
        _m1 = np.min(_joint); _m2 = np.max(_joint - _m1) or 1.0
        _rescale = lambda a: 2 * (np.divide(a - _m1, _m2)) - 1
        xform = [_rescale(xi) for xi in xform]
```

`_mean`/`_m1`/`_m2`/`_rescale` are function-locals that are gone by frame time, and `xform` has
already been overwritten in place. A per-frame forecast computed "in the already-reduced
plotting space" (plan line 7) is **already in display coordinates**, so re-applying `_rescale`
double-transforms it; recomputing the statistics to *include* it would require re-transforming
every dataset and every artist mid-animation, producing a visible jump. The two invariants in
Step 4 are mutually contradictory and neither is achievable.

The failure is not hypothetical. Animated axes limits are hard-set **before** `FuncAnimation` is
constructed, to exactly `[-1, 1]`:

```
matplotlib_backend.py:1785   cube_scale_anim = 1
matplotlib_backend.py:1888-1890
        ax.set_xlim3d([-cube_scale_anim, cube_scale_anim])
        ax.set_ylim3d([-cube_scale_anim, cube_scale_anim])
        ax.set_zlim3d([-cube_scale_anim, cube_scale_anim])
```

Measured (`ax.get_xlim3d()` on a live `animate=True` figure): `(-1.0, 1.0)`.

Kalman forecasts (t=5) from *partial* display-space history of a 60×4 random walk:

```
  frac=0.2   n_hist=12  fc min=+0.182 max=+0.999  OUTSIDE_CUBE=False
  frac=0.3   n_hist=18  fc min=-0.823 max=+0.637  OUTSIDE_CUBE=False
  frac=0.4   n_hist=24  fc min=-0.535 max=+2.259  OUTSIDE_CUBE=True
  frac=0.5   n_hist=30  fc min=+0.142 max=+0.676  OUTSIDE_CUBE=False
  ...
1/7 partial-history forecasts fall OUTSIDE the fixed [-1,1] animated cube
```

`test_forecast_stays_inside_the_axes_limits` (plan line 252) genuinely fails. The plan's own
"Known risk 2" names this test as the guard — the guard is the thing that breaks.

### C5. `test_t_equals_one_next_step_forecast` contradicts the documented antialias contract

Plan line 264-269 asserts `fc.shape[1] == 2` ("a t=1 forecast is the head plus one step").
Measured on today's drawn forecasts:

```
t=1: drawn forecast vertices (antialias=True, default) = 900 ; antialias=False -> 2
t=3: drawn forecast vertices (antialias=True, default) = 901 ; antialias=False -> 4
t=5: drawn forecast vertices (antialias=True, default) = 901 ; antialias=False -> 6
```

`plot.py:1904-1908` documents this as contract: *"Forecast overlays drawn by `predict=` are
smoothed the same way."* `_draw_forecast_overlays` (`plot.py:149-150`) applies
`_interp_static_line(fc)`, and the existing test `test_predict_with_spin_renders_dashed_forecast_overlay`
(`tests/plot/test_predict_integration.py:196`) pins it:

```python
        assert len(fc.get_xdata()) > t + 1  # smoothed (densified) beyond raw t+1
```

So the per-frame path must either drop antialias (breaking documented static/animated parity and
looking different from the `spin` overlay) or fail the plan's own test. Undecided in the plan.

### C6. Animations animate the *interpolated* array, so a per-frame `t` is 15× shorter than the user asked for

Measured on a 60-row input:

```
xform_data (analyze space) shape: [(60, 3)]
drawn head-line rows at frame 8: (904, 3)   (raw rows = 60)
interp factor ~15.1x  ->  t=3 "steps" on the DRAWN history = 0.20 raw samples
```

`update_lines_parallel` slices the antialiased array (`_aa_window`, `matplotlib_backend.py:1205`).
The plan says the forecast is computed "from the history revealed so far, in the already-reduced
plotting space" but never says *which* array. If the interpolated one, `t=3` forecasts 0.2 real
samples — a visually invisible stub, and a silent redefinition of the documented `t` unit
("Forecast horizon in steps", plan line 351). If the raw one, the forecast cannot be anchored on
the interpolated drawn head to `atol=1e-6`, so `test_forecast_starts_exactly_at_the_drawn_head`
fails. `pre_interp_lengths` (`plot.py:2998`) exists precisely because these two lengths diverge;
the plan never reconciles them.

### C7. Narrowing to `animate == "morph"` re-opens list-form morph, a regression

The replacement (plan line 310) is `if predict is not None and animate == "morph":`. But
`_resolve_animate_mode` — the only thing that maps a per-dataset list onto `'morph'` — is called
at `plot.py:3653`, roughly 1300 lines **after** the refusal at `plot.py:2338-2354`. At line 2338
`animate` is still the raw list, so `animate == "morph"` is `False`.

Today's behavior (verified) with `animate=['morph','morph'] + predict='Kalman'`:

```
NotImplementedError: predict= is only supported with static plots and with animate='spin' ...
contains 'morph': True
```

It raises today only via the `animate and animate != "spin"` truthiness of a non-empty list.
After the change it would be **silently accepted** into the per-frame forecast path for a morph —
the one mode the plan itself declares incoherent (`plot.py:2343` already documents "per-dataset
morph lists" as a morph form). `test_morph_still_refuses_predict` only covers scalar
`animate='morph'` and would not catch it.

### C8. Plotly gets the advertised support with no implementation behind it

The refusal at `plot.py:2346` is backend-agnostic, so removing it enables
`backend='plotly', animate=True, predict=` too. But `plotly_backend.py:901` draws forecasts
unconditionally and statically:

```python
    if forecasts is not None:
        for i, arr in enumerate(data):
```

and `_add_animation` (`plotly_backend.py:2517-2525`) takes no forecast argument at all. The plan
modifies only `hypertools/plot/matplotlib_backend.py` (File Structure, plan line 36). Result: a
newly-accepted call that silently renders a frozen full-history overlay on an animated plotly
figure. Existing parity test `test_plotly_predict_trace_parity`
(`tests/plot/test_predict_integration.py:226`) covers only the static case.

---

## IMPORTANT GAPS

### G1. `return_model=True` / `bundle_forecasts` contract left undefined (the reviewer's headline question)

`plot.py:1920-1941` and the `Returns` block at `plot.py:1955` document:

> ``predict`` is ``None`` unless `predict` was set, in which case it is
> ``{'model': ..., 'params': {'t': t}, 'forecasts': [...]}`` (one forecast array per input
> dataset, in the analyzed/plotted -- pre-center/scale -- space). Each bundled forecast has
> exactly `t` rows, matching what ``hyp.predict(xform_data, model=..., t=t)`` returns; the DRAWN
> dashed overlay additionally prepends the last observed row as a connector, so the drawn trace
> has `t + 1` vertices.

Under the plan, for a time-progressing animation:

- The second sentence becomes false — the drawn trace is a head-anchored per-frame path, not
  `bundle_forecasts` + a seam row.
- `bundle_forecasts` is still computed **once from the full history** at `plot.py:2979-2988`
  (`_predictor(xform, model=predict, t=t)`), which is not the forecast at any frame. So the
  bundle describes a forecast that is never drawn, while the library also pays for 240×N extra
  fits. The plan never states what `forecasts` should hold for an animated forecast
  (per-frame list? final frame? full-history?) — the reviewer's question is unanswered.
- Existing test `test_predict_with_spin_return_model_bundle_carries_forecasts`
  (`tests/plot/test_predict_integration.py:209-222`) pins `shape[0] == t` for `spin` only; no new
  test in the plan touches `return_model=` at all. **Claim not covered by any test.**

`bundle_forecasts` is not meaningful when the forecast is recomputed 240 times, and the plan
should either (a) keep it as the documented full-history forecast and say so explicitly in the
docstring, or (b) redefine it as the final-frame forecast — but it must pick one and test it.

### G2. The silent-drop guard at `plot.py:3999` has no per-frame analogue

```python
    if raw_forecasts is not None and len(raw_forecasts) != len(xform):
        raw_forecasts = None
```

No warning is emitted. With `hue=`/`cluster=`, `xform` is regrouped by category and forecasts are
silently dropped. The per-frame path has no equivalent guard, and forecasting *per drawn hue
group* is semantically wrong — a hue group is a non-contiguous subset of rows, not a time series.
No test in the plan or in `tests/plot/test_predict_integration.py` covers `hue`/`cluster` +
`predict` (`grep -n hue tests/plot/test_predict_integration.py` → no matches).

### G3. Per-frame refitting cost is prohibitive, and "Known risk 1" understates it

Measured (500-row history, t=3):

```
Kalman: 447 ms/fit -> 240 frames x 3 datasets = 322s
ARIMA:  181 ms/fit -> 240 frames x 3 datasets = 130s
```

The plan proposes `forecast_every=` only as a conditional follow-up ("If Task 2 makes animations
sluggish"). At 5+ minutes for a standard animation it is not optional, and no test asserts any
time bound. Task 2's own tests use only 8 frames (`duration=2, frame_rate=4`), so they will never
surface this.

### G4. `min_history=2` yields degenerate forecasts, and `None` handling is unspecified

Measured with a 2-row history:

```
2-row out: [[ 0.  0.  0.]
            [-0.14874371 -0.14874371 -0.14874371]
            [-0.14874371 -0.14874371 -0.14874371]
            [-0.14874371 -0.14874371 -0.14874371]]
```

Every forecast step is identical — a flat stub drawn for the opening frames of every animation.
`hypertools/predict/predict.py:92` raises `ValueError: input has no observations (got an array of
shape (0, 3))` for empty history, so the per-frame updater must handle `forecast_from_history(...)
is None` (hide the artist / set empty data). The plan never specifies this.

### G5. Ring-buffer trail state is not idempotent and is never reset

Task 3 Step 4 says "Each frame, shift the ring buffer". `FuncAnimation` updaters are re-invoked
non-monotonically — `ani.save()` and `to_jshtml()` replay from frame 0, and the plan's own tests
call `_func(4)`, `_func(8)`, `_func(12)` and `_func(2)`, `_func(12)` out of sequence. No reset
step is defined, so a saved GIF and an interactively-played animation would show different fans.

### G6. Serial mode reveals one dataset at a time; unstarted/finished datasets are unspecified

Measured at frame 4 of `animate='serial'` with 3 datasets:

```
=== n=3 animate='serial': 3 lines, updater=update_lines_serial
  line0: ls='-' shape=(3, 904)   <- fully drawn
  line1: ls='-' shape=(3, 517)   <- mid-reveal
  line2: ... IndexError: index -1 is out of bounds for axis 1 with size 0   <- ZERO points
```

The plan gives no rule for the forecast artist of a dataset with 0 revealed points, or of one
already fully revealed (does its forecast freeze, or keep re-fitting an unchanging history?).

### G7. Undocumented API assumption: `predict` returns a DataFrame, not an ndarray

`forecast_from_history`'s docstring (plan line 145-148) says "Returns ... numpy.ndarray". The
underlying call returns a pandas object:

```
type(_predict(ramp, model='Kalman', t=3)): <class 'pandas.DataFrame'>
```

The implementation's `np.asarray(..., dtype=float)` makes this harmless *for plain 2-D ndarray
input*, but the plan never states the assumption, and the DataFrame index is silently discarded —
which matters for the MultiIndex work in the sibling `2026-07-28-hypertools-1.1-multiindex.md`.

---

## TEST WEAKNESSES

### T1. `test_live_forecast_is_the_most_opaque`'s first assertion is a tautology

```python
    assert max(alphas) == alphas[0] or max(alphas) >= sorted(alphas)[-1]
```

`sorted(alphas)[-1]` **is** `max(alphas)`, so the right-hand disjunct is `max >= max` → always
`True`. The assertion can never fail and verifies nothing about the live forecast being most
opaque. Only the second line (`min(alphas) < max(alphas)`) tests anything, and it merely
requires two distinct alphas somewhere.

### T2. `test_trail_accumulates_past_forecasts` contradicts Task 3 Step 4's preallocation

Step 4 preallocates `n_retained` dashed artists **at setup**. The test counts artists with
`get_alpha() > 0`. But `trail_alpha` never returns 0:

```python
    decay = 1.0 - (age / max(1, n_retained))     # age == n_retained -> decay = 0
    return max(floor, floor + (live_alpha - floor) * decay)   # -> 0.08
```

so every preallocated artist is already `alpha > 0` at frame 0 and `late > early` cannot hold.
The plan defines no "not yet written" alpha state. Either preallocation or this test must change.

### T3. `test_trail_is_capped_by_an_integer` passes vacuously

It calls `ani._func(20, *ani._args)` **once**. A per-frame-shift ring buffer will contain at most
one entry after a single call, so `len(visible) <= 5` holds regardless of whether capping works.
Same defect in `test_trail_accumulates_past_forecasts` (frames 2 then 12 → 2 entries, not 12).

### T4. Stated pass counts are wrong

Task 2 Step 5 (plan line 331) says "Expected: 13 passed". Task 1 creates 5 tests; Task 2 appends
9 → **14**. (Task 1's "5 passed" and Task 3's "8 passed" are correct — 5 named + 3 params.)

### T5. `_dashed()` linestyle heuristic is not forecast-specific

```python
def _dashed(ax):
    return [ln for ln in ax.lines if ln.get_linestyle() not in ('-', 'solid')]
```

Any user data drawn with `'--'`/`':'`/`linestyle='--'` is classified as a forecast. The existing
suite uses a stronger discriminator where it can (`l.get_label() == '_nolegend_'`,
`tests/plot/test_predict_integration.py:153`), and `_draw_forecast_overlays` sets
`label='_nolegend_'` + `alpha=0.6` (`plot.py:156`) — both more reliable than linestyle. No test
guards the false-positive case.

---

## VERIFIED CORRECT (no defect — recorded so it is not re-litigated)

Task 1's central claim holds. All of the following was run:

- `from hypertools.predict.predict import predict as _predict` — **import path valid**
  (`<function predict at 0x12018d580>`).
- `predict(<raw 2-D ndarray>, model='Kalman', t=3)` accepts the raw array and returns a
  `(t, n_dims)` result — `(3, 3)` for a `(30, 3)` input.
- **All `t` returned rows are future steps.** On a unit ramp (last observation `29.0`):

  ```
  [[29.99999986 29.99999986 29.99999986]
   [31.00000008 31.00000008 31.00000008]
   [31.99999997 31.99999997 31.99999997]]
  ```

  `forecast[0] ≈ 30.0 = last_obs + 1`, confirming that anchoring on `f[0]` (as the market example
  does) discards a whole step. Holds for Kalman, ARIMA, and GP:

  ```
  Kalman   first_row[0]=+30.0000  displacement_steps=[1. 1. 1.]  unit_ramp_ok=True
  ARIMA    first_row[0]=+30.0000  displacement_steps=[1. 1. 1.]  unit_ramp_ok=True
  GP       first_row[0]=+30.0000  displacement_steps=[1. 1. 1.]  unit_ramp_ok=True
  Laplace  first_row[0]=+30.0000  displacement_steps=[1. 1.328 1.909]  unit_ramp_ok=False
  ```

- **`test_displacement_is_anchored_on_the_last_observation` PASSES.** Kalman reproduces the unit
  ramp far tighter than `atol=0.25`: measured steps `[0.99999986 1.00000022 0.99999988]`.
  (Caveat: `Laplace` would fail the same assertion — the test covers only Kalman, so the plan
  should either restrict the claim to Kalman/ARIMA/GP or parametrize.)

- All 5 Task 1 tests pass verbatim against the plan's `forecast_from_history` implementation:

  ```
  test_returns_none_below_min_history PASSED
  test_shape_is_t_plus_one_and_starts_at_the_origin PASSED
  test_displacement_is_anchored_on_the_last_observation PASSED
  test_horizon_of_one_is_supported PASSED
  test_history_must_be_two_dimensional PASSED
  ============================== 6 passed in 2.04s ===============================
  ```

- `ani._func(frame, *ani._args)` works: `hyp.plot(...)` unpacks to a plain
  `matplotlib.animation.FuncAnimation`; `_func` and a 6-tuple `_args` both exist, and
  `ani._func(1, *ani._args)` runs clean. `get_data_3d()` is available on the 3-D line artists.
- `test_spin_keeps_its_existing_static_overlay` **would pass**: `update_lines_spin`
  (`matplotlib_backend.py:1229`) never touches the forecast artists, and the `spin` overlay is
  drawn once at `plot.py:4339-4350`.
- `test_morph_still_refuses_predict` **passes today** for scalar `animate='morph'` — the current
  message contains `'morph'` (see C7 for the list-form hole).
- `_validate_forecast_trail` correctly routes all three `test_invalid_forecast_trail_raises`
  parameters (`-1` → ValueError, `'yes'` → TypeError, `2.5` → TypeError), and both `match=`
  regexes in Task 3 are valid.

---

## Recommended minimum before this plan can be executed

1. Rewrite Task 2 Step 6: `tests/plot/test_predict_integration.py:169-178` must be **edited** in
   Task 2 (drop `True`/`'parallel'`/`'serial'`/`'window'` from the parametrize, keep `'morph'`
   plus a new list-form-morph case), and drop the non-existent `tests/test_predict.py`.
2. Add an explicit step gating `plot.py:4339` on `animate in (False, 'spin')`.
3. Replace the `revealed_counts` invariant with something that exists for parallel animations
   (or extend animation-core Task 6 to populate `revealed_counts` for parallel too), and add
   animation-core **Task 4** to the prerequisite list.
4. Decide and document the display-space contract: state that the forecast is computed in and
   emitted directly into the already-rescaled display space, drop the impossible "same
   center/scale transform" requirement, and replace
   `test_forecast_stays_inside_the_axes_limits` with either an explicit clamp or a documented
   allowance that a forecast may extend past the cube during an animation.
5. Decide `t`'s unit (raw samples vs. interpolated samples) and the antialias treatment of a
   per-frame forecast; reconcile `test_t_equals_one_next_step_forecast` with `plot.py:1904-1908`.
6. Specify what `return_model=True`'s `predict.forecasts` holds for an animated forecast, update
   `plot.py:1920-1941` + `1955`, and add a test.
7. Gate or implement the plotly path.
8. Fix T1/T2/T3 (tautology, preallocation-vs-alpha, single-frame trail tests) and make the trail
   tests drive frames sequentially.
