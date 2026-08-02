# Plan 3 closure audit — `2026-07-27-hypertools-1.1-forecast-animation.md` (v3)

**Date:** 2026-08-01
**Repo:** `/Users/jmanning/hypertools`, branch `dev-1.0`
**Audit base commit:** `065c841e` (worktree HEAD moved to `37f5b8f8` mid-audit; `git diff --name-only 065c841e HEAD` over the plan, `hypertools/plot/`, `tests/plot/` and `tests/test_backend_window_parity.py` is **empty** — every audited file is byte-identical at both commits, so no result is invalidated).
**Python:** `/Users/jmanning/hypertools/.venv/bin/python` for every command below.
**Scratch:** `/private/tmp/claude-501/-Users-jmanning-hypertools/7e6531b3-066a-4ce2-b1f6-7c07c5e87b15/scratchpad`

The plan file was **not modified**. The only file created in the main worktree is this report.

---

## Check 1 — Execute Task 4's prescribed test and updater blocks verbatim

### 1.1 Verbatim extraction

Every block was cut out of the plan by line range, with the fences verified first (`sed -n '961p;1056p;1190p;1475p;2187p;2245p;1572p;1605p'` → all ```` ```python ````/```` ``` ````):

| block | plan lines | scratch file | lines |
|-|-|-|-|
| Task 3 test header + 6 test defs | 962–1055 | `verbatim/task3_block.py` | 94 |
| **Task 4 test block** | **1191–1474** | `verbatim/task4_block.py` | 284 |
| Task 7 test block | 2188–2244 | `verbatim/task7_block.py` | 57 |
| **Task 4 Step 5 `_update_forecasts`** | **1573–1604** | `verbatim/task4_updater_block.py` | 32 |

### 1.2 (a) Syntactic validity

```
$ .venv/bin/python -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" <each file>
OK   test_predict_animation_thru_t4.py
OK   test_predict_animation_thru_t7.py
OK   test_task4_block_only.py
OK   task3_block.py
OK   task4_block.py
OK   task7_block.py
```

The updater block is a 16-space-indented fragment of `plot()`; it was `textwrap.dedent`ed and `compile`d — parses clean. **PASS.**

### 1.3 Actually running the Task 4 tests against real hypertools

```
$ .venv/bin/python -m pytest $S/collect/test_task4_block_only.py -q
...
E  NotImplementedError: predict= is only supported with static plots and with
   animate='spin' (which just rotates the camera around the static forecast
   overlay); it is not yet supported with animate=True, which reveals/appends
   data over time. ...
hypertools/plot/plot.py:2749: NotImplementedError
18 failed in 2.82s
```

**This is the correct and expected result.** All 18 fail at the *same* single cause — the shipped refusal at `hypertools/plot/plot.py:2749`, which **Task 3 Step 4** removes and which is upstream of everything Task 4 adds. No test fails for an unrelated reason (no `NameError`, no `TypeError`, no signature mismatch, no fixture error). Library code Plan 3 has not yet built and that these tests need: `hypertools/plot/forecast.py` (`ForecastSchedule`, `DisplayTransform`, `revealed_raw_counts`, `forecast_from_history` — Tasks 1–2), `FrameHooks.add_internal` (Task 0), the `_hyp_forecast_role` tag (Task 3 Step 3 / Task 4 Step 5), and the narrowed refusal (Task 3 Step 4).

### 1.4 Actually running the `_update_forecasts` updater block

Rather than stop at static analysis, the updater was **executed for real**. Build:

* `hypertools/plot/forecast.py` assembled verbatim from Task 1 (plan 457–531) + Task 2 (plan 768–933), only the three intra-package relative imports rewritten to absolute (`..predict.predict`, `.trails`, `.matplotlib_backend`).
* `hypertools/plot/animation_context.py` patched with Task 0 (in the disposable worktree, see Check 3), giving a real `FrameHooks.add_internal`.
* Real `matplotlib` `Line3D`/`Line2D` artists on a real `Agg` figure, a real `ForecastSchedule` over 2×(60, 3) random-walk histories with **real Kalman fits** (`n_fits=14`), and the real `hypertools.plot.plot._interp_static_line`.
* The updater block itself `exec`ed **verbatim** (dedented only).

```
$ .venv/bin/python $S/live/run_updater.py $S/live $S/verbatim/task4_updater_block.py
hypertools FrameHooks has add_internal: True
schedule: n_fits=14, n_frames=8, n_datasets=2
  ndims=3 antialias=False: f0 vis=False shape=(3, 0) | f4 vis=True shape=(3, 4)   | f7 shape=(3, 4)   | idempotent(4)=True idempotent(0)=True moved=True
  ndims=3 antialias=True : f0 vis=False shape=(3, 0) | f4 vis=True shape=(3, 901) | f7 shape=(3, 901) | idempotent(4)=True idempotent(0)=True moved=True
  ndims=2 antialias=False: f0 vis=False shape=(2, 0) | f4 vis=True shape=(2, 4)   | f7 shape=(2, 4)   | idempotent(4)=True idempotent(0)=True moved=True
  ndims=2 antialias=True : f0 vis=False shape=(2, 0) | f4 vis=True shape=(2, 901) | f7 shape=(2, 901) | idempotent(4)=True idempotent(0)=True moved=True
  ndims=1 antialias=False: f0 vis=False shape=(2, 0) | f4 vis=True shape=(2, 4)   | f7 shape=(2, 4)   | idempotent(4)=True idempotent(0)=True moved=True
  ndims=1 antialias=True : f0 vis=False shape=(2, 0) | f4 vis=True shape=(2, 901) | f7 shape=(2, 901) | idempotent(4)=True idempotent(0)=True moved=True
  ordering frame 4: user callback saw current frame = True
  ordering frame 7: user callback saw current frame = True
```

All three `_ndims` branches (`>= 3` / `== 2` / else) execute without error; the frame-0 "no forecast" branch hides the artist with empty data; frames replayed out of order (`0, 4, 7, 4, 0`) are byte-identical (**Contract 6 holds**); `antialias=True` densifies 4 → 901 vertices exactly as the static overlay does; and a user `on_frame=` callback registered *after* the internal updater still observes the **current** frame (the v2 Fatal is genuinely fixed).

v2's exact mistake is confirmed absent: the block is `def _update_forecasts(ctx, ...)` reading `ctx.frame` (plan line 1573, 1578), never `def _update_forecasts(frame, ...)`.

### 1.5 (b)+(c) Symbol-by-symbol audit against real source

Every library symbol either exists today with the exact signature used, or is defined by an earlier task in this plan with the exact signature used.

**Exists today — signature verified against real source:**

| symbol | real definition | as used by Task 4 | verdict |
|-|-|-|-|
| `hyp.plot(predict=, t=, animate=, duration=, frame_rate=, show=, on_frame=, order=, ndims=, antialias=, hue=, return_model=, morph_samples=)` | `hypertools/plot/plot.py` | all 14 kwargs used by the block | **all present** (`inspect.signature`); `on_frame` default `None`, `order` default `None` — animation-core Tasks 5+7 confirmed landed |
| `hyp.predict(data, model='Kalman', t=10, ...)` | `hypertools/predict/predict.py` | `hyp.predict(arr, model='Kalman', t=4)` (Task 7 block) | match |
| `FrameContext.frame`, `.figure` | `hypertools/plot/animation_context.py:202,204` | `ctx.frame`, `_ax(ctx.figure)` | match |
| `_interp_static_line(arr)` | `hypertools/plot/plot.py:282` | `_interp_static_line(pts)` | match; module-level in `plot.py`, so in scope inside `plot()` |
| `_display_ndims` | `hypertools/plot/plot.py:3310` (`= ndims if (ndims and ndims < 3) else 3`) | bound as `_ndims=_display_ndims` | correct; the plan's warning against binding raw `ndims` is sound (`ndims` default is `3`, but an explicit `ndims=None` is accepted and would make `_ndims >= 3` raise `TypeError`) |
| `_frame_hooks` | `hypertools/plot/plot.py:4740` (`= FrameHooks(...)`) | `_frame_hooks.add_internal(...)` | name correct; threaded to `_draw` at `:4811`/`:4897`, dispatched at `:5004`, so it is live at the Step 5 insertion point |
| `anim_window_bounds(num, total_frames, n_points, window_frames)` | `hypertools/plot/trails.py:24` | `anim_window_bounds(num, total_frames, n_grid, 0)` | match |
| `serial_reveal_counts(lengths, num, total_frames)` | `hypertools/plot/matplotlib_backend.py:399` | `serial_reveal_counts(list(grid_lengths), f, n_frames)` | match |
| `ani._func` / `ani._args` | matplotlib `FuncAnimation` | `ani._func(6, *ani._args)` | verified live (`hyp.plot(..., animate=True)` returns a real `FuncAnimation`; call succeeds) |
| `return_model=True` bundle keys | verified live | `out['animation'] / ['fig'] / ['xform_data'] / ['predict']['model'] / ['params'] / ['forecasts']` | all present; measured `params == {'t': 4}`, `forecasts[0].shape == (4, 3)` — exactly what Task 7's block asserts |

**Defined by an earlier task in this plan — signature verified against the plan's own prescribed code:**

| symbol | prescribed at | as used by Task 4 | verdict |
|-|-|-|-|
| `FrameHooks.add_internal(updater) -> self` | Task 0, plan:297 | `_frame_hooks.add_internal(_update_forecasts)` | match (executed, Check 3) |
| `ForecastSchedule.for_parallel(histories, grid_lengths, model, t, n_frames, min_history=…)` | Task 2, plan:853 | `_builder(analyze_histories, _grid_lengths, model=predict, t=t, n_frames=_n_frames)` | match |
| `ForecastSchedule.for_serial(...)` (same signature) | Task 2, plan:861 | same call site | match |
| `ForecastSchedule.stacked_paths()` | Task 2, plan:903 | `forecast_schedule.stacked_paths()` | match |
| `ForecastSchedule.to_display(transform)` | Task 2, plan:918 | `forecast_schedule.to_display(DisplayTransform(_mean, _m1, _m2))` | match |
| `ForecastSchedule.polyline(dataset, frame)` | Task 2, plan:896 | `_sched.polyline(i, ctx.frame)` | match |
| `DisplayTransform(mean, offset, scale)` | Task 2, plan:804 | `DisplayTransform(_mean, _m1, _m2)` | match |
| `_hyp_forecast_role` tag | Task 3 Step 3 (`'static'`), Task 4 Step 5 (`'live'`) | `_forecasts(ax, role=…)` helper | match |

No reference to a non-existent `ForecastDisplay` anywhere (`grep -n ForecastDisplay` → only the plan's own line 939 explicitly denying such a class exists).

**Insertion-point anchors — every line number the plan cites was checked against real source:**

| plan claim | real source | verdict |
|-|-|-|
| Step 3: `raw_forecasts`/`bundle_forecasts` built at `plot.py:3391-3402` | `plot.py:3391` `raw_forecasts = None` … `:3402` closing the `zip(xform, _fc)` comprehension | **exact** |
| Step 3: correspondence guard at `plot.py:4552` | `plot.py:4552` `if raw_forecasts is not None and len(raw_forecasts) != len(xform):` / `:4553` `raw_forecasts = None` | **exact** |
| Step 4: centre/scale block begins `plot.py:4555`, arithmetic at `:4568-4585` | `plot.py:4555` is the `# center + scale.` comment; `:4568` `if raw_forecasts is not None:` opens the block the plan rewrites | **exact** |
| Step 5: after `_draw(...)` returns, `plot.py:4858-4898` | `plot.py:4858` `fig, ax, data, line_ani = _draw(` … `:4898` `)` | **exact** |
| Step 5 / Task 3 Step 5: static overlay at `plot.py:4907` | `plot.py:4907` `if raw_forecasts is not None:` / `:4908` `_forecast_artists = _draw_forecast_overlays(` | **exact** |
| `_frame_hooks` local at `plot.py:4740` | `plot.py:4740` `_frame_hooks = FrameHooks([on_frame] if on_frame is not None else [])` | **exact** |
| `_n_frames = max(2, int(round(frame_rate * duration)))` | `plot.py:4477` uses that **identical** expression; `xform` is resampled at `:4478` — *before* `:4555`, so Step 4's `_grid_lengths = [len(xi) for xi in xform]` really is the frame-grid length | **consistent** |
| `_resolve_animate_mode` at `plot.py:4158` rebinds `animate` **and** `order` | `plot.py:4158` `animate, morph_tags, order = _resolve_animate_mode(...)`; `plot.py:738-741` folds `order='serial'` into `mode='serial'` for `mode in (True, 'parallel')` | Step 4's `(animate == 'serial' or order == 'serial')` is correct (belt-and-braces: the first disjunct already fires for `animate=True, order='serial'`) |

**Test-helper sanity, measured live:**

* `_solid(ax)` picks the data line, not a frame artist — a 3-D animated `n=1` plot has `len(ax.lines) == 1` (the cube frame is not in `ax.lines`). `_solid(ax)[0]` has 9 vertices at frame 8 of 16 with `antialias=False`.
* `_plot_ax(fig)`'s fallback is genuinely needed: `ndims=2` produces a figure with **no** zaxis-bearing axes, so the Task 3 `_ax` helper would `IndexError` — Task 4 correctly defines `_plot_ax` for the 2-D parametrization.
* `test_forecast_artists_are_not_identified_by_linestyle`: `fmt='--'` with 3 datasets gives 3 dashed data lines today; +3 live forecast artists → `6 > 3`, so the assertion has real bite.
* Every `ani._func(N, …)` index is in range for its `duration`/`frame_rate` (frame 6 of 8; frames 12/15 and `range(16)` of 16).

### 1.6 Mismatches found

**M1 (MEDIUM — prescription gap, not a contradiction).** `_live_forecast_artists` is referenced at **plan line 1574** as a closure default:

```python
                def _update_forecasts(ctx, _sched=forecast_schedule,
                                      _artists=_live_forecast_artists,   # <- line 1574
```

`grep -n "_live_forecast_artists" <plan>` returns **exactly one hit — line 1574**. No prescribed code block anywhere in the plan creates or binds that list. Task 4 Step 5 specifies the artists only in prose (plan line 1570): *"create one dashed artist per dataset in that dataset's colour with `alpha=0.6`, `label='_nolegend_'`, `set_clip_on(False)` and `_hyp_forecast_role = 'live'` — the same styling `_draw_forecast_overlays` applies (`plot.py:168-171`)"*. The prose is complete enough to implement from (count, linestyle, colour source, alpha, label, clip, role tag), and `hypertools/plot/plot.py:137-180` is a working template, so this is a **gap the implementer must close**, not a defect that makes the plan wrong. It is the only symbol in either block without a prescribed definition.

**M2 (LOW — cosmetic).** `test_forecast_is_anchored_near_the_drawn_head`'s docstring (plan 1328–1331) quotes the measurement *"60 raw rows -> 8 grid rows -> 904 drawn vertices at duration=2/frame_rate=4"*, but the test itself runs `duration=4, frame_rate=4` (16 grid rows). The assertion is derived from the drawn data at runtime (`one_grid_step = max |diff|`), so the number in the prose does not affect correctness — only the explanatory arithmetic ("59/15 ≈ 3.9 raw steps", which *is* the duration=4 figure) is internally inconsistent with the sentence above it.

**M3 (LOW — redundant, harmless).** Step 4 re-computes `_n_frames = max(2, int(round(frame_rate * duration)))`, shadowing the identical local already assigned at `plot.py:4477`. Same value; and because `:4477` sits inside a nested `if animate:` in the *string*-`fmt` branch only, recomputing is in fact the safer choice.

**No mismatch found** in: the updater signature, the `ctx.frame` read, the 3-D/2-D/1-D dispatch, `_display_ndims` vs `ndims`, the `_frame_hooks` name, any `ForecastSchedule`/`DisplayTransform` call signature, any cited `plot.py` line number, or any test-helper assumption.

### Check 1 verdict: **PASS** (with M1 flagged as a prescription gap to close during implementation)

---

## Check 2 — Real collected counts: 27 (Task 4) and 31 (cumulative)

The plan's claims: Task 3 → **9**, Task 4 block → **18**, cumulative after Task 4 → **27** (plan line 1626), cumulative after Task 7 → **31** (plan line 2274).

**No stubbing was required.** The prescribed test modules import only `matplotlib`, `numpy`, `pytest` and `hypertools` — nothing from `hypertools.plot.forecast` — so collection resolves against the real package as-is.

```
$ cd /Users/jmanning/hypertools
$ .venv/bin/python -m pytest $S/collect/test_task3_only.py --collect-only -q
9 tests collected in 1.67s

$ .venv/bin/python -m pytest $S/collect/test_task4_block_only.py --collect-only -q
18 tests collected in 1.61s

$ .venv/bin/python -m pytest $S/collect/test_predict_animation_thru_t4.py --collect-only -q
27 tests collected in 1.62s

$ .venv/bin/python -m pytest $S/collect/test_predict_animation_thru_t7.py --collect-only -q
31 tests collected in 1.57s
```

### Manual `@pytest.mark.parametrize` expansion (independent cross-check)

An AST walk over each block, expanding every `parametrize` decorator by the literal length of its argument list and multiplying stacked decorators:

```
--- verbatim/task3_block.py: 6 test defs, manual expansion total = 9
      test_time_progressing_animation_draws_no_static_full_history_overlay: x4  (sizes [4])
--- verbatim/task4_block.py: 16 test defs, manual expansion total = 18
      test_frames_drawn_out_of_order_give_the_same_geometry: x2  (sizes [2])
      test_the_live_forecast_updates_in_both_2d_and_3d:      x2  (sizes [2])
--- verbatim/task7_block.py: 4 test defs, manual expansion total = 4
```

`9 + 18 = 27`; `9 + 18 + 4 = 31`. **No module-level `STYLES` list of `pytest.param` objects exists in any of these three blocks** (the AST scan reported no upper-case module-level constants) — the `STYLES` construct the maintainer warned about lives in **Task 6**'s separate file `tests/plot/test_forecast_animation_plotly.py`, which is out of scope for this check and is not part of the 27/31 arithmetic.

The plan's own derivation of the +6 delta over v2 (plan line 1626) is also correct: `test_a_user_callback_sees_this_frames_forecast_not_the_last_ones` (1) + `test_the_forecast_updater_runs_with_no_user_callback_registered` (1) + `test_frames_drawn_out_of_order_give_the_same_geometry` (2) + `test_the_live_forecast_updates_in_both_2d_and_3d` (2) = 6, and 12 + 6 = 18.

| claim | plan says | really collected | manual expansion |
|-|-|-|-|
| Task 3 block | 9 | **9** | 9 |
| Task 4 block alone | 18 | **18** | 18 |
| **cumulative after Task 4** | **27** | **27** | **27** |
| Task 7 block alone | 4 | **4** | 4 |
| **cumulative after Task 7** | **31** | **31** | **31** |

### Check 2 verdict: **PASS** — every count is exact, by real collection *and* by independent manual expansion

---

## Check 3 — Task 0 in a disposable worktree; on_frame + backend-parity regression

### Setup

```
$ git worktree add /tmp/p3audit 065c841e
Preparing worktree (detached HEAD 065c841e)
HEAD is now at 065c841e docs(plans): Plan 4 v2 review — fix the 3 defects v2 itself introduced
```

Import isolation verified before running anything (the venv has hypertools installed editable from the *main* tree, so this mattered):

```
$ cd /private/tmp/p3audit && .venv/bin/python -c "import hypertools; print(hypertools.__file__)"
/private/tmp/p3audit/hypertools/__init__.py     # <- the worktree copy, not the main tree
```

**Real suite paths verified before running** (both exist; note one is *not* under `tests/plot/`):
* `/Users/jmanning/hypertools/tests/plot/test_on_frame_hook.py`
* `/Users/jmanning/hypertools/tests/test_backend_window_parity.py`
(There is no `tests/plot/test_animation_context.py`, exactly as the plan's Step 5 note states.)

### BEFORE (worktree at 065c841e, unmodified)

```
$ cd /private/tmp/p3audit && .venv/bin/python -m pytest \
      tests/plot/test_on_frame_hook.py tests/test_backend_window_parity.py -q
73 passed in 4.89s
```

Task 0's own 9 tests, extracted verbatim from plan lines 182–275 into `tests/plot/test_frame_hooks_ordering.py`, run **before** the implementation:

```
$ .venv/bin/python -m pytest tests/plot/test_frame_hooks_ordering.py -q
E       AttributeError: 'FrameHooks' object has no attribute 'add_internal'
FAILED ... test_internal_updaters_run_before_user_callbacks
FAILED ... test_a_user_callback_sees_what_the_internal_updater_just_wrote
FAILED ... test_both_phases_share_one_frame_context
FAILED ... test_internal_updaters_run_with_no_user_callbacks_registered
FAILED ... test_an_internal_updater_must_be_callable
FAILED ... test_add_internal_returns_self_for_chaining
FAILED ... test_an_exception_in_an_internal_updater_propagates
7 failed, 2 passed in 1.75s
```

Exactly the plan's Step 2 prediction, including *which* two already pass (`test_dispatch_is_a_no_op_when_nothing_is_registered`, `test_user_callbacks_still_run_in_registration_order`).

### Task 0 applied

All three prescribed edits applied to `/private/tmp/p3audit/hypertools/plot/animation_context.py` **exactly as written** (plan lines 288–294, 297–317, 321–332): `__slots__ = ('callbacks', 'internal', 'state')`, `self.internal = []` in `__init__`, the `add_internal` method, and the `dispatch` rewrite with the two-phase loop and the `if not (self.internal or self.callbacks) or not self.state:` guard.

### AFTER

```
$ .venv/bin/python -m pytest tests/plot/test_frame_hooks_ordering.py -q
9 passed in 1.77s                                       # Task 0's own 9 -> plan claims 9 ✓

$ .venv/bin/python -m pytest \
      tests/plot/test_on_frame_hook.py tests/test_backend_window_parity.py -q
73 passed in 4.85s

$ .venv/bin/python -m pytest tests/plot/test_on_frame_hook.py -q
44 passed in 2.56s                                      # plan claims 44 ✓
$ .venv/bin/python -m pytest tests/test_backend_window_parity.py -q
29 passed in 4.23s                                      # plan claims 29 ✓
```

### Wider sweep (not asked for, run anyway — `dispatch` is central)

```
BEFORE (main tree, unmodified, same content):
$ .venv/bin/python -m pytest tests/plot tests/test_backend_window_parity.py -q
209 passed in 19.77s

AFTER (worktree, Task 0 applied):
$ .venv/bin/python -m pytest tests/plot tests/test_backend_window_parity.py -q
218 passed in 20.27s
```

`218 − 209 = 9`, accounted for **exactly** by Task 0's 9 new tests. Zero failures, zero errors, zero pre-existing failures either side.

| suite | BEFORE | AFTER | delta |
|-|-|-|-|
| `tests/plot/test_on_frame_hook.py` | 44 passed | 44 passed | 0 |
| `tests/test_backend_window_parity.py` | 29 passed | 29 passed | 0 |
| combined (plan claims 73) | **73 passed** | **73 passed** | 0 |
| Task 0's own tests (plan claims 9) | 7 failed / 2 passed | **9 passed** | +9 |
| `tests/plot` + parity, whole dir | 209 passed | 218 passed | +9 (the new file) |

**No regressions.** The plan's Step 4 (`9 passed`) and Step 5 (`44 + 29 = 73, unchanged`) claims are both confirmed by measurement.

### Cleanup

```
$ git worktree remove --force /private/tmp/p3audit && git worktree prune
$ git status --porcelain hypertools/ tests/ docs/superpowers/
(empty)
```

Worktree removed; the main worktree's `hypertools/`, `tests/` and `docs/superpowers/` are untouched.

### Check 3 verdict: **PASS**

---

## Check 4 — No stale `matplotlib_backend._anim_window_bounds` references

### 4.1 The plan

```
$ grep -n "_anim_window_bounds" docs/superpowers/plans/2026-07-27-hypertools-1.1-forecast-animation.md
(no output; exit 1)
```

**Zero hits — the private/old spelling `_anim_window_bounds` does not appear anywhere in the 2386-line plan.**

Every reference the plan does make uses the correct public name and, where a module is named, the correct module:

| plan line | text | module named | correct? |
|-|-|-|-|
| 37 | "Replaced by `anim_window_bounds` (`trails.py:24-94`), which **is** the parallel reveal and is what `update_lines_parallel` itself calls (`matplotlib_backend.py:1185`)" | `trails.py` for the definition, `matplotlib_backend.py:1185` only as a **call site** | **yes** — `matplotlib_backend.py:1185` really is `start, end, trail_stop = anim_window_bounds(` |
| 121 | "`anim_window_bounds(total-1, total, n, w)` → `end = n`" | none | n/a |
| 557 | "Delegates to `trails.anim_window_bounds` … which `update_lines_parallel` itself calls (`matplotlib_backend.py:1185`)"; "(`trails.py:85-86`: `end = int(np.ceil(...))`)" | `trails` | **yes** |
| 601 | `from hypertools.plot.trails import anim_window_bounds` (Task 2 test) | `hypertools.plot.trails` | **yes** — this import executes; the Task 2 module runs 14/14 green against it |
| 771–783 | "`end` comes from `trails.anim_window_bounds` … `matplotlib_backend.py:1185`"; `from .trails import anim_window_bounds` (Task 2 impl) | `.trails` | **yes** |
| 1618 | "`revealed_raw_counts`, which delegates to `anim_window_bounds` — the library's single reveal implementation" | none | n/a |
| 2179 | "`anim_window_bounds(total-1, total, n, w)` → `end = n`" | none | n/a |

### 4.2 The real current location

```
$ cd /private/tmp/p3neutral && .venv/bin/python -c "
    import inspect, importlib
    mb = importlib.import_module('hypertools.plot.matplotlib_backend')
    tr = importlib.import_module('hypertools.plot.trails')
    ..."
mb.__dict__ has 'anim_window_bounds' (imported name): True
  __module__ = hypertools.plot.trails
  defined in file: /Users/jmanning/hypertools/hypertools/plot/trails.py
mb has _anim_window_bounds: False
tr has _anim_window_bounds: False
tr.anim_window_bounds defined in: /Users/jmanning/hypertools/hypertools/plot/trails.py
tr.__all__: ['broadcast_trail_flag', 'anim_window_bounds']
```

```
$ grep -rn "def anim_window_bounds\|def _anim_window_bounds" hypertools/ tests/
hypertools/plot/trails.py:24:def anim_window_bounds(num, total_frames, n_points, window_frames):
```

* **Confirmed present** in `hypertools/plot/trails.py:24` (sole definition; exported via `__all__` at `trails.py:21`).
* **Confirmed absent** as a definition in `hypertools/plot/matplotlib_backend.py` — that module only *imports* the name (`matplotlib_backend.py:41: from .trails import anim_window_bounds, broadcast_trail_flag`) and otherwise mentions it in 8 comments. The private `_anim_window_bounds` spelling does not exist on either module.

### 4.3 Incidental finding outside the plan

The only `_anim_window_bounds` string anywhere under the repo root is:

```
docs/_build/html/_modules/hypertools/plot/plot.html:6466:
    <span class="c1"># matplotlib_backend._anim_window_bounds)</span>
```

This is a **stale, gitignored, untracked** Sphinx build artifact (`.gitignore:15: docs/_build`; `git ls-files` returns nothing for it). The live source it was generated from has since been corrected — `hypertools/plot/plot.py:6099` now reads `# hypertools.plot.trails.anim_window_bounds)`. Nothing to fix in tracked source; the artifact will be overwritten on the next `make html`.

### Check 4 verdict: **PASS**

---

## VERDICT

# IMPLEMENTABLE

All five checks pass. **No blocking items.**

| # | check | result |
|-|-|-|
| 1 | Task 4 test + `_update_forecasts` blocks executed verbatim | **PASS** |
| 2 | 27 Task 4 / 31 cumulative, by real collection | **PASS** |
| 3 | Task 0 applied in a worktree; on_frame + parity suites | **PASS** |
| 4 | No stale `_anim_window_bounds` in the plan | **PASS** |
| 5 | Verdict | **IMPLEMENTABLE** |

### Evidence summary

* Every one of the 18 Task 4 tests fails today for **one** reason — the shipped refusal at `hypertools/plot/plot.py:2749` that Task 3 removes — and for no other. No `NameError`, no signature mismatch, no fixture error.
* The `_update_forecasts` block was **executed for real** against real matplotlib artists, a real `ForecastSchedule` (14 real Kalman fits) and the real Task-0 `FrameHooks`: all three `_ndims` branches work, frames are idempotent under out-of-order replay, and a user callback observes the current frame. v2's Fatal (`def _update_forecasts(frame, …)`) is confirmed fixed.
* Every cited `plot.py` line number (`3391-3402`, `4552`, `4555`, `4568-4585`, `4740`, `4858-4898`, `4907`, `3310`, `282`, `4158`, `738-741`) was checked against real source and is **exact**.
* Collected counts are exact by two independent methods (pytest collection and AST parametrize expansion): 9 / 18 / **27** / 4 / **31**.
* Task 0 applied cleanly: `7 failed, 2 passed` → `9 passed`, with `44 + 29 = 73` unchanged and a wider `tests/plot` sweep going `209 → 218` (+9 = exactly the new file).
* Task 2's prescribed implementation was run against its own prescribed test module: **14 passed**, confirming the plan's Task 2 Step 4 claim and, transitively, every `ForecastSchedule`/`DisplayTransform` signature Task 4 depends on.
* `_anim_window_bounds` appears **zero** times in the plan; `anim_window_bounds` is defined only in `hypertools/plot/trails.py:24` and is not defined in `matplotlib_backend.py`.

### Non-blocking items to close during implementation

1. **M1 (MEDIUM).** `_live_forecast_artists` is consumed at plan line 1574 but never defined by prescribed code — only by prose at plan line 1570. The implementer must write the artist-creation loop (one dashed artist per dataset, dataset colour, `alpha=0.6`, `label='_nolegend_'`, `set_clip_on(False)`, `_hyp_forecast_role = 'live'`), using `hypertools/plot/plot.py:137-180` (`_draw_forecast_overlays`) as the template. Consider adding that block to the plan before handing it to a worker.
2. **M2 (LOW).** `test_forecast_is_anchored_near_the_drawn_head`'s docstring (plan 1328–1331) quotes the `duration=2/frame_rate=4` measurement while the test runs `duration=4/frame_rate=4`. Cosmetic; the assertion derives its tolerance from the drawn data at runtime.
3. **M3 (LOW).** Step 4 re-binds `_n_frames`, shadowing `plot.py:4477`. Same value, and safer given `:4477` only executes in the string-`fmt` branch — worth a one-line comment rather than a change.
4. **Incidental (LOW, outside the plan).** `docs/_build/html/_modules/hypertools/plot/plot.html:6466` still contains `matplotlib_backend._anim_window_bounds` in a stale, gitignored, untracked Sphinx artifact. Live source is already correct (`hypertools/plot/plot.py:6099`). Regenerating docs clears it.
