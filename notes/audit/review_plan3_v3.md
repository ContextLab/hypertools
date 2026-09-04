# ADDENDUM 2026-08-01 — VERDICT: IMPLEMENTABLE

*(This supersedes the "NOT IMPLEMENTABLE" verdict below, which was written against commit
`4ecb3b6d`. The eleven findings it raised were fixed in `3f94c5bf`; this addendum records the
closure audit the maintainer required before implementation, run against `065c841e`.)*

Closure audit: `notes/audit/plan3_closure_audit.md`. All five maintainer-specified checks PASS.

| # | check | result |
|-|-|-|
| 1 | Task 4 test + `_update_forecasts` blocks extracted and executed verbatim | **PASS** |
| 2 | 27 Task 4 / 31 cumulative confirmed by REAL collection, not AST | **PASS** |
| 3 | Task 0 applied in a disposable worktree; `on_frame` + parity suites re-run | **PASS** |
| 4 | no stale `matplotlib_backend._anim_window_bounds` anywhere in the plan | **PASS** |
| 5 | verdict | **IMPLEMENTABLE** |

Evidence that mattered:

- **Check 1 went past static analysis.** `forecast.py` was built from Tasks 1+2 verbatim, Task 0
  applied, and the `_update_forecasts` block executed against real `Line3D`/`Line2D` artists and a
  real `ForecastSchedule` (14 real Kalman fits). All three `_ndims` branches run; the out-of-order
  replay `0,4,7,4,0` is byte-identical; user callbacks observe the current frame. All 18 Task 4
  tests fail for exactly ONE cause — the shipped refusal at `plot.py:2749` that Task 3 removes —
  with no `NameError` and no signature mismatch. v2's Fatal is genuinely dead.
- **Check 2 needed no stubbing.** Real collection gave 9 / 18 / **27** / 4 / **31**, matching an
  independent AST parametrize expansion. (The `STYLES` construct that trips naive counters lives in
  Task 6's file, not these blocks.)
- **Check 3**: BEFORE 73 passed (44 + 29); Task 0's own tests 7 failed / 2 passed exactly as the
  plan predicts; AFTER 9 passed with 73 unchanged. Wider `tests/plot` sweep 209 → 218 (+9 = the new
  file). No regression.
- **Check 4**: zero hits in the plan. `anim_window_bounds` is defined only at `trails.py:24`;
  `matplotlib_backend.py` merely imports it at `:41`.

## Findings from the closure audit, and their disposition

| sev | finding | disposition |
|-|-|-|
| MEDIUM | `_live_forecast_artists` was **consumed** by Task 4 Step 5's code block but only **described in prose** — the creation code was never written. A plan may not reference an object no task defines. | **FIXED.** The artist-creation block is now written out in full. It snapshots `list(ax.lines)` *before* the loop — otherwise artist *i* would take its colour from forecast *i-1*, the same guard `_draw_forecast_overlays` opens with (`plot.py:157`). Executed against real 1-D, 2-D and 3-D axes: 3 artists each, forecast colours identical to trajectory colours, `role='live'`, `clip_on=False`, `ls='--'`, `alpha=0.6`, and the artists drive correctly afterwards. |
| LOW | a test docstring quoted `duration=2` densification figures while the test runs `duration=4` | **FIXED.** The test's real grid is 16 rows, not 8; the docstring's own derivation (`59/15 ≈ 3.9`) was already the 16-row figure, so the citation was wrong, not the arithmetic. The ~15.1x figure is now explicitly attributed to the other configuration. |
| LOW | Step 4 re-binds `_n_frames`, shadowing `plot.py:4477` | **NO CHANGE.** Same value, and the local binding is the safer of the two. |
| LOW | a stale artifact retains the old `_anim_window_bounds` name | **NO CHANGE NEEDED.** Verified it is only `docs/_build/html/_modules/hypertools/plot/plot.html`, which `.gitignore:15` ignores. It is regenerated on the next docs build and never ships. No live source or committed doc references the old symbol. |

Plan counts re-derived after these edits: unchanged at 9 / 8 / 14 / 11 / **18** / 11 / 18 / 4,
grand total **93**. The added block contains no tests and parses as valid Python (31 lines).

**Implementation order** (maintainer's, with their Task 4 split):
Task 0 → Tasks 1-2 → Task 3 → **Task 4a** (schedule + bounding box) → **Task 4b** (live drawing) →
Task 5 → Task 6 → Task 7. Task 4 is split into two commits because the two halves have different
failure modes and can be reviewed independently.

---

VERDICT: NOT IMPLEMENTABLE

Adversarial re-review of Plan 3 v3 (`docs/superpowers/plans/2026-07-27-hypertools-1.1-forecast-animation.md`,
2345 lines) against `dev-1.0` HEAD `4ecb3b6d`. Method as instructed: the plan's prescribed code was
EXTRACTED AND EXECUTED, never reasoned about. Nothing in the repo was modified (`git status --porcelain`
empty at the end; no worktree created). All work in
`/private/tmp/claude-501/-Users-jmanning-hypertools/7e6531b3-.../scratchpad/rev`.

**Every v2 finding is fixed.** The v2 Fatal is genuinely dead: the prescribed callback was executed
against a real `FrameContext` and real `Line3D`/`Line2D` artists through all three branches and works.
Task 0 was applied verbatim to a copy of the real `animation_context.py` and produces exactly the
claimed 9 passed (7 failed / 2 passed before, and the 2 pre-passing tests are exactly the two named).
Tasks 1+2 run verbatim against HEAD and pass 22. Task 3's predicted "7 failed, 2 passed" is exact.
**Every one of the 11 stated test counts re-derives correctly** by `ast` with parametrize expansion,
including the `STYLES` module-level list. **All 31 spot-checked citations are correct** (0 wrong).

It is NOT implementable because Task 4 — the core task — cannot reach its own stated exit gate
("18 passed"): the prescribed test file contains a call that raises `TypeError` in both
parametrizations, its 2-D half cannot work with the plan's own axes helper, and the prescribed
implementation snippet names a variable that does not exist in `plot()`. Three mechanical edits away
from implementable; no Fatal.

---

## Prior findings

Prior finding 1 (**Fatal**, `_update_forecasts(frame, …)` vs `dispatch(callback(ctx))`): **FIXED** — plan
L1557 is now `def _update_forecasts(ctx, _sched=…, _artists=…, _antialias=…, _ndims=…)` reading
`ctx.frame`. Extracted the block verbatim and ran it against a real `FrameContext(figure=fig, axes=ax,
frame=f, n_frames=8)`, a real `ForecastSchedule.to_display(...)` and real matplotlib artists:
signature `(ctx, _sched=<ForecastSchedule>, _artists=[Line3D], _antialias=False, _ndims=3)`; frame 0
(`pts is None`) → `visible=False, data shape (3,0)`; frames 1/4/7 → `visible=True, shape (3,4)`;
`_antialias=True` → `shape (3,901)`; 2-D branch → `Line2D`, `shape (2,4)`. Confirmed against the REAL
`hypertools/plot/animation_context.py`: `dispatch` does `ctx = FrameContext(...)` then `callback(ctx)`
(`:267-275`) — one argument, a `FrameContext`.

Prior finding 2 (**High**, `test_to_display_...` asserted on `disp.path()` = displacement): **FIXED** —
plan L701-733 now asserts on `.polyline()`, counts iterations (`assert checked > 0`), and L736 adds
`test_display_paths_are_displacements_not_positions`. Ran the plan's own Task 2 test file against the
plan's own Task 2 implementation: **14 passed**.

Prior finding 3 (**High**, `test_predict_integration.py` "17 passed"): **FIXED** — plan L1150-1161 now
says **15**, derived in a table. Measured today: `pytest tests/plot/test_predict_integration.py -q` →
`18 passed in 3.21s`; `--collect-only` → `18 tests collected`, of which
`test_time_progressing_animate_and_predict_raises_not_implemented` contributes 5 IDs
(`True/parallel/serial/window/morph`, real file `:169-178`). 18 − 5 + 2 = **15**. ✔

Prior finding 4 (**Med**, plotly parity tested `animate=True` only): **FIXED** — plan L1972-1977 defines
`STYLES` = 4 `pytest.param`s (`parallel`/`serial`/`window`/`order-serial`), 3 tests parametrized over
it, plus `test_plotly_serial_reveals_one_datasets_forecast_at_a_time`. animation-core Task 4 is now
listed in Prerequisites (L92) and is really shipped: measured
`hyp.plot(..., animate='serial', chemtrails=True, backend='plotly')` → `traces=5 frames=8 warnings=[]`,
2 trail-alpha traces — trails are drawn, not warned-and-dropped.

Prior finding 5 (**Med**, internal updater ran after user `on_frame`s): **FIXED** — new Task 0. Applied
the plan's `__slots__`/`add_internal`/`dispatch` blocks (plan L288-293, L297-317, L321-332) verbatim to
a COPY of the real file outside the repo and ran the plan's test file verbatim.
BEFORE: `7 failed, 2 passed` — all 7 `AttributeError: 'FrameHooks' object has no attribute
'add_internal'`, the 2 passes exactly `test_dispatch_is_a_no_op_when_nothing_is_registered` and
`test_user_callbacks_still_run_in_registration_order`, precisely as plan L281 predicts.
AFTER: **9 passed**, precisely as plan L344 claims.
Regression gate (plan L348) also verified: `pytest tests/plot/test_on_frame_hook.py
tests/test_backend_window_parity.py -q` → **73 passed**.

Prior finding 6 (**Med**, multiindex.md cited a non-existent test name): **FIXED** — 
`docs/superpowers/plans/2026-07-28-hypertools-1.1-multiindex.md:2797` now cites
`test_hue_regrouping_drops_forecasts_exactly_like_the_static_path`; grep for the old
`test_forecast_dropped_under_hue_regrouping` returns nothing.

Prior finding 7 (**Low**, multiindex cites plan3 `:289`/`:153`/`:389`): **FIXED** — now `:522`/`:386`/`:621`,
and all three land exactly: plan A `:522` = `if len(history) < max(2, min_history):`, `:386` =
`def test_returns_none_below_min_history():`, `:621` = `def test_early_frames_have_no_forecast():`.

Prior finding 8 (**Low**, counts at plan3 `:306`/`:1421`/`:1918`): **FIXED** — Task 1 Step 4 now reads
"8 passed (7 named tests, one of them parametrized over 2 models)" (ast: 7 defs, 8 IDs ✔); Task 5 Step 2
"11 items: 9 named tests, one parametrized over 3 values" (ast: 9 defs, 11 IDs ✔); the Self-Review
"T4 wrong pass counts" row now reads Task 1→8, Task 2→14, Task 3→9 (+15), Task 4→27, Task 5→11,
Task 6→18, Task 7→31 — **all seven re-derive exactly** (see Verification).

Supplementary rows of the same v2 audit table:
Prior finding 9 (**Low**, plan3 `:1704` plotly reveal `:2898`→`:2899`; plan4 `:118` refusal `2347`→`2346`):
**FIXED** — `grep -n "289[89]\|234[67]"` over plan A returns nothing; those citations no longer exist.
Prior finding 10 (**Low**, Task 4 "Files" listed `matplotlib_backend.py`; Task 3 Step 2 said the morph
tests fail): **FIXED** — Task 4's Files bullet (L1181) now names `plot.py` only; Task 3 Step 2 (L1061)
now states "**7 failed, 2 passed** (the 2 passes are the morph tests)", which I measured to be exactly
right by running the block against HEAD.
Prior finding 11 (**Low**, env: stale non-editable 1.0.0 in `.venv`): **FIXED** — `.venv/bin/python -c
"import hypertools"` → `/Users/jmanning/hypertools/hypertools/__init__.py`, version `1.0.1`,
`antialias` present in `plot()`'s signature.

---

## New findings

**High | Task 4 Step 1, plan L1281 (`test_the_live_forecast_updates_in_both_2d_and_3d`) | The test v3
added to close the 2-D/3-D dispatch finding cannot run at all: it calls `_series(n=1, d=max(ndims, 2))`,
and `_series` has no `d` parameter.** The module's helper is defined at plan L974 as
`def _series(n=3, rows=60, dims=4, seed=0)`. Measured with the plan's own helper verbatim:
`ndims=2: TypeError: _series() got an unexpected keyword argument 'd'` and the same for `ndims=3`.
Both parametrizations error. Consequence: Task 4 Step 6's "Expected: **27 passed** (9 from Task 3 +
**18** here)" is unreachable — 16 here, 25 cumulative — and Task 7 Step 4's "**31 passed**" becomes 29.
Fix: `_series(n=1, dims=max(ndims, 2))`.

**High | Task 4 Step 1, same test, `ndims=2` parametrization | Even with the kwarg fixed, the 2-D case
dies in the plan's own `_ax()` helper, so the 2-D branch of the `_ndims >= 3` dispatch stays untested.**
`_ax(fig)` (plan L979) is `[a for a in fig.axes if hasattr(a, 'zaxis')][0]`. Measured on a real 2-D
animated hypertools figure: `fig.axes` types `['Axes']`, `hasattr zaxis` `[False]` →
`IndexError: list index out of range`. The repo's own `FrameContext` docstring says this
(`animation_context.py:56-58`: "an ``Axes3D`` for 3-D plots, a plain ``Axes`` for 2-D ones (which have
no ``zaxis``)"). This matters more than the count: v3's revision table (L25) sells this test as the
guard against "`set_data` alone leaves z-data stale", and the cheapest way for an executor to get to
green is to drop the `ndims=2` parametrization — which silently restores exactly the gap v3 claims to
have closed. The test needs a dimension-aware axes helper (e.g. `fig.axes[0]` when `ndims < 3`).
*(The 3-D justification itself is CONFIRMED, and is in fact stronger than the plan states — see
Verification: on a real `Line3D`, `set_data` leaves `get_data_3d()` completely unchanged, x and y
included, not merely z.)*

**Med | Task 4 Step 5, plan L1581 | `hooks.add_internal(_update_forecasts)` names a variable that does
not exist anywhere in `plot()`; verbatim use raises `NameError` at `plot()` call time.** The registry
local is `_frame_hooks`, created at `plot.py:4740`
(`_frame_hooks = FrameHooks([on_frame] if on_frame is not None else [])`) and referenced 11 times;
grep for a bare `hooks` local in `plot.py` returns nothing. In `matplotlib_backend._draw` the same
object arrives as the parameter `frame_hooks`, so neither scope spells it `hooks`. The prose one
paragraph below ("**The signature is the contract.**") is meticulous about the callback's argument and
silent about its receiver. Fix: `_frame_hooks.add_internal(_update_forecasts)`. (The block is also
indented 16 spaces while the insertion point after `_draw(...)` at `plot.py:4858-4898` is at 12.)

**Med | Task 4 Step 5, `_ndims=ndims` | `plot(ndims=None)` is a supported call and would make the
prescribed closure raise `TypeError: '>=' not supported between instances of 'NoneType' and 'int'` on
every frame.** Measured: `hyp.plot(d, '-', ndims=None, show=False)` builds an `Axes3D` today.
`plot.py:3310` — `_display_ndims = ndims if (ndims and ndims < 3) else 3` — exists precisely because a
falsy `ndims` is expected, and `_display_ndims` (not `ndims`) is the value that decides the artist's
dimensionality: it is what `plot.py:3327`'s reducer call uses. The closure should capture
`_ndims=_display_ndims`. `ndims` defaults to `3`, so the common path is unaffected; this bites only an
explicit `ndims=None`.

**Med | Self-Review, plan L2341 | "**Placeholders.** None: every step carries runnable code, an exact
command, and an expected result" is false for the three largest un-coded steps.** Task 4 Step 5's
live-artist CREATION (only the updater closure is code; the artists themselves are one prose
sentence), Task 5 Step 4's preallocation and per-frame write (one prose paragraph, L1871), and Task 6
Steps 3/4/5 in their entirety (bullet specifications, L2105-2122) carry no code block. They are
specific and well-argued, but they are specifications, and Tasks 5-6 are exactly where an executor has
the most latitude to diverge from the tests that gate them.

**Low | Task 4 Step 2, plan L1464 | "Expected: the **12 new tests** FAIL" contradicts Step 6's 18.**
Stale v2 figure; the citation sweep corrected the Self-Review row and Step 6 but not this one. AST:
the Step 1 block has 16 `def test_`, two parametrized ×2 → 18 IDs.

**Low | Global Constraints L82 and Task 3 Step 8 L1166 | "Baseline today: **2564 collected / 2
deselected**" is stale by 218 tests.** Measured: `2782/2784 tests collected (2 deselected) in 5.24s`.

**Low | Task 2 Step 4, plan L939 | Two false statements about
`test_display_paths_are_displacements_not_positions`.** It says the test "has **not** been run against
an implementation and is expected to fail until `ForecastDisplay` exists". Measured: the plan's own
Task 1 + Task 2 implementation and test blocks, extracted verbatim (only `from ..predict.predict` /
`from .trails` / `from .matplotlib_backend` rewritten to absolute imports for the scratch location) →
**22 passed in 10.08s**, that test included. And `ForecastDisplay` appears nowhere else in the plan or
the codebase — the prescribed implementation is `ForecastSchedule.to_display`.

**Low | Task 5 Step 4, plan L1842 | `trail_frames(frame, n_retained, n_frames, stride=1)`'s `n_frames`
parameter is dead.** AST: `n_frames` is never referenced in the body. Harmless, but the Interfaces
contract at L1653 presents it as if it constrains the result ("the frame indices whose forecasts are
retained at `frame`"), and every call site the plan writes passes it.

**Low | Prerequisites L93 vs Task 4 Step 1 L1414 | test-name mismatch.** Prerequisites promises
`test_forecast_composes_with_serial_order`; the test is `test_forecast_composes_with_order_serial`.

**Low | Task 4 Step 4, plan L1519 | `_n_frames = max(1, int(round(frame_rate * duration)))` disagrees
with the frame grid's own `max(2, ...)`.** `plot.py:4475` is
`_n_frames = max(2, int(round(frame_rate * duration)))`. Only differs when
`round(frame_rate*duration) < 2`, where the schedule would carry 1 frame while the drawn grid carries
2 rows.

**Low | Task 4 Step 5 prose L1593 vs its own code block | the prose over-promises a three-way
dispatch.** "For 2-D and 1-D animations use `set_data` alone … mirroring `_draw_forecast_overlays`'
`d >= 3 / d == 2 / else` dispatch (`plot.py:167-179`)" — but the prescribed code has two branches, and
its `else` does `art.set_data(pts[:, 0], pts[:, 1])`, which would `IndexError` on 1-D. Not reachable:
measured `hyp.plot(..., animate=True, ndims=1)` →
`ValueError: Animations are only supported for 2-D or 3-D plots (got 1-D data)`. The code is right; the
prose should not claim the 1-D branch.

**Low | Task 4 Step 3 | the `analyze_histories` snapshot inherits a length-only correspondence guard
that two reorderings can slip past.** `plot.py:3795` and `plot.py:4087` both do
`xform = [xform[i] for i in _order]` inside the `cluster=`/`hue=` regrouping; the guard the plan extends
(`plot.py:4552`) only compares `len(raw_forecasts) != len(xform)`. When the regrouped count happens to
equal the input count, the guard does not fire and the schedule is indexed against a reordered `xform`.
This is PRE-EXISTING for `raw_forecasts` (which is never reordered either — grep shows no reorder of it)
and the plan is honest that the snapshot "keeps the same 1:1 dataset correspondence the guard at
`plot.py:4552` checks". Flagged only because it is the one place the new snapshot could silently
mis-pair rather than crash.

### Explicitly checked and clean (no finding)

- **Contract 9 is consistent with what animation-core shipped.** All seven `frame_hooks.record(...)`
  sites in `matplotlib_backend.py` build `artists=` from the backend's own line lists
  (`:1230` `artists=list(lines) + [t for t in trail_lines if t is not None]`, and likewise `:1302`,
  `:1423`, `:1511`, `:2102`, `:2168`, `:2201`), so artists created in `plot()` after `_draw` are
  structurally excluded. Nothing anywhere in the plan puts a forecast artist into `ctx.artists`: the
  only `ctx.artists` mentions are Contract 9 itself and its rationale, and every test reaches forecast
  artists through `ax.lines` by `_hyp_forecast_role`.
- **All three prerequisites are really shipped.** `'order' in inspect.signature(hyp.plot).parameters`
  → `True`; `'on_frame'` → `True`; `FrameHooks` exists with `add`/`record`/`dispatch`
  (`animation_context.py:226-275`); plotly draws serial trails with no warning; commits `7c859581`
  ("fix(plot): reject non-string title=…") and `f6084c7d` ("fix(plot): whole-branch review fixes for
  animation-core…") both resolve.
- `_frame_snapshots` — which Task 6's test helper imports — really exists at
  `plotly_backend.py:1488`, and it copies the base figure before applying each frame's updates, so the
  base traces' `meta` (which Task 6 Step 4 sets) survives into the snapshot the tests inspect.
- `_validate_forecast_trail` routes every case its tests demand: `False/None/0`→0, `True`→16, `4`→4,
  `True` with `predict=None`→`ValueError` matching "forecast_trail= requires predict=", and
  `-1`→`ValueError` / `'yes'`→`TypeError` / `2.5`→`TypeError`, all matching "forecast_trail".
- `trail_alpha` is strictly decreasing in age and always below the live 0.6
  (ages 0-5 @ n=16: 0.6, 0.5694, 0.5388, 0.5082, 0.4776, 0.4471; ages 1-16 span 0.1106-0.5694).
- `_resolve_animate_mode(True, 3, order='serial')` → `('serial', None, 'serial')`, so Task 4 Step 4's
  `animate == 'serial' or order == 'serial'` selects `for_serial` correctly (redundantly) for both
  spellings.
- The return bundle really has `fig`/`animation`/`xform_data`/`predict`; `ani._func`/`ani._args` are
  real and `ani._func(3, *ani._args)` runs; `fig, ani = hyp.plot(..., animate=True, ...)` unpacks;
  plotly returns a bare `go.Figure` with `len(fig.frames) == round(frame_rate*duration)`;
  `xform_data[0].shape == (60, 3)` for a 60×4 input, so Task 7's `(4, 3)` and `(t+1, 3)` shape
  assertions are right.
- `_to_plotly_color(color, alpha)` returns `f'rgba(r,g,b,{a})'` with the full float repr
  (`plotly_backend.py:2552-2558`), so Task 6's `f'{a}' in tr.line.color` assertion holds.
- `_interp_static_line` is module-level in `plot.py:282`, so the closure resolves it.
- `matplotlib_backend` builds every animation with `blit=False` (`:2006/:2017/:2027/:2042/:2422`), so
  mutating the forecast artist inside `dispatch` — which runs AFTER `_orig(num, *fargs)` in
  `_hyp_frame_with_hooks` (`plot.py:4987-5007`) — is rendered.

---

## Verification performed

Every command below was run with `/Users/jmanning/hypertools/.venv/bin/python` and `MPLBACKEND=Agg`.

**1. Integration baseline (plan's own claim, plan L82/L1156)**
```
$ MPLBACKEND=Agg .venv/bin/python -m pytest tests/plot/test_predict_integration.py -q
18 passed in 3.21s
$ ... --collect-only | tail
18 tests collected in 1.59s
   ...test_time_progressing_animate_and_predict_raises_not_implemented[True|parallel|serial|window|morph]  (5 IDs)
```
→ 18 − 5 + 2 = **15**. Plan's Step 7 table is exact.

**2. Full-suite collection (plan L82, L1166 claim "2564 collected / 2 deselected")**
```
$ MPLBACKEND=Agg .venv/bin/python -m pytest -q --collect-only | tail -1
2782/2784 tests collected (2 deselected) in 5.24s
```
→ STALE.

**3. Task 0 — prescribed change applied to a COPY of the real file, outside the repo**
Copied `hypertools/plot/animation_context.py` to
`scratchpad/rev/t0/pkg/animation_context_ORIG.py`; replaced the `__slots__`/`__init__` block and the
whole `dispatch` method with plan L288-293 and L321-332 verbatim, inserted plan L297-317 verbatim
between `add` and `record`; wrote the plan's test file (L182-275) verbatim with only the import path
retargeted.
```
BEFORE:  7 failed, 2 passed in 0.03s
         all 7 = AttributeError: 'FrameHooks' object has no attribute 'add_internal'
         passing: test_dispatch_is_a_no_op_when_nothing_is_registered,
                  test_user_callbacks_still_run_in_registration_order
AFTER:   9 passed in 0.01s
$ MPLBACKEND=Agg .venv/bin/python -m pytest tests/plot/test_on_frame_hook.py \
      tests/test_backend_window_parity.py -q
73 passed in 4.76s
```
→ plan L281, L344 and L349 all exact.

**4. Tasks 1+2 — prescribed implementation AND tests, extracted verbatim and run**
Concatenated plan L457-531 + L768-933 + L1842-1868 into `forecast_scratch.py` (three relative imports
rewritten to absolute for the scratch location; no other edit), plus the test files L376-446 and
L564-755.
```
$ MPLBACKEND=Agg .venv/bin/python -m pytest test_forecast_core.py test_forecast_schedule.py -q
22 passed in 10.08s
```
→ Task 1 "8 passed" ✔, Task 2 "14 passed" ✔, and plan L939's "expected to fail until `ForecastDisplay`
exists" is false.

**5. Task 3 test block, run verbatim against HEAD (plan L1061 claim)**
```
$ MPLBACKEND=Agg .venv/bin/python -m pytest test_predict_animation_task3.py -q
7 failed, 2 passed in 2.59s
   failing: test_predict_with_animate_true_no_longer_raises,
            ...draws_no_static_full_history_overlay[True|parallel|serial|window],
            test_spin_still_draws_the_static_overlay,
            test_static_plot_still_draws_the_static_overlay
```
→ exact, and the 2 passes are the morph tests, as the plan says.

**6. The `_update_forecasts` callback, executed (plan L1557-1581)**
Body `exec`'d verbatim; called with a real `FrameContext` built the way `dispatch` builds it, a real
`ForecastSchedule.to_display(DisplayTransform(...))` and real artists from `ax.plot`.
```
signature: (ctx, _sched=<ForecastSchedule>, _artists=[Line3D], _antialias=False, _ndims=3)
3-D, antialias=False:  frame (0, visible=False, shape (3,0), None)      <- pts is None branch
                       frame (1, True, (3,4), [ 0.2546 -0.3457 -0.9937])
                       frame (4, True, (3,4), [-0.1034 -0.0559  0.1004])
                       frame (7, True, (3,4), [-0.5895  0.0945  0.0986])
3-D, antialias=True:   frame (4, True, (3,901), ...)                    <- PCHIP densification
2-D (Line2D):          frame (0, False, (2,0)); (4, True, (2,4)); (7, True, (2,4))
```
And the real `FrameHooks.dispatch` (`animation_context.py:267-275`) does
`ctx = FrameContext(figure=figure, axes=axes, **self.state)` then `callback(ctx)` — one arg, a context.

**7. `set_data` vs `set_data_3d` on a REAL `Line3D` (the plan's stated justification)**
```
ln = ax.plot([0.,1.],[0.,1.],[100.,200.],'--')[0]      # type: Line3D
get_data_3d()                       -> [[0,1],[0,1],[100,200]]
ln.set_data([5.,6.],[7.,8.])
get_data_3d()                       -> [[0,1],[0,1],[100,200]]   z STALE? True
ln.set_data_3d([5.,6.],[7.,8.],[9.,10.])
get_data_3d()                       -> [[5,6],[7,8],[9,10]]
```
→ CONFIRMED, and stronger than the plan claims: through the 3-D API `set_data` leaves x and y stale
too, not just z.

**8. Every stated test count, re-derived by `ast` with parametrize expansion**
(module-level `STYLES` resolved by name, not missed)
```
Task 0 test file  L182-275 :  9 defs, 9 IDs      claim 9   OK
Task 1 test file  L376-446 :  7 defs, 8 IDs      claim 8   OK   (1 param x2)
Task 2 test file  L564-755 : 14 defs, 14 IDs     claim 14  OK
Task 3 block      L962-1055:  6 defs, 9 IDs      claim 9   OK   (1 param x4)
Task 4 append    L1191-1458: 16 defs, 18 IDs     claim 18  OK   (2 params x2)
Task 4 cumulative                    27          claim 27  OK
Task 5 test file L1658-1796:  9 defs, 11 IDs     claim 11  OK   (1 param x3)
Task 6 plotly    L1921-2097:  9 defs, 18 IDs     claim 18  OK   (3 params x4 via STYLES)
Task 7 append    L2155-2211:  4 defs, 4 IDs      claim 4   OK
Task 7 cumulative                    31          claim 31  OK
Task 3 Step 6 replacement L1127-1143: 1 def, 2 IDs  claim +2  OK
```
→ **11 of 11 correct.** (The 18-count is nevertheless unreachable — see the two High findings.)

**9. The two High findings, measured with the plan's OWN helpers**
```
_series(n=1, d=max(2,2)) -> TypeError: _series() got an unexpected keyword argument 'd'
_series(n=1, d=max(3,2)) -> TypeError: _series() got an unexpected keyword argument 'd'
fig,ani = hyp.plot(_series(n=1),'-',animate=True,ndims=2,duration=2,frame_rate=4,show=False)
  fig.axes types: ['Axes']   hasattr zaxis: [False]
  _ax(fig) -> IndexError: list index out of range
hyp.plot(..., animate=True, ndims=1) -> ValueError: Animations are only supported for 2-D or 3-D plots
hyp.plot(d,'-',ndims=None,show=False) -> Figure with ['Axes3D']       (so None >= 3 would TypeError)
```

**10. `hooks` vs `_frame_hooks`**
```
$ grep -rn "FrameHooks(" hypertools tests --include='*.py'
hypertools/plot/plot.py:4740:    _frame_hooks = FrameHooks([on_frame] if on_frame is not None else [])
$ grep -n "^\s*hooks\b\|[^_]hooks\." hypertools/plot/plot.py     # (no output)
$ grep -c "_frame_hooks" hypertools/plot/plot.py
11
```

**11. Prerequisites**
```
$ .venv/bin/python -c "import inspect, hypertools as hyp; s=inspect.signature(hyp.plot).parameters; ..."
order present=True default=None      on_frame present=True default=None
ndims present=True default=3         antialias present=True default=True
forecast_trail present=False   (correct: this plan adds it)
$ git cat-file -t 7c859581 -> commit ; f6084c7d -> commit
plotly serial trails: animate='serial' -> traces=5 frames=8 warnings=[] ; trail-alpha traces: 2
                      animate=True     -> traces=5 frames=8 warnings=[] ; trail-alpha traces: 2
```

**12. Citation spot-check — 31 unique `file.py:NNN` citations opened at the cited line (10 required)**
Extracted all 54 unique citations / 98 occurrences from the plan by regex, then printed the real file
at each of the following and confirmed the prose:
`plot.py` 137-180 (`_draw_forecast_overlays`), 167-179 (`d>=3/d==2/else`), 282 (`_interp_static_line`),
293-315 (`_interp_anim_line`), 803 (`chemtrails=` in the signature), 2255-2259 ("Forecast overlays
drawn by `predict=` are smoothed the same way"), 2289-2295 (bundle docstring), 2740-2756 (the refusal;
`if` really at 2748), 3391-3402 (`raw_forecasts` construction), 3412 (`pre_interp_lengths`), 4158
(`_resolve_animate_mode` call), 4460-4478 (the frame-grid resample), 4552 (the length guard nulling
`raw_forecasts`), 4555 (the centre/scale comment), 4569-4582 (the centre/scale arithmetic), 4795
(`forecasts=raw_forecasts` into `plotly_draw`), 4858-4898 (the `_draw(...)` call), 4907-4909 (the
static overlay);
`matplotlib_backend.py` 1185 (`anim_window_bounds` call in `update_lines_parallel`), 1941-1943
(`set_xlim3d([-cube_scale_anim, cube_scale_anim])`);
`plotly_backend.py` 465 (`forecasts=None`), 928-958 (`if forecasts is not None:` block), 965-970 (why
`trail_trace_start` exists), 980-984 (`trail_trace_start`/`trail_dataset_indices`), 1000
(`_to_plotly_color(color, _trail_alpha)`), 2580-2593 (`_add_animation` signature), 3003-3006
(`trace_indices` extension), 3241-3250 (per-frame trail rewrite);
`trails.py` 24-94 (`anim_window_bounds`), 85-86 (`end = ceil((num+1)*n_points/total)`);
`streaming.py` 382-401 (the frozen box + `np.clip`), 498-508 (the >25% `RuntimeWarning`).
→ **31 of 31 correct, 0 wrong.** The sweep's "51 of 53 fixed" claim is corroborated on a 31-citation
independent sample. Also confirmed: `grep -n "289[89]\|234[67]"` over the plan returns nothing, and the
cross-plan citations in the multiindex plan (`:522`/`:386`/`:621`, and the real test name) all land.

**13. Task 5 helpers, executed (plan L1809-1834, L1842-1868)**
```
DEFAULT_FORECAST_TRAIL = 16
False/None/0 -> 0 ; True -> 16 ; 4 -> 4 ; (True, predict=None) -> ValueError "forecast_trail= requires predict="
-1 -> ValueError(match forecast_trail) ; 'yes' -> TypeError(match) ; 2.5 -> TypeError(match)
trail_frames params ['frame','n_retained','n_frames','stride'] ; 'n_frames' referenced in body? False
trail_frames(3,16,16)=[2,1,0] ; trail_frames(14,16,16)=[13..0]
trail_alpha ages 0..5 @n=16 = [0.6,0.5694,0.5388,0.5082,0.4776,0.4471] strictly decreasing: True
live 0.6 > every trail alpha: True (min 0.1106, max 0.5694)
```

**14. Repo untouched**
```
$ git status --porcelain      # (no output)
$ git worktree list
/Users/jmanning/hypertools  4ecb3b6d [dev-1.0]
```

---

## What would flip this to IMPLEMENTABLE

1. `_series(n=1, d=max(ndims, 2))` → `_series(n=1, dims=max(ndims, 2))` (plan L1281).
2. Make `test_the_live_forecast_updates_in_both_2d_and_3d` dimension-aware in its axes lookup, so the
   `ndims=2` parametrization actually exercises the `else: art.set_data(...)` branch instead of dying
   in `_ax`.
3. `hooks.add_internal(...)` → `_frame_hooks.add_internal(...)` (plan L1581), and
   `_ndims=ndims` → `_ndims=_display_ndims`.

The five Low items (Step 2's "12", the 2564 baseline, the `ForecastDisplay` sentence, the dead
`n_frames`, the test-name mismatch) are cosmetic and can ride along. The Med Self-Review "no
placeholders" claim should simply be corrected to name Tasks 5-6 as specified-not-coded.
