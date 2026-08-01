# Plan 3 (forecast animation) — re-review of v2 against the contracts settled after it

Dispatched 2026-07-31. Plan 3 was reviewed once before (`review_plan3_forecast_animation.md`,
8 defects / 4 fatal → v2). This is the **first re-review since**, checking v2 against the contracts
decided in later Plan 1 / Plan 2 rounds: tuple-valued `FrameContext` fields, all four plotly frame
branches, spin's shared traces, `trace_data` vs `xform_data`, the both-axis ≥2-row precondition, and
`min_history` as a separate transient mechanism.

Method: every cited line number, symbol and API opened in the real source; the plan's own Task 1+2
code extracted and executed rather than reasoned about.

## Findings

| Sev | Location | What is wrong | Evidence (verified) |
|-|-|-|-|
| **Fatal** | plan3 `:1188` (T4 S5) | `_update_forecasts(frame, …)` is appended to `hooks.callbacks`, but `FrameHooks.dispatch` calls `callback(ctx)` with a **FrameContext**. `sched.polyline(i, ctx)` → `TypeError`. Must take `ctx` and use `ctx.frame` | animation-core.md:2596-2604 |
| **High** | plan3 `:470-487` | `test_to_display_maps_every_scheduled_forecast_into_the_cube` asserts on `disp.path()` = **displacement**, not `.polyline()` = position, so Contract 4 is never tested. Ran the plan's Task 1+2 code: paths peak 0.86 while polylines hit exactly ±1.000; a displacement is only bounded by ±2, so it can also fail spuriously | measured; plan3 `:481-487` |
| **High** | plan3 `:881` (T3 S7) | Expects "17 passed" for `test_predict_integration.py`. 18 − 5 (removed parametrize) + 2 = **15** | `pytest` → 18 passed; `:169-178` = 5 params |
| Med | plan3 Task 6 | Plotly parity tests only `animate=True`; no serial/window/`order='serial'`, though serial is a separate frame branch (`:2865` vs `:2975`). animation-core **Task 4** (plotly serial + trail plumbing) is an unlisted prerequisite | plotly_backend.py:2865, :2890-2897 |
| Med | plan3 `:1185` | Internal updater lands *after* user `on_frame`s in the shared list → user callbacks see the previous frame's forecast artists; also forces a FrameContext build every frame on every animated `predict=` | animation-core.md:2600 |
| Med | multiindex.md:2797 | Cites plan3's `test_forecast_dropped_under_hue_regrouping` — no such test; real name is `test_hue_regrouping_drops_forecasts_exactly_like_the_static_path` | plan3 `:1079` |
| Low | multiindex.md:64/2448 | plan3 `:289/:153/:389` → actually `:291/:155/:390` | sed |
| Low | plan3 `:306/:1421/:1918` | Counts: T1 "6 named"→7; T5 S2 "10 items"→11; T4 row "Task 2→12"→13 | counted |
| Low | plan3 `:1704`; plan4 `:118` | plotly reveal is `:2899` not `:2898`; refusal `if` is plot.py:**2346** not 2347 | source |
| Low | plan3 `:900-902` / `:793` | T4 "Files" lists matplotlib_backend.py but all code is plot.py; T3 S2 says the morph tests fail — they pass today | plan text |
| Low | env | `.venv/.../site-packages/hypertools` is a **stale non-editable 1.0.0** (no `antialias=`) | dist-info; `TypeError` on `antialias` |

**The Fatal was independently re-verified** (not taken on the reviewer's word): the plan defines
`def _update_forecasts(frame, _sched=..., _artists=..., _antialias=...)` and calls
`_sched.polyline(i, frame)`, while `FrameHooks.dispatch` does `for callback in self.callbacks:
callback(ctx)` — passing a `FrameContext`, not an index.

## Confirmed clean

- **FrameContext tuple contract honoured** (`:22`, `:78`, `:1209`) — no list-equality assertions survive.
- **`trace_data` / `xform_data`** correct: bundle measured `(4,3)` == `hyp.predict(xform_data)`.
- **`min_history` vs the precondition are not conflated** (`predict/common.py:254`).
- Task 1+2 code extracted and run verbatim → **12 passed + the expected `serial_reveal_counts`
  `ImportError`**, exactly as the plan claims.
- All spot-checked citations correct: plot.py 2338/2346/3999/4015-4032/4291/4339/1904-08/1935-41;
  matplotlib_backend 319/357/1888-90; plotly_backend 901/945/1425/2729/2819/2865/2975;
  streaming 382-401. `fig, ani` unpack and `ani._func`/`_args` are real.
- hue/cluster forecast drop measured: 0 dashed of 60 traces.

## Open decisions (README "Decisions still open / Plan 3")

| decision | status | recommendation |
|-|-|-|
| Silent forecast drop under `hue=`/`cluster=` | still open | **warn on both paths** |
| Throttling beyond memoization | still open | **memoization only for 1.1**, revisit with real market data |
| `min_history` | still open | **keep 2**, no new kwarg |
| A finished dataset's forecast under `order='serial'` | still open | **freeze** (as implemented) |
