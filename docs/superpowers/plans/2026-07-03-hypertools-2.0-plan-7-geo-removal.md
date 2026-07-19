# HyperTools 2.0 — Plan 7: DataGeometry removal + API finalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. **REQUIRED READING for every task:** `.superpowers/sdd/plan7-geo-removal-research.md` — the exhaustive coupling map, per-test rewrite table (§5), tools-shim inventory + importer list (§6), and sequencing (§10). This plan gives task boundaries + Jeremy's decisions; the research file gives the exact `file:line` details.

**Goal:** Delete `DataGeometry` entirely, flip `plot()`/`load()` to return figures/raw-data, vendor RSRM, retire the `tools/` shims, and keep the suite green at the start (335) and end — rewriting ~100 geo-coupled test bodies across 17 files in the process.

**Architecture:** Unlike Plans 1–6 (incremental strangler, green at every commit), geo removal is a **coordinated change**: some intermediate commits are partially red (a return-type flip can't be half-applied), converging back to fully green. Sequence per research §10. `DataGeometry` stays alive through the plot-flip so `load()`'s geo path keeps working, and is deleted only in the final step alongside the `load()` flip.

**Tech Stack:** Python 3.12 (`.venv`), pytest, datawrangler 0.5.0, matplotlib, plotly, numpy, pandas, scikit-learn.

## Jeremy's decisions (AUTHORITATIVE — 2026-07-03)

1. **Legacy example datasets → RE-HOST as plain pickles** (arrays/DataFrames/lists), NOT a private unpickle-shim. `load()` returns raw data. The regeneration script must run **while `DataGeometry` still exists** (to extract `.data` from the current hosted geo pickles); the actual **Google Drive upload is a MANUAL step Jeremy performs** — a blocking coordination point (Task 6). Until re-hosted, `test_load` downloads geo pickles, so the `load()`-flip + geo deletion cannot fully land.
2. **`plot()` return:** plain `plot()` → `fig` (static) / `(fig, animation)` (animated matplotlib) / `fig` (plotly, frames embedded). `ax = fig.axes[0]`. **`return_model=True` → a bundle `{'fig': fig, 'models': {...}, 'xform_data': [...]}`** (or a small namespace object) carrying the fitted reduce/align/cluster models AND the analyzed `xform_data`.
3. **Streaming:** attach `stream_info` as a dynamic attribute on the returned Figure — **`fig.stream_info = {'n_samples','reduce_model','truncated'}`**. Uniform Figure return; ~23 `test_streaming` assertions use `fig.stream_info`.
4. **RSRM:** VENDOR in Plan 7 (port from `jeremy/master:hypertools/external/brainiak.py` L646-1064). **Name-collision: DEFERRED** (document `from hypertools.align import HyperAlign`; not resolved here).

## Global Constraints

- **Interpreter:** ALL commands `/Users/jmanning/hypertools/.venv/bin/python`. Never bare `python`/`pip`/`pytest`. `MPLBACKEND=Agg` for pytest.
- **pandas `>=2.2.0`** (dw 0.5.0); validated on pandas 3.0.3 / numpy 2.3.5.
- **Branch** `dev-2.0-refactor`; never push master. Commit trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Don't push.
- **Green gates:** the FULL suite must be green after Tasks 1, 2, 3, and 7 (the convergence points). Tasks 4–5 may leave unrelated geo tests red (fixed by Task 7) — each such task's OWN touched-file tests must be green, and the task report must list which suite files are expectedly-red and why.
- **No mocks, real calls, eval-free.**

---

### Task 1: repoint production importers off the 9 shims (research §6/§10-S1)

**Files:** Modify `hypertools/__init__.py`, `hypertools/core/model.py`, `hypertools/plot/plot.py`, `hypertools/tools/text2mat.py`. **No test changes.** Leave the 9 shim files in place (tests still import them).

- [ ] **Step 1:** In each file, repoint SHIM imports to real locations (research §6 lists every line): `tools.load`→`io.load`, `tools.reduce`→`reduce.reduce`, `tools.describe`→`reduce.describe`, `tools.cluster`→`cluster.cluster`, `tools.apply_model`→`core.model`, `tools.streaming`→`io.streaming`. Leave `tools.analyze`/`tools.align`/`tools.normalize`/`tools.format_data` (real code) untouched. Fix `tools/text2mat.py:9` `from .load import load`→`from ..io.load import load`.
- [ ] **Step 2:** `.venv/bin/python -c "import hypertools; print('OK')"` then FULL suite: `MPLBACKEND=Agg .venv/bin/python -m pytest -q -p no:cacheprovider`. Expected: **335 passed** (pure import-path change, zero behavior change).
- [ ] **Step 3:** Commit `refactor(tools): repoint production imports off tools.* shims to real modules`.

---

### Task 2: repoint test imports + delete the 9 shims (research §6/§10-S2)

**Files:** Modify the test files in research §6 (test-code list); delete shim-parity assertions in `tests/{cluster/test_cluster_module,plot/test_colors_module,reduce/test_reduce_module,io/test_io_module}.py` (READ each full file first — keep non-parity tests); delete 9 shim files (`tools/{apply_model,cluster,colors,describe,load,procrustes,reduce,sources,streaming}.py`); prune their re-exports from `tools/__init__.py`.

- [ ] **Step 1:** Repoint every test import in research §6's test list to real locations. For the 4 shim-parity test files, read them fully and delete ONLY the `old is new` / `old_X == new_X` assertions (keep any real-behavior tests).
- [ ] **Step 2:** Delete the 9 shim files; prune `tools/__init__.py`.
- [ ] **Step 3:** `.venv/bin/python -c "import hypertools"` + FULL suite. Expected: **~335 passed** (minus any deleted parity assertions; report the new count). Every remaining test green.
- [ ] **Step 4:** Commit `refactor(tools): retire 9 re-export shims; repoint tests to real modules`.

---

### Task 3: vendor RSRM + wire RobustSharedResponseModel (research §8; geo-independent)

**Files:** Modify `hypertools/external/brainiak.py` (add RSRM), `hypertools/align/srm.py` (add adapter), `hypertools/align/__init__.py`, `hypertools/align/align.py` (ALIGNERS), `hypertools/tools/align.py` (`_ALIAS`). Test: `tests/align/test_rsrm.py`.

- [ ] **Step 1:** Write failing `tests/align/test_rsrm.py` mirroring `test_srm.py`: `RobustSharedResponseModel(features=3).fit_transform(rotated_pair)` returns a list of 2 with the requested feature dim; and `hyp.align(data, align='RSRM')` (or `'RobustSharedResponseModel'`) works.
- [ ] **Step 2:** Port the `RSRM` class from `jeremy/master:hypertools/external/brainiak.py` (L646-1064, ~418 lines) into `hypertools/external/brainiak.py`; add `"RSRM"` to `__all__`. Validate it imports and fits on real data (no mocks). Fix any numpy-2/pandas-3/sklearn-1.8 incompatibilities discovered (real test).
- [ ] **Step 3:** In `align/srm.py` add `rsrm_fitter` + `RobustSharedResponseModel(Aligner)` (mirror the SRM/DetSRM adapters); export from `align/__init__.py`; add to `align/align.py`'s `ALIGNERS`; add `'RSRM': 'RobustSharedResponseModel'` to `tools/align.py`'s `_ALIAS`.
- [ ] **Step 4:** `MPLBACKEND=Agg .venv/bin/python -m pytest tests/align/ tests/test_align.py -q` → green. Then FULL suite → green (RSRM is additive).
- [ ] **Step 5:** Commit `feat(align): vendor RSRM + RobustSharedResponseModel adapter`.

---

### Task 4: flip plot() return + return_model bundle + rewrite plot-adjacent tests (research §3/§5/§10-S4+S6; DataGeometry still exists)

**Files:** Modify `hypertools/plot/plot.py`; rewrite the plot-return assertions in `tests/{test_plot,test_plotly_trails,test_nested,test_interactive,test_regressions,test_round3,test_round4}.py` per research §5. `DataGeometry` is NOT deleted yet (load's geo path still works). Add `return_model` plumbing to `reduce/reduce.py`/`cluster/cluster.py`/`align/align.py` as needed (research §3 flags this as new plumbing).

- [ ] **Step 1:** Change `plot.py:862-875` `return DataGeometry(...)` → per Jeremy decision #2: `return fig` (static), `return fig, line_ani` (animated mpl), `return fig` (plotly). Remove the module-level `from ..datageometry import DataGeometry` in plot.py. For `return_model=True`, assemble `{'fig': fig, 'models': {'reduce':..., 'align':..., 'cluster':...}, 'xform_data': xform_data}` — thread `return_model` down to the internal `reducer()`/`clusterer()` calls (mirror `core/model.py`'s `(result, fitted)` convention) and capture align's fitted instance from `align/align.py`'s dispatcher (add a `return_model` flag there). `xform_data` is the `copy.copy(xform)` already captured at plot.py:399.
- [ ] **Step 2:** Rewrite the tests per research §5's table: `geo.fig`→`fig`, `geo.ax`→`fig.axes[0]`, `geo.line_ani`→the `ani` from `(fig, ani)`, plotly frames via `fig.frames`, `isinstance(geo, DataGeometry)`→`isinstance(fig, Figure)`. For the `xform_data`-dependent tests (`test_plot_reduce3d/2d/1d`, `test_plot_reduce_align5d/10d`, `test_fmt_list_line_interpolation_keeps_arrays`), use `return_model=True` and read `result['xform_data']` (per decision #2). `test_plot_geo` / geo-as-input tests: keep passing raw data (DataGeometry still exists so a geo input still works via format_data — but prefer rewriting to plain data; these fully retire in Task 7).
- [ ] **Step 3:** `MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_plot.py tests/test_plotly_trails.py tests/test_nested.py tests/test_interactive.py tests/test_regressions.py tests/test_round3.py tests/test_round4.py -q` → these files green. NOTE: `test_streaming.py` (Task 5), `test_geo.py`, and geo-input tests in other files may now be red — that's expected; list them in the report.
- [ ] **Step 4:** Commit `feat(plot): return figure/(fig,ani) + return_model bundle; drop DataGeometry return`.

---

### Task 5: streaming fig.stream_info + rewrite test_streaming (research §5/§10-S5; decision #3)

**Files:** Modify `hypertools/io/streaming.py` (`plot_stream`); rewrite `tests/test_streaming.py` (+ streaming-adjacent asserts in `tests/test_load_sources.py`).

- [ ] **Step 1:** `plot_stream` currently gets a geo from an internal `plot()` call then mutates it. Rework it for the new plot return: capture `fig` from `plot(..., show=False)` (now returns `fig`), get the artist via `fig.axes[0].lines`, and set `fig.stream_info = {'n_samples':..., 'reduce_model': model, 'truncated':...}` (decision #3). Return `fig`.
- [ ] **Step 2:** Rewrite `test_streaming.py`'s ~23 assertions: `geo.fig`→`fig`, `geo.ax`→`fig.axes[0]`, `geo.stream_info`→`fig.stream_info`, `geo.xform_data`/`geo.data` → recompute independently or read from `fig.stream_info` where applicable. Fix `test_load_sources.py::test_load_huggingface_streaming_flows_to_plot` to `fig.stream_info['n_samples']`.
- [ ] **Step 3:** `MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_streaming.py tests/test_load_sources.py -q` → green.
- [ ] **Step 4:** Commit `feat(io): streaming plots carry fig.stream_info; rewrite test_streaming`.

---

### Task 6: regenerate example datasets as plain pickles (BLOCKS on Jeremy's manual upload)

**Files:** Create `scripts/rehost_example_datasets.py`. **This task requires Jeremy to upload the regenerated files to Google Drive.**

- [ ] **Step 1:** Write `scripts/rehost_example_datasets.py` that (while `DataGeometry` still exists) downloads each hosted geo dataset (`weights`, `weights_avg`, `weights_sample`, `spiral`, `mushrooms`, and the `*_model` pipelines / shapes-zoo / datasaurus already-plain ones), extracts `.data` (or `.get_data()`) from the geo ones, and writes plain-pickle files (list-of-arrays or DataFrame) to a local `rehost/` dir. Print each new file's path + a manifest.
- [ ] **Step 2:** Run it (`.venv/bin/python scripts/rehost_example_datasets.py`), producing the plain pickles locally. Verify each unpickles to a plain array/DataFrame/list with NO DataGeometry.
- [ ] **Step 3:** **STOP + surface to controller:** the regenerated files must be uploaded by Jeremy to the same Drive/Dropbox IDs (or new IDs + `EXAMPLE_DATA` updated). Report the manifest + which IDs need replacing. Do NOT proceed to Task 7's `load()` flip until re-hosting is confirmed (else `test_load` downloads stale geo pickles and the flip goes red). Commit the script: `chore(scripts): add example-dataset re-host regeneration script`.

---

### Task 7: delete DataGeometry + purge _shared geo + flip load() + rewrite remaining geo tests (research §5/§7/§9/§10-S7) — GATED on Task 6 re-hosting

**Files:** Delete `hypertools/datageometry.py`, `tests/test_geo.py`. Modify `hypertools/__init__.py`, `hypertools/_shared/helpers.py`, `hypertools/tools/format_data.py`, `hypertools/tools/text2mat.py`, `hypertools/io/load.py`. Rewrite geo-input tests: `tests/test_{reduce,normalize,cluster,describe,format_data,align,procrustes,load,load_sources}.py`.

- [ ] **Step 1 (PREREQUISITE):** Confirm Task 6's re-hosted datasets are LIVE (or `EXAMPLE_DATA` IDs updated to new plain-pickle files). If not confirmed, STOP — this task is blocked.
- [ ] **Step 2:** `io/load.py`: remove `DataGeometry` import + the DataFrame-rehydration isinstance check (§9); change the `reduce`/`align`/`normalize` branch to `return d` (analyzed data) instead of `return plot(d, show=False)`; delete `_load_legacy` (or raise a clear "unsupported" error). `load()` now returns raw data / analyzed data.
- [ ] **Step 3:** Delete `hypertools/datageometry.py`; remove `from .datageometry import DataGeometry` from `hypertools/__init__.py`. In `_shared/helpers.py` remove `check_geo` (dead) + the `DataGeometry` branches in `get_type`/`get_dtype`. In `tools/format_data.py` remove the dead `DataGeometry` import + the `elif dtype=='geo':` branch. Fix `tools/text2mat.py:82` (drop `.get_data()`).
- [ ] **Step 4:** Delete `tests/test_geo.py`. Rewrite the geo-input tests per research §5 (`test_reduce_geo`, `test_normalize_geo`, `test_cluster_mixture_via_plot`, `test_describe_geo`, `test_format_data::test_geo`/`test_missing_data`, `test_align` geo lines + `test_align_geo`, `test_procrustes` load line, `test_load.py` entire file, `test_load_sources.py` geo-typed asserts + retire `test_datageometry_plot_accepts_source_strings`).
- [ ] **Step 5:** FULL suite → **must be green** (convergence point). Report the final count.
- [ ] **Step 6:** Commit `refactor!: remove DataGeometry entirely; load returns raw data; purge geo coupling`.

---

## Self-Review

**1. Spec coverage:** DataGeometry deletion (§2 decision) → Tasks 4–7; `plot()` figure return + `return_model` bundle (decision #2) → Task 4; streaming `fig.stream_info` (decision #3) → Task 5; `load()` raw-data return + re-host (decision #1) → Tasks 6–7; RSRM vendored (decision #4) → Task 3; 9 shims retired → Tasks 1–2. Name-collision explicitly DEFERRED (decision #4). ~100 test rewrites enumerated in research §5.

**2. Placeholder scan:** each task points to exact research sections for `file:line` detail + lists gates; the one genuine blocker (dataset re-hosting) is called out as a manual Jeremy step, not hidden.

**3. Sequencing integrity:** DataGeometry survives through Task 4–5 (load's geo path intact) and is deleted only in Task 7 alongside the load-flip; Tasks 1–3 are fully green; Tasks 4–5 have documented expected-red converging to green at Task 7. Re-host (Task 6) gates Task 7 only.

## Execution Handoff

After Plan 7, Plan 8 (docs/gallery/notebooks migrated to the figure-return API + Playwright visual verify + PR evidence), then whole-branch review (opus) + PR into dev-2.0. Name-collision resolution can fold into Plan 8 or a follow-up. Lift is complete: the full 2.0 class-based library with geo removed.
