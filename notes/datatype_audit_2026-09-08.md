# Datatype-handling audit (2026-09-08)

Jeremy: functions doing their own datatype checking should defer to datawrangler (wrangle/funnel/zoo predicates) so polars and future formats come for free; check EVERYWHERE, not only new code.

Read-only survey by a Claude subagent on the working tree at c700c85f + round-7/8 fixes (before any refactor). Line numbers will drift.

## A. datawrangler contract (installed: **pydata-wrangler 0.5.1**, `.venv/lib/python3.12/site-packages/datawrangler/`)

- **Polars: yes** — 67 hits across 10 files, incl. `zoo/polars_dataframe.py` (`is_polars_dataframe`, `is_polars_lazyframe`, `pandas_to_polars`, `create_polars_dataframe`).
- `dw.wrangle(x, return_dtype=False, backend=None, **kwargs)` — `zoo/format.py`. Accepts arrays, pandas/Polars DataFrames, LazyFrames, dataframe-like duck types, str/text/corpora, file paths & URLs, nested/mixed lists. kwargs: `array_kwargs`, `dataframe_kwargs`, `text_kwargs`, `null_kwargs`; `backend='pandas'|'polars'`. Format priority comes from config `supported_formats.types`.
- `dw.funnel` (`decorate/decorate.py:234`) — coerces every positional input to DataFrame(s); passes through `backend=`. Also `list_generalizer`, `apply_stacked`, `apply_unstacked`, `interpolate` (polars-aware, lines 306-337), `stack`/`unstack`.
- **`dw.zoo` real names**: `wrangle, is_array, wrangle_array, is_dataframe, wrangle_dataframe, is_multiindex_dataframe, is_null, wrangle_null, is_text, wrangle_text, get_corpus, apply_text_model, get_text_model, to_str_list, get_text, dataframe_like, array_like`. `is_dataframe` explicitly handles pandas + modin + polars DataFrame/LazyFrame + duck-typed.
- `dw.util`: `btwn, dataframe_like, array_like, depth`. `dw.core`: `get_default_options, apply_defaults, update_dict, __version__, set_dataframe_backend, get_dataframe_backend, reset_dataframe_backend`.
- hypertools currently uses **only** `dw.zoo.is_multiindex_dataframe` (3 sites) — `is_array`/`is_dataframe`/`is_text`/`array_like`/`dataframe_like`/`backend=` are **never** used.

## B. hypertools sites

775 `isinstance(` total; **482** are datatype checks. Roughly **~85 are REPLACEABLE** (class 1), ~395 legit (class 2: model specs, Colormap, str kwargs, scalars).

Replaceable per module: plot 24 · tools 12 · predict 9 · impute 9 · core 7 · manip 6 · io 6 · _shared 5 · align 4 · reduce 2 · cluster 1.

Top 25 replaceable:

1. `/Users/jmanning/hypertools/hypertools/_shared/helpers.py:530` — `get_type` isinstance ladder dispatcher
2. `_shared/helpers.py:619` — `get_dtype` duplicate type ladder
3. `tools/format_data.py:212` — Series→numpy on user input
4. `tools/format_data.py:34` — `_contains_dataset` recursive type test
5. `tools/format_data.py:264` — per-element Series coercion loop
6. `core/shared.py:27` — `as_dataframe` hand-rolled DataFrame coercion
7. `predict/predict.py:87` — `_normalize_data` ndarray branch guard
8. `predict/predict.py:63` — `_coerce_dataset` Series/1-D reshape
9. `impute/impute.py:94` — `_normalize_data` ndarray branch guard
10. `impute/impute.py:57` — `_coerce_dataset` Series/1-D reshape
11. `impute/impute.py:159` — post-funnel all-DataFrame re-check (**double work**)
12. `impute/impute.py:225` — second post-funnel DataFrame re-check
13. `align/align.py:211` — manual DataFrame rewrap of datasets
14. `align/align.py:179` — post-funnel re-wrap after format_data
15. `manip/manip.py:69` — Series/empty-shape guard on wrangled data
16. `manip/delay.py:103` — DataFrame re-check after funnel (**double work**)
17. `manip/smooth.py:224` — DataFrame re-check after funnel (**double work**)
18. `plot/plot.py:1851` — `_capture_row_indices` pandas-only index capture
19. `plot/plot.py:1883` — `_capture_column_names` pandas-only column capture
20. `plot/plot.py:9167` — hue matrix DataFrame/Series column capture
21. `plot/plot.py:3267` — `_panel_frame` Series/DataFrame relabel
22. `plot/colors.py:105` — hue matrix `.values` extraction
23. `plot/plot.py:2063` — `.values` on dataset items
24. `io/streaming.py:78` — `is_stream` negative type whitelist
25. `tools/damage.py:31` — `to_numpy()`-vs-`asarray` dispatch

Other notable: `plot/plot.py:3313, 3403, 6864, 7640, 7664, 7688`, `core/pipeline.py:296/298/633`, `core/hierarchy.py:87/104/140`, `tools/stack.py:39/45/291`, `align/procrustes.py:117-119` (`hasattr(x,'values')`), `io/save.py:279/298`, `plot/fonts.py:192`, `tools/text2mat.py:532/551`.

**Double-work / re-check-after-wrangle** (highest-value fixes): `impute/impute.py:159,225`; `manip/delay.py:103`; `manip/smooth.py:224`; `align/align.py:179-211` (funnel → `format_data` → manual `pd.DataFrame(np.asarray(...))` rewrap).

**Wrangled fn still accepting raw**: `predict/common.py:325,500` and `impute/common.py:134,193` call `_as_dataframe` on data the dispatcher already funneled.

**Class 3 UNCLEAR**: `io/streaming.py:78` (`is_stream` — a Polars LazyFrame is neither list nor iterator, so it would be *misread as a stream*); `plot/plot.py:6740` (`hasattr(_xf,'shape')`); `tools/text2mat.py:339`.

## C. Entry points

`__all__` = plot, analyze, reduce, align, normalize, describe, cluster, manip, predict, impute, load, save, apply_model, supported_models, Pipeline, set_interactive_backend, set_autoinstall, HyperAnimation, FrameContext, io, 5 exception classes, damage, stack, text_windows, subplots.

| entry | data path |
|---|---|
| `plot` (`plot/plot.py:3947`) | **format_data only** (line 7816); no dw at all — `plot/` never imports datawrangler |
| `align` | `@dw.decorate.funnel` on `_align` (align.py:234) **then** `format_data` (`_apply_format_data`) |
| `manip` | `@dw.decorate.funnel` (manip.py:114) |
| `impute` | `@dw.decorate.funnel` (impute.py:165,184) after custom `_normalize_data` |
| `predict` | `@dw.decorate.funnel` (predict.py:230,250) after custom `_normalize_data` |
| `reduce` | **format_data only** (reduce.py:250,421) |
| `cluster` | **format_data only** (cluster.py:477) |
| `normalize` (tools) | **format_data only** (normalize.py:324) |
| `analyze` | format_data (analyze.py:21) then per-stage |
| `describe`, `damage`, `stack`, `text_windows`, `apply_model`, `Pipeline` | **neither** — own isinstance ladders (`core/model.py:186` uses format_data) |
| `load`/`save`/`io` | neither — own pandas checks |

**Tests to model a polars test on**: `tests/core/test_dw_probe.py:82` (`test_funnel_accepts_polars`, already exists — the only polars test in the repo), `tests/test_dataset_compat.py`, `tests/test_format_data.py:19/47/93` (df, mixed list, column reordering), `tests/test_input_coercion_hardening.py:31-54` (1-D array, flat list, tuple/Series).

## Plan (Claude)

1. Wave 0: make `tools/format_data.py` the ONE coercion point built on `dw.wrangle` (arrays, pandas/polars DataFrames and LazyFrames, dataframe-likes, text) with `dw.zoo.is_dataframe/is_array/is_text` replacing the isinstance ladders in `_shared/helpers.get_type/get_dtype`, `core/shared.as_dataframe`, `predict/impute _coerce_dataset/_normalize_data`; `io/streaming.is_stream` must not misread a LazyFrame as a stream.
2. Wave 1 (parallel, exclusive files): plot.py index/column capture (`_capture_row_indices`, `_capture_column_names`, hue-matrix capture, `_panel_frame`), colors.py hue-matrix `.values`, tools (damage/stack/text2mat), align (funnel then format_data rewrap), manip (post-funnel re-checks), core (pipeline/hierarchy/model), io (save/load own pandas checks).
3. Tests: a polars input test per public entry point (modelled on tests/core/test_dw_probe.py::test_funnel_accepts_polars and tests/test_format_data.py), real polars (add to the dev extra if not present), plus a static gate that no module outside format_data/dw shims uses `isinstance(x, (np.ndarray, pd.DataFrame, pd.Series))` on user data.
4. Docs: 'Input data' section listing accepted forms incl. polars; CHANGELOG.
