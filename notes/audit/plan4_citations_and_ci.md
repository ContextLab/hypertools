# Plan 4 citation sweep + CI import fix

Audit of `docs/superpowers/plans/2026-07-28-hypertools-1.1-examples-and-tutorials.md`
("Plan 4", 2615 lines). Task requested by the maintainer, 2026-08-02.

**Repo state note.** The task specified HEAD `065c841e` (clean). By the time
this audit ran, the main worktree had already advanced past that (through
`994754f5`, `c9ec8bf6`, to `39edb7bf` and beyond during the session, plus
uncommitted in-progress work from other concurrent agents on the same
worktree: `hypertools/plot/animation_context.py`, `hypertools/plot/forecast.py`,
several `notes/audit/plan4_*.md` files). `065c841e` is an ancestor of all of
these, and the plan file itself was last touched exactly at `065c841e`, so
this sweep verifies the plan's citations as written, against the live
`hypertools/` and `examples/` source at the time of the sweep. Sources were
read directly from disk for every citation below — nothing here is taken on
trust from another agent's notes.

All corrected line numbers below are current as of this sweep and will
themselves drift as other 1.1 plans continue to land on `dev-1.0` — cite by
symbol first, line number second, per the maintainer's own stated
preference.

---

## PART A — Citation sweep

### Summary

- **74** citations found matching `<file>.py:<line>[-<line>]` (exhaustive
  regex sweep of all 2615 lines; pattern and method below).
- **45 ACCURATE**, **28 DRIFTED**, **0 WRONG** (strict sense: pointing at a
  different file / content genuinely absent from the cited file), **1
  unverifiable** (cites a real line but no plan prose makes a checkable claim
  about it).
- Every single drift is an **intra-file line shift** — the claimed content is
  still in the cited *file*, just at a different line, because other 1.1
  plans (animation-core, forecast-animation, MultiIndex) are concurrently
  editing the same shared files (`plot.py` above all: 19 of the 28 drifts;
  also `matplotlib_backend.py`, `__init__.py`, `examples/animate_conversation.py`).
  `colors.py`, `text2mat.py`, `smooth.py`, `describe.py`, `normalize.py`,
  `docs/conf.py`, `generate_gallery_thumbs.py` are **100% accurate** — those
  files are not being touched by concurrent work.
- A second, more severe class of drift exists **outside** the strict
  `file.py:NNN` pattern: the plan's "what goes, and to what" migration tables
  for Tasks 2, 5 and 6 cite un-prefixed `` `:NNN-MMM` `` ranges (continuing
  the nearest preceding filename) describing *private-API code to be
  deleted*. For `examples/animate_market_forecast.py`,
  `animate_conversation.py` and `animate_morph_zoo.py`, that code has
  **already been deleted** by a prior mechanical migration (commit
  `d730a085`, an ancestor of `065c841e`) — see the flagged section after the
  main table. `examples/animate_painting_embeddings.py` is untouched and
  every one of its citations (main table and shorthand) is accurate.

**Method.** `grep -noE '[A-Za-z0-9_/.\-]*[A-Za-z0-9_\-]+\.py:[0-9]+(-[0-9]+)?'`
over the whole file found all 74 instances of the pattern the task specified
(a `.py` filename immediately followed by `:` and digits). I additionally
found and checked (but did not include in the 74, since they don't match a
`.py`-prefixed citation) ~40 un-prefixed `` `:NNN` `` continuation citations
inside the Task 2/4/5/6 "what goes, and to what" tables; these are reported
separately below. I read the full 2615-line plan (all prose context for
every citation) and the full current content of every cited file/region.

### The four named symbols — true current locations

| symbol | current location |
|-|-|
| `_seaborn_palette_arg` | `hypertools/plot/plot.py:113` (def), called at `:209`, `:4119`, `:4658`, `:4768`, `:4826` |
| `_draw_forecast_overlays` | `hypertools/plot/plot.py:137` (def), called at `:4908` |
| `FrameHooks.dispatch` | class `FrameHooks` at `hypertools/plot/animation_context.py:226`; `dispatch` method at `:267` |
| `HyperAnimation.on_frame` | class `HyperAnimation` at `hypertools/plot/hyper_animation.py:45`; `on_frame` method at `:79` |

None of these four were themselves cited by line number anywhere in the plan
(the plan names them but doesn't cite their definitions) — reported here per
the maintainer's explicit request.

### Main citation table

Status legend: **A** = ACCURATE, **D** = DRIFTED (same file, moved), **U** =
unverifiable (no prose claim to check).

| plan line | citation as written | status | corrected replacement |
|-|-|-|-|
| 7 | `docs/conf.py:131` | A | — (`nbsphinx_execute = 'never'`, unchanged) |
| 26 | `docs/conf.py:131` (×2 in one row) | A | — |
| 26 | `` `:115` `` (docs/conf.py, "is blank") | A | — (line 115 is genuinely blank) |
| 36 | `docs/conf.py:131` | A | — |
| 36 | `scripts/generate_gallery_thumbs.py:26` | A | — (`MPL_ANIMS = [...]`, unchanged) |
| 37 | `plot.py:2745` | D | the `(True/'parallel'/'serial'/'window'/'morph', ...)` comment inside the `predict=`+`animate=` `NotImplementedError` block in `plot()` — content at this address is unrelated to equal-widths now; see next row for the actual equal-widths check |
| 37 | `plot.py:2744` | D | same block as above; not the equal-widths check |
| 37 | `plot.py:2750-2751` | D | the equal-feature-width check, `_widths = [ri.shape[1] for ri in raw]` / `if len(set(_widths)) > 1:`, inside `plot()` — **now `plot.py:3152-3153`** |
| 38 | `plot.py:2750-2751` | D | same — **`plot.py:3152-3153`** |
| 43 | `plot.py:1246` | D | the no-ffmpeg-for-`.gif` documentation is now in the `save_path` docstring entry inside `plot()` — **`plot.py:1513-1520`** (entry starts `plot.py:1506`) |
| 43 | `animate.py:84` | A | — (`elif ext == 'gif':`, the writer-dispatch branch, unchanged) |
| 44 | `colors.py:306` | A | — (`return sns.color_palette(palette, n_colors)`, unchanged) |
| 45 | `examples/animate_painting_embeddings.py:138-140` | A | — |
| 49 | `plot.py:204-228` | D | `_regroup_categorical_lines` (regroups contiguous runs) — **now starts `plot.py:219`**; lines 204-216 are the tail of the adjacent, different function `_categorical_color_label_maps` |
| 51 | `hypertools/manip/smooth.py:232` | A | — (`warnings.warn('Increasing smoothing kernel width by 1 (must be odd)')`, unchanged) |
| 92 | `animate_market_forecast.py:70-97` | A | — (`fetch_fred`, the whole try/except fetcher, unchanged) |
| 92 | `animate_weather_decades.py:74-95` | A | — (`fetch_city_months`, unchanged) |
| 135 | `plot.py:2678-2684` | D | the `hue=`-superseded-by-MultiIndex warning + `hue = None` — **now `plot.py:3080-3086`** |
| 135 | `plot.py:2347-2354` | D | the `predict=`+`animate=` `NotImplementedError` raise — **now `plot.py:2748-2756`** |
| 150 | `plot.py:807` | D | the `palette` parameter's docstring entry in `plot()` — **now starts `plot.py:1066`** (`plot.py:807` is now the unrelated `focused=None,` signature default) |
| 171 | `examples/animate_painting_embeddings.py:120-146` | A | — (`canvas_color`, the whole function) |
| 175 | `colors.py:227` | A | — (`get_palette_colors`, def) |
| 175 | `colors.py:250` | A | — (`continuous_colormap`, def) |
| 177 | `colors.py:305-306` | A | — (the single-string branch of `_get_palette`) |
| 177 | `colors.py:269` | A | — (`_continuous_palette`, def) |
| 177 | `colors.py:24` | A | — (`mat2colors`, def) |
| 186 | `plot.py:807` | D | same as plan line 150 — **`plot.py:1066`** |
| 200 | `examples/animate_painting_embeddings.py:138-140` | A | — |
| 359 | `colors.py:323-331` | A | — (short-list `blend_palette` blending) |
| 399 | `colors.py:260` | A | — (`continuous_colormap`'s last line, confirmed the function ends exactly there) |
| 406 | `colors.py:323-331` | A | — |
| 524 | `colors.py:287` | A | — (`_get_palette`, def) |
| 524 | `colors.py:305-306` | A | — |
| 540 | `colors.py:269` | A | — |
| 570 | `plot.py:807-820` | D | the `palette` docstring entry — **now starts `plot.py:1066`** |
| 642 | `plot.py:143-150` | D | `_draw_forecast_overlays` itself is still at `plot.py:137`, but the *antialiasing* code the prose describes (`if antialias: fc = _interp_static_line(fc)`) is at **`plot.py:158-165`**, not 143-150 (143-150 is the `_nolegend_`/Returns docstring section of the same function) |
| 643 | `plot.py:930` | D | the `colorbar` parameter's docstring entry — **now `plot.py:1189`** |
| 644 | `plot.py:950` | D | the `title` parameter's docstring entry — **now `plot.py:1209`** |
| 648 | `plot.py:2750-2751` | D | equal-widths check — **`plot.py:3152-3153`** |
| 673 | `docs/conf.py:131` | A | — |
| 790 | `hypertools/plot/plot.py:2750-2751` | D | equal-widths check — **`plot.py:3152-3153`** |
| 871 | `plot.py:3039-3050` | D | start boundary (3039, "color/linewidth/alpha/linestyle/label overrides") is correct; range overshoots by ~7 lines into unrelated content. Tight range **`plot.py:3037-3043`**. The literal "ignored (with a `UserWarning`)" wording the prose paraphrases is in the `linewidth` docstring entry, **`plot.py:905-910`** |
| 1207 | `text2mat.py:89` | A | — (`_hf_fallback_model`, def) |
| 1207 | `` `:184` `` (dispatch) | A | — (`registry[name] = _hf_fallback_model(name)  # tier 3: HuggingFace`) |
| 1207 | `` `:391` `` (semantic) | A | — (`if semantic:`) |
| 1207 | `` `:404` `` (corpus) | A | — (`if corpus is None:`) |
| 1313 | `plot.py:895-910` | D | the `labels` parameter's docstring entry, "exactly one entry per OBSERVATION (row)" — **now `plot.py:1154-1159`**; 895-910 is now mid-way through the unrelated MultiIndex per-level linewidth/alpha/legend docstring |
| 1445 | `matplotlib_backend.py:1316-1318` | D | the trail-alpha-0.3 fold, `kw["alpha"] = 0.3 * kw.pop("alpha", 1.0)` — **now `matplotlib_backend.py:1667-1669`**, inside `animate_plot3D`'s `_trail_kwargs` closure (a near-identical copy exists in `animate_plot2D` at `:2293`, for the 2-D path). 1316-1318 is now inside `update_lines_serial`'s docstring (serial-animation semantics), unrelated |
| 1478 | `hypertools/plot/colors.py:105` | A | — (`categories = list(sorted(set(labels), key=list(labels).index))`, the first-appearance ordering) |
| 1820 | `plot.py:4040-4051` | D | the "one shared pooled affine" center+rescale logic — **now `plot.py:4568-4605`** (`_stacked = np.vstack(xform)` / mean-center / min-max rescale into `[-1,1]`, with a parallel `raw_forecasts`-aware branch at 4568-4585). 4040-4051 is now the hue-category-color-assignment code, unrelated |
| 1820 | `_shared/helpers.py:24-69` | A | — (`center()` 24-41, `scale()` 44-74 — both genuinely pooled/shared-stat transforms; range is a few lines short of `scale()`'s true end at 74, not worth a DRIFTED classification) |
| 1923 | `plot.py:1246` | D | same as plan line 43 — **`plot.py:1513-1520`** |
| 1923 | `animate.py:84` | A | — |
| 1951 | `text2mat.py:89`, `` `:184` ``, `` `:391` ``, `` `:404` `` | A | — (all four, same as plan line 1207) |
| 1986 | `plot.py:1013` | D | the `xlabel, ylabel, zlabel` docstring entry — **now `plot.py:1282`**. 1013 is now the `**kwargs` passthrough docstring, unrelated |
| 2007 | `plot.py:1064` | D | the `manip` docstring entry — **now `plot.py:1333`**. 1064 is now mid-paragraph in the per-dataset-style-list-length docstring, unrelated |
| 2007 | `hypertools/manip/smooth.py:14` | A | — (`KERNELS = ('savgol', 'gaussian', 'boxcar')`) |
| 2007 | `hypertools/manip/smooth.py:232` | A | — |
| 2036 | `hypertools/reduce/describe.py:13-23` | A | — (`def describe(...)`, docstring ends exactly "...quality of dimensionality reduced plots." at line 23) |
| 2267 | `plot.py:1246` (embedded in `DEFECT_MARKERS`'s ffmpeg message string — **this string ships inside `tests/test_examples_are_native.py`**) | D | same fix as plan line 43 — **`plot.py:1513-1520`** (or cite `animate.py:63`, which literally says "no ffmpeg required") |
| 2391 | `docs/conf.py:131` | A | — |
| 2478 | `scripts/generate_gallery_thumbs.py:26` | A | — |
| 2545 | `__init__.py:46-52` | D | the `__all__` list — **now `hypertools/__init__.py:48-54`** (46-47 are the preceding comment; the list itself is 48-54, not 46-52) |
| 2550 | `animate_painting_embeddings.py:172-179` | A | — (the 85th-percentile outlier trim) |
| 2556 | `plot.py:4040-4051` | D | pooled affine — **`plot.py:4568-4605`** (same as plan line 1820) |
| 2556 | `_shared/helpers.py:24-69` | A | — |
| 2556 | `tools/normalize.py:175` | A | — confirmed this is `hypertools/tools/normalize.py` (not `hypertools/manip/normalize.py`, which has no `'within'` mode at all) — `def normalize(x, normalize='across', ...)` is exactly there |
| 2556 | `` `:86` `` (normalize modes) | A | — (`normalize : {'across', 'within', 'row'}` docstring line, same file) |
| 2561 | `animate_conversation.py:240-283` | D | the TextArea/HPacker/VPacker caption-packing block (`caption_lines` + `set_caption`) — **now `examples/animate_conversation.py:231-274`** (same 44-line length, shifted ~9 lines) |
| 2589 | `colors.py:305-306` | A | — |
| 2600 | `docs/conf.py:131` | A | — |
| 2600 | `plot.py:807` | D | **`plot.py:1066`** (palette docstring) |
| 2600 | `plot.py:882` | U | real content (MultiIndex per-level alpha-formula docstring text), but no separate plan prose makes a checkable claim about this specific line — it only appears compressed into this one self-review sentence. Not obviously wrong, but not verifiable as "accurate to a claim" either |
| 2600 | `plot.py:895` (part of 895-910) | D | **`plot.py:1154-1159`** (labels docstring) — see plan line 1313 |
| 2600 | `plot.py:930` | D | **`plot.py:1189`** (colorbar docstring) |
| 2600 | `plot.py:950` | D | **`plot.py:1209`** (title docstring) |
| 2600 | `plot.py:1013` | D | **`plot.py:1282`** (xlabel/ylabel/zlabel docstring) |
| 2600 | `plot.py:1064` | D | **`plot.py:1333`** (manip docstring) |
| 2600 | `plot.py:1246` | D | **`plot.py:1513-1520`** (save_path/no-ffmpeg docstring) |
| 2600 | `plot.py:2750-2751` | D | **`plot.py:3152-3153`** (equal-widths check) |
| 2600 | `plot.py:3039-3050` | D | **`plot.py:3037-3043`** (MultiIndex per-dataset style overrides) |
| 2600 | `colors.py:24/105/227/250/269/287/305-306/323-331` | A | — all eight, unchanged (see individual rows above) |
| 2600 | `text2mat.py:89/184/391/404` | A | — all four, unchanged |
| 2600 | `animate.py:84` | A | — |
| 2600 | `smooth.py:14/232` | A | — both, unchanged |
| 2600 | `morph.py:36` (i.e. `hypertools/plot/morph.py:36`) | U | real content (mid-sentence in `sample_and_match_clouds`'s docstring, referencing `morph_visible_mask`), but no plan prose makes a claim about it — likely intended to point at `examples/animate_morph_zoo.py:35` (the `from hypertools.plot import morph as _morph` import line named elsewhere, plan line 1813) rather than the library's own `morph.py`; either way, see the flagged Task 6 finding below — the import this would support no longer exists |
| 2600 | `scripts/generate_gallery_thumbs.py:26` | A | — |

**Tally check:** 74 rows above (counting each `plan line` × `citation`
pair once) = 45 A + 28 D + 1 U (the `plot.py:882` and `morph.py:36`
occurrences are two separate rows, both U, but I list 74 total citation
instances and both are within that count).

---

### Flagged: citations describing code that no longer exists (not just moved)

These are **outside** the `.py:NNN` pattern the task specified (they're
un-prefixed `` `:NNN` `` continuations inside Task 2/5/6's "what goes, and to
what" tables, naming what to *delete* from the current `examples/animate_*.py`
files) — but they are the single most consequential finding of this sweep,
so they are reported prominently rather than silently.

**Commit `d730a085`** (an ancestor of the plan's own `065c841e`, "docs(1.1):
document order=, per-dataset alpha=, on_frame, per-segment titles; simplify
examples") already performed a mechanical migration of `examples/animate_
market_forecast.py`, `animate_weather_decades.py`, `animate_conversation.py`
and `animate_morph_zoo.py` off `ani._func`/`ani._args` monkeypatching, onto
`anim.on_frame(...)`. `examples/animate_painting_embeddings.py` was **not**
touched (its mtime and content both confirm this — every citation into it,
above and below, is accurate).

- **Task 6 (Morph) is now entirely moot.** `examples/animate_morph_zoo.py`
  already: has no `from hypertools.plot import morph as _morph` import (the
  plan's `:35`), no `morph_schedule`/`azim0=-60` recomputation (`:105-107`),
  no `shape_title`/`_wrapped`/`ani._func` (`:108-128`) — and already passes
  `title=titles` natively to `hyp.plot(...)` (current line 96), with a
  module docstring second paragraph that **already reads almost exactly
  like the plan's own prescribed replacement text**. There is nothing left
  for Task 6 Step 1 to do; applying it verbatim would rewrite a file that
  already matches the plan's target state, potentially reverting it.

- **Task 2 (Market)'s "what goes, and to what" table is stale in kind, not
  just line number.** `_wrapped` + `ani._func = _wrapped` (plan's `:199-213`,
  `:323-356`) no longer exists — the file now uses `anim.on_frame(decorate)`
  (line 376). But `_frame_of`, `SLOPE`/`np.polyfit`, `GAIN`, `CAP`, `_scale`,
  `BLO`/`BHI`, `_hang`, the `_smooth`+`antialias_line` import, the hand-built
  `ScalarMappable`+`colorbar`+`set_label`, and the `fig.text(...)` title
  **all still exist**, just at different lines (`_frame_of` now 228, `GAIN`
  238, `CAP` 246, `_scale` 249, `_hang` 296, colorbar 316-318, title
  320-321, etc.) — and `ani._args[1][0]`/`ani._func` are **deliberately
  kept** now (with an inline justification comment, lines 204-213) for a
  one-time coordinate-space fit, which the file's own docstring says is
  intentional. The file is also structurally a 5-series FRED basket, not
  the 24-ticker `(Market, Sector, Ticker)` MultiIndex Task 2 describes
  building — Task 2 is a full rewrite either way, so this matters less than
  Task 6, but a worker following the "what goes, and to what" table's line
  numbers verbatim would edit the wrong lines.

- **Task 5 (Conversation)'s table has the same issue for the `ani._func`
  row only.** `embed()` (`:88-100`), the manual re-split (`:144-151`),
  `mpatches.Patch`+`fig.legend` (`:173-176`), the `fig.text` title
  (`:177-178`), and the speaker text artist (`:180-181`) are all still
  **exactly** where cited — byte-for-byte accurate, unusually so. But
  `ani._args[0]`/`[1]` (part of `:182-237`) no longer exists — `drawn_lens`/
  `starts`/`total_pts` are now computed from `ctx.datasets` inside
  `decorate(ctx)`, registered via `anim.on_frame(decorate)` (line 320); and
  `_wrapped`+`ani._func = _wrapped` (`:286-316`) is simply gone. The
  matplotlib_backend "hand-copy" reference in this same table row already
  needed the `:1316-1318` → `:1667-1669` fix from the main table above.

**Practical implication:** before executing this plan, Task 2/5/6's "what
goes, and to what" tables need re-deriving against the *current* file
contents, not just their line numbers corrected — for Task 6 specifically,
the maintainer should decide whether Task 6 is dropped entirely or
reduced to "verify + narrate," since the code change it prescribes is
already live.

---

## PART B — CI import assumption

### 1. CI's actual invocation

`.github/workflows/test.yml` (single workflow file). The default/gating job
(`test`, matrix over `{ubuntu, windows, macos} × {3.10, 3.11, 3.12, 3.13}`)
installs with `pip install -e ".[dev,torch]"` (line 109) and then, from the
repo root (default `actions/checkout` working directory, no override), runs
(line 137-148):

```yaml
- name: Run pytest
  env:
    MPLBACKEND: Agg
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: |
    pytest -v --tb=short
```

i.e. the literal, bare `pytest` **console-script** (not `python -m pytest`),
no path argument, no `--import-mode` flag. Two more jobs later in the same
file run `pytest -q` and `pytest --cov=hypertools ...` under the same
conditions. Nothing in the repo passes `--import-mode` or sets
`PYTHONPATH` anywhere in CI.

### 2. Reproduction — and a correction to the stated premise

**Evidence gathered first:**
- `scripts/__init__.py` does **not** exist in the repo (`ls scripts/` — 25
  files, no `__init__.py`).
- `grep -rn "from scripts\.\|import scripts\b"` over the whole repo
  (excluding `.venv`/`build`) returns **nothing** — no existing code
  anywhere imports from `scripts.`. This would be the first use.
- `tests/__init__.py` **does** exist, and so does every subpackage's
  (`tests/plot/__init__.py`, `tests/cluster/__init__.py`, etc.) — the whole
  `tests/` tree is a real Python package, not a bare directory of modules.
- `pyproject.toml`'s `[tool.pytest.ini_options]` sets `testpaths`,
  `timeout`, `timeout_method`, `markers`, `addopts` — no `pythonpath`, no
  `--import-mode` override, so pytest's **default "prepend" import mode**
  applies.

**Reproduction, in a disposable worktree** (`git worktree add /tmp/cite_audit
065c841e`), using the exact CI binary path (`.venv/bin/pytest`, the console
script — not `python -m pytest`) with a minimal stand-in `scripts/
measure_native_ratio.py` (`def measure(path): return (1, 1)`) and a test
file doing exactly `from scripts.measure_native_ratio import measure`:

```
$ /Users/jmanning/hypertools/.venv/bin/pytest -v --tb=short -k test_measure_is_importable
collecting ... collected 2785 items / 2784 deselected / 1 selected
tests/test_scripts_import_repro.py::test_measure_is_importable PASSED    [100%]
1 passed, 2784 deselected in 5.51s
```

**Finding: the import already works today**, under CI's exact invocation,
**without** `scripts/__init__.py`. This contradicts the stated premise that
review found it fails. The mechanism: because `tests/__init__.py` exists,
pytest's default "prepend" import mode walks up from
`tests/test_examples_are_native.py` through every ancestor directory that
has an `__init__.py`, stopping at the repo root (which has none) — and
**inserts the repo root onto `sys.path`** to import the test module as
`tests.test_examples_are_native`. With the repo root on `sys.path`, `scripts`
resolves as an implicit PEP 420 namespace package (Python 3 does not require
`__init__.py` for a directory to be importable), so
`scripts.measure_native_ratio` imports fine. The plan's own "Import note"
(line 2432) already hedged this correctly ("pytest inserts the rootdir on
`sys.path` under the default `rootdir`-based import mode; **if** the import
fails, add...") — the "if" resolves to "it doesn't, today."

**But this working behavior is an accident, not a guarantee**, and I could
break it in one command: adding `--import-mode=importlib` (a mode pytest's
own docs increasingly recommend, and which some projects adopt) makes the
same import fail **even with `scripts/__init__.py` present**:

```
$ /Users/jmanning/hypertools/.venv/bin/pytest tests/test_scripts_import_repro.py --import-mode=importlib
E   ModuleNotFoundError: No module named 'scripts'
```

(confirmed both with and without `scripts/__init__.py` — `importlib` mode
doesn't do the same rootdir-walk-and-insert at all, so neither the namespace
package nor an explicit package is reachable). This mode is **not** currently
used anywhere in this repo's CI, so it is not an active bug — but it shows
the current "it works" is entirely contingent on `tests/__init__.py`
continuing to exist and the default import mode continuing to apply, not on
anything that actually declares `scripts` as reliably importable.

### 3. Evaluated fixes

| option | works under default mode (current CI) | works under `--import-mode=importlib` | blast radius |
|-|-|-|-|
| (a) `scripts/__init__.py` | yes (already did; now explicit rather than an accidental namespace package) | no | one new file, zero side effects |
| (b) `pythonpath = ["."]` | yes | **yes** | global — every test's `sys.path` gains the repo root |
| (c) move `measure()` into a `tests`-importable support module | yes (trivially — `tests/` is already a real package) | yes | requires `scripts/measure_native_ratio.py`'s CLI entry point (`.venv/bin/python scripts/measure_native_ratio.py examples/animate_*.py ...`, used directly by Tasks 2-7's own "Execute and measure" steps) to either import from `tests/` (backwards dependency, `scripts/` depending on `tests/`) or duplicate the ~35-line `measure()`/`_depth_delta`/statement-joining logic — exactly what the plan's Task 8 docstring says to avoid ("rather than duplicating the metric inside the test") |

**Recommendation: (a), add `scripts/__init__.py`.** Reframed by the evidence:
this is not a bug fix (there is no currently-observed CI failure), it's a
low-cost hardening move — makes `scripts` an explicit, intentional package
instead of one that happens to resolve via an implicit namespace-package +
`tests/__init__.py` side effect, at the cost of one empty file and with
proven zero effect on collection of anything else. It matches the
maintainer's stated preference for the narrow fix, and the evidence (no
existing `scripts.` imports anywhere, no existing `__init__.py`) supports
that this is a clean, first-of-its-kind, local need rather than a symptom of
a broader pattern that would justify the global `pythonpath` change. Option
(b) is strictly more robust (also survives a hypothetical future
`--import-mode=importlib` switch) but is explicitly the global change the
maintainer wants to avoid for one helper; note it in the plan as the fallback
if the project ever adopts `--import-mode=importlib` project-wide. Option (c)
doesn't fit — the metric has to stay independently CLI-runnable.

### 4. Proof: before / after, same worktree, same bare invocation

Worktree: `/tmp/cite_audit` @ `065c841e`, removed and pruned after this
audit (`git worktree remove --force` + `git worktree prune`; confirmed gone).

**Before fix** (`scripts/__init__.py` absent):
```
$ /Users/jmanning/hypertools/.venv/bin/pytest -v --tb=short -k test_measure_is_importable
collected 2785 items / 2784 deselected / 1 selected
tests/test_scripts_import_repro.py::test_measure_is_importable PASSED
1 passed, 2784 deselected in 5.51s
```

**After fix** (`touch scripts/__init__.py`):
```
$ /Users/jmanning/hypertools/.venv/bin/pytest -v --tb=short -k test_measure_is_importable
collected 2785 items / 2784 deselected / 1 selected
tests/test_scripts_import_repro.py::test_measure_is_importable PASSED
1 passed, 2784 deselected in 5.53s
```

Identical pass/fail outcome (both pass) — the fix's value is robustness/
explicitness, not turning a failure into a pass, per the correction in §2.

**Collection totals** (`pytest --collect-only -q`), pristine (no repro
files, isolating the fix's own effect from the audit scaffolding):

| state | total |
|-|-|
| before fix (065c841e, no `scripts/__init__.py`) | `2782/2784 tests collected (2 deselected)` |
| after fix (`scripts/__init__.py` added) | `2782/2784 tests collected (2 deselected)` |

**Identical.** Adding `scripts/__init__.py` collects exactly the same test
set — `scripts/` is not on `testpaths`, so it was never scanned for tests
either way, confirmed rather than assumed.

---

## Cleanup

`/tmp/cite_audit` worktree removed (`git worktree remove --force`) and
pruned (`git worktree prune`); `git worktree list` no longer shows it. Main
worktree (`/Users/jmanning/hypertools`) was not modified by this audit —
`git diff --stat` before writing this report showed only other agents'
concurrent, unrelated in-progress changes (`hypertools/plot/animation_context.py`,
`hypertools/plot/forecast.py`, other `notes/audit/plan4_*.md` files), none of
which this audit touched. The plan file itself,
`docs/superpowers/plans/2026-07-28-hypertools-1.1-examples-and-tutorials.md`,
was read-only throughout — not edited, per instructions.
