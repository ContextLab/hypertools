# PR Evidence — HyperTools 2.0 Refactor (`dev-2.0-refactor`)

Factual summary of the class-based refactor (Plans 1-7) and docs migration (Plan 8),
assembled for the PR into `dev-2.0`. `dev-2.0..HEAD` = 55 commits.

## Test counts

- Baseline (pre-refactor, `dev-2.0`, pandas 2.3.3): **242 passed**.
- After Plans 1-7 (class-based refactor complete, dw 0.5.0 / pandas 3.0.3): **318 passed**,
  0 failed (independent controller gate run `b7tw3iusu`, 12m30s).
- After Plan 8 (docs migration + Jeremy's 4 review fixes, incl. the gif full-frame-pacing
  fix and its regression test): **325 passed** + the gif-pacing test
  (`test_animation_export`, 9 passed in that file), 0 failed.
- This task (Plan 8 Task 5) added no library tests — it is a docs-verification script only
  (`scripts/verify_docs_playwright.py`) and does not touch `hypertools/` or `tests/`.

## Dependency upgrade

- `datawrangler` (`pydata-wrangler`) **0.4.0 -> 0.5.0**. Upgrade fixed dw#30/#31
  (pandas-3 type-detection in `is_dataframe`/`is_multiindex_dataframe` breaking
  `stack`/`unstack`).
- The **`pandas<3` ceiling has been lifted**. `pyproject.toml` now declares
  `pandas>=2.2.0` (no upper bound) and `pydata-wrangler>=0.5.0`. CI (`test.yml`) gained a
  pinned-pandas-3 acceptance gate on ubuntu/py3.12. Verified on both dw0.5/pandas2.3.3 and
  dw0.5/pandas3.0.3 with 0 regressions.

## Geo (`DataGeometry`) removal

- `hyp.plot()` now returns a plain `matplotlib.figure.Figure` (static / plotly) or
  `(fig, animation)` for matplotlib animations, instead of a `DataGeometry`.
  `return_model=True` bundles `{'fig', 'models', 'xform_data'}`.
- `hyp.load()` now returns **raw/analyzed data** directly (list of arrays / DataFrame),
  not a `DataGeometry`.
- `DataGeometry` is **hidden**: trimmed to a minimal internal unpickle-only class at its
  original import path (`hypertools/datageometry.py`, ~30 lines) so the 6 hosted
  legacy-pickle example datasets (spiral/weights/weights_avg/weights_sample/mushrooms/wiki)
  still unpickle correctly. It is no longer exported as `hyp.DataGeometry` and carries an
  "internal, not public API" docstring. Streaming plots (`hyp.plot()` on a stream) attach a
  `fig.stream_info` dict (`data`, `xform_data`, `n_samples`, `reduce_model`, `truncated`)
  to the returned `Figure` instead of mutating a geo.
- All 9 dead `tools/*.py` compatibility shims (`apply_model`, `cluster`, `colors`,
  `describe`, `load`, `procrustes`, `reduce`, `sources`, `streaming`) were deleted; the 7
  real ones (`align`, `analyze`, `df2mat`, `format_data`, `missing_inds`, `normalize`,
  `text2mat`) remain as the documented classic-callable API surface.

## Jeremy's 4 review fixes (docs review pass, 2026-07-04)

1. **3D legend clipped off-canvas** — legend was always placed to the right, but long
   labels pushed it past the plot's x1 boundary; fixed layout so it stays in-canvas on
   both backends.
2. **No gallery example for shape morphing** — added `examples/plot_shape_morph.py`
   (zoo-shape morph animation), referencing the existing
   `scripts/generate_shape_morph.py` asset pipeline.
3. **Plotly `','` format string rendered as solid lines, not markers** — root-caused to
   `plotly_backend.py`'s `_MARKER_SYMBOLS` table being a hand-picked subset of matplotlib
   marker characters that omitted `,` (and `1 2 3 4 8 P X | _`); completed the table from
   `matplotlib.lines.Line2D.markers` so all 24 mpl marker chars map to Plotly
   `mode='markers'`. This directly affects Colab, since Colab's HTML output defaults to the
   plotly backend.
4. **Exported GIFs ~6x too fast** — animation export (both plotly and matplotlib backends)
   was subsampling frames for file size without compensating per-frame delay, collapsing
   total playback time (e.g., 75 frames / 4.5s instead of the intended ~27-30s). Fixed to
   export the **full frame set** with correct per-frame delay (frame subsampling now only
   applies to interactive HTML embedding, not exported GIF/MP4 files). Added a regression
   test (`test_animation_export`) asserting exported animation duration matches the
   requested duration.

## Doc build + media validity

- `PATH=.venv/bin:$PATH make -C docs html` — build succeeded, 0 tracebacks, all
  example/tutorial pages generated under `docs/_build/html/`.
- All embedded `.mp4` files are `ffprobe`-valid (6 x ~30s clips, confirmed at Plan 8 Task 4
  close, including the new `plot_shape_morph` clip).
- Gallery GIFs are full-length (not subsampled) per the fix above; the reference
  `docs/auto_examples/spin.gif` committed artifact (900 frames / ~27-30s) was refreshed
  after being found stale.

## Playwright visual verification (this task)

`scripts/verify_docs_playwright.py` serves the already-built `docs/_build/html/` tree over
a real local `http.server` and drives it with a real headless Chromium (Playwright
1.61.0 / Chrome for Testing 149.0.7827.55) — no mocks, no stubs. It makes hard,
non-weakened assertions against the live DOM and exits non-zero on any failure.

Run: `.venv/bin/python scripts/verify_docs_playwright.py` — **8/8 pages passed**:

| Page | Kind | Key assertions |
|-|-|-|
| `auto_examples/index.html` | gallery index | 40 thumbnails (>= 20 required); 5 sampled thumbnails non-blank (pixel stddev 15.3-36.0) |
| `auto_examples/plot_basic.html` | static example | `sphx_glr` image decodes to 640x480, non-blank (std 35.1); Colab badge present, href branch-aware |
| `auto_examples/animate_spin.html` | animated (mp4) | `<video>` decodes to 638x476, seeked frame non-blank (std 34.5); Colab badge branch-aware |
| `auto_examples/animate_plotly.html` | animated (plotly) | `.plotly-graph-div` renders real SVG/canvas children; page contains `Plotly.addFrames(`/`Plotly.animate(`; rendered plot non-blank (std 21.0); Colab badge branch-aware |
| `auto_examples/plot_shape_morph.html` | animated (mp4) | `<video>` decodes to 638x476, seeked frame non-blank (std 38.8); Colab badge branch-aware |
| `auto_examples/plot_clusters.html` | static example | `sphx_glr` image decodes to 640x480, non-blank (std 40.1); Colab badge branch-aware |
| `tutorials/plot.html` | tutorial (nbsphinx) | rendered figure non-blank (std 53.1); branch-aware `pip install ... @dev-2.0-refactor` cell present |
| `tutorials/align.html` | tutorial (nbsphinx) | rendered figure non-blank (std 34.2); branch-aware `pip install ... @dev-2.0-refactor` cell present |

Notes on the Colab check: gallery/example pages carry a literal "Open in Colab" badge
image whose link href was verified to contain `dev-2.0-refactor`
(e.g. `https://colab.research.google.com/github/ContextLab/hypertools/blob/dev-2.0-refactor/docs/auto_examples/plot_basic.ipynb`).
Tutorial pages (rendered from notebooks via nbsphinx) do not carry that image badge; instead
their first cell is a branch-aware install line
(`%pip install -q "hypertools[interactive] @ git+https://github.com/ContextLab/hypertools.git@dev-2.0-refactor"`),
which was verified present on both tutorial pages checked.

"Non-blank" is a real pixel-content check, not a size/existence check: each verified
image/video-frame/plotly-render element is screenshotted and its pixel standard deviation
is computed (via PIL + numpy); a blank/solid-color render would have std ~= 0, and the
threshold is 2.0. All 8 pages' checked elements measured std between 15.3 and 53.1.

Screenshots (full-page + cropped element captures used for the non-blank checks) are saved
in this directory (`docs/images/v2.0-docs/`), files `01_*.png` through `08_*.png` plus
`_crop`/`_frame`/`_plot`/`_thumb_N` element-level captures. Visually spot-checked
(`02_plot_basic.png`, `03_animate_spin_frame.png`, `04_animate_plotly_plot.png`): real
rendered 3D scatter/trajectory plots and a real Plotly animated line plot with
Play/Pause controls, not blank pages.
