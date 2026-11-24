# Session 2026-11-22: GH #295, animated panel grids and `caption=`

Branch `feat/295-animated-panels` (not pushed, no version bump, no tag).

## What was built

- `animate=` with `panels=`/`subplots=` returns ONE `HyperAnimation` of the
  whole grid. `_plot_panels` still re-enters `plot()` once per panel; in
  "panel cell" mode (`_panel_animation=` private kwarg, `_draw(panel_cell=True)`)
  the matplotlib backend draws into the given axes and returns a
  `PanelAnimationStub` (frame function + frame count) in place of a
  `FuncAnimation`. `_animate_panel_grid` then drives every stub from one
  `HyperFuncAnimation`.
- `caption=` (matplotlib): string, `'{index}'` pattern, or callable taking a
  `FrameContext`. Drawn by `_add_caption` as a `fig.text` with gid
  `hypertools-caption`, placed just under the lowest drawn content; the
  figure grows downward only when there is not enough blank room.
- `FrameContext.panels`: one context per panel for a grid; the grid context's
  `axes` is a tuple.
- `labels=` bookkeeping is now per axes (`ax._hyp_labels_and_points`), so
  labels follow the reveal in every panel.

## Layout notes

- A rotating 3-D cube overflows its (square) viewport at some azimuths;
  `_fit_rotating_cubes` lowers a per-cell zoom scale until the projected
  cube stays inside at every azimuth. With axis labels the figure is
  enlarged (or, with an explicit `size=`, a ValueError says to pass a larger
  one or drop the labels).
- In cell mode `fig.tight_layout()` must run per panel as it does for a
  static panel, and animated artists are excluded from layout.

## Refused combinations

plotly backend (NotImplementedError naming `backend='matplotlib'`),
`companion=`, `explore=`, `ax=`, panels of different widths
(`panel_fit='independent'`), axis labels that cannot fit an explicit `size=`,
`caption=` on plotly / with `ax=` / wrong type / `{index}` without an index.
The public `ax=` + `animate=` refusal is unchanged.

## Known, not changed

- A STATIC 3-D grid with axis labels clips the labels at the right edge of
  the last panel (seen while comparing; static output was kept
  pixel-identical here). Needs its own fix.

## Tests

`tests/test_animated_panels.py`, `tests/test_plot_caption.py`; the three
old-refusal tests in `tests/test_plot_panels.py` and
`tests/test_plot_panels_audit.py` now assert the new behavior. The full
suite was not run (about 25 minutes); the panel, animation, hook, title,
docs and example test files were.

## Shared colors, legend and colorbar for panel grids (branch feat/295-shared-legend)

Owner requirement (2026-11-22): "Legends and colorbars should be consolidated
(shown once across panels) and coloring should not be repeated to mean
different things across panels".

What was built (all in `hypertools/plot/plot.py` unless noted):

- `_plot_panels` resolves what the single-axes call of every dataset draws and
  hands each panel its share: `_panel_dataset_colors` (dataset i = palette
  position i of n, passed as `color=`), `legend=True` -> `[i + 1]`, and the
  grid's `hue=` scale (`_panel_apply_hue_scale`: `_panel_shared['limits']` for
  a continuous hue, a narrowed `{category: color}` palette for a categorical
  one). The scale is the shared probe's `'colors'` (`_panel_probe` now returns
  it) or, for `panel_fit='independent'`, `_panel_color_probe`.
- `plot()` takes an internal `_panel_shared` dict: `'limits'` pins the
  continuous hue range (`_multicolor_line_colors(limits=)`,
  `_build_colorbar_info(limits=)`), and `'colorbars'` collects the panel's
  colorbar instead of drawing it (and skips `_fit_right_legend`).
- After the panels are drawn: `_merge_panel_legends`, `_merge_panel_colorbars`,
  `_add_shared_furniture` (strip right of the grid; grid laid out without the
  strip, then `_grow_figure_rightward`), `_place_shared_furniture` (re-run
  after the caption and the animated cube fit).
- Per-panel legends/colorbars remain when `_panel_meanings_differ` (clusters
  fit per panel, forecast clusters) or on `_PanelFurnitureConflict` (a label
  that looks different in two panels: the grid is drawn again per panel).
- Plotly: `merge_panel_legends`, `place_shared_furniture` and
  `make_panel_grid(strip_px=)` in `plotly_backend.py`.
- Bundle `'colors'`: `_merge_panel_color_scales`.

Gotchas met: `tight_layout(rect=...)` and a single `tight_layout()` pass after
removing the panels' own legends both under-reserve a 3-D panel's title room
(titles left the canvas); three passes on a strip-less figure settle it.

Tests: `tests/test_panels_shared_legend_colors.py` (new); the tests that
asserted per-panel legends/colorbars/colors were rewritten, each with a
"CHANGED for 1.2.0" comment.

## Review round 1 fixes (2026-11-22, after the shared-legend commits)

Source list: the scratchpad's `feat295/fix_round1.md` (C1-C5, T1-T6, S1) plus
A1-A3. Tests: `tests/test_animated_panels_review.py` (contexts, leaks,
captions, titles, warnings) and `tests/test_panel_layout_review.py` (3-D axis
labels, explicit sizes, shared-legend placement).

- `_resolve_animate_argument` is the one resolver of `animate=` (dict form,
  sugar styles, scalars); `plot()` and `_plot_panels_impl` both call it.
- `_plot_panels` is now a wrapper that closes every pyplot figure a failed
  grid call registered; the implementation is `_plot_panels_impl`.
- Captions: `_wrap_caption` wraps to the figure width (per frame, through
  `_set_caption_text`); `_add_caption(samples=)` sizes the room for the
  first/middle/last frames (`_caption_sample_frames`). `caption=''` is no
  caption. Contexts are real: `_real_frame_context` in `plot()`,
  `_static_grid_context` for a static grid.
- Grid `{index}` caption: `_make_title_pattern_resolver` reads the context of
  the panel that draws the first indexed dataset (panel 0 for reducer grids).
- A panel cell with a per-frame title draws frame 0 before `tight_layout`, so
  the title has room.
- `_fit_panel_cubes` (was `_fit_rotating_cubes`): matplotlib's 3-D label
  offset SCALES with the cube (measured: label reach ~1.45 x cube reach in a
  3.2 in cell), so labels are fitted by shrinking the cube proportionally;
  the figure grows only when a cube would fall under 70% of an unlabeled one
  (`_PANEL_LABELED_CUBE_SHARE`), in steps of at most 20%, in the limiting
  direction. An explicit `size=` is refused only under 50%
  (`_REFUSED_LABELED_CUBE_SHARE`); the suggested size keeps the caller's
  proportions and is found by fitting at that size. Static grids use the
  same fitter through `_fit_static_cube_labels` (65%,
  `_STATIC_LABELED_CUBE_SHARE`; explicit `size=` honored, never refused). The camera views come from `_camera_azimuths`
  (`rotations=0` is one view).
- Shared legend placement: `_shared_legend_side` maps a `legend_kwargs` `loc`
  to a strip side; `_grow_figure` adds room on any side; `bbox_to_anchor` /
  `loc='center'` / plotly `x`/`y` reserve nothing.
- `analyze(internal=True)` no longer warns about the display `ndims` on its
  Pipeline path.
- `tests/test_animated_panels.py::test_axis_labels_enlarge_the_figure...`
  still asserts labeled > plain width for long labels; it passes because the
  70% floor makes long labels grow the figure slightly. It was not edited.

## Review round 2 fixes (2026-11-22)

Source list: the scratchpad's `feat295/fix_round2.md` (Codex 1-10, tester
R1-R11). Tests: `tests/test_panels_global_resolution.py` (new: every color /
legend item, both backends, each compared with the single-axes call), the
renamed-for-real `test_plotly_grid_categorical_hue_and_legend_false`, and
the layout/title tests added to `tests/test_panel_layout_review.py` and
`tests/test_animated_panels_review.py`.

Structure of the color/legend fix (one mechanism instead of per-item cases):

- `_resolve_grid_colors(rows, grid_kw, ndims, caught, clustering=)` MAKES the
  single-axes call of every dataset (on the analyzed rows, `reduce=None`,
  matplotlib, figure thrown away) with the color kwargs
  (`_GRID_COLOR_KWARGS`: fmt, hue, palette, color(s), legend, names,
  colorbar, hue_mode, color_reduce) and the shared clustering replayed. It
  returns the call's color scale, its `{category: color}` keyed by the hue's
  own categories, its colorbar info, `{category: legend label}` and the
  legend's order. Its errors are the grid's errors (legend-list length with
  the grid's counts, palette validation, colorbar validation); its warnings
  are issued once (`_reemit_panel_warnings`).
- `plot()`'s internal `_panel_shared` carries a panel's share:
  `'hue_kind'` (forces the integer-hue categorical/continuous decision),
  `'limits'`, `'legend_names'`, `'quiet'` (no per-panel repeat of the
  whole-call warnings), `'colorbars'` (collector) and `'report'` (what the
  probe call's legend resolved to). A consolidated grid's panels get
  `colorbar=None`: the grid's one colorbar is the resolution's.
- `panel_fit='independent'`: `_panel_hue_scale_probe` (the same resolver on
  placeholder rows, hue/palette/fmt only) runs BEFORE the per-panel fits so
  their kwargs are already narrowed; the full resolution runs after them.
- Not resolved globally (no grid-wide meaning exists): reducer-list grids
  (each panel already draws every dataset; their legends/colorbars are
  merged as before) and per-panel clusters (`panel_fit='independent'` with
  `cluster=`).
- Single-axes change that came with it: a `legend=` list under a categorical
  hue names the CATEGORIES in `legend=True` order on the line path too
  (was: one entry per run, in data order).
- Plotly: `_legend_trace_look` compares line width and marker size;
  `merge_panel_legends(order=)` ranks the kept entries; a grid's legend is
  wanted by `legend=` or by `names=` alone.

Layout half of round 2 (done by a subagent in a side worktree, cherry-picked):
R1 static labeled grid at a too-small explicit `size=` clamps cubes at 50%
(`_REFUSED_LABELED_CUBE_SHARE`), warns once with a size that works; Codex 9
title room = tallest of first/middle/last frames (`_caption_sample_frames`),
single-axes too (2-D figure grows upward); R11 `_wrap_shared_legend`,
`_thin_horizontal_colorbar_ticks`; R8 `Animation.save` ->
`FuncAnimation._init_draw` draws frame 0 before the loop (documented); R9
example text. Left open: an animated grid's figure is not enlarged for a
tall title (panels shrink under it).

## Static grids sized by the grid (owner decision 2026-11-22, for 1.2.0)

Owner chose: static `panels=` grids get 3.2 in per panel, like animated ones;
explicit `size=` still honored.

- `_PANEL_CELL_INCHES` (was `_ANIMATED_PANEL_CELL_INCHES`) and
  `_default_panel_grid_size(nrows, ncols)` =
  `(max(6.4, 3.2 * ncols), 3.2 * nrows)`; `_plot_panels_impl` uses it for
  static and animated alike. The grid's shape is still chosen first
  (`_auto_panel_grid` from `size=` or the 6.4 x 4.8 in aspect).
- Plotly: `make_panel_grid(default_size=)`; `_plot_panels_plotly` passes the
  same inches (x 100 px), gutters/strip added as before.
  `hyp.subplots()` (both backends) keeps its own default (6.4 x 4.8 in /
  640 x 480 px).
- Static vs animated default size differ only for 3-D panels WITH axis
  labels (measured on 1x4, 2x2, 3x1 with titles, captions, legends,
  colorbars, per-panel legends, 2-D labels: all equal). Animated: labels
  swing with the cube, floor 70% (`_PANEL_LABELED_CUBE_SHARE`), cube drawn at
  the animation zoom; static: one view, floor 65%
  (`_STATIC_LABELED_CUBE_SHARE`). Also differs with `rotations=0`.
- New tests: tests/test_static_panel_grid_size.py. Rewritten (old default
  asserted): tests/test_animated_panels.py
  (`test_a_static_grid_is_sized_by_its_grid_too`),
  tests/test_panel_layout_review.py.
- examples/plot_shapes_zoo.py: dropped `size=(12, 6)` (a workaround for the
  old default). docs/tutorials/reduce.ipynb: the one static `panels=` cell's
  stored output re-executed (scripts/execute_tutorial.py --out-dir, cell 19
  copied in).
- docs/prebuilt_gallery.json NOT touched: every static `panels=` gallery
  figure changes size and needs republishing at release.

## Shared axis labels on the outer panels only; one layout path (owner decision 2026-11-22, for 1.2.0)

Branch `feat/295-outer-labels` (worktree), on top of f31a022b. Tests:
`tests/test_panel_outer_labels.py`.

Owner: "if x and/or y axis labels are shared across panels, we can just show
them on the left-most columns (shared x labels) and bottom-most rows (shared
y labels)"; implemented as the usual outer-labels convention (x under the
lowest panel of each column, y on the left-most panel of each row), static
and animated through one path.

- `_outer_label_panels(labels, nrows, ncols, ndims, z_side)` is the ONE
  decision function (`_LABEL_SHARED_ALONG`: 2-D x=column, y=row; 3-D x and y
  =column, z=row). Same text in every occupied panel of the column/row ->
  one carrier; texts differ (a missing label counts) -> every panel keeps
  its own. Spare cells are not occupied.
- matplotlib: `_apply_outer_labels` (after the panels are drawn and the
  legends merged, before any layout) empties the other panels' label text.
  3-D z carrier: the panel at the end of the row matplotlib draws the z
  label toward, measured on a real layout (`_z_label_side`; right at the
  default view, left at other azimuths).
- plotly: `_apply_outer_labels_plotly` (same decision, on the drawn panel
  figures before the transplant). z titles go to the row's LEFT panel
  (`_PLOTLY_Z_LABEL_SIDE`); a scene's own z title falls outside a grid
  cell's scene and was clipped on every panel before, so first-column z
  titles are written as paper annotations in the left margin
  (`_add_plotly_z_titles`, `PLOTLY_Z_TITLE_NAME`, 30 px, default figure
  widened). z titles kept on INNER plotly panels (rows whose z texts
  differ) are still scene titles and still clipped.
- One layout: `_tighten_panel_grid` (tight_layout to convergence, negative
  wspace/hspace set to 0: labeled 3-D boxes used to overlap) ->
  `_lay_out_panel_grid` (cube fit, outer-label margins, strip, caption) for
  static and animated alike. `_fit_static_cube_labels` and
  `_STATIC_LABELED_CUBE_SHARE` (65%) are gone: one 70% floor.
  Every cube starts at matplotlib's default box zoom (`start_scale` on
  `PanelAnimationStub` = 1 / `_anim_box_zoom(1)`, on `_StaticCubeZoom` = 1),
  so static == animated(rotations=0): same figure size, axes positions,
  cubes, labels, strip and caption boxes (tested). Static cubes are
  unchanged from 1.1 (plotly parity test kept); an animated grid's cubes
  are ~7% smaller with rotations=0 and ~1% smaller rotating than in the
  first 1.2.0 draft.
- 3-D label room: seen from ONE view (static, rotations=0, default size)
  the outer panels' labels may hang past their squares toward the grid's
  outer edge (`_open_label_sides`: 'edge' or 'spare' cell) and the figure
  grows there by the measured overhang (`_outer_label_margins`,
  `_grow_figure`; the right-hand strip moves out with it). Otherwise
  (rotating; explicit `size=`; inner panels; per-panel legends/colorbars;
  a shared legend left of/under the grid; a shared colorbar not in the
  right strip) labels are kept inside the panel's own cell by drawing the
  cube smaller.
- EVERY cube of a grid is drawn at the same share of its unlabeled size
  (the smallest any panel's labels leave). Drawing only the label
  carriers smaller looked broken (kept renders:
  scratchpad feat295/outer/evidence_uneven_cubes_*.png). So in a ROTATING
  grid the inner panels do not get their room back (all cubes at >= 70%,
  as in the first draft); in a one-view grid they do.
- Still different between static and animated: an explicit `size=` too
  small for labeled cubes raises for an animated grid and warns for a
  static one (existing tests assert both; a static labeled grid always
  drew in 1.1).
- Side fixes: `panel_fit='independent'` grids of DataFrames with different
  column names were refused by the grid color resolution (stacked by
  feature name); it now resolves on the bare rows.
- No opt-out was added. Labels that differ are all kept, so different
  column names per panel (needs `panel_fit='independent'`) label every
  panel; `xlabel=` etc. take one string for the whole grid.
- Existing tests rewritten to the new behavior (each marked "CHANGED for
  1.2.0 (owner decision 2026-11-22)"): tests/test_plot_panels_fit.py (5),
  tests/test_static_panel_grid_size.py (1), tests/test_panel_layout_review.py
  (`_assert_labels_clear`), tests/test_plot_panels.py (1),
  tests/test_panels_shared_legend_colors.py (1).
- docs/prebuilt_gallery.json and docs/tutorials notebooks NOT touched:
  static and animated `panels=` gallery figures with axis labels change and
  need republishing at release.

## Review round 3 fixes (2026-11-22, after c7b6d0b2)

Brief: scratchpad `feat295/fix_round3.md` (Codex 1-8, tester Q1-Q5, suite Q6). All fixed; commits in order:

| Commit | Items | Tests |
|-|-|-|
| ed4a491d | Codex 1, 2, 4, 5, 7, 8 (grid color resolution) | tests/test_panel_color_invariant.py (403 cases), tests/test_panels_review_round3.py |
| ba7f9ddc | Codex 3 (RGB(A) tuple under colorbar=) | tests/test_panels_review_round3.py |
| bca7c05d | Codex 6 (animated title at loc left/right) | tests/test_panels_review_round3.py |
| 56c82d34 | Q1 (per-frame caption added an on_frame= call) | tests/test_panels_review_round3.py |
| 936ce719 | Q2, Q3, Q4 (predict= grids) | tests/test_panels_forecast_review_round3.py |
| 4fa57422 | Q5 (morph legend keys) | tests/test_morph_legend_keys.py |
| 69d1db57 | Q6 (process tree on browser give-up) | tests/test_multibyte.py, tests/test_animation_export.py |

How the color resolution works now:
- `_resolve_grid_colors` re-enters `plot()` with `_panel_shared={'resolve_only': True, ...}`; `plot()` returns right after `colorbar_info`/`colors_info` are built, before interpolation, figure creation or drawing. Arguments are forwarded by blacklist (`_GRID_RESOLUTION_DROPPED_KWARGS`), so `animate=`, `font=`, `antialias=` and the rest reach it.
- `_grid_needs_color_resolution`: no `hue=`/cluster/legend/colorbar means no resolution call at all.
- Categories are keyed by the category (never the legend label). Matrix hue: the resolver returns the resolved rows (`matrix_hue`: RGB rows, or shifted mixture weights) and the quantized blend groups (`blend_groups`); `_panel_apply_hue_scale` slices them per panel.
- Rule chosen: two categories given one `legend=` name keep their colors; the shared legend cannot show one name for two colors, so each panel keeps its own legend (existing conflict rule).
- Timings (best of 3/2, same machine, back to back): 12 panels x 5,000 rows with continuous hue 0.57 s -> 0.50 s; two 500,000-row datasets, no hue 0.24 s -> 0.18 s.

Other rules chosen:
- Q3: identical draw-time warnings from several panels are re-issued once (`_reemit_panel_warnings`); the forecast-cost notice is issued by the first schedule able to project, as spent + that schedule's projection x schedules still to come (`forecast.grid_cost_report`).
- Q5: every legend hypertools builds forces its keys visible (`show_legend_keys`).
- Q6: `_descendant_pids` reads one `ps` snapshot (PowerShell on Windows; untested there) and `_kill_process_tree` now also kills descendants outside the process group.
- Q1 cause: `_wrap_caption` measured through `_measurement_renderer`, which replaces the figure's canvas; mid-save that drops matplotlib's `_is_saving` guard and the animation starts (extra frame 0). Now measured on a bare `RendererAgg` (`_text_metrics_renderer`).

Scratch `repro.py` once wrote `tmp*/x.mp4` into the repo root; moved out to the scratchpad (`fix3/stray/`).

Final run of the brief's test list (99 files, 8 shards) at 03f986ef: 4132 passed, 13 skipped (release gate x6, example smoke x6, CI-only font check x1), 0 failed, no warnings. `ruff check .` clean.

## Review round 4 fixes (2026-11-22, after 35d515d2)

Brief: scratchpad `feat295/fix_round4.md` (Codex fourth review, 3 items). All fixed; tests in `tests/test_panel_color_invariant.py`.

| Commit | Item | Tests |
|-|-|-|
| 92d5ac93 | 1: matrix `hue=` + `color_reduce=` reduced twice per grid | `test_matrix_hue_is_reduced_once_per_grid`, `test_seeded_tsne_hue_colors_are_the_single_axes_calls`, `test_reducer_list_grid_reduces_its_matrix_hue_once` |
| b18c34af | 2: named mixture hue columns lost in the shared legend | `test_mixture_hue_column_names_reach_the_shared_legend`, `..._reach_an_animated_grids_legend`, `test_mixture_hue_names_are_in_the_grids_color_scale` |
| dc30f14c | 3: `ndims=1` series names lost under `panel_fit='independent'` with differing column names | `test_independent_series_grid_keeps_its_series_names`, `..._names_its_color_scale_by_series`, `test_independent_two_column_series_grid_keeps_its_series_names` |

How:
- Item 1: `_plot_panels` owns one `matrix_cache` dict per grid call, passed in `_panel_shared['matrix_cache']` to every call made on the WHOLE matrix (shared probe, early and final resolution, and a reducer-list grid's probes and panel draws). `_matrix_hue_to_rgb_once` reduces on the first call and returns the stored RGB (matched by value) afterwards. Reducer-list grids went from 2 fits per panel to 1 per grid (noted in CHANGELOG, since 1.1.0 behavior changes there for a stochastic reducer).
- Item 2: the resolution reports `hue_columns`; `_panel_apply_hue_scale` puts them in `shared['hue_columns']`, which `plot()` uses as `_hue_column_names` when the panel's hue is a bare array.
- Item 3: the single-axes call REFUSES frames with differing column names (GH #132), so there is no single-axes legend to match for that exact data. Rule chosen: each series is called by its column name, which is what each dataset's own call and the single-axes call of same-named series read. The names are captured before the rows are stripped and handed to the resolution as `_panel_shared['series_columns']`, so legend labels, legend order and the bundle's `colors['labels']` all carry them.

## matplotlib 3.11 text metrics (2026-11-22, branch fix/mpl311, after b81e1d2d)

Hosted CI "test (ubuntu-latest, 3.13)" failed 9 tests on b81e1d2d with matplotlib 3.11.2; all passed locally on 3.10.8. Reproduced on macOS with a Python 3.13 venv pinned to matplotlib 3.11.2, numpy 2.5.5, pillow 13.0.0, fonttools 4.68.0 (no Docker needed: the default face is the bundled Noto Sans on every OS).

Cause: two behavior changes in matplotlib 3.11, three places in hypertools that assumed the older behavior.

- `Text._get_layout`: a line's height is now the font's own ascent + descent (+ line gap between lines), read from its `OS/2`/`hhea` tables. Through 3.10 it was the measured height of the glyphs `lp` (times 1.2 between lines). Noto Sans: 1.362 em per line, against about 1.0 em (1.2 em between lines) before.
- `Axis3D._update_all_positions`: a label with no text (or hidden) is no longer positioned on a draw.

| What failed | Why | Fix (`hypertools/plot/plot.py`) |
|-|-|-|
| `caption=` ended below the canvas (7 tests) | `_add_caption` reserved `lines * 1.2 * font size` | `_caption_text_height_in` measures the wrapped text with the caption's own font at the figure's DPI; the room is the larger of that and the nominal line grid (so 3.10 output is unchanged) |
| `test_axis_labels_enlarge_the_figure_and_stay_inside_their_cells`, `(0, 0, '')` | a shared label emptied on an inner 3-D panel kept the position of the draw before the cube fit | `_place_emptied_cube_labels`: one measurement draw of the finished layout with a stand-in text in the emptied labels, then emptied again |
| `test_a_static_labeled_grid_that_fits_its_size_does_not_warn[long-1x4-8x3]` | the taller label boxes of the last panel's y and z labels overlap by more than the 4 px allowed, inside the panel's square, at any cube size | `_label_gutter_room`: when a static grid at an explicit `size=` has big enough cubes but colliding labels, it is fitted once more with the labels let into the blank room beside each square (half the gutter to the next panel, or up to the canvas edge / a legend / a colorbar less the label pad); kept only if the labels then clear, otherwise the grid is put back and the warning is raised as before |

Tried and dropped: comparing the labels' painted pixels instead of their layout boxes (the tests' own overlap count uses layout boxes, so the library must too); pushing colliding labels out with `labelpad` (the fit shrinks the cube by exactly as much, the labels do not move).

Not changed: `_axis_label_reach` still measures layout boxes, so labeled 3-D cubes are a few percent smaller under 3.11 than under 3.10 (taller boxes need more room). No CHANGELOG entry: all of this is 1.2.0 code that 1.1.0 does not have.

Tests: `tests/test_text_room_measured.py` (caption room against the real renderer in four fonts; emptied labels against matplotlib's own placement; gutter room; the 1x4 grid at three sizes).

Pixel check under matplotlib 3.10.8, before/after: `examples/animate_panels.py` (frames 0, 150, 299), `plot_datasaurus.py`, `plot_shapes_zoo.py`, `plot_impute.py`, `plot_autoencoders.py` are byte-identical (canvas RGBA buffers).

Final checks after the last change (panel, caption, animation, plot-review, docstring and datatype-gate files plus the new one; `ruff check .` clean): matplotlib 3.11.2 / Python 3.13: 3053 passed, 0 failed; matplotlib 3.10.8 / Python 3.12: 3053 passed, 0 failed.

## Windows CI: the suite died in the first test that kills a render worker (after 5f104d0a)

- CI on 5f104d0a: 12 jobs green (all ubuntu, windows 3.10/3.12/3.13, the gates, docs-clean); the four macOS jobs were still queued at the waiter's deadline; `test (windows-latest, 3.11)` ended with exit code 1 and NO pytest summary, its last line `tests/test_animation_export.py::test_real_export_succeeds_after_a_killed_export` (job 137294099026).
- Reading (not reproduced: no Windows machine here): `_kill_process_tree` now also kills `_descendant_pids(proc.pid)` (round 3, Q6). On Windows that list came from `Win32_Process` parent ids, which are never cleared when a parent exits, while process ids are reused. A render worker handed the id of a dead ancestor of the test runner makes the runner its "descendant", and `taskkill /F` ends it (exit code 1, no summary). Intermittent by nature, which fits three green Windows jobs beside one dead one.
- Fix: `_descendants_in_table` follows a parent link only when the child is not older than the parent (Windows creation ticks; a row without a creation time is never taken for a child), and `_descendant_pids` never returns the calling process or its ancestors. POSIX tables carry no creation times and are walked as before. Tests: `test_a_stale_parent_link_is_not_followed`, `test_this_process_and_its_ancestors_are_never_descendants`, `test_descendant_pids_finds_a_real_child_and_never_this_process` (real process table).
- Also in this stretch: the gallery's own videos are remuxed fast start by the scraper (fa4e765b), two example comments corrected (b81e1d2d), matplotlib 3.11 text metrics (5f104d0a). The user confirmed on 2026-11-22 that the 1.2.0 gallery figures are correct ("Current versions are correct for all figures").

## Released: 1.2.0 (2026-11-22, about 23:45 EST)

Jeremy's instruction: "Go for 1.2.0 release on pypi and then on conda forge".

- `master` == tag `v1.2.0` == 8086b5b8. Master CI run 45077352879 and tag CI run 45089891535: 17 of 17 jobs green.
- The tag run needed one re-run: `test (macos-latest, 3.13)` failed `tests/test_multibyte.py::test_render_script_exits_NO_BROWSER_when_the_browser_never_answers` on its own sanity check, `the stand-in browser was never seen running: {43413: '(Python)', 43416: '<defunct>'}` (8431 passed). The re-run passed. Cause and fix below.
- PyPI: `hypertools-1.2.0-py3-none-any.whl` sha256 3aa7d00e04bdfe21a8f234c86faabd4b9684f9dc812468da24e79b0692fd9d3b, `hypertools-1.2.0.tar.gz` sha256 b1ba6eb4d59c8d2f2460783ce28d05665da378cf7db1c7f404e83621af55e125 (the digests recorded at build time; read back from PyPI's simple index). `https://pypi.org/pypi/hypertools/1.2.0/json` kept answering a cached 404 for some minutes after the upload; `https://pypi.org/pypi/hypertools/json` and the simple index showed the files at once.
- GitHub release "HyperTools 1.2.0" published on the tag (Latest), both files attached; the seven links in its body answer 200.
- Read the Docs: `stable` moved to the tag by itself (API: `stable 8086b5b8 built`); `/en/stable/` and `/en/latest/` `optional_dependencies.html` answer 200; a gallery page under stable links `docs-notebooks/v1.2.0/` and that notebook's install cell is `%pip install -q "hypertools[interactive]>=1.2.0"`.
- Clean environment (Python 3.12): `pip install hypertools` gives 1.2.0; static and animated panel grids with `caption=` run. Published-wheel check (`hypertools[interactive]==1.2.0 --only-binary=hypertools --report`): the report names the wheel and its digest; a plotly figure, a Kalman forecast and an animated grid saved as a fast-start mp4 all run.
- On Python 3.9, `pip install hypertools` gives 0.8.2 without a message (`requires-python >=3.10`); the first smoke environment was built on 3.9 by mistake.
- PyPI page: 13 README images at the `v1.2.0` tag and the version badge all answer 200.
- conda-forge: no bot pull request had appeared, so https://github.com/conda-forge/hypertools-feedstock/pull/2 was opened from the `jeremymanning` fork (version and sha256 only; `[project] dependencies` is identical at v1.1.0 and v1.2.0). Checked first: a conda-forge-only Python 3.10 environment with the recipe's requirements solves, the PyPI sdist installs into it, `pip check` is clean, `import hypertools` gives 1.2.0.

### The flaky sanity check (branch `post-1.2.0`, not on master)

- Cause: `_run_render_watching_its_processes` kept one command line per process, the latest. A process is described differently over its life (`(Python)` between fork and exec, `sleep 600`, a bare `(sleep)` under load, `<defunct>` once ended), so a last snapshot that caught the stand-in already ended erased the `sleep 600` seen before it.
- Fix: the watcher keeps every command line each process was seen with; the "was it ever seen" check reads the whole history; `_alive` counts a process as still there when its current command line is any of those seen for it (stricter than before, never looser). Checked on real processes: a running `sleep` is reported, an ended one not yet collected by its parent is reported, a collected one is not.
- Master gets no commits between releases (release gate), so this waits on the branch for the next release.
- `notes/release_notes_v1.2.0_draft.md` now equals the published release body (it lacked the outer-labels bullet).

## Issue #299 (2026-11-23, branch `post-1.2.0`, for 1.2.1; not pushed)

Cause of the bug: `animate_plot3D`/`animate_plot2D` (hypertools/plot/matplotlib_backend.py) created the trail artists only inside `if any(is_line(f) for f in fmt)` (and `if is_line(fmt)`). `is_line` means "a PURE line format, no marker", so `'-o'`, `'o-'`, `'.-'`, `'--s'` built no trail artist and the updaters, which tolerate a missing trail through `zip_longest`, drew nothing and said nothing. A markers-only dataset got a trail of faded markers only when another dataset of the same call was a pure line. Plotly was never affected (it builds a trail trace for every flagged dataset).

Fix: the gate is gone; every dataset with a trail flag gets its trail, drawn with its own format. The trail of a line+marker format carries faded markers on the observations because `_aa_window(..., artist=trail)` already ran `_mark_observations` on it, exactly as for the tail. Markers-only: trail of faded markers (matplotlib now equals plotly), so there is no case left where a trail flag is dropped and no new warning was needed for it.

Second bug found on the way (fixed, in the CHANGELOG): a trail took its color from the axes' color cycle, so with `chemtrails=[False, True]` dataset 1's trail was drawn in dataset 0's color. `_own_color` gives a trail its head line's color unless the kwargs or the format string name one.

Added (all real `plot()` arguments, all four animation ones also accepted inside the `animate={...}` dict):

- `head_marker=`: marker str / True / dict (`marker`, `size`, `color`, `alpha`, `edgecolor`, `edgewidth`, `fillstyle`, `zorder`). One extra `Line2D` per dataset, created after the trails (so `ax.lines[:n]` are still the heads and the trails follow: `_apply_multicolor_animation` finds them by position), placed by `_place_head_marker` in the four updaters. Tagged `_hyp_trace_role='head_marker'`. With continuous `hue=` the multicolor frame function recolors it to the head row's color unless the dict pinned a color. Not in `FrameContext.artists` (the `dataset_fade` updater maps artists to datasets by position); the fade updater follows `line._hyp_head_marker` instead.
- `tail_linewidth=`: head line width; trail keeps `linewidth=`. Serial: switched per frame (`_set_serial_linewidth`), thick only while the dataset is the comet head.
- `trail_alpha=`: replaces the constant 0.3 (matplotlib `_trail_kwargs`, the multicolor trail collection, plotly's trail traces).
- `hold=` seconds: `FuncAnimation` is given the frame LIST `0..n-1` then `n-1` repeated (`held_frame_sequence`), so every wrapper `plot()` puts round the frame function later (multicolor, hooks, titles, captions) just sees the last index again. `_hyp_content_frames` on the animation is the count of distinct frames; `content_frame_count()` is what the caption/title sampling uses, and `HyperAnimation.draw_frame` clamps to it. A panel cell never holds; `_animate_panel_grid` holds for the grid.
- `caption_kwargs=`: size/color/weight/style/family/alpha/linespacing/position/loc, applied to the `Text` before `_wrap_caption`/`_caption_text_height_in` run.
- `suptitle=` + `suptitle_kwargs=`: drawn by `_add_caption` itself (gid `hypertools-suptitle`, position top), in `_lay_out_panel_grid` for grids (static and animated, one path) and at the caption site of `plot()` for single plots.

Decisions:

- `title=` as one string with `panels=` names every panel (documented, tested), so the figure title is a new argument, `suptitle=`.
- `caption_kwargs=`/`suptitle_kwargs=` follow `title_kwargs=`/`legend_kwargs=` rather than separate `caption_size=`... arguments.
- Matplotlib-only (NotImplementedError on plotly, the `caption=`/`companion=` wording): `head_marker`, `tail_linewidth`, `hold`, `suptitle`, `caption_kwargs` (through `caption=`). `trail_alpha` works on plotly.
- `head_marker`/`tail_linewidth`/`trail_alpha` on a static plot or `spin`/`morph` warn like the trail flags do; `hold` on a static plot raises like `on_frame=`.
- `_grow_figure` now also keeps hypertools' own figure texts (caption, suptitle) at their place in inches: adding the suptitle's strip at the top would otherwise have moved a bottom caption.
- Room above an animated figure: measured over the first/middle/last frames (grid: the cells' own frame functions set per-frame panel titles), and for a single plot with the tallest per-frame/per-segment title set on the axes while measuring (`_real_frame_context` runs no hooks, so the title was not there).

Tried and dropped:

- Clamping the frame number inside the backend's frame function for `hold=`: the wrappers `plot()` adds afterwards would still have received the raw index. The frame list does it for all of them.
- Default head marker size = the dataset's `markersize=` alone: with `'-o', markersize=3, tail_linewidth=3` the marker disappeared under the tail. Now never less than 2.5 x the tail width.
- A warning for markers-only + trail flags: not needed once the trail is drawn.
- Scalar only: `head_marker`, `tail_linewidth`, `trail_alpha` take one value for all datasets (no per-dataset lists).
- Not changed: the legend key of a dataset shows the tail's width when `tail_linewidth=` is set. No example in examples/ was edited (the prebuilt gallery record is not to be touched).

Tests: tests/test_issue299_animation_arguments.py (new).

## Issue #297 (2026-11-23, branch `post-1.2.0`, for 1.2.1; not pushed)

Six gallery layout points plus one suspicion. Three library fixes, four example edits. Tests: tests/test_issue297_layout.py (new, 32). Scratch renders (before/after per point) were kept outside the repo.

Per point:

1. `animate_conversation`, cube on the left canvas edge. Cause: a single animated 3-D axes fills the canvas and the frame functions draw the cube at box zoom 1.125 (`_anim_box_zoom`); on a square figure the cube seen corner-on is wider than the canvas (measured on 8 x 8 in: x from -4 to 826 px of 800 over the turn; in landscape 6.4 x 4.8 it is 78 to 575 of 640, which is why the old margin tests never saw it). Fix: `plot._keep_cube_clear` (called in `plot()` after the title margin, colorbar and legend fit, before the companions): projects the cube's eight corners at every camera azimuth of the turn (`_camera_azimuths`, as the panel fit does), and lowers the zoom factor the frame functions already multiply by (`_cell_zoom_scale`, now also on the animation as `_hyp_zoom_scale`) until the corners are 2% of the canvas inside it and the cube's outline (separating-axis test against a box, closed form in `_cube_excess`) is that far from the legend and from the figure's other axes. Static single plots go through the same function with `_StaticCubeZoom` (the handle static grids use); a static cube only ever changes when the legend was put inside the axes. Not run for a plot drawn into an `ax=` (other panels of the figure would count as obstacles), for `panels=` grids, or with `zoom=` above 1.
2. `animate_weather_decades`, last marker cut. Cause: the example's own `fig.add_axes([0.615, .., 0.38, ..])` ends at 0.995 of the width and its head marker is drawn unclipped, centered on that edge at the last month. The library's `companion=` panel was checked and is fine (ends at 0.98; marker 14 px inside). Fix in the example (and the tutorial notebook's source): width 0.37.
3. `plot_labels`, label boxes on each other. Cause: every label is annotated at the fixed offset (-20, 20) pt. Fix: `matplotlib_backend.declutter_labels`, run at draw time by `_LabelDeclutter`, an invisible child artist of the axes with the lowest zorder (so it runs before the labels are drawn, with the final layout, at any dpi, at the frame's camera angle: one path for static, hand-rotated and animated). Greedy in label order: a label keeps its place unless its box covers an earlier one, else the first free corner (4 corners x 3 distances), on the canvas if possible, else the least overlap. No hysteresis on purpose: frames must be idempotent. A label moved by the reader or the caller (its `xyann` no longer what hypertools set) is left alone. More than 300 visible labels: nothing is moved.
4. `plot_shapes_zoo`, teapot. Cause: the example's `,` pixel marker; the teapot has 1,728 rows but only 301 distinct points (it is the Utah teapot's control points). Fix in the example: `.` at markersize 2.5 for clouds under 5,000 points. It is visible now; it still does not read as a teapot from the PCA view, because it is 301 control points.
5. `animate_market_sectors`, legend against the cube. Cause: the example added its own `ax.legend(loc='upper left')` to the full-canvas axes after the call, so the library knew nothing of it. Fix: the example uses the library's legend (`legend_colors=` pairs, `legend_kwargs={'loc': 'upper left', 'fontsize': 8}`) and `_keep_cube_clear` keeps the cube off it. Side effect: the "Market" key line is drawn at the width of the other keys (it was 3.4).
6. `plot_datasets_tour`, Bechdel panel. Cause: the data. Dollar columns from $828 to $3.2e9; PCA's first axis is stretched by a few blockbusters, and the display transform (documented: mean-center, then one pooled min/max into [-1, 1]) centers on the MEAN, so skewed data sits in a corner; the point "outside the box" is the maximum, whose center is exactly on a cube face. Fix in the example: log10 of the dollar columns.
7. Animated markers-only plot with continuous `hue=`: real defect. `plot()` sent it to `_apply_multicolor_markers`, which removes the animated line artists and scatters every observation once. Measured: 80 of 80 points on frames 0, 15 and 29; 2-D frames byte-identical; 3-D only the camera moved; trails never drawn. Fix: `_apply_multicolor_animation(markers=True)`: the same per-frame path as lines, with a scatter (the hidden artist's glyph and size) per head and per trail, filled from the hidden artist's `_hyp_row_window`. `spin` keeps the static scatter (all points, camera only).

Behavior changes for existing figures (all in the CHANGELOG): animated single 3-D plots on square/portrait figures draw the cube about 10% smaller (zoom factor 0.90 on 8 x 8, 5 x 8, 6 x 6); landscape with the default legend 0.946 (the cube corner was 5.6 px inside the legend's box at the widest view); landscape with a colorbar 0.98; landscape without either unchanged (1.0). Static plots unchanged unless a legend is inside the axes. `zoom=` above 1 is not fitted. An animated figure shows frame 0 when `plot()` returns (the fit draws frame 0 through the updater alone, no hooks). Regression renders, old library against new: 11 static examples (23 figures: basic, legend, hue, colorbar, 2D, multiindex, nested_lists, density, surface, predict, align) and animate_panels, animate_painting_embeddings, animate_spin, animate_trails, animate_trails_mix, animate_surface_morph, save_movie pixel-identical at frames first/mid/last; animate (legend), animate_forecast and animate_morph_zoo changed as described (smaller cube). animate.py's MDS panel differs run to run on the same code (unseeded MDS).

Tried and dropped:

- Room = the axes' own box (`get_position(original=True)`): after `_fit_right_legend` that box is the SQUARE the cube is drawn in (it re-sets the position from the active one), so every legend plot shrank to 0.90. The room is the canvas; the legend and other axes are obstacles.
- Changing the display transform to center on the bounding box (it would cure point 6 in the library): it is the documented `axis_scale='unit'` contract and moves the data in every figure. Left for the owner to decide; not done.
- Hysteresis for label corners (keep the previous corner): breaks frame idempotence.
- Deciding label places once at annotate time: stale after `tight_layout`, a resize or a rotation.
- Explicit calls of the declutter in `update_position`/`_sync_anim_labels`: three call sites instead of the one draw-time artist.

Not done: the two tutorial notebooks (market_sectors, weather_decades) have their SOURCE cells updated to match the scripts but were not re-executed, so their stored outputs are from the old code. docs/prebuilt_gallery.json untouched. Plotly: labels and the cube fit are matplotlib-only changes.

## 1.2.1 review round 1 (2026-11-23, branch `post-1.2.0`; not pushed)

Two reviews of the #299/#297 work: a tester who ran the code (T1-T7) and a code-only review (C1-C8). Each item was reproduced with a real call first, on the unmodified head in a scratch worktree, then fixed. Tests: tests/test_review_round1_1_2_1.py (new, 86; 69 of them fail on the head before this round, the rest are "good values still work" controls).

Per item:

- T1 labels jump between frames. Confirmed: 71 changes of place in 60 frames for 8 labels on a spin, 777 for 30, 233 on a window animation. Cause: `declutter_labels` put every label back up-left and placed all of them afresh on every draw, so any small move of the points reshuffled the greedy result. (Superseded for ANIMATED plots by the follow-up below: placed once, then held.) Fix (matplotlib_backend.py `declutter_labels`): the placement is stateful. A label keeps its place (`_hyp_place`) until its box meets the box of an EARLIER label (or a place it was moved to has left the canvas); then it takes, among the 8 directions x 3 distances that are clear of every other label (with 3 pt of room) and on the canvas, the shortest reach and at that reach the nearest place; the place it last left ranks last at its reach; with no clear place it moves only if that halves what it covers. It never goes back on its own. Up to 3 passes per draw. Now 26 / 130 / 27 changes on the same three runs, 0 or 1 straight-back hops, no overlaps for 8 labels (30 labels on 640 x 480 still overlap in places: 3.8 pairs on average, 6.0 before). Because a draw now depends on where the labels were, `reset_label_places` puts them home: at the end of `plot()` and of `_lay_out_panel_grid` (so a draw made during layout leaves nothing behind), when an animation's frame number goes down (`_sync_anim_labels`: loop, save after preview, `draw_frame` of an earlier frame), and for a label that is hidden. Two passes through an animation give identical placements (tested). Dropped: ordering the candidates by distance moved alone (labels walked out to the 70 pt reach); a 1 px tolerance on "covered" (left sub-pixel overlaps that the box test counts).
- T2 / C3 animated `'-o'` + continuous hue draws no markers. Confirmed (2-D, 3-D, panels; tail and trail). Cause: `_apply_multicolor_animation` hides the line artist, which carries the markers too, and stood a line collection in for it. Fix: each such collection gets a scatter (`coll._hyp_markers`, tagged `_hyp_markers_of=(dataset, role)`) filled from the hidden artist's `markevery` (the observation vertices) with the per-observation colors.
- T3 `tail_linewidth=` on a markers-only format silently ignored. Confirmed. `_warn_ignored_animation_styling(any_line=)` warns when no dataset has a line.
- T4 values inside the dicts. Confirmed for all the listed values except `suptitle=b'bytes'`: that is decoded as UTF-8 on purpose, exactly as `caption=b'...'` is (documented and tested now, not changed). Fix: `_check_style_values` (plot.py), called by `_validate_head_marker`, `_validate_figure_text_kwargs` and `_normalize_title_kwargs`; errors read `head_marker['size'] must be a marker size in points greater than 0; got -3.` (the key is the one the caller wrote).
- T5 docstring said plotly raises; a static plotly plot warns. Confirmed. Decided by what `chemtrails=` does on a static plotly plot (warns): code kept, docstring/guide/changelog corrected (static: warning on either backend; animated plotly: NotImplementedError). Grammar: "head_marker only affects".
- T6 `hold=` in a gif. Confirmed: 10 stored frames, the last 1100 ms, for a 20-frame animation (Pillow merges identical frames); mp4 has all 20. Kept; docstring, guide and changelog now say exactly that; test reads the gif back.
- T7 cube-to-legend gap under 2%. Not confirmed as stated: the tester divided a bounding-box gap by the canvas WIDTH. Measured on every frame with the real renderer, cube outline to legend box: 1.99% of the canvas's shorter side; cube to canvas edge 1.95-1.97%. The changelog now says "about 2% of the shorter side (1.95% to 2%)"; a test measures it (at least 1.9%).
- C1 serial marker trail + hue crashes at frame 0. Confirmed (AttributeError `set_segments`, 2-D and 3-D, chemtrails/precog/bullettime). Fix: an empty window goes through `_set_segments`, which knows scatters.
- C2 serial hue trails keep stale geometry. Confirmed (frame 5 then frame 0: 432 segments still drawn; same at the last frame). Fix: the serial updaters (2-D and 3-D) record `_hyp_row_window = (0, 0)` on a trail they clear.
- C4 trail alpha with hue. Confirmed (0.5 instead of 0.1). Fix: the stand-in collection takes the hidden artist's alpha. Found with it: an animated hue LINE ignored `alpha=` (drawn opaque); fixed the same way.
- C5 suptitle over a top caption that grows. Confirmed (single 2-D/3-D and grids). Fix in `_add_caption`: each hypertools figure text remembers its tallest text (`_hyp_tallest_text`) and earlier ones are shown at it while a later text's room is measured.
- C6 label anchors stale after `_keep_cube_clear`. Confirmed (8.7 px off). Fix: `declutter_labels` projects every 3-D label's anchor at draw time.
- C7 head markers do not fade with hue. Confirmed. Fix: the head collection carries `_hyp_head_marker`; the fade updater also fades `_hyp_markers`.
- C8 spin draws the whole line at `tail_linewidth=`. Confirmed. Fix: `_head_linewidths(linewidths, style)` leaves spin/morph alone.
- N1 notebooks: docs/tutorials/market_sectors.ipynb and weather_decades.ipynb were regenerated (scripts/generate_tutorial_notebook.py; sources identical to the committed ones once the generator's prose was in American spelling) and executed with scripts/execute_tutorial.py (kernel hypertools-venv, install cell skipped, venv still editable afterwards). Market fetched live data: 318 months through Oct 2026 (317 through Sep before). Both mp4 clips were re-rendered by the notebooks (market 5.4 MB, weather 4.5 MB); a frame of each was looked at (legend clear of the cube; last-month marker inside the canvas). scripts/add_colab_install_cell.py was not run (the generator only imports `portable_video_source` from it; its `main()` is guarded).
- N2 neighbors found and fixed: mixed formats with hue (`['-', 'o']`) drew every dataset as markers, animated (`markers=` is per dataset now) and static (`only=` in `_apply_multicolor_lines`/`_apply_multicolor_markers`, each dataset's own glyph); `dataset_fade=` mapped a trail to a dataset by its place among the trails, wrong with `chemtrails=[False, True]` (trail lines carry `_hyp_trace_index` now); `title_kwargs=` values were as unchecked as the new dicts'; scripts/generate_tutorial_notebook.py still had British spellings in its prose, so regenerating undid the American spelling of the committed notebooks.

Behavior changes this round adds (all in the changelog): `title_kwargs={'size': -3 | 0 | nan}` raises instead of drawing nothing; an animated hue-colored line honors `alpha=`; a hue trail is `trail_alpha` x `alpha`; static and animated mixed-format hue plots draw lines as lines.

### T1 follow-up: an animation's labels are placed once and hold (2026-11-23)

Design decision from the coordinator after round 1: the stateful declutter still moved labels during an animation (26 switches in 60 frames with 8 labels) and made `draw_frame(i)` depend on the frames drawn before it. Rule now: frames are deterministic.

- matplotlib_backend.py `freeze_label_places(ax, draw)` (new): puts the labels home, shows every label (hidden ones included), makes one real draw (which runs the same `declutter_labels`), restores visibility, sets `ax._hyp_labels_frozen`. `declutter_labels` on a frozen axes only reprojects 3-D anchors and returns; `reset_label_places` skips frozen axes; the rewind reset in `_sync_anim_labels` is gone. No second placement implementation.
- plot.py `plot()`: after the final layout, for an animated single axes: frame 0 through the frame function, then `freeze_label_places`. `_lay_out_panel_grid`: the same per cell for an animated grid.
- Static figures are unchanged (declutter at draw time, stateful under hand rotation, `reset_label_places` at the end of `plot()`).
- Measured (60 frames): switches 0 for spin 8, spin 30, window 3-D, window 2-D and reveal, on two passes. Cost of holding: label boxes meet at later views (overlapping pairs mean/max: spin 8 3.2/6, spin 30 20.9/36, window 30 3.6/11; the stateful version had 0 for spin 8). With 30 crowded 2-D labels the one placement itself leaves 3 overlapping pairs (26 at the usual places).
- Determinism, pixel for pixel (scratch `det.py`, 14 configurations: labeled spin/window/reveal, parallel, line+marker with trails and head marker, hue window, serial, serial hue 2-D, dataset_fade, caption+suptitle, morph, panels with labels, legend spin): frame i in order == shuffled == reversed == the frames a real Pillow writer received from `animation.save` == the same frames of a second identical call. Static `plot()` twice: identical. So history dependence applied to labels only (the stale serial hue trail of C2 was the other case and was fixed in round 1).
- Tests: section A of tests/test_review_round1_1_2_1.py rewritten (zero switches, first-view placement, shuffled/reversed pixels, saved == drawn, panels, static twice, hysteresis moved to the hand-rotated static figure). tests/test_issue297_layout.py passes unmodified.


## 1.2.1 review round 2 (2026-11-23, branch `post-1.2.0`; not pushed)

Library commit 9fed77ec, then tests/docs/CHANGELOG in the next commit.
Tests: `tests/test_review_round2_1_2_1.py` (117 of its tests fail on
82933121). 1.2.0 reference venv: scratchpad `review121/old` (mpl 3.10.8).

Per item (confirmed? cause; fix):

- T1 confirmed. `_check_style_values` had its own list of allowed types.
  Now it asks matplotlib (`Text().update` / `Line2D(**{k: v})`), and adds
  only: size/markersize/linespacing must be positive and finite (also as
  numeric strings), text color not `'auto'`, family a str or list of str.
  None-valued keys are dropped in `_validate_figure_text_kwargs` and
  `_validate_head_marker`. Against 1.2.0 over 290 `title_kwargs` values the
  only differences left: size False/0/negative/NaN (and as strings) and
  linespacing 0/negative/NaN now raise; 1.2.0 accepted them.
- T2 confirmed. The axes kept only the last call's label list.
  `_watch_labels` merges lists per axes.
- T3 confirmed. Each panel warned for itself. `_warn_ignored_grid_styling`
  collects one case per panel and warns only if true of all.
- T4 confirmed. Placement used the renderer's pixels. `declutter_labels`
  now works in points; text sizes come from ONE Agg renderer at 100 dpi
  (`_label_size_pt`). Tried first: base `RendererBase` metrics; dropped,
  its sizes were up to 4.5 px smaller than Agg's drawn boxes and labels
  "clear" in points overlapped by a pixel on screen.
- T5 fixed the garbled line and rewrote the claims that changed.
- T6 `{'marker': True}` is 'o', `{'marker': False}` no marker.
- T7 made true: caption and dynamic title text is resolved on every frame
  (`_caption_text_frames`, `_sampled_dynamic_titles`). Cost about 0.9 ms a
  frame for a callable; no extra draws.
- T8 see timings in the hand-back. Causes: a full declutter on every
  draw (now memoized in `ax._hyp_label_answer`, skipped while `plot()`
  lays out via `labels_held_while_plotting`), a draw in
  `freeze_label_places` (now none), `_cube_corners_px` projecting per view
  (now one vectorized projection, equal to 1e-12 px).
- P1 confirmed. Trail lists were checked against drawn runs.
  `_trail_flag_per_run` maps dataset flags onto runs (also `surface=`).
- P2 confirmed, NOT changed: `random_state=None` is the documented
  default for every random stage; `random_state=0` gives the same picture.
  Added a sentence to the `cluster` docstring.
- P3 decided: a warning on both backends; plotly makes no annotations, matplotlib keeps its hidden label artists (an existing test reads them).
- C1 confirmed (in 1.2.0 too for legend/caption/panels).
  `_measurement_renderer` restores the canvas and `_original_dpi`.
- C2 confirmed. `_watch_labels` reinstalls the declutter artist when it
  is not among the axes' children and clears the frozen flag when no
  earlier label survives.
- C3 confirmed. `_TraceFollowers` (one invisible artist per axes) makes
  the hue marker scatter and the head marker follow their leader. Tried
  and dropped: a `_hyp_shown` flag that re-showed followers each draw
  (broke a round 1 test that hides the dots directly); closures on the
  artists (figure no longer pickles).
- C4 confirmed, static plot too. `_HueMarkerLook`.
- C5 confirmed with real MouseEvents (1 of 12 labels changed place after
  a 0.7 degree drag on 82933121, 0 now). Hidden labels keep their place.
- C6 confirmed. `isinstance(marker, str)` before the emptiness test.
- C7 confirmed. No measurement draw at all for label placement.

Test note: a caller-supplied 3-D axes is smaller than hypertools' own, so
the same 30 crowded labels leave 3 overlaps there and 0 in an own figure;
that is crowding, not a C2 defect (cleared axes == never-used axes).

## 1.2.1 review round 3 (2026-11-23, branch `post-1.2.0`; not pushed)

Four items from the third review, each reproduced with real code before
the fix. Tests: `tests/test_review_round3_1_2_1.py` (20 of them fail on
20192664). Scratch renders: session scratchpad `round3fix/`.

| Item | Confirmed | Fix |
|-|-|-|
| 1. `test_datatype_gate` failed on three `isinstance(..., np.ndarray)` sites in `plot.py` | yes | The morph `labels=` check is now `_any_label_given` (shared predicates `is_array_dataset` / `is_series_like`; a Series of labels is read per entry). `_trail_flag_per_run` unwraps an OPTION array, so it has an ALLOWLIST entry like the same line in `trails.py`. |
| 2. Stale label callbacks after `ax.clear()` | yes, 2-D and 3-D: `AttributeError: 'NoneType' object has no attribute 'transData'` out of the button release, new labels left hidden. 1.2.0 did not raise (round 2 regression); 1.2.0 did add a pair of callbacks per call | `_connect_label_events(fig, ax)`: one press/release pair per axes per canvas, reading `_label_entries(ax)` at event time. Which axes a canvas serves is a `WeakSet` on the canvas, so nothing unpicklable is on the axes or figure. |
| 3. `_label_size_pt` cache key | yes: after `set_rotation(90)` the size stayed (78.6, 10.8) pt, and after `set_parse_math(False)` too | Key now also holds `parse_math`, rotation, rotation mode, `transform_rotates_text`. Not the alignment: placing a label changes it, and a key that held it made every redraw a new problem (`test_a_redraw_of_an_unchanged_figure_places_nothing_again`). A wrapped text is never cached. |
| 4. Half `fillstyle=` with continuous `hue=` | yes: only the filled half was drawn (`scatter` ignores `get_alt_path()`). 1.2.0 ignored `fillstyle=` and drew full discs | `_HueMarkerLook.wear` turns the scatter into `_HalfFilledMarkers` / `_HalfFilledMarkers3D` (module-level subclasses, so figures still pickle), whose `draw` draws the collection and then the alt path with the same edges and the `markerfacecoloralt` fill (none by default). One artist, one offset per observation, so the animated path (`_make_scatter`, `_set_points`) needed one line. |

- Item 4 pixels were compared with `ax.plot(..., fillstyle='left')`.
  With `'-o'` the hue line shows through the empty half, as in matplotlib;
  a static plot draws the line over the markers (zorder 2 over 1, as
  before), so the tests count outline pixels there and read single pixels
  for `'o'` only.
- matplotlib's own per-label drag callbacks (`DraggableAnnotation`, 2-D)
  are six `button_release_event` entries per labeled call; matplotlib
  drops the ones of removed labels at the next release. Not ours.
- CHANGELOG 1.2.1: the labels entry (shared callbacks, restyled labels
  measured again) and the hue marker entry (half fill styles,
  `markerfacecoloralt=`).

## 1.2.1 review round 4 (2026-11-23, branch `post-1.2.0`; not pushed)

Nine items from the fourth review, each reproduced with real code first
(scratch `round4fix/repro.py`, the tester's scripts in `review121c/`,
the 1.2.0 venv `review121/old`). Tests:
`tests/test_review_round4_1_2_1.py` (111; 64 of them fail on a0ef72b8).
Before/after renders: session scratchpad `round4fix/`
(`before_item*.png`, `after_item*.png`, `old120_item*.png`).

| Item | Confirmed | Cause; fix |
|-|-|-|
| 1. Panel labels on the shared legend (regression) | yes: 6 of 12 seeds in 3-D and 6 of 12 in 2-D put a label on the legend, a panel title or the suptitle; 1.2.0: 0 on the legend | The declutter only knew the other labels, and it runs after the layout that made room for the labels at their usual place. `_label_obstacles_pt` (matplotlib_backend.py): every legend (axes and figure), every colorbar axes with its tick labels, every axes title, every figure text, in points, widened to whole points. `declutter_labels` moves a label that lies on one; a candidate place on one is a last resort (it costs the label's whole box plus `_LABEL_OBSTACLE_WEIGHT` = 4 times the covered area, as an off-canvas place costs the whole box); they are part of the memo key. One path: static draw and `freeze_label_places`. |
| 2. Labels on title text (1.2.0 too) | yes | Same rule. A single label is looked at too (it used to return early with fewer than two labels). Tiny figure (3 x 2.5 in, size 18 texts, 30 labels): labels on texts 9 -> 0, on the legend 1 -> 0 (matplotlib 3.10 and 3.11), label pairs on each other 15 -> 29 (3.11: 23 -> 33): there is no room, and labels lie on each other rather than on the texts. First version (furniture cost = label cost) left 1 label on the title on 3.10 and 2 on the legend on 3.11, although with every label at its usual place none is on furniture; hence the last-resort cost. |
| 3. 2-D `legend_kwargs={'loc': ...}` | yes, static and animated, 1.2.0 too | Not a stray `bbox_to_anchor`: the 2-D axes fills the canvas and the frame is a square drawn inside it (`frame_2d`), so the corner of the axes is outside the frame. `frame_2d` records the square (`ax._hyp_frame_box`, data coordinates) and `legend_call_kwargs(ax=)` anchors a NAMED `loc` to it (`bbox_to_anchor` = the square, `bbox_transform=ax.transData`). Not touched: a caller's `bbox_to_anchor`/`bbox_transform`, a coordinate pair as `loc`, `axis_scale='data'` (no square), 3-D (the legend is inside the axes and `_keep_cube_clear` shrinks the cube), `panels=` (documented: `loc` names the side of the grid). |
| 4. Two head markers with categorical `hue=` | yes | Each category run is its own line with its own marker artist, and a run the reveal had passed kept its marker at its end. `_head_holders`: per input dataset (the `ownership`), only the last reached run places its marker; the others are emptied. Color is the run's, so the category's. |
| 5. Serial frame 0 empty | yes (2-D, 3-D, panels) | `serial_reveal_counts` is 0 everywhere on frame 0. Not changed (an existing contract with tests); `_serial_head_holders` gives the first dataset the marker on frame 0 and it is placed on `data[:1]`. With a continuous `hue=` it wears row 0's color (`_follow_head`). |
| 6. `head_marker=` dict keys | yes | Aliases `ms`, `mec`, `mew`, `mfc`, `mfcalt`, `c`, and `facecolor`/`facecoloralt`; keys `markerfacecolor`, `markerfacecoloralt`. One option under two names raises. Found by the new test: a bad face color was not refused by `_check_style_values` (a `Line2D` takes it and fails in the draw); now checked like the edge color. |
| 7a. `trail_alpha=` warning under `'window'` | yes | Own message for `animate='window'` (it draws no trail whatever is passed). |
| 7b. `hold=0.01` | yes | DECIDED: the rule stays `round(hold * frame_rate)` and a positive hold that rounds to no frame WARNS, naming `1 / frame_rate`. The coordinator's default was "at least one frame"; not taken because `tests/test_issue299_animation_arguments.py::test_hold_zero_is_the_animation_it_always_was` asserts `hold=0.04` at 10 fps adds no frame, and existing tests are not to be changed. Switching is `max(1, ...)` in `_validate_hold` plus that assertion. |
| 8. Undocumented definitions | yes | Docstrings for `hide_labels`, `update_position`, `_HalfFilledMarkers.draw`, `_HalfFilledMarkers3D.draw`. |
| 9. LSL tests under load | failure NOT reproduced | Measured on 12 cores: discovery by name 6 ms idle, a steady 1.0 s with 144 CPU-bound processes (load average 118); first sample 20 ms either way; the file passed (27) with 48 burners in 197 s. So I don't know the exact cause; the quantized 1.0 s points at missed resolve queries, and a few more of them use up 5 s. Tests only: discovery `timeout=5.0` -> `30.0` (returns as soon as the outlet answers) in test_lsl_streaming.py, test_io_audit_lsl.py, test_io_audit_final.py; the stall test's `timeout=2.0` -> `10.0` (it is also the wait for the first sample). No assertion changed. `hypertools/io/lsl.py` untouched. |

Tried and dropped:

- Obstacle boxes measured on the label metrics renderer at 100 dpi
  (`Text.get_window_extent(dpi=)`): it resets the figure's dpi inside a
  draw and is wrong during a `bbox_inches='tight'` save. The boxes are
  measured on the draw's renderer and widened to whole points instead.
  UNCERTAIN: a place that is free at one dpi and covered by a fraction
  of a point at another could still differ between a PNG and a PDF; the
  round 2 cross-renderer tests pass.
- Changing `serial_reveal_counts` so frame 0 reveals one row: it moves
  every serial animation's first frame and its `revealed_counts`.

Not done: a `companion=` panel and other panels of a grid are not label
obstacles (labels may reach over a neighboring panel, as before). A
per-frame title or caption is an obstacle at its frame 0 extent only.

Checks run for round 4 (machine load average 120 to 430 throughout, from
other suites running at the same time): new file 111 passed (also 111
under matplotlib 3.11.2); review rounds 1 to 3: 343 passed, 1 skipped;
test_issue297_layout 32; test_issue299_animation_arguments 122; the 73
other files named label/legend/hue/panel/caption/title/animat/docstring/
gate/signature/changelog/lsl: 2687 passed, 9 skipped, 5 failed, then all
5 passed on a rerun of their files (363 passed, 8 skipped with the
docstring, gate, signature, changelog and new files). Of the 5: one was
mine (`test_panel_rosters_cover_every_documented_per_dataset_argument`
reads the phrase "per dataset" in a docstring as a per-dataset argument;
the `head_marker` text was reworded). Three were time bounds in
tests/test_animation_export.py (kill within 20 s / 30 s) and one was
`test_io_audit_lsl.py::test_multiple_matching_streams_warn` ("DID NOT
WARN": with `minimum=2` and 30 s only one of two outlets was heard) at
load average about 400; not reproduced on the rerun, cause not
established beyond the load. `ruff check .` clean.

## 1.2.1 round 5: per-frame labels (owner decision 2026-11-23)

Branch `post-1.2.0`, after a11c09e9; not pushed. The owner rejected the
design of 82933121 (an animation's labels decluttered once and held,
`freeze_label_places`). The decision, word for word:

> for axis labels, they should move with the rotations so they remain
> aligned with the axes. same for labels of individual datapoints (e.g.,
> in every frame the labels should have correct positioning). for static
> text like legends, titles, and other annotations: compute placement
> once based on the *worst case* rotations of the axes, and then place
> once so that the static text (and other static content) remains stable
> across frames.

Added while the work was under way (relayed by the coordinator), also the
owner's words:

> slight shifts between frames are OK but not if they are wobbling back
> and forth so that it looks glitchy. movement needs to be in a
> consistent direction

So a frame's picture is a function of (the animation, the frame index),
not of the frame alone and never of the render order.

### What was built (hypertools/plot/matplotlib_backend.py, plot.py)

- `_arrange_labels(places, left, anchors, sizes, pads, canvas, obstacles, ...)`:
  the placement rule, taken out of `declutter_labels` unchanged. ONE
  function places a static figure's labels (`declutter_labels`, from where
  the labels hang) and every frame of an animation (`solve_label_tracks`,
  from where the frame before left them). The animation passes four
  things a static draw does not: `ahead` (the anchors of the coming
  frames: of equally clear places the one that stays clear longest is
  taken), `avoid` (places left a short while ago), `resting` (a label that
  has just moved is not moved again for another label) and `strict` (a
  label at its usual place that leaves the canvas moves too).
- `solve_label_tracks(ax)`: runs the frame function for every frame
  (updater only, nothing rendered, no `on_frame=` call), records each
  label's anchor in points and whether it is shown, then goes through the
  frames in order. A loop without a seam (`_labels_loop`: same labels on
  the first and last frame, and their points no further apart than between
  other neighboring frames) is gone round until a round repeats the round
  before, at most 16 times (fewer above 40 labels: 640 // labels, at
  least 3). `_settle_short_stays` then removes every stay shorter than
  three glides (the label keeps the place before, or goes straight to the
  one after, where that keeps it on the canvas and off static content);
  this is what guarantees no out-and-back. `_ease_label_tracks` turns the
  steps into glides (Hann weights, 0.5 s, 5 to 15 frames, finished at the
  frame the label had to move at). A glide that would cross static content
  whose both ends are clear of becomes a jump (`'forced'` in the table).
- The table (`label_tracks(ax)`: `offsets` (frames, labels, 2) in points
  from the point to the middle of the text, `places`, `shown`, `forced`,
  `periodic`, `ease_frames`, `rounds`) is kept in a module
  `WeakKeyDictionary`, not on the axes: it holds the frame function, and
  an attribute on the axes broke `pickle.dumps(fig)`
  (tests/test_review_round2_1_2_1.py::test_an_animated_figure_still_pickles).
- `_sync_anim_labels` (every animation style's updater) calls
  `_apply_label_tracks(ax, num)`; a table whose signature (canvas size in
  points, axes box, label ids) no longer matches is worked out again
  there. `declutter_labels` on such an axes only reprojects and applies
  the row. `label.update_positions(renderer)` was dropped from
  `_sync_anim_labels`: the draw does it, and it cost about 1 ms a label on
  every frame.
- Static content, C: `_keep_cube_clear` measured the cube's corners only
  (every 3 degrees of the turn, which is sufficient for the cube: its
  reach is smooth in the angle, so between two samples it exceeds the
  larger by at most 1 - cos(1.5 degrees) = 0.03%). The AXIS LABELS were
  not measured at all: on a 5 x 5 inch spin "the z axis" was up to 61 px
  off the canvas, and with legend + title + caption + suptitle the axis
  labels crossed static content on 6 of 60 frames (scratch
  `axes_check.py`). Now the axis texts are measured at EVERY frame's
  camera angle (`frame_azimuths`, read by running each frame's updater;
  which edge a label hangs from changes discretely with the view, so a
  sample is not provably enough) by drawing the three axis objects alone
  (`axis_text_boxes_px`), and the cube is zoomed out until each is on the
  canvas and off `_label_obstacles_pt`. Grids keep `_fit_panel_cubes`
  (axis labels sampled every 30 degrees); checked on every frame of a
  2 x 2 labeled grid with and without legend/titles/caption: nothing off
  the canvas, on static content or on another panel's label
  (`test_a_grids_axis_labels_stay_on_the_canvas_and_apart_every_frame`).
- `_label_obstacles_pt` gained two kinds of obstacle, for static figures
  too: every OTHER axes of the figure (a label that reached into the next
  panel of a grid was painted over by that panel's background; seen on the
  first panels contact sheet), and a 3-D axes' own axis texts.
- Plotly, F: `labels=` are layout annotations in data coordinates
  (`_build_point_annotations`), added once to the base layout. They follow
  their points as the scene turns (plotly projects them), at one fixed
  offset, are never decluttered and are shown on every frame, also for
  points not revealed yet. Not changed (placement happens in the browser).

### Measured

Turn-backs (an offset component goes more than 1 pt one way and more than
1 pt back within one glide's time on either side), largest step between
two frames in points, and changes of place, 60 frames at 15 fps; "alone" =
every frame placed on its own from the usual places (`_independent=True`):

|case|turn-backs table|turn-backs alone|max step table|max step alone|changes table|changes alone|
|-|-|-|-|-|-|-|
|spin, 8 labels|0|29|33.2|107.8|3|95|
|spin, 12 labels, legend, title, suptitle, caption|0|46|12.6|188.5|3|139|
|spin, 30 labels, legend, title, caption|0|164|47.1|188.6|31|720|
|window, 3-D, 30 labels|0|59|0.0|188.5|0|223|
|window, 2-D, two lines|0|0|0.0|57.5|0|8|
|serial, 2-D|0|2|0.0|188.5|0|15|
|2 x 2 panels, 48 labels|0|251|33.9|166.3|22|700|

In a 2-D animation the points do not move on screen (only a 3-D camera
turns), so labels there appear and disappear and need no glide.

Timings, 3-D spin of 200 points, 60 frames, CPU seconds (`time.process_time`,
best of 3; the machine's load average was 90 to 180 throughout, so wall
times were useless: the same frame measured 107 ms and 598 ms minutes
apart). Before = a11c09e9 (scratch `round5fix/head`), scratch `bench_cpu.py`:

|labels|plot() before|plot() after|frame before|frame after|
|-|-|-|-|-|
|0|0.008 s|0.008 s|2.9 ms|2.9 ms|
|20|0.054 s|0.46 s|37.4 ms|37.0 ms|
|60|0.138 s|1.89 s|106.7 ms|106.4 ms|
|200|0.481 s|5.39 s|365.1 ms|324.4 ms|

A frame costs what drawing the label boxes costs (about 1.7 ms a label);
applying the table is a lookup. `plot()` pays for the table: one updater
call per frame plus up to 16 rounds of the loop (3 rounds at 200
labels). No cap was added.

### Tests

New: tests/test_review_round5_1_2_1.py (50 tests). Rewritten, because
they pinned the rejected design (every other assertion kept or made
stricter):

- tests/test_review_round1_1_2_1.py:
  `test_animated_labels_hold_their_places` ->
  `test_animated_labels_follow_one_table`;
  `test_animated_labels_are_placed_for_the_first_view` ->
  `test_animated_labels_are_placed_for_every_view`;
  `test_animated_panels_hold_their_label_places` ->
  `test_animated_panels_show_a_frame_the_same_way_every_time`.
- tests/test_review_round2_1_2_1.py:
  `test_an_axes_reused_after_an_animation_places_labels_again` (reads
  `label_tracks(ax)` instead of `_hyp_labels_frozen`);
  `test_labels_added_to_an_animations_axes_are_placed_once_and_held` ->
  `test_labels_added_to_an_animations_axes_join_its_table`;
  `test_an_animated_grid_places_its_labels_without_a_draw_per_panel`
  (same attribute change);
  `test_an_animations_labels_are_placed_as_a_draw_would_place_them`
  (a row of the table placed on its own against a real static draw).
- tests/test_review_round4_1_2_1.py:
  `test_animated_panel_labels_are_placed_off_the_legend_and_texts` (off
  legend, titles and figure texts on EVERY frame, and a frame shown again
  has its labels where they were).

### Known limits

- With many labels on a small cube labels still overlap each other (30
  labels on the default figure). Static content and the canvas edge are
  kept clear on every frame of every tested case.
- A label is allowed to overlap another for up to three glides (1.5 s)
  rather than move twice in a row: that is the price of no wobble.
- A dynamic (per-frame) title is measured as it stands when the table is
  worked out, not per frame.
- Looping animations whose rounds do not repeat within the round limit
  keep the last round; `_settle_short_stays` still removes short stays
  round the seam.
- The table costs time in `plot()` (see timings), most of it in the
  rounds of a loop.

Contact sheets and GIFs: scratch `round5fix/sheet_*.png`, `anim_*.gif`.

## LSL discovery failure in full-suite runs

2026-11-23. `tests/test_lsl_streaming.py::test_lsl_stream_name_takes_precedence_over_type`
failed in two full-suite runs ("no LSL stream found (name='HypertoolsTestStreamB-...')",
at 5 s with four more tests of the file, then alone at 30 s). The machine crashed
at about 17:45 during this work; everything below marked "after the restart" was
measured again on the fresh boot, the rest could not be repeated.

What was established:

- Why that test: liblsl's unicast discovery query (127.0.0.1:16571) reaches only
  the oldest socket bound to the port, so only one outlet on the machine can be
  found that way. The test's second outlet is the one case in the file that needs
  a multicast query to arrive. (Unicast to the oldest socket only: plain sockets,
  10 of 10 against 0 of 10.)
- A race in liblsl 1.17.7 on macOS 26 (after the restart too): the resolver sends
  each multicast query on loopback and then switches the same socket to en0
  (`resolve_attempt_udp::send_next_query`); a loopback datagram followed at once
  by that switch is lost (plain sockets: 0 of 30 with no delay, 30 of 30 with
  1 ms, 30 of 30 without the switch), and nothing sent on en0 is looped back (0
  of 30). Per 20 query waves 18 / 8 / 4 delivered the 224.0.0.1 / 224.0.0.183 /
  239.255.172.215 datagram. After the restart, default configuration: the second
  of two outlets found at once in 8 of 12 resolves, after 0.5, 0.5, 1.0, 2.5 s in
  the rest. This alone costs seconds, not 30 s.
- Before the restart the kernel dropped loopback UDP for minutes at a time, on any
  port: a fresh socket got 0 of 20 datagrams; "dropped due to full socket buffers"
  (`netstat -s -p udp`) +3,096 in 10 s (1,006,018 at 16:42, 1,699,882 at 17:39);
  `netstat -m` 360,106 KB allocated to network, 94.7% in use, 2,034 requests
  denied, 41,615 mbufs of ancillary data; the `com.apple.content-filter` kernel
  control socket's receive queue full (523,880 of 524,288 bytes; Malwarebytes and
  GlobalProtect network extensions are installed). In milder phases only multicast
  to sockets already on 16571 stopped, 16 to 143 s after resolvers started and
  until they ended. After the restart: 21,559 KB, 2.8% in use, unicast 20 of 20.
  A 30 s failure needs this state; the race does not produce one.
- NOT established: what filled the network buffers (the content filter's queue is
  the lead, unproven), whether that state existed at 12:53 and 15:20 when the two
  suite runs failed (the drop counter was already 1,006,018 at 16:42), and whether
  this work's own experiments (long resolvers, two LSL test runs at once, a
  400-process CPU burner for 75 s) made it worse or contributed to the crash.
  Do not repeat them on a loaded machine.
- Ruled out: leaked outlets or sockets in the pytest process (liblsl leaks 3 file
  descriptors per outlet, a kqueue and a pipe pair, no sockets); port-range
  exhaustion (26 of 33 UDP ports busy with two resolvers, none leaked); order
  dependence (test_io_audit_final.py through test_lsl_streaming.py, 23 files in
  one process: all passed); a stalled pusher thread; a member leaving a multicast
  group silencing the others; an undrained full socket silencing the others.
- Corrected along the way: 14 to 15 s durations of the two-outlet tests before the
  restart were outlet destruction (7 s per outlet then; a pair took 5.8 to 14 s
  after the restart in a bare script), not lost discovery.

What changed:

- `tests/lsl_loopback.cfg` (`[lab] KnownPeers = {127.0.0.1}`): liblsl also sends
  each query straight to every port of its range, where each outlet has its own
  socket. `tests/conftest.py` points `LSLAPICFG` at it on macOS only, unless the
  caller set one. Before the restart, same outlets, same minutes: default
  resolver 13 of 13 misses, two KnownPeers resolvers 138 of 138 and 154 of 154
  found. After the restart: second outlet within 0.51 s in 12 of 12. It does
  nothing while unicast is dropped too (9 of 9 discovery tests failed then).
- New test `test_every_discovery_wave_queries_each_outlet_port_directly` (real
  socket on a port of the range, real resolve): 0 queries without the setting,
  7 with.
- Timeouts stay at 30 s (discovery) and 10 s (stall test), as in a11c09e9: with
  5 s restored, 9 discovery tests failed in a degraded phase. The comment that
  explained them was rewritten; "Discovery" in the test module's docstring has
  the measurements. `tests/test_io_audit_final.py` has one more 30 s; untouched.
- `[multicast] Interfaces = {127.0.0.1}` removes the race (20 of 20 waves) and
  was dropped: it did not prevent the multicast outage, and a test that counted
  multicast deliveries failed once in nine runs with it in place.

Library: `hypertools/io/lsl.py` is unchanged and uses liblsl's defaults, so a
user on macOS with two outlets on one machine (or one outlet next to another LSL
program) has the same race: a few query waves, covered by the default
`timeout=10`. The outage is the machine's. A sentence in the LSL tutorial about
`KnownPeers` for same-machine outlets is a candidate; not written (user-facing
advice, for Jeremy to decide). Linux and Windows CI runners do not get the
configuration (`sys.platform == 'darwin'`); the setting is plain unicast to
127.0.0.1 and should be harmless there, which was not tested.

Verified after the restart (load average 45 to 95): `.venv/bin/ruff check tests
hypertools/io` clean; `pytest tests/test_io_audit_lsl.py tests/test_lsl_streaming.py
tests/test_io_audit_final.py tests/test_notebook_install_gate.py`: 73 passed, 3
skipped in 114 s; the new test alone with the setting commented out: 1 failed
("assert 0 >= 6"). No full-suite run was made.

### Checks (after the machine crash of about 17:45; code as in checkpoint d9058dff)

The machine crashed while the timings were being written up; the
coordinator checkpointed the working tree as d9058dff. Every earlier test
result was lost with the scratch folder, so all runs were made again on
that code (load average 25 to 85):

- tests/test_review_round5_1_2_1.py: 50 passed (matplotlib 3.10); 50
  passed with matplotlib 3.11.2 (installed into a scratch target folder
  and put first on `PYTHONPATH`; the earlier `mpl311` venv was lost).
- tests/test_review_round1_1_2_1.py + round2: 311 passed, 1 skipped.
- round3 + round4 + tests/test_issue297_layout.py +
  tests/test_issue299_animation_arguments.py: 297 passed.
- The 71 other files named label/legend/hue/panel/caption/title/animat/
  docstring/gate/signature/changelog (not the LSL files): 2661 passed, 8
  skipped, 0 failed. (Before the crash one time bound failed once in
  tests/plot/test_forecast_animated_regrouped.py, the slow-forecast notice
  at load 180, and its file passed on a rerun, 15 passed.)
- `ruff check .`: clean. The whole suite was not run (the coordinator
  runs it).

Contact sheets and GIFs were made again after the crash:
`<scratchpad>/round5fix/sheet_{spin30_legend_title_caption,spin12_axis_labels,window2d,panels2x2}.png`
and `anim_*.gif` of the same names; all four sheets were looked at.

### Round 5, second pass: every label planned over the whole animation (2026-11-23 evening)

Coordinator feedback on the first table: labels overlapped EACH OTHER in
most sampled frames (the table kept labels still at the price of leaving
them stacked, "allowed up to 1.5 s"). The frame-by-frame hysteresis solver
(`avoid`, `resting`, `ahead`, loop rounds, `_settle_short_stays`) is gone;
`_arrange_labels` is again only the static rule (plus `strict`).

What `solve_label_tracks` does now (matplotlib_backend.py):

- Each label gets a PLAN for all frames: the cheapest path
  (`_cheapest_places`, exact dynamic programming over place and frames
  spent there) through 72 places (`_LABEL_TRACK_CANDIDATES`: 16 directions,
  reaches 20 to 130 pt). Costs: off canvas / on static content / on the
  cube's turning axis labels 1e7 a frame (with the static 3 pt clearance);
  on another label's DRAWN box 1e3 a frame (1 pt clearance); far out
  0.4 per usual distance per second; pointing into the cloud 1.0 per
  second; a change of place 3 + 0.5 per usual distance, none longer than
  240 pt. At least `_LABEL_REST_EASES` (3) glides at a place: that is what
  gives 0 turn-backs. Glide 0.35 s (5 frames at 15 fps).
- Labels in the given order; then a glide that crosses something bars that
  change on the frames it would cross (`_bar`) and the label is planned
  again (`_LABEL_TRACK_REPLANS` 4 at 30 labels, up to 12 with fewer, 1
  above 120); then up to 3 rounds in which labels still not clear, and the
  labels they lie on, are planned again among ALL others (1 round above 40
  labels, none above 80). The effort limits are in the docstring.
- A glide that would still cross static content or the canvas edge becomes
  a jump, made as early as the place left stops being clear.
- A loop is planned round the loop (three rounds, then once more pinned in
  the middle of the longest stay).
- The other animated axes of the figure that are planned already count as
  labels to keep off (`tracks['boxes']`): grid panels.

Tried and dropped: a shorter rest for covered labels (lazy barring of
reversals, or accepting a short-rest plan with no measured turn-back): next
to no gain on 30 labels for 2 to 3 times the time, and turn-backs when it
was used to get off static content.

Measured (matplotlib 3.10; full table in the scratch `pairs_per_frame.txt`),
overlapping pairs per frame, mean / max / frames with a pair of 60:

| case | still rule per frame | table of 66c33a32 | now |
|-|-|-|-|
| spin, 8 labels | 0 / 0 / 0 | 0.67 / 3 / 30 | 0 / 0 / 0 |
| spin, 12 labels, dressed | 0 / 0 / 0 | 0.55 / 4 / 20 | 0 / 0 / 0 |
| spin, 30 labels, dressed | 5.58 / 10 / 60 | 10.88 / 20 / 60 | 0.75 / 3 / 30 |
| window, 3-D | 0.05 / 2 / 2 | 0.68 / 3 / 28 | 0 / 0 / 0 |
| 2 x 2 panels, 12 each | 0 / 0 / 0 | 2.88 / 7 / 55 | 0.07 / 1 / 4 |

Turn-backs now 0 in all five (still rule: 29, 46, 164, 59, 251). With
matplotlib 3.11 (wider text) 30 labels give 1.5 / 4 / 45 and the grid
0.2 / 2 / 11; 8 and 12 labels stay at 0.

plot() CPU seconds, 200 points, 3-D spin, 60 frames, labels 0/20/60/200:
0.01 / 0.41 / 1.46 / 2.42 (table of 66c33a32, same run: 0.01 / 0.53 / 2.29
/ 5.33; placed once, a11c09e9: 0.008 / 0.054 / 0.138 / 0.481). The test
spins cost more than the benchmark at equal count because their labels are
crowded: 30 labels 1.8 to 2.2 s (1.56 before), the grid 2.7 s (2.6).

Not achieved: with 30 labels on a default cube a pair still overlaps on
half the frames (some at rest: no place stays clear for a second in a cube
that turns in four), and leader lines are long (mean offset about 118 pt
against 77 for the still rule) and many cross the cube. The outward
preference only breaks ties; what keeps a label still for a second is a far
place.

Tests (tests/test_review_round5_1_2_1.py, now 58): added
`test_no_two_labels_overlap_on_any_frame` (5 cases, all labels of the
figure together), `test_a_grids_labels_overlap_on_few_frames`,
`test_thirty_labels_overlap_less_than_stills_of_each_frame`; the
turn-back test runs on all 7 animations; `PASSING` is now the two cases in
which labels do change place (30 labels, grid); the 30-label share against
the usual place is 0.25 (was 0.6).

### Round 5, third pass: four defects from a source review of the planner (2026-11-23 night)

Head 185202a1 passed the full suite and was pushed by the coordinator. A
source-only review reported four defects; all four REPRODUCED with real
code (scratch `round5fix3/repro.py`) before anything was changed, and the
8 tests written for them fail against 185202a1.

| item | reproduced | fix |
|-|-|-|
| 1 memory and time | 400 labels x 600 frames: 13.9 MB table (58 B a label and frame); 60 labels x 600 frames: 94.7 CPU s | above 300 labels (now counted in total, not per frame) no frame is run and the table has one row (11 kB); otherwise at most `_LABEL_TRACK_BUDGET` = 3600 labels x frames are planned (evenly spaced frames, at least 40, `tracks['frames']`), frames between are interpolated in `_apply_label_tracks`; table float32/int16, 28 B a label and planned frame |
| 2 leak | closed animated figures never freed (`_LABEL_STATES` value held the frame function); also the last static labeled figure (module global `labels_and_points`) | state holds the frame function weakly (`weakref.ref` / `WeakMethod`); the animation owns it (`line_ani._hyp_label_frames`); the global now holds weak proxies (tests/test_plotly_labels.py and test_plot_save_audit_fixes.py read it, so it stays) |
| 3 pickle | `pickle.loads(pickle.dumps(anim.figure)).canvas.draw()` and `copy.deepcopy` raised TypeError | `_label_planned`: a planned label with no plan stays where the frame had it (not arranged, not sent home) |
| 4 stale plan | `fig.text` on a label, `set_fontsize`, `set_text`: no re-plan | signature also has every label's size and pad and the obstacle boxes (0.27 ms a frame); re-plan once; a figure changing every frame is planned at the first two changes and once when it stops |

Planning time, 60 labels (all 60 points labeled, crowded), CPU s: 60 frames
8.5, 600 frames 10.2 (was 94.7), 3000 frames 10.3; 12 labels x 3000 frames
8.6. Edge inputs (no labels, one label, None entries, one and two frames,
NaN point, panels with unequal counts, two saves) did not raise before
either; they now have tests. Known limit: with planned frames far apart (a
long animation with many labels) a glide lasts at least 5 planned frames,
so it is slower than 0.35 s, and labels are only checked on planned frames.
