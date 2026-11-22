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
