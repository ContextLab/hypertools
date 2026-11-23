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

