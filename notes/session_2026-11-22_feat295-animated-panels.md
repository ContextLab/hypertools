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
