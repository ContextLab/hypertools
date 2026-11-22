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
