# HyperTools 1.2.0

A feature release: animated panel grids, a per-frame caption, and panel grids whose colors, legend and colorbar are shared across the figure. It also carries the fixes for noise and side effects met when `hyp.plot` draws into a figure the caller built.

Every change is listed in the [changelog](https://github.com/ContextLab/hypertools/blob/v1.2.0/CHANGELOG.md); these notes summarize it.

## Install

```bash
pip install --upgrade hypertools
```

Python 3.10 through 3.13. Extras are unchanged from 1.1.0.

## Added

- **Animated panel grids ([#295](https://github.com/ContextLab/hypertools/issues/295)).** `hyp.plot([a, b, c, d], panels=(1, 4), animate=True, save_path='grid.mp4')` gives one animation of the whole grid: every panel runs on the same clock, each panel is animated as the single-axes call for its data is, and one `HyperAnimation` is returned for the figure. The combination used to raise `ValueError`. Animated grids are drawn with matplotlib.
- **`caption=`.** One line of text under a plot or a grid, given as a string, a `'{index}'` pattern or a function of the frame, for example `caption=lambda ctx: f'timepoint {ctx.frame + 1} of {ctx.n_frames}'`. It is updated on every frame of an animation, wrapped to the figure's width, and drawn once on a static plot.
- **Fast-start video files.** Saved `.mp4`, `.m4v` and `.mov` files start playing in a browser before the whole file has downloaded.

## Changed

- **Panel grids share their colors, one legend and one colorbar.** In a `panels=` grid each dataset is drawn in the color it has in the single-axes plot of all the datasets, `hue=` uses one scale (or one label-to-color mapping) for the whole figure, and the grid shows one legend and one colorbar beside the panels instead of one per panel. Legends stay per panel only where the colors mean different things in each panel (for example clusters fit separately per panel).
- **A panel grid's default figure is sized by the grid:** 3.2 in per panel each way, at least 6.4 in wide, instead of a fixed 6.4 x 4.8 in. `size=` is honored as before, and `hyp.subplots()` keeps its default.

## Fixed

- **The backend warning is issued once, and only when it applies ([#290](https://github.com/ContextLab/hypertools/issues/290)).**
- **`hyp.plot(..., ax=ax)` leaves the layout of the caller's figure alone ([#291](https://github.com/ContextLab/hypertools/issues/291)).** A legend or title added to the caller's axes is fitted inside that axes' own room, automatically, on every draw.
- **No `findfont` log lines for ordinary text under matplotlib 3.11 ([#292](https://github.com/ContextLab/hypertools/issues/292)).**
- **A Yahoo Finance rate limit is reported as one.**
- **A seeded UMAP no longer warns about `n_jobs`.**
- **A `legend=` list names the categories of a categorical `hue=` on line plots too,** as it already did for markers.

## Upgrading from 1.1.0

- **Static `panels=` figures look different:** per-dataset colors, one shared legend and colorbar, and a figure sized by the grid. Pass `size=` to set the figure size yourself.
- **If you relied on the automatic `tight_layout`** of a figure made with `plt.subplots` and drawn into with `ax=`, call `fig.tight_layout()` after plotting, create the figure with `layout='constrained'`, or build the grid with `hyp.subplots`.
- On a single line plot with a categorical `hue=`, `legend=[...]` now takes one name per category (it took one per drawn run).

## Known limitations

- Animated panel grids and `caption=` are matplotlib-only; with `backend='plotly'` they raise `NotImplementedError`. Static plotly grids are supported.
- The automatic legend fit on a caller's axes does not detect an inset or other axes placed by hand where the legend goes.

## Links

- Documentation: https://hypertools.readthedocs.io/en/stable/
- Changelog: https://github.com/ContextLab/hypertools/blob/v1.2.0/CHANGELOG.md
- PyPI: https://pypi.org/project/hypertools/1.2.0/
