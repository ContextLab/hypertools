# HyperTools 1.1.1

A patch release with no API changes. It fixes noise and side effects met when `hyp.plot` draws into a figure the caller built (`ax=`), found by executing notebooks with `jupyter nbconvert` under matplotlib 3.11, and one misleading error message from the Yahoo Finance loader.

Every change is listed in the [changelog](https://github.com/ContextLab/hypertools/blob/v1.1.1/CHANGELOG.md); these notes summarize it.

## Install

```bash
pip install --upgrade hypertools
```

Python 3.10 through 3.13. Extras are unchanged from 1.1.0.

## Fixed

- **The backend warning is issued once, and only when it applies ([#290](https://github.com/ContextLab/hypertools/issues/290)).** When hypertools cannot set up an interactive matplotlib backend at import (for example in a notebook kernel without `ipympl`), `hyp.plot` repeated the warning on every call, including static plots and calls that show no live figure. It is now issued only by a call that displays an animated or interactive plot that matplotlib renders, and once per session per message.
- **`hyp.plot(..., ax=ax)` leaves the layout of the caller's figure alone ([#291](https://github.com/ContextLab/hypertools/issues/291)).** Static plots ended with `plt.tight_layout()` even when the caller passed its own axes, which moved every axes of the figure, discarded `subplots_adjust` margins, replaced a `layout='constrained'` engine, and acted on pyplot's current figure rather than the one that owns `ax`. A figure the caller built is now left as the caller arranged it. Figures hypertools makes itself (its own figure, a `panels=` grid, a `hyp.subplots` grid) are laid out as before.
- **A legend or title added to the caller's axes is fitted automatically, inside that axes' own room.** Through 1.1.0 a `legend=` beside a caller's axes was fitted by widening the caller's figure, which stretched its other axes and switched off its layout engine. Now, when the figure is drawn, only the axes passed as `ax=` is narrowed just far enough that its legend ends before the canvas edge or the next panel, and lowered just far enough that its title stays on the canvas. No call is needed, and the fit follows a later `fig.tight_layout()`, `fig.subplots_adjust(...)`, resize or save. A legend with no room beside the axes is drawn inside it; one too large for its panel either way is left in place with a `UserWarning` naming the remedies.
- **No `findfont` log lines for ordinary text under matplotlib 3.11 ([#292](https://github.com/ContextLab/hypertools/issues/292)).** matplotlib 3.11 logs a line each time a family in the font list has no face at a text's exact weight; one of the fallback families hypertools listed has no regular face on macOS, so every figure logged the line several times. Fallback families with no regular-weight face are now left out of the list.
- **A Yahoo Finance rate limit is reported as one.** `hyp.load('yahoo:<TICKER>')` reported an HTTP 429 from Yahoo as a JSON decoding failure. The error now says that Yahoo is rate-limiting the address and to try again later.
- **A seeded UMAP no longer warns about `n_jobs`.** A seed passed in the `reduce` dict (`reduce={'model': 'UMAP', 'kwargs': {'random_state': 0}}`) drew umap-learn's "n_jobs value 1 overridden to 1" warning; it is now quiet, like the top-level `random_state=`.

## Documentation

- Two gallery examples (`plot_colorbar.py`, `plot_digits.py`) are now seeded, so their figures are the same on every docs build, and `plot_shapes_zoo.py` no longer shows stray axis labels beside one panel.

## Upgrading from 1.1.0

- **If you relied on the automatic `tight_layout`** of a figure made with `plt.subplots`, call `fig.tight_layout()` after plotting, create the figure with `layout='constrained'`, or build the grid with `hyp.subplots`.
- On an axes whose legend or title was fitted, `ax.get_position()` reports the position of the last draw (the fit is an axes locator, recomputed on each draw).

## Known limitations

- The automatic fit does not detect an inset or other axes placed by hand where the legend goes, or a legend over the tick labels of its own axes' twin.
- Bold or other non-regular text can still log `findfont` lines under matplotlib 3.11 when an installed fallback family ships a regular face only; see "Documented limitations" in the changelog.

## Links

- Documentation: https://hypertools.readthedocs.io/en/stable/
- Changelog: https://github.com/ContextLab/hypertools/blob/v1.1.1/CHANGELOG.md
- PyPI: https://pypi.org/project/hypertools/1.1.1/
