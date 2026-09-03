# -*- coding: utf-8 -*-
"""
==================================================
Six sectors, their stocks, and each sector's mean
==================================================

Twenty-four large-cap stocks, four per sector, each drawn as a path through
two measures every stock shares -- cumulative 12-month return (x) and
drawdown below the 24-month peak (y) -- over the most recent five years at
six-month strokes. The whole figure is **one** ``hyp.plot`` call on a
DataFrame whose columns carry a ``(Sector, Ticker, Measure)``
``MultiIndex``. The innermost level is the feature axis and every level
above it groups, so the library finds the four stocks of each sector in the
index, draws each as a faint leaf, and draws the sector's **mean** as a
heavier line on top. That mean is computed by the hierarchy, not by this
file, and nothing here computes or draws a trajectory by hand.

**Six panels, one call.** An animated ``hyp.plot`` owns its figure and
normalizes everything it is given into one unit box, so six sectors cannot
be six axes of one animation -- and they do not need to be. Each sector's
block is translated into its own region of one shared coordinate box
*before* the call, and the six panels become six column groups of a single
frame: one call, one animation, one ``.save()``, and one normalization
shared by construction rather than by assertion. The boxes, the sector
labels, the darker mean and the dot riding its head are annotation of
artists the library drew.

**How to read it.** Every panel has the same units and the same gain, so
the length, direction and shape of a path are comparable across panels,
and each panel is read against its own rectangle. Position *between*
panels is not comparable: the offsets are layout, not market data, and the
caption says so. The bold line is that **sector's** mean of the four stocks
beside it. There is deliberately no market-wide mean: the offsets are
applied before the hierarchy computes its means, so a top-level parent
would average six layout translations and mean nothing. Sector means stay
exact because all four stocks of a sector receive the same translation.

**Data & graceful degradation.** Ten years of ADJUSTED daily closes are
pulled from Yahoo Finance's chart endpoint and cached on disk; adjusted, so
a split does not read as a -50% day. Month-end levels are a decimation, so
no future observation reaches back into a bar. If the network is
unavailable the example falls back to a seeded synthetic basket with the
same sector structure, so it always renders -- the technique (hierarchy ->
tiled panels -> one animated call) is identical either way.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import json
import os
import tempfile
import urllib.request
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

import hypertools as hyp

# This file keeps its 1.0 name so the published docs URL survives. Nothing in
# it forecasts anything; see the module docstring for what it shows instead.
CACHE = os.path.join(tempfile.gettempdir(), 'hypertools_gallery_cache')
RANGE = '10y'
# six sectors x four EQUALLY WEIGHTED tickers each
SECTORS = {
    'Technology': ['AAPL', 'MSFT', 'ORCL', 'IBM'],
    'Financials': ['JPM', 'BAC', 'GS', 'AXP'],
    'Healthcare': ['JNJ', 'PFE', 'MRK', 'ABT'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB'],
    'Consumer': ['KO', 'PG', 'WMT', 'MCD'],
    'Industrials': ['BA', 'CAT', 'GE', 'HON'],
}
SECTOR_COLORS = ['#c1272d', '#f2a900', '#1b7f4f', '#2d5fa8', '#7d3c98',
                 '#00808a']
CUM_WINDOW, DD_WINDOW = 12, 24          # months
MONTHS, STRIDE = 60, 6                  # five years, one stroke per half-year
PANEL_COLS, PANEL_STEP = 3, 2.6         # the grid -- layout only, no meaning
DURATION, FPS = 6, 15                   # seconds, frames per second
CAPTION = ('same cumulative-return (x) and drawdown (y) scale in every '
           'panel; panel positions are layout only')


class Market(NamedTuple):
    stocks: pd.DataFrame        # (Sector, Ticker, Measure) -- what is PLOTTED
    source: str                 # which path produced it


# --- the data half: the ONLY code here that reaches the network -------------
def fetch_closes(sectors=SECTORS):
    """Adjusted daily closes per (sector, ticker), or ``None`` if anything
    (network, parsing) goes wrong."""
    # outside the try, so it raises instead of being caught and quietly
    # downgraded: a test that sets HYPERTOOLS_OFFLINE is asserting that no
    # fetch happened, and a swallowed exception would hide one
    if os.environ.get('HYPERTOOLS_OFFLINE'):
        raise RuntimeError('HYPERTOOLS_OFFLINE is set: refusing to fetch')
    os.makedirs(CACHE, exist_ok=True)
    try:
        series = {}
        for sector, tickers in sectors.items():
            for ticker in tickers:
                dest = os.path.join(CACHE, f'yahoo_adj_{ticker}_{RANGE}.json')
                if not os.path.exists(dest):
                    url = ('https://query1.finance.yahoo.com/v8/finance/chart/'
                           f'{ticker}?range={RANGE}&interval=1d')
                    req = urllib.request.Request(
                        url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        payload = resp.read()
                    with open(dest + '.part', 'wb') as f:
                        f.write(payload)
                    # rename, so an interrupted download can never leave a
                    # truncated cache file that every later run would trust
                    os.replace(dest + '.part', dest)
                with open(dest) as f:
                    result = json.load(f)['chart']['result'][0]
                stamps = pd.to_datetime(result['timestamp'], unit='s')
                series[(sector, ticker)] = pd.Series(
                    result['indicators']['adjclose'][0]['adjclose'],
                    index=stamps.normalize()).astype(float)
        # no ffill: a stale ticker padded flat would contribute exactly zero
        # return while its peers moved. Dropping the row makes it visible.
        return pd.DataFrame(series).sort_index().dropna()
    except Exception:
        return None


def synthetic_closes(sectors=SECTORS, days=2500, seed=0):
    """The same sector structure, seeded, so the figure renders offline."""
    rng = np.random.default_rng(seed)
    index = pd.date_range('2016-08-15', periods=days, freq='B')
    columns = [(sector, t) for sector, ts in sectors.items() for t in ts]
    drift = rng.normal(0.0003, 0.0002, size=(1, len(columns)))
    steps = rng.normal(0, 0.013, size=(days, len(columns))) + drift
    return pd.DataFrame(100.0 * np.exp(steps.cumsum(axis=0)), index=index,
                        columns=pd.MultiIndex.from_tuples(columns))


def stock_paths(closes):
    """``(Sector, Ticker, Measure)``: each stock's path through the two
    measures. Month-end LOG levels (a month still in progress is dropped),
    both measures backward-looking, the leading incomplete window dropped
    rather than filled, and one stroke per STRIDE months counted back from
    the most recent month."""
    levels = np.log(closes.resample('ME').last().loc[:closes.index[-1]])
    columns = {}
    for sector, ticker in closes.columns:
        level = levels[(sector, ticker)]
        columns[(sector, ticker, 'cumulative return')] = (
            level - level.shift(CUM_WINDOW))
        columns[(sector, ticker, 'drawdown')] = (
            level - level.rolling(DD_WINDOW, min_periods=1).max())
    paths = pd.DataFrame(columns).dropna()
    paths.columns = pd.MultiIndex.from_tuples(
        paths.columns, names=['Sector', 'Ticker', 'Measure'])
    return paths.iloc[-MONTHS:].apply(np.expm1).iloc[::-STRIDE].iloc[::-1]


def load_market(sectors=SECTORS):
    """The ONLY function here that may touch the network."""
    try:
        closes, source = fetch_closes(sectors), 'Yahoo Finance adjusted closes'
    except RuntimeError:
        closes = None
    if closes is None:
        closes, source = synthetic_closes(sectors), 'synthetic basket (offline)'
    return Market(stock_paths(closes), source)


def fixture_data():
    """The same payload from the seeded synthetic basket. No network."""
    return Market(stock_paths(synthetic_closes()), 'synthetic basket (fixture)')


# --- the figure half: no network, deterministic given its input -------------
def cell_offset(index):
    """Where panel `index` sits in the grid, in the tiled frame's units."""
    return (index % PANEL_COLS) * PANEL_STEP, -(index // PANEL_COLS) * PANEL_STEP


def tile(paths):
    """Lay the panels out IN THE DATA.

    One display gain per measure, pooled once over the complete frame (so no
    panel can be rescaled on its own), then each sector's block translated
    into its own cell. Returns the tiled frame and the extent of one cell.
    """
    measures = paths.columns.get_level_values('Measure')
    half = {m: np.ptp(paths.xs(m, axis=1, level='Measure').to_numpy()) / 2
            for m in measures.unique()}
    tiled = paths / [half[m] for m in measures]
    cell = {m: (tiled.xs(m, axis=1, level='Measure').min().min(),
                tiled.xs(m, axis=1, level='Measure').max().max())
            for m in half}
    sectors = tiled.columns.get_level_values('Sector')
    for index, sector in enumerate(sectors.unique()):
        dx, dy = cell_offset(index)
        tiled.loc[:, (sectors == sector) & (measures == 'cumulative return')] += dx
        tiled.loc[:, (sectors == sector) & (measures == 'drawdown')] += dy
    return tiled, cell


def drawn_affine(tiled):
    """Recover the affine ``hyp.plot`` applies (its docstring: mean-centred
    and rescaled into ``[-1, 1]``, one shared transform), so annotations can
    be placed in data terms. It takes no part in drawing the data. Read off a
    STATIC draw: an animated frame stops a hair short of the last vertex."""
    probe, ax = plt.subplots()
    hyp.plot(tiled, '-', reduce=None, ndims=2, normalize=None, colorbar=False,
             ax=ax, show=False)
    lines = [line for line in ax.lines if len(line.get_xdata()) > 2]
    plt.close(probe)
    affine = {}
    for measure, drawn in (('cumulative return', [ln.get_xdata() for ln in lines]),
                           ('drawdown', [ln.get_ydata() for ln in lines])):
        data = tiled.xs(measure, axis=1, level='Measure').to_numpy()
        lo, hi = min(d.min() for d in drawn), max(d.max() for d in drawn)
        gain = (hi - lo) / (data.max() - data.min())
        affine[measure] = (gain, lo - gain * data.min())
    return affine


def draw_panel_boxes(ax, sectors, cell, affine, pad=0.06):
    """One identical rectangle per panel, from the OFFSETS that built the
    grid (not from where each sector's paths happen to sit), plus its label.
    Annotation only: no trajectory is touched."""
    (gx, sx), (gy, sy) = affine['cumulative return'], affine['drawdown']
    boxes = {}
    for index, sector in enumerate(sectors):
        dx, dy = cell_offset(index)
        x0 = gx * (dx + cell['cumulative return'][0]) + sx - pad
        x1 = gx * (dx + cell['cumulative return'][1]) + sx + pad
        y0 = gy * (dy + cell['drawdown'][0]) + sy - pad
        y1 = gy * (dy + cell['drawdown'][1]) + sy + pad
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   ec='#cccccc', lw=0.8, zorder=0))
        ax.annotate(sector, xy=(x0, y1 + 0.012), color=SECTOR_COLORS[index],
                    fontsize=8.5, fontweight='bold', va='bottom')
        boxes[sector] = (x0, x1, y0, y1)
    return boxes


def construct_artifact(data):
    """`data.stocks` in, the animation out. No network, no module globals.
    Returns the HyperAnimation wrapper, never the unpacked pair."""
    tiled, cell = tile(data.stocks)
    sectors = list(tiled.columns.get_level_values('Sector').unique())
    span = f'{tiled.index[0]:%b %Y} - {tiled.index[-1]:%b %Y}'
    # THE call. The column MultiIndex is the whole layout instruction: 24
    # leaves, six automatic sector means drawn heavier, one colour per
    # top-level group. The trail window equals the clip, so every path stays
    # in view and the last frame is fully revealed (the default 2 s window
    # leaves 16-54% of a path on screen at the end -- measured). The
    # library's own frame box spans the full unit square, so once the view
    # is cropped to the panel grid its edges would cross the figure;
    # `frame_kwargs` is the documented knob for that.
    anim = hyp.plot(tiled, '-', palette=SECTOR_COLORS, reduce=None, ndims=2,
                    normalize=None, animate='parallel', duration=DURATION,
                    tail_duration=DURATION, frame_rate=FPS, colorbar=False,
                    size=(7.36, 4.9), frame_kwargs={'visible': False},
                    show=False,
                    title=f"Six sectors, their stocks, and each sector's "
                          f'mean ({span})')
    ax = anim.figure.axes[0]
    # the panel labels already name the sectors. `legend=` stays at its
    # default so the parent traces keep their group labels, which is how
    # they are told apart from the leaves below.
    ax.get_legend().remove()
    parents = {ln.get_label(): ln for ln in ax.lines if ln.get_label() in sectors}
    leaves = [ln for ln in ax.lines if ln.get_label() not in sectors]
    per_panel = len(leaves) // len(sectors)
    boxes = draw_panel_boxes(ax, sectors, cell, drawn_affine(tiled))

    # restyle what the library drew: a darker, heavier mean than the
    # hierarchy's own 2x width gives, paler leaves, and a dot at each head
    heads = {}
    for index, sector in enumerate(sectors):
        dark = tuple(0.62 * channel for channel in to_rgb(SECTOR_COLORS[index]))
        parents[sector].set(color=dark, linewidth=2.6, zorder=4)
        for leaf in leaves[index * per_panel:(index + 1) * per_panel]:
            leaf.set(alpha=0.55, linewidth=0.9)
        heads[sector] = ax.plot([], [], 'o', color=dark, ms=4.0, mfc='white',
                                mew=1.3, zorder=7)[0]

    def move_head_dots(context):
        """`on_frame` is for decoration that must CHANGE with the frame."""
        for sector, parent in parents.items():
            heads[sector].set_data(parent.get_xdata()[-1:],
                                   parent.get_ydata()[-1:])

    anim.on_frame(move_head_dots)
    # crop to the grid; limits and spines persist across frames and saves
    ax.set_xlim(min(b[0] for b in boxes.values()) - 0.02,
                max(b[1] for b in boxes.values()) + 0.02)
    ax.set_ylim(min(b[2] for b in boxes.values()) - 0.02,
                max(b[3] for b in boxes.values()) + 0.075)
    for spine in ax.spines.values():
        spine.set_visible(False)
    anim.figure.text(0.5, 0.025, CAPTION, ha='center', fontsize=8,
                     color='#555555')
    return anim


if __name__ == '__main__':
    market = load_market()
    print(f'market data: {market.stocks.shape[0]} strokes x '
          f'{market.stocks.columns.get_level_values("Ticker").nunique()} '
          f'stocks in {len(SECTORS)} sectors ({market.source})')
    anim = construct_artifact(market)
    fig = anim.figure
