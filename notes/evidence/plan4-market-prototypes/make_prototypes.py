"""Market prototypes v2 -- both compositions, corrected against measurement.

Two things measured on v1 and fixed here:

* `hyp.plot` draws in a NORMALIZED unit box (xlim == [-1.1, 1.1] whatever
  the data) and removes ticks by design, so no number can be read off an
  axis. But `xlabel=`/`ylabel=` are native and DO render -- so the axes can
  say what they are, just not what they equal. Used natively here.
* each `hyp.plot` call normalizes ITS OWN inputs, so v1's small multiples
  put each panel on a private scale and the shared market curve came out a
  different shape in every one. Fixed by passing the WHOLE frame to every
  panel and varying only the hue: identical input, identical normalization,
  genuinely comparable panels.

The panel highlight is itself the hierarchy's arithmetic -- `hue_mode=
'mixture'` with a 7th near-white palette entry. The focused sector takes
its primary, the rest take white, and the market mean falls out as the mean
of the seven weight rows without anything computing it.
"""
import matplotlib
matplotlib.use('Agg')          # headless: this writes PNGs, it never shows one
import pathlib
import sys

# run from anywhere: the repo root is four levels up from this file
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
# E402: both imports must follow the sys.path insert above, which is what
# lets this run from outside the repo
import hypertools as hyp                                        # noqa: E402
from scripts.market_representation_study import (                 # noqa: E402
    SECTORS, fetch_closes, synthetic_closes, monthly_levels, CUM_WINDOW,
    DD_WINDOW)

# written beside this script, which is where the committed PNGs live
OUT = f'{pathlib.Path(__file__).resolve().parent}/'
WINDOW = 30
SECTOR_COLORS = ['#c1272d', '#f2a900', '#1b7f4f', '#2d5fa8', '#7d3c98', '#00808a']
PALE = '#dcdcdc'
NAMES = list(SECTORS)
XLAB = f'cumulative {CUM_WINDOW}-month return  (right = higher)'
YLAB = f'drawdown below the {DD_WINDOW}-month peak  (up = nearer the peak)'

closes = fetch_closes(SECTORS)
if closes is None:
    closes = synthetic_closes(SECTORS)
levels = monthly_levels(closes)

columns = {}
for sector in NAMES:
    lvl = levels[sector].mean(axis=1)
    columns[('Market', sector, 'cumulative return')] = lvl - lvl.shift(CUM_WINDOW)
    columns[('Market', sector, 'drawdown')] = (
        lvl - lvl.rolling(DD_WINDOW, min_periods=1).max())
frame = pd.DataFrame(columns).dropna()
frame.columns = pd.MultiIndex.from_tuples(
    frame.columns, names=['Market', 'Sector', 'Measure'])
frame = frame.iloc[-WINDOW:].apply(np.expm1)
dates, n = frame.index, len(frame)
span = f'{dates[0]:%b %Y} - {dates[-1]:%b %Y}'
print(f'{n} months, {span}')


def weights(focus=None):
    """One weight row per observation per leaf. `focus=None` colours every
    sector; an index pales all but that one."""
    rows = []
    for leaf in range(6):
        column = leaf if (focus is None or leaf == focus) else 6
        rows.append(np.tile(np.eye(7)[column], (n, 1)))
    return rows


def draw(ax, focus=None, **kw):
    hyp.plot(frame, '-', hue=weights(focus), palette=SECTOR_COLORS + [PALE],
             hue_mode='mixture', reduce=None, ndims=2, normalize=None,
             colorbar=False, ax=ax, show=False, **kw)
    return sorted([c for c in ax.collections
                   if getattr(c, '_hyp_trace_index', None) is not None],
                  key=lambda c: c._hyp_trace_index)


def path_of(coll):
    segs = coll.get_segments()
    return np.vstack([segs[0]] + [s[-1:] for s in segs[1:]])


def mark(ax, pts, color, lw=2.0):
    """Where it started, and which way it is going."""
    ax.plot(*pts[0], 'o', color=color, ms=5.5, mfc='white', mew=1.7, zorder=6)
    ax.annotate('', xy=pts[-1], xytext=pts[-4],
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                mutation_scale=16), zorder=6)


# ---------------- A: one fixed 2-D panel ----------------------------------
figa, axa = plt.subplots(figsize=(9.6, 7.2))
colls = draw(axa, xlabel=XLAB, ylabel=YLAB)
for i, name in enumerate(NAMES):
    pts = path_of(colls[i])
    mark(axa, pts, SECTOR_COLORS[i])
    axa.annotate(name, xy=pts[-1], xytext=(10, 3), textcoords='offset points',
                 color=SECTOR_COLORS[i], fontsize=9.5, fontweight='bold')
market = path_of(colls[-1])
mark(axa, market, '#2b2b2b', lw=3.0)
axa.annotate('MARKET', xy=market[-1], xytext=(10, -13),
             textcoords='offset points', color='#2b2b2b', fontsize=10.5,
             fontweight='bold')
axa.set_title(f'Six sectors and the market mean, {span}\n'
              'circle = start of the window, arrow = most recent month',
              fontsize=12.5, fontweight='bold', loc='left')
figa.tight_layout()
figa.savefig(f'{OUT}PROTO_A2_single_panel.png', dpi=115)
print('saved A2')

# ---------------- B: small multiples, ONE shared normalization ------------
figb, axes = plt.subplots(2, 3, figsize=(13.2, 8.4))
for i, (name, axb) in enumerate(zip(NAMES, axes.ravel())):
    panel = draw(axb, focus=i,
                 xlabel=(XLAB if i >= 3 else None),
                 ylabel=(YLAB if i % 3 == 0 else None))
    mark(axb, path_of(panel[i]), SECTOR_COLORS[i])
    mark(axb, path_of(panel[-1]), '#2b2b2b', lw=2.4)
    axb.set_title(name, fontsize=11.5, fontweight='bold',
                  color=SECTOR_COLORS[i], loc='left')
figb.suptitle(f'One sector at a time, against the market mean (dark) and its '
              f'five peers (pale) -- {span}', fontsize=13, fontweight='bold')
figb.tight_layout()
figb.savefig(f'{OUT}PROTO_B2_small_multiples.png', dpi=115)
print('saved B2')

# the claim that makes B2 comparable at all, asserted rather than hoped
lims = [(round(float(a.get_xlim()[0]), 6), round(float(a.get_xlim()[1]), 6),
         round(float(a.get_ylim()[0]), 6), round(float(a.get_ylim()[1]), 6))
        for a in axes.ravel()]
print('every panel shares one normalization:', len(set(lims)) == 1, lims[0])
