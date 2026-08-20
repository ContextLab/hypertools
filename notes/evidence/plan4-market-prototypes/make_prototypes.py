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


def mark(ax, pts, color, lw=2.0, ms=5.5, mew=1.7, head=16):
    """Where it started, and which way it is going.

    Sized by argument because the annotation that reads well on a 1500 px
    prototype is a blob on a 736 px documentation page -- which is the
    whole reason C is rendered at the documentation width.
    """
    ax.plot(*pts[0], 'o', color=color, ms=ms, mfc='white', mew=mew, zorder=6)
    ax.annotate('', xy=pts[-1], xytext=pts[-4],
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                mutation_scale=head), zorder=6)


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


# ---------------- C: B2, with the pooled per-measure display gain --------
# MEASURED on the library, and the reason B2's traces sat in a thin band:
# `hyp.plot` centres each dimension separately but applies ONE gain to all
# of them, so a dimension with 1/100th the spread is drawn at 1/100th the
# height (probe: x in [0, 100], y in [0, 1] -> drawn y range +-0.01).
# MEASURED here: the two measures' half-ranges over the complete frame are
# 0.456 (return) and 0.129 (drawdown), a factor of 3.5 -- so drawdown was
# being drawn into a band under a third of the height it deserved, and each
# individual sector covers only part of even that.
#
# So each MEASURE gets one display gain, computed ONCE over the complete
# frame -- every sector, every row -- and applied to every panel. It is a
# display gain, not a per-panel rescale: within-panel and across-panel
# comparisons are untouched, and the only thing it sets is how tall one
# unit of drawdown looks next to one unit of return, which the raw units
# were deciding by accident.
half_range = {}
for measure in frame.columns.get_level_values('Measure').unique():
    block = frame.xs(measure, axis=1, level='Measure').to_numpy()
    half_range[measure] = (block.max() - block.min()) / 2.0
scaled = frame / [half_range[c[2]] for c in frame.columns]
print('per-measure half-ranges:',
      {k: round(v, 4) for k, v in half_range.items()})


#: C pales the peers further than B2 did. At 736 px the #dcdcdc peers were
#: not "context" but clutter competing with the sector the panel is about.
#:
#: This value also decides whether the MARKET MEAN can be seen, and the
#: three rendered candidates say it cannot -- measured parent luminance vs
#: peer luminance, from `parent colour` below:
#:
#:   FAINT      peer   parent   gap    what it looks like at 736 px
#:   #f4f4f4    0.957  0.863    0.094  peers gone; mean a faint tint
#:   #ededed    0.929  0.841    0.088  peers a ghost; mean lost among them
#:   #c4c4c4    0.769  0.707    0.062  peers compete with the focus; mean
#:                                     still not separable from the peers
#:
#: The parent is the MEAN OF ITS CHILDREN by construction, so it always
#: lands inside the peer greys, roughly a tenth of a luminance step from
#: them, whatever this value is. Twice the leaf linewidth (measured: 2.0 vs
#: 1.0) is not enough to recover a 0.1 gap at this size. That is a
#: composition decision, not a tuning problem -- see README.md.
FAINT = '#ededed'


def draw_scaled(ax, focus=None, **kw):
    hyp.plot(scaled, '-', hue=weights(focus), palette=SECTOR_COLORS + [FAINT],
             hue_mode='mixture', reduce=None, ndims=2, normalize=None,
             colorbar=False, ax=ax, show=False, **kw)
    return sorted([c for c in ax.collections
                   if getattr(c, '_hyp_trace_index', None) is not None],
                  key=lambda c: c._hyp_trace_index)


# 7.36in x 100dpi = 736px = furo's content column, which is the width that
# actually decides whether these panels are legible
figc, axesc = plt.subplots(2, 3, figsize=(7.36, 4.9), dpi=100)
for i, (name, axc) in enumerate(zip(NAMES, axesc.ravel())):
    panel = draw_scaled(axc, focus=i)
    mark(axc, path_of(panel[i]), SECTOR_COLORS[i], lw=1.4, ms=4.0, mew=1.2,
         head=10)
    # the hierarchy mean is a BLEND, not a dark line: one focused leaf and
    # five pale ones average to mostly pale, and the hierarchy's own
    # arithmetic cannot make it black. What it DOES give it is twice the
    # leaf linewidth (measured: 2.0 vs 1.0). Its endpoint gets a neutral
    # marker so it can be found; nothing redraws its path.
    mark(axc, path_of(panel[-1]), '#5f5f5f', lw=1.2, ms=4.0, mew=1.2,
         head=10)
    axc.set_title(name, fontsize=9.5, fontweight='bold',
                  color=SECTOR_COLORS[i], loc='left', pad=3)
figc.supxlabel(XLAB, fontsize=8.5)
figc.supylabel(YLAB, fontsize=8.5)
# the title claims only what the figure delivers: it does NOT say the mean
# is visible, because at this width it is findable only by its marker
figc.suptitle(f'One sector at a time, on one shared frame  ({span})',
              fontsize=10, fontweight='bold')
figc.tight_layout()
figc.savefig(f'{OUT}PROTO_C_small_multiples.png')
width_px, height_px = figc.get_size_inches() * figc.dpi
print(f'saved C at {width_px:.0f}x{height_px:.0f} px '
      f"(furo's content column is 736 px wide)")

# what the hierarchy's own arithmetic gives the parent, rather than what a
# caption might claim it gives: linewidth and blended colour, measured
parent, leaf = panel[-1], panel[0]
print('parent linewidth', np.atleast_1d(parent.get_linewidth())[0],
      'vs leaf', np.atleast_1d(leaf.get_linewidth())[0])
print('parent colour (first segment)', parent.get_colors()[0].round(3))
peer = panel[0]          # panel 6 focuses leaf 5, so leaf 0 is a peer


def luminance(rgba):
    return round(float(0.2126 * rgba[0] + 0.7152 * rgba[1]
                       + 0.0722 * rgba[2]), 3)


print(f'luminance: peer {luminance(peer.get_colors()[0])}  '
      f'parent {luminance(parent.get_colors()[0])}  '
      f'(the gap the market mean has to be found across)')

limsc = [(round(float(a.get_xlim()[0]), 6), round(float(a.get_xlim()[1]), 6),
          round(float(a.get_ylim()[0]), 6), round(float(a.get_ylim()[1]), 6))
         for a in axesc.ravel()]
print('C: every panel shares one normalization:', len(set(limsc)) == 1,
      limsc[0])
probe_fig, probe_ax = plt.subplots()
spans = [(np.ptp(path_of(c)[:, 0]), np.ptp(path_of(c)[:, 1]))
         for c in draw_scaled(probe_ax)]
plt.close(probe_fig)
print('C: drawn x/y spans per trace, out of the 2.0 the box allows:',
      [(round(a, 2), round(b, 2)) for a, b in spans])


# ---------------- D: explicit trace colours, dark hierarchy mean ---------
# WHY C COULD NOT WORK, stated as the theorem it is: with `hue_mode=
# 'mixture'` the parent's colour is the MEAN of its children's colours, and
# a mean lies in the convex hull of what it averages. So the parent can
# never be darker than the darkest leaf. Five pale peers therefore FORCE a
# pale mean -- measured gap 0.088, and no palette setting escapes it,
# because the constraint is arithmetic rather than aesthetic.
#
# Review round 12 relaxes the requirement accordingly: the hierarchy must
# be native in DISCOVERY (leaves from the column MultiIndex), GEOMETRY (the
# parent is the mean of its children) and STYLE (its heavier line), but the
# parent's COLOUR need not come out of hue-weight averaging.
#
# What the library supports today, measured:
#   * 3-level frame + `palette`  -> 6 leaves + 1 parent, ALL one colour
#     (leaves alpha 0.7 lw 1.0, parent alpha 1.0 lw 2.0)
#   * matrix `hue` + 'mixture'   -> arbitrary per-leaf colours, parent = the
#     mean of them
#   * there is NO spelling that gives per-leaf colours AND an independently
#     coloured parent -- that is the API gap this prototype documents
#
# So D draws the SAME complete frame twice: once for the leaves (matrix
# hue) and once for the parent (a dark single-colour palette), and hides
# the second call's leaves. Both calls receive identical input, so they
# normalize identically -- asserted below by comparing the two parent paths
# point for point. Nothing is drawn by hand; the dark mean is the library's
# own parent trace, in the library's own heavier hierarchy style.
MEAN_DARK = '#3f3f3f'


def draw_mean(ax):
    """The hierarchy's parent trace, drawn dark, with its leaves hidden."""
    before = set(map(id, ax.lines))
    # legend=False: this call exists only for its parent trace, and its
    # legend entry would otherwise steal a third of every panel's width
    hyp.plot(scaled, '-', palette=[MEAN_DARK], reduce=None, ndims=2,
             normalize=None, colorbar=False, legend=False, ax=ax, show=False)
    drawn = [ln for ln in ax.lines if id(ln) not in before]
    # the parent is the one the hierarchy styles as a parent: the heavier
    # line. Found by the style the LIBRARY assigned, not by position.
    parent = max(drawn, key=lambda ln: ln.get_linewidth())
    for line in drawn:
        if line is not parent:
            line.set_visible(False)
    parent.set_zorder(5)
    return parent


figd, axesd = plt.subplots(2, 3, figsize=(7.36, 4.9), dpi=100)
for i, (name, axd) in enumerate(zip(NAMES, axesd.ravel())):
    panel = draw_scaled(axd, focus=i)
    mean_line = draw_mean(axd)
    mark(axd, path_of(panel[i]), SECTOR_COLORS[i], lw=1.4, ms=4.0, mew=1.2,
         head=10)
    # the mean gets a start dot only: it is already the heaviest line in
    # the panel, and a second arrowhead competes with the sector's
    axd.plot(*np.array(mean_line.get_data()).T[0], 'o', color=MEAN_DARK,
             ms=4.0, mfc='white', mew=1.2, zorder=6)
    axd.set_title(name, fontsize=9.5, fontweight='bold',
                  color=SECTOR_COLORS[i], loc='left', pad=3)
figd.supxlabel(XLAB, fontsize=8.5)
figd.supylabel(YLAB, fontsize=8.5)
figd.suptitle(f'One sector at a time, against the market mean  ({span})',
              fontsize=10, fontweight='bold')
figd.tight_layout()
figd.savefig(f'{OUT}PROTO_D_small_multiples.png')
print('saved D at %.0fx%.0f px' % tuple(figd.get_size_inches() * figd.dpi))

# the claim that makes the overlay legitimate: both calls see the same
# frame, so the parent they each compute is the SAME curve
hue_parent = path_of(panel[-1])
dark_parent = np.array(mean_line.get_data()).T
print('D: the two calls agree on the parent to',
      f'{np.abs(hue_parent - dark_parent).max():.2e}')
print('D: mean linewidth', mean_line.get_linewidth(), 'vs leaf',
      np.atleast_1d(panel[0].get_linewidth())[0],
      '| mean luminance', luminance(matplotlib.colors.to_rgb(MEAN_DARK)),
      'vs peer', luminance(panel[0].get_colors()[0]))
limsd = [(round(float(a.get_xlim()[0]), 6), round(float(a.get_xlim()[1]), 6),
          round(float(a.get_ylim()[0]), 6), round(float(a.get_ylim()[1]), 6))
         for a in axesd.ravel()]
print('D: every panel shares one normalization:', len(set(limsd)) == 1,
      limsd[0])
