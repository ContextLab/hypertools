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
frame_all = frame
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


# ---------------- D2: D, with the mean restrained -------------------------
# Review round 12, finding 3: D's mean is unmistakable but reads as a dense
# black knot competing with the sector the panel is about. Four changes,
# all of them display-side:
#   * a shorter window -- 30 months of a wandering mean is knot, not path
#   * the mean drawn at 1.6 rather than the hierarchy's own 2.0, still
#     heavier than the 1.0 leaves, and at alpha 0.85 rather than opaque
#   * peers paler still
#   * only the focused sector keeps a direction arrow
# The per-measure gain is recomputed over the SHORTER frame: it is defined
# as "one gain per measure over the complete frame", and the complete frame
# is now 21 months.
D2_WINDOW = 21
frame2 = frame.iloc[-D2_WINDOW:]
half2 = {}
for measure in frame2.columns.get_level_values('Measure').unique():
    block2 = frame2.xs(measure, axis=1, level='Measure').to_numpy()
    half2[measure] = (block2.max() - block2.min()) / 2.0
scaled2 = frame2 / [half2[c[2]] for c in frame2.columns]
span2 = f'{frame2.index[0]:%b %Y} - {frame2.index[-1]:%b %Y}'
PALEST = '#f2f2f2'
MEAN_GREY = '#4a4a4a'


def draw_hierarchy_mean(ax, data, color=MEAN_GREY, lw=1.6, alpha=0.85):
    """Draw ONLY the hierarchy's parent trace, in a colour of our choosing.

    THIS IS A WORKAROUND, and it is isolated here so that it is obvious
    where it starts and stops. 1.1 has no way to give a hierarchy's parent
    a colour independent of its leaves: `palette` colours every trace in a
    group identically, and `hue_mode='mixture'` forces the parent to the
    MEAN of its leaves' colours -- which can never be darker than the
    darkest leaf, so five pale peers force a pale mean. See README.md for
    the five spellings that were measured.

    So the same complete frame is drawn a second time with a dark
    single-colour palette, and every trace except the parent is hidden.
    Both calls receive identical input, so they normalize identically and
    the two parent paths coincide exactly (asserted below). This is NOT
    how one normally plots a hierarchy -- ordinary usage is one call -- and
    a gallery example carrying it must say so where a reader will see it.
    """
    before = set(map(id, ax.lines))
    hyp.plot(data, '-', palette=[color], reduce=None, ndims=2,
             normalize=None, colorbar=False, legend=False, ax=ax, show=False)
    drawn = [ln for ln in ax.lines if id(ln) not in before]
    # the parent is found by the style the LIBRARY gave it -- the heavier
    # line -- not by its position in the list
    parent = max(drawn, key=lambda ln: ln.get_linewidth())
    for line in drawn:
        line.set_visible(line is parent)
    parent.set(linewidth=lw, alpha=alpha, zorder=5)
    return parent


def draw_scaled2(ax, focus):
    hyp.plot(scaled2, '-', hue=weights2(focus),
             palette=SECTOR_COLORS + [PALEST], hue_mode='mixture',
             reduce=None, ndims=2, normalize=None, colorbar=False, ax=ax,
             show=False)
    return sorted([c for c in ax.collections
                   if getattr(c, '_hyp_trace_index', None) is not None],
                  key=lambda c: c._hyp_trace_index)


def weights2(focus):
    rows = []
    for leaf in range(6):
        column = leaf if leaf == focus else 6
        rows.append(np.tile(np.eye(7)[column], (len(frame2), 1)))
    return rows


figd2, axesd2 = plt.subplots(2, 3, figsize=(7.36, 4.9), dpi=100)
for i, (name, axd) in enumerate(zip(NAMES, axesd2.ravel())):
    panel2 = draw_scaled2(axd, focus=i)
    mean2 = draw_hierarchy_mean(axd, scaled2)
    mark(axd, path_of(panel2[i]), SECTOR_COLORS[i], lw=1.4, ms=4.0, mew=1.2,
         head=10)
    axd.plot(*np.array(mean2.get_data()).T[0], 'o', color=MEAN_GREY, ms=3.6,
             mfc='white', mew=1.1, zorder=6)
    axd.set_title(name, fontsize=9.5, fontweight='bold',
                  color=SECTOR_COLORS[i], loc='left', pad=3)
figd2.supxlabel(XLAB, fontsize=8.5)
figd2.supylabel(YLAB, fontsize=8.5)
figd2.suptitle(f'One sector at a time, against the market mean  ({span2})',
               fontsize=10, fontweight='bold')
figd2.tight_layout()
figd2.savefig(f'{OUT}PROTO_D2_small_multiples.png')
print(f'saved D2 ({D2_WINDOW} months, {span2})')
print('D2: the two calls agree on the parent to %.2e'
      % np.abs(path_of(panel2[-1]) - np.array(mean2.get_data()).T).max())
print('D2: mean lw %.1f vs leaf %.1f, alpha %.2f'
      % (mean2.get_linewidth(), np.atleast_1d(panel2[0].get_linewidth())[0],
         mean2.get_alpha()))
limsd2 = [(round(float(a.get_xlim()[0]), 6), round(float(a.get_xlim()[1]), 6),
           round(float(a.get_ylim()[0]), 6), round(float(a.get_ylim()[1]), 6))
          for a in axesd2.ravel()]
print('D2: every panel shares one normalization:', len(set(limsd2)) == 1)


# ---------------- D3: the knot, diagnosed and fixed ----------------------
# D2 followed round 12's prescription (shorter window, restrained mean) and
# the mean was STILL a knot. Measured, rather than re-tuned:
#
#   roughness = total turning / total drawn length
#
#   30 months, monthly     mean 8.59   sector 2.94
#   21 months, monthly     mean 8.59   sector 2.94   (the window is not it)
#   + Smooth(kernel=5)     mean 10.10  sector 3.23   (smoothing makes it WORSE:
#                                                     it removes length faster
#                                                     than it removes turning)
#   60 months, 6-monthly   mean 4.18   sector 2.74
#
# The cause is in the data, not the styling: over the same span the market
# mean covers 0.46x the ground of a single sector while turning as often,
# because averaging six sectors cancels DIRECTION rather than noise. No
# linewidth, alpha or window fixes that -- only fewer, wider strides do.
# Sampling every 6 months over 5 years gives 10 strokes instead of 30, and
# shows more history rather than less.
D3_MONTHS, D3_STEP = 60, 6
frame3 = frame_all.iloc[-D3_MONTHS:].apply(np.expm1).iloc[::D3_STEP]
half3 = {}
for measure in frame3.columns.get_level_values('Measure').unique():
    half3[measure] = np.ptp(
        frame3.xs(measure, axis=1, level='Measure').to_numpy()) / 2.0
scaled3 = frame3 / [half3[c[2]] for c in frame3.columns]
span3 = f'{frame3.index[0]:%b %Y} - {frame3.index[-1]:%b %Y}'


def weights3(focus):
    return [np.tile(np.eye(7)[leaf if leaf == focus else 6], (len(frame3), 1))
            for leaf in range(6)]


figd3, axesd3 = plt.subplots(2, 3, figsize=(7.36, 4.9), dpi=100)
for i, (name, axd) in enumerate(zip(NAMES, axesd3.ravel())):
    hyp.plot(scaled3, '-', hue=weights3(i), palette=SECTOR_COLORS + [PALEST],
             hue_mode='mixture', reduce=None, ndims=2, normalize=None,
             colorbar=False, ax=axd, show=False)
    panel3 = sorted([c for c in axd.collections
                     if getattr(c, '_hyp_trace_index', None) is not None],
                    key=lambda c: c._hyp_trace_index)
    mean3 = draw_hierarchy_mean(axd, scaled3)
    mark(axd, path_of(panel3[i]), SECTOR_COLORS[i], lw=1.4, ms=4.0, mew=1.2,
         head=10)
    axd.plot(*np.array(mean3.get_data()).T[0], 'o', color=MEAN_GREY, ms=3.6,
             mfc='white', mew=1.1, zorder=6)
    axd.set_title(name, fontsize=9.5, fontweight='bold',
                  color=SECTOR_COLORS[i], loc='left', pad=3)
figd3.supxlabel(XLAB, fontsize=8.5)
figd3.supylabel(YLAB, fontsize=8.5)
figd3.suptitle(f'One sector at a time, against the market mean  ({span3}, '
               f'every {D3_STEP} months)', fontsize=10, fontweight='bold')
figd3.tight_layout()
figd3.savefig(f'{OUT}PROTO_D3_small_multiples.png')
print(f'saved D3 ({len(frame3)} strokes, {span3})')


# ---------------- E: the ANIMATED composition ----------------------------
# Round 12, finding 1, is right that six panels x two calls cannot be one
# animation -- and the reason is sharper than "twelve schedules". MEASURED:
# `hyp.plot` IGNORES `ax=` when `animate=` is set and builds its own figure
# (probe: `out.figure is fig` -> False, and the passed axes stays empty). So
# panels cannot be composed into one figure through `ax=` at all, and the
# two-call dark-mean workaround is impossible in an animated plot.
#
# `HyperAnimation.draw_frame(i)` IS public and documented as "the supported
# way to drive an animation from a test or a script", so N animations could
# be stepped in lockstep -- but they live on N separate figures, so there
# is nothing to save them into.
#
# The way out is to stop asking matplotlib for the panels. Translate each
# panel group into its own region of ONE shared coordinate box and the six
# panels become six column groups of a single frame: ONE `hyp.plot` call,
# ONE animation, an ordinary `.save()`. The panels are laid out in the
# DATA, which is also what makes them share one normalization by
# construction rather than by assertion.
#
# And with the panels in the data, colour-by-group becomes the RIGHT
# behaviour rather than an obstacle: each panel is one sector, its leaves
# are that sector's constituent stocks, and its parent -- automatically
# computed, automatically heavier -- is that sector's mean. No hue matrix,
# no second call, no hidden artists. The hierarchy is demonstrated six
# times over instead of being worked around once.
PANEL_COLS, STEP = 3, 2.6

ticker_cols = {}
for sector, tickers in SECTORS.items():
    for ticker in tickers:
        lvl = levels[(sector, ticker)]
        ticker_cols[(sector, ticker, 'cumulative return')] = (
            lvl - lvl.shift(CUM_WINDOW))
        ticker_cols[(sector, ticker, 'drawdown')] = (
            lvl - lvl.rolling(DD_WINDOW, min_periods=1).max())
stocks = pd.DataFrame(ticker_cols).dropna()
stocks.columns = pd.MultiIndex.from_tuples(
    stocks.columns, names=['Sector', 'Ticker', 'Measure'])
stocks = stocks.iloc[-D3_MONTHS:].apply(np.expm1).iloc[::D3_STEP]

# one display gain per measure over the COMPLETE frame, exactly as C/D3
halfs = {m: np.ptp(stocks.xs(m, axis=1, level='Measure').to_numpy()) / 2.0
         for m in stocks.columns.get_level_values('Measure').unique()}
tiled = stocks / [halfs[c[2]] for c in stocks.columns]
untiled = tiled.copy()          # the same content, before any translation
# ...then translate each sector's block into its own cell of the grid
for p, sector in enumerate(NAMES):
    ox, oy = (p % PANEL_COLS) * STEP, -(p // PANEL_COLS) * STEP
    for column in tiled.columns:
        if column[0] == sector:
            tiled[column] = tiled[column] + (
                ox if column[2] == 'cumulative return' else oy)

anim = hyp.plot(tiled, '-', palette=SECTOR_COLORS, reduce=None, ndims=2,
                normalize=None, animate='parallel', duration=6, frame_rate=15,
                legend=False, colorbar=False, show=False,
                title=f"Six sectors, their stocks, and each sector's mean "
                      f'({span3})')
axe = anim.figure.axes[0]
anim.figure.set_size_inches(7.36, 4.9)
anim.draw_frame(anim.n_frames - 1)
# the library draws its own frame box as a patch on the axes (NOT the axes
# spines, which is what the first attempt at this hid). It spans the full
# normalized box, so once the view is cropped to the panel grid only its
# left and right edges remain -- two mystery vertical rules through the
# figure. Recorded here, before any of this script's own patches exist,
# so it can be told apart from them by identity rather than by looks.
library_patches = list(axe.patches)
print(f'E: the library drew {len(library_patches)} patch(es) of its own')

# The panels are laid out in the data, so their positions on screen are a
# fact to be READ OFF the drawn artists rather than assumed -- which is
# also the check that the tiling worked. Labels and cell frames are
# annotations; nothing here touches a trajectory.
cells = {}
for ln in axe.lines:
    key = tuple(np.round(ln.get_color(), 3))
    x, y = np.asarray(ln.get_xdata()), np.asarray(ln.get_ydata())
    lo_x, hi_x, lo_y, hi_y = cells.get(key, (np.inf, -np.inf, np.inf, -np.inf))
    cells[key] = (min(lo_x, x.min()), max(hi_x, x.max()),
                  min(lo_y, y.min()), max(hi_y, y.max()))
# The cells belong on a GRID, so their positions come from the offsets
# that built the grid -- not from where each sector's content happens to
# sit, which staggered them. `hyp.plot` maps data to the drawn box with one
# affine transform (one gain, per-dimension centring), and that transform
# is recoverable from the extents: the drawn extent IS the data extent
# mapped, since interpolation along a segment stays inside it.
def _affine(drawn_lo, drawn_hi, data_lo, data_hi):
    gain = (drawn_hi - drawn_lo) / (data_hi - data_lo)
    return gain, drawn_lo - gain * data_lo


def _extent(frame_, measure):
    block = frame_.xs(measure, axis=1, level='Measure').to_numpy()
    return float(block.min()), float(block.max())


# recovered from a STATIC draw of the same frame, not from the animated
# one: an animated last frame is a hair short of the full path, which
# biases the extents differently in each dimension (measured: it made one
# gain look 7% larger than the other, which is an artefact of measuring
# mid-reveal rather than a property of the transform).
probe_fig, probe_ax = plt.subplots()
hyp.plot(tiled, '-', palette=SECTOR_COLORS, reduce=None, ndims=2,
         normalize=None, legend=False, colorbar=False, ax=probe_ax, show=False)
static_x = (min(np.min(ln.get_xdata()) for ln in probe_ax.lines),
            max(np.max(ln.get_xdata()) for ln in probe_ax.lines))
static_y = (min(np.min(ln.get_ydata()) for ln in probe_ax.lines),
            max(np.max(ln.get_ydata()) for ln in probe_ax.lines))
plt.close(probe_fig)
gain_x, off_x = _affine(*static_x, *_extent(tiled, 'cumulative return'))
gain_y, off_y = _affine(*static_y, *_extent(tiled, 'drawdown'))
print('E: one gain for both dimensions:',
      abs(gain_x - gain_y) < 1e-9, f'({gain_x:.4f}, {gain_y:.4f})')

# one cell box, in data units: the union of every sector's content BEFORE
# translation, so all six cells are identical and the panels differ only
# in what they contain
base_x, base_y = _extent(untiled, 'cumulative return'), _extent(untiled, 'drawdown')
pad = 0.06 / gain_x
for p_, (name, colour) in enumerate(zip(NAMES, SECTOR_COLORS)):
    ox, oy = (p_ % PANEL_COLS) * STEP, -(p_ // PANEL_COLS) * STEP
    x0 = gain_x * (ox + base_x[0] - pad) + off_x
    x1 = gain_x * (ox + base_x[1] + pad) + off_x
    y0 = gain_y * (oy + base_y[0] - pad) + off_y
    y1 = gain_y * (oy + base_y[1] + pad) + off_y
    axe.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                ec='#cccccc', lw=0.8, zorder=0))
    axe.annotate(name, xy=(x0, y1 + 0.012), color=colour, fontsize=8.5,
                 fontweight='bold', va='bottom')
    cells[name] = (x0, x1, y0, y1)
print('E: axes artists ->', {k: v for k, v in (
    ('lines', len(axe.lines)), ('patches', len(axe.patches)),
    ('collections', len(axe.collections)), ('texts', len(axe.texts)),
    ('figure axes', len(anim.figure.axes)),
    ('figure lines', len(anim.figure.lines)),
    ('figure patches', len(anim.figure.patches)))})
# and drop the letterbox the normalized square leaves around a 3x2 grid
boxes = [cells[name] for name in NAMES]
axe.set_xlim(min(b[0] for b in boxes) - 0.02, max(b[1] for b in boxes) + 0.02)
axe.set_ylim(min(b[2] for b in boxes) - 0.02, max(b[3] for b in boxes) + 0.075)

per_colour = {}
for ln in axe.lines:
    per_colour.setdefault(tuple(np.round(ln.get_color(), 3)), []).append(
        round(float(ln.get_linewidth()), 1))
print(f'E: ONE call -> {anim.n_frames} frames, {len(anim.figure.axes)} axes, '
      f'{len(axe.lines)} lines, {len(per_colour)} colours')
print('E: linewidths within one panel (leaves then parent):',
      sorted(sorted(v) for v in per_colour.values())[0])
# MEASURED: the animated backend re-applies its axis styling on EVERY
# frame, so decoration applied once is undone by the next frame -- the
# first render of this prototype came back with the axes spines restored
# after `save()`. `on_frame` is the supported hook for exactly this, so
# the panel-grid styling is re-applied per frame rather than once.
def restyle(context):
    for spine in axe.spines.values():
        spine.set_visible(False)
    for patch in library_patches:
        patch.set_visible(False)
    axe.set_xlim(*xlim)
    axe.set_ylim(*ylim)


xlim, ylim = axe.get_xlim(), axe.get_ylim()
restyle(None)
anim.on_frame(restyle)
anim.save(f'{OUT}PROTO_E_animated.gif', dpi=100)
print('E: still restyled after save:',
      not any(sp.get_visible() for sp in axe.spines.values())
      and not any(pt.get_visible() for pt in library_patches))
anim.figure.savefig(f'{OUT}PROTO_E_lastframe.png', dpi=100)
print('E: GIF %.2f MB'
      % (pathlib.Path(f'{OUT}PROTO_E_animated.gif').stat().st_size / 1e6))
