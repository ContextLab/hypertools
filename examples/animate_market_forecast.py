# -*- coding: utf-8 -*-
"""
======================================================
Many markets as one path: chemtrails + a live forecast
======================================================

Several daily financial series are bundled into a single moving 3-D "market
path": each day is a point, the whole history is reduced with
``hyp.reduce(..., reduce='IncrementalPCA', ndims=3, manip='Smooth',
normalize='across')`` (hypertools' canonical ``manip -> normalize -> reduce``
order: a Savitzky-Golay smooth per dataset, an across-the-stack z-score, then
the 3 highest-variance directions in mini-batches), and ``hyp.plot``
draws it with **one slow quarter-turn** of the camera over the whole clip
(``rotations=0.25`` -- enough parallax to read the 3-D shape without the box
spinning out from under the overlay), colored by an equal-weight index (a
continuous ``hue`` + labeled ``colorbar``) and animated with
``chemtrails=True`` -- a moving head that leaves the whole history glowing
faintly behind it.

On top of the library call we overlay a **live forecast**: at each step a
Kalman filter (``hyp.predict(..., model='Kalman')``) extrapolates the next few
months from the history so far, drawn as a dashed red tail off the current
point, and every *past* forecast stays on-screen as a thin faint fan in the
SAME red (they are all forecasts, so they read as one family, separated by
weight and opacity rather than hue) as a "history of
predictions" fan. A subtitle keeps a running **directional-accuracy** score:
each forecast is compared with what the path actually did over the same
horizon, and only counts once that horizon has elapsed on screen.

**Coordinate note (important for any overlay on an animation).** ``hyp.plot``
internally normalizes the reduced path into its drawn cube, so points in the
original ``reduce`` space do NOT line up with what's on screen. We therefore
read the TRUE on-screen head straight from the plotted line artist
(``ani._args[1][0]``) each frame -- guaranteeing the forecast starts exactly at
the visible head and stays in the box -- and recover the (reduce -> drawn)
scale with a small per-axis fit so the red-space forecast delta lands in drawn
units. The forecast carries a single visual GAIN so it is legible; it is
illustrative, not a price target.

**Data & graceful degradation.** The series are pulled from FRED
(`fred.stlouisfed.org <https://fred.stlouisfed.org>`_) as small CSVs and cached
on disk. If the network is unavailable the example falls back to a synthetic
basket of correlated random-walk "assets", so it always renders -- the
technique (reduce -> hue/colorbar -> chemtrails -> forecast overlay) is
identical either way.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import os
import tempfile
import urllib.request

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

import hypertools as hyp

CACHE = os.path.join(tempfile.gettempdir(), 'hypertools_gallery_cache')
os.makedirs(CACHE, exist_ok=True)

# five broad daily FRED series -- a rough cross-section of "the market"
FRED_IDS = ['SP500', 'NASDAQCOM', 'DGS10', 'DCOILWTICO', 'VIXCLS']
START, END = '2004-01-01', '2024-01-01'


def fetch_fred(ids, start, end):
    """Return (dates, (days, n_series) matrix) from FRED, forward-filled and
    aligned; or ``None`` if anything (network, parsing) goes wrong."""
    try:
        import pandas as pd
        frames = []
        for sid in ids:
            url = (f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}'
                   f'&cosd={start}&coed={end}')
            dest = os.path.join(CACHE, f'fred_{sid}_{start}_{end}.csv')
            if not (os.path.exists(dest) and os.path.getsize(dest) > 0):
                req = urllib.request.Request(
                    url, headers={'User-Agent': 'hypertools-gallery/1.0'})
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = r.read()
                with open(dest, 'wb') as f:
                    f.write(data)
            df = pd.read_csv(dest, na_values=['.'])
            df.columns = ['date', sid]
            df['date'] = pd.to_datetime(df['date'])
            frames.append(df.set_index('date'))
        # sort=False + an explicit sort_index(): pandas is deprecating the
        # implicit sort here, and we sort the union index ourselves anyway
        merged = pd.concat(frames, axis=1,
                           sort=False).sort_index().ffill().dropna()
        return merged.index.to_numpy(), merged.to_numpy(dtype=float)
    except Exception:
        return None


def synthetic_basket(n_days=1000, n_assets=5, seed=0):
    """Fallback: correlated geometric-random-walk 'assets' (a stand-in market
    with enough shared structure to trace a coherent low-D path)."""
    rng = np.random.default_rng(seed)
    market = np.cumsum(rng.standard_normal(n_days)) * 0.6      # shared factor
    prices = []
    for k in range(n_assets):
        idio = np.cumsum(rng.standard_normal(n_days)) * (0.4 + 0.1 * k)
        prices.append(100 * np.exp(0.02 * (market + idio) / 10))
    dates = np.arange(n_days)
    return dates, np.column_stack(prices)


fetched = fetch_fred(FRED_IDS, START, END)
if fetched is None:
    dates, prices = synthetic_basket()
    source = 'synthetic basket (offline fallback)'
else:
    dates, prices = fetched
    source = 'FRED daily series'
print(f'market data: {prices.shape[0]} days x {prices.shape[1]} series '
      f'({source})')

# thin to a manageable number of points, take LOG prices (assets grow
# ~exponentially; log linearizes the path), then ONE hyp.reduce call in the
# canonical stage order (manip -> normalize -> reduce): Smooth strips
# day-to-day jitter, normalize='across' z-scores the columns across the
# stacked rows (replacing a hand-rolled z-score), and IncrementalPCA keeps the
# 3 highest-variance directions, so every day becomes one 3-D point.
THIN = max(1, len(prices) // 800)
prices = prices[::THIN]
idx_level = (prices / prices[0]).mean(axis=1) * 100.0     # equal-weight index
logp = np.log(np.clip(prices, 1e-9, None))
red = np.asarray(hyp.reduce(logp, reduce='IncrementalPCA', ndims=3,
                            manip='Smooth', normalize='across'))
T = len(red)

# Precompute Kalman forecasts at monthly anchors from the history-so-far, as a
# reduce-space DELTA (running Kalman on the full daily path every frame is
# slow). STEP is derived from the thinning factor so it stays ~1 trading month
# of the ORIGINAL series no matter how much we thinned.
#
# MIN_HIST: require a real run-up of history before forecasting at all. Kalman
# fits from only a couple of samples are wildly over-confident: measured on the
# FRED series above, dropping the requirement lets the largest forecast reach
# ~25x the median forecast length, and the visual GAIN below turns that into an
# arrow that streaks across the box. Requiring two years of monthly history
# halves the worst case (to ~12x the median) without changing directional
# skill; the CAP below handles what remains of the tail (95th pct ~7x median).
#
# MODEL CHOICE, measured rather than assumed. On a 20-stock daily version of
# this pipeline (293 anchors, scoring the predicted 4-month displacement
# against what actually happened), direction was called correctly by:
#     Kalman   51%        ARIMA   51%        Laplace   65%
# against 62% for the trivial "assume it keeps drifting the way it has been"
# rule. A reduced market path is close to a random walk with drift, so a single
# linear-Gaussian fit has little to grip; hyp's 'Laplace' is a
# likelihood-weighted Bayesian ensemble over a population of candidate
# forecasters and holds up better. Kalman is kept HERE only because it is ~30x
# faster (this file runs on every docs build). Swap the one keyword below to
# model='Laplace' for the better forecast.
STEP = max(2, round(21 / THIN))                           # ~1 trading month
HORIZON = 4                                               # months ahead
MIN_HIST = 24                                             # monthly samples
anchors = list(range(MIN_HIST * STEP, T, STEP))
raw_fc, raw_hit = {}, {}
for a in anchors:
    hist = red[:a + 1:STEP]
    if len(hist) < 2:
        continue
    f = np.asarray(hyp.predict(hist, model='Kalman', t=HORIZON))
    # hyp.predict returns exactly `t` NEW rows, all of them future steps, so
    # f[0] is the FIRST forecast step and not the last observation: anchor the
    # displacement on the last OBSERVED row. (`f - f[0]` would discard a whole
    # step and force the first displacement to zero.) The prepended zero row
    # keeps the drawn forecast starting exactly at the current head.
    raw_fc[a] = np.vstack([np.zeros((1, f.shape[1])),
                           f - hist[-1]])                 # reduce-space delta
    # score it against what ACTUALLY happened over the same horizon: did the
    # forecast point the right way? (directional hit = positive dot product)
    j = min(T - 1, a + HORIZON * STEP)
    d_pred, d_act = raw_fc[a][-1], red[j] - red[a]
    n1, n2 = np.linalg.norm(d_pred), np.linalg.norm(d_act)
    raw_hit[a] = (float(d_pred @ d_act) > 0) if (n1 > 1e-12 and n2 > 1e-12) \
        else None

# THE hypertools call: one market path, hue = equal-weight index, chemtrails,
# and ONE slow quarter-turn of the camera (rotations=0.25) over the clip.
# duration/frame_rate MUST be passed: otherwise hyp falls back to its 30s/30fps
# defaults while this script's `total` says otherwise, desyncing the forecasts.
duration, fps = 8, 20
anim = hyp.plot(red, fmt='-', reduce=None, hue=idx_level, colorbar=False,
                palette='plasma', animate=True, chemtrails=True,
                rotations=0.25, duration=duration, frame_rate=fps,
                linewidth=2.2, size=(9, 6.5), show=False)
fig, ani = anim
ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
ax.set_position([0.0, 0.03, 0.78, 0.9])
total = int(round(fps * duration))

# read the visible line artist so forecasts anchor at the TRUE drawn head, and
# fit the (reduce -> drawn) per-axis scale so the reduce-space delta lands in
# drawn units (see the module docstring for why this is necessary).
#
# This ONE-TIME setup step is the one place this example still reaches into
# matplotlib's private FuncAnimation internals (`ani._args`/`ani._func`),
# deliberately: it needs the fully-revealed, ANTIALIASED on-screen line (this
# is a synchronous "force a render, then read it back" operation, not a
# per-frame callback), and there is no public equivalent -- `ctx.datasets`
# (from `on_frame=`) is the pre-antialiasing array at a coarser resolution
# and fits a measurably different (~2-8%, checked empirically) slope. The
# RECURRING per-frame decoration below has a clean public replacement and
# uses it (`anim.on_frame`); this setup step does not, so it is left alone
# rather than silently changing the fitted forecast geometry.
market_line = ani._args[1][0]
_orig = ani._func
_orig(total - 1, *ani._args)                              # reveal fully, once
_fx, _fy, _fz = market_line.get_data_3d()
full_drawn = np.column_stack([_fx, _fy, _fz])
K = len(full_drawn)
_red_rs = np.column_stack([np.interp(np.linspace(0, T - 1, K), np.arange(T),
                                     red[:, c]) for c in range(3)])
SLOPE = np.array([np.polyfit(_red_rs[:, c], full_drawn[:, c], 1)[0]
                  for c in range(3)])
BLO = np.array([ax.get_xlim3d()[0], ax.get_ylim3d()[0], ax.get_zlim3d()[0]])
BHI = np.array([ax.get_xlim3d()[1], ax.get_ylim3d()[1], ax.get_zlim3d()[1]])


def _frame_of(a):
    """Frame at which reduce-space sample ``a`` is the animated head."""
    return int(round(a / max(1, T - 1) * (total - 1)))


# forecasts as DRAWN-space deltas, keyed by the frame they were made; a single
# GAIN makes them legible (hyp packs this path into a sub-region of the cube)
FC = {_frame_of(a): SLOPE[None, :] * d for a, d in raw_fc.items()}
_ends = [np.linalg.norm(d[-1]) for d in FC.values() if len(d)]
MED_LEN = 0.20                                            # median arrow length
GAIN = MED_LEN / (np.median(_ends) or 1.0)
# A few months of movement is tiny next to a 20-year path, so the arrow has to
# be amplified to be visible at all -- which also amplifies the heavy tail of
# Kalman forecast magnitudes. Capping each forecast at 1.8x the MEDIAN length
# keeps relative differences readable (a bigger predicted move is still a
# longer arrow) while stopping any single forecast from streaking across the
# box; an absolute cap would instead flatten every large forecast onto the same
# length.
CAP = 1.8 * MED_LEN


def _scale(d):
    d = GAIN * d
    L = np.linalg.norm(d[-1])
    return d * (CAP / L) if L > CAP else d


FC = {f: _scale(d) for f, d in FC.items()}
frame_list = sorted(FC)
HEAD_CACHE = {}

# running directional accuracy: a forecast only counts once its horizon has
# actually elapsed on screen (no peeking at the future)
_matured = sorted((_frame_of(min(T - 1, a + HORIZON * STEP)), raw_hit[a])
                  for a in raw_fc if raw_hit.get(a) is not None)
ACC = np.full(total, np.nan)                              # frame -> accuracy %
_n = _k = _mi = 0
for _f in range(total):
    while _mi < len(_matured) and _matured[_mi][0] <= _f:
        _n += 1
        _k += int(_matured[_mi][1])
        _mi += 1
    if _n:
        ACC[_f] = 100.0 * _k / _n
N_SCORED = _n
print(f'forecasts: {len(FC)} drawn, {N_SCORED} scored; '
      f'final directional accuracy = {ACC[-1]:.0f}%')


def _smooth(pts, n=80):
    """Densify a short polyline so it draws smooth.

    ``antialias_line`` is the exact routine ``hyp.plot(antialias=True)`` runs
    on every library-drawn line; we call it directly here because this
    forecast overlay is hand-drawn matplotlib rather than a plotted dataset.
    There is no public re-export of it (unlike ``title=``/``on_frame=``, this
    is smoothing, not a per-frame callback, so it is outside plan 1.1's
    scope) -- reimplementing PCHIP antialiasing by hand here would risk
    silently drifting from what ``hyp.plot`` actually draws, so the private
    import stays.
    """
    from hypertools._shared.helpers import antialias_line
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return pts
    return antialias_line(pts, n)[0]


def _hang(head, delta):
    return _smooth(np.clip(head + delta, BLO, BHI))


# history fan + the current forecast, in the SAME red: these are all forecasts,
# so what separates them is weight and opacity (thin/faint = already made and
# left behind, thick/dashed/bright = the one being made right now)
N_HIST = 16
FC_COLOR = '#E23B2E'
HIST_COLOR = FC_COLOR
hist_lines = [ax.plot([], [], [], '-', color=HIST_COLOR, lw=1.1, alpha=0.0,
                      zorder=6)[0] for _ in range(N_HIST)]
for _ln in hist_lines:
    _ln.set_clip_on(False)
fc_line, = ax.plot([], [], [], '--', color=FC_COLOR, lw=2.6, alpha=0.98,
                   zorder=10)
fc_line.set_clip_on(False)

# labeled colorbar for the equal-weight index
cax = fig.add_axes([0.82, 0.14, 0.02, 0.66])
sm = ScalarMappable(Normalize(idx_level.min(), idx_level.max()), cmap='plasma')
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label('equal-weight index (start = 100)', fontsize=9)

title = fig.text(0.40, 0.965, '', ha='center', va='top', fontsize=14,
                 fontweight='bold', color='#1a1a1a')
# the running score lives UNDER the title, in its own lighter subtitle line
acc_label = fig.text(0.40, 0.925, '', ha='center', va='top', fontsize=11.5,
                     color='#555')
# legend built from REAL Line2D handles, so each entry is drawn in exactly the
# style it has on screen (thick red dashed = live forecast; thin faint red
# solid = the past-forecast fan) instead of two identical text labels
fig.legend(handles=[
    Line2D([], [], color=FC_COLOR, lw=2.6, ls='--',
           label=f'forecast from today (next {HORIZON} months)'),
    Line2D([], [], color=HIST_COLOR, lw=1.1, ls='-', alpha=0.75,
           label='past forecasts, as they were made'),
], loc='lower left', bbox_to_anchor=(0.055, 0.02), ncol=1, frameon=False,
    fontsize=12.5, handlelength=3.2, labelspacing=0.6, labelcolor='#444')
fig.text(0.40, 0.005, 'arrows amplified for visibility; length is relative, '
         'not a price target', ha='center', va='bottom', fontsize=9,
         color='#8a8a8a', style='italic')


def decorate(ctx):
    """Per-frame decoration: the live forecast arrow, the past-forecast fan,
    and the running accuracy subtitle. Registered below via ``anim.on_frame``
    -- hyp.plot() has already moved the market path's head for this frame by
    the time this runs, so (unlike the pre-1.1 ``ani._func`` monkeypatch this
    replaces) there is no original updater to call through to."""
    num = ctx.frame
    hx, hy, hz = market_line.get_data_3d()
    head = np.array([hx[-1], hy[-1], hz[-1]])
    HEAD_CACHE[num] = head
    passed = [f for f in frame_list if f <= num]
    for ln in hist_lines:
        ln.set_alpha(0.0)
    if passed:
        cur = _hang(head, FC[passed[-1]])
        fc_line.set_data(cur[:, 0], cur[:, 1])
        fc_line.set_3d_properties(cur[:, 2])
        prior = passed[:-1][-N_HIST:]
        for slot, f in enumerate(prior):
            hp = _hang(HEAD_CACHE.get(f, head), FC[f])
            hist_lines[slot].set_data(hp[:, 0], hp[:, 1])
            hist_lines[slot].set_3d_properties(hp[:, 2])
            hist_lines[slot].set_alpha(0.08 + 0.30 * (slot + 1) / len(prior))
    else:
        fc_line.set_data([], [])
        fc_line.set_3d_properties([])
    # running directional accuracy of every forecast whose horizon has already
    # elapsed on screen (50% = a coin flip)
    acc = ACC[min(num, total - 1)]
    title.set_text('many markets as one path')
    acc_label.set_text(
        'forecast direction correct so far: waiting for the first horizon'
        if np.isnan(acc) else
        f'forecast direction correct so far: {acc:.0f}%   (50% = coin flip)')


anim.on_frame(decorate)
