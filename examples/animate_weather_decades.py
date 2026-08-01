# -*- coding: utf-8 -*-
"""
===================================================================
Decades of weather: bold means, faint cities, temperature colormaps
===================================================================

Hierarchical data as a single ``hyp.plot`` call. Monthly weather (temperature,
precipitation, humidity, wind) for several cities is reduced to 3-D with
``hyp.reduce``; each city's decades trace a seasonal loop that slowly drifts.
The cities are grouped by **hemisphere**: we draw a bold hemisphere-*mean* loop
per hemisphere plus the individual cities as faint context -- the classic
bold-means / faint-leaves hierarchy -- smoothed (``manip='Smooth'``) and
animated with ``chemtrails=True`` so the history accumulates as the years play.

Every loop is colored by **temperature**, with a *separate* hot-cold colormap
per hemisphere (two labeled colorbars). We build the hierarchy as an explicit
**list** (city loops + hemisphere-mean loops) rather than a row ``MultiIndex``
DataFrame: hyp's MultiIndex expansion draws the same bold-means/faint-leaves
hierarchy automatically, but it colors by *group* and ignores a continuous
``hue=`` (GH #95), so the temperature coloring would be lost. With a list, the
per-point temperature ``hue`` applies. The color axis is kept tight to the
prominent hemisphere-mean range so the bold loops sweep the full colormap over
the seasons; the faint cities that run hotter/colder saturate at the ends.

Alongside the 3-D view a **second panel** plots raw *daily* temperature: one
thin, translucent line per city under two bold hemisphere-mean lines, drawn
with the *same* per-hemisphere colorscales as the loops and revealed in
lockstep with the animation, with a vertical "now" cursor. The daily series are
read back out of the same cached archive responses the monthly matrices were
built from, so the panel costs no extra request. It is the same data seen the
ordinary way, so the 3-D shape can be read against a familiar time series.

**Data & graceful degradation.** Monthly means are built from the open-meteo
historical archive (`open-meteo.com <https://open-meteo.com>`_) and cached on
disk. If the network is unavailable the example synthesizes seasonal loops and
matching daily temperatures (hemispheres in opposite phase, with a slow warming
drift), so it always renders. To keep the gallery build fast this lighter version uses 6 cities
(3 per hemisphere) over 1990-2024 rather than the full 12-city, 1960-2024 demo.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import json
import os
import tempfile
import urllib.request

import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import hypertools as hyp

CACHE = os.path.join(tempfile.gettempdir(), 'hypertools_gallery_cache')
os.makedirs(CACHE, exist_ok=True)

START, END = '1990-01-01', '2024-12-31'
CITIES = {                       # 3 per hemisphere: loops stay followable
    'New York': (40.71, -74.01, 'Northern'),
    'London': (51.51, -0.13, 'Northern'),
    'Tokyo': (35.68, 139.69, 'Northern'),
    'Sydney': (-33.87, 151.21, 'Southern'),
    'Cape Town': (-33.92, 18.42, 'Southern'),
    'Santiago': (-33.45, -70.66, 'Southern'),
}
FEATS = ['temperature_2m_mean', 'precipitation_sum',
         'relative_humidity_2m_mean', 'windspeed_10m_max']


def fetch_city_months(name, lat, lon):
    """Monthly-mean feature matrix for one city from open-meteo, or ``None``."""
    try:
        url = (f'https://archive-api.open-meteo.com/v1/archive?latitude={lat}'
               f'&longitude={lon}&start_date={START}&end_date={END}'
               f'&daily={",".join(FEATS)}&timezone=auto')
        dest = os.path.join(CACHE, f'wx_{name.replace(" ", "_")}.json')
        if not (os.path.exists(dest) and os.path.getsize(dest) > 0):
            req = urllib.request.Request(
                url, headers={'User-Agent': 'hypertools-gallery/1.0'})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            with open(dest, 'wb') as f:
                f.write(data)
        d = json.load(open(dest))['daily']
        df = pd.DataFrame({f: pd.to_numeric(pd.Series(d[f]), errors='coerce')
                           for f in FEATS}).interpolate().ffill().bfill()
        dt = pd.to_datetime(d['time'])
        df['ym'] = dt.year * 12 + dt.month
        return df.groupby('ym')[FEATS].mean().to_numpy()
    except Exception:
        return None


def synthetic_city_months(hemi, n_months=420, seed=0):
    """Fallback: a seasonal loop (hemispheres in opposite phase) that drifts."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_months)
    phase = 0.0 if hemi == 'Northern' else np.pi          # opposite seasons
    season = np.sin(2 * np.pi * t / 12 + phase)
    warming = t / n_months                                # slow drift
    temp = 14 + 11 * season + 3 * warming + rng.standard_normal(n_months) * 0.6
    precip = 60 + 25 * np.cos(2 * np.pi * t / 12 + phase) \
        + rng.standard_normal(n_months) * 5
    humid = 70 + 10 * season + rng.standard_normal(n_months) * 2
    wind = 20 + 5 * np.sin(2 * np.pi * t / 12 + phase + 1.0) \
        + rng.standard_normal(n_months) * 1.5
    return np.column_stack([temp, precip, humid, wind])


mats, hemis, offline = [], [], False
for seed, (name, (lat, lon, hemi)) in enumerate(CITIES.items()):
    m = fetch_city_months(name, lat, lon)
    if m is None:
        offline = True
        m = synthetic_city_months(hemi, seed=seed)
    mats.append(m)
    hemis.append(hemi)
print(f'weather: {len(CITIES)} cities '
      f'({"synthetic (offline fallback)" if offline else "open-meteo archive"})')

# align lengths (synthetic + real runs may differ), then ONE hyp.reduce call
# on the LIST of cities: normalize='across' z-scores the four weather features
# across the stacked rows, a single IncrementalPCA is fit on that stack (so the
# loops are directly comparable), and the rows come back split into one array
# per city -- no manual vstack, z-score, or slicing. With ndims=3 each month
# becomes a single 3-D point on the top 3 principal components.
min_len = min(len(m) for m in mats)
mats = [m[:min_len] for m in mats]
city_loops = [np.asarray(loop) for loop in
              hyp.reduce(mats, reduce='IncrementalPCA', ndims=3,
                         normalize='across')]
city_temp = [mats[i][:, 0] for i in range(len(mats))]

# hemisphere-mean loops (bold) from the per-city loops
N_idx = [i for i in range(len(mats)) if hemis[i] == 'Northern']
S_idx = [i for i in range(len(mats)) if hemis[i] == 'Southern']
Nmean_loop = np.mean([city_loops[i] for i in N_idx], axis=0)
Smean_loop = np.mean([city_loops[i] for i in S_idx], axis=0)
Nmean_temp = np.mean([city_temp[i] for i in N_idx], axis=0)
Smean_temp = np.mean([city_temp[i] for i in S_idx], axis=0)

# a SEPARATE hot-cold colormap per hemisphere, spliced into one palette; hue
# maps each hemisphere onto its own half. Color axis tight to the bold means'
# range so they sweep the full colormap over the seasons.
Ncm = LinearSegmentedColormap.from_list('N', ['#08306b', '#e31a1c'])   # blue->red
Scm = LinearSegmentedColormap.from_list('S', ['#0c7c8c', '#f4a300'])   # teal->amber
combined = LinearSegmentedColormap.from_list(
    'combo', [Ncm(x) for x in np.linspace(0, 1, 128)]
    + [Scm(x) for x in np.linspace(0, 1, 128)])
Nlo, Nhi = float(Nmean_temp.min()), float(Nmean_temp.max())
Slo, Shi = float(Smean_temp.min()), float(Smean_temp.max())


def enc(temps, hemi):
    t = np.asarray(temps, float)
    if hemi == 'Northern':
        return np.clip(0.49 * (t - Nlo) / (Nhi - Nlo), 0.0, 0.49)     # [0, 0.49]
    return np.clip(0.51 + 0.49 * (t - Slo) / (Shi - Slo), 0.51, 1.0)  # [0.51, 1.0]


# datasets: faint city loops, then the two bold hemisphere-mean loops
datasets = list(city_loops) + [Nmean_loop, Smean_loop]
hue = np.concatenate(
    [enc(city_temp[i], hemis[i]) for i in range(len(mats))]
    + [enc(Nmean_temp, 'Northern'), enc(Smean_temp, 'Southern')])
CITY_LW, MEAN_LW = 1.0, 2.2          # means bold, but not heavy-handed
lws = [CITY_LW] * len(mats) + [MEAN_LW, MEAN_LW]

# ``hyp.plot`` antialiases every drawn line by default (``antialias=True``), so
# these long seasonal loops render smooth at any frame rate.
duration, fps = 8, 20
date_labels = pd.date_range(START, periods=min_len, freq='MS')

# THE hypertools call: a list of loops colored by a temperature hue, smoothed,
# chemtrails; the two bold means are emphasized below. manip='Smooth' is a
# per-dataset Savitzky-Golay pass applied before anything is drawn, so each
# loop reads as a clean seasonal cycle; chemtrails=True leaves the loop already
# traversed glowing faintly behind each moving head; legend=False and
# colorbar=False suppress the library's own annotations because this figure
# draws its own two per-hemisphere colorbars (one colorbar cannot describe two
# different colormaps).
anim = hyp.plot(datasets, fmt='-', hue=hue, palette=combined,
                colorbar=False, linewidth=lws, animate=True,
                chemtrails=True, manip='Smooth', duration=duration,
                frame_rate=fps, legend=False, elev=20, azim=-70,
                size=(13, 6), show=False)
fig = anim.figure
ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
ax.set_position([-0.01, 0.03, 0.52, 0.90])       # 3-D view: left half

# hyp's multicolor collections don't inherit the per-dataset linewidth, so set
# the emphasis ourselves. Collections are created head-first in dataset order
# (plot._apply_multicolor_lines), so the first `n` are the per-dataset HEADS
# and the two means are the LAST two of those; the rest are the faint trails.
_colls = [c for c in ax.collections if isinstance(c, Line3DCollection)]
NDS = len(datasets)
heads, trails = _colls[:NDS], _colls[NDS:]
MEAN_IDX = {NDS - 2, NDS - 1}
for k in MEAN_IDX:
    heads[k].set_linewidth(MEAN_LW)

# --- 2nd panel: every city's DAILY temperature ------------------------------
# The same hierarchy as the 3-D view, in ordinary axes: thin, translucent
# per-city lines (the "leaves") under two bold hemisphere-mean lines, every
# line colored point-by-point by that day's temperature through its OWN
# hemisphere colormap and norm -- the same two the colorbars label. It reveals
# in lockstep with the animation.


def fetch_city_daily_temp(name):
    """Daily mean temperature for one city, or ``None``.

    Read back out of the SAME cached archive response ``fetch_city_months``
    built its monthly matrix from, so this panel costs no extra request.
    """
    try:
        d = json.load(open(os.path.join(
            CACHE, f'wx_{name.replace(" ", "_")}.json')))['daily']
        v = pd.to_numeric(pd.Series(d['temperature_2m_mean']), errors='coerce')
        return v.interpolate().ffill().bfill().to_numpy(dtype=float)
    except Exception:
        return None


def synthetic_city_daily(hemi, n_days, seed=0):
    """Fallback: the daily-resolution twin of ``synthetic_city_months``."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    phase = 0.0 if hemi == 'Northern' else np.pi          # opposite seasons
    season = np.sin(2 * np.pi * t / 365.25 + phase)
    return (14 + 11 * season + 3 * (t / n_days)
            + rng.standard_normal(n_days) * 1.5)


daily = []
for seed, (name, (_, _, hemi)) in enumerate(CITIES.items()):
    v = fetch_city_daily_temp(name)
    if v is None:
        v = synthetic_city_daily(hemi, int(round(min_len * 365.25 / 12)),
                                 seed=seed)
    daily.append(v)
ND = min(len(v) for v in daily)
daily = [v[:ND] for v in daily]
# daily samples laid out on the SAME x axis as the animation's month clock, so
# the reveal cursor and the month index line up exactly
day_x = np.linspace(0, min_len - 1, ND)
Nmean_daily = np.mean([daily[i] for i in N_idx], axis=0)
Smean_daily = np.mean([daily[i] for i in S_idx], axis=0)

ax_t = fig.add_axes([0.575, 0.145, 0.30, 0.70])
nrmN, nrmS = Normalize(Nlo, Nhi), Normalize(Slo, Shi)
temp_colls = []


def temp_line(y, hemi, lw, alpha, z):
    """One per-day-colored temperature line, added hidden (revealed below).

    The weights are deliberately far apart. Decades of seasonal cycles land
    only a few pixels apart across this panel, so every line here is a dense
    picket fence rather than a followable curve; the means separate from the
    city haze only if the cities are drawn very faint and very thin. (A white
    halo under the means was tried and removed: at this cycle density the halo
    fills the band with white instead of outlining a curve.)
    """
    pts = np.column_stack([day_x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=Ncm if hemi == 'Northern' else Scm,
                        norm=nrmN if hemi == 'Northern' else nrmS,
                        linewidth=lw, alpha=alpha, zorder=z)
    lc.set_segments([])                        # revealed progressively below
    ax_t.add_collection(lc)
    temp_colls.append((lc, segs, y[:-1]))      # y[:-1] = per-segment temp


for i in range(len(mats)):                     # faint city "leaves"
    temp_line(daily[i], hemis[i], 0.3, 0.12, 2)
temp_line(Nmean_daily, 'Northern', 1.8, 1.0, 4)    # bold hemisphere means,
temp_line(Smean_daily, 'Southern', 1.8, 1.0, 4)    # opaque, as in the 3-D view
_allt = np.concatenate(daily)
ax_t.set_xlim(0, min_len - 1)
ax_t.set_ylim(_allt.min() - 1.5, _allt.max() + 1.5)
_yr_ticks = list(range(0, min_len, 60))                    # every 5 years
ax_t.set_xticks(_yr_ticks)
ax_t.set_xticklabels([date_labels[i].strftime('%Y') for i in _yr_ticks],
                     fontsize=9)
ax_t.tick_params(axis='y', labelsize=9)
ax_t.set_ylabel('daily mean temperature (°C)', fontsize=10)
ax_t.set_title('every city, every day', fontsize=11, color='#333', pad=6)
for _s in ('top', 'right'):
    ax_t.spines[_s].set_visible(False)
ax_t.grid(alpha=0.18, linewidth=0.6)
now_line = ax_t.axvline(0, color='#444', lw=1.2, alpha=0.8)   # "now" cursor

# two labeled colorbars, one per hemisphere
caxN = fig.add_axes([0.925, 0.55, 0.016, 0.33])
cbN = fig.colorbar(ScalarMappable(Normalize(Nlo, Nhi), Ncm), cax=caxN)
cbN.set_label('Northern temp (°C)', fontsize=9)
caxS = fig.add_axes([0.925, 0.145, 0.016, 0.33])
cbS = fig.colorbar(ScalarMappable(Normalize(Slo, Shi), Scm), cax=caxS)
cbS.set_label('Southern temp (°C)', fontsize=9)

# the title spans the WHOLE figure (3-D box + temperature panel), not the box
title = fig.text(0.47, 0.965, '', ha='center', va='top', fontsize=13.5,
                 fontweight='bold', color='#1a1a1a')
total = int(round(fps * duration))


def decorate(ctx):
    """Per-frame decoration: bold hemisphere means vs. faint cities (the
    multicolor updater resets per-segment alpha every frame, so it is
    re-applied here), and the 2nd panel's lockstep reveal + "now" cursor.
    Registered below via ``anim.on_frame`` -- by the time this runs,
    hyp.plot() has already drawn the frame, so (unlike the pre-1.1
    ``ani._func`` monkeypatch this replaces) there is no original updater
    to call through to, and nothing to return."""
    frame = ctx.frame
    for k, c in enumerate(heads):
        c.set_alpha(1.0 if k in MEAN_IDX else 0.16)
    for c in trails:
        c.set_alpha(0.10)
    idx = min(min_len - 1, int(frame / max(1, total - 1) * min_len))
    # 2nd panel reveals in lockstep with the 3-D animation (month -> day index)
    kd = int(np.clip(round(idx / max(1, min_len - 1) * (ND - 1)), 0, ND - 1))
    for lc, segs, vals in temp_colls:
        lc.set_segments(segs[:kd])
        lc.set_array(vals[:kd])            # keep colors aligned to segments
    now_line.set_xdata([idx, idx])
    title.set_text('decades of weather, 6 cities  '
                   f'{date_labels[idx].strftime("%b %Y")}')


anim.on_frame(decorate)
