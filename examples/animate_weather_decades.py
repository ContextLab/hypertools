# -*- coding: utf-8 -*-
"""
=======================================================================
A century of weather: twenty cities as twenty features, one hot path
=======================================================================

The figure from the HyperTools paper, in one library call, with two
companion panels that read off the same clock. Monthly mean temperatures
for **twenty cities spread across both hemispheres** (Bangkok to Montreal,
Sydney to Moscow) are treated not as twenty separate series but as **twenty
features of one measurement**: each month is a single 20-dimensional
observation of "what the world's weather was doing", and ``hyp.plot``
reduces that stream to a 3-D path.

Every point on the path is coloured by the **average temperature across all
twenty cities** on a diverging blue-cold / red-hot scale
(``palette='RdBu_r'``), so the seasons show up as the path sweeping between
the ends of the colormap and the slow warming trend shows up as where the
sweep sits. ``manip='Smooth'`` takes out month-to-month jitter before
anything is drawn, ``normalize='across'`` z-scores the twenty city columns
over the stacked rows so a hot city cannot dominate the reduction purely by
scale, and ``chemtrails=True`` leaves the traversed path glowing faintly
behind the moving head as 138 years play in one two-minute camera orbit.

The colour axis, the colorbar and the trail are all the library's; the
example adds no ``ScalarMappable`` and reaches for nothing private. What it
does add is one ``anim.on_frame`` hook that keeps three things in step with
the head of the path, every frame:

* a **figure-wide title** naming the month the head is passing through
  ("March 1954");
* a **world map** (Natural Earth 110 m coastlines, fetched once and cached)
  with one dot per city, coloured by *that city's* temperature that month
  through the *same* colormap and value range as the path's colorbar --
  so tropical cities sit at the red end all year while the mid-latitude
  ones flip between blue and red with the seasons, out of phase across the
  equator;
* a **mean-temperature-vs-time** panel, sharing its left and right edges
  with the map: the full 138-year series drawn faintly in grey once, the
  months revealed so far growing over it as a line coloured **segment by
  segment by the mean temperature it is drawn at** (same colormap and
  range again), and a head marker on the current month coloured like the
  head of the path. The raw monthly mean swings by ~15 \N{DEGREE SIGN}C
  every year (the coloured line and the marker bounce with it), so a
  trailing 12-month rolling mean is drawn in plain black over the revealed
  months to let the warming drift show through the seasons.

The callback derives the current month from the frame index (a parallel
animation exposes no reveal count) and assigns every artist's state from it
on every frame, as the hook contract requires. (Before 1.1 this example
monkeypatched ``ani._func`` to redraw a second panel every frame; the public
hook is what replaced that reach, and it now drives three.)

**Data & graceful degradation.** The temperature matrix and the city
coordinates are the ones published with the HyperTools paper (verified
2026-09-03: 1645 complete months, May 1875 - August 2013, 20 cities),
fetched once and cached. If the network is unavailable the example says
which error it hit and synthesizes twenty seasonal series in opposite
hemispheric phase with a slow warming drift, at synthetic coordinates, so
it always renders; if only the coastline file cannot be fetched, the map
panel says so and draws the dots on a bare longitude/latitude frame.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import json
import os
import tempfile
import urllib.request
from typing import NamedTuple

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

import hypertools as hyp

CACHE = os.path.join(tempfile.gettempdir(), 'hypertools_gallery_cache')
BASE = ('https://raw.githubusercontent.com/ContextLab/'
        'hypertools-paper-notebooks/master/data/')
LAND = ('https://raw.githubusercontent.com/nvkelso/natural-earth-vector/'
        'master/geojson/ne_110m_land.geojson')
MONTHS = ('January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December')
# map-label nudges (x points, y points, alignment) for the crowded corners
# of the archive's twenty cities; everything else is labelled to the right
LABEL_OFFSETS = {'Chicago': (-5, 4, 'right'), 'Montreal': (4, 5, 'left'),
                 'New_York': (4, -9, 'left'), 'Seoul': (-5, 4, 'right'),
                 'Shanghai': (3, -9, 'left'), 'Tokyo': (4, 3, 'left'),
                 'London': (-5, 3, 'right'), 'Rome': (-5, -9, 'right')}


class Weather(NamedTuple):
    temps: np.ndarray           # months x cities, monthly mean temperature
    cities: list                # the column names, in the archive's order
    lats: np.ndarray            # one latitude per city
    lons: np.ndarray            # one longitude per city
    years: np.ndarray           # calendar year of each row
    months: np.ndarray          # calendar month (1-12) of each row
    land: list                  # coastline rings, (n, 2) lon/lat arrays
    source: str                 # which path produced them


# --- the data half: the ONLY code here that reaches the network -------------
def _cached_download(name, url):
    """Fetch `url` into the gallery cache once; never leave a truncated file."""
    dest = os.path.join(CACHE, name)
    if not os.path.exists(dest):
        req = urllib.request.Request(
            url, headers={'User-Agent': 'hypertools-gallery/1.1'})
        with urllib.request.urlopen(req, timeout=60) as response:
            payload = response.read()
        with open(dest + '.part', 'wb') as handle:
            handle.write(payload)
        os.replace(dest + '.part', dest)   # never a truncated cache
    return dest


def fetch_temperatures():
    """Temperatures, city coordinates and the calendar of each row, or
    ``None``.

    The fallback is announced with the exception that caused it, so a run
    that degraded cannot be mistaken for one that fetched."""
    if os.environ.get('HYPERTOOLS_OFFLINE'):
        raise RuntimeError('HYPERTOOLS_OFFLINE is set: refusing to fetch')
    os.makedirs(CACHE, exist_ok=True)
    try:
        temps = pd.read_csv(_cached_download('temperatures.csv',
                                             BASE + 'temperatures.csv'))
        locs = pd.read_csv(_cached_download('temperature_locs.csv',
                                            BASE + 'temperature_locs.csv'))
        # temperatures.csv carries 'Year' and 'Month' columns (from January
        # 1850) plus both '<City>' (absolute) and '<City>_anomaly' columns;
        # temperature_locs.csv ('City', 'Lat', 'Long') fixes the city order.
        # The complete rows run May 1875 - August 2013 with 15 months
        # missing in between, so the calendar is read from the columns
        # rather than counted from the start.
        cities = list(locs['City'])
        complete = temps.dropna()
        return (complete[cities].to_numpy(float), cities,
                locs['Lat'].to_numpy(float), locs['Long'].to_numpy(float),
                complete['Year'].to_numpy(int), complete['Month'].to_numpy(int))
    except Exception as error:
        print(f'weather archive unavailable ({error!r}); using the synthetic '
              'fallback')
        return None


def fetch_coastlines():
    """Natural Earth's 110 m land polygons as a list of (n, 2) lon/lat
    rings, or ``[]`` (announced) when they cannot be fetched."""
    if os.environ.get('HYPERTOOLS_OFFLINE'):
        raise RuntimeError('HYPERTOOLS_OFFLINE is set: refusing to fetch')
    os.makedirs(CACHE, exist_ok=True)
    try:
        with open(_cached_download('ne_110m_land.geojson', LAND)) as handle:
            features = json.load(handle)['features']
    except Exception as error:
        print(f'coastlines unavailable ({error!r}); the map panel will have '
              'no land fill')
        return []
    rings = []
    for feature in features:            # outer ring of each (Multi)Polygon
        geometry = feature['geometry']
        polygons = ([geometry['coordinates']] if geometry['type'] == 'Polygon'
                    else geometry['coordinates'])
        rings.extend(np.asarray(polygon[0], float) for polygon in polygons)
    return rings


def synthetic_weather(source, n_months=1645, n_cities=20, seed=0):
    """Fallback: seasonal cycles in opposite hemispheric phase, drifting,
    at made-up coordinates spread across both hemispheres. No coastlines."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_months)
    columns = []
    for city in range(n_cities):
        phase = 0.0 if city % 2 == 0 else np.pi          # opposite seasons
        columns.append(14 + 11 * np.sin(2 * np.pi * t / 12 + phase)
                       + 3 * (t / n_months)
                       + rng.standard_normal(n_months) * 0.6)
    index = np.arange(n_cities)
    lats = np.where(index % 2 == 0, 30 + index, -20 - index)   # even: north
    lons = np.linspace(-165, 145, n_cities)
    ordinal = 1875 * 12 + 4 + t                               # from May 1875
    return Weather(np.column_stack(columns),
                   [f'city {i + 1}' for i in index], lats, lons,
                   ordinal // 12, ordinal % 12 + 1, [], source)


def load_weather():
    """The ONLY function here that may touch the network."""
    try:
        fetched = fetch_temperatures()
        land = fetch_coastlines() if fetched is not None else []
    except RuntimeError:
        fetched = None
    if fetched is None:
        return synthetic_weather('synthetic (offline)')
    return Weather(*fetched, land, 'HyperTools paper archive')


def fixture_data():
    """The same payload from the seeded synthetic path. No network."""
    return synthetic_weather('synthetic (fixture)')


# --- the figure half: no network, deterministic given its input -------------
def construct_artifact(data):
    """`data` in, the animation out. Returns the HyperAnimation wrapper,
    never the unpacked pair."""
    mean = data.temps.mean(axis=1)              # the hue: raw monthly mean
    n_months = len(mean)
    # THE hypertools call: twenty cities as twenty FEATURES of one path,
    # coloured by the average temperature across them on a blue-cold /
    # red-hot scale, one camera orbit over the two-minute reveal. Stage
    # order is the library's: manip -> normalize -> reduce -> animate.
    # An animated line is resampled onto the frame grid, so the frame rate
    # is also the path's resolution: 20 fps x 120 s = 2400 points, more than
    # the 1645 months, so the 12-month loop keeps every one of its vertices
    # (a 300-frame grid aliased it into chords -- measured 2026-09-03).
    anim = hyp.plot(
        data.temps, '-',
        hue=mean, palette='RdBu_r',
        colorbar={'label': 'Average temperature ($^\\circ$C)'},
        manip='Smooth', normalize='across',
        animate=True, chemtrails=True, rotations=1,
        duration=120, frame_rate=20, size=(14, 7), show=False)
    fig = anim.figure

    # Layout: the library's 3-D axes and colorbar take the left ~55%; the
    # map and the time series stack on the right under one suptitle.
    path_ax, cbar_ax = fig.axes[0], fig.axes[1]
    path_ax.set_position([0.0, 0.0, 0.50, 0.92])
    cbar_ax.set_position([0.50, 0.22, 0.014, 0.50])
    cbar_ax.yaxis.label.set_fontsize(11)

    # The map and the path share ONE colour scale: the colorbar spans the
    # hue's actual range, so the same norm sends each city's own monthly
    # temperature through the same colormap (extremes saturate).
    norm = matplotlib.colors.Normalize(mean.min(), mean.max())
    cmap = plt.get_cmap('RdBu_r')
    map_ax = fig.add_axes([0.615, 0.50, 0.38, 0.44])
    for ring in data.land:
        map_ax.add_patch(Polygon(ring, closed=True, facecolor='0.86',
                                 edgecolor='0.62', linewidth=0.4))
    map_ax.set(xlim=(-180, 180), ylim=(-60, 85), aspect='equal',
               xticks=[], yticks=[])
    if not data.land:                       # bare frame with faint gridlines
        map_ax.set(xticks=range(-180, 181, 60), yticks=range(-60, 91, 30))
        map_ax.tick_params(length=0, labelbottom=False, labelleft=False)
        map_ax.grid(True, color='0.88', linewidth=0.6)
        map_ax.text(0.5, 0.03, 'coastlines unavailable', fontsize=8,
                    color='0.5', ha='center', transform=map_ax.transAxes)
    dots = map_ax.scatter(data.lons, data.lats, c=data.temps[0], cmap=cmap,
                          norm=norm, s=55, edgecolors='black',
                          linewidths=0.6, zorder=5)
    for name, lon, lat in zip(data.cities, data.lons, data.lats):
        dx, dy, align = LABEL_OFFSETS.get(name, (4, 3, 'left'))
        map_ax.annotate(name.replace('_', ' '), (lon, lat), xytext=(dx, dy),
                        textcoords='offset points', ha=align, fontsize=6,
                        color='0.25')

    # Mean temperature over time: the whole raw series once, faintly; then
    # the raw months revealed so far, each segment coloured by the value it
    # ends at, a trailing 12-month mean over them in black, and a marker on
    # the current month coloured like the head of the path. The panel takes
    # its left/right edges from the map's DRAWN box (equal aspect shrinks
    # the map inside the box it was given), so the two line up exactly.
    when = data.years + (data.months - 1) / 12
    rolling = pd.Series(mean).rolling(12).mean().to_numpy()
    points = np.column_stack([when, mean])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    map_ax.apply_aspect()
    drawn = map_ax.get_position()
    line_ax = fig.add_axes([drawn.x0, 0.10, drawn.width, 0.37])
    line_ax.plot(when, mean, color='0.85', linewidth=0.6)
    revealed = line_ax.add_collection(
        LineCollection([], cmap=cmap, norm=norm, linewidths=0.9))
    trend, = line_ax.plot([], [], color='black', linewidth=1.6)
    head, = line_ax.plot([], [], 'o', markersize=8, markeredgecolor='black',
                         clip_on=False)          # whole marker at the ends
    line_ax.set(xlim=(when[0], when[-1]), ylim=(mean.min() - 1, mean.max() + 1),
                xlabel='year', ylabel='Average temperature ($^\\circ$C)')
    line_ax.spines[['top', 'right']].set_visible(False)

    def on_frame(ctx):
        # A parallel reveal exposes no reveal count: the head sits at
        # fraction frame / (n_frames - 1) of the path, hence of the months.
        i = min(round(ctx.frame / max(ctx.n_frames - 1, 1) * (n_months - 1)),
                n_months - 1)
        fig.suptitle(f'{MONTHS[data.months[i] - 1]} {data.years[i]}',
                     fontsize=17, y=0.965)
        dots.set_array(data.temps[i])
        revealed.set_segments(segments[:i])
        revealed.set_array(mean[1:i + 1])
        trend.set_data(when[:i + 1], rolling[:i + 1])
        head.set_data([when[i]], [mean[i]])
        head.set_markerfacecolor(cmap(norm(mean[i])))

    anim.on_frame(on_frame)
    return anim


if __name__ == '__main__':
    weather = load_weather()
    print(f'weather: {weather.temps.shape[0]} months x '
          f'{weather.temps.shape[1]} cities ({weather.source})')
    anim = construct_artifact(weather)
    fig = anim.figure
