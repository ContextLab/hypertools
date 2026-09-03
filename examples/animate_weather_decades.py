# -*- coding: utf-8 -*-
"""
=======================================================================
A century of weather: twenty cities as twenty features, one hot path
=======================================================================

The figure from the HyperTools paper, in one library call. Monthly mean
temperatures for **twenty cities spread across both hemispheres** (Bangkok
to Montreal, Sydney to Moscow) are treated not as twenty separate series
but as **twenty features of one measurement**: each month is a single
20-dimensional observation of "what the world's weather was doing", and
``hyp.plot`` reduces that stream to a 3-D path.

Every point on the path is coloured by the **average temperature across all
twenty cities** on a diverging blue-cold / red-hot scale
(``palette='RdBu_r'``), so the seasons show up as the path sweeping between
the ends of the colormap and the slow warming trend shows up as where the
sweep sits. ``manip='Smooth'`` takes out month-to-month jitter before
anything is drawn, ``normalize='across'`` z-scores the twenty city columns
over the stacked rows so a hot city cannot dominate the reduction purely by
scale, and ``chemtrails=True`` leaves the traversed path glowing faintly
behind the moving head as 138 years play.

There is no hand-built hierarchy, no hand-spliced colormap, no
``ScalarMappable``, and no per-frame callback: the colour axis, the
colorbar and the trail are all the library's.

**Data & graceful degradation.** The temperature matrix and the city
coordinates are the ones published with the HyperTools paper (verified
2026-09-03: 1645 complete months, 1875-2013, 20 cities), fetched once and
cached. If the network is unavailable the example says which error it hit
and synthesizes twenty seasonal series in opposite hemispheric phase with
a slow warming drift, so it always renders.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import os
import tempfile
import urllib.request
from typing import NamedTuple

import numpy as np
import pandas as pd

import hypertools as hyp

CACHE = os.path.join(tempfile.gettempdir(), 'hypertools_gallery_cache')
BASE = ('https://raw.githubusercontent.com/ContextLab/'
        'hypertools-paper-notebooks/master/data/')


class Weather(NamedTuple):
    temps: np.ndarray           # months x cities, monthly mean temperature
    cities: list                # the column names, in the archive's order
    source: str                 # which path produced them


# --- the data half: the ONLY code here that reaches the network -------------
def fetch_temperatures():
    """(months x 20 cities) monthly means and the city names, or ``None``.

    The fallback is announced with the exception that caused it, so a run
    that degraded cannot be mistaken for one that fetched."""
    if os.environ.get('HYPERTOOLS_OFFLINE'):
        raise RuntimeError('HYPERTOOLS_OFFLINE is set: refusing to fetch')
    os.makedirs(CACHE, exist_ok=True)
    try:
        frames = {}
        for name in ('temperatures.csv', 'temperature_locs.csv'):
            dest = os.path.join(CACHE, name)
            if not os.path.exists(dest):
                req = urllib.request.Request(
                    BASE + name, headers={'User-Agent': 'hypertools-gallery/1.1'})
                with urllib.request.urlopen(req, timeout=60) as response:
                    payload = response.read()
                with open(dest + '.part', 'wb') as handle:
                    handle.write(payload)
                os.replace(dest + '.part', dest)   # never a truncated cache
            frames[name] = pd.read_csv(dest)
        # the CSV carries both '<City>' (absolute) and '<City>_anomaly'
        # columns; the locations file fixes the city order
        cities = list(frames['temperature_locs.csv']['City'])
        complete = frames['temperatures.csv'].dropna()
        return complete[cities].to_numpy(float), cities
    except Exception as error:
        print(f'weather archive unavailable ({error!r}); using the synthetic '
              'fallback')
        return None


def synthetic_temperatures(n_months=1645, n_cities=20, seed=0):
    """Fallback: seasonal cycles in opposite hemispheric phase, drifting."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_months)
    columns = []
    for city in range(n_cities):
        phase = 0.0 if city % 2 == 0 else np.pi          # opposite seasons
        columns.append(14 + 11 * np.sin(2 * np.pi * t / 12 + phase)
                       + 3 * (t / n_months)
                       + rng.standard_normal(n_months) * 0.6)
    return (np.column_stack(columns),
            [f'city {i + 1}' for i in range(n_cities)])


def load_weather():
    """The ONLY function here that may touch the network."""
    try:
        fetched, source = fetch_temperatures(), 'HyperTools paper archive'
    except RuntimeError:
        fetched = None
    if fetched is None:
        fetched, source = synthetic_temperatures(), 'synthetic (offline)'
    return Weather(*fetched, source)


def fixture_data():
    """The same payload from the seeded synthetic path. No network."""
    return Weather(*synthetic_temperatures(), 'synthetic (fixture)')


# --- the figure half: no network, deterministic given its input -------------
def construct_artifact(data):
    """`data.temps` in, the animation out. Returns the HyperAnimation
    wrapper, never the unpacked pair."""
    # THE hypertools call: twenty cities as twenty FEATURES of one path,
    # coloured by the average temperature across them on a blue-cold /
    # red-hot scale. Stage order is the library's: manip -> normalize ->
    # reduce -> animate.
    anim = hyp.plot(
        data.temps, '-',
        hue=data.temps.mean(axis=1), palette='RdBu_r',
        colorbar={'label': 'average temperature across '
                           f'{len(data.cities)} cities (\N{DEGREE SIGN}C)'},
        manip='Smooth', normalize='across',
        animate=True, chemtrails=True,
        title=f'{len(data.cities)} cities, {data.temps.shape[0]} months, '
              'as one moving path',
        duration=8, frame_rate=20, size=(8, 7), show=False)
    return anim


if __name__ == '__main__':
    weather = load_weather()
    print(f'weather: {weather.temps.shape[0]} months x '
          f'{weather.temps.shape[1]} cities ({weather.source})')
    anim = construct_artifact(weather)
    fig = anim.figure
