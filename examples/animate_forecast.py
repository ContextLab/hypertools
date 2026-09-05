# -*- coding: utf-8 -*-
"""
=====================================================================
Forecasting three regions' weather while it is drawn
=====================================================================

An animated forecast: ``hyp.plot(..., animate=True, predict='Kalman')``
refits the forecaster on the history revealed so far and re-anchors it on
the last revealed observation, so the prediction grows and bends with the
animation instead of standing still. ``forecast_trail=True`` keeps the
earlier forecasts on screen as a fading fan, so you can watch the
prediction *change* as history accumulates.

The data are monthly mean temperatures from the HyperTools paper's archive
(the same file the weather example uses), over its last five years: three
regions -- the Americas, Europe and Africa, Asia and the Pacific -- with six
cities each, chosen so every region spans both hemispheres. Each region is
one dataset whose six cities are its six **features**, so a year is one
loop through a shared 3-D space (the northern and southern cities in a
region peak six months apart, which is what opens the loop up). Five years
are five turns of each loop, and ``predict='Kalman'`` forecasts a year
ahead (``t=12`` raw observations, not frames): on this archive a Kalman fit
on the first four years predicts the fifth to within 1.4-3.1 degrees per
region, against a seasonal swing of about 9 (measured 2026-09-04).

Three keywords restyle the forecasts without touching the observed paths:
``forecast_hue=`` groups the forecasts (the two Old World regions share a
colour, the Americas have their own), ``forecast_palette=`` gives that
grouping its own colours, and ``forecast_fmt=`` draws every forecast dashed.
Everything they do not name is inherited from the trace a forecast
continues, drawn at half its alpha.

``slow_warning_seconds=`` is the one keyword here that changes no pixel.
An animated forecast needs one fit per distinct revealed history length,
so this clip needs 180 fits before its first frame; that measured about
6 s on 2026-09-04, under the 10 s at which the library would otherwise
warn that a long wait is expected. The example passes ``None`` to silence
the notice on slower machines, since the wait is known; a smaller number
makes it fire sooner.

**Data & graceful degradation.** The archive is fetched once and cached.
If the network is unavailable the example says which error it hit and
synthesizes three seasonal regions (a hemispheric mix in each, a slow
drift) so it always renders; ``HYPERTOOLS_OFFLINE`` makes the fetch refuse
rather than degrade, which is how the test-suite proves the import path
fetches nothing.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

# a pre-rendered gif of the forecast fan as this example's gallery thumbnail,
# as the other animated examples do
# sphinx_gallery_thumbnail_path = '_static/thumbnails/sphx_glr_animate_forecast_thumb.gif'

import os
import tempfile
import urllib.request
from typing import NamedTuple

import numpy as np
import pandas as pd

import hypertools as hyp

CACHE = os.path.join(tempfile.gettempdir(), 'hypertools_gallery_cache')
ARCHIVE = ('https://raw.githubusercontent.com/ContextLab/'
           'hypertools-paper-notebooks/master/data/temperatures.csv')
# three regions, six cities each, every region spanning both hemispheres:
# the cities are the FEATURES of a region's trajectory, and equal widths
# let the regions share one reduction
REGIONS = {'Americas': ['Chicago', 'Los_Angeles', 'Mexico', 'Montreal',
                        'New_York', 'Santiago'],
           'Europe & Africa': ['Cairo', 'Cape_Town', 'Istanbul', 'London',
                               'Moscow', 'Rome'],
           'Asia & Pacific': ['Bangkok', 'Bombay', 'Seoul', 'Shanghai',
                              'Sydney', 'Tokyo']}
N_MONTHS = 60                    # the archive's last five years
HORIZON = 12                     # forecast a year ahead, in months
DURATION, FRAME_RATE = 12, 15    # 180 frames


class Climate(NamedTuple):
    regions: list               # one (months x 6 cities) array per region
    names: list                 # the region names, in `regions` order
    source: str                 # which path produced them


# --- the data half: the ONLY code here that reaches the network -------------
def fetch_temperatures():
    """The archive's last `N_MONTHS` complete months, one array per region,
    or ``None`` (announced with the error) when it cannot be fetched."""
    if os.environ.get('HYPERTOOLS_OFFLINE'):
        raise RuntimeError('HYPERTOOLS_OFFLINE is set: refusing to fetch')
    os.makedirs(CACHE, exist_ok=True)
    dest = os.path.join(CACHE, 'temperatures.csv')
    try:
        if not os.path.exists(dest):
            req = urllib.request.Request(
                ARCHIVE, headers={'User-Agent': 'hypertools-gallery/1.1'})
            with urllib.request.urlopen(req, timeout=60) as response:
                payload = response.read()
            with open(dest + '.part', 'wb') as handle:
                handle.write(payload)
            os.replace(dest + '.part', dest)   # never a truncated cache
        # the archive carries '<City>' (absolute) and '<City>_anomaly'
        # columns; its complete rows end in August 2013
        recent = pd.read_csv(dest).dropna().tail(N_MONTHS)
        return [recent[cities].to_numpy(float)
                for cities in REGIONS.values()]
    except Exception as error:
        print(f'weather archive unavailable ({error!r}); using the synthetic '
              'fallback')
        return None


def synthetic_climate(source, n_months=N_MONTHS, seed=0):
    """Fallback: six seasonal cities per region, half of them in southern
    phase, at three different mean temperatures, with a slow drift."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_months)
    regions = []
    for base in (12.0, 15.0, 20.0):
        cities = []
        for k in range(6):
            phase = np.pi if k % 2 else 0.0          # alternate hemispheres
            cities.append(base + rng.uniform(6, 12)
                          * np.sin(2 * np.pi * t / 12 + phase
                                   + rng.uniform(-0.4, 0.4))
                          + 0.02 * t + rng.standard_normal(n_months) * 0.5)
        regions.append(np.column_stack(cities))
    return Climate(regions, list(REGIONS), source)


def load_climate():
    """The ONLY function here that may touch the network."""
    try:
        fetched = fetch_temperatures()
    except RuntimeError:
        fetched = None
    if fetched is None:
        return synthetic_climate('synthetic (offline)')
    return Climate(fetched, list(REGIONS), 'HyperTools paper archive')


def fixture_data():
    """The same payload from the seeded synthetic path. No network."""
    return synthetic_climate('synthetic (fixture)')


# --- the figure half: no network, deterministic given its input -------------
def construct_artifact(data):
    """`data` in, the animation out. Returns the HyperAnimation wrapper,
    never the unpacked pair."""
    # THE hypertools call. The three regions are three datasets in one
    # shared space; `predict=` refits on every distinct revealed history
    # and `forecast_trail=` keeps the earlier fits as a fading fan. The
    # three `forecast_*` keywords restyle the forecasts only: the two Old
    # World regions' forecasts share a colour, the Americas' has its own,
    # all three are dashed, and each still continues the path it belongs
    # to. `slow_warning_seconds=None` silences the long-schedule notice:
    # the 180 fits this clip needs measured about 6 s, a known wait.
    return hyp.plot(
        data.regions, '-', names=data.names,
        animate=True, duration=DURATION, frame_rate=FRAME_RATE,
        predict='Kalman', t=HORIZON, forecast_trail=True,
        forecast_hue=['New World', 'Old World', 'Old World'],
        forecast_palette=['#d62728', '#1f77b4'], forecast_fmt='--',
        slow_warning_seconds=None,
        title='Three regions, one year ahead', size=(8, 6), show=False)


if __name__ == '__main__':
    climate = load_climate()
    print(f'climate: {len(climate.regions)} regions x '
          f'{climate.regions[0].shape[0]} months x '
          f'{climate.regions[0].shape[1]} cities ({climate.source})')
    anim = construct_artifact(climate)
    fig = anim.figure
