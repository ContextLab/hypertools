# -*- coding: utf-8 -*-
"""Does SEASONAL WEATHER earn the forecast claim the Market example lost?

`notes/market_representation_study_2026-08-17.md` retired the Market
example's forecast story: applied mechanically, the preregistered rule
passed three drawdown specifications, and the audit that file committed to
in advance then killed all three -- a parameter-free "predict full
recovery" rule beat the models in 10 of 12 cells. Plan 4 still owes the
gallery a prediction story, and the maintainer's suggestion was seasonal
weather, motion, or sensor data.

This file asks the same question of the weather data, under the SAME rule,
and it is written before any of it has been run.

THE ACCEPTANCE RULE, PREREGISTERED
----------------------------------
A forecast claim survives only if, on one measure, a model

  1. beats EVERY baseline listed below, on that same measure,
  2. with the same sign in BOTH evaluation-anchor blocks,
  3. scoring POSITIVE (a model that is consistently less wrong than an
     even-worse baseline has not predicted anything), and
  4. at t=1 -- the horizon an animated example actually draws.

Nothing is added to the baseline set after seeing a result. That is the
whole point of writing them down here:

  zero              predict no change
  persistence       the last observed change, repeated
  mean_change       the mean change over all history
  ew_continuation   an exponentially weighted recent change
  seasonal_naive    the change over the SAME calendar month a year ago
  climatology       the historical mean for the target calendar month

The last two are the ones that matter, and they are baselines rather than
post-hoc audits BECAUSE the data is seasonal: for a series with a known
annual cycle, the trivial competitor is obvious in advance, so it belongs
in the rule from the start. The Market study had to carry its parameter-free
competitor as a separate audit precisely because that competitor (drawdown
recovers to zero) only became obvious after looking at which measure
passed. Preregistering it here is the cheaper, more honest version of the
same discipline.

A claim that survives this is worth drawing. A claim that does not is worth
saying so, and the Market example is the precedent for saying it.
"""
import json
import os
import sys
import tempfile
import time
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The scoring primitives are IMPORTED, not re-implemented. Two studies that
# each define "the predicted change" can drift into scoring different things
# and then get compared to each other; a shared callee cannot drift from
# itself. This is the same rule the examples gate applies to its two line
# counters.
from scripts.market_representation_study import (          # noqa: E402
    _baseline_deltas, _predict_delta)

DRAWN_HORIZON = 1                  # the horizon an animated example draws
EXPECTED_BLOCKS = ('block1', 'block2')
MEASURES = ['temperature', 'precipitation', 'humidity', 'windspeed']
FEATS = ['temperature_2m_mean', 'precipitation_sum',
         'relative_humidity_2m_mean', 'windspeed_10m_max']
START, END = '1990-01-01', '2024-12-31'
CACHE = os.path.join(tempfile.gettempdir(), 'hypertools_gallery_cache')
SEASON = 12                        # months in the cycle

#: the example's own six cities, three per hemisphere
CITIES = {
    'New York': (40.71, -74.01, 'Northern'),
    'London': (51.51, -0.13, 'Northern'),
    'Tokyo': (35.68, 139.69, 'Northern'),
    'Sydney': (-33.87, 151.21, 'Southern'),
    'Cape Town': (-33.92, 18.42, 'Southern'),
    'Santiago': (-33.45, -70.66, 'Southern'),
}


def fetch_city_months(name, lat, lon):
    """Monthly-mean feature matrix for one city, or None.

    Same shape as the example's fetcher -- cached, and a failure returns
    None so an offline run degrades instead of dying -- with one deliberate
    difference: it SAYS WHY it fell back. The example's `except Exception:
    return None` is silent, and this study's first run was silently
    synthetic end to end because of it. That mattered: the offline generator
    is a sine wave plus noise, so `climatology` is literally the
    data-generating process and every model loses to it by construction. A
    study cannot be allowed to answer a question about real weather with
    fabricated weather and not mention it.
    """
    try:
        os.makedirs(CACHE, exist_ok=True)
        url = (f'https://archive-api.open-meteo.com/v1/archive?latitude={lat}'
               f'&longitude={lon}&start_date={START}&end_date={END}'
               f'&daily={",".join(FEATS)}&timezone=auto')
        dest = os.path.join(CACHE, f'wx_{name.replace(" ", "_")}.json')
        if not (os.path.exists(dest) and os.path.getsize(dest) > 0):
            req = urllib.request.Request(
                url, headers={'User-Agent': 'hypertools-gallery/1.0'})
            # open-meteo rate-limits: six of these fired back to back gets
            # HTTP 429 on the last three, MEASURED. Retry with backoff
            # rather than treating a throttle as "offline" -- the first run
            # of this study did exactly that and answered the question with
            # fabricated data.
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(req, timeout=60) as r:
                        data = r.read()
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code != 429 or attempt == 4:
                        raise
                    wait = 2 ** attempt
                    print(f'  . {name}: 429, retrying in {wait}s',
                          file=sys.stderr)
                    time.sleep(wait)
            tmp = dest + '.part'
            with open(tmp, 'wb') as f:
                f.write(data)
            os.replace(tmp, dest)
        with open(dest) as f:
            d = json.load(f)['daily']
        df = pd.DataFrame({f: pd.to_numeric(pd.Series(d[f]), errors='coerce')
                           for f in FEATS}).interpolate().ffill().bfill()
        dt = pd.to_datetime(d['time'])
        df['ym'] = dt.year * 12 + dt.month
        return df.groupby('ym')[FEATS].mean().to_numpy()
    except Exception as exc:
        print(f'  ! {name}: {type(exc).__name__}: {exc}', file=sys.stderr)
        return None


def synthetic_city_months(hemi, n_months=420, seed=0):
    """The example's own fallback, verbatim in shape: a seasonal loop with
    the hemispheres in opposite phase, plus a slow drift."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_months)
    phase = 0.0 if hemi == 'Northern' else np.pi
    season = np.sin(2 * np.pi * t / 12 + phase)
    warming = t / n_months
    temp = 14 + 11 * season + 3 * warming + rng.standard_normal(n_months) * 0.6
    precip = 60 + 25 * np.cos(2 * np.pi * t / 12 + phase) \
        + rng.standard_normal(n_months) * 5
    humid = 70 + 10 * season + rng.standard_normal(n_months) * 2
    wind = 20 + 5 * np.sin(2 * np.pi * t / 12 + phase + 1.0) \
        + rng.standard_normal(n_months) * 1.5
    return np.column_stack([temp, precip, humid, wind])


def city_arrays():
    """One (n_months, 4) array per city, trimmed to a common length."""
    mats, offline = [], False
    for seed, (name, (lat, lon, hemi)) in enumerate(CITIES.items()):
        m = fetch_city_months(name, lat, lon)
        if m is None:
            offline = True
            m = synthetic_city_months(hemi, seed=seed)
        mats.append(np.asarray(m, dtype=float))
    n = min(len(m) for m in mats)
    return [m[:n] for m in mats], offline


def seasonal_deltas(series, anchor, horizon):
    """The two SEASONAL baselines, both computable at `anchor`.

    `seasonal_naive` repeats the change observed over the same calendar
    month one year earlier. `climatology` targets the historical mean for
    the target month, taken over every earlier year present in the history,
    and predicts the change needed to reach it.

    Both return a zero vector when the history is too short to contain a
    full prior cycle, which is why `evaluate` also refuses anchors below
    `2 * SEASON`: a baseline silently degrading to "predict no change" would
    make the model look like it beat something it never faced.
    """
    last = series[anchor - 1]
    target = anchor + horizon - 1
    if anchor - 1 - SEASON < 0:
        naive = np.zeros(series.shape[1])
    else:
        naive = series[target - SEASON] - series[anchor - 1 - SEASON]
    months = np.arange(anchor)
    same = months[(months % SEASON) == (target % SEASON)]
    clim = (series[same].mean(axis=0) - last if len(same)
            else np.zeros(series.shape[1]))
    return {'seasonal_naive': naive, 'climatology': clim}


BASELINES = ('zero', 'persistence', 'mean_change', 'ew_continuation',
             'seasonal_naive', 'climatology')


def evaluate(arrays, model, horizon, anchors, block):
    """Pooled predicted-vs-realised Pearson correlation, per measure."""
    predicted, realised = [], []
    baselines = {name: [] for name in BASELINES}
    t0 = time.time()
    for anchor in anchors:
        for series in arrays:
            if anchor + horizon - 1 >= len(series) or anchor < 2 * SEASON:
                continue
            history = series[:anchor]
            predicted.append(_predict_delta(history, model, horizon))
            realised.append(series[anchor + horizon - 1] - history[-1])
            trivial = _baseline_deltas(history, horizon)
            trivial.update(seasonal_deltas(series, anchor, horizon))
            for name in BASELINES:
                baselines[name].append(trivial[name])
    elapsed = time.time() - t0
    if not predicted:
        return None
    predicted, realised = np.array(predicted), np.array(realised)

    def _pearson(pred):
        return [float(np.corrcoef(pred[:, j], realised[:, j])[0, 1])
                if np.std(pred[:, j]) > 0 and np.std(realised[:, j]) > 0
                else float('nan') for j in range(realised.shape[1])]

    return {'model': model, 'horizon': horizon, 'block': block,
            'n': len(predicted), 'seconds': elapsed,
            'pearson': _pearson(predicted),
            'baselines': {name: _pearson(np.array(vals))
                          for name, vals in baselines.items()}}


def best_baseline(row, measure_index):
    """The STRONGEST trivial competitor for one measure.

    `zero` has no variance, so its correlation is nan by construction --
    take the max over the FINITE ones only, or a single nan becomes "the
    baseline" and every model wins by default.
    """
    finite = [row['baselines'][name][measure_index]
              for name in BASELINES
              if np.isfinite(row['baselines'][name][measure_index])]
    return max(finite) if finite else float('nan')


def verdict(rows, drawn_horizon=DRAWN_HORIZON):
    """Apply the preregistered rule mechanically, and report what survived."""
    survivors, considered = [], {}
    for row in rows:
        if row is None or row['horizon'] != drawn_horizon:
            continue
        for j, measure in enumerate(MEASURES):
            key = (row['model'], measure)
            considered.setdefault(key, []).append(
                (row['block'], row['pearson'][j], best_baseline(row, j)))
    for key, entries in considered.items():
        blocks = [block for block, _, _ in entries]
        if sorted(blocks) != sorted(EXPECTED_BLOCKS):
            continue
        if all(np.isfinite(score) and np.isfinite(base)
               and score > max(0.0, base) for _, score, base in entries):
            survivors.append((key, entries))
    return survivors


def main():
    arrays, offline = city_arrays()
    n = len(arrays[0])
    print(f'{len(arrays)} cities x {n} months x {len(MEASURES)} measures '
          f'({"SYNTHETIC (offline)" if offline else "open-meteo archive"})')
    if offline and os.environ.get('WEATHER_STUDY_ALLOW_SYNTHETIC') != '1':
        raise SystemExit(
            'REFUSING to run: at least one city fell back to the synthetic\n'
            'generator, which is a sine wave plus noise -- `climatology` IS\n'
            'its data-generating process, so the answer would be rigged\n'
            'against every model before a single fit ran. Fix the network or\n'
            'set WEATHER_STUDY_ALLOW_SYNTHETIC=1 to see the rigged numbers on\n'
            'purpose (they are not evidence about weather).')

    # two NON-OVERLAPPING evaluation-anchor blocks. The models' histories
    # still overlap -- every fit expands from the start of the series -- so
    # this splits WHERE each forecast is evaluated, not what it was fitted
    # on, and the wording matters: it is not two independent samples.
    lo, hi = 2 * SEASON + 1, n - 1
    mid = (lo + hi) // 2
    blocks = {'block1': np.linspace(lo, mid, 20, dtype=int),
              'block2': np.linspace(mid + 1, hi - 1, 20, dtype=int)}

    rows = []
    for model in ('Kalman', 'ARIMA'):
        for block, anchors in blocks.items():
            row = evaluate(arrays, model, DRAWN_HORIZON, anchors, block)
            rows.append(row)
            if row is None:
                continue
            print(f'\n{model:8s} {block}  n={row["n"]:4d}  '
                  f'{row["seconds"]:.1f}s')
            for j, measure in enumerate(MEASURES):
                base = best_baseline(row, j)
                mark = 'BEATS' if (np.isfinite(row['pearson'][j])
                                   and np.isfinite(base)
                                   and row['pearson'][j] > max(0.0, base)) else '.'
                print(f'   {measure:14s} r={row["pearson"][j]:+.3f}   '
                      f'best baseline {base:+.3f} '
                      f'({max(BASELINES, key=lambda b: (row["baselines"][b][j] if np.isfinite(row["baselines"][b][j]) else -9))})'
                      f'   {mark}')

    survivors = verdict(rows)
    print('\n' + '=' * 68)
    if not survivors:
        print('NOTHING SURVIVES the preregistered rule at t=1.')
        print('Weather does not earn a forecast claim either.')
    else:
        print(f'{len(survivors)} specification(s) survive at t=1:')
        for (model, measure), entries in survivors:
            print(f'  {model} / {measure}')
            for block, score, base in sorted(entries):
                print(f'     {block}: r={score:+.3f} vs baseline {base:+.3f}')
    return 0 if True else 1


if __name__ == '__main__':
    raise SystemExit(main())
