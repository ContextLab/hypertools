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

AMENDMENT, 2026-08-19 -- HOW THE SCORES ARE POOLED ACROSS CITIES
----------------------------------------------------------------
Written and committed BEFORE the corrected numbers were produced, and the
first run's numbers are superseded by them.

The first version of `evaluate` concatenated every city's predicted and
realised changes and took ONE Pearson correlation per measure, in raw
units. Pearson is invariant to a scale shared by both of its arguments, but
NOT to different scales in different blocks of the pooled sample: a city
whose precipitation swings by 80 mm contributes ~1600x the covariance of a
city that swings by 2 mm, so the pooled number is close to that one city's
number and the other five barely vote. `market_representation_study.py`
already refuses this on the MEASURE axis, in its own docstring ("never
pooled across measures in raw units -- the three measures differ in scale,
and pooling them lets one dominate"). The same objection applies on the
CITY axis, and the first version of this file did not apply it.

The correction, fixed here before rerunning:

  unit scale     for each city and measure, s = the standard deviation of
                 that city's month-over-month changes over the whole
                 series. A pure units constant: it never reaches a model,
                 and it divides the prediction and the realisation of EVERY
                 competitor identically, so it cannot move one competitor
                 relative to another.

  pooled_scaled  the same pooled correlation, computed after dividing each
                 city's predicted and realised changes by that city's s.
                 THIS IS THE HEADLINE AGGREGATION.

  fisher_z       an independent check: correlate within each city, then
                 average with Fisher's z weighted by (n - 3), and report
                 the spread across cities. It never pools cities' units at
                 all, so it answers the same question a different way.

  pooled_raw     the defective original, kept and printed so the size of
                 the defect is on the record rather than described.

The verdict is applied under all three. A claim that survives under a
scale-free aggregation but not under `pooled_raw` would be a NEW claim: it
would not inherit this preregistration, and it would need its own.

WHY THERE IS NO END-TO-END "MULTIPLY A CITY BY 100" INVARIANCE TEST
-------------------------------------------------------------------
Because the shipped forecaster is not scale-equivariant, and pretending
otherwise would hide that. MEASURED on a 60x2 seasonal series, multiplying
the input by 100 and dividing the forecast change back by 100:

  Kalman  n_iter=1  rel. change 0.012 / 2.03      ARIMA  0.0005 / 0.32
  Kalman  n_iter=5  rel. change 0.003 / 0.41
  Kalman  n_iter=25 rel. change 0.021 / 0.91
  Kalman  n_iter=100 rel. change 0.084 / 0.62

More EM iterations do not close the gap, so this is not "5 is too few":
pykalman's EM starts from identity covariances, which mean something
different relative to data scaled by 100, and it settles into a different
optimum. `hypertools.predict` documents no scaling requirement.

So the invariance that CAN be asserted is the one this amendment is about:
the SCORING layer is invariant to a city's units. `tests/test_weather_
forecast_study.py` asserts exactly that, on the aggregation and on the
exactly-equivariant baselines, and states the model measurement above as
the reason it stops there.
"""
import datetime
import email.utils
import json
import os
import random
import sys
import tempfile
import time
import urllib.error
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


#: waits, in seconds, before retry attempts 2..7 of a throttled request.
#: open-meteo 429s six back-to-back archive requests on the last three,
#: MEASURED, and the last of those still 429'd after a 1+2+4+8s schedule --
#: it cleared after roughly another 30s. This schedule therefore spans 62s
#: before giving up, and a `Retry-After` header overrides it whenever the
#: server states its own number.
_RETRY_WAITS = (2, 4, 8, 16, 32)

#: pause between cities that actually reach the network, so six requests are
#: not fired back to back in the first place. Cached cities skip it.
_PACING_SECONDS = 1.5


def _get_with_retries(req, name, waits=_RETRY_WAITS, sleep=time.sleep,
                      opener=urllib.request.urlopen):
    """Fetch `req`, retrying a throttle instead of calling it "offline".

    Retries only 429 and 5xx -- a 404 or a 400 is a bug in the URL and
    repeating it is pointless. Honours `Retry-After` when the server sends
    one (in seconds or as an HTTP date), and jitters the fallback waits so
    six cities that were throttled together do not all come back at the
    same instant and throttle each other again.

    `sleep` and `opener` are seams for the tests: the real schedule spans
    62 seconds, which no test should actually wait through, and a builder
    that is not allowed to bind a socket still has to be able to check the
    retry policy. Both default to the real thing.
    """
    for attempt, wait in enumerate((*waits, None)):
        try:
            with opener(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or wait is None:
                raise
            stated = _retry_after_seconds(exc.headers.get('Retry-After'))
            delay = stated if stated is not None else wait * random.uniform(
                0.8, 1.3)
            print(f'  . {name}: HTTP {exc.code}, retrying in {delay:.1f}s'
                  f'{" (Retry-After)" if stated is not None else ""}'
                  f' [attempt {attempt + 1}/{len(waits) + 1}]',
                  file=sys.stderr)
            sleep(delay)
    raise AssertionError('unreachable: the last attempt re-raises')


def _retry_after_seconds(header):
    """`Retry-After` as a float, or None. Accepts both legal spellings."""
    if not header:
        return None
    try:
        return max(0.0, float(header.strip()))
    except ValueError:
        pass
    try:
        # raises on anything unparseable (it does NOT return None, which is
        # what this code assumed until a test fed it "not a number")
        stamp = email.utils.parsedate_to_datetime(header.strip())
    except (TypeError, ValueError):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=datetime.timezone.utc)
    return max(0.0, (stamp - datetime.datetime.now(datetime.timezone.utc)
                     ).total_seconds())


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
            data = _get_with_retries(req, name)
            tmp = dest + '.part'
            with open(tmp, 'wb') as f:
                f.write(data)
            os.replace(tmp, dest)
            # this city cost a network request; pace the next one rather
            # than racing it into the same rate-limit window
            time.sleep(_PACING_SECONDS)
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

AGGREGATIONS = ('pooled_scaled', 'fisher_z', 'pooled_raw')
HEADLINE = 'pooled_scaled'


def unit_scale(series, calibration_end):
    """One positive units constant per measure for one city.

    The standard deviation of that city's month-over-month changes over
    `series[:calibration_end]` -- data that is entirely BEFORE the first
    evaluation anchor. The cutoff is the point of the parameter, and it is
    required rather than defaulted: computing this over the whole series
    (as the first corrected version did) uses evaluation-period volatility
    to decide how loudly each city votes in the pooled score, which is
    held-out outcome data selecting the weights on the outcomes. The scale
    never reaches a model either way; that is not what makes it leakage.

    ONE scale per city serves BOTH blocks, so the two blocks are also
    weighted the same way as each other.

    A measure with no variation in the calibration window yields nan, and
    that city is dropped from that measure's pooling rather than dividing
    by zero.
    """
    window = np.asarray(series, dtype=float)[:calibration_end]
    if len(window) < 2:
        return np.full(np.shape(series)[1], np.nan)
    spread = np.std(np.diff(window, axis=0), axis=0)
    return np.where(np.isfinite(spread) & (spread > 0), spread, np.nan)


def _corr_columns(pred, real):
    """Per-column Pearson correlation, nan where it is not defined.

    nan-tolerant on purpose: `zero` predicts a constant, so its column has
    no variance and its correlation genuinely does not exist -- and one
    city dropped for one measure must not take the other five with it.
    """
    pred, real = np.asarray(pred, dtype=float), np.asarray(real, dtype=float)
    out = []
    for j in range(real.shape[1]):
        ok = np.isfinite(pred[:, j]) & np.isfinite(real[:, j])
        if ok.sum() < 3 or np.std(pred[ok, j]) == 0 or np.std(real[ok, j]) == 0:
            out.append(float('nan'))
        else:
            out.append(float(np.corrcoef(pred[ok, j], real[ok, j])[0, 1]))
    return out


def _fisher_z_mean(per_city):
    """Correlate WITHIN each city, then average with Fisher's z.

    `per_city` is a list of (correlations, n) pairs, one per city. Weighting
    by (n - 3) is the usual variance weighting for z. Cities are never
    pooled in the same units here at all, so this agrees with
    `pooled_scaled` only if the answer does not depend on how the pooling
    was done -- which is exactly what it is here to check.
    """
    n_measures = len(per_city[0][0])
    means, spreads = [], []
    for j in range(n_measures):
        z, w, raw = [], [], []
        for corrs, n in per_city:
            c = corrs[j]
            if np.isfinite(c) and n > 3:
                z.append(np.arctanh(np.clip(c, -0.999999, 0.999999)))
                w.append(n - 3)
                raw.append(c)
        if not z:
            means.append(float('nan'))
            spreads.append((float('nan'), float('nan')))
            continue
        means.append(float(np.tanh(np.average(z, weights=w))))
        spreads.append((float(min(raw)), float(max(raw))))
    return means, spreads


def _aggregate(per_city_pred, per_city_real, scales):
    """The three aggregations of one competitor's (city -> arrays) scores."""
    scaled_pred = [p / s for p, s in zip(per_city_pred, scales)]
    scaled_real = [r / s for r, s in zip(per_city_real, scales)]
    per_city = [(_corr_columns(p, r), len(r))
                for p, r in zip(per_city_pred, per_city_real)]
    fisher, spread = _fisher_z_mean(per_city)
    return {
        'pooled_scaled': _corr_columns(np.concatenate(scaled_pred),
                                       np.concatenate(scaled_real)),
        'fisher_z': fisher,
        'pooled_raw': _corr_columns(np.concatenate(per_city_pred),
                                    np.concatenate(per_city_real)),
        'per_city': [corrs for corrs, _ in per_city],
        'spread': spread,
    }


def evaluate(arrays, model, horizon, anchors, block, calibration_end):
    """Predicted-vs-realised correlation per measure, under each aggregation.

    Scores are accumulated PER CITY and pooled afterwards, because how the
    cities are combined turned out to be a load-bearing choice -- see the
    2026-08-19 amendment at the top of this file. `calibration_end` is the
    index the per-city units are measured up to; it must be at or before
    the first anchor in `anchors`, and it is asserted rather than trusted.
    """
    if calibration_end > min(anchors):
        raise ValueError(
            f'calibration_end={calibration_end} reaches past the first '
            f'anchor ({min(anchors)}): the units would be measured on data '
            f'the score is taken over')
    pred_by_city, real_by_city, scales = [], [], []
    base_by_city = {name: [] for name in BASELINES}
    t0 = time.time()
    for series in arrays:
        pred, real = [], []
        trivials = {name: [] for name in BASELINES}
        for anchor in anchors:
            if anchor + horizon - 1 >= len(series) or anchor < 2 * SEASON:
                continue
            history = series[:anchor]
            pred.append(_predict_delta(history, model, horizon))
            real.append(series[anchor + horizon - 1] - history[-1])
            trivial = _baseline_deltas(history, horizon)
            trivial.update(seasonal_deltas(series, anchor, horizon))
            for name in BASELINES:
                trivials[name].append(trivial[name])
        if not pred:
            continue
        # the scale is appended HERE, beside the arrays it belongs to, so a
        # city that contributed no anchors cannot shift every later city's
        # units by one position
        scales.append(unit_scale(series, calibration_end))
        pred_by_city.append(np.array(pred))
        real_by_city.append(np.array(real))
        for name in BASELINES:
            base_by_city[name].append(np.array(trivials[name]))
    elapsed = time.time() - t0
    if not pred_by_city:
        return None
    return {'model': model, 'horizon': horizon, 'block': block,
            'n': sum(len(p) for p in pred_by_city), 'seconds': elapsed,
            'cities': len(pred_by_city),
            'scores': _aggregate(pred_by_city, real_by_city, scales),
            'baselines': {name: _aggregate(vals, real_by_city, scales)
                          for name, vals in base_by_city.items()}}


def best_baseline(row, measure_index, aggregation=HEADLINE):
    """The STRONGEST trivial competitor for one measure, one aggregation.

    `zero` has no variance, so its correlation is nan by construction --
    take the max over the FINITE ones only, or a single nan becomes "the
    baseline" and every model wins by default.
    """
    finite = [row['baselines'][name][aggregation][measure_index]
              for name in BASELINES
              if np.isfinite(row['baselines'][name][aggregation][measure_index])]
    return max(finite) if finite else float('nan')


def verdict(rows, drawn_horizon=DRAWN_HORIZON, aggregation=HEADLINE):
    """Apply the preregistered rule mechanically, and report what survived."""
    survivors, considered = [], {}
    for row in rows:
        if row is None or row['horizon'] != drawn_horizon:
            continue
        for j, measure in enumerate(MEASURES):
            key = (row['model'], measure)
            considered.setdefault(key, []).append(
                (row['block'], row['scores'][aggregation][j],
                 best_baseline(row, j, aggregation)))
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

    # the per-city units are measured on everything BEFORE the first
    # evaluation anchor, and the same scale then serves both blocks
    calibration_end = lo
    print(f'city units calibrated on months [0, {calibration_end}) -- '
          f'{calibration_end - 1} changes, all before the first anchor')

    rows = []
    for model in ('Kalman', 'ARIMA'):
        for block, anchors in blocks.items():
            row = evaluate(arrays, model, DRAWN_HORIZON, anchors, block,
                           calibration_end)
            rows.append(row)
            if row is not None:
                report_block(row)

    print('\n' + '=' * 70)
    outcomes = {}
    for aggregation in AGGREGATIONS:
        survivors = verdict(rows, aggregation=aggregation)
        outcomes[aggregation] = survivors
        label = f'{aggregation}{"  (HEADLINE)" if aggregation == HEADLINE else ""}'
        if not survivors:
            print(f'{label:28s} nothing survives at t=1')
            continue
        print(f'{label:28s} {len(survivors)} specification(s) survive:')
        for (model, measure), entries in survivors:
            for block, score, base in sorted(entries):
                print(f'      {model} / {measure} / {block}: '
                      f'r={score:+.3f} vs baseline {base:+.3f}')

    print('=' * 70)
    if not outcomes[HEADLINE]:
        print('NOTHING SURVIVES the preregistered rule at t=1.')
        print('Weather does not earn a forecast claim either.')
    else:
        print(f'{len(outcomes[HEADLINE])} specification(s) survive at t=1 '
              f'under {HEADLINE}.')
    disagree = [a for a in AGGREGATIONS
                if bool(outcomes[a]) != bool(outcomes[HEADLINE])]
    if disagree:
        print(f'NOTE: {", ".join(disagree)} disagree(s) with {HEADLINE}. A '
              'claim that survives\nonly under some aggregations is a NEW '
              'claim and does not inherit this\npreregistration.')
    return 0


def report_block(row):
    """Print one model-block cell: every aggregation, side by side."""
    print(f'\n{row["model"]:8s} {row["block"]}  {row["cities"]} cities  '
          f'n={row["n"]:4d}  {row["seconds"]:.1f}s')
    for j, measure in enumerate(MEASURES):
        for k, aggregation in enumerate(AGGREGATIONS):
            score = row['scores'][aggregation][j]
            base = best_baseline(row, j, aggregation)
            beats = (np.isfinite(score) and np.isfinite(base)
                     and score > max(0.0, base))
            # the margin is printed, not just the verdict: a cell that
            # reads "+0.296 vs +0.296 BEATS" at three decimals is a
            # near-tie, and rounding it away would look like a broken rule
            mark = f' BEATS by {score - max(0.0, base):.4f}' if beats else ''
            label = measure if k == 0 else ''
            print(f'   {label:14s} {aggregation}: r={score:+.3f} vs '
                  f'{base:+.3f} [{_winning_baseline(row, j, aggregation)}]'
                  f'{mark}')
        lo, hi = row['scores']['spread'][j]
        print(f'   {"":14s} {"":13s}  per-city r in [{lo:+.2f}, {hi:+.2f}]')


def _winning_baseline(row, measure_index, aggregation):
    """WHICH trivial competitor is the one to beat, for this aggregation.

    Named per aggregation rather than once per cell: the winner is not
    always the same one. Under the calibrated headline, `seasonal_naive`
    takes windspeed in block 2 while `climatology` takes the other seven
    cells, and a single label would have quietly reported the wrong
    baseline for that cell.
    """
    finite = [(row['baselines'][name][aggregation][measure_index], name)
              for name in BASELINES
              if np.isfinite(row['baselines'][name][aggregation][measure_index])]
    return max(finite)[1] if finite else 'none'


if __name__ == '__main__':
    raise SystemExit(main())
