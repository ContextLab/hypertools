# -*- coding: utf-8 -*-
"""Preregistered comparison behind the Market gallery example's design.

NOT gallery code, and deliberately not imported by anything that ships: this
is the diagnostic that decides what the Market example should plot and
whether it may claim a forecast result at all. It exists in the repo because
the decision it supports is recorded in the plan and in the CHANGELOG, and a
decision whose evidence lives only in a chat transcript cannot be re-checked.

WHAT IT COMPARES (fixed before any result was inspected -- that is the point
of writing it down):

  representations
    D1  six SECTOR trajectories: cumulative return / drawdown / realized
        volatility, equal-weight within each sector
    D2  three sectors x two stocks, the same three measures PER STOCK, as a
        (Market, Sector, Ticker, Measure) column hierarchy -- 6 leaves,
        3 sector means, 1 market mean
    (a PCA representation is deliberately NOT included: `Pipeline.inverse_
    transform` cannot round-trip a hypertools stage pipeline today --
    `_DispatchStep` has no `inverse_transform` -- so scoring in inverted
    coordinates would require reaching into private fitted state.)

  models     Kalman, AutoRegressor, Laplace
  horizons   1 and 3 monthly steps
  blocks     two NON-OVERLAPPING halves of the sample, so a specification
             cannot be chosen on the same data it is then scored on

  metrics    per-measure Pearson correlation between PREDICTED and REALISED
             change (never pooled across measures in raw units -- the three
             measures differ in scale, and pooling them lets one dominate);
             Spearman cross-sectional correlation per date; the same
             correlation for four trivial baselines; wall clock; and, for the
             visual half, path roughness and cube occupancy.

  baselines  zero change, persistence (repeat the last change), historical
             mean change, exponentially-weighted continuation

ACCEPTANCE RULE, written before the numbers existed: a forecast claim
survives only if it beats every baseline on the same measure, keeps the same
sign in BOTH time blocks, and does so at a horizon the example actually
draws. Otherwise the example shows forecasts without scoring them.

    .venv/bin/python scripts/market_representation_study.py            # both halves
    .venv/bin/python scripts/market_representation_study.py --quick    # D1 only, Kalman only
"""

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hypertools as hyp                                          # noqa: E402

# D2's universe: fixed by a STRUCTURAL rule (the two largest US listings in
# each of three sectors, by market capitalisation, as of the plan's writing),
# chosen before any forecast was scored. Picking constituents by how well
# they forecast would make the gallery a cherry-picked backtest.
D2_SECTORS = {
    'Technology': ['AAPL', 'MSFT'],
    'Financials': ['JPM', 'BAC'],
    'Energy': ['XOM', 'CVX'],
}
DRAWN_HORIZON = 1       # steps -- the example draws predict=..., t=1
#: the blocks a claim must survive, by NAME. Checked as an exact set, not a
#: count: two rows both labelled 'block1' are one block scored twice, and
#: 'block1' plus a typo'd 'blokc2' is a claim tested on half the sample.
EXPECTED_BLOCKS = ('block1', 'block2')
VOL_WINDOW = 6          # months
CUM_WINDOW = 12         # months -- "cumulative return over a fixed recent window"
DD_WINDOW = 24          # months -- running peak for the drawdown
MEASURES = ('cum_return', 'drawdown', 'volatility')


#: The universe and the fetcher are LIFTED from the gallery example rather
#: than imported from it. The study is the evidence that decides what the
#: example should BE, so it must not depend on the example existing in any
#: particular form -- when the artifact was discarded pending a redesign,
#: an `import` here took the evidence down with it. Reproducing a recorded
#: result cannot require the artifact the result was used to judge.
CACHE = os.path.join(tempfile.gettempdir(), 'hypertools_gallery_cache')
RANGE = '10y'
#: six sectors x four EQUALLY WEIGHTED tickers each
SECTORS = {
    'Technology': ['AAPL', 'MSFT', 'ORCL', 'IBM'],
    'Financials': ['JPM', 'BAC', 'GS', 'AXP'],
    'Healthcare': ['JNJ', 'PFE', 'MRK', 'ABT'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB'],
    'Consumer': ['KO', 'PG', 'WMT', 'MCD'],
    'Industrials': ['BA', 'CAT', 'GE', 'HON'],
}


def fetch_closes(sectors=SECTORS):
    """Adjusted daily closes per (sector, ticker), or ``None`` if anything
    (network, parsing) goes wrong. Adjusted, not raw: a split would
    otherwise read as a -50% day."""
    # the offline check is OUTSIDE the try, so it raises instead of being
    # caught and quietly downgraded to the synthetic fallback: a test that
    # sets HYPERTOOLS_OFFLINE is asserting that no fetch happened, and a
    # swallowed exception would make a real fetch look the same as a refused
    # one. `load_market` catches it and degrades; nothing else calls this.
    if os.environ.get('HYPERTOOLS_OFFLINE'):
        raise RuntimeError('HYPERTOOLS_OFFLINE is set: refusing to fetch')
    os.makedirs(CACHE, exist_ok=True)
    try:
        series = {}
        for sector, tickers in sectors.items():
            for ticker in tickers:
                dest = os.path.join(CACHE, f'yahoo_adj_{ticker}_{RANGE}.json')
                if not os.path.exists(dest):
                    url = ('https://query1.finance.yahoo.com/v8/finance/chart/'
                           f'{ticker}?range={RANGE}&interval=1d')
                    req = urllib.request.Request(
                        url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        payload = resp.read()
                    with open(dest + '.part', 'wb') as f:
                        f.write(payload)
                    # rename, so an interrupted download can never leave a
                    # truncated cache file that every later run would trust
                    os.replace(dest + '.part', dest)
                with open(dest) as f:
                    result = json.load(f)['chart']['result'][0]
                stamps = pd.to_datetime(result['timestamp'], unit='s')
                series[(sector, ticker)] = pd.Series(
                    result['indicators']['adjclose'][0]['adjclose'],
                    index=stamps.normalize()).astype(float)
        # NO ffill: a stale ticker would be padded flat and then contribute
        # exactly zero return while its peers moved, silently breaking the
        # equal-weight claim. Dropping the row instead makes it visible.
        return pd.DataFrame(series).sort_index().dropna()
    except Exception:
        return None


def synthetic_closes(sectors=SECTORS, days=2500, seed=0):
    """Same sector structure -- the REQUESTED one -- so the figure renders
    offline. Takes `sectors` for the same reason `fetch_closes` does: a
    caller who asks for three sectors must not silently get six back when
    the network is down."""
    rng = np.random.default_rng(seed)
    index = pd.date_range('2016-08-15', periods=days, freq='B')
    columns = [(sector, t) for sector, ts in sectors.items() for t in ts]
    drift = rng.normal(0.0003, 0.0002, size=(1, len(columns)))
    steps = rng.normal(0, 0.013, size=(days, len(columns))) + drift
    return pd.DataFrame(100.0 * np.exp(steps.cumsum(axis=0)),
                        index=index, columns=pd.MultiIndex.from_tuples(columns))


def monthly_levels(closes):
    """Month-end log price levels. A decimation, not an average: no future
    observation reaches backwards into a bar."""
    return np.log(closes.resample('ME').last().dropna())


def level_features(levels):
    """The three LEVEL-LIKE measures, all backward-looking.

    `cum_return` is the trailing CUM_WINDOW-month log return, `drawdown` the
    log distance below the trailing DD_WINDOW-month peak (<= 0), and
    `volatility` the realized standard deviation of monthly log returns over
    VOL_WINDOW months. Unlike return/volatility/momentum these persist --
    which is the whole point: a state that reverses every step draws a knot.
    """
    rets = levels.diff()
    return pd.DataFrame({
        'cum_return': levels - levels.shift(CUM_WINDOW),
        'drawdown': levels - levels.rolling(DD_WINDOW, min_periods=1).max(),
        'volatility': rets.rolling(VOL_WINDOW).std(),
    })


def d1_frame(closes, sectors):
    """Six sector trajectories: (Market, Sector, Measure)."""
    levels = monthly_levels(closes)
    columns = {}
    for sector in sectors:
        # equal weight WITHIN the sector, on log levels, so the sector's
        # level is the mean of its constituents' levels
        sector_level = levels[sector].mean(axis=1)
        feats = level_features(sector_level)
        for measure in MEASURES:
            columns[('Market', sector, measure)] = feats[measure]
    frame = pd.DataFrame(columns).dropna()
    frame.columns = pd.MultiIndex.from_tuples(
        frame.columns, names=['Market', 'Sector', 'Measure'])
    return frame


def d2_frame(closes, sectors):
    """Six stock leaves under three sectors: (Market, Sector, Ticker, Measure)."""
    levels = monthly_levels(closes)
    columns = {}
    for sector, tickers in sectors.items():
        for ticker in tickers:
            feats = level_features(levels[(sector, ticker)])
            for measure in MEASURES:
                columns[('Market', sector, ticker, measure)] = feats[measure]
    frame = pd.DataFrame(columns).dropna()
    frame.columns = pd.MultiIndex.from_tuples(
        frame.columns, names=['Market', 'Sector', 'Ticker', 'Measure'])
    return frame


def leaf_arrays(frame):
    """One (T, 3) array per LEAF, in column order.

    Positional, not `.loc[:, key]`: these frames are not lexsorted, and label
    indexing into them emits a pandas PerformanceWarning -- which this repo
    treats as a failure rather than as noise.
    """
    groups = {}
    for position, column in enumerate(frame.columns):
        groups.setdefault(tuple(column[:-1]), []).append(position)
    return [frame.iloc[:, positions].to_numpy(dtype=float)
            for positions in groups.values()]


def scale_per_measure(frame):
    """One positive constant per MEASURE, pooled over every leaf.

    The display applies a single shared gain to all three axes, so measures
    that differ in spread collapse the picture. Diagonal and positive, so it
    changes no sign and no ratio within a measure -- and it inverts exactly,
    which is what lets a forecast be scored in the original units.
    """
    spread = {m: frame.xs(m, axis=1, level='Measure').to_numpy().std()
              for m in MEASURES}
    scaled = frame.div([spread[c[-1]] for c in frame.columns], axis=1)
    return scaled, spread


def roughness(arrays):
    """Path length divided by bounding span. Lower is more path-like."""
    out = []
    for a in arrays:
        length = float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum())
        span = float(np.linalg.norm(a.max(axis=0) - a.min(axis=0)))
        out.append(length / span if span else np.inf)
    return out


def occupancy(arrays):
    """Fraction of each axis the pooled data fills, after the shared
    center/scale a hypertools figure applies (which maps the widest axis
    onto [-1, 1])."""
    pooled = np.vstack(arrays)
    centered = pooled - pooled.mean(axis=0)
    gain = np.abs(centered).max()
    if not gain:
        return [0.0, 0.0, 0.0]
    unit = centered / gain
    return list(unit.max(axis=0) - unit.min(axis=0))


def _predict_delta(history, model, horizon):
    """(predicted change over `horizon`, from the last observed state)."""
    forecast = np.asarray(hyp.predict(history, model=model, t=horizon),
                          dtype=float)
    return forecast[horizon - 1] - history[-1]


def _baseline_deltas(history, horizon):
    """The four trivial predictions, in the same units as `_predict_delta`."""
    steps = np.diff(history, axis=0)
    ew = np.zeros(history.shape[1])
    if len(steps):
        weights = 0.7 ** np.arange(len(steps))[::-1]
        ew = (steps * weights[:, None]).sum(axis=0) / weights.sum()
    return {
        'zero': np.zeros(history.shape[1]),
        'persistence': steps[-1] * horizon if len(steps) else np.zeros(history.shape[1]),
        'mean_change': steps.mean(axis=0) * horizon if len(steps) else np.zeros(history.shape[1]),
        'ew_continuation': ew * horizon,
    }


def leaf_levels(closes, sectors, kind):
    """The log LEVEL series behind each leaf, in the frame's column order.

    `window_dropout` needs the series a leaf's features were built FROM, and
    a D1 leaf is a whole sector -- the equal-weight mean of its constituents'
    log levels -- so it cannot be recovered by indexing `closes` with the
    leaf key. Built by the same expressions `d1_frame`/`d2_frame` use, in the
    same iteration order, so position i here is leaf i there.
    """
    levels = monthly_levels(closes)
    if kind == 'D1':
        return [levels[sector].mean(axis=1) for sector in sectors]
    return [levels[(sector, ticker)]
            for sector, tickers in sectors.items() for ticker in tickers]


def window_dropout(leaf_series, index, spread, horizon):
    """The part of the `cum_return` change over `horizon` already known at t.

    `cum_return[t] = L[t] - L[t - CUM_WINDOW]`, so its change over h steps is

        cum[t+h] - cum[t] = (L[t+h] - L[t]) - (L[t+h-12] - L[t-12])

    and the second term is a span of h returns that all ENDED on or before
    t (for h <= CUM_WINDOW): it is known exactly, with no forecast at all.
    Measured at h=1 on the live data, that known term carries 37-52% of the
    change's variance and correlates 0.57-0.74 with it.

    This is a BASELINE, not a nuisance to be removed -- and the four trivial
    baselines cannot see it, which is why every one of them scored NEGATIVE
    while any model that had seen the history scored positive. A comparison
    that omits it flatters the model.

    HORIZON-AWARE. It used to return the one-step term `-rets.shift(12)` for
    every horizon. At h=3 three known returns leave the window, not one, so
    the baseline was scored on a third of the information actually available
    to it -- which understates it exactly where the model has most room to
    look good. Written as `-(L - L.shift(h)).shift(CUM_WINDOW + 1 - h)`,
    which collapses to the old expression at h=1 and generalises correctly.

    `evaluate` reads position `anchor`, one past the last observed row, so
    the extra +1 in the shift is that offset and not an adjustment to the
    identity above.
    """
    if not 1 <= horizon <= CUM_WINDOW:
        raise ValueError(
            f"the dropout identity holds for 1 <= horizon <= {CUM_WINDOW} "
            f"(beyond that the span leaving the window is not yet "
            f"observed); got horizon={horizon}")
    known = {}
    for i, level in enumerate(leaf_series):
        span = level - level.shift(horizon)         # L[t] - L[t-h]
        known[i] = (-span.shift(CUM_WINDOW + 1 - horizon).reindex(index)
                    / spread['cum_return']).to_numpy()
    return known


def evaluate(arrays, model, horizon, anchors, label, mechanical=None,
             representation=None, block=None):
    """Pooled predicted-vs-realised correlation, PER MEASURE, plus baselines."""
    predicted, realised, baselines = [], [], {k: [] for k in
                                             ('zero', 'persistence',
                                              'mean_change', 'ew_continuation',
                                              'window_dropout')}
    per_date = []
    # The AUDIT baseline this file's notes committed to before the verdict
    # existed: drawdown is bounded above at 0 and recovers, so "predict full
    # recovery" (change = -drawdown[t]) is a parameter-free rule that needs
    # no model. Kept OUT of `baselines` on purpose -- the acceptance rule is
    # preregistered and adding a competitor after seeing who it would beat
    # is the move preregistration exists to prevent. Reported alongside.
    audit = []
    t0 = time.time()
    for anchor in anchors:
        date_pred, date_real = [], []
        for index_of, series in enumerate(arrays):
            if anchor + horizon >= len(series) or anchor < 12:
                continue
            history = series[:anchor]
            actual = series[anchor + horizon - 1] - history[-1]
            predicted.append(_predict_delta(history, model, horizon))
            realised.append(actual)
            trivial = _baseline_deltas(history, horizon)
            # the window-aware baseline predicts ONLY the known dropout, on
            # the cum_return axis, and nothing on the other two
            known = np.zeros(history.shape[1])
            if mechanical is not None:
                value = mechanical.get(index_of, [np.nan] * (anchor + 1))[anchor]
                known[0] = 0.0 if not np.isfinite(value) else value
            trivial['window_dropout'] = known
            recovery = np.zeros(history.shape[1])
            recovery[MEASURES.index('drawdown')] = -history[-1][
                MEASURES.index('drawdown')]
            audit.append(recovery)
            for name, value in trivial.items():
                baselines[name].append(value)
            date_pred.append(predicted[-1])
            date_real.append(actual)
        if len(date_pred) > 2:
            per_date.append((np.array(date_pred), np.array(date_real)))
    elapsed = time.time() - t0
    if not predicted:
        return None
    predicted, realised = np.array(predicted), np.array(realised)

    def _pearson(pred):
        return [float(np.corrcoef(pred[:, j], realised[:, j])[0, 1])
                if np.std(pred[:, j]) > 0 and np.std(realised[:, j]) > 0
                else float('nan') for j in range(realised.shape[1])]

    # Spearman ACROSS TRACES within each date, then averaged: "did it rank
    # this month's movers correctly", which is a different question from the
    # pooled one and is not answered by it.
    spearman = []
    for pred, real in per_date:
        for j in range(real.shape[1]):
            if len(pred) > 2:
                rp = pd.Series(pred[:, j]).rank().to_numpy()
                rr = pd.Series(real[:, j]).rank().to_numpy()
                if np.std(rp) > 0 and np.std(rr) > 0:
                    spearman.append((j, float(np.corrcoef(rp, rr)[0, 1])))
    cross = [float(np.mean([v for k, v in spearman if k == j]))
             if any(k == j for k, _ in spearman) else float('nan')
             for j in range(realised.shape[1])]

    return {
        'label': label, 'model': model, 'horizon': horizon,
        'representation': representation, 'block': block,
        'n': len(predicted), 'seconds': elapsed,
        'pearson': _pearson(predicted), 'spearman': cross,
        'baselines': {name: _pearson(np.array(vals))
                      for name, vals in baselines.items()},
        'audit_reversion': _pearson(np.array(audit)),
    }


def _best_baselines(row):
    """The STRONGEST trivial competitor per measure.

    `zero` has no variance so its correlation is nan by construction --
    nanmax, or a single nan would silently become "the baseline" and every
    real model would look like it wins.
    """
    best = {}
    for j, measure in enumerate(MEASURES):
        finite = [v for v in (row['baselines'][b][j] for b in row['baselines'])
                  if np.isfinite(v)]
        best[measure] = max(finite) if finite else float('nan')
    return best


def verdict(rows, drawn_horizon=DRAWN_HORIZON):
    """Apply this module's ACCEPTANCE RULE mechanically, and say so.

    The rule, quoted from the top of this file: a forecast claim survives
    only if it beats every baseline on the same measure, keeps the same sign
    in BOTH time blocks, and does so at a horizon the example actually
    draws. Evaluated in code rather than read off the table by eye, because
    a rule applied by inspection is a rule that can be applied leniently.

    Two clauses are read STRICTLY, because reading them loosely admits
    claims the rule plainly does not intend:

    * "keeps the same sign" is read as POSITIVE, and the comparison is
      `score > max(0, best_baseline)`. Taken literally an all-NEGATIVE set
      keeps its sign too, so a model correlating -0.10 with the outcome
      passed against a -0.50 baseline -- consistently wrong, and merely
      less wrong than the trivial competition. That supports no forecast
      claim at all, and it is live rather than hypothetical here: every
      trivial baseline on `drawdown` is anti-correlated.
    * "in BOTH time blocks" is read as an EXACT block set. A count check
      accepted one block scored twice, or a renamed block, as coverage.
    """
    claims = {}
    for row in rows:
        if row is None:
            continue
        # "...and does so at a horizon the example actually draws". The
        # example forecasts one step (`predict='Kalman', t=1`), so a result
        # that only appears at h=3 describes a figure nobody is drawing.
        # This clause was in the rule from the start and was omitted from
        # the first mechanical pass, which reported five extra passes.
        if row['horizon'] != drawn_horizon:
            continue
        best = _best_baselines(row)
        for j, measure in enumerate(MEASURES):
            key = (row['representation'], row['model'], row['horizon'], measure)
            claims.setdefault(key, []).append(
                (row['block'], row['pearson'][j], best[measure]))

    print(f'\n--- acceptance rule (POSITIVE, beats every baseline, in each '
          f'of {len(EXPECTED_BLOCKS)} blocks, h={drawn_horizon} as drawn) ---')
    survivors = []
    for key in sorted(claims):
        entries = claims[key]
        # EXACT block set. `len(blocks) > 1` accepted a duplicated block, a
        # renamed one, or three rows covering two blocks -- none of which is
        # the out-of-sample test the rule describes.
        blocks = [block for block, _, _ in entries]
        covered = sorted(blocks) == sorted(EXPECTED_BLOCKS)
        # `score > max(0, base)`, not merely `score > base`. Every trivial
        # baseline on `drawdown` is anti-correlated, so "beats every
        # baseline" alone let a model at -0.10 pass against a baseline at
        # -0.50: consistently WRONG, merely less wrong than the trivial
        # competition, and no support for a positive forecast claim. The old
        # sign clause permitted it explicitly by accepting an all-negative
        # set as "the same sign".
        beats = all(np.isfinite(score) and np.isfinite(base)
                    and score > max(0.0, base)
                    for _, score, base in entries)
        if covered and beats:
            survivors.append((key, entries))
    if survivors:
        for key, entries in survivors:
            detail = ', '.join(f'{block} {score:+.3f} vs {base:+.3f}'
                               for block, score, base in entries)
            print(f'  PASS  {key[0]} {key[1]} h={key[2]} {key[3]}: {detail}')
    else:
        print(f'  NOTHING PASSES: 0 of {len(claims)} '
              f'(representation, model, horizon, measure) specifications '
              f'are POSITIVE and beat every baseline in both blocks.')
        print('  Under the rule as written, the example may not claim a '
              'forecast result.')
    return survivors


def reversion_diagnostic(rows):
    """POST-HOC, and printed apart from the rule for that reason.

    Every trivial baseline on the `drawdown` axis scores NEGATIVE -- -0.504
    at the extreme -- because drawdown is bounded above at zero and mean
    reverts: persistence predicts "keep falling" exactly when a recovery is
    most likely. "Beats every baseline" is therefore a low bar on that axis,
    and unlike `cum_return` it has no mechanical baseline at all (the
    trailing max is not linear, so there is no dropout identity to compute).

    A model can clear the preregistered bar on drawdown while carrying no
    information a one-line mean-reversion rule does not already have. This
    reports how far above that rule the survivors actually sit. It is NOT
    folded into the rule: adding a baseline after seeing which results it
    would kill is the move preregistration exists to prevent. It is
    evidence for a decision, and the decision is the maintainer's.
    """
    print('\n--- drawdown audit (NOT part of the acceptance rule) ---')
    print('  Every trivial baseline on `drawdown` is anti-correlated with '
          'the realised change, so')
    print('  clearing them there is a weaker result than clearing them on '
          '`cum_return`. The')
    print('  parameter-free comparison is "predict full recovery": change '
          '= -drawdown[t].')
    axis = MEASURES.index('drawdown')
    for row in rows:
        if row is None or row['horizon'] != DRAWN_HORIZON:
            continue
        best = _best_baselines(row)['drawdown']
        reversion = row['audit_reversion'][axis]
        beaten = ('BEATS' if row['pearson'][axis] > reversion else 'LOSES TO')
        print(f'  {row["label"]:26s} {row["model"]:14s} drawdown '
              f'{row["pearson"][axis]:+.3f}  trivial baselines '
              f'{best:+.3f}  full-recovery rule {reversion:+.3f}  '
              f'-> {beaten} the recovery rule')


def _report(rows):
    print(f'\n{"representation":22s} {"model":14s} {"h":>2s} {"n":>5s} '
          f'{"secs":>6s}  ' + '  '.join(f'{m:>12s}' for m in MEASURES))
    for r in rows:
        if r is None:
            continue
        print(f'{r["label"]:22s} {r["model"]:14s} {r["horizon"]:2d} '
              f'{r["n"]:5d} {r["seconds"]:6.1f}  '
              + '  '.join(f'{v:12.3f}' for v in r['pearson']))
        best = _best_baselines(r)
        print(f'{"":22s} {"  best baseline":14s} {"":2s} {"":5s} {"":6s}  '
              + '  '.join(f'{best[m]:12.3f}' for m in MEASURES))
        print(f'{"":22s} {"  spearman/date":14s} {"":2s} {"":5s} {"":6s}  '
              + '  '.join(f'{v:12.3f}' for v in r['spearman']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--anchors', type=int, default=14)
    opts = parser.parse_args()

    universe = dict(D2_SECTORS) if opts.quick else SECTORS
    closes = fetch_closes(universe)
    source = 'live/cached Yahoo'
    if closes is None:
        closes = synthetic_closes(universe)
        source = 'SYNTHETIC (network unavailable -- results are not evidence)'
    print(f'universe: {len(universe)} sectors, {closes.shape[1]} tickers '
          f'({source}); {closes.shape[0]} daily closes')

    frames = {'D1 sector levels': d1_frame(closes, universe)}
    if not opts.quick:
        frames['D2 stock hierarchy'] = d2_frame(closes, D2_SECTORS)

    print('\n--- geometry (scaled per measure, as plotted) ---')
    scaled_arrays, dropout_inputs = {}, {}
    for name, frame in frames.items():
        scaled, spread = scale_per_measure(frame)
        arrays = leaf_arrays(scaled)
        scaled_arrays[name] = arrays
        # BOTH representations need the window-aware baseline: a D1 sector's
        # cum_return is a trailing CUM_WINDOW sum of the sector level, so it
        # carries exactly the same mechanical dropout a D2 stock does.
        kind = 'D1' if name.startswith('D1') else 'D2'
        dropout_inputs[name] = (
            leaf_levels(closes, universe if kind == 'D1' else D2_SECTORS, kind),
            frame.index, spread)
        rough = roughness(arrays)
        print(f'{name:22s} {frame.shape[0]:4d} months x {len(arrays)} leaves  '
              f'roughness min/median/max {min(rough):5.1f}/'
              f'{float(np.median(rough)):5.1f}/{max(rough):5.1f}  '
              f'axis occupancy ' + '/'.join(f'{v:.2f}' for v in occupancy(arrays)))

    judged = []
    models = ['Kalman'] if opts.quick else ['Kalman', 'AutoRegressor', 'Laplace']
    horizons = [1] if opts.quick else [1, 3]
    for name, arrays in scaled_arrays.items():
        length = len(arrays[0])
        half = length // 2
        blocks = {'block1': (12, half), 'block2': (half, length - 4)}
        for block, (lo, hi) in blocks.items():
            anchors = np.linspace(max(lo, 14), hi - 1, opts.anchors).astype(int)
            series, index, spread = dropout_inputs[name]
            rows = [evaluate(arrays, model, horizon, anchors,
                             f'{name} {block}',
                             mechanical=window_dropout(series, index, spread,
                                                       horizon),
                             representation=name, block=block)
                    for model in models for horizon in horizons]
            _report(rows)
            judged.extend(rows)


    verdict(judged)
    reversion_diagnostic(judged)


if __name__ == '__main__':
    main()
