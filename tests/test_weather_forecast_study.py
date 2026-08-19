# -*- coding: utf-8 -*-
"""The weather study's rule must be as hard to fool as the Market study's.

`scripts/weather_forecast_study.py` decides whether the gallery may claim a
successful prediction. Two ways a study like that goes wrong, both of which
have already happened once in this plan:

1. a baseline that is quietly WRONG, so the model clears a bar that was
   never really there (the Market study's first `window_dropout` used a
   one-step value at every horizon);
2. an acceptance rule read LENIENTLY, so consistently-bad forecasts pass by
   being less bad than something worse.

These tests pin the arithmetic against directly computed values and then
attack the rule with results designed to slip through it.
"""
import numpy as np
import pytest

from scripts.weather_forecast_study import (SEASON, best_baseline,
                                            seasonal_deltas, verdict)


def _series(n=120, dims=2, seed=0):
    """A seasonal series with a trend -- the shape the study actually sees."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    out = []
    for d in range(dims):
        out.append(10 * np.sin(2 * np.pi * t / SEASON + d)
                   + 0.02 * t + rng.standard_normal(n))
    return np.column_stack(out)


@pytest.mark.parametrize('horizon', [1, 3])
def test_seasonal_naive_IS_last_years_change_over_the_same_span(horizon):
    """Computed independently of the implementation, from the definition.

    `seasonal_naive` must be the change observed between the same two
    calendar months one year earlier -- and nothing else. Written out at
    the anchor by hand rather than by re-running the function's own
    arithmetic, which is the only version of this test that can fail.
    """
    series = _series()
    anchor = 60
    got = seasonal_deltas(series, anchor, horizon)['seasonal_naive']
    start = series[anchor - 1 - SEASON]
    end = series[anchor + horizon - 1 - SEASON]
    assert np.allclose(got, end - start)


def test_seasonal_naive_uses_LAST_YEAR_not_last_month():
    """The mutation that would make the baseline trivially weak.

    A one-month offset instead of a twelve-month one turns `seasonal_naive`
    into persistence, which is already in the baseline set -- the seasonal
    baseline would then add nothing and every model would face a bar 11
    months too low. On a seasonal series the two are far apart, so this
    fails loudly if the offset is ever changed.
    """
    series = _series()
    anchor = 60
    seasonal = seasonal_deltas(series, anchor, 1)['seasonal_naive']
    one_month = series[anchor - 1] - series[anchor - 2]
    assert not np.allclose(seasonal, one_month, atol=0.5), (
        'the seasonal baseline collapsed onto persistence')


def test_climatology_TARGETS_the_month_and_measures_from_the_last_value():
    """It predicts a CHANGE, so it must be the target month's historical
    mean MINUS where the series actually is -- not the mean itself. Getting
    this wrong would put a level into a table of changes, where it would
    correlate with nothing and look like a weak baseline rather than a
    broken one."""
    series = _series()
    anchor, horizon = 60, 1
    got = seasonal_deltas(series, anchor, horizon)['climatology']
    target = anchor + horizon - 1
    months = np.arange(anchor)
    same = months[(months % SEASON) == (target % SEASON)]
    assert len(same) >= 4, 'test setup must contain several prior years'
    assert np.allclose(got, series[same].mean(axis=0) - series[anchor - 1])


def test_a_history_shorter_than_a_CYCLE_yields_no_seasonal_prediction():
    """Both seasonal baselines degrade to "predict no change" when there is
    no prior cycle to look back at. That is the right degradation, and it is
    also why `evaluate` refuses anchors below `2 * SEASON` -- a baseline
    silently reduced to zeros is a bar the model does not have to clear."""
    series = _series()
    got = seasonal_deltas(series, SEASON - 2, 1)
    assert np.allclose(got['seasonal_naive'], 0.0)


def test_evaluate_REFUSES_anchors_with_a_degraded_baseline():
    """Pins the guard itself, on the real function.

    Every anchor is below `2 * SEASON`, so nothing is scorable and
    `evaluate` must return None rather than quietly scoring a model against
    zeroed-out seasonal baselines.
    """
    from scripts.weather_forecast_study import evaluate
    arrays = [_series(n=40, seed=s) for s in range(2)]
    assert evaluate(arrays, 'Kalman', 1, [10, 15, 20], 'block1') is None


def test_best_baseline_SKIPS_the_nan_from_the_zero_baseline():
    """`zero` has no variance so its correlation is nan by construction. A
    plain `max` over the raw list returns nan, nan compares False against
    everything, and every model then "beats the best baseline"."""
    row = {'baselines': {'zero': [float('nan')], 'persistence': [0.4],
                         'mean_change': [0.1], 'ew_continuation': [0.2],
                         'seasonal_naive': [0.3], 'climatology': [0.9]}}
    assert best_baseline(row, 0) == pytest.approx(0.9)


def _row(model, block, scores, baselines, horizon=1):
    """A real result dict, in the shape `evaluate` returns."""
    from scripts.weather_forecast_study import BASELINES, MEASURES
    assert len(scores) == len(MEASURES)
    return {'model': model, 'block': block, 'horizon': horizon,
            'pearson': list(scores),
            'baselines': {name: list(baselines) for name in BASELINES}}


GOOD = [0.8, 0.1, 0.1, 0.1]
WEAK = [0.2, 0.9, 0.9, 0.9]


def test_a_model_that_beats_every_baseline_in_BOTH_blocks_passes():
    """The control. Without it, every refusal test below would pass just as
    well against a `verdict` that refused everything unconditionally."""
    rows = [_row('Kalman', 'block1', GOOD, WEAK),
            _row('Kalman', 'block2', GOOD, WEAK)]
    survivors = verdict(rows)
    assert [key for key, _ in survivors] == [('Kalman', 'temperature')]


def test_consistently_NEGATIVE_correlations_do_not_pass():
    """Being less wrong than something more wrong is not a prediction."""
    rows = [_row('Kalman', 'block1', [-0.2, -0.9, -0.9, -0.9], [-0.8] * 4),
            _row('Kalman', 'block2', [-0.3, -0.9, -0.9, -0.9], [-0.8] * 4)]
    assert verdict(rows) == []


def test_ONE_block_is_not_both_blocks():
    """The real weather result: Kalman beats climatology on precipitation in
    block 2 and loses in block 1. A rule that only checked "beats every
    baseline" would have accepted it."""
    rows = [_row('Kalman', 'block2', GOOD, WEAK)]
    assert verdict(rows) == []


def test_the_SAME_block_twice_is_not_two_blocks():
    """`len(blocks) > 1` would accept this. Requiring the exact expected set
    is what makes a duplicated row fail."""
    rows = [_row('Kalman', 'block1', GOOD, WEAK),
            _row('Kalman', 'block1', GOOD, WEAK)]
    assert verdict(rows) == []


def test_a_result_at_an_UNDRAWN_horizon_cannot_carry_the_claim():
    """The example draws t=1. A win at t=3 is a fact about a figure nobody
    is looking at."""
    rows = [_row('Kalman', 'block1', GOOD, WEAK, horizon=3),
            _row('Kalman', 'block2', GOOD, WEAK, horizon=3)]
    assert verdict(rows) == []


def test_the_LIVE_weather_numbers_are_refused_by_the_rule():
    """The measured 2026-08-19 result, pinned so the conclusion in
    *Revision note (v5)* cannot drift from the rule that produced it.

    Kalman beats climatology on precipitation (+0.651 vs +0.631) and
    windspeed (+0.468 vs +0.382) in block 2, and loses on both in block 1.
    Nothing survives -- and it must be the BOTH-BLOCKS clause that refuses
    it, not an accident of the numbers.
    """
    b1 = _row('Kalman', 'block1', [0.697, 0.440, 0.221, 0.238],
              [0.940, 0.729, 0.710, 0.773])
    b2 = _row('Kalman', 'block2', [0.903, 0.651, 0.583, 0.468],
              [0.935, 0.631, 0.656, 0.382])
    assert verdict([b1, b2]) == []
    # and block 2 alone really would have passed on two measures, which is
    # exactly what the both-blocks clause exists to catch
    assert best_baseline(b2, 1) < b2['pearson'][1]
    assert best_baseline(b2, 3) < b2['pearson'][3]
