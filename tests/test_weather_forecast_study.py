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
import datetime
import email.message
import email.utils
import http.server
import io
import socket
import threading
import time
import urllib.error
import urllib.request
import urllib.response

import numpy as np
import pytest

from scripts.weather_forecast_study import (AGGREGATIONS, HEADLINE, SEASON,
                                            best_baseline, seasonal_deltas,
                                            verdict)


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
    assert evaluate(arrays, 'Kalman', 1, [10, 15, 20], 'block1',
                    calibration_end=10) is None


def test_best_baseline_SKIPS_the_nan_from_the_zero_baseline():
    """`zero` has no variance so its correlation is nan by construction. A
    plain `max` over the raw list returns nan, nan compares False against
    everything, and every model then "beats the best baseline"."""
    row = {'baselines': {name: {a: [value] for a in AGGREGATIONS}
                         for name, value in [('zero', float('nan')),
                                             ('persistence', 0.4),
                                             ('mean_change', 0.1),
                                             ('ew_continuation', 0.2),
                                             ('seasonal_naive', 0.3),
                                             ('climatology', 0.9)]}}
    assert best_baseline(row, 0) == pytest.approx(0.9)


def _row(model, block, scores, baselines, horizon=1):
    """A real result dict, in the shape `evaluate` returns."""
    from scripts.weather_forecast_study import BASELINES, MEASURES
    assert len(scores) == len(MEASURES)
    # every aggregation carries the same numbers: these tests are about the
    # RULE, and a rule that behaved differently per aggregation would be a
    # separate defect from the one the aggregation tests below cover
    return {'model': model, 'block': block, 'horizon': horizon,
            'scores': {a: list(scores) for a in AGGREGATIONS},
            'baselines': {name: {a: list(baselines) for a in AGGREGATIONS}
                          for name in BASELINES}}


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

    These are the HEADLINE (`pooled_scaled`) numbers from the run with the
    city units calibrated before the first anchor, on 6 cities x 420 months
    of open-meteo archive. Kalman beats the best baseline on precipitation
    (+0.674 vs +0.660) and windspeed (+0.495 vs +0.296) in block 2, and
    loses on both in block 1. Nothing survives -- and it must be the
    BOTH-BLOCKS clause that refuses it, not an accident of the numbers.
    """
    b1 = _row('Kalman', 'block1', [0.539, 0.520, 0.201, 0.355],
              [0.921, 0.758, 0.719, 0.824])
    b2 = _row('Kalman', 'block2', [0.895, 0.674, 0.558, 0.495],
              [0.925, 0.660, 0.648, 0.296])
    assert verdict([b1, b2]) == []
    # and block 2 alone really would have passed on two measures, which is
    # exactly what the both-blocks clause exists to catch
    assert best_baseline(b2, 1) < b2['scores'][HEADLINE][1]
    assert best_baseline(b2, 3) < b2['scores'][HEADLINE][3]


def test_the_CORRECTED_pooling_did_not_rescue_any_specification():
    """The scale correction moved the numbers -- Kalman's block-1
    temperature fell from +0.697 (raw) to +0.539 (scaled), because the raw
    pooling had been letting the loudest cities speak for the rest -- but it
    moved no verdict. Pinned under all three aggregations so a later change
    to `HEADLINE` cannot quietly change the published conclusion.
    """
    from scripts.weather_forecast_study import AGGREGATIONS
    measured = {
        'pooled_scaled': ([0.539, 0.520, 0.201, 0.355],
                          [0.921, 0.758, 0.719, 0.824],
                          [0.895, 0.674, 0.558, 0.495],
                          [0.925, 0.660, 0.648, 0.296]),
        'fisher_z': ([0.713, 0.543, 0.345, 0.355],
                     [0.934, 0.768, 0.750, 0.818],
                     [0.900, 0.684, 0.629, 0.502],
                     [0.934, 0.674, 0.685, 0.433]),
        'pooled_raw': ([0.697, 0.440, 0.221, 0.238],
                       [0.940, 0.729, 0.710, 0.773],
                       [0.903, 0.651, 0.583, 0.468],
                       [0.935, 0.631, 0.656, 0.382]),
    }
    assert sorted(measured) == sorted(AGGREGATIONS)
    for aggregation, (s1, base1, s2, base2) in measured.items():
        rows = [_row('Kalman', 'block1', s1, base1),
                _row('Kalman', 'block2', s2, base2)]
        assert verdict(rows, aggregation=aggregation) == [], aggregation



# ---------------------------------------------------------------- pooling
# The 2026-08-19 amendment: cities were pooled in RAW units, so a city whose
# numbers happen to be large voted many times and a city whose numbers are
# small barely voted at all. These tests pin both halves of that -- that the
# defect is real, and that the correction removes it.

def _two_cities(loud_scale=1.0, seed=3):
    """Two cities, one predictable and one not, in whatever units are asked.

    City A: predictions ANTI-correlated with what happens.
    City B: predictions that track it closely.
    Nothing about which city is right depends on `loud_scale` -- that only
    changes A's units, which is exactly what must not matter.
    """
    rng = np.random.default_rng(seed)
    real_a = rng.standard_normal((40, 1))
    pred_a = -real_a + 0.1 * rng.standard_normal((40, 1))
    real_b = rng.standard_normal((40, 1))
    pred_b = real_b + 0.1 * rng.standard_normal((40, 1))
    return ([pred_a * loud_scale, pred_b], [real_a * loud_scale, real_b])


def test_pooled_raw_lets_ONE_LOUD_CITY_decide_the_answer():
    """The defect, demonstrated rather than described.

    One city is measured in units 100x the other's. Under raw pooling its
    covariance term is ~10,000x its neighbour's, so the pooled correlation
    is essentially that one city's -- here, strongly NEGATIVE, even though
    the other city is nearly perfectly predicted. Under the scale-free
    aggregations both cities vote and the answer lands between them.
    """
    from scripts.weather_forecast_study import _aggregate
    pred, real = _two_cities(loud_scale=100.0)
    scales = [np.std(np.diff(r, axis=0), axis=0) for r in real]
    got = _aggregate(pred, real, scales)
    assert got['pooled_raw'][0] < -0.5, 'the loud city should dominate'
    assert got['pooled_scaled'][0] > 0.0
    assert got['fisher_z'][0] > 0.0


def test_multiplying_ONE_CITY_by_100_leaves_the_scale_free_scores_UNCHANGED():
    """The invariance the review asked for, at the layer where it is real.

    Exactly unchanged, not approximately: the units constant is divided out
    of both sides of the same city's contribution.
    """
    from scripts.weather_forecast_study import _aggregate

    def aggregate(loud_scale):
        pred, real = _two_cities(loud_scale=loud_scale)
        scales = [np.std(np.diff(r, axis=0), axis=0) for r in real]
        return _aggregate(pred, real, scales)

    plain, loud = aggregate(1.0), aggregate(100.0)
    for aggregation in ('pooled_scaled', 'fisher_z'):
        assert np.allclose(plain[aggregation], loud[aggregation], atol=1e-12), (
            f'{aggregation} moved when one city changed units')
    assert not np.allclose(plain['pooled_raw'], loud['pooled_raw'], atol=1e-3)


def test_multiplying_ONE_CITY_by_100_leaves_the_VERDICT_unchanged():
    """The same invariance carried all the way to the accept/reject call.

    Scores AND baselines are re-aggregated from the rescaled data -- if the
    units leaked into either side, the two verdicts would differ.
    """
    from scripts.weather_forecast_study import (BASELINES, MEASURES,
                                                _aggregate)

    def rows_for(loud_scale):
        pred, real = _two_cities(loud_scale=loud_scale)
        # one measure per column of MEASURES, all the same series: this test
        # is about the units, not about the measures
        pred = [np.repeat(p, len(MEASURES), axis=1) for p in pred]
        real = [np.repeat(r, len(MEASURES), axis=1) for r in real]
        scales = [np.std(np.diff(r, axis=0), axis=0) for r in real]
        scores = _aggregate(pred, real, scales)
        # a genuinely weaker competitor: the same prediction with the sign
        # flipped, so it is anti-correlated rather than merely rescaled --
        # correlation is invariant to a positive rescale, so a "0.01x
        # baseline" would tie with the model to within float noise and the
        # comparison would be decided by rounding
        weak = _aggregate([-p for p in pred], real, scales)
        return [{'model': 'Kalman', 'horizon': 1, 'block': block,
                 'scores': scores,
                 'baselines': {name: weak for name in BASELINES}}
                for block in ('block1', 'block2')]

    plain = [key for key, _ in verdict(rows_for(1.0))]
    loud = [key for key, _ in verdict(rows_for(100.0))]
    assert plain, 'the control must ACCEPT, or the equality below is vacuous'
    assert plain == loud


def test_the_BASELINES_are_exactly_scale_equivariant():
    """Why the invariance above stops at the scoring layer.

    Every baseline is a linear function of the series, so scaling the
    series scales its prediction by the same factor -- to float precision.
    The shipped FORECASTERS are not: measured at x100 on a 60x2 seasonal
    series, Kalman's one-step change moves by 41% on one column and ARIMA's
    by 32%, and more EM iterations do not close it (see the module
    docstring). So an end-to-end "rescale a city, get the same verdict"
    test would be asserting something about pykalman's EM, not about this
    study's arithmetic.
    """
    from scripts.weather_forecast_study import BASELINES
    from scripts.market_representation_study import _baseline_deltas
    series = _series(n=60)
    for factor in (100.0, 0.01):
        plain = _baseline_deltas(series, 1)
        plain.update(seasonal_deltas(series, 48, 1))
        scaled = _baseline_deltas(series * factor, 1)
        scaled.update(seasonal_deltas(series * factor, 48, 1))
        for name in BASELINES:
            assert np.allclose(scaled[name] / factor, plain[name],
                               atol=1e-9), f'{name} is not scale-equivariant'


def test_unit_scale_REFUSES_a_flat_series_instead_of_dividing_by_zero():
    """A measure that never changes has no units to divide by. It must come
    back nan and be dropped from that measure's pooling -- not raise, and
    not silently become 1.0, which would put that city back into the pool
    at whatever scale it happened to have."""
    from scripts.weather_forecast_study import _corr_columns, unit_scale
    flat = np.column_stack([np.ones(30), _series(n=30)[:, 0]])
    scale = unit_scale(flat, len(flat))
    assert not np.isfinite(scale[0]) and np.isfinite(scale[1])
    dropped = _corr_columns(flat / scale, flat / scale)
    assert not np.isfinite(dropped[0]), 'the flat measure must drop out'
    assert np.isfinite(dropped[1]), 'and must not take the other one with it'


def test_fisher_z_weights_by_n_and_reports_the_per_city_spread():
    """Computed independently from the definition: average arctanh with
    weights (n - 3), then tanh back. A plain mean of correlations would give
    a different (and slightly biased) answer, so the two are compared."""
    from scripts.weather_forecast_study import _fisher_z_mean
    per_city = [([0.2], 10), ([0.9], 50)]
    means, spread = _fisher_z_mean(per_city)
    expected = np.tanh((np.arctanh(0.2) * 7 + np.arctanh(0.9) * 47) / 54)
    assert means[0] == pytest.approx(float(expected))
    assert means[0] != pytest.approx(float(np.mean([0.2, 0.9])))
    assert spread[0] == (0.2, 0.9)


# ------------------------------------------------------------------ retry
# The study's first run answered a question about real weather with
# fabricated weather, because open-meteo returned 429 and the fetcher
# treated a throttle as "offline". These run against a REAL local HTTP
# server that really returns 429 -- there is no way to check retry
# behaviour by inspecting the code, and a stubbed opener would only test
# the stub.

class _Throttler(http.server.BaseHTTPRequestHandler):
    """429s the first `fail_times` requests, then serves a body."""

    fail_times = 2
    retry_after = '0'
    status_after_failures = 200
    seen = 0

    def do_GET(self):
        type(self).seen += 1
        if type(self).seen <= type(self).fail_times:
            self.send_response(429)
            if type(self).retry_after is not None:
                self.send_header('Retry-After', type(self).retry_after)
            self.end_headers()
            self.wfile.write(b'slow down')
            return
        self.send_response(type(self).status_after_failures)
        self.end_headers()
        self.wfile.write(b'{"served": true}')

    def log_message(self, *args):
        """Silence: a passing test should not print an access log."""


def _can_bind_a_local_socket():
    """Whether this builder is allowed to listen on the loopback address.

    Asked by TRYING it, not by guessing from the platform: sandboxed and
    managed builders refuse the bind, and the four integration tests below
    cannot run there. Everything they check is also checked without a
    socket by the unit tests further down, so skipping them loses coverage
    of the transport, not of the retry policy.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(('127.0.0.1', 0))
        return True
    except OSError:
        return False
    finally:
        probe.close()


needs_a_socket = pytest.mark.skipif(
    not _can_bind_a_local_socket(),
    reason='this environment does not allow binding 127.0.0.1')


@pytest.fixture
def throttled_server():
    """A real server on a real port, reset for each test."""
    _Throttler.seen = 0
    _Throttler.fail_times = 2
    _Throttler.retry_after = '0'
    _Throttler.status_after_failures = 200
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _Throttler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_address[1]}/archive'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@needs_a_socket
def test_a_throttled_request_is_RETRIED_until_it_succeeds(throttled_server):
    """Two real 429s, then a real body -- and the body is the one returned."""
    from scripts.weather_forecast_study import _get_with_retries
    request = urllib.request.Request(throttled_server)
    body = _get_with_retries(request, 'test', waits=(0.01, 0.02, 0.03))
    assert body == b'{"served": true}'
    assert _Throttler.seen == 3, 'both throttles must have been retried'


@needs_a_socket
def test_the_server_STATED_wait_is_honoured_over_the_backoff_schedule(
        throttled_server):
    """`Retry-After: 1` must be obeyed even though the local schedule would
    have waited 0.01s. Measured on the clock: a run that ignored the header
    would come back in milliseconds."""
    from scripts.weather_forecast_study import _get_with_retries
    _Throttler.fail_times = 1
    _Throttler.retry_after = '1'
    started = time.monotonic()
    body = _get_with_retries(urllib.request.Request(throttled_server), 'test',
                             waits=(0.01, 0.02))
    assert body == b'{"served": true}'
    assert time.monotonic() - started >= 1.0


@needs_a_socket
def test_a_request_that_is_WRONG_rather_than_throttled_is_not_retried(
        throttled_server):
    """A 404 means the URL is broken. Repeating it six times wastes a minute
    and reports the same failure at the end, so it must raise immediately."""
    from scripts.weather_forecast_study import _get_with_retries
    _Throttler.fail_times = 0
    _Throttler.status_after_failures = 404
    with pytest.raises(urllib.error.HTTPError) as caught:
        _get_with_retries(urllib.request.Request(throttled_server), 'test',
                          waits=(0.01, 0.02))
    assert caught.value.code == 404
    assert _Throttler.seen == 1, 'a 404 must not be retried at all'


@needs_a_socket
def test_a_throttle_that_never_clears_RAISES_rather_than_going_synthetic(
        throttled_server):
    """The failure that started all of this. When the retries are exhausted
    the error must reach `fetch_city_months`, which prints it and returns
    None -- and `main` then refuses to report a synthetic run. What must not
    happen is a quiet fallback that looks like a real result."""
    from scripts.weather_forecast_study import _get_with_retries
    _Throttler.fail_times = 99
    with pytest.raises(urllib.error.HTTPError) as caught:
        _get_with_retries(urllib.request.Request(throttled_server), 'test',
                          waits=(0.01, 0.01))
    assert caught.value.code == 429
    assert _Throttler.seen == 3, 'two waits means three attempts'


@pytest.mark.parametrize('header, expected', [
    ('12', 12.0),
    ('  7.5 ', 7.5),
    ('-3', 0.0),                       # never sleep a negative duration
    ('not a number', None),
    ('', None),
    (None, None),
])
def test_retry_after_parses_the_numeric_spelling(header, expected):
    from scripts.weather_forecast_study import _retry_after_seconds
    got = _retry_after_seconds(header)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)


def test_retry_after_parses_the_HTTP_DATE_spelling():
    """RFC 9110 allows an absolute date, and open-meteo is not the only
    server this pattern will be copied to. A date 30s out must become ~30
    seconds, not None (which would fall back to the local schedule) and not
    a crash."""
    from scripts.weather_forecast_study import _retry_after_seconds
    when = datetime.datetime.now(datetime.timezone.utc) + \
        datetime.timedelta(seconds=30)
    got = _retry_after_seconds(email.utils.format_datetime(when))
    assert 25 <= got <= 31


def test_a_date_in_the_PAST_means_retry_now_not_a_negative_sleep():
    from scripts.weather_forecast_study import _retry_after_seconds
    when = datetime.datetime.now(datetime.timezone.utc) - \
        datetime.timedelta(seconds=60)
    assert _retry_after_seconds(email.utils.format_datetime(when)) == 0.0


# --------------------------------------------------- retry, without a socket
# The four tests above are the integration layer: a real server, real
# sockets, real 429s. They are skipped where a builder may not bind. These
# check the same policy through the `opener` seam, and they check two things
# the server tests CANNOT: the real 62-second wait schedule (no test should
# sit through it) and the exact jitter bounds.
#
# Nothing here is a mock object. `_scripted_opener` hands back the same two
# real types urllib itself hands back -- `urllib.error.HTTPError` for a
# failure and `urllib.response.addinfourl` for a body -- so the code under
# test cannot tell the difference, and there is no stub whose behaviour
# could drift from the library's.

def _http_error(code, retry_after=None, url='http://example.invalid/archive'):
    """A real `urllib.error.HTTPError`, built the way urllib builds one."""
    headers = email.message.Message()
    if retry_after is not None:
        headers['Retry-After'] = retry_after
    return urllib.error.HTTPError(url, code, f'HTTP {code}', headers,
                                  io.BytesIO(b''))


def _response(body, url='http://example.invalid/archive'):
    """A real `urllib.response.addinfourl` -- what `urlopen` returns."""
    return urllib.response.addinfourl(io.BytesIO(body), email.message.Message(),
                                      url, 200)


def _scripted_opener(script):
    """Play `script` in order: raise the exceptions, return the responses."""
    remaining = list(script)
    calls = []

    def opener(request, timeout=None):
        calls.append(timeout)
        step = remaining.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    opener.calls = calls
    return opener


def _recording_sleep():
    waited = []

    def sleep(seconds):
        waited.append(seconds)

    sleep.waited = waited
    return sleep


def test_the_REAL_wait_schedule_spans_the_measured_throttle():
    """The default schedule, which no test can sit through for real.

    The sixth of six back-to-back archive requests was measured needing
    roughly 30 s; a 1+2+4+8 schedule gave up after 15 and reported the run
    as offline. Five waits of 2/4/8/16/32, jittered, span 62 s, and each
    wait must stay inside its jitter band -- a bug that dropped the
    multiplier would still "retry", just uselessly fast.
    """
    from scripts.weather_forecast_study import (_RETRY_WAITS,
                                                _get_with_retries)
    opener = _scripted_opener([_http_error(429)] * 5 + [_response(b'ok')])
    sleep = _recording_sleep()
    body = _get_with_retries(urllib.request.Request('http://example.invalid'),
                             'test', sleep=sleep, opener=opener)
    assert body == b'ok'
    assert len(sleep.waited) == len(_RETRY_WAITS) == 5
    for waited, nominal in zip(sleep.waited, _RETRY_WAITS):
        assert 0.8 * nominal <= waited <= 1.3 * nominal
    assert sum(_RETRY_WAITS) == 62


def test_the_waits_are_JITTERED_rather_than_identical_every_run():
    """Six cities throttled together must not all come back at the same
    instant and throttle each other again. Two runs of the same schedule
    must therefore differ."""
    from scripts.weather_forecast_study import _get_with_retries

    def run():
        sleep = _recording_sleep()
        _get_with_retries(urllib.request.Request('http://example.invalid'),
                          'test', sleep=sleep,
                          opener=_scripted_opener([_http_error(503),
                                                   _response(b'ok')]))
        return sleep.waited[0]

    assert len({round(run(), 9) for _ in range(8)}) > 1


def test_a_stated_Retry_After_REPLACES_the_schedule_without_a_socket():
    from scripts.weather_forecast_study import _get_with_retries
    sleep = _recording_sleep()
    body = _get_with_retries(
        urllib.request.Request('http://example.invalid'), 'test',
        waits=(2, 4), sleep=sleep,
        opener=_scripted_opener([_http_error(429, retry_after='11'),
                                 _response(b'ok')]))
    assert body == b'ok'
    assert sleep.waited == [11.0]


@pytest.mark.parametrize('code, retried', [
    (429, True),                       # throttled: the whole point
    (500, True), (503, True),          # the server is briefly unwell
    (400, False), (404, False),        # the request itself is wrong
])
def test_only_a_RETRYABLE_status_is_retried(code, retried):
    from scripts.weather_forecast_study import _get_with_retries
    sleep = _recording_sleep()
    opener = _scripted_opener([_http_error(code), _response(b'ok')])
    request = urllib.request.Request('http://example.invalid')
    if retried:
        assert _get_with_retries(request, 'test', waits=(0.5,), sleep=sleep,
                                 opener=opener) == b'ok'
        assert len(sleep.waited) == 1
    else:
        with pytest.raises(urllib.error.HTTPError) as caught:
            _get_with_retries(request, 'test', waits=(0.5,), sleep=sleep,
                              opener=opener)
        assert caught.value.code == code
        assert sleep.waited == [], 'a wrong request must not be slept on'


def test_the_LAST_attempt_re_raises_rather_than_returning_nothing():
    """Exhausting the schedule must raise, so `fetch_city_months` prints the
    failure and `main` refuses to report a synthetic run. Returning None
    here is how the study answered a question about real weather with
    fabricated weather the first time."""
    from scripts.weather_forecast_study import _get_with_retries
    sleep = _recording_sleep()
    opener = _scripted_opener([_http_error(429)] * 3)
    with pytest.raises(urllib.error.HTTPError):
        _get_with_retries(urllib.request.Request('http://example.invalid'),
                          'test', waits=(0.1, 0.2), sleep=sleep, opener=opener)
    assert len(opener.calls) == 3, 'two waits means exactly three attempts'


def test_the_request_carries_a_TIMEOUT_so_a_hung_server_cannot_stall_the_run():
    """A throttled request that never answers would hang the whole study."""
    from scripts.weather_forecast_study import _get_with_retries
    opener = _scripted_opener([_response(b'ok')])
    _get_with_retries(urllib.request.Request('http://example.invalid'), 'test',
                      opener=opener)
    assert opener.calls == [60]


# ------------------------------------------------------- calibration cutoff
# Review round 12, finding 2: the units each city is scored in must not be
# measured on the outcomes being scored. `unit_scale` decides how loudly a
# city votes in the pooled headline, so computing it over the whole series
# used held-out evaluation-period volatility to set those weights.

def test_the_UNITS_are_blind_to_everything_after_the_cutoff():
    """The leakage test. Rewrite the post-cutoff data as violently as you
    like -- x100, sign-flipped, a different process entirely -- and the
    stored scale must not move by one ulp."""
    from scripts.weather_forecast_study import unit_scale
    series = _series(n=200)
    cutoff = 25
    before = unit_scale(series, cutoff)
    for mutation in (lambda x: x * 100.0, lambda x: -x,
                     lambda x: np.zeros_like(x),
                     lambda x: np.cumsum(x, axis=0)):
        mutated = series.copy()
        mutated[cutoff:] = mutation(mutated[cutoff:])
        assert np.array_equal(unit_scale(mutated, cutoff), before), (
            'the calibration window is not blind to the evaluation period')


def test_the_units_DO_move_when_the_calibration_window_itself_changes():
    """The control. Without it the test above would pass just as well
    against a `unit_scale` that ignored its input entirely."""
    from scripts.weather_forecast_study import unit_scale
    series = _series(n=200)
    mutated = series.copy()
    mutated[:25] *= 100.0
    assert not np.allclose(unit_scale(mutated, 25), unit_scale(series, 25))


def test_evaluate_REFUSES_a_cutoff_that_reaches_past_the_first_anchor():
    """Asserted rather than trusted: a caller that passes the series length
    (the pre-fix behaviour) must fail loudly, not score quietly."""
    from scripts.weather_forecast_study import evaluate
    arrays = [_series(n=120, seed=s) for s in range(2)]
    with pytest.raises(ValueError, match='reaches past the first anchor'):
        evaluate(arrays, 'Kalman', 1, [40, 60, 80], 'block1',
                 calibration_end=len(arrays[0]))


def test_a_calibration_window_too_short_to_DIFF_yields_no_units():
    """One observation gives no changes, so there is no spread to measure.
    That must be nan -- a city dropped from the pooling -- not a crash and
    not a silent 1.0."""
    from scripts.weather_forecast_study import unit_scale
    assert not np.any(np.isfinite(unit_scale(_series(n=50), 1)))


def test_the_WINDSPEED_cell_is_a_near_tie_and_the_rule_still_refuses_it():
    """The one cell where the calibrated headline changes which baseline is
    the one to beat: in block 2, `seasonal_naive` (+0.296) edges out
    `climatology`, and ARIMA's +0.2961 clears it by 0.0001.

    A margin that small is a rounding artefact, not a result -- and the
    rule refuses it anyway, because ARIMA loses the same measure in block
    1 (+0.257 vs +0.824). Pinned because "climatology is the strongest
    baseline in all eight cells" was TRUE before this calibration and is
    now true in seven of eight; a claim like that has to be re-checked, not
    carried forward.
    """
    b1 = _row('ARIMA', 'block1', [0.641, 0.695, 0.418, 0.257],
              [0.921, 0.758, 0.719, 0.824])
    b2 = _row('ARIMA', 'block2', [0.578, 0.609, 0.365, 0.2961],
              [0.925, 0.660, 0.648, 0.2960])
    assert b2['scores'][HEADLINE][3] > best_baseline(b2, 3), (
        'block 2 alone really does clear it, by 0.0001')
    assert verdict([b1, b2]) == []
