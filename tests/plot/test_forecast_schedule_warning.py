# -*- coding: utf-8 -*-
"""A slow forecast schedule must SAY it will be slow.

The maintainer ruled out striding the schedule to make it cheaper: sampling
the reveal instead of tracking it exactly would change *what is plotted*,
and "the critical thing is the outcome, not the speed". The accepted cost of
that decision is that a large dataset simply takes a long time before the
first frame appears. What the library owes the user is NOTICE -- a long
render should be expected, not mysterious.
"""

import warnings

import numpy as np
import pytest

import hypertools as hyp
from hypertools.plot.forecast import (ForecastSchedule,
                                      project_schedule_cost)


def _walk(rows, dims=3, seed=0):
    return np.random.default_rng(seed).normal(size=(rows, dims)).cumsum(axis=0)


def test_a_small_schedule_warns_about_nothing():
    """The common case must stay silent. A warning users see on every plot
    is a warning they learn to ignore."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')          # any warning fails the test
        ForecastSchedule.for_parallel([_walk(30)], [30], model='Kalman', t=3,
                                      n_frames=20)


def test_a_slow_schedule_warns_before_doing_the_work():
    """The warning must arrive EARLY -- a notice delivered after a five
    minute wait is not a notice. It is emitted once the first fit has been
    timed and the projection exceeds the threshold, so the user learns what
    they are in for while there is still something to do about it.
    """
    with pytest.warns(UserWarning, match='forecast') as record:
        ForecastSchedule.for_parallel(
            [_walk(120)], [120], model='Kalman', t=3, n_frames=120,
            # a threshold low enough that the real (fast) schedule trips it,
            # so the test needs no artificially huge dataset
            slow_warning_seconds=0.0)
    msg = ' '.join(str(w.message) for w in record)
    assert 'fit' in msg.lower(), msg
    # it must quantify, not merely say "this may be slow"
    assert any(ch.isdigit() for ch in msg), msg


def test_the_warning_names_the_kwarg_that_silences_it():
    """A warning with no escape hatch is noise. It must say how to turn it
    off, and the named kwarg must actually exist."""
    with pytest.warns(UserWarning) as record:
        ForecastSchedule.for_parallel([_walk(80)], [80], model='Kalman', t=3,
                                      n_frames=80, slow_warning_seconds=0.0)
    msg = ' '.join(str(w.message) for w in record)
    assert 'slow_warning_seconds' in msg, msg
    # and it really is silenceable
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        ForecastSchedule.for_parallel([_walk(80)], [80], model='Kalman', t=3,
                                      n_frames=80, slow_warning_seconds=None)


def test_warning_or_not_the_schedule_is_byte_identical():
    """The whole point of the maintainer's ruling: notice must not change
    the outcome. A warned schedule and an unwarned one must produce exactly
    the same forecasts -- this is what rules out striding.
    """
    kw = dict(model='Kalman', t=3, n_frames=60)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        warned = ForecastSchedule.for_parallel([_walk(70)], [70],
                                               slow_warning_seconds=0.0, **kw)
    quiet = ForecastSchedule.for_parallel([_walk(70)], [70],
                                          slow_warning_seconds=None, **kw)
    assert warned.n_fits == quiet.n_fits
    assert sorted(warned._paths) == sorted(quiet._paths)
    for key, path in warned._paths.items():
        other = quiet._paths[key]
        if path is None or other is None:
            assert path is other, key
        else:
            assert np.array_equal(path, other), key


def _reported(rows, frames, dims=3):
    """(fit count, projected seconds) as the warning reports them."""
    import re
    with pytest.warns(UserWarning) as rec:
        ForecastSchedule.for_parallel([_walk(rows, dims=dims)], [rows],
                                      model='Kalman', t=3, n_frames=frames,
                                      slow_warning_seconds=0.0)
    msg = ' '.join(str(w.message) for w in rec)
    fits = re.findall(r'needs (\d+) forecast fits', msg)
    secs = re.findall(r'roughly ([\d.]+) s', msg)
    assert fits and secs, f'message did not quantify the job: {msg}'
    return int(fits[0]), float(secs[0])


def test_the_reported_cost_scales_with_DATA_not_with_frames():
    """The number the warning quotes must be the real size of the job.

    This is the memoization property itself, and it is what makes the
    maintainer's "no striding" ruling affordable at all: a fit is needed per
    DISTINCT revealed history length, so the count is bounded by ROWS, not by
    frames. Drawing ten times as many frames of the same series costs
    nothing extra.

    Asserted on the reported count rather than on elapsed time on purpose.
    An earlier version of this test compared the projected SECONDS for a
    3-dimensional against a 12-dimensional fit and required the latter to be
    larger. It passed alone and failed under full-suite load: at these sizes
    a fit is sub-millisecond, the projection quantises to ~0.1 s, and the
    first timed fit carries warm-up that spiked one run to 14.6 s. The
    property was real but the probe could not show it -- a test that cannot
    reliably demonstrate what it claims is not evidence.
    """
    few_frames, _ = _reported(rows=60, frames=60)
    many_frames, _ = _reported(rows=60, frames=600)
    assert few_frames == many_frames == 60, (
        f'10x the frames changed the fit count ({few_frames} -> '
        f'{many_frames}); memoization is not keyed on revealed length')

    more_rows, _ = _reported(rows=120, frames=600)
    assert more_rows == 120, more_rows
    assert more_rows > many_frames, (
        'more DATA must cost more fits, even at the same frame count')


def test_the_projection_comes_from_a_real_fit():
    """It must be extrapolated from timing actual work.

    The specific bug this pins: the first entries in the schedule are
    histories SHORTER than `min_history`, where `forecast_from_history`
    returns None without fitting anything. Timing one of those and
    extrapolating projected **0.0 s for a job that takes minutes** -- a
    notice worse than none, because it actively reassures. The projection
    must therefore be strictly positive whenever any real fit ran.
    """
    fits, seconds = _reported(rows=60, frames=60)
    assert fits > 0
    assert seconds > 0.0, (
        'projected 0 s while really doing work -- the timed fit was one of '
        'the below-min_history no-ops, not a real fit')


def test_the_projection_is_within_an_order_of_magnitude_of_the_real_cost():
    """The notice has to be usable as an estimate, not merely present.

    It projected `first_fit_time * fits_remaining`. `todo` is ordered by
    GROWING revealed history and a Kalman fit costs ~linearly in rows, so
    the first real fit is the cheapest one there will ever be: the estimate
    was low by the ratio of the mean history length to the shortest. On a
    real gallery figure it projected 12.9 s for a schedule that took 176 s
    -- 13.6x low -- and the example ended up passing
    `slow_warning_seconds=None` to suppress a number that wrong.

    Projecting per ROW instead makes it an estimate again. This asserts an
    order of magnitude, not a tight bound: the point is that a caller can
    act on it, and a tolerance tight enough to be flaky on a loaded machine
    would be a worse test than none.
    """
    import re
    import time

    data = [np.random.default_rng(s).normal(size=(150, 3)).cumsum(axis=0)
            for s in range(3)]
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        hyp.plot(data, '-', predict='Kalman', t=1, animate=True,
                 duration=2, frame_rate=8, slow_warning_seconds=0.0,
                 show=False)
    actual = time.perf_counter() - start

    notices = [str(w.message) for w in caught
               if 'forecast fits' in str(w.message)]
    assert len(notices) == 1, f'expected one projection notice, got {notices}'
    projected = float(re.search(r'roughly ([\d.]+) s', notices[0]).group(1))

    # the schedule dominates this call (measured: the same plot without
    # predict= is ~1% of it), so the wall clock is a fair yardstick
    assert projected > actual / 10, (
        f'projected {projected:.1f} s for a call that took {actual:.1f} s -- '
        'the estimate is low by more than an order of magnitude')
    assert projected < actual * 10, (
        f'projected {projected:.1f} s for a call that took {actual:.1f} s -- '
        'the estimate is high by more than an order of magnitude')


def test_project_schedule_cost_separates_the_SETUP_from_the_SLOPE():
    """The arithmetic, on timings chosen so the answer is checkable by hand.

    A fit that takes 0.5 s at 100 rows and 0.9 s at 300 rows costs
    0.002 s/row with 0.3 s of one-off setup. Ten more fits at 200 rows must
    therefore project 10 * (0.3 + 0.4) = 7.0 s. Neither degenerate model
    gets this right: a constant per fit says 9.0 s and a pure per-row rate
    says 4.0 s.
    """
    projected, per_row, setup, lengths = project_schedule_cost(
        {100: 0.5, 300: 0.9}, [200] * 10)
    assert per_row == pytest.approx(0.002)
    assert setup == pytest.approx(0.3)
    assert lengths == (100, 300)
    assert projected == pytest.approx(7.0)


def test_project_schedule_cost_REFUSES_a_single_history_length():
    """Two points at the same length cannot separate constant from slope.

    Returning a silent zero slope here is exactly the failure this module
    had: it looks like a linear estimator and behaves like a constant one.
    """
    with pytest.raises(ValueError, match='two DIFFERENT history lengths'):
        project_schedule_cost({100: 0.5}, [200, 300])


def test_a_noisy_pair_cannot_project_a_NEGATIVE_cost():
    """A longer fit that happens to time faster must not imply a refund."""
    projected, per_row, setup, _ = project_schedule_cost(
        {100: 0.9, 300: 0.5}, [200] * 4)
    assert per_row == 0.0
    assert setup == pytest.approx(0.5)
    assert projected == pytest.approx(2.0)


def test_the_projection_is_sampled_at_TWO_DIFFERENT_history_lengths():
    """The specific bug: `todo` is ordered by frame and THEN by dataset.

    With more than one dataset, every dataset is fitted at one revealed
    length before any of them advances -- measured, the first three entries
    of a 3-dataset parallel schedule all reveal 7 rows. Taking "the first
    two timed fits" therefore sampled one length twice, so the row
    difference was zero, the slope was clamped to 0, and the projection
    collapsed into the constant-per-fit estimate it replaced. Nothing
    failed, because an order-of-magnitude tolerance hides the difference on
    small data.

    Asserting on the RECORDED sample lengths is what makes that visible:
    under the old rule these two numbers were equal.
    """
    data = [np.random.default_rng(s).normal(size=(150, 3)).cumsum(axis=0)
            for s in range(3)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schedule = ForecastSchedule.for_parallel(
            data, [150] * 3, model='Kalman', t=1, n_frames=24,
            slow_warning_seconds=0.0)

    projection = schedule.projection
    assert projection is not None, 'no projection was ever made'
    short, long = projection['lengths']
    assert short != long, (
        f'both timing samples came from a {short}-row history, so the '
        'per-row slope is unidentifiable and the estimator is constant')
    assert projection['timed_fits'] >= 2
    assert projection['per_row'] >= 0.0
    assert np.isfinite(projection['per_row'])
    # and the reported total really is spent + remaining, not one of them
    assert projection['total'] == pytest.approx(
        projection['spent'] + projection['remaining'])


def test_the_warning_reports_the_TOTAL_not_just_what_is_left():
    """It claims a time "before the first frame can be drawn", so the number
    it quotes has to include the fits it already did to make the estimate --
    otherwise the sentence and the figure disagree."""
    import re
    with pytest.warns(UserWarning) as record:
        ForecastSchedule.for_parallel(
            [_walk(120)], [120], model='Kalman', t=3, n_frames=120,
            slow_warning_seconds=0.0)
    msg = ' '.join(str(w.message) for w in record)
    total = float(re.search(r'roughly ([\d.]+) s in total', msg).group(1))
    spent = float(re.search(r'([\d.]+) s already spent', msg).group(1))
    left = float(re.search(r'about ([\d.]+) s still to come', msg).group(1))
    assert total == pytest.approx(spent + left, abs=0.15), msg
    # and it must no longer claim a single timed fit
    assert 'one timed fit' not in msg, msg
    assert 'two history lengths' in msg, msg
