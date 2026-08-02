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

from hypertools.plot.forecast import ForecastSchedule


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
