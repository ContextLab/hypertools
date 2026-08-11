# tests/plot/test_forecast_schedule.py
"""The precomputed forecast schedule: reveal mapping, memoization, purity."""

import numpy as np

from hypertools.plot.forecast import (DisplayTransform, ForecastSchedule,
                                      forecast_from_history,
                                      revealed_raw_counts)

N_RAW, N_GRID, N_FRAMES = 60, 8, 8

#: Every schedule built here passes this, and none of these tests is about the
#: warning -- `test_forecast_schedule_warning.py` pins both of its branches
#: deterministically (`0.0` always warns, `None` never does). At the DEFAULT
#: threshold the warning fires when `one_fit_seconds * remaining_fits > 10`,
#: which makes it a function of how fast the machine is at that moment rather
#: than of anything the code does: the 900-frame memoization test below
#: projects ~9.7 s on this laptop, so a machine a hair slower emits a warning
#: here and a hair faster does not. The suite is held to ZERO warnings, so
#: that difference is the difference between green and red -- a CI failure
#: nobody can reproduce locally. Measured: under `PYTHONDEVMODE=1`, which
#: slows a Kalman fit from ~54 ms to ~1.3 s, the default threshold warns.
NO_SPEED_WARNING = dict(slow_warning_seconds=None)


def _history(n=N_RAW, d=3, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d)).cumsum(axis=0)


# --- the reveal mapping ----------------------------------------------------

def test_revealed_raw_counts_is_monotonic():
    counts = [revealed_raw_counts(N_RAW, N_GRID, f, N_FRAMES)
              for f in range(N_FRAMES)]
    assert counts == sorted(counts), counts


def test_revealed_raw_counts_never_exceeds_the_raw_row_count():
    for f in range(N_FRAMES):
        assert 0 <= revealed_raw_counts(N_RAW, N_GRID, f, N_FRAMES) <= N_RAW


def test_the_last_frame_reveals_the_whole_history():
    """Contract 7 depends on this: the final-frame forecast IS the
    full-history forecast, so `return_model`'s bundle stays truthful."""
    assert revealed_raw_counts(N_RAW, N_GRID, N_FRAMES - 1, N_FRAMES) == N_RAW


def test_reveal_matches_the_library_formula_not_a_second_copy_of_it():
    from hypertools.plot.trails import anim_window_bounds
    for f in range(N_FRAMES):
        _, end, _ = anim_window_bounds(f, N_FRAMES, N_GRID, 0)
        pos = (end - 1) * (N_RAW - 1) / (N_GRID - 1)
        assert revealed_raw_counts(N_RAW, N_GRID, f, N_FRAMES) == int(pos) + 1


# --- the schedule ----------------------------------------------------------

def test_schedule_has_one_entry_per_frame_per_dataset():
    sched = ForecastSchedule.for_parallel(
        [_history(seed=s) for s in range(2)], [N_GRID] * 2,
        model='Kalman', t=3, n_frames=N_FRAMES, **NO_SPEED_WARNING)
    assert sched.n_datasets == 2 and sched.n_frames == N_FRAMES
    for i in range(2):
        for f in range(N_FRAMES):
            p = sched.path(i, f)
            assert p is None or p.shape == (4, 3)


def test_early_frames_have_no_forecast():
    """Frame 0 reveals 1 raw row; min_history=2 refuses to fit it."""
    sched = ForecastSchedule.for_parallel(
        [_history()], [N_GRID], model='Kalman', t=3, n_frames=N_FRAMES,
        **NO_SPEED_WARNING)
    assert sched.path(0, 0) is None


def test_final_frame_forecast_equals_the_full_history_forecast():
    hist = _history()
    sched = ForecastSchedule.for_parallel(
        [hist], [N_GRID], model='Kalman', t=3, n_frames=N_FRAMES,
        **NO_SPEED_WARNING)
    direct = forecast_from_history(hist, 'Kalman', t=3)
    assert np.allclose(sched.path(0, N_FRAMES - 1), direct)


def test_fits_are_memoized_by_revealed_history_length():
    """A 900-frame animation of a 60-row dataset can only have <= 60 distinct
    revealed lengths, so it must cost <= 60 fits, not 900. Measured cost of a
    single 60-row Kalman fit: 54 ms -- 900 fits would be 48s PER DATASET."""
    sched = ForecastSchedule.for_parallel(
        [_history(seed=s) for s in range(3)], [900] * 3,
        model='Kalman', t=3, n_frames=900, **NO_SPEED_WARNING)
    assert sched.n_fits <= 3 * N_RAW
    assert sched.n_fits < 900 * 3 / 4, sched.n_fits


def test_the_schedule_is_a_pure_lookup_so_frames_are_idempotent():
    sched = ForecastSchedule.for_parallel(
        [_history()], [N_GRID], model='Kalman', t=3, n_frames=N_FRAMES,
        **NO_SPEED_WARNING)
    forward = [sched.path(0, f) for f in range(N_FRAMES)]
    backward = [sched.path(0, f) for f in reversed(range(N_FRAMES))]
    for a, b in zip(forward, reversed(backward)):
        assert (a is None and b is None) or np.allclose(a, b)


def test_stacked_paths_covers_every_forecast_vertex():
    """Task 4 builds the bounding box from this, so a vertex it misses is a
    forecast that could render outside the cube."""
    sched = ForecastSchedule.for_parallel(
        [_history(seed=s) for s in range(2)], [N_GRID] * 2,
        model='Kalman', t=3, n_frames=N_FRAMES, **NO_SPEED_WARNING)
    stacked = sched.stacked_paths()
    assert stacked.ndim == 2 and stacked.shape[1] == 3
    for i in range(2):
        for f in range(N_FRAMES):
            drawn = sched.polyline(i, f)
            if drawn is None:
                continue
            for row in drawn:
                nearest = np.abs(stacked - row).sum(axis=1).min()
                assert np.isclose(nearest, 0.0), (
                    f'dataset {i} frame {f} vertex {row} is not in '
                    'stacked_paths(), so the bounding box would not hold it')


def test_serial_schedule_reveals_datasets_in_order():
    hists = [_history(seed=s) for s in range(3)]
    sched = ForecastSchedule.for_serial(hists, [N_GRID] * 3, model='Kalman',
                                        t=3, n_frames=16,
                                        **NO_SPEED_WARNING)
    early = [sched.revealed(i, 1) for i in range(3)]
    assert early[0] >= early[1] >= early[2]
    assert [sched.revealed(i, 15) for i in range(3)] == [N_RAW] * 3


# --- the display transform -------------------------------------------------

def test_display_transform_reproduces_plot_s_centre_scale_arithmetic():
    """Mirrors plot.py:4569-4582 exactly, on the same inputs."""
    rng = np.random.default_rng(4)
    data = rng.normal(size=(40, 3))
    mean = data.mean(axis=0)
    centred = data - mean
    m1 = centred.min()
    m2 = (centred - m1).max() or 1.0
    expected = 2 * ((centred - m1) / m2) - 1
    got = DisplayTransform(mean, m1, m2)(data)
    assert np.allclose(got, expected)
    assert got.min() >= -1.0 - 1e-12 and got.max() <= 1.0 + 1e-12


def test_to_display_maps_every_scheduled_forecast_into_the_cube():
    """Contract 4: no clamping is needed because the box was built to hold
    them. Build the transform from data + schedule, exactly as Task 4 does.

    Assert on `.polyline()`, NOT `.path()`. `path()` is the DISPLACEMENT
    (`to_display` rescales it by `2 / scale` only -- the mean cancels), so
    it is bounded by +/-2 and says nothing about where the forecast lands.
    `polyline()` is the drawn POSITION -- the anchor plus the displacement --
    and is the only thing the cube can contain. Measured on this exact
    fixture: paths peak at 0.86 while polylines reach exactly +/-1.000, so a
    path-based assertion passes no matter where the forecast is drawn, and
    can also fail spuriously on a large displacement that never leaves the
    box. `polyline()` is what Task 4 Step 5 feeds to the artists.
    """
    hists = [_history(seed=s) for s in range(2)]
    sched = ForecastSchedule.for_parallel(hists, [N_GRID] * 2, model='Kalman',
                                          t=5, n_frames=N_FRAMES,
                                          **NO_SPEED_WARNING)
    joint = np.vstack([np.vstack(hists), sched.stacked_paths()])
    mean = joint.mean(axis=0)
    joint_c = joint - mean
    m1 = joint_c.min()
    m2 = (joint_c - m1).max() or 1.0
    disp = sched.to_display(DisplayTransform(mean, m1, m2))
    checked = 0
    for i in range(2):
        for f in range(N_FRAMES):
            p = disp.polyline(i, f)
            if p is None:
                continue
            assert p.min() >= -1.0 - 1e-9 and p.max() <= 1.0 + 1e-9
            checked += 1
    # a loop that `continue`d every iteration would assert nothing
    assert checked > 0


def test_display_paths_are_displacements_not_positions():
    """Pins WHY the test above uses `polyline()`: `path()` is a difference,
    so it is not the quantity the cube bounds. If these two ever return the
    same thing, the test above has silently stopped testing Contract 4."""
    hists = [_history(seed=s) for s in range(2)]
    sched = ForecastSchedule.for_parallel(hists, [N_GRID] * 2, model='Kalman',
                                          t=5, n_frames=N_FRAMES,
                                          **NO_SPEED_WARNING)
    joint = np.vstack([np.vstack(hists), sched.stacked_paths()])
    mean = joint.mean(axis=0)
    m1 = (joint - mean).min()
    m2 = ((joint - mean) - m1).max() or 1.0
    disp = sched.to_display(DisplayTransform(mean, m1, m2))
    for i in range(2):
        for f in range(N_FRAMES):
            path, poly = disp.path(i, f), disp.polyline(i, f)
            if path is None:
                continue
            assert not np.allclose(path, poly)
            # the polyline IS the anchor plus the displacement
            assert np.allclose(poly - poly[0], path)
