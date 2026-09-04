# -*- coding: utf-8 -*-
"""Cross-backend parity for the parallel/'window' sliding window.

Both backends pace every dataset with the SAME function
(`hypertools.plot.trails.anim_window_bounds`), per dataset, per frame. They
did not always: the plotly renderer carried its own transcription of that
arithmetic, computing ONE window per frame from the LONGEST dataset and
clamping everyone else into it. Four divergences rode on that, all shipped
in 1.0 and all fixed together:

* a 5-row marker dataset plotted beside a 15-row line went BLANK for 9 of
  its 15 frames -- 60% of its own animation -- because the shared window
  slid past its end, while matplotlib kept a correctly-paced 2-point window
  alive to the last frame;
* a missing ``- 1`` in ``start`` left every steady-state head window one
  point shorter than matplotlib's, which also opened a one-segment GAP
  between a chemtrails trail and the head it is supposed to join;
* ``end`` floored at 2 where matplotlib floors at 1, so frame 0 of a
  ``precog`` trail was one point short;
* the frame COUNT floored at 2 where matplotlib floors at 1, so a
  sub-frame request animated in plotly and held still in matplotlib.

Every assertion here compares real renders from both backends against each
other -- not against a transcription of either one's arithmetic, which is
the failure mode that let the originals ship. Tests that pin a formula pin
whichever code wrote it.

All renders are real (no mocks): MPLBACKEND=Agg, show=False.
"""

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

import hypertools as hyp

pytest.importorskip('plotly')


def _walks(n=200, d=3, k=2, seed=0):
    rng = np.random.default_rng(seed)
    return [np.cumsum(rng.standard_normal((n, d)), axis=0) + 5 * i
            for i in range(k)]


def _mpl_head_counts(data, n_frames, **kw):
    """Per-frame, per-dataset head-artist point counts (matplotlib)."""
    ani = hyp.plot(data, show=False, return_model=True, **kw)['animation']
    counts = []
    for num in range(n_frames):
        ani._func(num, *ani._args)
        counts.append([len(line.get_data()[0]) for line in ani._args[1]])
    return counts


def _mpl_trail_counts(data, n_frames, **kw):
    """Per-frame, per-dataset TRAIL-artist point counts (matplotlib)."""
    ani = hyp.plot(data, show=False, return_model=True, **kw)['animation']
    counts = []
    for num in range(n_frames):
        ani._func(num, *ani._args)
        counts.append([len(t.get_data()[0]) for t in ani._args[2]
                       if t is not None])
    return counts


def _plotly_counts(fig, n_datasets, offset=0):
    """Per-frame, per-dataset point counts from `offset` (plotly)."""
    return [[len(f.data[offset + i].x) for i in range(n_datasets)]
            for f in fig.frames]


# ---------------------------------------------------------------------------
# head windows: identical on both backends, every frame, every dataset
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('style', [True, 'window'])
def test_head_window_matches_matplotlib_every_frame_3d(style):
    data = _walks()
    kw = dict(animate=style, duration=4, frame_rate=30, tail_duration=1,
              antialias=False)
    fig = hyp.plot(data, backend='plotly', show=False, **kw)
    n_frames = len(fig.frames)
    assert n_frames == 120
    mpl = _mpl_head_counts(data, n_frames, **kw)
    assert _plotly_counts(fig, 2) == mpl


def test_head_window_matches_matplotlib_every_frame_2d():
    data = _walks(n=150, d=2)
    kw = dict(animate=True, duration=2.5, frame_rate=30, tail_duration=1,
              ndims=2, antialias=False)
    fig = hyp.plot(data, backend='plotly', show=False, **kw)
    n_frames = len(fig.frames)
    assert n_frames == 75
    assert _plotly_counts(fig, 2) == _mpl_head_counts(data, n_frames, **kw)


# ---------------------------------------------------------------------------
# a short dataset keeps its own pacing beside a long one (never goes blank)
# ---------------------------------------------------------------------------

def test_short_dataset_never_blanks_beside_a_long_one():
    rng = np.random.default_rng(0)
    short = np.cumsum(rng.standard_normal((5, 3)), axis=0)
    long = np.cumsum(rng.standard_normal((15, 3)), axis=0) + 5
    kw = dict(animate=True, duration=3, frame_rate=5, tail_duration=0.5,
              fmt=['.', '-'], antialias=False)
    fig = hyp.plot([short, long], backend='plotly', show=False, **kw)
    n_frames = len(fig.frames)
    assert n_frames == 15
    mpl = _mpl_head_counts([short, long], n_frames, **kw)
    plotly = _plotly_counts(fig, 2)
    # the 5-row dataset used to show 0 points for frames 6-14 in plotly
    assert [c[0] for c in plotly] == [c[0] for c in mpl]
    assert all(c[0] > 0 for c in plotly)
    assert plotly == mpl


def test_single_point_dataset_stays_visible_on_both_backends():
    """A 1-row dataset is the degenerate case of the above: its whole
    trajectory is one vertex, which must be drawn on every frame rather
    than sliding out of a window sized for its longer neighbour."""
    rng = np.random.default_rng(1)
    point = np.zeros((1, 3))
    walk = np.cumsum(rng.standard_normal((20, 3)), axis=0)
    kw = dict(animate=True, duration=2, frame_rate=10, tail_duration=0.5,
              fmt=['.', '-'], antialias=False)
    fig = hyp.plot([point, walk], backend='plotly', show=False, **kw)
    n_frames = len(fig.frames)
    plotly = _plotly_counts(fig, 2)
    assert [c[0] for c in plotly] == [1] * n_frames
    assert plotly == _mpl_head_counts([point, walk], n_frames, **kw)


# ---------------------------------------------------------------------------
# trails: chemtrails joins the head, precog is right on frame 0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('duration,frame_rate,tail_duration', [
    # NOTE: this first config detects NONE of the four documented bugs -- it
    # passes at the pre-fix commit too. On a 120-row frame grid the old
    # `max(2, round(120 * 1 / 4)) == 30` coincides with the new
    # `int(30 * 1) == 30`, and the `end` floors differ only at k=0, where
    # `trail_stop` clamps to 0 either way. It is kept as a forward-looking
    # parity assertion, NOT as regression coverage; the config below is what
    # carries this case.
    (4, 30, 1),
    # a window only ONE frame long: plotly used to floor its window at 2
    # frames, so this config diverged at every frame rather than none
    (1, 10, 0.1),
])
def test_chemtrails_trail_matches_matplotlib_every_frame(duration, frame_rate,
                                                         tail_duration):
    data = _walks(n=120)
    kw = dict(animate=True, duration=duration, frame_rate=frame_rate,
              tail_duration=tail_duration, chemtrails=True, antialias=False)
    fig = hyp.plot(data, backend='plotly', show=False, **kw)
    n_frames = len(fig.frames)
    mpl = _mpl_trail_counts(data, n_frames, **kw)
    assert _plotly_counts(fig, 2, offset=2) == mpl


def test_chemtrails_trail_joins_the_head_with_no_gap():
    """The past trail's last vertex IS the head window's first vertex on
    both backends -- otherwise the faded trail and the opaque head are
    drawn with a one-segment hole between them."""
    data = _walks(n=120)
    kw = dict(animate=True, duration=4, frame_rate=30, tail_duration=1,
              chemtrails=True, antialias=False)
    fig = hyp.plot(data, backend='plotly', show=False, **kw)
    for k in (len(fig.frames) // 2, len(fig.frames) - 1):
        head, trail = fig.frames[k].data[0], fig.frames[k].data[2]
        assert trail.x[-1] == head.x[0]
        assert trail.y[-1] == head.y[0]
        assert trail.z[-1] == head.z[0]

    ani = hyp.plot(data, show=False, return_model=True, **kw)['animation']
    for k in (len(fig.frames) // 2, len(fig.frames) - 1):
        ani._func(k, *ani._args)
        hx, hy = ani._args[1][0].get_data()
        tx, ty = ani._args[2][0].get_data()
        assert tx[-1] == hx[0]
        assert ty[-1] == hy[0]


def test_precog_trail_matches_matplotlib_on_frame_zero():
    rng = np.random.default_rng(2)
    data = [np.cumsum(rng.standard_normal((12, 3)), axis=0)]
    kw = dict(animate=True, duration=1, frame_rate=12, precog=True,
              antialias=False)
    fig = hyp.plot(data, backend='plotly', show=False, **kw)
    n_frames = len(fig.frames)
    assert n_frames == 12
    # frame 0 is the whole point: plotly's `end` used to floor at 2 where
    # matplotlib floors at 1, costing the precog trail its first point
    assert _plotly_counts(fig, 1, offset=1) == \
        _mpl_trail_counts(data, n_frames, **kw)


# ---------------------------------------------------------------------------
# frame count: same floor on both backends
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('style', [True, 'serial', 'spin', 'morph'])
def test_frame_count_matches_for_a_sub_frame_request(style):
    """`duration * frame_rate` below 1 still animates on BOTH backends.

    Every style, not just the parallel one: the frame count is resolved
    once, before `_add_animation` branches, so a floor that applies to one
    style applies to all four. It is also the denominator every dataset's
    window is paced against, so a floor that differs by one does not merely
    add a frame -- it shifts the pacing of every frame.

    matplotlib floored only its parallel/'window' path, so `'serial'` and
    `'spin'` asked `FuncAnimation` for ZERO frames here -- an animation that
    draws nothing at all. `'morph'` builds its own schedule from segment
    counts and lands on 3 frames on both backends.
    """
    rng = np.random.default_rng(3)
    data = [np.cumsum(rng.standard_normal((10, 3)), axis=0),
            np.cumsum(rng.standard_normal((10, 3)), axis=0) + 5]
    kw = dict(animate=style, duration=0.05, frame_rate=10, antialias=False)
    fig = hyp.plot(data, backend='plotly', show=False, **kw)
    ani = hyp.plot(data, show=False, return_model=True, **kw)['animation']
    assert ani._save_count >= 1
    assert len(fig.frames) == ani._save_count


@pytest.mark.parametrize('style', [True, 'serial', 'spin', 'morph'])
def test_frame_count_matches_across_backends_for_ordinary_durations(style):
    # NOTE: the `morph` arm is a control -- it passes at the pre-fix commit
    # too, because morph builds its own schedule from segment counts. The
    # other three arms are the regression coverage.
    rng = np.random.default_rng(4)
    data = [np.cumsum(rng.standard_normal((40, 3)), axis=0),
            np.cumsum(rng.standard_normal((40, 3)), axis=0) + 5]
    for duration, frame_rate in ((1, 12), (2.5, 30), (0.2, 5), (3, 7),
                                 (1.7, 9)):
        kw = dict(animate=style, duration=duration, frame_rate=frame_rate,
                  antialias=False)
        fig = hyp.plot(data, backend='plotly', show=False, **kw)
        ani = hyp.plot(data, show=False, return_model=True,
                       **kw)['animation']
        assert len(fig.frames) == ani._save_count, (style, duration,
                                                    frame_rate)


# ---------------------------------------------------------------------------
# playback speed: both backends request the same ms per frame
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('frame_rate,duration', [
    # non-integer `frame_rate * duration` is where these diverged: plotly
    # derived playback speed from the FRAME COUNT (1000*duration/n_frames)
    # while matplotlib derives it from the frame RATE
    (3, 1.4), (7, 2.5), (9, 1.7), (10, 0.05), (30, 4),
])
def test_play_button_speed_matches_matplotlib(frame_rate, duration):
    rng = np.random.default_rng(6)
    data = [np.cumsum(rng.standard_normal((30, 3)), axis=0)]
    kw = dict(animate=True, duration=duration, frame_rate=frame_rate,
              antialias=False)
    fig = hyp.plot(data, backend='plotly', show=False, **kw)
    ani = hyp.plot(data, show=False, return_model=True, **kw)['animation']
    play = fig.layout.updatemenus[0].buttons[0]
    plotly_ms = play.args[1]['frame']['duration']
    assert plotly_ms == pytest.approx(ani._interval, rel=1e-9)
    # ...and that is the true inter-frame interval, not a derived one
    assert plotly_ms == pytest.approx(1000.0 / frame_rate, rel=1e-9)


# ---------------------------------------------------------------------------
# the spin camera completes the same orbit on both backends
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('frame_rate,duration', [
    # a NON-integer frame_rate * duration is the case that diverged: the
    # matplotlib spin updater paced its orbit over the raw product while
    # every other path (and plotly) paces over the rounded frame count
    (7, 2.5), (9, 1.7), (30, 4), (12, 1),
])
def test_spin_camera_azimuth_matches_matplotlib(frame_rate, duration):
    rng = np.random.default_rng(5)
    data = [np.cumsum(rng.standard_normal((30, 3)), axis=0)]
    kw = dict(animate='spin', duration=duration, frame_rate=frame_rate,
              rotations=1, antialias=False)
    bundle = hyp.plot(data, show=False, return_model=True, **kw)
    ani, mfig = bundle['animation'], bundle['fig']
    fig = hyp.plot(data, backend='plotly', show=False, **kw)
    n = ani._save_count
    assert len(fig.frames) == n
    for k in (0, n // 2, n - 1):
        ani._func(k, *ani._args)
        eye = fig.frames[k].layout.scene.camera.eye
        plotly_azim = np.degrees(np.arctan2(eye.y, eye.x))
        # same angle modulo a full turn
        gap = (mfig.axes[0].azim - plotly_azim + 180) % 360 - 180
        assert abs(gap) < 1e-6, (k, mfig.axes[0].azim, plotly_azim)


# ---------------------------------------------------------------------------
# the window really is `focused`/`tail_duration` seconds long, both backends
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('frame_rate,seconds', [(20, 0.5), (10, 1), (30, 2)])
def test_window_spans_the_requested_seconds_on_both_backends(frame_rate,
                                                             seconds):
    data = _walks(n=200)
    kw = dict(animate='window', duration=4, frame_rate=frame_rate,
              focused=seconds, show=False, antialias=False)
    fig = hyp.plot(data, backend='plotly', **kw)
    n_frames = len(fig.frames)
    mid = n_frames // 2
    expected = int(frame_rate * seconds) + 1
    assert len(fig.frames[mid].data[0].x) == expected
    ani = hyp.plot(data, return_model=True, **kw)['animation']
    ani._func(mid, *ani._args)
    assert len(ani._args[1][0].get_data()[0]) == expected
