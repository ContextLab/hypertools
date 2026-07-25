# -*- coding: utf-8 -*-
"""Per-dataset chemtrails/precog/bullettime trail styles (GH #127).

Historically `chemtrails`/`precog`/`bullettime` were single scalar bools
applied uniformly to every animated dataset; passing a list silently
truthy-broadcast (any non-empty list counted as True for every dataset,
ignoring its actual per-dataset content). This module locks in the fix:
each of the three kwargs now accepts a bool (broadcast) OR a list/tuple of
bool (one entry per FINAL -- post cluster/hue-reshape -- drawn dataset),
with mixed per-dataset combinations honored on both backends. The `animate`
MODE itself (True/'parallel'/'spin'/'serial') stays a single global camera/
frame-loop setting -- only the trail FLAGS become per-dataset.
"""

import warnings

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

import hypertools as hyp
from hypertools.plot.matplotlib_backend import _draw
from hypertools.plot.plotly_backend import plotly_draw


def _walks(n=60, d=3, k=3, seed=0):
    rng = np.random.default_rng(seed)
    return [np.cumsum(rng.standard_normal((n, d)), axis=0) + 5 * i
            for i in range(k)]


# ---------------------------------------------------------------------------
# matplotlib: mixed per-dataset trail styles
# ---------------------------------------------------------------------------

def test_mpl_mixed_trail_styles_per_dataset():
    """3 datasets: dataset 0 chemtrails only, dataset 1 precog only, dataset
    2 bullettime only. Drive the animation to a fixed mid-animation frame and
    assert directly on the trail artists' data.

    Ground truth is `line_ani._args[0]` -- the EXACT (post analyze/reduce/
    center/scale/interpolate) arrays `update_lines_parallel` itself operates
    on -- rather than the raw input `data`, since the analyze pipeline
    (PCA reduction, centering, scaling, line interpolation) transforms
    coordinates and point counts before animation.
    """
    data = _walks()
    bundle = hyp.plot(
        data, animate=True, duration=2, tail_duration=1, frame_rate=30,
        chemtrails=[True, False, False],
        precog=[False, True, False],
        bullettime=[False, False, True],
        show=False, return_model=True,
    )
    line_ani = bundle['animation']
    assert line_ani is not None

    data_lines = line_ani._args[0]
    tail_duration_frames = line_ani._args[4]
    d0, d1, d2 = data_lines[0], data_lines[1], data_lines[2]

    num = d0.shape[0] // 2  # a fixed mid-animation frame index
    lines, trail_lines = line_ani._func(num, *line_ani._args)

    assert len(trail_lines) == 3
    assert all(t is not None for t in trail_lines)

    xs0, ys0, zs0 = trail_lines[0].get_data_3d()
    xs1, ys1, zs1 = trail_lines[1].get_data_3d()
    xs2, ys2, zs2 = trail_lines[2].get_data_3d()

    # dataset 0 (chemtrails): past window only -- trail ends AT the current
    # frame index and starts at the beginning of the trajectory.
    expected0 = d0[0: num - tail_duration_frames + 1]
    assert len(xs0) == len(expected0)
    np.testing.assert_allclose(xs0, expected0[:, 0])
    np.testing.assert_allclose(xs0[-1], d0[num - tail_duration_frames, 0])
    np.testing.assert_allclose(xs0[0], d0[0, 0])

    # dataset 1 (precog): future window only -- trail starts AT the current
    # frame index (sharing the opaque head's last vertex, so head and trail
    # form one continuous line -- release-1.0 audit F05-008: the historical
    # `d1[num + 1:]` slice left a one-segment gap between them) and runs to
    # the end of the trajectory.
    expected1 = d1[num:]
    assert len(xs1) == len(expected1)
    np.testing.assert_allclose(xs1, expected1[:, 0])
    np.testing.assert_allclose(xs1[0], d1[num, 0])
    np.testing.assert_allclose(xs1[-1], d1[-1, 0])

    # dataset 2 (bullettime): full trail, start to finish.
    assert len(xs2) == d2.shape[0]
    np.testing.assert_allclose(xs2, d2[:, 0])
    np.testing.assert_allclose(xs2[0], d2[0, 0])
    np.testing.assert_allclose(xs2[-1], d2[-1, 0])

    # the three trail lengths genuinely differ (different windows)
    assert len({len(xs0), len(xs1), len(xs2)}) == 3

    import matplotlib.pyplot as plt
    plt.close('all')


def test_mpl_no_flags_dataset_gets_no_trail_artist():
    """A dataset with none of chemtrails/precog/bullettime set gets NO trail
    artist at all (not an inert stub), while a flagged sibling does."""
    data = _walks(k=2)
    bundle = hyp.plot(
        data, animate=True, duration=2, tail_duration=1, frame_rate=30,
        chemtrails=[True, False], show=False, return_model=True,
    )
    line_ani = bundle['animation']
    num = 20
    lines, trail_lines = line_ani._func(num, *line_ani._args)
    assert trail_lines[0] is not None
    assert trail_lines[1] is None

    import matplotlib.pyplot as plt
    plt.close('all')


# ---------------------------------------------------------------------------
# plotly: mixed per-dataset trail trace lengths
# ---------------------------------------------------------------------------

def test_plotly_mixed_trail_styles_per_dataset():
    pytest.importorskip('plotly')
    data = _walks()
    fig = plotly_draw(
        data, animate=True, duration=2, tail_duration=1,
        chemtrails=[True, False, False],
        precog=[False, True, False],
        bullettime=[False, False, True],
        show=False,
    )
    # 3 data traces + 3 trail traces (one per dataset, since ALL three
    # datasets have some flag set) + 1 cube
    assert len(fig.data) == 7

    mid = fig.frames[len(fig.frames) // 2]
    trail_by_idx = dict(zip(mid.traces, mid.data))
    head_lens = [len(fig.data[i].x) for i in range(3)]

    # trail traces sit at indices 3, 4, 5 (right after the 3 data traces,
    # no forecasts in play)
    trail0, trail1, trail2 = trail_by_idx[3], trail_by_idx[4], trail_by_idx[5]

    # dataset 0 (chemtrails): past trail starts at the trajectory origin
    assert trail0.x[0] == fig.data[0].x[0]
    # dataset 1 (precog): future trail ends at the trajectory's end
    assert trail1.x[-1] == fig.data[1].x[-1]
    # dataset 2 (bullettime): full trail, same length as the full dataset
    assert len(trail2.x) == len(fig.data[2].x)

    # the three trail lengths differ from each other (different windows)
    assert len({len(trail0.x), len(trail1.x), len(trail2.x)}) == 3


def test_plotly_no_flags_dataset_gets_no_trail_trace():
    """A dataset with no trail flags gets NO trail trace at all -- only
    the flagged dataset does."""
    pytest.importorskip('plotly')
    data = _walks(k=2)
    fig = plotly_draw(data, animate=True, duration=2, tail_duration=1,
                      chemtrails=[True, False], show=False)
    # 2 data + 1 trail (dataset 0 only) + 1 cube
    assert len(fig.data) == 4
    for frame in fig.frames:
        assert set(frame.traces) == {0, 1, 2}


# ---------------------------------------------------------------------------
# scalar broadcast equivalence
# ---------------------------------------------------------------------------

def test_mpl_scalar_broadcast_equivalent_to_explicit_list():
    data = _walks(k=2)
    b1 = hyp.plot(data, animate=True, duration=2, tail_duration=1,
                 chemtrails=True, show=False, return_model=True)
    b2 = hyp.plot(data, animate=True, duration=2, tail_duration=1,
                 chemtrails=[True, True], show=False, return_model=True)

    num = 15
    lines1, trail1 = b1['animation']._func(num, *b1['animation']._args)
    lines2, trail2 = b2['animation']._func(num, *b2['animation']._args)

    for t1, t2 in zip(trail1, trail2):
        np.testing.assert_allclose(t1.get_data_3d()[0], t2.get_data_3d()[0])
        np.testing.assert_allclose(t1.get_data_3d()[1], t2.get_data_3d()[1])
        np.testing.assert_allclose(t1.get_data_3d()[2], t2.get_data_3d()[2])

    import matplotlib.pyplot as plt
    plt.close('all')


def test_plotly_scalar_broadcast_equivalent_to_explicit_list():
    pytest.importorskip('plotly')
    data = _walks(k=2)
    f1 = plotly_draw(data, animate=True, duration=2, tail_duration=1,
                     precog=True, show=False)
    f2 = plotly_draw(data, animate=True, duration=2, tail_duration=1,
                     precog=[True, True], show=False)
    assert len(f1.data) == len(f2.data)
    for fr1, fr2 in zip(f1.frames, f2.frames):
        for d1, d2 in zip(fr1.data, fr2.data):
            np.testing.assert_allclose(np.asarray(d1.x, dtype=float),
                                       np.asarray(d2.x, dtype=float))


# ---------------------------------------------------------------------------
# bad-length ValueError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('kwarg', ['chemtrails', 'precog', 'bullettime'])
def test_bad_length_list_raises_value_error(kwarg):
    data = _walks(k=3)
    with pytest.raises(ValueError, match=r"4 entries but there are 3"):
        # deliberately wrong length (4 entries for 3 datasets)
        hyp.plot(data, animate=True, duration=1, show=False,
                 **{kwarg: [True, False, True, False]})


def test_bad_length_list_names_actual_counts():
    data = _walks(k=2)
    with pytest.raises(ValueError) as excinfo:
        hyp.plot(data, animate=True, duration=1, show=False,
                 chemtrails=[True, True, True])
    msg = str(excinfo.value)
    assert '3 entries' in msg
    assert '2 datasets' in msg


# ---------------------------------------------------------------------------
# spin / morph / window modes (GH #127 follow-up): trail styles are
# semantically meaningless -- 'spin' has no "current position" for a trail to
# lead/follow (only the camera moves), 'morph' draws one traveling cloud, and
# 'window' is bullettime MINUS its trail by definition. `plot()` warns ONCE
# (naming the mode, the ignored flag(s), and which dataset indices had them
# set) and neither backend creates a trail artist/trace at all in these modes.
#
# 'serial' is the EXCEPTION on the matplotlib backend: it now COMPOSES with
# the trail flags (chemtrails-serial / precog-serial / bullettime-serial),
# tested separately below. The plotly backend still reveals 'serial' fully
# opaque with no trail, so it keeps warning there.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('mode', ['spin'])
def test_mpl_spin_warns_and_skips_trail_artists(mode):
    data = _walks(k=3, n=20)
    with pytest.warns(UserWarning,
                       match=r"animate=%r.*chemtrails.*\[0, 2\]" % mode):
        bundle = hyp.plot(
            data, animate=mode, duration=1, tail_duration=1, frame_rate=5,
            chemtrails=[True, False, True],
            show=False, return_model=True,
        )
    assert bundle['animation'] is not None

    fig = bundle['fig']
    ax = fig.axes[0]
    # exactly one Line3D per dataset -- no trail artists were ever created
    # (not merely hidden/frozen)
    assert len(ax.lines) == 3

    import matplotlib.pyplot as plt
    plt.close('all')


@pytest.mark.parametrize('mode', ['spin'])
def test_mpl_spin_warning_names_multiple_ignored_flags(mode):
    """precog on dataset 1 only -- the warning names precog (not
    chemtrails/bullettime, which are unset) and dataset index [1]."""
    data = _walks(k=2, n=20)
    with pytest.warns(UserWarning,
                       match=r"animate=%r.*precog.*\[1\]" % mode):
        hyp.plot(
            data, animate=mode, duration=1, tail_duration=1, frame_rate=5,
            precog=[False, True], show=False,
        )
    import matplotlib.pyplot as plt
    plt.close('all')


# ---------------------------------------------------------------------------
# serial COMPOSES with the trail flags on the matplotlib backend (new family:
# chemtrails-serial / precog-serial / bullettime-serial). Datasets still
# reveal one at a time; the ONE currently-revealing dataset also traces out a
# faded trail relative to its own reveal, past datasets stay fully drawn, and
# future datasets stay invisible.
# ---------------------------------------------------------------------------

def _serial_trail_bundle(**flags):
    """3 datasets, animate='serial'; return (bundle, data_lines, lines, trail)
    with NO 'trail styles' warning emitted (matplotlib serial now supports
    trails)."""
    data = _walks(k=3, n=20)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        bundle = hyp.plot(
            data, fmt='-', animate='serial', duration=2, frame_rate=10,
            show=False, return_model=True, **flags,
        )
    assert not any('trail styles' in str(w.message) for w in caught)
    line_ani = bundle['animation']
    assert line_ani is not None
    # serial 3D fargs: (x, lines, trail, cube_scale, window_frames,
    # rotations, zoom, chemtrails, precog, bullettime, elev)
    data_lines, lines, trail = (line_ani._args[0], line_ani._args[1],
                                line_ani._args[2])
    return bundle, line_ani, data_lines, lines, trail


def _len3d(artist):
    return len(artist.get_data_3d()[0])


@pytest.mark.parametrize('flag', ['chemtrails', 'precog', 'bullettime'])
def test_mpl_serial_composes_with_trail_flags(flag):
    """serial + one trail flag (all datasets): trail artists are CREATED, and
    at a mid-animation frame the currently-revealing dataset carries a faded
    trail while a fully-revealed earlier dataset is drawn in full and an
    unstarted later dataset is empty."""
    bundle, line_ani, data_lines, lines, trail = _serial_trail_bundle(
        **{flag: True})

    # a trail artist exists for every dataset (flag set on all), faded
    assert all(t is not None for t in trail)
    assert all(abs(t.get_alpha() - 0.3) < 1e-9 for t in trail)

    # pick a frame well into dataset 1's reveal so its opaque comet-head is
    # strictly shorter than the revealed span (a fade is actually visible).
    # tf = frame_rate*duration = 20; total_points = 3*20 = 60.
    tf = 20
    num = 11
    line_ani._func(num, *line_ani._args)
    n0, n1 = data_lines[0].shape[0], data_lines[1].shape[0]
    total_points = sum(d.shape[0] for d in data_lines)
    revealed = total_points * num / max(1, tf - 1)
    shown = int(np.clip(revealed - n0, 0, n1))       # dataset 1's reveal count
    assert 0 < shown < n1                            # ds1 is mid-reveal

    # past dataset 0 fully drawn as opaque head, no trail
    assert _len3d(lines[0]) == n0
    assert _len3d(trail[0]) == 0
    # future dataset 2 invisible (head + trail empty)
    assert _len3d(lines[2]) == 0
    assert _len3d(trail[2]) == 0

    # currently-revealing dataset 1: opaque comet-head STRICTLY shorter than
    # its revealed span (so the faded trail shows), plus a faded trail whose
    # extent matches the flag's semantics
    assert 0 < _len3d(lines[1]) < shown
    tl = _len3d(trail[1])
    if flag == 'chemtrails':
        assert tl == shown                 # revealed-so-far past
    elif flag == 'precog':
        assert tl == n1 - (shown - 1)      # not-yet-revealed future
    else:  # bullettime
        assert tl == n1                    # whole trajectory

    import matplotlib.pyplot as plt
    plt.close('all')


def test_mpl_serial_plain_still_has_no_trail_artists():
    """Plain animate='serial' (no trail flag) is UNCHANGED: no trail artists
    are created and each revealed dataset is drawn fully opaque (data[:shown])."""
    data = _walks(k=3, n=20)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        bundle = hyp.plot(data, fmt='-', animate='serial', duration=2,
                          frame_rate=10, show=False, return_model=True)
    assert not any('trail styles' in str(w.message) for w in caught)
    line_ani = bundle['animation']
    data_lines, lines, trail = (line_ani._args[0], line_ani._args[1],
                                line_ani._args[2])
    # trail list is present (one slot per dataset) but every slot is None --
    # no trail artist was ever built
    assert all(t is None for t in trail)
    # exactly one Line3D per dataset (heads only)
    assert len(bundle['fig'].axes[0].lines) == 3
    # head is the whole revealed span, opaque, exactly as before
    line_ani._func(9, *line_ani._args)
    assert _len3d(lines[0]) == data_lines[0].shape[0]   # ds0 fully revealed
    assert _len3d(lines[1]) == 8                          # ds1 shown=8
    assert _len3d(lines[2]) == 0                          # ds2 not started

    import matplotlib.pyplot as plt
    plt.close('all')


def test_mpl_serial_mixed_trail_flags_per_dataset():
    """Per-dataset flags compose with serial: dataset 0 chemtrails, dataset 1
    precog, dataset 2 bullettime, dataset 3 none -> a trail artist only for
    the first three."""
    data = _walks(k=4, n=20)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        bundle = hyp.plot(
            data, fmt='-', animate='serial', duration=2, frame_rate=10,
            chemtrails=[True, False, False, False],
            precog=[False, True, False, False],
            bullettime=[False, False, True, False],
            show=False, return_model=True,
        )
    assert not any('trail styles' in str(w.message) for w in caught)
    trail = bundle['animation']._args[2]
    assert [t is not None for t in trail] == [True, True, True, False]

    import matplotlib.pyplot as plt
    plt.close('all')


def test_mpl_serial_trail_renders_and_saves(tmp_path):
    """End-to-end: a serial+chemtrails animation saves a multi-frame GIF
    without error (the motivating conversation-turns use case)."""
    rng = np.random.default_rng(0)
    turns = [np.cumsum(rng.standard_normal((8, 3)), 0)
             + rng.standard_normal(3) * 4 for _ in range(6)]
    out = tmp_path / 'chemserial.gif'
    fig, ani = hyp.plot(turns, fmt='-', animate='serial', chemtrails=True,
                        duration=2, frame_rate=6, show=False)
    ani.save(str(out))
    assert out.exists() and out.stat().st_size > 0

    import matplotlib.pyplot as plt
    plt.close('all')


def test_mpl_serial_2d_composes_with_trail_flags():
    """2-D serial also composes with the trail flags (fixed viewport)."""
    rng = np.random.default_rng(3)
    turns = [np.cumsum(rng.standard_normal((10, 2)), 0)
             + rng.standard_normal(2) * 4 for _ in range(3)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        bundle = hyp.plot(turns, fmt='-', animate='serial', chemtrails=True,
                          ndims=2, duration=2, frame_rate=10, show=False,
                          return_model=True)
    assert not any('trail styles' in str(w.message) for w in caught)
    line_ani = bundle['animation']
    # 2D serial fargs: (x, lines, trail, window_frames, chemtrails,
    # precog, bullettime)
    trail = line_ani._args[2]
    assert all(t is not None for t in trail)
    line_ani._func(9, *line_ani._args)
    # dataset 1 revealing -> its 2-D trail has data
    assert any(len(t.get_xdata()) > 0 for t in trail)

    import matplotlib.pyplot as plt
    plt.close('all')


@pytest.mark.parametrize('mode', ['spin', 'serial'])
def test_plotly_spin_serial_warns_and_skips_trail_traces(mode):
    pytest.importorskip('plotly')
    data = _walks(k=2, n=20)
    with pytest.warns(UserWarning,
                       match=r"animate=%r.*bullettime.*\[0\]" % mode):
        fig = hyp.plot(data, animate=mode, duration=1, tail_duration=1,
                       frame_rate=5, bullettime=[True, False],
                       backend='plotly', show=False)
    # no trail traces in spin/serial mode -- only 'parallel'/True builds them
    assert len(fig.data) == 3  # 2 data + cube
    assert len(fig.frames) > 0


def test_mpl_parallel_not_warned_for_trails():
    """animate=True/'parallel' is the one mode where trail styles DO apply
    -- no warning should fire, and the trail artist should still be built
    normally."""
    data = _walks(k=2, n=20)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        bundle = hyp.plot(
            data, animate=True, duration=1, tail_duration=1, frame_rate=5,
            chemtrails=[True, False], show=False, return_model=True,
        )
    assert not any('trail styles' in str(w.message) for w in caught)
    num = 5
    lines, trail_lines = bundle['animation']._func(
        num, *bundle['animation']._args)
    assert trail_lines[0] is not None

    import matplotlib.pyplot as plt
    plt.close('all')


def test_plotly_parallel_not_warned_for_trails():
    pytest.importorskip('plotly')
    data = _walks(k=2, n=20)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig = hyp.plot(data, animate=True, duration=1, tail_duration=1,
                       frame_rate=5, precog=[True, False],
                       backend='plotly', show=False)
    assert not any('trail styles' in str(w.message) for w in caught)
    assert len(fig.data) == 4  # 2 data + 1 trail (dataset 0) + cube


# ---------------------------------------------------------------------------
# layout-lock regression (commit 30dac241): explicit trail-trace indexing
# must still hold with per-dataset flags in play.
# ---------------------------------------------------------------------------

def test_plotly_layout_lock_still_holds_with_per_dataset_flags():
    pytest.importorskip('plotly')
    rng = np.random.default_rng(0)
    d1 = np.cumsum(rng.standard_normal((30, 3)), axis=0)
    d2 = np.cumsum(rng.standard_normal((30, 3)), axis=0) + 3
    fc1 = np.vstack([d1[-1], d1[-1] + rng.standard_normal((5, 3))])
    fc2 = np.vstack([d2[-1], d2[-1] + rng.standard_normal((5, 3))])

    fig = plotly_draw([d1, d2], animate=True, duration=2, tail_duration=1,
                      chemtrails=[True, True], forecasts=[fc1, fc2],
                      show=False)
    # 2 data + 2 forecast + 2 trail + 1 cube
    assert len(fig.data) == 7
    for frame in fig.frames:
        assert 2 not in frame.traces and 3 not in frame.traces
        assert 4 in frame.traces and 5 in frame.traces


# ---------------------------------------------------------------------------
# fractional duration must not crash (QC 2026-07)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('style', ['spin', 'serial', True])
def test_mpl_fractional_duration_frame_count(style):
    """A fractional `duration` made `frame_rate * duration` a float, which
    matplotlib's FuncAnimation rejected with `range(float)` ->
    "'float' object cannot be interpreted as an integer" for the spin/serial
    styles (the parallel/window styles used an int frame count already)."""
    from hypertools import HyperAnimation
    out = hyp.plot(_walks(n=40), animate=style, duration=2.5, frame_rate=20,
                   show=False)
    assert isinstance(out, HyperAnimation)


@pytest.mark.parametrize('style', ['spin', 'serial'])
def test_plotly_fractional_duration_frame_count(style):
    pytest.importorskip('plotly')
    fig = plotly_draw(_walks(n=40), animate=style, duration=2.5, frame_rate=20,
                      show=False)
    assert len(fig.frames) > 0


# ---------------------------------------------------------------------------
# per-point labels must track their datapoint's visibility window (QC 2026-07,
# previously a documented "known limitation": labels drawn on EVERY frame)
# ---------------------------------------------------------------------------

def _labeled_helix():
    t = np.linspace(0, 4 * np.pi, 40)
    traj = np.column_stack([np.cos(t), np.sin(t), t / 4.0])
    labels = [None] * 40
    labels[8], labels[32] = 'AAA', 'BBB'
    return traj, labels


def test_window_animation_labels_scroll_with_their_datapoint():
    traj, labels = _labeled_helix()
    anim = hyp.plot(traj, '-o', labels=labels, animate='window', focused=1.0,
                    duration=5, frame_rate=20, show=False)
    fig, fa = anim[0], anim[1]
    anns = {a.get_text(): a for a in fig.axes[0].get_children()
            if getattr(a, '_hyp_point_idx', None) is not None}
    assert set(anns) == {'AAA', 'BBB'}
    assert anns['AAA']._hyp_point_idx < anns['BBB']._hyp_point_idx

    def vis_at(frame):
        fa._func(frame, *fa._args)
        return {k: a.get_visible() for k, a in anns.items()}

    frames = [vis_at(f) for f in range(0, 100, 4)]
    # each label appears on SOME frames and is hidden on others (the old bug
    # left both visible on every frame)
    assert 0 < sum(f['AAA'] for f in frames) < len(frames)
    assert 0 < sum(f['BBB'] for f in frames) < len(frames)
    # AAA (earlier point) enters the window before BBB
    first_aaa = next(i for i, f in enumerate(frames) if f['AAA'])
    first_bbb = next(i for i, f in enumerate(frames) if f['BBB'])
    assert first_aaa < first_bbb


def test_spin_and_static_labels_stay_visible():
    traj, labels = _labeled_helix()
    # spin draws every point every frame -> labels stay shown
    anim = hyp.plot(traj, '-o', labels=labels, animate='spin', duration=3,
                    frame_rate=20, show=False)
    fig, fa = anim[0], anim[1]
    anns = [a for a in fig.axes[0].get_children()
            if getattr(a, '_hyp_point_idx', None) is not None]
    fa._func(7, *fa._args)
    assert all(a.get_visible() for a in anns)
    # static plot: labels visible
    stat = hyp.plot(traj, '-o', labels=labels, show=False)
    stat_anns = [a for a in stat.axes[0].get_children()
                 if getattr(a, '_hyp_point_idx', None) is not None]
    assert stat_anns and all(a.get_visible() for a in stat_anns)


def _anns(fig):
    return {a.get_text(): a for a in fig.axes[0].get_children()
            if getattr(a, '_hyp_point_idx', None) is not None}


def test_serial_animation_labels_reveal_cumulatively_multi_dataset():
    """Serial reveals points cumulatively across datasets: a label shows once
    its GLOBAL index is revealed and then stays. A label in the SECOND dataset
    must only appear after the first dataset is fully drawn (global-index
    mapping), not on every frame (red-team of 52bcff88)."""
    t = np.linspace(0, 4 * np.pi, 30)
    traj = np.column_stack([np.cos(t), np.sin(t), t / 4.0])
    labs = [[None] * 30, [None] * 30]
    labs[0][2] = 'START'
    labs[1][27] = 'END'
    anim = hyp.plot([traj, traj + 3], '-o', labels=labs, animate='serial',
                    duration=4, frame_rate=20, show=False)
    fig, fa = anim[0], anim[1]
    anns = _anns(fig)

    def vis(frame):
        fa._func(frame, *fa._args)
        return {k: a.get_visible() for k, a in anns.items()}

    early, late = vis(2), vis(78)
    assert early == {'START': False, 'END': False}   # nothing revealed yet
    assert late == {'START': True, 'END': True}        # both revealed by the end
    # END (2nd dataset) is NOT shown while only the 1st dataset is drawing
    assert vis(30)['START'] and not vis(30)['END']


def test_2d_window_animation_labels_scroll():
    t = np.linspace(0, 4 * np.pi, 30)
    tr = np.column_stack([np.cos(t), np.sin(t)])          # 2-D
    labels = [None] * 30
    labels[5], labels[25] = 'A', 'B'
    anim = hyp.plot(tr, '-o', labels=labels, animate='window', focused=1.0,
                    duration=4, frame_rate=20, show=False)
    fig, fa = anim[0], anim[1]
    anns = _anns(fig)

    def vis(frame):
        fa._func(frame, *fa._args)
        return {k: a.get_visible() for k, a in anns.items()}

    res = [vis(f) for f in range(0, 80, 5)]
    assert 0 < sum(r['A'] for r in res) < len(res)     # scrolls in and out
    assert 0 < sum(r['B'] for r in res) < len(res)


def test_morph_animation_hides_per_point_labels():
    t = np.linspace(0, 4 * np.pi, 30)
    traj = np.column_stack([np.cos(t), np.sin(t), t / 4.0])
    labs = [[None] * 30, [None] * 30]
    labs[0][2] = 'X'
    anim = hyp.plot([traj, traj + 2], '.', labels=labs, animate='morph',
                    duration=3, frame_rate=20, show=False)
    fig, fa = anim[0], anim[1]
    anns = _anns(fig)
    fa._func(20, *fa._args)
    assert anns and all(not a.get_visible() for a in anns.values())
