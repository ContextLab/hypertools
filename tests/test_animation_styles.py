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

    # dataset 1 (precog): future window only -- trail starts AFTER the
    # current frame index and runs to the end of the trajectory.
    expected1 = d1[num + 1:]
    assert len(xs1) == len(expected1)
    np.testing.assert_allclose(xs1, expected1[:, 0])
    np.testing.assert_allclose(xs1[0], d1[num + 1, 0])
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
# spin / serial modes: trails still resolve per-dataset (though, matching
# pre-existing global behavior, spin/serial never actually ANIMATE the trail
# artists/traces -- only 'parallel'/True does). This locks in that the
# per-dataset flags don't crash or change spin/serial's own behavior.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('mode', ['spin', 'serial'])
def test_mpl_spin_serial_unaffected_by_per_dataset_trails(mode):
    data = _walks(k=2)
    bundle = hyp.plot(
        data, animate=mode, duration=1, tail_duration=1,
        chemtrails=[True, False], precog=[False, True],
        show=False, return_model=True,
    )
    assert bundle['animation'] is not None
    import matplotlib.pyplot as plt
    plt.close('all')


@pytest.mark.parametrize('mode', ['spin', 'serial'])
def test_plotly_spin_serial_unaffected_by_per_dataset_trails(mode):
    pytest.importorskip('plotly')
    data = _walks(k=2)
    fig = hyp.plot(data, animate=mode, duration=1, tail_duration=1,
                   chemtrails=[True, False], precog=[False, True],
                   backend='plotly', show=False)
    # no trail traces in spin/serial mode -- only 'parallel'/True builds them
    assert len(fig.data) == 3  # 2 data + cube
    assert len(fig.frames) > 0


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
