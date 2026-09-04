# -*- coding: utf-8 -*-
"""animate='serial' must mean the same thing on both backends, trail flags
included. Measured before this task: plotly warned "does not support trail
styles" and produced a figure identical to plain serial."""
import matplotlib
matplotlib.use("Agg")

import warnings

import numpy as np
import pytest

import hypertools as hyp

pytest.importorskip('plotly')

DURATION, FRAME_RATE = 3, 4
N_FRAMES = DURATION * FRAME_RATE          # 12
PROBE_FRAME = 3


def _datasets(n=3, rows=40, dims=4, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, dims)).cumsum(axis=0) for _ in range(n)]


def _mpl_counts(**kw):
    """(head vertex counts, trail vertex counts) at PROBE_FRAME."""
    fig, ani = hyp.plot(_datasets(), '-', duration=DURATION,
                        frame_rate=FRAME_RATE, show=False, **kw)
    ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
    ani._func(PROBE_FRAME, *ani._args)
    n = 3
    heads = [len(ln.get_data_3d()[0]) for ln in ax.lines[:n]]
    trails = [len(ln.get_data_3d()[0]) for ln in ax.lines[n:2 * n]]
    return heads, trails


def _plotly_fig(**kw):
    hyp.set_interactive_backend('plotly')
    try:
        return hyp.plot(_datasets(), '-', duration=DURATION,
                        frame_rate=FRAME_RATE, show=False, **kw)
    finally:
        hyp.set_interactive_backend('matplotlib')


def _plotly_counts(fig, n=3):
    frame = fig.frames[PROBE_FRAME]
    npts = [0 if t.x is None else len(t.x) for t in frame.data]
    return npts[:n], npts[n:2 * n]


def _alpha_of(color):
    """rgba(r,g,b,a) -> a"""
    return float(color.rsplit(',', 1)[1].rstrip(') '))


def test_plotly_serial_with_chemtrails_emits_no_ignore_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _plotly_fig(animate='serial', chemtrails=True)
    assert not [w for w in caught if 'does not support trail' in str(w.message)]


def test_plotly_serial_with_chemtrails_creates_trail_traces():
    plain = _plotly_fig(animate='serial')
    trailed = _plotly_fig(animate='serial', chemtrails=True)
    assert len(trailed.data) == len(plain.data) + 3


def test_plotly_serial_trail_traces_are_faded():
    """Same 0.3 opacity the parallel trails already use
    (plotly_backend.py:953)."""
    fig = _plotly_fig(animate='serial', chemtrails=True)
    alphas = [_alpha_of(t.line.color) for t in fig.data
              if t.line is not None and t.line.color is not None
              and t.line.color.startswith('rgba')]
    assert alphas[:3] == pytest.approx([1.0, 1.0, 1.0])
    assert alphas[3:6] == pytest.approx([0.3, 0.3, 0.3])


@pytest.mark.parametrize('flags', [
    {'chemtrails': True},
    {'precog': True},
    {'bullettime': True},
    {'chemtrails': True, 'precog': True},
])
def test_serial_trail_geometry_matches_matplotlib_frame_for_frame(flags):
    """The strong parity assertion: identical head AND trail point counts."""
    mpl_heads, mpl_trails = _mpl_counts(animate='serial', **flags)
    ply_heads, ply_trails = _plotly_counts(
        _plotly_fig(animate='serial', **flags))
    assert ply_heads == mpl_heads
    assert ply_trails == mpl_trails


def test_plain_serial_parity_is_unchanged():
    """Regression guard: the no-trail serial reveal already matched."""
    mpl_heads, _ = _mpl_counts(animate='serial')
    ply_heads, _ = _plotly_counts(_plotly_fig(animate='serial'))
    assert ply_heads == mpl_heads == [657, 0, 0]


def test_parallel_trail_parity_is_unchanged():
    mpl_heads, mpl_trails = _mpl_counts(animate=True, chemtrails=True)
    ply_heads, ply_trails = _plotly_counts(
        _plotly_fig(animate=True, chemtrails=True))
    assert ply_heads == mpl_heads
    assert ply_trails == mpl_trails


def test_spin_and_window_still_warn_and_ignore_on_both_backends():
    """Only 'serial' leaves the ignore list; 'spin'/'morph'/'window' keep the
    established warn-and-ignore behaviour (plot.py:3757-3781)."""
    for backend, setter in (('matplotlib', 'matplotlib'), ('plotly', 'plotly')):
        hyp.set_interactive_backend(setter)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                hyp.plot(_datasets(), '-', animate='window', chemtrails=True,
                         duration=DURATION, frame_rate=FRAME_RATE, show=False)
            assert [w for w in caught
                    if 'does not support trail' in str(w.message)], backend
        finally:
            hyp.set_interactive_backend('matplotlib')
