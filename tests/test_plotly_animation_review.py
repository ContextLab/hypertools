# -*- coding: utf-8 -*-
"""Plotly animation parity findings from the 1.1 release review.

* A 1-D (single-column, non-series) animation raised a clear ValueError on
  matplotlib but ran silently on plotly, drawing its frame-grid row numbers
  as the x axis.
* The serial reveal's `FrameContext.window_bounds` started at 0 on plotly
  while matplotlib reported the comet head's real start -- the one field
  the FrameContext contract says both backends fill identically.

Real figures and real `on_frame=` callbacks.
"""

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

import hypertools as hyp

pytest.importorskip('plotly')


def _walks(k=2, n=30, d=3, seed=0):
    rng = np.random.default_rng(seed)
    return [np.cumsum(rng.standard_normal((n, d)), 0) + 2 * i
            for i in range(k)]


# --------------------------------------------------------------------------
# finding 8: 1-D animation


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
@pytest.mark.parametrize('style', [True, 'window', 'serial'])
def test_single_column_animation_raises_on_both_backends(backend, style):
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.standard_normal(30))
    with pytest.raises(ValueError, match='Animations are only supported '
                       'for 2-D or 3-D plots'):
        hyp.plot([y, y + 1], backend=backend, show=False, animate=style,
                 duration=1, frame_rate=4)


def test_series_mode_animation_still_animates_on_plotly():
    """`ndims=1` series mode draws (row index, value) pairs -- a 2-D
    animation that reveals along x, documented in the CHANGELOG."""
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.standard_normal(30))
    fig = hyp.plot([y, y + 1], backend='plotly', show=False, animate=True,
                   ndims=1, duration=1, frame_rate=4)
    assert len(fig.frames) == 4


# --------------------------------------------------------------------------
# finding A: serial window_bounds


def _collect(backend):
    seen = []

    def grab(ctx):
        seen.append((ctx.frame, tuple(ctx.window_bounds),
                     tuple(ctx.revealed_counts)))

    hyp.plot(_walks(), backend=backend, show=False, animate=True,
             order='serial', chemtrails=True, tail_duration=0.1,
             duration=1, frame_rate=8, on_frame=grab)
    return seen


def _drive_matplotlib():
    seen = []

    def grab(ctx):
        seen.append((ctx.frame, tuple(ctx.window_bounds),
                     tuple(ctx.revealed_counts)))

    anim = hyp.plot(_walks(), backend='matplotlib', show=False,
                    animate=True, order='serial', chemtrails=True,
                    tail_duration=0.1, duration=1, frame_rate=8,
                    on_frame=grab)
    ani = anim.animation
    ani._init_draw()
    for k in range(8):
        ani._func(k, *ani._args)
    import matplotlib.pyplot as plt
    plt.close(anim.figure)
    return seen


def test_serial_comet_window_bounds_match_matplotlib():
    mpl = {f: (wb, rc) for f, wb, rc in _drive_matplotlib()}
    ply = {f: (wb, rc) for f, wb, rc in _collect('plotly')}
    assert set(ply) == set(range(8))
    for k in range(8):
        assert ply[k] == mpl[k], (k, ply[k], mpl[k])
    # and the reviewer's frames really are comet windows, not (0, n)
    assert any(wb[0][0] > 0 for wb, _ in ply.values())
