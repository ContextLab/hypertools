"""The animated continuous-hue overlay must render at the width of the
artist it replaces (plot.py:5150-5153 read `linewidth` off kwargs_list after
matplotlib_backend.py:1602-1606 had already popped it, so every collection
fell back to rcParams['lines.linewidth'] == 1.5)."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

import hypertools as hyp


def _hue_datasets(n=3, rows=30, dims=4, seed=1):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, dims)).cumsum(axis=0) for _ in range(n)]


def _hue_for(datasets):
    """One value per OBSERVATION -- plot.py:3368-3375 counts
    sum(len(xi) for xi in xform), not len(datasets[0])."""
    return np.linspace(0.0, 1.0, sum(d.shape[0] for d in datasets))


def _ax(fig):
    return [a for a in fig.axes if hasattr(a, 'zaxis')][0]


def _overlay_widths(ax):
    """Widths of the head/trail collections built by
    `_apply_multicolor_animation._make_collection`, which is the only place
    that labels a collection '_nolegend_' (plot.py:5172). The cube-plane
    collections carry matplotlib's auto '_childN' labels."""
    return [float(np.atleast_1d(c.get_linewidth())[0])
            for c in ax.collections if c.get_label() == '_nolegend_']


def _static_widths(ax, n):
    """`_apply_multicolor_lines` (plot.py:5075-5101) removes every Line2D and
    appends its collections last, so the multicolour ones are the final n."""
    return [float(np.atleast_1d(c.get_linewidth())[0])
            for c in ax.collections[-n:]]


def test_animated_continuous_hue_honors_per_dataset_linewidth():
    ds = _hue_datasets()
    fig, ani = hyp.plot(ds, '-', hue=_hue_for(ds), linewidth=[0.5, 0.5, 5.0],
                        animate=True, duration=1, frame_rate=2, show=False)
    ax = _ax(fig)
    ani._func(1, *ani._args)
    assert _overlay_widths(ax) == pytest.approx([0.5, 0.5, 5.0])


def test_animated_hue_trails_share_their_head_linewidth():
    """matplotlib_backend.py:1639 gives each trail its head's linewidth; the
    hue overlay must too (3 heads + 3 trails, in that order)."""
    ds = _hue_datasets()
    fig, ani = hyp.plot(ds, '-', hue=_hue_for(ds), linewidth=[0.5, 0.5, 5.0],
                        chemtrails=True, animate=True, duration=1,
                        frame_rate=2, show=False)
    ax = _ax(fig)
    ani._func(1, *ani._args)
    assert _overlay_widths(ax) == pytest.approx([0.5, 0.5, 5.0,
                                                 0.5, 0.5, 5.0])


def test_animated_hue_default_width_matches_the_artist_it_replaces():
    """With no linewidth=, the hidden head Line2Ds are 1.0 (the backend's
    pop default, matplotlib_backend.py:1603). The overlay must agree; it
    used to render at rcParams 1.5."""
    ds = _hue_datasets()
    fig, ani = hyp.plot(ds, '-', hue=_hue_for(ds), animate=True,
                        duration=1, frame_rate=2, show=False)
    ax = _ax(fig)
    ani._func(1, *ani._args)
    hidden = [ln.get_linewidth() for ln in ax.lines]
    assert hidden == pytest.approx([1.0, 1.0, 1.0])
    assert _overlay_widths(ax) == pytest.approx(hidden)


def test_static_continuous_hue_linewidth_still_correct():
    """Control: the bug is animation-only, so this passes before AND after."""
    ds = _hue_datasets()
    fig = hyp.plot(ds, '-', hue=_hue_for(ds), linewidth=[0.5, 0.5, 5.0],
                   show=False)
    assert _static_widths(_ax(fig), len(ds)) == pytest.approx([0.5, 0.5, 5.0])


def test_2d_animated_hue_honors_per_dataset_linewidth():
    """The 2-D twin pops linewidth at matplotlib_backend.py:2197-2201."""
    ds = _hue_datasets()
    fig, ani = hyp.plot(ds, '-', hue=_hue_for(ds), linewidth=[0.5, 0.5, 5.0],
                        ndims=2, animate=True, duration=1, frame_rate=2,
                        show=False)
    ax = fig.axes[0]
    assert not hasattr(ax, 'zaxis'), 'expected a 2-D axes'
    ani._func(1, *ani._args)
    assert _overlay_widths(ax) == pytest.approx([0.5, 0.5, 5.0])
