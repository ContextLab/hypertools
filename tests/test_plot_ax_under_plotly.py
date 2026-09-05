"""``ax=`` names the surface to draw into: a matplotlib Axes for the
matplotlib backend, or a plotly Figure (the object an earlier ``hyp.plot``
returned) for the plotly backend, whose traces are appended to it.

Until 1.1 the plotly backend silently ignored a matplotlib Axes: it built its
own ``plotly.graph_objects.Figure`` and left the caller's axes empty (on
Colab, where ``backend='auto'`` resolves to plotly, a two-panel before/after
layout drawn with ``ax=axes[0]`` / ``ax=axes[1]`` showed two empty 3-D
cubes; measured 2026-09-04).
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import hypertools as hyp

matplotlib.use('Agg')

pytest.importorskip('plotly')


def _walk(seed=0):
    return np.cumsum(np.random.default_rng(seed).standard_normal((30, 3)), axis=0)


def test_matplotlib_axes_under_plotly_is_refused_before_drawing():
    ax = plt.figure().add_subplot(projection='3d')
    with pytest.raises(ValueError, match='plotly backend cannot draw into it'):
        hyp.plot(_walk(), ax=ax, backend='plotly')
    assert not ax.lines and not ax.collections     # nothing was drawn into it
    plt.close('all')


def test_matplotlib_axes_inside_set_interactive_backend_plotly_is_refused():
    ax = plt.figure().add_subplot(projection='3d')
    with hyp.set_interactive_backend('plotly'):
        with pytest.raises(ValueError, match='plotly backend cannot draw into it'):
            hyp.plot(_walk(), ax=ax)
    plt.close('all')


def test_a_plotly_figure_as_ax_receives_the_new_traces():
    import plotly.graph_objects as go
    first = hyp.plot(_walk(0), backend='plotly', show=False)
    assert isinstance(first, go.Figure)
    n_before = len(first.data)
    second = hyp.plot(_walk(1), ax=first, backend='plotly', show=False)
    assert second is first                          # drawn INTO the caller's figure
    assert len(first.data) > n_before
    third = hyp.plot([_walk(2), _walk(3)], ax=first, backend='plotly', show=False)
    assert third is first and len(first.data) > n_before + 1


def test_a_plotly_figure_as_ax_with_animate_is_refused():
    first = hyp.plot(_walk(0), backend='plotly', show=False)
    with pytest.raises(ValueError, match='cannot be combined with animate='):
        hyp.plot(_walk(1), ax=first, animate=True, backend='plotly', show=False)


def test_a_plotly_figure_as_ax_under_matplotlib_is_a_type_error():
    first = hyp.plot(_walk(0), backend='plotly', show=False)
    with pytest.raises(TypeError, match='draws with matplotlib'):
        hyp.plot(_walk(1), ax=first, backend='matplotlib', show=False)


def test_ax_still_works_on_matplotlib_and_inside_a_matplotlib_context():
    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    hyp.plot(_walk(), ax=axes[0], backend='matplotlib', show=False)
    with hyp.set_interactive_backend('matplotlib'):
        hyp.plot(_walk(), ax=axes[1], show=False)
    assert axes[0].lines or axes[0].collections
    assert axes[1].lines or axes[1].collections
    plt.close(fig)


def test_a_non_axes_non_figure_ax_is_a_type_error():
    with pytest.raises(TypeError, match='ax= must be'):
        hyp.plot(_walk(), ax='not an axes', show=False)
