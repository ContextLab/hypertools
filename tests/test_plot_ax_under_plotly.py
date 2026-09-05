"""``ax=`` is refused under the plotly backend instead of being ignored.

A matplotlib Axes cannot host a plotly figure. Until 1.1 the plotly backend
built its own ``plotly.graph_objects.Figure`` and left the caller's axes
empty: on Colab, where ``backend='auto'`` resolves to plotly, a two-panel
before/after layout drawn with ``ax=axes[0]`` / ``ax=axes[1]`` showed two
empty 3-D cubes (measured 2026-09-04). The refusal mirrors the one for
``ax=`` together with ``animate=``.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import hypertools as hyp

matplotlib.use('Agg')

pytest.importorskip('plotly')


def _walk():
    return np.cumsum(np.random.default_rng(0).standard_normal((30, 3)), axis=0)


def test_ax_with_backend_plotly_raises_before_drawing():
    ax = plt.figure().add_subplot(projection='3d')
    with pytest.raises(ValueError, match='plotly backend cannot draw'):
        hyp.plot(_walk(), ax=ax, backend='plotly')
    assert not ax.lines and not ax.collections     # nothing was drawn into it
    plt.close('all')


def test_ax_under_set_interactive_backend_plotly_raises():
    ax = plt.figure().add_subplot(projection='3d')
    with hyp.set_interactive_backend('plotly'):
        with pytest.raises(ValueError, match='plotly backend cannot draw'):
            hyp.plot(_walk(), ax=ax)
    plt.close('all')


def test_ax_still_works_on_matplotlib_and_inside_a_matplotlib_context():
    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    hyp.plot(_walk(), ax=axes[0], backend='matplotlib', show=False)
    with hyp.set_interactive_backend('matplotlib'):
        hyp.plot(_walk(), ax=axes[1], show=False)
    assert axes[0].lines or axes[0].collections
    assert axes[1].lines or axes[1].collections
    plt.close(fig)
