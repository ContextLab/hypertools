"""Closing an animated figure must not raise, even when matplotlib's notebook
backend processes the close event twice.

matplotlib's ``nbAgg`` backend (the one hypertools selects in Colab and
classic Jupyter) fires ``close_event`` re-entrantly from
``FigureManagerNbAgg.destroy()``, so ``Animation._stop`` runs twice and the
second call raises ``AttributeError: 'NoneType' object has no attribute
'remove_callback'``. Measured 2026-09-04 on Colab (Python 3.13) and locally
with matplotlib 3.10.8 and 3.11.1, with and without hypertools: after any
displayed hypertools animation the NEXT static-plot cell failed in IPython's
end-of-cell ``plt.close('all')``, and ``show=False`` animated plots failed
inside ``plot()``. ``HyperFuncAnimation._stop`` skips the repeat call.
"""

import textwrap

import matplotlib
import matplotlib.animation
import numpy as np
import pytest
from matplotlib.backend_bases import CloseEvent

import hypertools as hyp
from hypertools.plot.animate import HyperFuncAnimation


def _walk(n=40, d=4, seed=0):
    return np.cumsum(np.random.default_rng(seed).standard_normal((n, d)), axis=0)


def test_plot_animation_is_a_funcanimation_with_idempotent_stop():
    anim = hyp.plot(_walk(), animate='window', show=False)
    fa = anim.animation
    assert isinstance(fa, HyperFuncAnimation)
    assert isinstance(fa, matplotlib.animation.FuncAnimation)
    # plot(show=False) already closed the figure; on GUI/notebook backends
    # that stopped the animation, on Agg it did not -- either way a stop
    # followed by another stop must be a no-op, not an AttributeError.
    fa._stop()
    assert fa.event_source is None
    fa._stop()
    # the re-entrant nbAgg pattern: the close event delivered twice
    CloseEvent('close_event', anim.figure.canvas)._process()
    CloseEvent('close_event', anim.figure.canvas)._process()


def test_plain_matplotlib_stop_is_not_idempotent_so_the_guard_is_needed():
    """Documents the upstream behaviour the subclass exists for: if this
    starts passing, matplotlib fixed it and the guard can go."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    line, = ax.plot([0, 1], [0, 1])
    fa = matplotlib.animation.FuncAnimation(
        fig, lambda i: (line.set_ydata([0, i % 2]),), frames=2, interval=50)
    fa._stop()
    with pytest.raises(AttributeError, match='remove_callback'):
        fa._stop()
    plt.close(fig)


_NOTEBOOK_CELLS = [
    # the Colab sequence: inline backend, then an animation displayed
    # (hypertools switches to nbAgg for it), then an ordinary static plot
    """
    %matplotlib inline
    import matplotlib, matplotlib.pyplot as plt, numpy as np
    import hypertools as hyp
    X = np.cumsum(np.random.default_rng(0).standard_normal((40, 4)), axis=0)
    print('backend', matplotlib.get_backend())
    """,
    """
    anim = hyp.plot(X, animate='window', mpl_backend='nbAgg')
    print('animated ok')
    anim
    """,
    """
    fig = hyp.plot(X, 'o')          # IPython flushes (closes) all figures after this cell
    print('static after animation ok')
    """,
    """
    anim2 = hyp.plot(X, animate='window', mpl_backend='nbAgg', show=False)
    print('show=False animation ok', anim2.n_frames > 0)
    plt.close('all')
    print('close all ok')
    """,
]


def test_nbagg_kernel_survives_closing_animated_figures(tmp_path):
    """Runs the exact Colab sequence in a real ipykernel with nbAgg."""
    nbformat = pytest.importorskip('nbformat')
    nbclient = pytest.importorskip('nbclient')
    pytest.importorskip('ipykernel')
    nb = nbformat.v4.new_notebook()
    for src in _NOTEBOOK_CELLS:
        nb.cells.append(nbformat.v4.new_code_cell(textwrap.dedent(src).strip()))
    nbclient.NotebookClient(nb, timeout=300, kernel_name='python3',
                            allow_errors=True,
                            resources={'metadata': {'path': str(tmp_path)}}).execute()
    errors = [(i, o['ename'], o['evalue']) for i, c in enumerate(nb.cells)
              for o in c.get('outputs', []) if o['output_type'] == 'error']
    assert errors == [], errors
    text = '\n'.join(''.join(o.get('text', '')) for c in nb.cells
                     for o in c.get('outputs', []) if o['output_type'] == 'stream')
    for expected in ('animated ok', 'static after animation ok',
                     'show=False animation ok True', 'close all ok'):
        assert expected in text, text
