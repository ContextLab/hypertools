"""1.2.0 review, round 3: a `panels=` grid's color resolution runs in the
panels' own drawing mode and stops before any geometry, categories are
identified by themselves, a matrix `hue=` is resolved once for the grid --
plus the single-axes defects found on the way (an RGB tuple under
`colorbar=`, a left/right animated title, a per-frame caption's extra
`on_frame=` call).

Every test makes the real call and reads the real figure back; call counts
are taken with `sys.setprofile` on the real functions (nothing is replaced).
The broad "every panel artist has the single-axes call's color" check is
`tests/test_panel_color_invariant.py`.
"""
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgb

import hypertools as hyp
from hypertools.plot import matplotlib_backend, plotly_backend
from hypertools.plot import plot as plot_module
from hypertools.plot.plot import (SHARED_COLORBAR_GID, SHARED_LEGEND_GID,
                                  _title_artist)

X = np.random.default_rng(0).normal(size=(20, 2)).cumsum(0)
ANIM = dict(animate=True, duration=1, frame_rate=4)


@pytest.fixture(autouse=True)
def _close_figures():
    """Leave no figure behind."""
    yield
    plt.close('all')


def _quiet(*args, **kwargs):
    """`hyp.plot(..., show=False)` with its warnings ignored."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return hyp.plot(*args, show=False, **kwargs)


def _shared_colorbar(fig):
    """The grid's one colorbar axes."""
    found = [ax for ax in fig.axes if ax.get_gid() == SHARED_COLORBAR_GID]
    assert len(found) == 1
    return found[0]


def _swatches(cax):
    """The RGB swatches of a discrete colorbar axes, in drawn order."""
    colors = []
    for coll in cax.collections:
        array = coll.get_array()
        if array is None or not len(array):
            continue
        rgba = coll.cmap(coll.norm(np.asarray(array).ravel()))
        colors.extend(tuple(round(float(v), 3) for v in c[:3])
                      for c in rgba)
    return colors


class _Calls:
    """Count the real calls of the given functions made inside a ``with``
    block, with `sys.setprofile` -- the functions themselves run
    untouched."""

    def __init__(self, **functions):
        self._codes = {name: fn.__code__ for name, fn in functions.items()}
        self.count = dict.fromkeys(functions, 0)

    def _profile(self, frame, event, arg):
        """Tally a ``call`` event of one of the watched functions."""
        if event == 'call':
            for name, code in self._codes.items():
                if frame.f_code is code:
                    self.count[name] += 1

    def __enter__(self):
        self._previous = sys.getprofile()
        sys.setprofile(self._profile)
        return self

    def __exit__(self, *exc):
        sys.setprofile(self._previous)
        return False


# --------------------------------------------------------------------------
# Codex 1: the resolution runs in the panels' drawing mode
# --------------------------------------------------------------------------

def test_animated_mixture_clusters_take_a_colorbar():
    """An animated mixture clustering is quantized into groups, which a
    discrete colorbar can show; the static one is a continuous blend,
    which it cannot. The grid resolved its colors statically whatever
    the panels drew, and refused the animated grid's colorbar."""
    kwargs = dict(reduce=None, ndims=2, cluster='GaussianMixture',
                  n_clusters=2, random_state=0, colorbar=True)
    anim = _quiet([X, X + 3], 'o', panels=True, **kwargs, **ANIM)
    single = _quiet([X, X + 3], 'o', **kwargs, **ANIM)
    grid_swatches = _swatches(_shared_colorbar(anim.figure))
    assert grid_swatches
    # the one colorbar is the single-axes call's
    assert grid_swatches == _swatches(single.figure.axes[-1])
    # ...and every panel draws its groups in those colors
    anim.draw_frame(anim.n_frames - 1)
    drawn = {tuple(np.round(to_rgb(line.get_color()), 3))
             for ax in anim.figure.axes[:2] for line in ax.lines
             if len(line.get_xdata())}
    assert drawn and drawn <= set(grid_swatches)

    with pytest.raises(ValueError, match='colorbar is not supported'):
        _quiet([X, X + 3], 'o', panels=True, **kwargs)


# --------------------------------------------------------------------------
# Codex 2: a category is identified by itself, never by its legend label
# --------------------------------------------------------------------------

@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_two_categories_with_one_legend_name_keep_their_colors(backend):
    fig = _quiet([X, X + 3], 'o', panels=True, reduce=None, ndims=2,
                 hue=[['a'] * 20, ['b'] * 20],
                 palette={'a': 'red', 'b': 'blue'},
                 legend=['same', 'same'], colorbar=True, backend=backend)
    if backend == 'plotly':
        drawn = [trace.marker.color for trace in fig.data
                 if trace.name == 'same']
        assert drawn == ['rgba(255,0,0,1.0)', 'rgba(0,0,255,1.0)']
        return
    drawn = [to_rgb(ax.lines[0].get_color()) for ax in fig.axes[:2]]
    assert drawn == [to_rgb('red'), to_rgb('blue')]
    assert _swatches(_shared_colorbar(fig)) == [(1.0, 0.0, 0.0),
                                                (0.0, 0.0, 1.0)]
    # one name for two colors cannot be one shared entry: each panel
    # keeps its own legend, in its own color
    assert not [lg for lg in fig.legends
                if lg.get_gid() == SHARED_LEGEND_GID]
    for ax, color in zip(fig.axes[:2], ('red', 'blue')):
        handle, = ax.get_legend().legend_handles
        assert to_rgb(handle.get_color()) == to_rgb(color)
        assert [t.get_text() for t in ax.get_legend().get_texts()] \
            == ['same']


# --------------------------------------------------------------------------
# Codex 4: MultiIndex frames in a list are datasets like any other
# --------------------------------------------------------------------------

@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_row_multiindex_frames_keep_their_dataset_colors(backend):
    frame = pd.DataFrame(
        X, index=pd.MultiIndex.from_product([['a', 'b'], range(10)]))
    kwargs = dict(reduce=None, ndims=2, legend=True, colorbar=True,
                  backend=backend)
    grid = _quiet([frame, frame + 3], 'o', panels=True, **kwargs)
    single = _quiet([frame, frame + 3], 'o', **kwargs)
    if backend == 'plotly':
        want = [t.marker.color for t in single.data if t.name in ('1', '2')]
        got = [t.marker.color for t in grid.data if t.name in ('1', '2')]
        assert len(set(want)) == 2 and got == want
        return
    want = [to_rgb(line.get_color()) for line in single.axes[0].lines]
    got = [to_rgb(ax.lines[0].get_color()) for ax in grid.axes[:2]]
    assert len(set(want)) == 2 and got == want
    assert _swatches(_shared_colorbar(grid)) \
        == [tuple(round(float(v), 3) for v in c) for c in want]


# --------------------------------------------------------------------------
# Codex 5: a matrix hue is resolved once for the grid
# --------------------------------------------------------------------------

@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_one_rgb_hue_row_is_one_color_in_every_panel(backend):
    """Rows 19 and 20 of the hue are both ``[1, 1, 1]``; the grid's range
    is 0..2, so both are mid gray, as the single-axes call draws them --
    not white in the first panel and black in the second."""
    hue = np.repeat(np.r_[np.linspace(0, 1, 20),
                          np.linspace(1, 2, 20)][:, None], 3, axis=1)
    fig = _quiet([X, X + 3], 'o', panels=True, reduce=None, ndims=2,
                 hue=hue, hue_mode='rgb', backend=backend)
    if backend == 'plotly':
        first, second = [t.marker.color for t in fig.data[:2]]
        assert first[0] == 'rgb(0,0,0)' and second[-1] == 'rgb(255,255,255)'
        assert first[-1] == second[0] == 'rgb(128,128,128)'
        return
    first, second = [ax.collections[0].get_facecolor()[:, :3]
                     for ax in fig.axes[:2]]
    np.testing.assert_allclose(first[0], 0.0)
    np.testing.assert_allclose(second[-1], 1.0)
    np.testing.assert_allclose(first[-1], 0.5)
    np.testing.assert_allclose(second[0], 0.5)


def test_a_wide_matrix_hue_is_reduced_once_for_the_grid():
    """More than three columns are reduced to RGB by ONE fit over every
    panel's rows; each panel refitting its own gave unrelated colors."""
    rng = np.random.default_rng(3)
    hue = rng.normal(size=(40, 6)).cumsum(axis=0)
    kwargs = dict(reduce=None, ndims=2, hue=hue)
    grid = _quiet([X, X + 3], 'o', panels=True, **kwargs)
    single = _quiet([X, X + 3], 'o', **kwargs)
    want = [c.get_facecolor()[:, :3] for c in single.axes[0].collections]
    got = [ax.collections[0].get_facecolor()[:, :3] for ax in grid.axes[:2]]
    assert len(want) == 2
    for panel, reference in zip(got, want):
        np.testing.assert_allclose(panel, reference, atol=1e-9)


def test_signed_mixture_weights_blend_alike_in_every_panel():
    """`mat2colors` shifts every row of a matrix that holds ANY negative
    entry; a panel whose own rows hold none must still be shifted when the
    grid's matrix is signed."""
    rng = np.random.default_rng(5)
    hue = rng.dirichlet([1.0, 1.0, 1.0], size=40)
    hue[20:] -= 0.4
    kwargs = dict(reduce=None, ndims=2, hue=hue)
    grid = _quiet([X, X + 3], 'o', panels=True, **kwargs)
    single = _quiet([X, X + 3], 'o', **kwargs)
    want = [c.get_facecolor()[:, :3] for c in single.axes[0].collections]
    got = [ax.collections[0].get_facecolor()[:, :3] for ax in grid.axes[:2]]
    for panel, reference in zip(got, want):
        np.testing.assert_allclose(panel, reference, atol=1e-9)


# --------------------------------------------------------------------------
# Codex 7: font= reaches the grid's one colorbar
# --------------------------------------------------------------------------

def test_the_shared_colorbar_uses_the_requested_font():
    fig = _quiet([X, X + 3], 'o', panels=True, reduce=None, ndims=2,
                 hue=np.linspace(0, 1, 40),
                 colorbar={'label': 'value'}, font='DejaVu Sans Mono')
    cax = _shared_colorbar(fig)
    fig.canvas.draw()
    texts = [t for t in cax.get_yticklabels() if t.get_text()]
    texts.append(cax.yaxis.label)
    assert len(texts) > 1
    for text in texts:
        assert text.get_fontname() == 'DejaVu Sans Mono', text.get_text()


# --------------------------------------------------------------------------
# Codex 8: resolving the colors draws nothing
# --------------------------------------------------------------------------

def _count_calls(data, **kwargs):
    """How often `plot()` and the two backends' draw entry points run for
    one grid call: ``(plot calls, draw calls)``."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with _Calls(plot=plot_module.plot,
                    mpl=matplotlib_backend._draw,
                    plotly=plotly_backend.plotly_draw) as calls:
            hyp.plot(data, show=False, panels=True, **kwargs)
    return (calls.count['plot'],
            calls.count['mpl'] + calls.count['plotly'])


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_color_resolution_never_draws(backend):
    """A shared-fit grid of n datasets makes the call itself, one probe
    (the fit) and n panel calls; an independent-fit grid one probe per
    panel. With nothing to resolve -- no `hue=`, clusters, legend or
    colorbar -- that is ALL: no call is made for the colors. With
    something to resolve, the resolution is one more `plot()` call (two
    for independent fits, one of them before the fits) and NOT ONE more
    draw: it stops before any geometry."""
    data = [X, X + 3, X - 3]
    n = len(data)
    base = dict(reduce=None, ndims=2, backend=backend)
    hue = dict(hue=np.linspace(0, 1, 60), legend=False, colorbar=True)

    assert _count_calls(data, **base) == (2 + n, 1 + n)
    assert _count_calls(data, **base, **hue) == (3 + n, 1 + n)
    assert _count_calls(data, **base, legend=True) == (3 + n, 1 + n)

    independent = dict(base, panel_fit='independent')
    assert _count_calls(data, **independent) == (1 + 2 * n, 2 * n)
    assert _count_calls(data, **independent, **hue) \
        == (3 + 2 * n, 2 * n)


def test_color_resolution_creates_no_figure():
    """Only the probe's figure and the grid's are ever created."""
    data = [X, X + 3]
    before = len(plt.get_fignums())
    with _Calls(new=plt.figure) as calls:
        fig = _quiet(data, panels=True, reduce=None, ndims=2,
                     hue=np.linspace(0, 1, 40), colorbar=True)
    assert calls.count['new'] <= 2
    assert len(plt.get_fignums()) == before
    assert _shared_colorbar(fig) is not None


# --------------------------------------------------------------------------
# Codex 3 (single axes): an RGB(A) tuple is ONE color under colorbar=
# --------------------------------------------------------------------------

@pytest.mark.parametrize('color', [(0.2, 0.4, 0.6), (0.2, 0.4, 0.6, 1.0)],
                         ids=['rgb-3-datasets', 'rgba-4-datasets'])
@pytest.mark.parametrize('panels', [False, True], ids=['single', 'grid'])
def test_a_color_tuple_as_long_as_the_dataset_count_is_one_color(
        color, panels):
    """Three datasets and a three-number tuple: the tuple is one RGB color
    (as the drawing reads it), not one entry per dataset. The colorbar
    resolved ONE color for three labels and failed on the tick count."""
    data = [X + 3 * i for i in range(len(color))]
    kwargs = dict(panels=True) if panels else {}
    fig = _quiet(data, reduce=None, ndims=2, color=color, colorbar=True,
                 backend='matplotlib', **kwargs)
    cax = _shared_colorbar(fig) if panels else fig.axes[-1]
    assert _swatches(cax) == [tuple(color[:3])] * len(data)
    fig.canvas.draw()
    assert [t.get_text() for t in cax.get_yticklabels()] \
        == [str(i + 1) for i in range(len(data))]
    lines = [line for ax in fig.axes if ax is not cax for line in ax.lines]
    assert len(lines) == len(data)
    assert all(to_rgb(line.get_color()) == tuple(color[:3])
               for line in lines)


# --------------------------------------------------------------------------
# Codex 6 (single axes): room for a tall left/right animated title
# --------------------------------------------------------------------------

@pytest.mark.parametrize('loc', ['left', 'right', 'center'])
@pytest.mark.parametrize('ndims', [2, 3])
def test_a_tall_animated_title_at_any_loc_is_inside_the_canvas(loc, ndims):
    rng = np.random.default_rng(0)
    data = rng.normal(size=(20, ndims)).cumsum(0)
    title = '\n'.join(['Tall title'] * 12)
    anim = _quiet(data, reduce=None, ndims=ndims, title=title,
                  title_kwargs={'loc': loc}, **ANIM)
    fig = anim.figure
    anim.draw_frame(0)
    fig.canvas.draw()
    artist = _title_artist(fig.axes[0], {'loc': loc})
    assert artist.get_text() == title
    box = artist.get_window_extent()
    assert box.height > 150
    assert box.y0 >= 0 and box.y1 <= fig.bbox.height + 0.5, (
        f'the title spans y={box.y0:.0f}..{box.y1:.0f} px on a '
        f'{fig.bbox.height:.0f} px canvas')
