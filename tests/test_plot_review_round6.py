"""1.1 release review, round 6: two defects pinned at the public API on
both backends. No mocks -- every assertion reads the drawn artists/traces.

1 the plotly backend dropped the COLOUR letter of a data ``fmt=`` string
  (``'r-'`` drew the palette colour; matplotlib drew red, and the `plot()`
  docstring promises fmt works "exactly as in matplotlib") -- on the
  single-axes path, animated, for per-dataset fmt lists, and inside
  ``panels=`` cells.
2 ``panels=True`` on 2-column (or 1-column) data with the default
  ``ndims=`` built 3-D cells: plotly refused the 2-D traces ("Trace type
  'scatter' is not compatible with subplot type 'scene'") and matplotlib
  drew a flat trajectory inside a cube. The cells now follow the data's
  drawn dimensionality, as the single-axes call does.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt                                 # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402
from matplotlib.colors import to_rgb                            # noqa: E402

import hypertools as hyp                                        # noqa: E402
from hypertools.plot.plotly_backend import _rgb_triplet         # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _walks(n=2, rows=20, cols=3, seed=601):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, cols)).cumsum(0) for _ in range(n)]


def _rgb255(color):
    """A matplotlib colour spelling or a plotly ``rgba()`` string as an
    integer ``(r, g, b)`` triplet, so the two backends' colours compare."""
    if isinstance(color, str) and color.startswith(('rgb(', 'rgba(')):
        return tuple(int(c) for c in _rgb_triplet(color))
    return tuple(int(round(c * 255)) for c in to_rgb(color))


def _mpl_line_colors(fig):
    """The colours of the lines on a matplotlib figure's first axes (a
    static hypertools plot draws its frame as patches/collections, so the
    `Line2D`s are the data lines, one per dataset)."""
    return [_rgb255(ln.get_color()) for ln in fig.axes[0].lines]


def _plotly_data_traces(fig):
    """The DATA traces of a plotly figure: hypertools' frame/box traces
    are drawn in black with ``hoverinfo='skip'`` and no legend entry;
    a data trace keeps its hover."""
    return [tr for tr in fig.data
            if getattr(tr, 'hoverinfo', None) != 'skip']


def _plotly_line_colors(fig):
    out = []
    for tr in _plotly_data_traces(fig):
        color = tr.line.color if 'lines' in (tr.mode or '') \
            else tr.marker.color
        out.append(_rgb255(color))
    return out


# ------------------------------------------ 1: fmt= colour letter, plotly

def test_fmt_colour_letter_single_axes():
    """``fmt='r-'``: plotly's trace is the same red matplotlib's line is,
    not the palette's first colour."""
    x = _walks(1)[0]
    mpl_fig = hyp.plot(x, fmt='r-', show=False)
    ply_fig = hyp.plot(x, fmt='r-', backend='plotly', show=False)
    mpl_colors = _mpl_line_colors(mpl_fig)
    ply_colors = _plotly_line_colors(ply_fig)
    assert mpl_colors == [_rgb255('r')]
    assert ply_colors == mpl_colors
    # ...and the dash/mode halves of the fmt are untouched by the fix
    (trace,) = _plotly_data_traces(ply_fig)
    assert trace.mode == 'lines'
    assert trace.line.dash in (None, 'solid')


def test_fmt_colour_letters_per_dataset_list():
    """``fmt=['g--', 'b:']``: each plotly trace wears its own letter's
    colour AND its own dash, exactly as matplotlib's two lines do."""
    data = _walks(2)
    mpl_fig = hyp.plot(data, fmt=['g--', 'b:'], show=False)
    ply_fig = hyp.plot(data, fmt=['g--', 'b:'], backend='plotly',
                       show=False)
    mpl_lines = mpl_fig.axes[0].lines
    assert [ln.get_linestyle() for ln in mpl_lines] == ['--', ':']
    assert _mpl_line_colors(mpl_fig) == [_rgb255('g'), _rgb255('b')]
    traces = _plotly_data_traces(ply_fig)
    assert [tr.line.dash for tr in traces] == ['dash', 'dot']
    assert _plotly_line_colors(ply_fig) == _mpl_line_colors(mpl_fig)


def test_fmt_colour_letter_animated():
    """An animated plotly call keeps the letter too: the base trace and
    every frame that restyles the line carry red, never the palette."""
    x = _walks(1, rows=12)[0]
    anim = hyp.plot(x, fmt='r-', backend='plotly', animate=True,
                    show=False)
    mpl_anim = hyp.plot(x, fmt='r-', animate=True, show=False)
    mpl_colors = [_rgb255(ln.get_color())
                  for ln in mpl_anim.figure.axes[0].lines
                  if ln.get_color() != 'black']
    assert mpl_colors and set(mpl_colors) == {_rgb255('r')}
    assert _plotly_line_colors(anim) == [_rgb255('r')]
    assert len(anim.frames) > 0
    for frame in anim.frames:
        for tr in frame.data:
            color = getattr(getattr(tr, 'line', None), 'color', None)
            if color is not None:
                assert _rgb255(color) == _rgb255('r')


def test_fmt_colour_letter_inside_panels():
    """Each `panels=` cell honours its own dataset's letter (the panel
    calls re-enter `plot()`, so this is the single-axes fix seen through
    the partitioned fmt list)."""
    data = _walks(2)
    ply_fig = hyp.plot(data, fmt=['g--', 'b:'], panels=True,
                       backend='plotly', show=False)
    mpl_fig = hyp.plot(data, fmt=['g--', 'b:'], panels=True, show=False)
    assert _plotly_line_colors(ply_fig) == [_rgb255('g'), _rgb255('b')]
    assert [_rgb255(ax.lines[0].get_color()) for ax in mpl_fig.axes[:2]] \
        == [_rgb255('g'), _rgb255('b')]


def test_fmt_colour_letter_precedence_matches_matplotlib():
    """The letter beats `palette=` but loses to an explicit `color=` on
    BOTH backends, and a lettered dataset consumes no palette slot: with
    ``fmt=['r-', '-']`` the second dataset takes the palette's FIRST
    colour, as matplotlib's colour cycle hands it out."""
    data = _walks(2)
    for kwargs in (dict(fmt='r-', palette='viridis'),
                   dict(fmt='r-', color=['g', 'b']),
                   dict(fmt=['r-', '-'])):
        mpl_fig = hyp.plot(data, show=False, **kwargs)
        ply_fig = hyp.plot(data, backend='plotly', show=False, **kwargs)
        assert _plotly_line_colors(ply_fig) == _mpl_line_colors(mpl_fig), \
            kwargs
    palette_first = _mpl_line_colors(hyp.plot(data[:1], show=False))[0]
    assert _mpl_line_colors(hyp.plot(data, fmt=['r-', '-'],
                                     show=False))[1] == palette_first


# --------------------------------- 2: panels= cells follow the data width

def _plotly_xy_cells(fig):
    """``(trace types, the xaxis of each data trace, layout scene keys)``."""
    traces = _plotly_data_traces(fig)
    # the SET layout entries (iterating `fig.layout` lists every property
    # name, set or not)
    scenes = sorted(k for k in fig.layout.to_plotly_json()
                    if k.startswith('scene'))
    return ([tr.type for tr in traces],
            [getattr(tr, 'xaxis', None) for tr in traces], scenes)


@pytest.mark.parametrize('fit', ('shared', 'independent'))
def test_panels_two_column_data_default_ndims_plotly(fit):
    """2-column data, no ndims=: xy cells with the traces drawn in them
    (before: ValueError "Trace type 'scatter' is not compatible with
    subplot type 'scene'")."""
    data = _walks(2, cols=2)
    fig = hyp.plot(data, panels=True, panel_fit=fit, backend='plotly',
                   show=False)
    types, xaxes, scenes = _plotly_xy_cells(fig)
    assert types == ['scatter', 'scatter']
    assert xaxes == ['x', 'x2']
    assert scenes == []
    for tr in _plotly_data_traces(fig):
        assert len(tr.x) > 0 and len(tr.x) == len(tr.y)
    # the same cells an explicit ndims=2 builds
    explicit = hyp.plot(data, panels=True, panel_fit=fit, ndims=2,
                        backend='plotly', show=False)
    assert _plotly_xy_cells(explicit) == (types, xaxes, scenes)


@pytest.mark.parametrize('fit', ('shared', 'independent'))
def test_panels_two_column_data_default_ndims_matplotlib(fit):
    """The matplotlib grid gives 2-column data 2-D axes -- what the
    single-axes call draws it on -- not 3-D cubes."""
    data = _walks(2, cols=2)
    single = hyp.plot(data[0], show=False)
    assert single.axes[0].name == 'rectilinear'
    fig = hyp.plot(data, panels=True, panel_fit=fit, show=False)
    axes = [ax for ax in fig.axes if ax.get_visible()]
    assert [ax.name for ax in axes] == ['rectilinear', 'rectilinear']
    assert all(len(ax.lines) >= 1 for ax in axes)
    # the panel draws the same points the single-axes call draws
    assert np.allclose(axes[0].lines[0].get_xydata(),
                       single.axes[0].lines[0].get_xydata())


@pytest.mark.parametrize('backend', ('matplotlib', 'plotly'))
@pytest.mark.parametrize('fit', ('shared', 'independent'))
def test_panels_one_column_data_default_ndims(backend, fit):
    """1-column data draws as an index-vs-value series on the single-axes
    path; the panel grid gives it the same 2-D cells (before: matplotlib
    raised inside Axes3D.plot, plotly refused the scatter traces)."""
    data = _walks(2, cols=1)
    fig = hyp.plot(data, panels=True, panel_fit=fit, backend=backend,
                   show=False)
    if backend == 'plotly':
        types, xaxes, scenes = _plotly_xy_cells(fig)
        assert types == ['scatter', 'scatter']
        assert xaxes == ['x', 'x2']
        assert scenes == []
        for tr in _plotly_data_traces(fig):
            assert len(tr.x) > 0 and len(tr.x) == len(tr.y)
    else:
        axes = [ax for ax in fig.axes if ax.get_visible()]
        assert [ax.name for ax in axes] == ['rectilinear', 'rectilinear']
        single = hyp.plot(data[0], show=False)
        assert np.allclose(axes[0].lines[0].get_xydata(),
                           single.axes[0].lines[0].get_xydata())


@pytest.mark.parametrize('backend', ('matplotlib', 'plotly'))
def test_panels_reducer_list_on_two_column_data(backend):
    """One panel per reducer over 2-column data: 2-D cells too."""
    x = _walks(1, cols=2)[0]
    fig = hyp.plot(x, panels=True, reduce=['PCA', 'IncrementalPCA'],
                   backend=backend, show=False)
    if backend == 'plotly':
        types, xaxes, scenes = _plotly_xy_cells(fig)
        assert types == ['scatter', 'scatter'] and scenes == []
    else:
        assert [ax.name for ax in fig.axes if ax.get_visible()] \
            == ['rectilinear', 'rectilinear']


@pytest.mark.parametrize('backend', ('matplotlib', 'plotly'))
def test_panels_three_column_data_keeps_3d_cells(backend):
    """The unchanged case: 3-column data (and ndims > 3, drawn in 3-D)
    still builds 3-D cells."""
    for data, kwargs in ((_walks(2), {}), (_walks(2, cols=6), {'ndims': 5})):
        fig = hyp.plot(data, panels=True, backend=backend, show=False,
                       **kwargs)
        if backend == 'plotly':
            assert {tr.type for tr in _plotly_data_traces(fig)} \
                == {'scatter3d'}
            assert sorted(k for k in fig.layout.to_plotly_json()
                          if k.startswith('scene')) == ['scene', 'scene2']
        else:
            assert [ax.name for ax in fig.axes if ax.get_visible()] \
                == ['3d', '3d']


def test_panels_two_column_bundle_axes_are_xy(fit='shared'):
    """`return_model=True` on the plotly grid records one (xaxis, yaxis)
    pair per 2-D cell, the bundle shape the 2-D path promises."""
    data = _walks(2, cols=2)
    bundle = hyp.plot(data, panels=True, panel_fit=fit, backend='plotly',
                      return_model=True, show=False)
    assert bundle['panels'] == (1, 2)
    assert all(isinstance(pair, tuple) and len(pair) == 2
               for pair in bundle['axes'])
