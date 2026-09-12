"""1.1 release review, round 7: two defects pinned at the public API on
both backends. No mocks -- every assertion reads the drawn artists/traces,
the returned bundle, or a real reducer instance's own fit counter.

1 ``panels=`` decided each cell's projection from the RAW column count,
  before the analysis pipeline ran (a round-6 regression). Two raw columns
  through a feature-expanding ``manip='Delay'`` come out three wide, and
  the 2-D cells then refused the 3-D rows (matplotlib, shared fit),
  silently reduced them to 2-D despite ``ndims=3`` (matplotlib,
  independent fit), or hit plotly's "Trace type 'scatter3d' is not
  compatible with subplot type 'xy'". The cells now follow the ANALYZED
  data, in every mode (shared/independent fits, reducer comparisons),
  with the requested ``ndims`` and the fitted pipeline kept and no panel
  fitted twice.
2 composing a second call into the figure/axes/cell a ``fmt='r-'`` call
  drew into gave the second dataset the SECOND palette colour on plotly
  (and in an initially empty ``hyp.subplots`` cell on both backends),
  where the single call ``fmt=['r-', '-']`` gives it the first: the
  colour-cycle count recorded every drawn dataset, although a colour
  letter consumes no palette slot. Only datasets coloured FROM the cycle
  count now, on both backends, for a plain figure and a cell alike.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt                                 # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402
from matplotlib.colors import to_rgb                            # noqa: E402
from sklearn.decomposition import PCA                           # noqa: E402

import hypertools as hyp                                        # noqa: E402
from hypertools.plot.plotly_backend import _rgb_triplet         # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


BACKENDS = ('matplotlib', 'plotly')
PALETTE = ['navy', 'gold', 'green']
DELAY3 = {'model': 'Delay', 'kwargs': {'dims': 3}}


def _walks(n=2, rows=20, cols=2, seed=701):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, cols)).cumsum(0) for _ in range(n)]


def _rgb255(color):
    """A matplotlib colour spelling or a plotly ``rgba()`` string as an
    integer ``(r, g, b)`` triplet, so the two backends' colours compare."""
    if isinstance(color, str) and color.startswith(('rgb(', 'rgba(')):
        return tuple(int(c) for c in _rgb_triplet(color))
    return tuple(int(round(c * 255)) for c in to_rgb(color))


def _plotly_data_traces(fig):
    """The DATA traces of a plotly figure (the frame/box traces carry
    ``hoverinfo='skip'``)."""
    return [tr for tr in fig.data
            if getattr(tr, 'hoverinfo', None) != 'skip']


def _line_colors(fig, backend):
    """The data-line colours a figure draws, in drawing order, on either
    backend (matplotlib: the `Line2D`s of every visible axes; plotly: the
    data traces)."""
    if backend == 'matplotlib':
        return [_rgb255(ln.get_color()) for ax in fig.axes
                if ax.get_visible() for ln in ax.lines]
    out = []
    for tr in _plotly_data_traces(fig):
        color = tr.line.color if 'lines' in (tr.mode or '') \
            else tr.marker.color
        # a hue-coloured trace carries one colour PER POINT: not a
        # single colour, so it reads as None here
        out.append(_rgb255(color) if isinstance(color, str) else None)
    return out


def _cell_kinds(fig, backend):
    """What the grid's cells are: matplotlib axes projection names, or
    the plotly data-trace types (``scatter3d`` only lives in a scene)."""
    if backend == 'matplotlib':
        return [ax.name for ax in fig.axes if ax.get_visible()]
    return sorted({tr.type for tr in _plotly_data_traces(fig)})


def _drawn_points(fig, backend):
    """The 3-D coordinates of each panel's first data trace."""
    if backend == 'matplotlib':
        return [np.column_stack(ax.lines[0].get_data_3d())
                for ax in fig.axes if ax.get_visible()]
    return [np.column_stack([tr.x, tr.y, tr.z])
            for tr in _plotly_data_traces(fig)]


class CountingPCA(PCA):
    """A real sklearn reducer that counts its own fits: the observable
    for "no panel is fitted twice"."""

    def __init__(self, n_components=None):
        super().__init__(n_components=n_components)
        self.fits = 0

    def fit(self, X, y=None):
        self.fits += 1
        return super().fit(X)

    def fit_transform(self, X, y=None):
        self.fits += 1
        return super().fit_transform(X)


# --------------------------- 1: panel cells follow the ANALYZED data

@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('fit', ('shared', 'independent'))
def test_feature_expanding_manip_gets_3d_cells(backend, fit):
    """Two raw columns, ``manip='Delay'`` (dims=3), ``reduce='PCA'``,
    ``ndims=3``: 3-D cells drawing the same 3-wide rows the single-axes
    call draws (before: refused / silently 2-D / plotly 'xy' cell)."""
    data = _walks(2)
    single = hyp.plot(data, manip=DELAY3, reduce='PCA', ndims=3,
                      backend=backend, show=False, return_model=True)
    assert [np.shape(a) for a in single['xform_data']] == [(18, 3), (18, 3)]
    bundle = hyp.plot(data, panels=True, panel_fit=fit, manip=DELAY3,
                      reduce='PCA', ndims=3, backend=backend, show=False,
                      return_model=True)
    fig = bundle['fig']
    assert [np.shape(a) for a in bundle['xform_data']] == [(18, 3), (18, 3)]
    for model in bundle['panel_models']:
        assert [np.shape(a) for a in model['xform_data']] == [(18, 3)]
        assert model['pipeline'] is not None
    if backend == 'matplotlib':
        assert _cell_kinds(fig, backend) == ['3d', '3d']
    else:
        assert _cell_kinds(fig, backend) == ['scatter3d']
    # every panel draws x, y AND z, none of them degenerate (the drawn
    # line is the display-interpolated trajectory, so only its width and
    # spread are the data's)
    for p in _drawn_points(fig, backend):
        assert p.shape[1] == 3 and p.shape[0] >= 18
        assert (np.ptp(p, axis=0) > 0).all()
    if fit == 'shared':
        # the shared fit IS the single-axes call's fit: same rows
        for got, want in zip(bundle['xform_data'], single['xform_data']):
            assert np.allclose(got, want)
        assert bundle['pipeline'] is not None
        assert all(m['pipeline'] is bundle['pipeline']
                   for m in bundle['panel_models'])


@pytest.mark.parametrize('backend', BACKENDS)
def test_independent_fit_keeps_the_requested_ndims(backend):
    """``panel_fit='independent'`` + ``ndims=3`` on Delay-expanded
    2-column data: each panel is fitted on its own and comes out 3-wide,
    exactly as its own single-axes call does (before: matplotlib silently
    returned (18, 2) per panel on rectilinear axes)."""
    data = _walks(2, seed=702)
    bundle = hyp.plot(data, panels=True, panel_fit='independent',
                      manip=DELAY3, reduce='PCA', ndims=3, backend=backend,
                      show=False, return_model=True)
    for i, model in enumerate(bundle['panel_models']):
        own = hyp.plot(data[i], manip=DELAY3, reduce='PCA', ndims=3,
                       backend=backend, show=False, return_model=True)
        assert np.allclose(model['xform_data'][0], own['xform_data'][0])
        assert model['pipeline'] is not None
    assert bundle['panel_models'][0]['pipeline'] \
        is not bundle['panel_models'][1]['pipeline']


@pytest.mark.parametrize('backend', BACKENDS)
def test_reducer_comparison_gets_3d_cells(backend):
    """A list-valued ``reduce=`` (one panel per reducer) on Delay-expanded
    2-column data: every panel draws every dataset in 3-D, and each
    panel's bundle carries that reducer's own fitted pipeline."""
    data = _walks(2, seed=703)
    bundle = hyp.plot(data, panels=True, reduce=['PCA', 'IncrementalPCA'],
                      manip=DELAY3, ndims=3, backend=backend, show=False,
                      return_model=True)
    fig = bundle['fig']
    if backend == 'matplotlib':
        assert _cell_kinds(fig, backend) == ['3d', '3d']
        assert [len(ax.lines) for ax in fig.axes if ax.get_visible()] \
            == [2, 2]
    else:
        assert _cell_kinds(fig, backend) == ['scatter3d']
        assert len(_plotly_data_traces(fig)) == 4
    for model in bundle['panel_models']:
        assert [np.shape(a) for a in model['xform_data']] \
            == [(18, 3), (18, 3)]
        assert model['pipeline'] is not None
    assert bundle['panel_models'][0]['pipeline'] \
        is not bundle['panel_models'][1]['pipeline']


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('fit', ('shared', 'independent'))
def test_pipeline_input_gets_3d_cells(backend, fit):
    """The same through ``pipeline=``: a Delay -> PCA(3) pipeline fitted
    by the grid expands 2 columns to 3, and the cells follow. A pipeline
    whose reduce stage keeps MORE than 3 (a bare 'PCA' keeps all 6 Delay
    columns) is drawn through the single-axes call's own 3-D display
    projection, as ``ndims > 3`` is."""
    data = _walks(2, seed=704)
    pipe = hyp.Pipeline([('manip', DELAY3),
                         ('reduce', {'model': 'PCA',
                                     'kwargs': {'n_components': 3}})])
    bundle = hyp.plot(data, panels=True, panel_fit=fit, pipeline=pipe,
                      ndims=3, backend=backend, show=False,
                      return_model=True)
    assert [np.shape(a) for a in bundle['xform_data']] == [(18, 3), (18, 3)]
    if backend == 'matplotlib':
        assert _cell_kinds(bundle['fig'], backend) == ['3d', '3d']
    else:
        assert _cell_kinds(bundle['fig'], backend) == ['scatter3d']
    wide = hyp.Pipeline([('manip', DELAY3), ('reduce', 'PCA')])
    single = hyp.plot(data, pipeline=wide, ndims=3, backend=backend,
                      show=False, return_model=True)
    assert [np.shape(a) for a in single['xform_data']] == [(18, 6), (18, 6)]
    assert [np.shape(a) for a in single['trace_data']] == [(18, 3), (18, 3)]
    wide = hyp.Pipeline([('manip', DELAY3), ('reduce', 'PCA')])
    bundle = hyp.plot(data, panels=True, panel_fit=fit, pipeline=wide,
                      ndims=3, backend=backend, show=False,
                      return_model=True)
    assert [np.shape(a) for a in bundle['xform_data']] == [(18, 3), (18, 3)]
    if backend == 'matplotlib':
        assert _cell_kinds(bundle['fig'], backend) == ['3d', '3d']
    else:
        assert _cell_kinds(bundle['fig'], backend) == ['scatter3d']
    if fit == 'shared':
        for got, want in zip(bundle['xform_data'], single['trace_data']):
            assert np.allclose(got, want)


@pytest.mark.parametrize('fit', ('shared', 'independent'))
def test_no_panel_is_fitted_twice(fit):
    """Deciding the cells from the analyzed data adds no fit: a real
    reducer instance counts ONE fit for the shared grid and one PER PANEL
    for the independent grid."""
    data = _walks(2, seed=705)
    reducer = CountingPCA(n_components=3)
    bundle = hyp.plot(data, panels=True, panel_fit=fit, manip=DELAY3,
                      reduce=reducer, ndims=3, show=False,
                      return_model=True)
    assert reducer.fits == (1 if fit == 'shared' else 2)
    assert [np.shape(a) for a in bundle['xform_data']] == [(18, 3), (18, 3)]
    assert _cell_kinds(bundle['fig'], 'matplotlib') == ['3d', '3d']


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('fit', ('shared', 'independent'))
def test_narrow_data_still_gets_2d_cells(backend, fit):
    """The round-6 rule is intact: 2-column data with no feature-expanding
    stage keeps 2-D cells (the single-axes call draws it on 2-D axes)."""
    data = _walks(2, seed=706)
    bundle = hyp.plot(data, panels=True, panel_fit=fit, backend=backend,
                      show=False, return_model=True)
    assert [np.shape(a) for a in bundle['xform_data']] == [(20, 2), (20, 2)]
    if backend == 'matplotlib':
        assert _cell_kinds(bundle['fig'], backend) \
            == ['rectilinear', 'rectilinear']
    else:
        assert _cell_kinds(bundle['fig'], backend) == ['scatter']


@pytest.mark.parametrize('backend', BACKENDS)
def test_unequal_widths_under_independent_fit_share_3d_cells(backend):
    """Independent fits of a 2-column and a 3-column dataset with
    ``reduce=None``: one 3-D grid, the narrow panel drawn flat in its
    cell on both backends (plotly's scene cell used to refuse it)."""
    rng = np.random.default_rng(707)
    data = [rng.normal(size=(20, 2)).cumsum(0),
            rng.normal(size=(20, 3)).cumsum(0)]
    bundle = hyp.plot(data, panels=True, panel_fit='independent',
                      reduce=None, backend=backend, show=False,
                      return_model=True)
    if backend == 'matplotlib':
        assert _cell_kinds(bundle['fig'], backend) == ['3d', '3d']
    else:
        assert _cell_kinds(bundle['fig'], backend) == ['scatter3d']
    pts = _drawn_points(bundle['fig'], backend)
    assert [p.shape[1] for p in pts] == [3, 3]
    # the narrow panel's z is flat; its x/y are the data's
    assert np.ptp(pts[0][:, 2]) == 0
    assert np.ptp(pts[0][:, 0]) > 0 and np.ptp(pts[0][:, 1]) > 0
    assert (np.ptp(pts[1], axis=0) > 0).all()


# ---------------------------- 2: a fmt colour letter takes no palette slot

def _target(fig, backend):
    """What a second call composes into: the first axes (matplotlib) or
    the figure (plotly)."""
    return fig.axes[0] if backend == 'matplotlib' else fig


@pytest.mark.parametrize('backend', BACKENDS)
def test_fmt_letter_then_uncoloured_call_matches_the_single_call(backend):
    """``hyp.plot(a, fmt='r-')`` then ``hyp.plot(b, ax=...)`` with the
    same palette draws ``b`` in the FIRST palette colour -- what the
    single call ``fmt=['r-', '-']`` draws it in (before: gold, the second
    colour, on plotly)."""
    a, b = _walks(2, cols=3, seed=708)
    single = hyp.plot([a, b], fmt=['r-', '-'], palette=PALETTE,
                      backend=backend, show=False)
    want = _line_colors(single, backend)
    assert want == [_rgb255('r'), _rgb255('navy')]
    fig = hyp.plot(a, fmt='r-', palette=PALETTE, backend=backend,
                   show=False)
    hyp.plot(b, ax=_target(fig, backend), palette=PALETTE, backend=backend,
             show=False)
    assert _line_colors(fig, backend) == want


@pytest.mark.parametrize('backend', BACKENDS)
def test_fmt_letter_in_an_empty_subplots_cell(backend):
    """The same composition into an initially empty ``hyp.subplots`` cell
    (before: gold on BOTH backends)."""
    a, b = _walks(2, cols=3, seed=709)
    fig, axes = hyp.subplots(1, 1, backend=backend)
    hyp.plot(a, fmt='r-', palette=PALETTE, ax=axes[0], backend=backend,
             show=False)
    hyp.plot(b, palette=PALETTE, ax=axes[0], backend=backend, show=False)
    assert _line_colors(fig, backend) == [_rgb255('r'), _rgb255('navy')]


@pytest.mark.parametrize('backend', BACKENDS)
def test_only_cycle_coloured_datasets_consume_slots(backend):
    """A mixed call (one lettered, one unlettered dataset) consumes ONE
    slot: the third dataset composed in afterwards is gold, as in the
    single call ``fmt=['r-', '-', '-']``."""
    a, b, c = _walks(3, cols=3, seed=710)
    single = hyp.plot([a, b, c], fmt=['r-', '-', '-'], palette=PALETTE,
                      backend=backend, show=False)
    want = _line_colors(single, backend)
    assert want == [_rgb255('r'), _rgb255('navy'), _rgb255('gold')]
    fig = hyp.plot([a, b], fmt=['r-', '-'], palette=PALETTE,
                   backend=backend, show=False)
    hyp.plot(c, ax=_target(fig, backend), palette=PALETTE, backend=backend,
             show=False)
    assert _line_colors(fig, backend) == want


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('pin', ('color', 'hue'))
def test_explicit_colour_and_hue_consume_no_slot(backend, pin):
    """An explicit ``color=`` or a ``hue=`` colouring pins its dataset
    without touching the cycle, so the next call starts the palette."""
    a, b = _walks(2, cols=3, seed=711)
    first = ({'color': 'k'} if pin == 'color'
             else {'hue': np.arange(len(a))})
    fig = hyp.plot(a, palette=PALETTE, backend=backend, show=False,
                   **first)
    hyp.plot(b, ax=_target(fig, backend), palette=PALETTE, backend=backend,
             show=False)
    assert _line_colors(fig, backend)[-1] == _rgb255('navy')


@pytest.mark.parametrize('backend', BACKENDS)
def test_uncoloured_datasets_still_advance_the_palette(backend):
    """The ordinary count is unchanged, and now the same on a plain figure
    as in a cell on both backends: two uncoloured calls into one figure
    draw navy then gold, like the single two-dataset call."""
    a, b = _walks(2, cols=3, seed=712)
    single = hyp.plot([a, b], palette=PALETTE, backend=backend, show=False)
    want = _line_colors(single, backend)
    assert want == [_rgb255('navy'), _rgb255('gold')]
    fig = hyp.plot(a, palette=PALETTE, backend=backend, show=False)
    hyp.plot(b, ax=_target(fig, backend), palette=PALETTE, backend=backend,
             show=False)
    assert _line_colors(fig, backend) == want
    grid, axes = hyp.subplots(1, 1, backend=backend)
    hyp.plot(a, palette=PALETTE, ax=axes[0], backend=backend, show=False)
    hyp.plot(b, palette=PALETTE, ax=axes[0], backend=backend, show=False)
    assert _line_colors(grid, backend) == want
