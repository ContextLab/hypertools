"""1.1 release review, round 8: three ``panels=``/composition defects
pinned at the public API on both backends. No mocks -- every assertion
reads the drawn artists/traces, the returned bundle, or a real clusterer
instance's own fit counter.

2 ``panels=`` re-clustered every panel's analyzed rows without the
  caller's ``random_state`` (a round-7 regression): the probe that fits
  each panel's pipeline clustered once, seeded, and the panel call then
  clustered AGAIN, unseeded, so ``panel_fit='independent'`` and reducer
  grids drew cluster memberships the seeded single-axes call never draws
  (and every clusterer fitted more often than before). Each panel now
  replays its probe's fitted labels -- the memberships ARE the individual
  call's, on both backends, in every fit mode -- and the bundle reports
  them as ``models['cluster_labels']``.
3 on plotly, a call pinning its colour (``color=`` or a categorical
  ``hue=``) into a figure/cell an ordinary call had drawn into reset the
  palette count, so the NEXT ordinary call restarted the palette (navy
  again where the single call, and matplotlib, give gold). The count is
  now read for every composed call, whether or not it colours from the
  cycle.
4 independent panels of a ONE-column and a three-column dataset crashed
  both backends (only two-column rows were padded into the 3-D grid).
  A one-column series is now drawn in the 3-D cell as row index vs value
  on the cell's floor.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt                                 # noqa: E402
import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402
import pytest                                                   # noqa: E402
from matplotlib.colors import to_rgb                            # noqa: E402
from sklearn.cluster import KMeans                              # noqa: E402

import hypertools as hyp                                        # noqa: E402
from hypertools.plot.plotly_backend import _rgb_triplet         # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


BACKENDS = ('matplotlib', 'plotly')
PALETTE = ['navy', 'gold', 'green']


def _clouds(n=2, rows=24, cols=5, seed=801):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, cols)) for _ in range(n)]


def _rgb255(color):
    if isinstance(color, str) and color.startswith(('rgb(', 'rgba(')):
        return tuple(int(c) for c in _rgb_triplet(color))
    return tuple(int(round(c * 255)) for c in to_rgb(color))


def _plotly_data_traces(fig):
    return [tr for tr in fig.data
            if getattr(tr, 'hoverinfo', None) != 'skip']


def _line_colors(fig, backend):
    """The data-line colours a figure draws, in drawing order."""
    if backend == 'matplotlib':
        return [_rgb255(ln.get_color()) for ax in fig.axes
                if ax.get_visible() for ln in ax.lines]
    out = []
    for tr in _plotly_data_traces(fig):
        color = tr.line.color if 'lines' in (tr.mode or '') \
            else tr.marker.color
        out.append(_rgb255(color) if isinstance(color, str) else None)
    return out


def _cluster_point_sets(fig, backend):
    """Per cell, the drawn clusters as a list of point arrays (one per
    cluster artist/trace), each sorted row-wise so two drawings of the
    same memberships compare equal whatever order the points were drawn
    in; the clusters themselves are ordered by size, then lexically."""
    cells = []
    if backend == 'matplotlib':
        for ax in fig.axes:
            if not ax.get_visible() or not ax.lines:
                continue
            cells.append([np.column_stack(ln.get_data_3d())
                          for ln in ax.lines])
    else:
        by_scene = {}
        for tr in _plotly_data_traces(fig):
            by_scene.setdefault(tr.scene, []).append(
                np.column_stack([tr.x, tr.y, tr.z]))
        cells = [by_scene[k] for k in sorted(by_scene)]
    out = []
    for groups in cells:
        groups = [g[np.lexsort(g.T[::-1])] for g in groups]
        groups.sort(key=lambda g: (len(g), g.round(6).tolist()))
        out.append(groups)
    return out


def _assert_same_clusters(got, want):
    assert len(got) == len(want)
    assert [len(g) for g in got] == [len(w) for w in want]
    for g, w in zip(got, want):
        assert np.allclose(g, w)


class CountingKMeans(KMeans):
    """A real sklearn clusterer that counts its own fits: the observable
    for "a panel grid clusters no more often than its probes do"."""

    def __init__(self, n_clusters=3, random_state=802, n_init=1):
        super().__init__(n_clusters=n_clusters, random_state=random_state,
                         n_init=n_init)
        self.fits = 0

    def fit(self, X, y=None, sample_weight=None):
        self.fits += 1
        return super().fit(X, y, sample_weight=sample_weight)


CLUSTER = dict(cluster='KMeans', n_clusters=3, random_state=88)
DRAW = dict(reduce='PCA', ndims=3, fmt='o', antialias=False, show=False,
            return_model=True)


# --------------------------- 2: panels replay the probe's seeded clustering

@pytest.mark.parametrize('backend', BACKENDS)
def test_independent_panels_draw_the_individual_calls_clusters(backend):
    """Every independent panel's drawn clusters (which points, in which
    group) are exactly the seeded single-axes call's for that dataset,
    and its bundle carries those memberships."""
    data = _clouds(seed=803)
    singles = [hyp.plot(d, backend=backend, **CLUSTER, **DRAW)
               for d in data]
    grid = hyp.plot(data, panels=2, panel_fit='independent',
                    backend=backend, **CLUSTER, **DRAW)
    want = [_cluster_point_sets(s['fig'], backend)[0] for s in singles]
    got = _cluster_point_sets(grid['fig'], backend)
    assert len(got) == 2
    for g, w in zip(got, want):
        assert len(g) == 3
        _assert_same_clusters(g, w)
    for panel, single in zip(grid['panel_models'], singles):
        assert (list(panel['models']['cluster_labels'])
                == list(single['models']['cluster_labels']))
    # the individual call's clustering is seeded and non-trivial: the
    # comparison above can only pass by matching it
    sizes = sorted(len(g) for g in want[0])
    assert sizes == sorted(len(g) for g in got[0]) and len(set(sizes)) > 1


@pytest.mark.parametrize('backend', BACKENDS)
def test_reducer_grid_panels_draw_the_individual_calls_clusters(backend):
    """A per-reducer grid: each panel's clusters are the seeded joint
    call's with that reducer (every dataset in the panel)."""
    data = _clouds(seed=804)
    reducers = ['PCA', 'IncrementalPCA']
    singles = [hyp.plot(data, backend=backend,
                        **{**DRAW, 'reduce': spec}, **CLUSTER)
               for spec in reducers]
    grid = hyp.plot(data, panels=2, backend=backend,
                    **{**DRAW, 'reduce': reducers}, **CLUSTER)
    got = _cluster_point_sets(grid['fig'], backend)
    assert len(got) == 2
    for g, single in zip(got, singles):
        _assert_same_clusters(g, _cluster_point_sets(single['fig'],
                                                     backend)[0])
    for panel, single in zip(grid['panel_models'], singles):
        assert (list(panel['models']['cluster_labels'])
                == list(single['models']['cluster_labels']))


@pytest.mark.parametrize('backend', BACKENDS)
def test_shared_panels_slice_the_one_seeded_clustering(backend):
    """A shared fit clusters ONCE across every dataset (the joint call);
    each panel replays its dataset's slice of those labels."""
    data = _clouds(seed=805)
    joint = hyp.plot(data, backend=backend, **CLUSTER, **DRAW)
    labels = list(joint['models']['cluster_labels'])
    assert len(labels) == 48
    grid = hyp.plot(data, panels=2, panel_fit='shared', backend=backend,
                    **CLUSTER, **DRAW)
    got = [list(m['models']['cluster_labels']) for m in grid['panel_models']]
    assert got == [labels[:24], labels[24:]]
    # ...and the drawn groups hold exactly those rows: as many drawn
    # clusters per panel as labels its slice has, with the right sizes
    for cell, slice_ in zip(_cluster_point_sets(grid['fig'], backend), got):
        counts = sorted(slice_.count(k) for k in set(slice_))
        assert sorted(len(g) for g in cell) == counts


@pytest.mark.parametrize('backend', BACKENDS)
def test_two_identical_seeded_grids_agree(backend):
    """The seed contract itself: the same seeded grid twice draws the
    same memberships (it did not, once each panel re-clustered
    unseeded)."""
    data = _clouds(seed=806)
    grids = [hyp.plot(data, panels=2, panel_fit='independent',
                      backend=backend, **CLUSTER, **DRAW) for _ in range(2)]
    first, second = (_cluster_point_sets(g['fig'], backend) for g in grids)
    for a, b in zip(first, second):
        _assert_same_clusters(a, b)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('mode', ('shared', 'independent', 'reducers'))
def test_no_panel_clusters_again(backend, mode):
    """A real clusterer instance fits exactly as often as the grid's
    probes (each an ordinary single-axes call) fit it: once per probe
    call's own count -- the panels' draws add NO fit."""
    data = _clouds(seed=807)
    baseline = CountingKMeans()
    hyp.plot(data, cluster=baseline, backend=backend, **DRAW)
    per_call = baseline.fits
    assert per_call >= 1
    counter = CountingKMeans()
    if mode == 'reducers':
        hyp.plot(data, panels=2, cluster=counter, backend=backend,
                 **{**DRAW, 'reduce': ['PCA', 'IncrementalPCA']})
        probes = 2
    else:
        hyp.plot(data, panels=2, panel_fit=mode, cluster=counter,
                 backend=backend, **DRAW)
        probes = 1 if mode == 'shared' else 2
    assert counter.fits == probes * per_call


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('spec', (
    dict(n_clusters=3),
    dict(cluster={'model': 'KMeans', 'kwargs': {'n_clusters': 3}}),
    dict(cluster='GaussianMixture', n_clusters=2),
))
def test_every_cluster_spelling_replays(backend, spec):
    """``n_clusters=`` alone, a dict spec and a mixture model (soft
    labels: one proportion row per observation) all replay the probe's
    labels, reported per panel and equal to the individual call's."""
    data = _clouds(seed=808)
    kw = {**DRAW, 'random_state': 5, 'backend': backend, **spec}
    singles = [hyp.plot(d, **kw) for d in data]
    grid = hyp.plot(data, panels=2, panel_fit='independent', **kw)
    for panel, single in zip(grid['panel_models'], singles):
        got = np.asarray(panel['models']['cluster_labels'])
        want = np.asarray(single['models']['cluster_labels'])
        assert got.shape == want.shape and np.allclose(got, want)
        assert panel['models']['cluster'] == single['models']['cluster']


def test_the_bundle_reports_the_figures_cluster_labels():
    """``models['cluster_labels']`` is the per-observation labelling the
    figure was grouped by (None without clustering), and it is what a
    fresh ``hyp.cluster`` of the same analyzed rows with the same seed
    gives."""
    data = _clouds(seed=809)
    plain = hyp.plot(data, **DRAW)
    assert plain['models']['cluster_labels'] is None
    bundle = hyp.plot(data, **CLUSTER, **DRAW)
    labels = bundle['models']['cluster_labels']
    assert len(labels) == 48 and set(labels) == {0, 1, 2}
    again = hyp.cluster(bundle['xform_data'], cluster='KMeans', n_clusters=3,
                        random_state=88)
    assert list(labels) == list(again)
    assert 'cluster_labels' in hyp.plot.__doc__


# ---------------- 3: a pinned call keeps the palette slots already taken

@pytest.fixture(params=('figure', 'cell'))
def composed(request):
    """A target to compose into, with one ordinary dataset already drawn
    there (one palette slot taken): a plain figure or a `hyp.subplots`
    cell."""
    def make(backend, x):
        if request.param == 'figure':
            fig = hyp.plot(x, palette=PALETTE, backend=backend, show=False)
            target = fig.axes[0] if backend == 'matplotlib' else fig
            return fig, target
        fig, axes = hyp.subplots(1, 1, backend=backend)
        hyp.plot(x, ax=axes[0], palette=PALETTE, backend=backend, show=False)
        return fig, axes[0]
    return make


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('pin', ('color', 'hue'))
def test_pinned_call_keeps_a_nonzero_offset(backend, pin, composed):
    """ordinary -> pinned (``color=`` / categorical ``hue=``) -> ordinary:
    the last dataset is GOLD (the second palette colour) on both backends,
    as the single three-dataset call draws it."""
    x = np.random.default_rng(810).normal(size=(20, 3))
    fig, target = composed(backend, x)
    pinned = ({'color': 'black'} if pin == 'color'
              else {'hue': ['a'] * 10 + ['b'] * 10})
    hyp.plot(x + 1, ax=target, palette=PALETTE, backend=backend, show=False,
             **pinned)
    hyp.plot(x + 2, ax=target, palette=PALETTE, backend=backend, show=False)
    colors = _line_colors(fig, backend)
    assert colors[0] == _rgb255('navy')
    assert colors[-1] == _rgb255('gold')
    if pin == 'color':
        assert colors[1] == _rgb255('black')


@pytest.mark.parametrize('backend', BACKENDS)
def test_pinned_call_between_two_taken_slots(backend, composed):
    """Two slots taken, a pinned call, then an ordinary one: GREEN --
    the offset is carried, not clamped or reset."""
    x = np.random.default_rng(811).normal(size=(20, 3))
    fig, target = composed(backend, x)
    hyp.plot(x + 1, ax=target, palette=PALETTE, backend=backend, show=False)
    hyp.plot(x + 2, ax=target, palette=PALETTE, backend=backend, show=False,
             color='black')
    hyp.plot(x + 3, ax=target, palette=PALETTE, backend=backend, show=False)
    colors = _line_colors(fig, backend)
    assert colors[:2] == [_rgb255('navy'), _rgb255('gold')]
    assert colors[-1] == _rgb255('green')


@pytest.mark.parametrize('backend', BACKENDS)
def test_pinned_call_consumes_no_slot_itself(backend, composed):
    """The round-7 rule still holds from a non-zero offset: the pinned
    call takes no slot, so the offset after it is exactly the one before
    (gold, not green, follows one ordinary + one pinned call)."""
    x = np.random.default_rng(812).normal(size=(20, 3))
    fig, target = composed(backend, x)
    hyp.plot(x + 1, ax=target, palette=PALETTE, backend=backend, show=False,
             hue=['a'] * 10 + ['b'] * 10)
    hyp.plot(x + 2, ax=target, palette=PALETTE, backend=backend, show=False)
    assert _line_colors(fig, backend)[-1] == _rgb255('gold')
    assert _rgb255('green') not in _line_colors(fig, backend)


# ------------------ 4: one-column panels in a mixed-width independent grid

def _cell_kinds(fig, backend):
    if backend == 'matplotlib':
        return [ax.name for ax in fig.axes if ax.get_visible()]
    return sorted({tr.type for tr in _plotly_data_traces(fig)})


def _first_trace_points(fig, backend):
    """Per cell, the coordinates of its first data line/trace."""
    if backend == 'matplotlib':
        return [np.column_stack(ax.lines[0].get_data_3d()
                                if ax.name == '3d' else ax.lines[0].get_data())
                for ax in fig.axes if ax.get_visible()]
    return [np.column_stack([tr.x, tr.y] + ([tr.z] if tr.type == 'scatter3d'
                                            else []))
            for tr in _plotly_data_traces(fig)]


def _mixed(widths, seed=813):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(24, w)).cumsum(0) for w in widths]


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('widths', ([1, 3], [1, 2], [2, 3]))
def test_mixed_widths_under_independent_fit(backend, widths):
    """Independent panels of unequal widths construct on both backends
    and share one grid: 3-D cells when any panel is 3 wide (the narrow
    panel on the floor), 2-D cells otherwise. A one-column panel is a
    series -- evenly spaced increasing x, its values on y."""
    data = _mixed(widths)
    bundle = hyp.plot(data, panels=2, panel_fit='independent', reduce=None,
                      backend=backend, show=False, return_model=True,
                      antialias=False)
    three_d = max(widths) == 3
    if backend == 'matplotlib':
        assert _cell_kinds(bundle['fig'], backend) == (
            ['3d', '3d'] if three_d else ['rectilinear', 'rectilinear'])
    else:
        assert _cell_kinds(bundle['fig'], backend) == (
            ['scatter3d'] if three_d else ['scatter'])
    pts = _first_trace_points(bundle['fig'], backend)
    assert [p.shape for p in pts] == [(24, 3 if three_d else 2)] * 2
    narrow, wide = pts
    values = data[0][:, 0]
    if widths[0] == 1:
        x = narrow[:, 0].astype(float)
        assert (np.diff(x) > 0).all() and np.allclose(np.diff(x), np.diff(x)[0])
        y = narrow[:, 1].astype(float)
        assert np.corrcoef(y, values)[0, 1] > 0.999999
    else:
        assert np.ptp(narrow[:, 0]) > 0 and np.ptp(narrow[:, 1]) > 0
    if three_d:
        assert np.ptp(narrow[:, 2].astype(float)) == 0
        assert (np.ptp(wide.astype(float), axis=0) > 0).all()


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('widths', ([1, 3], [1, 2], [2, 3]))
def test_mixed_widths_under_a_shared_fit_are_refused_as_documented(
        backend, widths):
    """One pipeline cannot be fit across datasets of unequal width; the
    shared grid says so (the single-axes error, naming the widths) rather
    than crashing inside a backend."""
    with pytest.raises(ValueError, match=r'column counts \[%d, %d\]'
                       % tuple(widths)):
        hyp.plot(_mixed(widths), panels=2, panel_fit='shared', reduce=None,
                 backend=backend, show=False)


@pytest.mark.parametrize('backend', BACKENDS)
def test_one_column_frame_with_a_date_index_in_a_3d_grid(backend):
    """A dated one-column frame beside a 3-column dataset: drawn by row
    position (a scene has no date axis), its values intact."""
    rng = np.random.default_rng(814)
    series = pd.DataFrame(rng.normal(size=(24, 1)).cumsum(0),
                          index=pd.date_range('2024-01-01', periods=24,
                                              freq='D'), columns=['v'])
    bundle = hyp.plot([series, rng.normal(size=(24, 3))], panels=2,
                      panel_fit='independent', reduce=None, backend=backend,
                      show=False, return_model=True, antialias=False)
    narrow = _first_trace_points(bundle['fig'], backend)[0].astype(float)
    assert narrow.shape == (24, 3)
    assert (np.diff(narrow[:, 0]) > 0).all()
    assert np.corrcoef(narrow[:, 1], series['v'].to_numpy())[0, 1] > 0.999999
    assert np.ptp(narrow[:, 2]) == 0


def test_panels_docstring_describes_the_cells_from_the_analyzed_width():
    doc = hyp.plot.__doc__
    assert 'ANALYZED data' in doc
    assert 'UNEQUAL analyzed' in doc
    assert '1-column series as row index vs value' in doc
