"""1.1 release review, round 9: three ``panels=``/composition defects
pinned at the public API on both backends. No mocks -- every assertion
reads the drawn artists/traces or the returned bundle, and compares them
with a separate individual call.

1 SHARED clustered panels lost the joint call's label-to-colour mapping:
  each panel replayed its own slice of the one seeded clustering but
  coloured it from the labels present in THAT slice, so two panels
  holding global clusters 1 and 0 both drew the first palette colour
  where the joint figure drew them in two. Every cell now colours (and
  names in its legend) each cluster exactly as the joint figure does.
2 a NARROW panel of a mixed-width independent grid (a 1- or 2-column
  dataset beside a 3-column one) was padded to ``(index, value, 0)``
  BEFORE the panel call, so the padded rows became the forecasting
  input: its Kalman forecast differed from the individual call's, a
  forecaster fitted on the one-column data refused the three padded
  features, and a one-column ``truth=`` was rejected. The panel now
  forecasts, resolves ``truth=`` and reports its bundle in the analyzed
  space (the individual call's numbers) and lifts only the DRAWN rows,
  forecast and truth into the 3-D cell.
3 a categorical ``hue=`` with a marker-only fmt (``'o'``) advanced the
  palette on a composed axes/figure/cell: its groups were drawn straight
  from the ambient cycle (which also let a colour letter, ``'ro'``, paint
  every group red). Hue ownership is now explicit in the palette
  accounting, and the marker path resolves its category colours as the
  line path always did.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt                                 # noqa: E402
import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402
import pytest                                                   # noqa: E402
from matplotlib.colors import to_rgb                            # noqa: E402

import hypertools as hyp                                        # noqa: E402
from hypertools.plot.plotly_backend import _rgb_triplet         # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


BACKENDS = ('matplotlib', 'plotly')
PALETTE = ['navy', 'gold', 'green', 'purple']


def _rgb255(color):
    if isinstance(color, str) and color.startswith(('rgb(', 'rgba(')):
        return tuple(int(c) for c in _rgb_triplet(color))
    return tuple(int(round(c * 255)) for c in to_rgb(color))


def _plotly_data_traces(fig):
    return [tr for tr in fig.data
            if getattr(tr, 'hoverinfo', None) != 'skip']


def _trace_color(tr):
    return tr.line.color if 'lines' in (tr.mode or '') else tr.marker.color


def _line_colors(fig, backend):
    """The data-line colours a figure draws, in drawing order."""
    if backend == 'matplotlib':
        return [_rgb255(ln.get_color()) for ax in fig.axes
                if ax.get_visible() for ln in ax.lines]
    return [_rgb255(_trace_color(tr)) for tr in _plotly_data_traces(fig)]


# --------------- 1: shared clustered panels keep the joint colour mapping

def _clouds(centres, rows=20, seed0=901):
    return [np.random.default_rng(seed0 + i).normal(scale=0.01,
                                                     size=(rows, 3)) + c
            for i, c in enumerate(centres)]


CLUSTER = dict(reduce=None, cluster='KMeans', random_state=88,
               legend=True, antialias=False, show=False, return_model=True)


def _labelled_colors(fig, backend, models):
    """{cluster label: colour} as drawn, per cell, read off the artists
    (matplotlib) or traces (plotly) together with the bundle's replayed
    labels -- what a reader of the figure sees for each global cluster."""
    out = []
    if backend == 'matplotlib':
        cells = [ax for ax in fig.axes if ax.get_visible() and ax.lines]
        for ax, m in zip(cells, models):
            legend = ax.get_legend()
            names = [t.get_text() for t in legend.get_texts()]
            handles = [_rgb255(h.get_color()) for h in legend.legend_handles]
            out.append({
                'labels': sorted(set(np.asarray(
                    m['models']['cluster_labels']).tolist())),
                'legend': dict(zip(names, handles)),
                'lines': [_rgb255(ln.get_color()) for ln in ax.lines],
            })
        return out
    by_scene = {}
    for tr in _plotly_data_traces(fig):
        by_scene.setdefault(tr.scene, []).append(tr)
    scenes = sorted(by_scene, key=lambda s: (len(s or ''), s or ''))
    for scene, m in zip(scenes, models):
        traces = by_scene[scene]
        out.append({
            'labels': sorted(set(np.asarray(
                m['models']['cluster_labels']).tolist())),
            'legend': {tr.name: _rgb255(_trace_color(tr)) for tr in traces},
            'lines': [_rgb255(_trace_color(tr)) for tr in traces],
        })
    return out


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('fmt', ('o', '-'))
@pytest.mark.parametrize('palette', ('hls', ['red', 'blue', 'green']))
def test_shared_panels_colour_each_cluster_as_the_joint_figure_does(
        backend, fmt, palette):
    """Three well-separated clouds, one seeded 3-cluster fit: every panel
    holds exactly one global cluster, and draws it -- and names it in
    its legend -- in the colour the joint figure gives that label. The
    default 'hls' palette is included because its colours depend on how
    many are asked for: a panel colouring from its OWN one-label set got
    hls(1), not the joint figure's hls(3)."""
    data = _clouds([-10, 0, 10])
    joint = hyp.plot(data, n_clusters=3, fmt=fmt, palette=palette,
                     backend=backend, **CLUSTER)
    grid = hyp.plot(data, n_clusters=3, fmt=fmt, palette=palette,
                    backend=backend, panels=3, **CLUSTER)
    joint_map = _labelled_colors(joint['fig'], backend, [joint])[0]['legend']
    assert sorted(joint_map) == ['0', '1', '2']
    assert len(set(joint_map.values())) == 3
    cells = _labelled_colors(grid['fig'], backend, grid['panel_models'])
    assert len(cells) == 3
    seen = []
    for cell in cells:
        assert len(cell['labels']) == 1
        label = cell['labels'][0]
        seen.append(label)
        # the legend names the GLOBAL label, in the joint figure's colour
        assert cell['legend'] == {str(label): joint_map[str(label)]}
        # ...and every drawn artist of the cell is that colour
        assert set(cell['lines']) == {joint_map[str(label)]}
    assert sorted(seen) == [0, 1, 2]
    # the three panels draw three DIFFERENT colours (no two clusters
    # collapsed onto the first palette entry)
    assert len({cell['lines'][0] for cell in cells}) == 3


@pytest.mark.parametrize('backend', BACKENDS)
def test_shared_panel_missing_a_cluster_keeps_the_others_colours(backend):
    """The reviewer's probe: two clouds, two clusters, ``palette=['red',
    'blue']``. The joint figure draws label 0 red and label 1 blue; the
    panel holding label 1 must be blue, not red."""
    data = _clouds([-10, 10], rows=24)
    joint = hyp.plot(data, n_clusters=2, fmt='o', palette=['red', 'blue'],
                     backend=backend, **CLUSTER)
    grid = hyp.plot(data, n_clusters=2, fmt='o', palette=['red', 'blue'],
                    backend=backend, panels=2, **CLUSTER)
    joint_map = _labelled_colors(joint['fig'], backend, [joint])[0]['legend']
    assert joint_map == {'0': _rgb255('red'), '1': _rgb255('blue')}
    cells = _labelled_colors(grid['fig'], backend, grid['panel_models'])
    for cell in cells:
        (label,) = cell['labels']
        assert cell['lines'] == [joint_map[str(label)]]
        assert cell['legend'] == {str(label): joint_map[str(label)]}
    assert {cell['lines'][0] for cell in cells} == {
        _rgb255('red'), _rgb255('blue')}


@pytest.mark.parametrize('backend', BACKENDS)
def test_independent_and_reducer_panels_still_colour_by_sorted_label(
        backend):
    """The mapping is applied in every fit mode: a panel drawing all of
    its probe's clusters colours them exactly as its individual call."""
    data = [np.vstack(_clouds([-10, 0, 10], rows=8, seed0=910 + 3 * k))
            for k in range(2)]
    for mode in ('independent', 'reducer'):
        kw = dict(n_clusters=3, fmt='o', palette='hls', backend=backend)
        kw.update(CLUSTER)
        if mode == 'reducer':
            kw['reduce'] = ['PCA', 'PCA']
        grid = hyp.plot(data, panels=2, **kw,
                        **({'panel_fit': 'independent'}
                           if mode == 'independent' else {}))
        cells = _labelled_colors(grid['fig'], backend, grid['panel_models'])
        for i, cell in enumerate(cells):
            single_kw = dict(kw)
            if mode == 'reducer':
                single_kw['reduce'] = 'PCA'
            single = hyp.plot(data[i] if mode == 'independent' else data,
                              **single_kw)
            want = _labelled_colors(single['fig'], backend, [single])[0]
            assert cell['legend'] == want['legend']
            assert cell['lines'] == want['lines']


# --------- 2: narrow panels forecast in the analyzed space, drawn lifted

def _mixed(widths, seed=913):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(24, w)).cumsum(0) for w in widths]


FORECAST = dict(reduce=None, predict='Kalman', t=3, antialias=False,
                show=False, return_model=True)


def _cell_traces(fig, backend, cell):
    """(data, forecast, truth) drawn rows of one grid cell, each an
    (n, 3) float array (None when that overlay is absent)."""
    if backend == 'matplotlib':
        ax = [a for a in fig.axes if a.get_visible()][cell]
        rows = [np.column_stack(ln.get_data_3d()).astype(float)
                for ln in ax.lines]
        data, rest = rows[0], rows[1:]
        forecast = rest[0] if rest else None
        truth = rest[1] if len(rest) > 1 else None
        return data, forecast, truth
    scene = 'scene' if cell == 0 else f'scene{cell + 1}'
    traces = [tr for tr in fig.data if tr.scene == scene
              and tr.type == 'scatter3d']
    by_role = {}
    for tr in traces:
        meta = tr.meta if isinstance(tr.meta, dict) else {}
        role = meta.get('hyp_forecast_role', 'data' if 'hyp_trace_index'
                        in meta else None)
        if role is not None:
            by_role[role] = np.column_stack([tr.x, tr.y, tr.z]).astype(float)
    return by_role['data'], by_role.get('static'), by_role.get('truth')


def _affine(values, drawn):
    """The affine map (slope, intercept) from analyzed `values` to the
    `drawn` display coordinate -- exact for the unit-box rescale -- with
    its fit residual asserted to be zero."""
    slope, intercept = np.polyfit(values, drawn, 1)
    assert np.allclose(slope * values + intercept, drawn, atol=1e-9)
    return slope, intercept


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('widths', ([1, 3], [2, 3]))
def test_narrow_panel_forecasts_are_the_individual_calls(backend, widths):
    """The reviewer's numeric probe: the narrow panel's bundled forecast
    is EXACTLY the individual call's (same shape, same numbers), and the
    bundle stays in the analyzed space."""
    data = _mixed(widths)
    single = hyp.plot(data[0], backend=backend, **FORECAST)
    grid = hyp.plot(data, panels=2, panel_fit='independent',
                    backend=backend, **FORECAST)
    panel = grid['panel_models'][0]
    want = np.asarray(single['predict']['forecasts'][0])
    got = np.asarray(panel['predict']['forecasts'][0])
    assert got.shape == want.shape == (3, widths[0])
    assert np.array_equal(got, want)
    assert np.asarray(panel['xform_data'][0]).shape == (24, widths[0])
    assert np.asarray(panel['trace_data'][0]).shape == (24, widths[0])
    # ...and both equal hyp.predict on the analyzed rows themselves
    direct = np.asarray(hyp.predict(data[0], model='Kalman', t=3))
    assert np.allclose(got, direct)


@pytest.mark.parametrize('backend', BACKENDS)
def test_one_column_panel_draws_its_forecast_along_the_series(backend):
    """The DRAWN forecast of a one-column panel continues the series:
    x steps on by one row per forecast step, its y values are the
    forecast values through the same affine the data went through, and
    it stays on the cell's floor."""
    data = _mixed([1, 3])
    grid = hyp.plot(data, panels=2, panel_fit='independent',
                    backend=backend, **FORECAST)
    drawn, forecast, _ = _cell_traces(grid['fig'], backend, 0)
    assert drawn.shape == (24, 3) and forecast.shape == (4, 3)
    step = drawn[1, 0] - drawn[0, 0]
    assert step > 0
    assert np.allclose(np.diff(drawn[:, 0]), step)
    # seam row = the last observed row; then one step per forecast row
    assert np.allclose(forecast[0], drawn[-1])
    assert np.allclose(forecast[:, 0], drawn[-1, 0] + step * np.arange(4))
    assert np.allclose(forecast[:, 2], drawn[0, 2])
    assert np.ptp(drawn[:, 2]) == 0
    values = data[0][:, 0]
    slope, intercept = _affine(values, drawn[:, 1])
    fc = np.asarray(grid['panel_models'][0]['predict']['forecasts'][0])[:, 0]
    assert np.allclose(forecast[1:, 1], slope * fc + intercept, atol=1e-9)


@pytest.mark.parametrize('backend', BACKENDS)
def test_two_column_panel_draws_its_forecast_on_the_floor(backend):
    data = _mixed([2, 3])
    grid = hyp.plot(data, panels=2, panel_fit='independent',
                    backend=backend, **FORECAST)
    drawn, forecast, _ = _cell_traces(grid['fig'], backend, 0)
    assert forecast.shape == (4, 3)
    assert np.allclose(forecast[0], drawn[-1])
    assert np.ptp(np.concatenate([drawn[:, 2], forecast[:, 2]])) == 0
    fc = np.asarray(grid['panel_models'][0]['predict']['forecasts'][0])
    for col in range(2):
        slope, intercept = _affine(data[0][:, col], drawn[:, col])
        assert np.allclose(forecast[1:, col], slope * fc[:, col] + intercept,
                           atol=1e-9)


@pytest.mark.parametrize('backend', BACKENDS)
def test_a_forecaster_fitted_on_the_narrow_data_is_reused(backend):
    """The reviewer's fitted-model case: a Kalman fitted on BOTH
    datasets (`hyp.predict(x, return_model=True)`) is bound to each
    panel's own dataset and forecasts its analyzed rows -- the same
    numbers `fitted.for_dataset(0)` gives the individual call -- instead
    of refusing three padded features."""
    data = _mixed([1, 3])
    fitted = hyp.predict(data, model='Kalman', t=3, return_model=True)[1]
    kw = dict(FORECAST)
    kw['predict'] = fitted
    grid = hyp.plot(data, panels=2, panel_fit='independent',
                    backend=backend, **kw)
    kw['predict'] = fitted.for_dataset(0)
    single = hyp.plot(data[0], backend=backend, **kw)
    got = np.asarray(grid['panel_models'][0]['predict']['forecasts'][0])
    want = np.asarray(single['predict']['forecasts'][0])
    assert got.shape == want.shape == (3, 1)
    assert np.array_equal(got, want)
    # a forecaster fitted on the one-column dataset ALONE is reused for
    # EVERY panel (the reviewer's first probe): it fits the one-column
    # panel and is refused by the three-column one -- the same refusal
    # the individual three-column call gives, naming ITS width (the
    # one-column panel no longer offers three padded features)
    kw['predict'] = hyp.predict(data[0], model='Kalman', t=3,
                                return_model=True)[1]
    with pytest.raises(ValueError, match='expects 1 feature') as grid_err:
        hyp.plot(data, panels=2, panel_fit='independent', backend=backend,
                 **kw)
    with pytest.raises(ValueError, match='expects 1 feature') as single_err:
        hyp.plot(data[1], backend=backend, **kw)
    assert str(grid_err.value) == str(single_err.value)
    assert 'new dataset has 3' in str(grid_err.value)


@pytest.mark.parametrize('backend', BACKENDS)
def test_one_column_truth_is_accepted_and_drawn_along_the_series(backend):
    """`truth=` is read in the analyzed space (one column for the
    one-column panel, as the individual call reads it) and drawn where
    the forecast is drawn: continuing the series' x, its values through
    the data's affine, on the floor."""
    data = _mixed([1, 3])
    held = [data[0][-3:, :] + 0.5, data[1][-3:, :] + 0.5]
    grid = hyp.plot(data, panels=2, panel_fit='independent', truth=held,
                    backend=backend, **FORECAST)
    drawn, forecast, truth = _cell_traces(grid['fig'], backend, 0)
    assert truth is not None and truth.shape == (4, 3)
    step = drawn[1, 0] - drawn[0, 0]
    assert np.allclose(truth[0], drawn[-1])
    assert np.allclose(truth[:, 0], drawn[-1, 0] + step * np.arange(4))
    assert np.allclose(truth[:, 2], drawn[0, 2])
    slope, intercept = _affine(data[0][:, 0], drawn[:, 1])
    assert np.allclose(truth[1:, 1], slope * held[0][:, 0] + intercept,
                       atol=1e-9)
    # the individual call accepts the very same truth
    hyp.plot(data[0], truth=held[0], backend=backend, **FORECAST)


@pytest.mark.parametrize('backend', BACKENDS)
def test_dated_one_column_panel_forecast_continues_by_position(backend):
    """A dated one-column frame is drawn by row position in a 3-D cell
    (no date axis in a scene); its forecast continues that position axis
    one row per step, and its bundle is still the individual call's."""
    rng = np.random.default_rng(914)
    series = pd.DataFrame(rng.normal(size=(24, 1)).cumsum(0),
                          index=pd.date_range('2024-01-01', periods=24,
                                              freq='D'), columns=['v'])
    data = [series, rng.normal(size=(24, 3)).cumsum(0)]
    grid = hyp.plot(data, panels=2, panel_fit='independent',
                    backend=backend, **FORECAST)
    single = hyp.plot(series, backend=backend, **FORECAST)
    assert np.array_equal(
        np.asarray(grid['panel_models'][0]['predict']['forecasts'][0]),
        np.asarray(single['predict']['forecasts'][0]))
    drawn, forecast, _ = _cell_traces(grid['fig'], backend, 0)
    step = drawn[1, 0] - drawn[0, 0]
    assert np.allclose(forecast[:, 0], drawn[-1, 0] + step * np.arange(4))


@pytest.mark.parametrize('backend', BACKENDS)
def test_two_d_mixed_grid_bundle_is_unchanged(backend):
    """A [1, 2] grid draws 2-D cells (no lift): its narrow panel's
    bundle is the individual call's, exactly as before."""
    data = _mixed([1, 2])
    single = hyp.plot(data[0], backend=backend, **FORECAST)
    grid = hyp.plot(data, panels=2, panel_fit='independent',
                    backend=backend, **FORECAST)
    assert np.array_equal(
        np.asarray(grid['panel_models'][0]['predict']['forecasts'][0]),
        np.asarray(single['predict']['forecasts'][0]))
    assert '_panel_lift' not in hyp.plot.__doc__


# ------------- 3: marker-only categorical hue consumes no palette slot

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


HUE = ['a'] * 12 + ['b'] * 12


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('fmt', ('o', '.', 'ro', 'r-o', '-'))
def test_categorical_hue_consumes_no_slot_whatever_its_fmt(
        backend, fmt, composed):
    """ordinary -> categorical hue (any fmt) -> ordinary: the last
    dataset is GOLD (the second palette colour), as it is for the line
    fmt and as the single call gives it; and the hue's own groups take
    the palette's first two colours regardless of what was drawn before
    -- a colour letter in the fmt does not paint them."""
    x = np.random.default_rng(916).normal(size=(24, 3))
    fig, target = composed(backend, x)
    hyp.plot(x + 1, ax=target, palette=PALETTE, backend=backend, show=False,
             hue=HUE, fmt=fmt)
    hyp.plot(x + 2, ax=target, palette=PALETTE, backend=backend, show=False)
    colors = _line_colors(fig, backend)
    assert colors[0] == _rgb255('navy')
    assert colors[-1] == _rgb255('gold')
    assert _rgb255('purple') not in colors
    hue_colors = set(colors[1:-1])
    assert hue_colors == {_rgb255('navy'), _rgb255('gold')}


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('fmt', ('ro', 'r.', 'r-'))
def test_hue_colours_beat_a_fmt_colour_letter_on_every_path(backend, fmt):
    """On a fresh figure too: ``hue=`` with ``'ro'`` colours the groups
    navy/gold, exactly as ``'r-'`` always did (the marker path used to
    let the letter paint every group red)."""
    x = np.random.default_rng(917).normal(size=(24, 3))
    fig = hyp.plot(x, hue=HUE, fmt=fmt, palette=PALETTE, backend=backend,
                   show=False)
    assert _line_colors(fig, backend) == [_rgb255('navy'), _rgb255('gold')]


@pytest.mark.parametrize('backend', BACKENDS)
def test_marker_hue_colours_match_the_single_three_dataset_call(backend):
    """The composed sequence ordinary -> hue (fmt='o') -> ordinary draws
    the same colours as one call drawing the same three things."""
    x = np.random.default_rng(918).normal(size=(24, 3))
    fig = hyp.plot(x, palette=PALETTE, backend=backend, show=False)
    target = fig.axes[0] if backend == 'matplotlib' else fig
    hyp.plot(x + 1, ax=target, palette=PALETTE, backend=backend, show=False,
             hue=HUE, fmt='o')
    hyp.plot(x + 2, ax=target, palette=PALETTE, backend=backend, show=False)
    composed_colors = _line_colors(fig, backend)
    two = hyp.plot([x, x + 2], palette=PALETTE, backend=backend, show=False)
    assert [composed_colors[0], composed_colors[-1]] == _line_colors(
        two, backend)
    grouped = hyp.plot(x + 1, hue=HUE, fmt='o', palette=PALETTE,
                       backend=backend, show=False)
    assert composed_colors[1:-1] == _line_colors(grouped, backend)


def test_palette_slots_docstring_names_hue_ownership():
    from hypertools.plot.plot import _palette_slots_consumed
    doc = _palette_slots_consumed.__doc__
    assert 'category_colored' in doc
    assert "fmt='o'" in doc
