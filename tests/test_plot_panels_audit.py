"""`panels=` partitions every per-dataset and per-forecast argument, and
exposes the shared fitted pipeline (release audit 2026-09-07, findings 3
and 4).

Finding 3: a shared-fit panel grid drew every panel through `transform=`,
so each panel bundle's ``pipeline`` was ``None`` and the ONE fitted pipeline
the panels share could not be replayed on held-out data.

Finding 4: `_plot_panels` handed every panel the WHOLE per-dataset /
per-forecast list for `palette=` (per-dataset form), `forecast_fmt=`,
`forecast_palette=`, a model-major `forecast_hue=`, and a forecaster fitted
on every dataset -- each of which works on the single-axes path and either
raised inside the panel or (`forecast_palette=`) drew every panel in the
first colour. Two more of the same shape turned up while fixing it: a
`legend=` list naming the datasets, and ``alpha=[a, b, c]`` for three
datasets (read as ONE RGB colour by the colour-tuple guard).

Every test reads the ACTUAL artists/traces -- colours, styles, forecast
values -- on both backends, in both `panel_fit=` modes, and compares them
with the single-axes call's, which is what a panel grid promises to draw.
"""

import inspect
import re
import warnings

import matplotlib
matplotlib.use('Agg')

import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402
import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib.colors import to_rgb                            # noqa: E402

import hypertools as hyp                                        # noqa: E402
from hypertools.core.pipeline import Pipeline                   # noqa: E402
from hypertools.plot import plot as plot_module                 # noqa: E402
from hypertools.plot.colors import palette_lead_color           # noqa: E402

BACKENDS = ('matplotlib', 'plotly')
FITS = ('shared', 'independent')


def walks(n=2, rows=20, cols=3, seed=201):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, cols)).cumsum(0) for _ in range(n)]


def quiet_plot(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return hyp.plot(*args, show=False, return_model=True, **kwargs)


# ---------------------------------------------------------------- readers

def _rgba(color):
    """(rgb tuple in [0, 1], alpha) from a matplotlib colour or a plotly
    ``rgb()``/``rgba()``/hex string."""
    if isinstance(color, str) and color.startswith(('rgb(', 'rgba(')):
        parts = [float(v) for v in re.findall(r'[\d.]+', color)]
        rgb = tuple(v / 255.0 for v in parts[:3])
        return rgb, (parts[3] if len(parts) > 3 else 1.0)
    return tuple(to_rgb(color)), None


def data_artists(bundle, i, backend):
    """The observed-data artists of panel `i` (forecast/truth overlays
    excluded), as ``[(rgb, alpha, style, label)]``."""
    if backend == 'matplotlib':
        out = []
        for line in bundle['axes'][i].lines:
            if getattr(line, '_hyp_forecast_role', None):
                continue
            rgb, _ = _rgba(line.get_color())
            out.append((rgb, line.get_alpha(), line.get_linestyle(),
                        line.get_label()))
        return out
    scene = bundle['axes'][i].plotly_name
    out = []
    for trace in bundle['fig'].data:
        if trace.scene != scene or not isinstance(trace.meta, dict) \
                or 'hyp_trace_index' not in trace.meta:
            continue
        rgb, alpha = _rgba(trace.line.color)
        out.append((rgb, alpha, trace.line.dash, trace.name))
    return out


def forecast_artists(fig_or_bundle, i, backend, panel=True):
    """The forecast overlays of panel `i` (or of the single axes when
    ``panel=False``), as ``[(rgb, style)]`` in drawing order -- model-major
    for a `predict=` collection."""
    if backend == 'matplotlib':
        axes = (fig_or_bundle['axes'][i] if panel
                else fig_or_bundle['fig'].axes[0])
        return [(_rgba(line.get_color())[0], line.get_linestyle())
                for line in axes.lines
                if getattr(line, '_hyp_forecast_role', None) == 'static'
                and (panel or getattr(line, '_hyp_forecast_dataset',
                                      None) == i)]
    scene = fig_or_bundle['axes'][i].plotly_name if panel else None
    out = []
    for trace in fig_or_bundle['fig'].data:
        if panel and trace.scene != scene:
            continue
        if not isinstance(trace.meta, dict) \
                or trace.meta.get('hyp_forecast_role') != 'static':
            continue
        if not panel and trace.meta.get('hyp_dataset') != i:
            continue
        out.append((_rgba(trace.line.color)[0], trace.line.dash))
    return out


def same_color(a, b):
    return np.allclose(a, b, atol=1.5 / 255)


def close_all():
    plt.close('all')


# ---------------------------------------------------- finding 3: pipeline

@pytest.mark.parametrize('backend', BACKENDS)
def test_shared_panels_expose_the_shared_pipeline(backend):
    """Every shared-fit panel bundle carries the ONE fitted pipeline, and
    replaying it on held-out data projects into the panels' common space
    without refitting -- byte-for-byte what the single-axes bundle's
    pipeline does with the same held-out data."""
    data = walks(n=2, rows=30, cols=6, seed=1)
    held_out = np.random.default_rng(99).normal(size=(9, 6))
    single = quiet_plot(data, reduce='PCA', backend=backend)
    grid = quiet_plot(data, panels=True, reduce='PCA', backend=backend)
    try:
        pipelines = [m['pipeline'] for m in grid['panel_models']]
        assert all(isinstance(p, Pipeline) for p in pipelines)
        assert pipelines[0] is pipelines[1]            # ONE shared fit
        assert grid['pipeline'] is pipelines[0]
        assert pipelines[0].is_fitted
        want = np.asarray(single['pipeline'].transform(held_out))
        got = np.asarray(pipelines[0].transform(held_out))
        assert got.shape == (9, 3)
        np.testing.assert_allclose(got, want, atol=1e-10)
        # the bundled pipeline is the fit the panels were DRAWN from: it
        # reproduces each panel's own rows ...
        for i, panel in enumerate(grid['panel_models']):
            np.testing.assert_allclose(
                np.asarray(pipelines[0].transform(data[i])),
                np.asarray(panel['xform_data'][0]), atol=1e-10)
        # ... and it was not refit on the held-out rows: a fresh fit on
        # them lands somewhere else
        fresh = np.asarray(hyp.reduce(held_out, reduce='PCA', ndims=3))
        assert not np.allclose(fresh, got, atol=1e-6)
    finally:
        close_all()


@pytest.mark.parametrize('backend', BACKENDS)
def test_independent_panels_expose_their_own_pipelines(backend):
    data = walks(n=2, rows=30, cols=6, seed=2)
    held_out = np.random.default_rng(98).normal(size=(9, 6))
    grid = quiet_plot(data, panels=True, panel_fit='independent',
                      reduce='PCA', backend=backend)
    try:
        assert grid['pipeline'] is None               # no shared fit
        pipelines = [m['pipeline'] for m in grid['panel_models']]
        assert all(isinstance(p, Pipeline) for p in pipelines)
        assert pipelines[0] is not pipelines[1]
        outs = []
        for i, pipeline in enumerate(pipelines):
            own = quiet_plot([data[i]], reduce='PCA', backend=backend)
            want = np.asarray(own['pipeline'].transform(held_out))
            got = np.asarray(pipeline.transform(held_out))
            np.testing.assert_allclose(got, want, atol=1e-10)
            outs.append(got)
        assert not np.allclose(outs[0], outs[1], atol=1e-6)
    finally:
        close_all()


# ------------------------------------------- finding 4: per-dataset lists

@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('fit', FITS)
def test_per_dataset_palette_list_is_partitioned(backend, fit):
    data = walks()
    # palette NAMES: each panel draws in its palette's lead colour ...
    grid = quiet_plot(data, panels=True, panel_fit=fit, backend=backend,
                      palette=['viridis', 'magma'])
    try:
        for i, name in enumerate(('viridis', 'magma')):
            artists = data_artists(grid, i, backend)
            assert len(artists) == 1
            assert same_color(artists[0][0], palette_lead_color(name))
    finally:
        close_all()
    # ... and NESTED explicit colour lists likewise, matching the
    # single-axes call dataset by dataset
    nested = [['red', 'green'], ['blue', 'black']]
    single = quiet_plot(data, backend=backend, palette=nested)
    grid = quiet_plot(data, panels=True, panel_fit=fit, backend=backend,
                      palette=nested)
    try:
        single_colors = ([_rgba(ln.get_color())[0]
                          for ln in single['fig'].axes[0].lines
                          if not getattr(ln, '_hyp_forecast_role', None)]
                         if backend == 'matplotlib' else
                         [_rgba(t.line.color)[0] for t in single['fig'].data
                          if isinstance(t.meta, dict)
                          and 'hyp_trace_index' in t.meta])
        assert same_color(single_colors[0], to_rgb('red'))
        assert same_color(single_colors[1], to_rgb('blue'))
        for i in range(2):
            artists = data_artists(grid, i, backend)
            assert len(artists) == 1
            assert same_color(artists[0][0], single_colors[i])
    finally:
        close_all()


@pytest.mark.parametrize('backend', BACKENDS)
def test_alpha_list_of_three_is_partitioned(backend):
    """``alpha=[0.3, 0.6, 0.9]`` for three datasets is three alphas, not
    one RGB colour -- the colour-tuple guard applies to `color=` only."""
    data = walks(n=3)
    grid = quiet_plot(data, panels=True, backend=backend,
                      alpha=[0.3, 0.6, 0.9])
    try:
        for i, want in enumerate((0.3, 0.6, 0.9)):
            artists = data_artists(grid, i, backend)
            assert len(artists) == 1
            assert artists[0][1] == pytest.approx(want)
    finally:
        close_all()


@pytest.mark.parametrize('backend', BACKENDS)
def test_legend_list_naming_the_datasets_is_partitioned(backend):
    data = walks(n=3)
    grid = quiet_plot(data, panels=True, backend=backend,
                      legend=['alpha', 'beta', 'gamma'])
    try:
        for i, name in enumerate(('alpha', 'beta', 'gamma')):
            artists = data_artists(grid, i, backend)
            assert [a[3] for a in artists] == [name]
            if backend == 'matplotlib':
                legend = grid['axes'][i].get_legend()
                assert legend is not None
                assert [t.get_text() for t in legend.get_texts()] == [name]
    finally:
        close_all()


# ------------------------------------------ finding 4: per-forecast lists

@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('fit', FITS)
def test_forecast_fmt_list_is_partitioned(backend, fit):
    data = walks()
    grid = quiet_plot(data, panels=True, panel_fit=fit, backend=backend,
                      predict='Kalman', t=3, forecast_fmt=['--', ':'])
    single = quiet_plot(data, backend=backend, predict='Kalman', t=3,
                        forecast_fmt=['--', ':'])
    try:
        want = ('--', ':') if backend == 'matplotlib' else ('dash', 'dot')
        for i in range(2):
            styles = [s for _, s in forecast_artists(grid, i, backend)]
            assert styles == [want[i]]
            single_styles = [s for _, s in
                             forecast_artists(single, i, backend, panel=False)]
            assert single_styles == [want[i]]
    finally:
        close_all()


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('fit', FITS)
@pytest.mark.parametrize('palette', [['red', 'blue'], 'husl'])
def test_forecast_palette_is_partitioned(backend, fit, palette):
    """One colour per forecast, dataset by dataset -- an explicit list and
    a palette NAME alike -- matching the single-axes figure (before the
    fix every panel drew its forecast in the palette's first colour)."""
    data = walks()
    grid = quiet_plot(data, panels=True, panel_fit=fit, backend=backend,
                      predict='Kalman', t=3, forecast_palette=palette)
    single = quiet_plot(data, backend=backend, predict='Kalman', t=3,
                        forecast_palette=palette)
    try:
        colors = []
        for i in range(2):
            got = forecast_artists(grid, i, backend)
            want = forecast_artists(single, i, backend, panel=False)
            assert len(got) == len(want) == 1
            assert same_color(got[0][0], want[0][0])
            colors.append(got[0][0])
        assert not same_color(colors[0], colors[1])
        if palette != 'husl':
            assert same_color(colors[0], to_rgb('red'))
            assert same_color(colors[1], to_rgb('blue'))
    finally:
        close_all()


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('fit', FITS)
def test_model_major_forecast_hue_is_partitioned(backend, fit):
    """Two models x two datasets, ``forecast_hue=['a', 'b', 'c', 'd']`` in
    the documented model-major order: each panel gets ITS two forecasts'
    labels, with the colour each label has on the single-axes figure."""
    data = walks()
    kwargs = dict(predict=['Kalman', 'AutoRegressor'], t=3,
                  forecast_hue=['a', 'b', 'c', 'd'],
                  forecast_palette=['red', 'green', 'blue', 'black'])
    single = quiet_plot(data, backend=backend, **kwargs)
    grid = quiet_plot(data, panels=True, panel_fit=fit, backend=backend,
                      **kwargs)
    try:
        # single-axes overlays are model-major: [K0, K1, A0, A1]
        flat = forecast_artists(single, 0, backend, panel=False) \
            + forecast_artists(single, 1, backend, panel=False)
        assert len(flat) == 4
        by_dataset = {0: [c for c, _ in forecast_artists(single, 0, backend,
                                                          panel=False)],
                      1: [c for c, _ in forecast_artists(single, 1, backend,
                                                          panel=False)]}
        # dataset 0's forecasts are 'a' (Kalman) and 'c' (AutoRegressor)
        assert same_color(by_dataset[0][0], to_rgb('red'))
        assert same_color(by_dataset[0][1], to_rgb('blue'))
        assert same_color(by_dataset[1][0], to_rgb('green'))
        assert same_color(by_dataset[1][1], to_rgb('black'))
        solid = '-' if backend == 'matplotlib' else 'solid'
        dashed = '--' if backend == 'matplotlib' else 'dash'
        for i in range(2):
            got = forecast_artists(grid, i, backend)
            assert len(got) == 2
            # the model is told by its linestyle (solid, dashed, ...)
            assert [s for _, s in got] == [solid, dashed]
            for (color, _), want in zip(got, by_dataset[i]):
                assert same_color(color, want)
    finally:
        close_all()


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('fit', FITS)
def test_fitted_multi_dataset_forecaster_is_bound_per_panel(backend, fit):
    """A forecaster fitted on two OTHER datasets is applied dataset by
    dataset (panel i reuses fitted model i), giving exactly the forecasts
    the single-axes call draws -- and not those of a refit."""
    data = walks(seed=201)
    other = walks(seed=7)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fitted = hyp.predict(other, model='Kalman', t=3,
                             return_model=True)[1]
    assert len(fitted.models_) == 2
    single = quiet_plot(data, backend=backend, predict=fitted, t=3)
    refit = quiet_plot(data, backend=backend, predict='Kalman', t=3)
    grid = quiet_plot(data, panels=True, panel_fit=fit, backend=backend,
                      predict=fitted, t=3)
    try:
        for i, panel in enumerate(grid['panel_models']):
            got = np.asarray(panel['predict']['forecasts'][0])
            want = np.asarray(single['predict']['forecasts'][i])
            assert got.shape == (3, 3)
            np.testing.assert_allclose(got, want, atol=1e-8)
            assert not np.allclose(
                got, np.asarray(refit['predict']['forecasts'][i]),
                atol=1e-6)
            assert len(forecast_artists(grid, i, backend)) == 1
    finally:
        close_all()
    # a forecaster fitted on a different NUMBER of datasets cannot be
    # paired with the panels ...
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        three = hyp.predict(walks(n=3, seed=8), model='Kalman', t=3,
                            return_model=True)[1]
    with pytest.raises(ValueError, match=r'fitted on 3 dataset\(s\).*'
                                         r'panels= draws 2'):
        quiet_plot(data, panels=True, panel_fit=fit, backend=backend,
                   predict=three, t=3)
    close_all()
    # ... while one fitted on a SINGLE dataset is reused by every panel,
    # as it is by every dataset of a single-axes call
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        one = hyp.predict(other[0], model='Kalman', t=3,
                          return_model=True)[1]
    single = quiet_plot(data, backend=backend, predict=one, t=3)
    grid = quiet_plot(data, panels=True, panel_fit=fit, backend=backend,
                      predict=one, t=3)
    try:
        for i, panel in enumerate(grid['panel_models']):
            np.testing.assert_allclose(
                np.asarray(panel['predict']['forecasts'][0]),
                np.asarray(single['predict']['forecasts'][i]), atol=1e-8)
    finally:
        close_all()


# ------------------------------------------------------ roster vs docstring

_PER_DATASET_PHRASES = re.compile(
    r'per[- ]dataset|per (?:input )?dataset|per DATASET|per FORECAST|'
    r'per forecast|model-major|per MODEL', re.IGNORECASE)

#: documented "per dataset" arguments that `panels=` handles WITHOUT
#: slicing, each with the reason -- a new per-dataset argument must be
#: added to a roster in plot.py or, with a reason, here.
_NOT_SLICED = {
    'x': 'the datasets themselves',
    'animate': 'rejected under panels= (an animation owns its figure)',
    'resample': 'its per-dataset arrays are its OUTPUT, not an argument',
    'ax': 'rejected under panels=; mentions per-dataset layouts only',
    'title_color': 'per SEGMENT of a serial/morph title, not per dataset',
    'forecast_cluster': 'groups each panel\'s own forecast endpoints',
    'forecast_n_clusters': 'a count for forecast_cluster=, not a list',
    'forecast_trail': 'animation only; panels= is static',
    'return_model': 'describes the bundle, takes no per-dataset value',
    'colorbar': 'drawn per panel',
    'label_anchor': 'positions a per-dataset labels= entry; one value',
    'legend_colors': 'per legend ENTRY, not per dataset',
    'hue_mode': 'describes the hue matrix, one value',
    'color_reduce': 'reduces a hue matrix, one spec',
    't': 'one horizon for every forecast',
    'dataset_fade': 'animation only; panels= is static',
    'companion': 'animation only; panels= is static',
    'frame_kwargs': 'one frame style for every panel',
    'save_path': 'one file for the whole grid',
    'size': 'one figure size for the whole grid',
    'loop': 'animation only; panels= is static',
    'on_frame': 'animation only; panels= is static',
    'panels': 'the grid itself (its description names the sliced arguments)',
}


def _documented_per_dataset_parameters():
    doc = inspect.getdoc(hyp.plot)
    params = doc.split('Parameters\n----------', 1)[1]
    params = params.split('\nReturns\n-------', 1)[0]
    heading = re.compile(r'^(\w[\w()]*)(?: \([^)]*\))? : ', re.MULTILINE)
    matches = list(heading.finditer(params))
    flagged = set()
    for k, match in enumerate(matches):
        body = params[match.end():
                      matches[k + 1].start() if k + 1 < len(matches)
                      else len(params)]
        if not _PER_DATASET_PHRASES.search(body):
            continue
        name = match.group(1)
        # ``linestyle(s)`` / ``marker(s)`` document both spellings
        if name.endswith('(s)'):
            flagged.update({name[:-3], name[:-3] + 's'})
        else:
            flagged.add(name)
    return flagged


def test_panel_rosters_cover_every_documented_per_dataset_argument():
    """The docstring is the contract: every parameter whose description
    says "per dataset" / "per forecast" / "model-major" is either sliced
    by `_plot_panels` (one of the rosters in plot.py) or listed above with
    the reason it is not."""
    sliced = (set(plot_module._PANEL_PER_DATASET_KWARGS)
              | set(plot_module._PANEL_PER_FORECAST_KWARGS)
              | set(plot_module._PANEL_PER_OBSERVATION_KWARGS)
              | set(plot_module._PANEL_SPECIAL_KWARGS))
    parameters = set(inspect.signature(hyp.plot).parameters) - {'kwargs'}
    unknown = sliced - parameters
    assert not unknown, f'roster names that are not plot() parameters: {unknown}'
    documented = _documented_per_dataset_parameters()
    assert documented, 'the per-dataset phrase scan found nothing'
    # the audit's own arguments are all documented per dataset/forecast
    assert {'palette', 'forecast_fmt', 'forecast_hue', 'forecast_palette',
            'truth', 'alpha', 'fmt', 'hue', 'names', 'predict'} <= documented
    missing = documented - sliced - set(_NOT_SLICED)
    assert not missing, (
        f'documented per-dataset/per-forecast argument(s) {sorted(missing)} '
        'are neither sliced by _plot_panels (add to a _PANEL_*_KWARGS '
        'roster in hypertools/plot/plot.py) nor listed with a reason in '
        '_NOT_SLICED here')
    stale = set(_NOT_SLICED) & sliced
    assert not stale, f'{stale} listed as not sliced but on a roster'
