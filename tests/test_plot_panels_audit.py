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


# --------------------- Codex round 6: forecast labels that share a colour

@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('fit', FITS)
def test_forecast_labels_sharing_a_colour_keep_their_palette_slots(backend,
                                                                   fit):
    """Codex round 6 (the reviewer's probe): ``forecast_hue=['a', 'a',
    'b', 'b']`` (model-major, two models x two datasets) with
    ``forecast_palette=['red', 'red']`` -- two DISTINCT labels drawn in one
    colour on purpose. The single-axes figure draws every forecast red;
    the panel path resolved the grid's label -> colour map and then
    deduplicated the COLOURS, handing each panel a one-entry palette for
    its two labels (``ValueError: palette= supplies 1 color(s) but 2 are
    required``) on both backends and in both `panel_fit=` modes."""
    data = walks()
    kwargs = dict(predict=['Kalman', 'ARIMA'], t=3,
                  forecast_hue=['a', 'a', 'b', 'b'],
                  forecast_palette=['red', 'red'])
    single = quiet_plot(data, backend=backend, **kwargs)
    grid = quiet_plot(data, panels=True, panel_fit=fit, backend=backend,
                      **kwargs)
    try:
        for i in range(2):
            want = [c for c, _ in forecast_artists(single, i, backend,
                                                   panel=False)]
            got = [c for c, _ in forecast_artists(grid, i, backend)]
            assert len(want) == len(got) == 2
            for color in want + got:
                assert same_color(color, to_rgb('red'))
    finally:
        close_all()


#: the sibling spellings of "several forecasts share a colour": a
#: per-DATASET `forecast_hue=` (broadcast over a collection's models, and
#: with a single model), `forecast_cluster=` grouping instead of labels,
#: and a palette NAME that cycles -- 'Set2' has eight colours, so the
#: ninth distinct label ('I', first seen at forecast 9 = model 1 of
#: dataset 4) reuses the first label's colour, and dataset 4's panel holds
#: both ('A' at forecast 4, 'I' at forecast 9).
_SHARED_COLOUR_FORMS = {
    'per-dataset hue, two models': (2, dict(
        predict=['Kalman', 'AutoRegressor'], t=3, forecast_hue=['a', 'b'],
        forecast_palette=['red', 'red'])),
    'per-dataset hue, one model': (2, dict(
        predict='Kalman', t=3, forecast_hue=['a', 'b'],
        forecast_palette=['red', 'red'])),
    'forecast_cluster': (2, dict(
        predict=['Kalman', 'AutoRegressor'], t=3, forecast_cluster='KMeans',
        forecast_n_clusters=2, forecast_palette=['red', 'red'])),
    'palette name that cycles': (5, dict(
        predict=['Kalman', 'AutoRegressor'], t=3,
        forecast_hue=['A', 'B', 'C', 'D', 'A', 'E', 'F', 'G', 'H', 'I'],
        forecast_palette='Set2')),
}


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('fit', FITS)
@pytest.mark.parametrize('form', sorted(_SHARED_COLOUR_FORMS))
def test_repeated_forecast_colours_match_the_single_axes_figure(backend, fit,
                                                                form):
    """Every panel's forecasts carry exactly the colours the single-axes
    figure gives that dataset's forecasts, however the repetition is
    spelled (`_SHARED_COLOUR_FORMS`) -- and the repetition is real: an
    explicit ``['red', 'red']`` draws every forecast red, and the cycling
    palette name gives dataset 4's two forecasts one colour."""
    n_datasets, kwargs = _SHARED_COLOUR_FORMS[form]
    data = walks(n_datasets)
    single = quiet_plot(data, backend=backend, **kwargs)
    grid = quiet_plot(data, panels=True, panel_fit=fit, backend=backend,
                      **kwargs)
    try:
        n_models = 1 if isinstance(kwargs['predict'], str) else 2
        for i in range(n_datasets):
            want = [c for c, _ in forecast_artists(single, i, backend,
                                                   panel=False)]
            got = [c for c, _ in forecast_artists(grid, i, backend)]
            assert len(want) == len(got) == n_models
            for color, expected in zip(got, want):
                assert same_color(color, expected)
            if kwargs['forecast_palette'] == ['red', 'red']:
                for color in got:
                    assert same_color(color, to_rgb('red'))
        if form == 'palette name that cycles':
            last = [c for c, _ in forecast_artists(grid, 4, backend)]
            assert same_color(last[0], last[1])
            first = [c for c, _ in forecast_artists(grid, 0, backend)]
            assert same_color(first[0], last[0])      # 'A' is red-ish
            assert not same_color(first[0], first[1])  # 'A' vs 'E'
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
    'palette_sort': 'a grid-wide option resolved before the panels branch; '
                    'it orders the colors of every palette, per-dataset '
                    'list entries included, and each panel receives the '
                    'already-prepared palette',
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


# ------------------- roster: each per-dataset value reaches only its panel

def _style(bundle, i, backend, panel=True):
    """Everything a `_PANEL_PER_DATASET_KWARGS` argument can change, read
    off the observed-data artist of panel `i` (or, ``panel=False``, of
    dataset `i` on the single axes): colour, alpha, linestyle, marker,
    marker size, line width, legend name, and the number of surfaces
    drawn beside it."""
    if backend == 'matplotlib':
        ax = bundle['axes'][i] if panel else bundle['fig'].axes[0]
        lines = [ln for ln in ax.lines
                 if not getattr(ln, '_hyp_forecast_role', None)]
        line = lines[0] if panel else lines[i]
        if panel:
            assert len(lines) == 1
        return dict(
            color=_rgba(line.get_color())[0], alpha=line.get_alpha(),
            linestyle=line.get_linestyle(), marker=line.get_marker(),
            markersize=line.get_markersize(),
            linewidth=line.get_linewidth(), name=line.get_label(),
            surfaces=sum(type(c).__name__ == 'Poly3DCollection'
                         for c in ax.collections))
    scene = bundle['axes'][i].plotly_name if panel else None
    traces = [tr for tr in bundle['fig'].data
              if not panel or tr.scene == scene]
    lines = [tr for tr in traces if isinstance(tr.meta, dict)
             and 'hyp_trace_index' in tr.meta]
    line = lines[0] if panel else lines[i]
    if panel:
        assert len(lines) == 1
    rgb, alpha = _rgba(line.line.color)
    return dict(
        color=rgb, alpha=alpha, linestyle=line.line.dash,
        marker=line.marker.symbol if 'markers' in (line.mode or '') else None,
        markersize=line.marker.size, linewidth=line.line.width,
        name=line.name,
        surfaces=sum(tr.type == 'mesh3d' for tr in traces))


#: `_PANEL_PER_DATASET_KWARGS` entry -> (two distinct values, the
#: `_style` keys they change). Every roster entry is either here, in
#: `_PER_DATASET_TESTED_ELSEWHERE`, or in `_PER_DATASET_STATIC_INVISIBLE`
#: (asserted below), so a new roster entry needs a partitioning test.
_PER_DATASET_CASES = {
    'fmt': (['r-', 'b--'], ('color', 'linestyle')),
    'marker': (['o', 's'], ('marker',)),
    'markers': (['o', 's'], ('marker',)),
    'linestyle': (['-', '--'], ('linestyle',)),
    'linestyles': (['-', '--'], ('linestyle',)),
    'color': (['red', 'blue'], ('color',)),
    'colors': (['red', 'blue'], ('color',)),
    'alpha': ([0.3, 0.7], ('alpha',)),
    'markersize': ([4, 10], ('markersize',)),
    'linewidth': ([1, 4], ('linewidth',)),
    'names': (['first', 'second'], ('name',)),
    'surface': ([True, False], ('surfaces',)),
}

_PER_DATASET_TESTED_ELSEWHERE = {
    'truth': 'test_truth_list_reaches_only_its_own_panel',
}

#: roster entries a STATIC grid cannot show: the three trails are
#: animation-only (`panels=` rejects `animate=`), and `density=` has no
#: per-dataset list form -- `plot()` rejects a list on the single-axes
#: path and on every panel alike.
_PER_DATASET_STATIC_INVISIBLE = {
    'chemtrails': 'animation only',
    'precog': 'animation only',
    'bullettime': 'animation only',
    'density': 'no per-dataset list form (plot() rejects a list)',
}


def test_every_per_dataset_roster_entry_has_a_partitioning_test():
    covered = (set(_PER_DATASET_CASES) | set(_PER_DATASET_TESTED_ELSEWHERE)
               | set(_PER_DATASET_STATIC_INVISIBLE))
    roster = set(plot_module._PANEL_PER_DATASET_KWARGS)
    assert roster == covered, (
        f'roster entries without a partitioning test: {roster - covered}; '
        f'tested names no longer on the roster: {covered - roster}')
    for name in _PER_DATASET_TESTED_ELSEWHERE.values():
        assert callable(globals().get(name)), name


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('fit', FITS)
@pytest.mark.parametrize('key', sorted(_PER_DATASET_CASES))
def test_per_dataset_argument_reaches_only_its_own_panel(backend, fit, key):
    """The reviewer's remark on the roster test: a name on
    `_PANEL_PER_DATASET_KWARGS` proves nothing about what a panel DRAWS.
    So, per entry: two datasets, two distinct values, and each panel's
    artist shows its own value -- the one the single-axes figure gives
    that dataset -- and not the other panel's."""
    values, props = _PER_DATASET_CASES[key]
    kwargs = {key: values}
    if key == 'names':
        kwargs['legend'] = True
    data = walks()
    single = quiet_plot(data, backend=backend, **kwargs)
    grid = quiet_plot(data, panels=True, panel_fit=fit, backend=backend,
                      **kwargs)
    try:
        styles = [_style(grid, i, backend) for i in range(2)]
        for i in range(2):
            want = _style(single, i, backend, panel=False)
            for prop in props:
                if prop == 'surfaces':
                    assert styles[i][prop] == int(values[i])
                    continue
                if prop != 'color':
                    assert styles[i][prop] == want[prop], (key, prop, i)
                else:
                    # includes fmt='s colour letter on plotly (round 6:
                    # 'r-' drew the palette colour there; the xfail that
                    # used to sit here is lifted)
                    assert same_color(styles[i][prop], want[prop]), \
                        (key, prop, i)
        for prop in props:
            assert styles[0][prop] != styles[1][prop], (key, prop)
    finally:
        close_all()


def _truth_points(bundle, i, backend):
    """The points of every `truth=` artist of panel `i`, in data units
    (2-D, ``axis_scale='data'``), the seam to the last observation
    dropped -- as `tests/test_plot_panels_fit.py` reads them."""
    if backend == 'matplotlib':
        return [np.asarray(ln.get_xydata())[1:]
                for ln in bundle['axes'][i].lines
                if getattr(ln, '_hyp_forecast_role', None) == 'truth']
    yaxis = bundle['axes'][i][0].anchor          # 'y', 'y2', ...
    return [np.column_stack([tr.x, tr.y])[1:] for tr in bundle['fig'].data
            if tr.yaxis == yaxis and isinstance(tr.meta, dict)
            and tr.meta.get('hyp_forecast_role') == 'truth']


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('fit', FITS)
def test_truth_list_reaches_only_its_own_panel(backend, fit):
    """``truth=[t0, t1]``: panel `i` draws exactly `t_i` (two different
    arrays), in both fit modes and on both backends. Drawn in the data's
    own 2-D units (`axis_scale='data'`): a 3-D frame is rescaled into the
    unit cube per axes, so its coordinates cannot be read back."""
    data = walks(cols=2)
    truth = [data[0][-1] + np.arange(1, 4)[:, None] * [1.0, 2.0],
             data[1][-1] - np.arange(1, 4)[:, None] * [3.0, 1.0]]
    grid = quiet_plot(data, panels=True, panel_fit=fit, backend=backend,
                      ndims=2, axis_scale='data', antialias=False,
                      predict='Kalman', t=3, truth=truth)
    try:
        for i in range(2):
            drawn = _truth_points(grid, i, backend)
            assert len(drawn) >= 1
            for points in drawn:
                assert np.allclose(points, truth[i])
                assert not np.allclose(points, truth[1 - i])
    finally:
        close_all()


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('fit', FITS)
def test_nested_hue_and_labels_reach_only_their_own_panel(backend, fit):
    """The per-OBSERVATION arguments, nested one sub-sequence per dataset:
    panel `i`'s single trace carries hue label `i`, and only annotation
    `i` is drawn on it."""
    data = walks()
    hue = [['first'] * 20, ['second'] * 20]
    labels = [['p0'] + [None] * 19, ['p1'] + [None] * 19]
    grid = quiet_plot(data, panels=True, panel_fit=fit, backend=backend,
                      hue=hue, labels=labels, legend=True)
    try:
        for i, (group, text) in enumerate((('first', 'p0'),
                                           ('second', 'p1'))):
            artists = data_artists(grid, i, backend)
            assert [a[3] for a in artists] == [group]
            if backend == 'matplotlib':
                drawn = [t.get_text() for t in grid['axes'][i].texts]
            else:
                drawn = [a.text for a in grid['axes'][i].annotations]
            assert drawn == [text]
    finally:
        close_all()
