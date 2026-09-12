"""Codex round 12 (2026-09-08) regressions: a pandas Series keeps its index
and name through the Manipulator classes and `hyp.Pipeline` (R12-4), a
polars Series beside an array in a `hyp.manip` list works (R12-1), a mixed
named-frame/array list keeps the frame's feature names for every model
(R12-3), `MatrixColormap` follows matplotlib's full RGBA extreme-color
rules (R12-2), `legend=` takes a polars Series of labels (R12-5), and a 1-D
array is one column for `hyp.manip` as it is for `hyp.normalize`. Real
data, real models, real figures on both backends; no mocks."""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import LinearSegmentedColormap

import hypertools as hyp
from hypertools.manip import ZScore, Normalize, Smooth, Resample, Delay
from hypertools.plot.colors import MatrixColormap

pl = pytest.importorskip('polars')

pytestmark = pytest.mark.filterwarnings(
    'ignore:.*(Missing data|DataFrame column|reordering|do not share columns'
    '|copy keyword).*')


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close('all')


def _same(a, b):
    if isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b)
        for x, y in zip(a, b):
            _same(x, y)
        return
    pd.testing.assert_frame_equal(pd.DataFrame(a), pd.DataFrame(b))


SERIES = pd.Series([0., 1., 4., 9., 16., 25., 36.],
                   index=[0., 1., 2., 8., 10., 15., 20.], name='signal')
# the values `Pipeline([Smooth(kernel_width=5), Resample(n_samples=9)])`
# produced on 650808f0 (before the datatype refactor): resampled at the
# Series' OWN positions 0..20, not at 0..6
PIPELINE_VALUES = [0.0, 4.6743116472, 6.5053827751, 8.3289778265, 16.0,
                   20.8197033898, 25.0, 30.2375, 36.0]


# --- R12-4: Series metadata through the direct classes and Pipeline ---------

def test_direct_smooth_and_delay_keep_a_series_index_and_name():
    smoothed = Smooth(kernel_width=5).fit_transform(SERIES)
    assert list(smoothed.index) == list(SERIES.index)
    assert list(smoothed.columns) == ['signal']
    delayed = Delay().fit_transform(SERIES)
    assert list(delayed.columns) == ['signal_lag1', 'signal_lag0']
    assert list(delayed.index) == list(SERIES.index)[1:]
    # the same through hyp.manip, and for a polars Series
    _same(hyp.manip(SERIES, model='Smooth', kernel_width=5), smoothed)
    polars_smoothed = Smooth(kernel_width=5).fit_transform(
        pl.Series('signal', SERIES.to_numpy()))
    assert list(polars_smoothed.columns) == ['signal']
    assert np.allclose(polars_smoothed.to_numpy(), smoothed.to_numpy())


def test_pipeline_resamples_a_series_at_its_own_positions():
    pipe = hyp.Pipeline([('smooth', Smooth(kernel_width=5)),
                         ('resample', Resample(n_samples=9))])
    out = pipe.fit_transform(SERIES)
    assert np.allclose(np.asarray(out.index, dtype=float), np.linspace(0, 20, 9))
    assert np.allclose(np.asarray(out).ravel(), PIPELINE_VALUES, atol=1e-9)
    assert list(out.columns) == ['signal']
    # a fitted manipulator reused on a Series keeps its index too
    fitted = Smooth(kernel_width=5).fit(SERIES)
    again = fitted.transform(SERIES * 2)
    assert list(again.index) == list(SERIES.index)
    assert np.allclose(again.to_numpy(), 2 * Smooth(kernel_width=5).fit_transform(SERIES).to_numpy())


@pytest.mark.parametrize('cls,kwargs', [
    (ZScore, {}), (Normalize, {}), (Smooth, {'kernel_width': 5}),
    (Resample, {'n_samples': 20}), (Delay, {'tau': 1, 'dims': 2})],
    ids=['ZScore', 'Normalize', 'Smooth', 'Resample', 'Delay'])
def test_direct_classes_on_a_dated_series_match_its_one_column_frame(cls, kwargs):
    dated = pd.Series(np.sin(np.arange(30.) / 3),
                      index=pd.date_range('2024-01-01', periods=30), name='v')
    _same(cls(**kwargs).fit_transform(dated),
          cls(**kwargs).fit_transform(dated.to_frame()))
    _same(cls(**kwargs).fit_transform(pl.Series('v', dated.to_numpy())),
          cls(**kwargs).fit_transform(dated.reset_index(drop=True).to_frame()))


# --- R12-1 / 1-D arrays: a polars Series beside an array -------------------

MODELS = [('ZScore', {}), ('Normalize', {}), ('Smooth', {'kernel_width': 5}),
          ('Resample', {'n_samples': 7}), ('Delay', {'dims': 2})]


@pytest.mark.parametrize('model,kwargs', MODELS, ids=[m for m, _ in MODELS])
def test_polars_series_beside_an_array_matches_pandas(model, kwargs):
    values = np.arange(12.) ** 1.5
    series = pl.Series('a', values)
    out = hyp.manip([series, values], model=model, **kwargs)
    ref = hyp.manip([series.to_pandas(), values], model=model, **kwargs)
    _same(out, ref)
    width = 2 if model == 'Delay' else 1          # Delay embeds dims=2
    assert [np.asarray(o).shape[1] for o in out] == [width, width]


@pytest.mark.parametrize('model,kwargs', MODELS, ids=[m for m, _ in MODELS])
def test_a_1d_array_is_one_column_for_manip_like_normalize(model, kwargs):
    values = np.arange(12.) ** 1.5
    alone = hyp.manip(values, model=model, **kwargs)
    column = hyp.manip(values.reshape(-1, 1), model=model, **kwargs)
    _same(alone, column)
    assert np.asarray(alone).shape[1] == (2 if model == 'Delay' else 1)
    if model == 'ZScore':
        # the same reading `hyp.normalize` gives a 1-D array (population
        # vs sample std aside): one feature, twelve observations
        assert hyp.normalize(values).shape == (12, 1)
        assert not np.allclose(np.asarray(alone), 0.0)


# --- R12-3: mixed lists keep every frame's own feature names ---------------

def _named():
    return pd.DataFrame(np.arange(20.).reshape(10, 2) ** 1.1,
                        columns=['a', 'b'],
                        index=pd.date_range('2020-01-01', periods=10))


@pytest.mark.parametrize('model,kwargs', [
    ('Smooth', {'kernel_width': 5}), ('Resample', {'n_samples': 7}),
    ('Delay', {'dims': 2})], ids=['Smooth', 'Resample', 'Delay'])
def test_independent_manipulators_keep_names_beside_an_array(model, kwargs):
    frame = _named()
    alone = hyp.manip(frame, model=model, **kwargs)
    mixed = hyp.manip([frame, frame.to_numpy()], model=model, **kwargs)
    _same(mixed[0], alone)
    assert list(mixed[0].columns) == list(alone.columns)
    assert list(mixed[1].columns) == (
        [0, 1] if model != 'Delay' else ['0_lag1', '0_lag0', '1_lag1', '1_lag0'])
    assert np.allclose(mixed[1].to_numpy(), alone.to_numpy())


@pytest.mark.parametrize('model', ['ZScore', 'Normalize'])
def test_shared_statistics_are_positional_and_names_survive(model):
    frame = _named()
    other = pd.DataFrame(np.random.default_rng(3).normal(size=(6, 2)) * 5,
                         columns=['x', 'y'])
    arr = other.to_numpy()
    mixed = hyp.manip([frame, arr], model=model)
    assert list(mixed[0].columns) == ['a', 'b']
    assert list(mixed[0].index) == list(frame.index)
    assert list(mixed[1].columns) == [0, 1]
    # the statistics are shared across BOTH datasets: identical to fitting
    # the same values as two unnamed arrays, and to a by-hand computation
    unnamed = hyp.manip([frame.to_numpy(), arr], model=model)
    for got, want in zip(mixed, unnamed):
        assert np.allclose(got.to_numpy(), want.to_numpy())
    stacked = np.vstack([frame.to_numpy(), arr])
    if model == 'ZScore':
        expected = (frame.to_numpy() - stacked.mean(axis=0)) / stacked.std(axis=0, ddof=1)
    else:
        lo, hi = stacked.min(axis=0), stacked.max(axis=0)
        expected = (frame.to_numpy() - lo) / (hi - lo)
    assert np.allclose(mixed[0].to_numpy(), expected)
    # two named frames with DIFFERENT labels are matched by position too
    two = hyp.manip([frame, other], model=model)
    assert [list(t.columns) for t in two] == [['a', 'b'], ['x', 'y']]
    assert np.allclose(two[0].to_numpy(), expected)
    # different widths cannot share statistics
    with pytest.raises(ValueError, match='same number of columns'):
        hyp.manip([frame, np.ones((4, 3))], model=model)


# --- R12-2: MatrixColormap extreme colors, alpha and infinities ------------

ANCHORS = [[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]]
X = np.array([-0.1, 1.1, -np.inf, np.inf, np.nan, 0.5])


def _pair(**extremes):
    ours = MatrixColormap('ours', ANCHORS)
    ref = LinearSegmentedColormap.from_list('ref', ANCHORS)
    for cmap in (ours, ref):
        for key, (color, alpha) in extremes.items():
            getattr(cmap, f'set_{key}')(color, alpha=alpha)
    return ours, ref


def test_extremes_keep_their_alpha_and_infinities_are_under_and_over():
    ours, ref = _pair(under=('red', .2), over=('blue', .4), bad=('green', .6))
    got, want = ours(X), ref(X)
    assert np.allclose(got[:5], want[:5])          # the extremes, RGBA
    assert got[:5, 3].tolist() == [.2, .4, .2, .4, .6]
    assert np.allclose(got[5], [0.5, 0.5, 0.5, 1.0], atol=1e-9)   # exact, not the LUT
    assert np.allclose(got[5, :3], want[5, :3], atol=1 / 255)


def test_alpha_override_reaches_the_extremes_but_not_a_transparent_bad():
    ours, ref = _pair(under=('red', .2), over=('blue', .4), bad=('green', .6))
    assert np.allclose(ours(X, alpha=.3)[:, 3], .3)
    assert np.allclose(ours(X, alpha=.3), ref(X, alpha=.3), atol=1 / 255)
    plain, plain_ref = MatrixColormap('p', ANCHORS), LinearSegmentedColormap.from_list('p', ANCHORS)
    got, want = plain(X, alpha=.3), plain_ref(X, alpha=.3)
    assert np.allclose(got[4], 0.0) and np.allclose(want[4], 0.0)   # bad stays transparent
    assert np.allclose(got[[0, 1, 5], 3], .3)
    per_element = np.linspace(0.1, 0.6, len(X))
    assert np.allclose(ours(X, alpha=per_element)[:, 3], per_element)
    with pytest.raises(ValueError, match='alpha is array-like'):
        ours(X, alpha=np.ones(3))


def test_bytes_and_masked_input_match_the_parent():
    ours, ref = _pair(under=('red', .2), over=('blue', .4), bad=('green', .6))
    assert ours(X, bytes=True)[:5].tolist() == ref(X, bytes=True)[:5].tolist()
    masked = np.ma.array([0.25, 0.75], mask=[False, True])
    assert np.allclose(ours(masked), ref(masked), atol=1 / 255)
    assert np.allclose(ours(masked)[1], ref.get_bad())
    assert ours(0.5) == pytest.approx((0.5, 0.5, 0.5, 1.0))   # scalar in, tuple out


# --- R12-5: legend labels as a polars Series -------------------------------

def _legend_names(fig, backend):
    if backend == 'matplotlib':
        return [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    return [tr.name for tr in fig.data if tr.showlegend]


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
@pytest.mark.parametrize('container', ['polars', 'pandas', 'index', 'array'])
def test_legend_labels_from_any_series_like(backend, container):
    x = pd.DataFrame(np.random.default_rng(0).normal(size=(20, 3)))
    labels = {'polars': pl.Series(['first', 'second']),
              'pandas': pd.Series(['first', 'second']),
              'index': pd.Index(['first', 'second']),
              'array': np.array(['first', 'second'])}[container]
    fig = hyp.plot([x, x + 2], legend=labels, backend=backend, show=False)
    assert _legend_names(fig, backend) == ['first', 'second']
    with pytest.raises(TypeError, match='legend= must be'):
        hyp.plot([x, x + 2], legend=7, backend=backend, show=False)


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_legend_length_mismatch_is_reported_for_a_polars_series(backend):
    x = pd.DataFrame(np.random.default_rng(0).normal(size=(20, 3)))
    with pytest.raises(ValueError, match='legend= was given as a list of length'):
        hyp.plot([x, x + 2], legend=pl.Series(['only']), backend=backend, show=False)
