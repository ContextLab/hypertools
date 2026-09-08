# -*- coding: utf-8 -*-
"""polars inputs through every public entry point (datatype audit, 2026-09-08).

hypertools functions must not classify input datatypes themselves; they defer
to datawrangler (``dw.wrangle`` / the ``dw.zoo`` predicates) so that every
backend datawrangler recognises -- polars DataFrames and LazyFrames today,
whatever it adds later -- works without hypertools changes. Each test here
feeds a REAL polars DataFrame (and, where datawrangler accepts one, a
LazyFrame, and a list mixing polars with numpy/pandas) to a public entry point
and asserts the result is the SAME as for the equivalent pandas DataFrame.

Wave 0 made the shared coercion layer polars-aware (``tools/format_data.py``,
``_shared/helpers.py``, ``core/shared.py``, ``predict/predict.py``,
``impute/impute.py``, ``io/streaming.py``). Entry points that still fail
because a module OUTSIDE that layer hand-checks pandas types are marked
``xfail(strict=True)`` naming the blocking file and line, so lifting the block
in wave 1 flips them to XPASS and the marker must be removed.

No mocks: real polars, real pandas, real hypertools calls (show=False).
"""
import warnings

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.tools.format_data import format_data
from hypertools._shared.helpers import get_type, get_dtype
from hypertools.core.shared import as_dataframe
from hypertools.io.streaming import is_stream

pl = pytest.importorskip("polars")

pytestmark = pytest.mark.filterwarnings(
    'ignore:.*(Missing data|DataFrame column|reordering|do not share columns).*:UserWarning')

N_ROWS, N_COLS = 40, 4
COLUMNS = ['a', 'b', 'c', 'd']


def _make_pandas(seed=0, nan=False):
    rng = np.random.RandomState(seed)
    values = rng.rand(N_ROWS, N_COLS)
    if nan:
        values[3, 1] = np.nan
        values[10, 0] = np.nan
        values[25, 3] = np.nan
    return pd.DataFrame(values, columns=COLUMNS)


@pytest.fixture
def pdf():
    return _make_pandas()


@pytest.fixture
def plf(pdf):
    frame = pl.from_pandas(pdf)
    assert isinstance(frame, pl.DataFrame)
    return frame


@pytest.fixture
def lazy(plf):
    frame = plf.lazy()
    assert isinstance(frame, pl.LazyFrame)
    return frame


def _same(a, b):
    """Recursively compare two hypertools results: same container shape,
    same array shapes, all-close values (NaN == NaN), same DataFrame
    columns."""
    if isinstance(a, dict) or isinstance(b, dict):
        assert isinstance(a, dict) and isinstance(b, dict)
        assert set(a) == set(b)
        for key in a:
            _same(a[key], b[key])
        return
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        assert isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _same(x, y)
        return
    if hasattr(a, 'shape') or hasattr(b, 'shape'):
        assert hasattr(a, 'shape') and hasattr(b, 'shape')
        assert a.shape == b.shape
        if hasattr(a, 'columns') or hasattr(b, 'columns'):
            assert list(a.columns) == list(b.columns)
        np.testing.assert_allclose(np.asarray(a, dtype=float),
                                   np.asarray(b, dtype=float), equal_nan=True)
        return
    assert a == b


def _inputs(pdf, plf, lazy, kind):
    """(polars-flavoured input, pandas reference input) for one `kind`."""
    arr = np.random.RandomState(1).rand(N_ROWS, N_COLS)
    return {
        'polars': (plf, pdf),
        'lazy': (lazy, pdf),
        'mixed': ([arr, plf, pdf], [arr, pdf, pdf]),
    }[kind]


KINDS = ['polars', 'lazy', 'mixed']


# --- the coercion layer itself ---------------------------------------------

def test_format_data_polars_dataframe_matches_pandas(pdf, plf, lazy):
    ref = format_data(pdf)
    _same(format_data(plf), ref)
    _same(format_data(lazy), ref)
    assert isinstance(format_data(plf)[0], np.ndarray)
    assert format_data(plf)[0].dtype == ref[0].dtype


def test_format_data_mixed_list_matches_pandas(pdf, plf, lazy):
    arr = np.random.RandomState(1).rand(N_ROWS, N_COLS)
    ref = format_data([arr, pdf, pdf])
    _same(format_data([arr, plf, pdf]), ref)
    _same(format_data([arr, lazy, pdf]), ref)
    # nested groups holding polars frames are flattened exactly like pandas
    _same(format_data([[arr, plf], pdf]), ref)


def test_format_data_polars_string_column_dummy_coded_like_pandas():
    sdf = pd.DataFrame({'v': [1., 2., 3., 4.], 'c': ['x', 'y', 'x', 'y']})
    _same(format_data(pl.from_pandas(sdf)), format_data(sdf))
    assert format_data(pl.from_pandas(sdf))[0].shape == (4, 3)


def test_format_data_polars_nulls_are_missing_data_like_pandas_nan():
    pdf_nan = _make_pandas(nan=True)
    plf_nan = pl.from_pandas(pdf_nan)
    assert plf_nan.null_count().sum_horizontal().item() == 3
    # PPCA's fill is randomly initialised from numpy's global state: seed it
    # identically before each call so the two fills are comparable exactly
    np.random.seed(0)
    with pytest.warns(UserWarning, match='filling missing values'):
        ref = format_data(pdf_nan)
    np.random.seed(0)
    with pytest.warns(UserWarning, match='filling missing values'):
        out = format_data(plf_nan)
    _same(out, ref)
    assert not np.isnan(out[0]).any()
    # ppca=False keeps the nulls as NaN in the same cells
    _same(format_data(plf_nan, ppca=False), format_data(pdf_nan, ppca=False))
    assert np.isnan(format_data(plf_nan, ppca=False)[0]).sum() == 3


def test_format_data_polars_series_is_a_1d_dataset():
    values = np.arange(6.)
    ref = format_data(pd.Series(values))
    out = format_data(pl.Series('s', values))
    _same(out, ref)
    assert out[0].shape == (6, 1)
    # and inside a list
    _same(format_data([pl.Series('s', values), values]),
          format_data([pd.Series(values), values]))


def test_format_data_polars_named_columns_align_by_name_like_pandas():
    # GH #132 column-name alignment applies to polars frames too
    df1 = pd.DataFrame({'a': [1., 2., 3.], 'b': [10., 20., 30.]})
    df2 = pd.DataFrame({'b': [100., 200., 300.], 'a': [1000., 2000., 3000.]})
    with pytest.warns(UserWarning, match='reordering'):
        ref = format_data([df1, df2])
    with pytest.warns(UserWarning, match='reordering'):
        out = format_data([pl.from_pandas(df1), pl.from_pandas(df2)])
    _same(out, ref)
    assert np.allclose(out[1][:, 0], [1000., 2000., 3000.])


def test_get_type_and_get_dtype_classify_polars_as_dataframe(plf, lazy):
    assert get_type(plf) == 'df' == get_type(lazy)
    assert get_dtype(plf) == 'df' == get_dtype(lazy)
    # the vocabulary for the other inputs is unchanged
    assert get_type(np.zeros((3, 2))) == 'arr_num'
    assert get_type(['a b', 'c d']) == 'list_str'
    assert get_type([1., 2.]) == 'list_num'
    assert get_type([np.zeros(2)]) == 'list_arr'
    assert get_type('a b') == 'str'
    with pytest.raises(TypeError, match=r"Unsupported data type 'dict'"):
        get_type({'a': 1})


def test_as_dataframe_polars_becomes_pandas(pdf, plf, lazy):
    out = as_dataframe(plf)
    assert isinstance(out, pd.DataFrame)
    _same(out, pdf)
    _same(as_dataframe(lazy), pdf)
    # pandas passes through untouched (same object), arrays become frames
    assert as_dataframe(pdf) is pdf
    assert as_dataframe(np.arange(3.)).shape == (3, 1)


def test_is_stream_polars_lazyframe_is_not_a_stream(plf, lazy):
    assert not is_stream(lazy)
    assert not is_stream(plf)
    assert not is_stream(pl.Series('s', [1., 2.]))
    # the stream cases are unchanged
    assert is_stream(iter([np.zeros(3)]))
    assert is_stream(x for x in [np.zeros(3)])
    assert not is_stream([np.zeros((3, 2))])
    assert not is_stream(pd.DataFrame(np.zeros((3, 2))))
    assert not is_stream(pd.Series([1., 2.]))
    assert not is_stream('text')


# --- public entry points ---------------------------------------------------

@pytest.mark.parametrize('kind', KINDS)
def test_reduce_polars_matches_pandas(pdf, plf, lazy, kind):
    x, ref = _inputs(pdf, plf, lazy, kind)
    _same(hyp.reduce(x, ndims=2), hyp.reduce(ref, ndims=2))


@pytest.mark.parametrize('kind', KINDS)
def test_align_polars_matches_pandas(pdf, plf, lazy, kind):
    x, ref = _inputs(pdf, plf, lazy, kind)
    if kind != 'mixed':
        x, ref = [x, x], [ref, ref]
    _same(hyp.align(x), hyp.align(ref))


@pytest.mark.parametrize('kind', KINDS)
def test_cluster_polars_matches_pandas(pdf, plf, lazy, kind):
    x, ref = _inputs(pdf, plf, lazy, kind)
    # KMeans is seeded so the pandas and polars labelings are comparable
    spec = {'model': 'KMeans', 'kwargs': {'random_state': 0}}
    _same(hyp.cluster(x, cluster=spec, n_clusters=3),
          hyp.cluster(ref, cluster=spec, n_clusters=3))


@pytest.mark.parametrize('kind', KINDS)
def test_normalize_polars_matches_pandas(pdf, plf, lazy, kind):
    x, ref = _inputs(pdf, plf, lazy, kind)
    _same(hyp.normalize(x), hyp.normalize(ref))


@pytest.mark.parametrize('kind', KINDS)
def test_manip_polars_matches_pandas(pdf, plf, lazy, kind):
    x, ref = _inputs(pdf, plf, lazy, kind)
    _same(hyp.manip(x, model='ZScore'), hyp.manip(ref, model='ZScore'))


@pytest.mark.parametrize('kind', KINDS)
def test_predict_polars_matches_pandas(pdf, plf, lazy, kind):
    x, ref = _inputs(pdf, plf, lazy, kind)
    out = hyp.predict(x, model='AutoRegressor', t=3)
    _same(out, hyp.predict(ref, model='AutoRegressor', t=3))
    frame = out[1] if kind == 'mixed' else out
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == COLUMNS


def test_predict_polars_series_matches_pandas_series():
    values = np.cumsum(np.random.RandomState(0).rand(30))
    out = hyp.predict(pl.Series('s', values), model='AutoRegressor', t=3)
    ref = hyp.predict(pd.Series(values, name='s'), model='AutoRegressor', t=3)
    _same(out, ref)
    assert list(out.columns) == ['s']


@pytest.mark.parametrize('kind', KINDS)
def test_impute_polars_matches_pandas(kind):
    pdf_nan = _make_pandas(nan=True)
    plf_nan = pl.from_pandas(pdf_nan)
    x, ref = _inputs(pdf_nan, plf_nan, plf_nan.lazy(), kind)
    out = hyp.impute(x, model='KNNImputer')
    _same(out, hyp.impute(ref, model='KNNImputer'))
    frame = out[1] if kind == 'mixed' else out
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == COLUMNS
    assert not np.isnan(frame.to_numpy()).any()


def test_impute_polars_series_with_null_matches_pandas_series():
    values = np.arange(12.)
    values[4] = np.nan
    out = hyp.impute(pl.Series('s', values), model='SimpleImputer')
    ref = hyp.impute(pd.Series(values, name='s'), model='SimpleImputer')
    _same(out, ref)
    assert list(out.columns) == ['s']
    assert not np.isnan(out.to_numpy()).any()


@pytest.mark.parametrize('kind', KINDS)
def test_analyze_polars_matches_pandas(pdf, plf, lazy, kind):
    x, ref = _inputs(pdf, plf, lazy, kind)
    _same(hyp.analyze(x, reduce='PCA', ndims=2),
          hyp.analyze(ref, reduce='PCA', ndims=2))


@pytest.mark.parametrize('kind', KINDS)
def test_describe_polars_matches_pandas(pdf, plf, lazy, kind):
    x, ref = _inputs(pdf, plf, lazy, kind)
    _same(hyp.describe(x, show=False), hyp.describe(ref, show=False))


def _mpl_drawn(fig):
    """Every drawn coordinate array of a matplotlib figure (lines and
    scatter collections), in drawing order."""
    drawn = []
    for ax in fig.axes:
        for line in ax.lines:
            data = (line.get_data_3d() if hasattr(line, 'get_data_3d')
                    else line.get_data())
            drawn.append(np.asarray(data, dtype=float))
        for coll in ax.collections:
            drawn.append(np.asarray(coll.get_offsets(), dtype=float))
    assert drawn, 'nothing was drawn'
    return drawn


@pytest.mark.parametrize('kind', KINDS)
def test_plot_matplotlib_polars_matches_pandas(pdf, plf, lazy, kind):
    x, ref = _inputs(pdf, plf, lazy, kind)
    out = hyp.plot(x, show=False)
    _same(_mpl_drawn(out), _mpl_drawn(hyp.plot(ref, show=False)))


def _plotly_traces(fig):
    traces = []
    for trace in fig.data:
        coords = [trace.x, trace.y]
        if getattr(trace, 'z', None) is not None:
            coords.append(trace.z)
        traces.append((trace.type, np.asarray(coords, dtype=float)))
    assert traces, 'no traces'
    return traces


@pytest.mark.parametrize('kind', KINDS)
def test_plot_plotly_polars_matches_pandas(pdf, plf, lazy, kind):
    x, ref = _inputs(pdf, plf, lazy, kind)
    hyp.set_interactive_backend('plotly')
    try:
        out = hyp.plot(x, show=False)
        expected = hyp.plot(ref, show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    got, want = _plotly_traces(out), _plotly_traces(expected)
    assert [t for t, _ in got] == [t for t, _ in want]
    _same([c for _, c in got], [c for _, c in want])


def test_plot_polars_column_names_become_axis_labels_like_pandas():
    # wave 1 lifted: `_capture_column_names` / `_dataframe_axis_labels`
    # read column names through the datawrangler predicates
    values = np.random.RandomState(0).rand(N_ROWS, 2)
    p2 = pd.DataFrame(values, columns=['height', 'weight'])
    ref = hyp.plot(p2, show=False)
    out = hyp.plot(pl.from_pandas(p2), show=False)
    _same(_mpl_drawn(out), _mpl_drawn(ref))
    assert ref.axes[0].get_xlabel() == 'height'
    assert (out.axes[0].get_xlabel(), out.axes[0].get_ylabel()) == \
        (ref.axes[0].get_xlabel(), ref.axes[0].get_ylabel())
    # and the 3-D case names all three axes
    p3 = pd.DataFrame(np.random.RandomState(1).rand(N_ROWS, 3),
                      columns=['x1', 'x2', 'x3'])
    ref3 = hyp.plot(p3, show=False)
    out3 = hyp.plot(pl.from_pandas(p3), show=False)
    assert ref3.axes[0].get_zlabel() == 'x3'
    assert (out3.axes[0].get_xlabel(), out3.axes[0].get_ylabel(),
            out3.axes[0].get_zlabel()) == ('x1', 'x2', 'x3')


def test_plot_plotly_polars_column_names_become_axis_labels_like_pandas():
    values = np.random.RandomState(0).rand(N_ROWS, 2)
    p2 = pd.DataFrame(values, columns=['height', 'weight'])
    hyp.set_interactive_backend('plotly')
    try:
        ref = hyp.plot(p2, show=False)
        out = hyp.plot(pl.from_pandas(p2), show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    assert ref.layout.xaxis.title.text == 'height'
    assert (out.layout.xaxis.title.text, out.layout.yaxis.title.text) == \
        (ref.layout.xaxis.title.text, ref.layout.yaxis.title.text)
    _same([c for _, c in _plotly_traces(out)],
          [c for _, c in _plotly_traces(ref)])


# --- plot-level polars arguments (hue / labels / truth / palette / panels) ---

@pytest.fixture
def frame3():
    """A 30 x 3 named frame as (polars, pandas)."""
    rng = np.random.RandomState(3)
    pdf3 = pd.DataFrame(rng.rand(30, 3), columns=['a', 'b', 'c'])
    return pl.from_pandas(pdf3), pdf3


def _both_backends(call):
    """``(matplotlib_result, plotly_result)`` of `call(backend)``, with the
    backend restored afterwards."""
    mpl = call('matplotlib')
    hyp.set_interactive_backend('plotly')
    try:
        ply = call('plotly')
    finally:
        hyp.set_interactive_backend('matplotlib')
    return mpl, ply


def _mpl_colors(fig):
    """Every drawn colour of a matplotlib figure, in drawing order."""
    colors = []
    for ax in fig.axes:
        for line in ax.lines:
            colors.append(np.asarray(mcolors.to_rgba(line.get_color())))
        for coll in ax.collections:
            fc = np.asarray(coll.get_facecolor(), dtype=float)
            if fc.size:
                colors.append(fc)
    return colors


def _plotly_colors(fig):
    return [(t.type, getattr(t.marker, 'color', None),
             getattr(t.line, 'color', None)) for t in fig.data]


def _plotly_annotation_texts(fig):
    """Every annotation text of a plotly figure (3-D labels live on the
    scene, 2-D ones on the layout)."""
    texts = [a.text for a in fig.layout.annotations]
    if fig.layout.scene is not None:
        texts += [a.text for a in fig.layout.scene.annotations]
    return texts


def _assert_same_figure(out, ref, backend):
    """The polars-argument figure is drawn exactly like the pandas one."""
    if backend == 'matplotlib':
        _same(_mpl_drawn(out), _mpl_drawn(ref))
        _same(_mpl_colors(out), _mpl_colors(ref))
        assert ([t.get_text() for t in out.axes[0].texts]
                == [t.get_text() for t in ref.axes[0].texts])
        assert len(out.axes) == len(ref.axes)
    else:
        got, want = _plotly_traces(out), _plotly_traces(ref)
        assert [t for t, _ in got] == [t for t, _ in want]
        _same([c for _, c in got], [c for _, c in want])
        assert _plotly_colors(out) == _plotly_colors(ref)
        assert [t.name for t in out.data] == [t.name for t in ref.data]
        assert _plotly_annotation_texts(out) == _plotly_annotation_texts(ref)


@pytest.mark.parametrize('form', ['series', 'frame'])
def test_plot_hue_from_polars_categorical(frame3, form):
    plf3, pdf3 = frame3
    labels = ['x', 'y', 'z'] * 10
    p_hue = pd.Series(labels, name='grp')
    hue = pl.Series('grp', labels)
    if form == 'frame':
        p_hue, hue = p_hue.to_frame(), hue.to_frame()

    def call(backend):
        out = hyp.plot(plf3, hue=hue, fmt='o-', legend=True, show=False)
        ref = hyp.plot(pdf3, hue=p_hue, fmt='o-', legend=True, show=False)
        _assert_same_figure(out, ref, backend)
        return out
    mpl, ply = _both_backends(call)
    # three categories, three colours: the polars hue was grouped, not
    # dropped
    assert len({tuple(c) for c in map(tuple, np.round(
        [mcolors.to_rgba(line.get_color()) for line in mpl.axes[0].lines],
        6))}) == 3
    assert len(ply.data) >= 3
    legend = mpl.axes[0].get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == ['x', 'y', 'z']


@pytest.mark.parametrize('form', ['series', 'frame'])
def test_plot_hue_from_polars_numeric(frame3, form):
    plf3, pdf3 = frame3
    values = np.linspace(0., 1., 30)
    p_hue = pd.Series(values, name='val')
    hue = pl.Series('val', values)
    if form == 'frame':
        p_hue, hue = p_hue.to_frame(), hue.to_frame()

    def call(backend):
        out = hyp.plot(plf3, hue=hue, fmt='.', show=False)
        ref = hyp.plot(pdf3, hue=p_hue, fmt='.', show=False)
        _assert_same_figure(out, ref, backend)
        return out
    mpl, _ = _both_backends(call)
    # a continuous hue: more than three distinct colours over 30 points
    colors = np.round(np.vstack([c for c in _mpl_colors(mpl)
                                 if c.ndim == 2]), 6)
    assert len({tuple(c) for c in colors}) > 3


def test_plot_hue_matrix_from_polars_frame_keeps_column_names(frame3):
    """A multi-column polars frame is a matrix hue whose column names label
    the legend, exactly like the pandas frame (GH #285)."""
    plf3, pdf3 = frame3
    rng = np.random.RandomState(5)
    weights = rng.rand(30, 2)
    weights /= weights.sum(axis=1, keepdims=True)
    p_hue = pd.DataFrame(weights, columns=['alpha', 'beta'])
    hue = pl.from_pandas(p_hue)
    out = hyp.plot(plf3, hue=hue, fmt='.', legend=True, show=False)
    ref = hyp.plot(pdf3, hue=p_hue, fmt='.', legend=True, show=False)
    _assert_same_figure(out, ref, 'matplotlib')
    legend = out.axes[0].get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert labels == ['alpha', 'beta']
    assert labels == [t.get_text()
                      for t in ref.axes[0].get_legend().get_texts()]


def test_plot_labels_from_polars_column(frame3):
    plf3, pdf3 = frame3
    names = [f'obs{i}' for i in range(30)]
    p_lab = pdf3.assign(name=names)
    pl_lab = pl.from_pandas(p_lab)

    def call(backend):
        out = hyp.plot(plf3, labels=pl_lab['name'], show=False)
        ref = hyp.plot(pdf3, labels=p_lab['name'], show=False)
        _assert_same_figure(out, ref, backend)
        return out
    mpl, ply = _both_backends(call)
    assert [t.get_text() for t in mpl.axes[0].texts] == names
    assert _plotly_annotation_texts(ply) == names


def test_plot_truth_from_polars_frame(frame3):
    plf3, pdf3 = frame3
    rng = np.random.RandomState(7)
    p_truth = pd.DataFrame(rng.rand(5, 3), columns=['a', 'b', 'c'])
    truth = pl.from_pandas(p_truth)

    def call(backend):
        out = hyp.plot(plf3, predict='AutoRegressor', t=5, truth=truth,
                       show=False)
        ref = hyp.plot(pdf3, predict='AutoRegressor', t=5, truth=p_truth,
                       show=False)
        _assert_same_figure(out, ref, backend)
        return out
    mpl, _ = _both_backends(call)
    # data line + forecast + truth were all drawn
    assert len(mpl.axes[0].lines) >= 3
    # and a truth of the wrong length is refused for polars as for pandas
    with pytest.raises(ValueError, match='exactly t=5 rows'):
        hyp.plot(plf3, predict='AutoRegressor', t=5,
                 truth=pl.from_pandas(p_truth.iloc[:3]), show=False)


def test_plot_mixed_polars_and_pandas_datasets(frame3):
    plf3, pdf3 = frame3
    rng = np.random.RandomState(11)
    other = pd.DataFrame(rng.rand(20, 3), columns=['a', 'b', 'c'])
    arr = rng.rand(10, 3)

    def call(backend):
        out = hyp.plot([plf3, other, arr], show=False)
        ref = hyp.plot([pdf3, other, arr], show=False)
        _assert_same_figure(out, ref, backend)
        return out
    mpl, ply = _both_backends(call)
    assert len(mpl.axes[0].lines) == 3
    assert len(ply.data) >= 3
    # per-dataset hue lists mixing a polars Series with a python list
    hue_pl = [pl.Series('g', ['p'] * 30), ['q'] * 20, ['r'] * 10]
    hue_pd = [pd.Series(['p'] * 30), ['q'] * 20, ['r'] * 10]
    out = hyp.plot([plf3, other, arr], hue=hue_pl, show=False)
    ref = hyp.plot([pdf3, other, arr], hue=hue_pd, show=False)
    _assert_same_figure(out, ref, 'matplotlib')


def test_plot_polars_frame_as_matrix_palette(frame3):
    from hypertools.plot.colors import is_palette_matrix, matrix_palette
    plf3, pdf3 = frame3
    rng = np.random.RandomState(13)
    p_pal = pd.DataFrame(rng.rand(30, 2), columns=['u', 'v'])
    pal = pl.from_pandas(p_pal)
    assert is_palette_matrix(pal) and is_palette_matrix(p_pal)
    assert is_palette_matrix(pal.lazy())
    # a polars frame of strings is not a palette matrix (pandas rule)
    assert not is_palette_matrix(pl.DataFrame({'s': ['a', 'b']}))
    cm_pl, cm_pd = matrix_palette(pal), matrix_palette(p_pal)
    samples = np.linspace(0., 1., 7)
    np.testing.assert_allclose(cm_pl(samples), cm_pd(samples))
    assert cm_pl(samples).shape == (7, 4)

    def call(backend):
        out = hyp.plot(plf3, hue=np.linspace(0, 1, 30), palette=pal,
                       fmt='.', show=False)
        ref = hyp.plot(pdf3, hue=np.linspace(0, 1, 30), palette=p_pal,
                       fmt='.', show=False)
        _assert_same_figure(out, ref, backend)
        return out
    _both_backends(call)


def test_plot_panels_with_polars_input(frame3):
    plf3, pdf3 = frame3
    rng = np.random.RandomState(17)
    other = pd.DataFrame(rng.rand(24, 3), columns=['a', 'b', 'c'])

    def call(backend):
        out = hyp.plot([plf3, pl.from_pandas(other)], panels=True,
                       show=False)
        ref = hyp.plot([pdf3, other], panels=True, show=False)
        _assert_same_figure(out, ref, backend)
        return out
    mpl, ply = _both_backends(call)
    assert len(mpl.axes) >= 2
    # panel axis labels come from the polars column names too
    assert [ax.get_xlabel() for ax in mpl.axes[:2]] == ['a', 'a']
    # series mode (ndims=1, one line per column with reduce=None) panels:
    # the polars frame's columns name the lines exactly as the pandas
    # frame's do (`_capture_column_names` through `_panel_frame`)
    out = hyp.plot([plf3, pl.from_pandas(other)], panels=True, ndims=1,
                   reduce=None, legend=True, show=False)
    ref = hyp.plot([pdf3, other], panels=True, ndims=1, reduce=None,
                   legend=True, show=False)
    _assert_same_figure(out, ref, 'matplotlib')
    legends = [[t.get_text() for t in ax.get_legend().get_texts()]
               for ax in out.axes if ax.get_legend()]
    assert legends == [['a', 'b', 'c'], ['a', 'b', 'c']]
    assert legends == [[t.get_text() for t in ax.get_legend().get_texts()]
                       for ax in ref.axes if ax.get_legend()]


def test_plot_series_mode_names_polars_columns_like_pandas(frame3):
    """`_capture_column_names` (wave 1): in ndims=1 series mode a polars
    frame's columns name the drawn lines, and a single named column names
    the y axis, exactly as for the pandas frame."""
    plf3, pdf3 = frame3

    def call(backend):
        out = hyp.plot(plf3, ndims=1, reduce=None, legend=True, show=False)
        ref = hyp.plot(pdf3, ndims=1, reduce=None, legend=True, show=False)
        _assert_same_figure(out, ref, backend)
        return out, ref
    (mpl_out, mpl_ref), (ply_out, ply_ref) = _both_backends(call)
    names = [t.get_text() for t in mpl_out.axes[0].get_legend().get_texts()]
    assert names == ['a', 'b', 'c']
    assert names == [t.get_text()
                     for t in mpl_ref.axes[0].get_legend().get_texts()]
    assert [t.name for t in ply_out.data][:3] == ['a', 'b', 'c']
    one_pl = hyp.plot(plf3.select('b'), ndims=1, show=False)
    one_pd = hyp.plot(pdf3[['b']], ndims=1, show=False)
    assert one_pl.axes[0].get_ylabel() == 'b' == one_pd.axes[0].get_ylabel()
    # a polars Series is named after itself, like a pandas Series
    s_pl = hyp.plot(plf3['c'], ndims=1, show=False)
    s_pd = hyp.plot(pdf3['c'], ndims=1, show=False)
    assert s_pl.axes[0].get_ylabel() == 'c' == s_pd.axes[0].get_ylabel()
    _same(_mpl_drawn(s_pl), _mpl_drawn(s_pd))


def test_plot_polars_date_column_matches_pandas_date_column():
    """polars has no row index: a frame with a datetime column plots exactly
    as the pandas frame with the same datetime COLUMN (and a RangeIndex),
    on both backends -- not as the pandas frame whose DatetimeIndex puts
    dates on the x axis of `ndims=1` series mode (that is a pandas index
    feature; ``pl_df.to_pandas().set_index('date')`` opts into it)."""
    dates = pd.date_range('2020-01-01', periods=12, freq='D', name='date')
    rng = np.random.RandomState(19)
    indexed = pd.DataFrame({'v': rng.rand(12), 'w': rng.rand(12)},
                           index=dates)
    with_column = indexed.reset_index()
    assert list(with_column.columns) == ['date', 'v', 'w']
    polars = pl.from_pandas(with_column)
    assert polars.schema['date'].is_temporal()

    def call(backend):
        out = hyp.plot(polars, ndims=1, reduce=None, show=False)
        ref = hyp.plot(with_column, ndims=1, reduce=None, show=False)
        _assert_same_figure(out, ref, backend)
        return out, ref
    (mpl_out, mpl_ref), _ = _both_backends(call)
    # the datetime index version draws real dates on x (pandas only)
    import matplotlib.dates as mdates
    idx_fig = hyp.plot(indexed, ndims=1, reduce=None, show=False)
    assert idx_fig.axes[0].lines[0].get_xdata()[0] == \
        mdates.date2num(dates[0])
    # ...and the polars frame, like the pandas column frame, draws row
    # positions on x (0..11), with the date column as one more series
    assert mpl_out.axes[0].lines[0].get_xdata()[0] == 0.0
    assert mpl_out.axes[0].get_xlim() == mpl_ref.axes[0].get_xlim()
    assert mpl_out.axes[0].get_xlim()[1] < 20
    assert len(mpl_out.axes[0].lines) == len(mpl_ref.axes[0].lines) == 3
    # a polars frame's default row index is the pandas default, so a
    # {index} title pattern is refused for both with the same message
    with pytest.raises(ValueError, match='index') as pl_err:
        hyp.plot(polars, ndims=1, title='{index}', animate=True, show=False)
    with pytest.raises(ValueError, match='index') as pd_err:
        hyp.plot(with_column, ndims=1, title='{index}', animate=True,
                 show=False)
    assert str(pl_err.value) == str(pd_err.value)


def test_polars_inputs_raise_no_warnings_beyond_pandas(pdf, plf):
    """A polars input must not add warnings of its own (e.g. a datawrangler
    deprecation on the conversion path) relative to the same pandas call."""
    def collect(x):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            hyp.reduce(x, ndims=2)
        return sorted({str(w.message) for w in caught})
    assert collect(plf) == collect(pdf)
