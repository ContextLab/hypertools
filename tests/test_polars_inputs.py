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


@pytest.mark.xfail(
    strict=True,
    reason='wave 1: hypertools/manip/manip.py:114 funnels `data` without '
           "backend='pandas', so a polars input reaches the manipulators as a "
           'polars frame and hypertools/manip/zscore.py:46 (pandas '
           'Series.mean(axis=0) / pd.concat) fails on it')
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


@pytest.mark.xfail(
    strict=True,
    reason='wave 1: hypertools/plot/plot.py:1885 `_capture_column_names` '
           '(and :1851 `_capture_row_indices`) only capture axis labels '
           'from a pandas DataFrame, so a polars frame plots the same '
           'coordinates but drops its column names from the axes')
def test_plot_polars_column_names_become_axis_labels_like_pandas():
    values = np.random.RandomState(0).rand(N_ROWS, 2)
    p2 = pd.DataFrame(values, columns=['height', 'weight'])
    ref = hyp.plot(p2, show=False)
    out = hyp.plot(pl.from_pandas(p2), show=False)
    _same(_mpl_drawn(out), _mpl_drawn(ref))
    assert ref.axes[0].get_xlabel() == 'height'
    assert (out.axes[0].get_xlabel(), out.axes[0].get_ylabel()) == \
        (ref.axes[0].get_xlabel(), ref.axes[0].get_ylabel())


def test_polars_inputs_raise_no_warnings_beyond_pandas(pdf, plf):
    """A polars input must not add warnings of its own (e.g. a datawrangler
    deprecation on the conversion path) relative to the same pandas call."""
    def collect(x):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            hyp.reduce(x, ndims=2)
        return sorted({str(w.message) for w in caught})
    assert collect(plf) == collect(pdf)
