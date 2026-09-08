# -*- coding: utf-8 -*-
"""polars inputs through the wave-1 entry points (datatype audit, 2026-09-08).

Wave 1 removed the last hand-rolled pandas/numpy type checks OUTSIDE the
shared coercion layer -- the manipulators (`hyp.manip` and the Manipulator
classes used directly), `hyp.align`, `hyp.stack`, `hyp.damage`,
`hyp.Pipeline`, `hyp.apply_model`, `hyp.normalize`'s fitted `Normalizer`,
`hyp.save`/`hyp.load`, and `text2mat` -- so that every DataFrame backend
datawrangler recognises works there too. Each test feeds a REAL polars
DataFrame (and a LazyFrame, and a list mixing polars with pandas/numpy) and
asserts the result is the SAME as for the equivalent pandas DataFrame.

The results are hypertools' internal types (pandas frames / numpy arrays):
polars in, pandas out, exactly as a pandas Series or a tuple of datasets is
normalised on the way in.

No mocks: real polars, real pandas, real hypertools calls (show=False).
"""
import os

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.manip import ZScore, Normalize, Smooth, Resample, Delay
from hypertools.tools.text2mat import text2mat

pl = pytest.importorskip("polars")

pytestmark = pytest.mark.filterwarnings(
    'ignore:.*(Missing data|DataFrame column|reordering|do not share columns'
    '|copy keyword).*')

N_ROWS, N_COLS = 40, 4
COLUMNS = ['a', 'b', 'c', 'd']


def _make_pandas(seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame(rng.rand(N_ROWS, N_COLS), columns=COLUMNS)


@pytest.fixture
def pdf():
    return _make_pandas(0)


@pytest.fixture
def pdf2():
    return _make_pandas(1)


@pytest.fixture
def plf(pdf):
    frame = pl.from_pandas(pdf)
    assert isinstance(frame, pl.DataFrame)
    return frame


@pytest.fixture
def plf2(pdf2):
    return pl.from_pandas(pdf2)


@pytest.fixture
def lazy(plf):
    frame = plf.lazy()
    assert isinstance(frame, pl.LazyFrame)
    return frame


def _same(a, b):
    """Recursively compare two hypertools results: same container shape,
    same DataFrame index/columns, all-close values (NaN == NaN)."""
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
    if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
        # polars in, pandas out: the SAME internal type as the pandas call
        assert isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame)
        assert list(a.columns) == list(b.columns)
        assert a.index.equals(b.index)
        np.testing.assert_allclose(a.to_numpy(dtype=float),
                                   b.to_numpy(dtype=float), equal_nan=True)
        return
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        assert isinstance(a, pd.Series) and isinstance(b, pd.Series)
        assert a.name == b.name and a.index.equals(b.index)
        np.testing.assert_allclose(a.to_numpy(dtype=float),
                                   b.to_numpy(dtype=float), equal_nan=True)
        return
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    np.testing.assert_allclose(a.astype(float), b.astype(float),
                               equal_nan=True)


# ---------------------------------------------------------------- hyp.manip

MANIPS = [('ZScore', {}), ('Normalize', {}), ('Smooth', {'kernel_width': 5}),
          ('Resample', {'n_samples': 20}), ('Delay', {'tau': 1, 'dims': 2})]


@pytest.mark.parametrize('model,kwargs', MANIPS, ids=[m for m, _ in MANIPS])
@pytest.mark.parametrize('kind', ['frame', 'lazy'])
def test_manip_polars_matches_pandas(pdf, plf, lazy, kind, model, kwargs):
    x = plf if kind == 'frame' else lazy
    out = hyp.manip(x, model=model, **kwargs)
    ref = hyp.manip(pdf, model=model, **kwargs)
    _same(out, ref)
    if model != 'Delay':      # Delay names its columns '<col>_lag<k>'
        assert list(out.columns) == COLUMNS


@pytest.mark.parametrize('model,kwargs', MANIPS, ids=[m for m, _ in MANIPS])
def test_manip_list_mixing_polars_pandas_numpy(pdf, pdf2, plf, model,
                                               kwargs):
    # a numpy array gets positional column labels; the shared-statistics
    # manipulators match columns by position when the labels differ, and
    # Smooth/Resample/Delay work per dataset, so EVERY model takes the mix
    # (Codex round 12: this test used to swap in an all-named list for
    # ZScore/Normalize)
    mixed = [plf, pdf2, pl.from_pandas(pdf2).lazy(), pdf2.to_numpy()]
    ref_in = [pdf, pdf2, pdf2, pdf2.to_numpy()]
    out = hyp.manip(mixed, model=model, **kwargs)
    _same(out, hyp.manip(ref_in, model=model, **kwargs))
    if model != 'Delay':
        assert [list(o.columns) for o in out] == [COLUMNS] * 3 + [[0, 1, 2, 3]]
    if model in ('ZScore', 'Normalize'):
        # the statistics are shared across all four datasets
        stacked = np.vstack([pdf.to_numpy()] + [pdf2.to_numpy()] * 3)
        if model == 'ZScore':
            expected = (pdf.to_numpy() - stacked.mean(axis=0)) / stacked.std(axis=0, ddof=1)
        else:
            lo, hi = stacked.min(axis=0), stacked.max(axis=0)
            expected = (pdf.to_numpy() - lo) / (hi - lo)
        assert np.allclose(out[0].to_numpy(), expected)


def test_manip_polars_series_is_one_column(pdf, plf):
    out = hyp.manip(plf['a'], model='ZScore')
    _same(out, hyp.manip(pdf['a'], model='ZScore'))
    assert list(out.columns) == ['a'] and out.shape == (N_ROWS, 1)
    # beside an array or a pandas Series in a list, too (Codex round 12,
    # R12-1: the polars frame from `.to_frame()` reached pandas-only code)
    mixed = hyp.manip([plf['a'], pdf['b'].to_numpy(), pdf['c']], model='Smooth',
                      kernel_width=5)
    ref = hyp.manip([pdf['a'], pdf['b'].to_numpy(), pdf['c']], model='Smooth',
                    kernel_width=5)
    _same(mixed, ref)
    assert [list(m.columns) for m in mixed] == [['a'], [0], ['c']]


def test_manip_polars_chain_and_stage_kwargs(pdf, plf):
    chain = [{'model': 'Resample', 'kwargs': {'n_samples': 15}}, 'ZScore']
    _same(hyp.manip(plf, model=chain), hyp.manip(pdf, model=chain))
    _same(hyp.manip(plf, model='ZScore', reduce='PCA', ndims=2),
          hyp.manip(pdf, model='ZScore', reduce='PCA', ndims=2))


def test_manip_polars_fitted_model_reuse(pdf, pdf2, plf, plf2):
    out, fitted = hyp.manip(plf, model='ZScore', return_model=True)
    ref, ref_fitted = hyp.manip(pdf, model='ZScore', return_model=True)
    _same(out, ref)
    _same(hyp.manip(plf2, model=fitted), hyp.manip(pdf2, model=ref_fitted))


def test_manip_empty_polars_frame_raises_no_observations():
    with pytest.raises(ValueError, match='no observations'):
        hyp.manip(pl.DataFrame({'a': []}), model='ZScore')


@pytest.mark.parametrize('cls,kwargs', [
    (ZScore, {}), (Normalize, {}), (Smooth, {'kernel_width': 5}),
    (Resample, {'n_samples': 20}), (Delay, {'tau': 1, 'dims': 2})],
    ids=['ZScore', 'Normalize', 'Smooth', 'Resample', 'Delay'])
def test_manipulator_classes_directly_on_polars(pdf, pdf2, plf, plf2, lazy,
                                                cls, kwargs):
    # the Manipulator classes are public (hyp.manip.MANIPULATORS) and are
    # used directly inside hyp.Pipeline: fit/transform never route through
    # hyp.manip's funnel, so they coerce their own input
    _same(cls(**kwargs).fit_transform(plf), cls(**kwargs).fit_transform(pdf))
    _same(cls(**kwargs).fit_transform(lazy), cls(**kwargs).fit_transform(pdf))
    _same(cls(**kwargs).fit_transform([plf, pdf2]),
          cls(**kwargs).fit_transform([pdf, pdf2]))
    _same(cls(**kwargs).fit(plf).transform(plf2),
          cls(**kwargs).fit(pdf).transform(pdf2))
    # a Series (either backend) is one column that keeps its name (Codex
    # round 12, R12-4: `as_dataframe` dropped a pandas Series' index/name)
    out = cls(**kwargs).fit_transform(plf['a'])
    ref = cls(**kwargs).fit_transform(pdf['a'])
    _same(out, ref)
    assert list(ref.columns) == (['a'] if cls is not Delay else ['a_lag1', 'a_lag0'])


@pytest.mark.parametrize('cls,kwargs', [
    (ZScore, {}), (Normalize, {}), (Resample, {'n_samples': 6})],
    ids=['ZScore', 'Normalize', 'Resample'])
def test_manipulator_classes_axis1_on_polars(pdf, plf, cls, kwargs):
    _same(cls(axis=1, **kwargs).fit_transform(plf),
          cls(axis=1, **kwargs).fit_transform(pdf))


# ---------------------------------------------------------------- hyp.align

@pytest.mark.parametrize('model', ['HyperAlign', 'Procrustes', None])
def test_align_list_mixing_polars_and_pandas(pdf, pdf2, plf, model):
    mixed = [plf, pdf2, pdf2.to_numpy()]
    ref = [pdf, pdf2, pdf2.to_numpy()]
    _same(hyp.align(mixed, model=model), hyp.align(ref, model=model))


def test_align_lazyframes_and_single_polars(pdf, pdf2, plf, plf2, lazy):
    _same(hyp.align([lazy, plf2]), hyp.align([pdf, pdf2]))
    _same(hyp.align(plf), hyp.align(pdf))


def test_align_polars_return_score_and_model(pdf, pdf2, plf, plf2):
    out, model, score = hyp.align([plf, pdf2], return_model=True,
                                  return_score=True)
    ref, ref_model, ref_score = hyp.align([pdf, pdf2], return_model=True,
                                          return_score=True)
    _same(out, ref)
    assert score['metric'] == ref_score['metric']
    np.testing.assert_allclose(score['before'], ref_score['before'])
    np.testing.assert_allclose(score['after'], ref_score['after'])
    # the fitted aligner is reusable on polars, like on pandas
    _same(hyp.align([plf2, plf], model=model),
          hyp.align([pdf2, pdf], model=ref_model))


# ---------------------------------------------------- hyp.stack / hyp.damage

def test_stack_polars_matches_pandas(pdf, pdf2, plf, plf2, lazy):
    _same(hyp.stack({'x': plf, 'y': plf2}), hyp.stack({'x': pdf, 'y': pdf2}))
    _same(hyp.stack([lazy, pdf2]), hyp.stack([pdf, pdf2]))
    _same(hyp.stack({'g': {'x': plf, 'y': pdf2}}, aggregate='mean'),
          hyp.stack({'g': {'x': pdf, 'y': pdf2}}, aggregate='mean'))
    # a polars Series is one single-column dataset named after the series
    _same(hyp.stack([plf['a'], plf2['a']]), hyp.stack([pdf['a'], pdf2['a']]))


def test_damage_polars_matches_pandas(pdf, pdf2, plf, lazy):
    _same(hyp.damage(plf, frac=0.2, seed=1, return_mask=True),
          hyp.damage(pdf, frac=0.2, seed=1, return_mask=True))
    _same(hyp.damage(lazy, frac=0.2, seed=1),
          hyp.damage(pdf, frac=0.2, seed=1))
    _same(hyp.damage([plf, pdf2.to_numpy()], frac=0.2, seed=1),
          hyp.damage([pdf, pdf2.to_numpy()], frac=0.2, seed=1))
    damaged = hyp.damage(plf, frac=0.2, seed=1)
    assert damaged.isna().to_numpy().sum() > 0
    # the caller's polars frame is untouched
    assert plf.null_count().to_numpy().sum() == 0


def test_damage_polars_series_matches_pandas(pdf, plf):
    _same(hyp.damage(plf['a'], frac=0.2, seed=1, return_mask=True),
          hyp.damage(pdf['a'], frac=0.2, seed=1, return_mask=True))


# ---------------------------------------- hyp.Pipeline / hyp.apply_model

def _pipeline():
    return hyp.Pipeline([('z', 'ZScore'),
                         ('pca', {'model': 'PCA',
                                  'kwargs': {'n_components': 2}})])


def test_pipeline_fit_transform_on_polars(pdf, pdf2, plf, plf2, lazy):
    pipe, ref = _pipeline(), _pipeline()
    _same(pipe.fit_transform(plf), ref.fit_transform(pdf))
    _same(pipe.transform(plf2), ref.transform(pdf2))
    _same(pipe.transform(lazy), ref.transform(pdf))
    _same(_pipeline().fit_transform(lazy), ref.fit_transform(pdf))


def test_pipeline_with_manipulator_instances_on_polars_list(pdf, pdf2, plf):
    pipe = hyp.Pipeline([('s', Smooth(kernel_width=5)), ('z', 'ZScore')])
    ref = hyp.Pipeline([('s', Smooth(kernel_width=5)), ('z', 'ZScore')])
    _same(pipe.fit_transform([plf, pdf2]), ref.fit_transform([pdf, pdf2]))


def test_apply_model_on_polars(pdf, pdf2, plf, plf2, lazy):
    _same(hyp.apply_model(plf, 'PCA', ndims=2),
          hyp.apply_model(pdf, 'PCA', ndims=2))
    _same(hyp.apply_model(lazy, 'PCA', ndims=2),
          hyp.apply_model(pdf, 'PCA', ndims=2))
    _same(hyp.apply_model([plf, pdf2], 'PCA', ndims=2),
          hyp.apply_model([pdf, pdf2], 'PCA', ndims=2))
    _same(hyp.apply_model(plf, 'PCA', ndims=2, format_data=False),
          hyp.apply_model(pdf, 'PCA', ndims=2, format_data=False))
    out, fitted = hyp.apply_model(plf, 'PCA', ndims=2, return_model=True)
    ref, ref_fitted = hyp.apply_model(pdf, 'PCA', ndims=2, return_model=True)
    _same(out, ref)
    _same(fitted.transform(plf2.to_numpy()),
          ref_fitted.transform(pdf2.to_numpy()))


def test_normalize_fitted_normalizer_on_polars(pdf, pdf2, plf, plf2, lazy):
    _same(hyp.normalize(plf), hyp.normalize(pdf))
    _, normalizer = hyp.normalize(pdf, return_model=True)
    _same(normalizer.transform(plf2), normalizer.transform(pdf2))
    _same(normalizer.transform(lazy), normalizer.transform(pdf))
    _same(normalizer.transform([plf, pdf2]), normalizer.transform([pdf, pdf2]))


# --------------------------------------------------------- hyp.save / load

@pytest.mark.parametrize('ext', ['csv', 'tsv', 'json', 'parquet', 'npy',
                                 'npz', 'mat', 'pkl'])
def test_save_load_round_trip_polars(tmp_path, pdf, plf, lazy, ext):
    target = tmp_path / f'polars.{ext}'
    ref_target = tmp_path / f'pandas.{ext}'
    hyp.save(plf, str(target))
    hyp.save(pdf, str(ref_target))
    assert target.exists() and os.path.getsize(target) > 0
    loaded = hyp.load(str(target))
    ref = hyp.load(str(ref_target))
    lazy_target = tmp_path / f'lazy.{ext}'
    hyp.save(lazy, str(lazy_target))
    if ext == 'pkl':
        # a pickle round-trips the object itself -- the LazyFrame included
        # (Codex round 12: an early return used to skip this branch)
        assert isinstance(loaded, pl.DataFrame)
        pd.testing.assert_frame_equal(loaded.to_pandas(), pdf)
        lazy_loaded = hyp.load(str(lazy_target))
        assert isinstance(lazy_loaded, pl.LazyFrame)
        pd.testing.assert_frame_equal(lazy_loaded.collect().to_pandas(), pdf)
        return
    _same(loaded, ref)
    if ext in ('csv', 'tsv', 'json', 'parquet'):
        assert list(loaded.columns) == COLUMNS
    # a LazyFrame is written like the frame it collects to
    _same(hyp.load(str(lazy_target)), ref)


def test_save_polars_csv_writes_no_index_column(tmp_path, plf):
    target = tmp_path / 'frame.csv'
    hyp.save(plf, str(target))
    with open(target) as f:
        header = f.readline().strip()
    assert header == ','.join(COLUMNS)


def test_load_passes_polars_through_and_analyzes_it(pdf, pdf2, plf, plf2,
                                                    lazy):
    assert hyp.load(plf) is plf
    assert hyp.load(lazy) is lazy
    assert [x is y for x, y in zip(hyp.load([plf, pdf2]), [plf, pdf2])] == \
        [True, True]
    _same(hyp.load([plf, plf2], reduce='PCA', ndims=2),
          hyp.load([pdf, pdf2], reduce='PCA', ndims=2))
    with pytest.raises(TypeError, match='unexpected keyword'):
        hyp.load(plf, n_samples=3)


# ------------------------------------------------------------ text2mat

def test_text2mat_polars_series_of_documents_is_one_dataset():
    docs = ['the cat sat on the mat', 'the dog sat on the log',
            'a bird sang']
    ref = text2mat(docs, vectorizer='CountVectorizer', semantic=None)
    out = text2mat(pl.Series('doc', docs), vectorizer='CountVectorizer',
                   semantic=None)
    pandas_out = text2mat(pd.Series(docs), vectorizer='CountVectorizer',
                          semantic=None)
    _same(out, ref)
    _same(pandas_out, ref)
    assert len(out) == 1 and out[0].shape[0] == len(docs)


def test_manip_aligns_an_unnamed_array_with_named_frames_by_position():
    """`manip([weights, df])` used to fail inside datawrangler ('All
    DataFrames must have the same columns'), pure pandas included, while
    `plot`/`reduce`/`align` accept that mix by position; now it does too."""
    import numpy as np
    import pandas as pd
    import polars as pl
    import hypertools as hyp
    arr = np.random.RandomState(0).rand(20, 3)
    pdf = pd.DataFrame(np.random.RandomState(1).rand(20, 3), columns=list('abc'))
    out = hyp.manip([arr, pdf, pl.DataFrame(pdf)], model='ZScore')
    ref = hyp.manip([arr, pdf.to_numpy(), pdf.to_numpy()], model='ZScore')
    assert len(out) == 3
    for got, want in zip(out, ref):
        assert np.allclose(np.asarray(got), np.asarray(want))
    # named frames are handed over untouched (see the mixed-list test below)


def test_manip_keeps_named_frames_and_indices_in_mixed_lists():
    """Codex round 11: the first mixed-list rule rejected two named frames
    with different labels (the independent manipulators never combine
    features) and rebuilt frames from their values, dropping a dated or
    irregular index so Resample interpolated at the wrong positions."""
    import numpy as np
    import pandas as pd
    import hypertools as hyp
    a = pd.DataFrame(np.random.RandomState(0).rand(30, 2), columns=['a', 'b'])
    b = pd.DataFrame(np.random.RandomState(1).rand(30, 2), columns=['x', 'y'])
    out = hyp.manip([a, b], model='Smooth')
    assert [list(o.columns) for o in out] == [['a', 'b'], ['x', 'y']]
    single = hyp.manip(a, model='Smooth')
    assert np.allclose(out[0].to_numpy(), single.to_numpy())
    # an irregular index survives beside an array, and the numbers match the
    # single-frame call exactly
    frame = pd.DataFrame({'v': [0.0, 1.0, 4.0, 9.0, 16.0]}, index=[0, 1, 2, 8, 10])
    arr = np.arange(5.0).reshape(-1, 1)
    mixed = hyp.manip([frame, arr], model='Resample', n_samples=7)
    alone = hyp.manip(frame, model='Resample', n_samples=7)
    assert np.allclose(mixed[0].to_numpy(), alone.to_numpy())
    assert list(mixed[0].index) == list(alone.index)
    dated = pd.DataFrame({'v': np.arange(30.0)}, index=pd.date_range('2024-01-01', periods=30))
    mixed = hyp.manip([dated, np.arange(30.0).reshape(-1, 1)], model='Smooth')
    assert list(mixed[0].index) == list(dated.index)
    # ...and its feature NAMES (Codex round 12, R12-3: the mixed-list rule
    # relabelled the named frame positionally for every model)
    assert list(mixed[0].columns) == ['v'] and list(mixed[1].columns) == [0]
    delayed = hyp.manip([dated, np.arange(30.0).reshape(-1, 1)], model='Delay')
    assert list(delayed[0].columns) == ['v_lag1', 'v_lag0']
    assert list(delayed[1].columns) == ['0_lag1', '0_lag0']
