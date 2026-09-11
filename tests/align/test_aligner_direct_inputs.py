"""The Aligner classes called directly, on the inputs every other hypertools
entry point accepts (1.1 release review, 2026-09-11).

`Aligner.fit` documents ``data : DataFrame, array, or list of these`` and the
1.1 CHANGELOG shows ``HyperAlign().fit(xs).transform(ys)``, but `fit` handed
its input straight to ``datawrangler.unstack``, which only understands
DataFrames: a list of numpy arrays -- the most common hypertools input --
raised a bare ``Exception: Unsupported datatype: <class 'list'>``, and so did
a single array. The classes now coerce their input the way ``hyp.align``
does, and hand results back in the input's own form: arrays for arrays,
DataFrames for DataFrames, a list for a list and a single dataset for a
single dataset.

Also here: the alignment warnings name the CALLER's line (not
``hypertools/align/common.py``), like the library's other warnings.

All data is real (small) numeric arrays -- no mocks.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.align import (HyperAlign, NullAlign, Procrustes,
                              SharedResponseModel)
from hypertools.align.srm import (DeterministicSharedResponseModel,
                                  RobustSharedResponseModel)
from hypertools.align.score import alignment_score
from hypertools.core.pipeline import Pipeline


def _rotated_copies(n=3, n_obs=40, n_features=3, noise=0.05, seed=0):
    """`n` rotated, noisy copies of one smooth trajectory (numpy arrays)."""
    t = np.linspace(0, 4 * np.pi, n_obs)
    base = np.stack([np.sin(t), np.cos(t), t / 10, np.sin(2 * t)],
                    axis=1)[:, :n_features]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        rot, _ = np.linalg.qr(rng.standard_normal((n_features, n_features)))
        out.append((base + noise * rng.standard_normal(base.shape)) @ rot)
    return out


ALIGNER_FACTORIES = {
    'HyperAlign': lambda: HyperAlign(n_iter=5),
    'Procrustes': lambda: Procrustes(),
    'SharedResponseModel': lambda: SharedResponseModel(features=3),
    'DeterministicSharedResponseModel':
        lambda: DeterministicSharedResponseModel(features=3),
    'RobustSharedResponseModel': lambda: RobustSharedResponseModel(features=3),
    'NullAlign': lambda: NullAlign(),
}


@pytest.mark.parametrize('name', sorted(ALIGNER_FACTORIES))
def test_fit_accepts_a_list_of_arrays_and_returns_arrays(name):
    xs = _rotated_copies()
    model = ALIGNER_FACTORIES[name]()
    assert model.fit(xs) is model
    out = model.transform(xs)
    assert isinstance(out, list) and len(out) == len(xs)
    for a, x in zip(out, xs):
        assert type(a) is np.ndarray
        assert a.shape[0] == x.shape[0]

    # the same numbers as the DataFrame route that always worked
    frames = [pd.DataFrame(x) for x in xs]
    reference = ALIGNER_FACTORIES[name]().fit(frames).transform(frames)
    for a, r in zip(out, reference):
        assert isinstance(r, pd.DataFrame)
        np.testing.assert_allclose(a, r.to_numpy())


@pytest.mark.parametrize('name', sorted(ALIGNER_FACTORIES))
def test_fit_transform_accepts_a_list_of_arrays(name):
    xs = _rotated_copies()
    out = ALIGNER_FACTORIES[name]().fit_transform(xs)
    frames = [pd.DataFrame(x) for x in xs]
    reference = ALIGNER_FACTORIES[name]().fit_transform(frames)
    assert isinstance(out, list) and len(out) == len(xs)
    for a, r in zip(out, reference):
        assert type(a) is np.ndarray
        np.testing.assert_allclose(a, r.to_numpy())


def test_changelog_chain_fit_on_arrays_transform_held_out_arrays():
    """The CHANGELOG's ``HyperAlign().fit(xs).transform(ys)``, with the
    usual array input: held-out data is projected into the fitted space
    and agrees measurably better than before alignment."""
    data = _rotated_copies(n=3, n_obs=80)
    xs = [d[:40] for d in data]
    ys = [d[40:] for d in data]
    aligned = HyperAlign(n_iter=10).fit(xs).transform(ys)
    assert all(type(a) is np.ndarray and a.shape == (40, 3) for a in aligned)
    score = alignment_score(ys, aligned=aligned)
    assert score['after'] < score['before'] - 0.2


@pytest.mark.parametrize('cls', [HyperAlign, Procrustes, NullAlign])
def test_a_single_array_is_one_dataset_and_comes_back_as_one_array(cls):
    x = _rotated_copies(n=1)[0]
    model = cls()
    assert model.fit(x) is model
    out = model.transform(x)
    assert type(out) is np.ndarray and out.shape == x.shape
    replay = cls().fit_transform(x)
    assert type(replay) is np.ndarray and replay.shape == x.shape


def test_a_single_dataframe_comes_back_as_one_dataframe():
    """`transform`'s docstring promises the input's list/single-item shape;
    a single DataFrame used to come back as a list of one."""
    frame = pd.DataFrame(_rotated_copies(n=1)[0],
                         index=pd.RangeIndex(100, 140))
    out = NullAlign().fit_transform(frame)
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == list(frame.index)
    np.testing.assert_allclose(out.to_numpy(), frame.to_numpy())


def test_a_mixed_list_returns_each_dataset_in_its_own_form():
    a, b = _rotated_copies(n=2)
    frame = pd.DataFrame(b, index=pd.RangeIndex(0, 40))
    out = HyperAlign(n_iter=5).fit_transform([a, frame])
    assert type(out[0]) is np.ndarray
    assert isinstance(out[1], pd.DataFrame)
    assert list(out[1].index) == list(frame.index)


def test_a_tuple_of_arrays_is_a_list_of_datasets():
    xs = _rotated_copies()
    out = Procrustes().fit_transform(tuple(xs))
    reference = Procrustes().fit_transform(list(xs))
    assert isinstance(out, list) and len(out) == 3
    for a, r in zip(out, reference):
        np.testing.assert_allclose(a, r)


def test_transform_replay_keeps_the_fit_time_input_form():
    xs = _rotated_copies()
    model = HyperAlign(n_iter=5).fit(xs)
    replay = model.transform()
    assert isinstance(replay, list)
    assert all(type(a) is np.ndarray for a in replay)
    np.testing.assert_allclose(replay[1], model.transform(xs)[1])


def test_held_out_shape_is_still_validated_for_arrays():
    xs = _rotated_copies()
    model = HyperAlign(n_iter=3).fit(xs)
    with pytest.raises(ValueError, match=r'3 dataset'):
        model.transform(xs[:2])
    with pytest.raises(ValueError, match=r'column'):
        model.transform([xs[0], xs[1], np.hstack([xs[2], xs[2]])])


def test_polars_frames_align_like_pandas_frames():
    pl = pytest.importorskip('polars')
    xs = _rotated_copies()
    out = HyperAlign(n_iter=5).fit_transform(
        [pl.DataFrame(x, schema=['a', 'b', 'c']) for x in xs])
    reference = HyperAlign(n_iter=5).fit_transform(
        [pd.DataFrame(x) for x in xs])
    assert isinstance(out, list) and len(out) == 3
    for a, r in zip(out, reference):
        assert isinstance(a, pd.DataFrame)
        np.testing.assert_allclose(a.to_numpy(), r.to_numpy())


def test_dispatcher_results_are_unchanged_by_the_direct_route():
    """`hyp.align` wraps the same classes: its (array) output equals the
    direct class call on arrays."""
    xs = _rotated_copies()
    via_dispatcher = hyp.align(xs, model=HyperAlign(n_iter=5))
    direct = HyperAlign(n_iter=5).fit_transform(xs)
    for a, b in zip(via_dispatcher, direct):
        np.testing.assert_allclose(a, b)
    single = hyp.align(xs[0], model='NullAlign')
    assert type(single) is np.ndarray and single.shape == xs[0].shape


def test_a_raw_aligner_pipeline_step_takes_a_list_of_arrays():
    xs = _rotated_copies()
    out = Pipeline(['HyperAlign']).fit_transform(xs)
    reference = HyperAlign().fit_transform(xs)
    assert isinstance(out, list) and len(out) == 3
    for a, r in zip(out, reference):
        assert type(a) is np.ndarray
        np.testing.assert_allclose(a, r)


# --- warnings name the caller's line (finding 4) ---------------------------

def test_the_row_trim_warning_names_the_callers_line_via_hyp_align():
    a, b = _rotated_copies(n=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        hyp.align([a, b[:25]], model='HyperAlign')
    trims = [w for w in caught if 'common to all datasets' in str(w.message)]
    assert len(trims) == 1
    assert trims[0].filename == __file__, trims[0].filename


def test_the_row_trim_warning_names_the_callers_line_via_the_class():
    a, b = _rotated_copies(n=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        HyperAlign().fit_transform([a, b[:25]])
    trims = [w for w in caught if 'common to all datasets' in str(w.message)]
    assert len(trims) == 1
    assert trims[0].filename == __file__, trims[0].filename


def test_the_duplicate_index_warning_names_the_callers_line():
    a, b = _rotated_copies(n=2)
    dup = pd.DataFrame(a, index=[0, 0] + list(range(2, 40)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        hyp.align([dup, pd.DataFrame(b)], model='NullAlign')
    dups = [w for w in caught if 'duplicated row-index' in str(w.message)]
    assert len(dups) == 1
    assert dups[0].filename == __file__, dups[0].filename
