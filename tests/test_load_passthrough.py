"""``hypertools.load`` passes already-loaded data through unchanged.

A tutorial author found that ``hyp.load(df)`` with a DataFrame in hand
raised ``TypeError``, so ``hyp.load`` could not serve as the uniform entry
point in a loop over mixed dataset names and in-memory data. These tests
make REAL calls (no mocks): in-memory passthrough (identity), post-load
kwargs honoured on passed-through data the same way as on loaded data, a
hosted built-in still downloading, and nonsense types still raising the
same ``TypeError`` as before.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.standard_normal((30, 4)), columns=list('abcd'))


@pytest.fixture
def arrays():
    rng = np.random.default_rng(1)
    return [rng.standard_normal((40, 6)), rng.standard_normal((40, 6))]


def test_dataframe_passes_through_by_identity(frame):
    out = hyp.load(frame)
    assert out is frame
    pd.testing.assert_frame_equal(out, frame)


def test_ndarray_passes_through_by_identity(arrays):
    arr = arrays[0]
    out = hyp.load(arr)
    assert out is arr
    np.testing.assert_array_equal(out, arr)


def test_list_of_arrays_passes_through_as_one_dataset(arrays):
    out = hyp.load(arrays)
    assert out is arrays
    for got, want in zip(out, arrays):
        assert got is want


def test_tuple_of_frames_and_arrays_passes_through(frame, arrays):
    data = (frame, arrays[0])
    out = hyp.load(data)
    assert out is data


def test_post_load_kwargs_apply_to_passed_through_data(arrays):
    """reduce/ndims/align on in-memory arrays == the same call on the
    same arrays after a real load: the list is analyzed as ONE
    multi-dataset (aligned across elements), not element-wise."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = hyp.load(arrays, reduce='PCA', ndims=3, align='HyperAlign')
        want = hyp.analyze(arrays, reduce='PCA', ndims=3, align='HyperAlign')
    assert isinstance(out, list) and len(out) == 2
    assert all(o.shape == (40, 3) for o in out)
    for got, expected in zip(out, want):
        np.testing.assert_allclose(got, expected)


def test_ndims_reduces_a_passed_through_dataframe(frame):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = hyp.load(frame, ndims=2)
    assert out.shape == (30, 2)


def test_mixed_list_resolves_names_and_keeps_in_memory_data(arrays):
    """The tutorial use case: one hyp.load loop over names and arrays."""
    arr = arrays[0]
    out = hyp.load([arr, 'spiral'])
    assert isinstance(out, list) and len(out) == 2
    assert out[0] is arr
    spiral = out[1]
    assert isinstance(spiral, list) and len(spiral) == 2
    assert all(isinstance(s, np.ndarray) and s.shape == (1000, 3)
               for s in spiral)


def test_hosted_dataset_name_still_loads():
    data = hyp.load('spiral')
    assert isinstance(data, list) and len(data) == 2
    assert all(isinstance(d, np.ndarray) and d.shape == (1000, 3)
               for d in data)


def test_fitted_model_name_still_returns_pipeline():
    from sklearn.pipeline import Pipeline
    assert isinstance(hyp.load('sotus_model'), Pipeline)


@pytest.mark.parametrize('bad', [{'a': [1, 2]}, 3, 2.5, object(), None])
def test_unsupported_types_still_raise_typeerror(bad):
    with pytest.raises(TypeError, match=r'hypertools\.load: dataset must be'):
        hyp.load(bad)


def test_list_mixing_in_an_unsupported_type_raises_typeerror(arrays):
    with pytest.raises(TypeError, match=r'got int$'):
        hyp.load([arrays[0], 3])


def test_typeerror_names_the_accepted_types():
    with pytest.raises(TypeError) as info:
        hyp.load({'a': 1})
    msg = str(info.value)
    assert 'DataFrame' in msg and 'numpy array' in msg and 'got dict' in msg
