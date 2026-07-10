"""Input coercion + error-quality hardening (QC 2026-07 release hunt).

Common, natural inputs used to crash or silently misbehave:
- a 1-D ndarray (data[0][0] on a scalar),
- a flat list of numbers (mapped get_type over each float -> "Unsupported"),
- a list of numeric lists ('list' has no attribute 'ndim'),
- a tuple / pandas Series,
- hue with the wrong length (silent truncation or cryptic IndexError),
- an unknown align model name ("'str' has no attribute fit_transform"),
- infinities / all-NaN data (opaque sklearn / PPCA errors).

Real data, no mocks; headless (Agg).
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.reduce.reduce import reduce as reducer
from hypertools.tools.normalize import normalize as normalizer


def _rng():
    return np.random.default_rng(0)


# --- input coercion: no crash on natural inputs ------------------------

def test_1d_ndarray_is_accepted():
    assert np.asarray(reducer(_rng().random(50))).shape == (50, 1)
    assert hyp.plot(_rng().random(50), show=False) is not None


def test_flat_list_of_numbers_is_one_dataset():
    out = reducer([1.0, 2.0, 3.0, 4.0, 5.0])
    assert not isinstance(out, list)
    assert np.asarray(out).shape == (5, 1)
    assert hyp.plot([1.0, 2.0, 3.0, 4.0, 5.0], show=False) is not None


def test_list_of_numeric_lists_does_not_crash():
    out = reducer([[1, 2, 3], [4, 5, 6]])
    assert isinstance(out, list) and len(out) == 2


def test_tuple_and_series_inputs_accepted():
    r = _rng()
    assert isinstance(reducer((r.random((30, 4)), r.random((30, 4)))), list)
    assert np.asarray(normalizer(pd.Series(r.random(40)))).shape == (40, 1)


def test_numpy_scalar_list_accepted():
    out = reducer([np.int64(1), np.int64(2), np.int64(3), np.int64(4)])
    assert np.asarray(out).shape == (4, 1)


# --- hue length validation ---------------------------------------------

@pytest.mark.parametrize('hue', [['a', 'b'], ['a'] * 8, list(range(3))])
def test_mismatched_hue_length_raises_clear_error(hue):
    x = _rng().normal(size=(6, 4))
    with pytest.raises(ValueError, match='hue has .* observations'):
        hyp.plot(x, 'o', hue=hue, show=False)


def test_matching_hue_length_still_works():
    x = _rng().normal(size=(6, 4))
    assert hyp.plot(x, 'o', hue=['a', 'b', 'c', 'a', 'b', 'c'], show=False) is not None
    assert hyp.plot(x, 'o', hue=np.linspace(0, 1, 6), show=False) is not None


@pytest.mark.parametrize('hue', ['red', 3])
def test_scalar_hue_broadcasts_to_one_group(hue):
    """A single string/number hue means 'one group for all observations'; it
    must not be mis-measured as len('red')==3 characters (red-team of 7d71975b)."""
    x = _rng().normal(size=(20, 3))
    assert hyp.plot(x, 'o', hue=hue, show=False) is not None


def test_zero_dim_ndarray_is_accepted():
    """A 0-d array (np.array(5)) used to raise an opaque 'tuple index out of
    range'; it is now one observation with one feature (red-team of 7d71975b)."""
    assert hyp.plot(np.array(5.0), show=False) is not None


# --- error quality -----------------------------------------------------

def test_unknown_align_model_clear_error():
    x = _rng().normal(size=(20, 4))
    with pytest.raises(ValueError, match='unknown align model'):
        hyp.align([x, x + 1], model='NotAnAligner')


@pytest.mark.parametrize('spec', [{'model': 'NotAnAligner'},
                                  {'model': 'NotAnAligner', 'kwargs': {}}])
def test_unknown_align_model_dict_form_clear_error(spec):
    """The dict spec form used to slip past the bare-string guard and hit the
    cryptic 'str object has no attribute fit_transform' (red-team of 7d71975b)."""
    x = _rng().normal(size=(20, 4))
    with pytest.raises(ValueError, match='unknown align model'):
        hyp.align([x, x + 1], model=spec)


def test_infinite_values_clear_error():
    x = _rng().normal(size=(20, 4))
    x[0, 0] = np.inf
    with pytest.raises(ValueError, match='infinite'):
        hyp.plot(x, show=False)


def test_all_nan_input_clear_error():
    with pytest.raises(ValueError, match='entirely missing'):
        reducer(np.full((20, 4), np.nan))


def test_scattered_nan_still_imputes():
    x = _rng().normal(size=(30, 4))
    x[_rng().random((30, 4)) < 0.1] = np.nan
    assert np.asarray(reducer(x, reduce='PCA', ndims=2)).shape == (30, 2)
