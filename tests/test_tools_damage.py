#!/usr/bin/env python
"""Tests for `hypertools.tools.damage` (GH #285).

`damage` replaces four hand-written knock-outs: the seeded flat-index
`rng.choice` in `docs/tutorials/plot.ipynb` (cell 40), the
``rng.random(shape) < 0.05`` mask in `docs/tutorials/pipelines.ipynb`
(cell 17), and the occlusion band plus scattered NaNs -- with its
read-only/Fortran-order write-through workaround -- in
`docs/tutorials/projectile_kalman.ipynb` (cell 8).
"""

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.tools import damage


def _walk(rows=60, cols=5, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal((rows, cols)), axis=0)


def test_scattered_fraction_and_untouched_original():
    full = _walk()
    holey = damage(full, frac=0.1, seed=0)
    assert holey.shape == full.shape
    assert int(np.isnan(holey).sum()) == round(0.1 * full.size)
    assert not np.isnan(full).any()
    # observed cells are passed through unchanged
    observed = ~np.isnan(holey)
    assert np.array_equal(holey[observed], full[observed])


def test_seed_is_reproducible_and_different_seeds_differ():
    full = _walk()
    a = damage(full, frac=0.1, seed=42)
    b = damage(full, frac=0.1, seed=42)
    c = damage(full, frac=0.1, seed=43)
    assert np.array_equal(a, b, equal_nan=True)
    assert not np.array_equal(np.isnan(a), np.isnan(c))
    # an explicit Generator works too
    d = damage(full, frac=0.1, seed=np.random.default_rng(42))
    assert np.array_equal(a, d, equal_nan=True)


def test_frac_zero_damages_nothing():
    full = _walk()
    assert np.array_equal(damage(full, frac=0), full)
    assert not damage(full, frac=0, return_mask=True)[1].any()


def test_scattered_damage_never_empties_a_row():
    full = _walk(rows=40, cols=4)
    for frac in (0.5, 0.9, 1.0):
        holey = damage(full, frac=frac, seed=1)
        assert not np.isnan(holey).all(axis=1).any()
    # ... and it takes as many cells as that guarantee allows
    assert int(np.isnan(damage(full, frac=1.0, seed=1)).sum()) == 40 * 3


def test_single_feature_damages_cells_directly():
    column = _walk(rows=20, cols=1)
    assert int(np.isnan(damage(column, frac=0.5, seed=0)).sum()) == 10
    flat = _walk(rows=20, cols=1).ravel()
    damaged = damage(flat, frac=0.5, seed=0)
    assert damaged.shape == flat.shape
    assert int(np.isnan(damaged).sum()) == 10


def test_rows_blanks_whole_rows():
    full = _walk(rows=20, cols=4)
    gapped = damage(full, frac=0, rows=slice(5, 10))
    blank = np.flatnonzero(np.isnan(gapped).all(axis=1))
    assert blank.tolist() == [5, 6, 7, 8, 9]
    assert int(np.isnan(gapped).sum()) == 5 * 4

    assert np.flatnonzero(np.isnan(
        damage(full, frac=0, rows=[0, -1])).all(axis=1)).tolist() == [0, 19]

    mask = np.zeros(20, dtype=bool)
    mask[[2, 3]] = True
    assert np.flatnonzero(np.isnan(
        damage(full, frac=0, rows=mask)).all(axis=1)).tolist() == [2, 3]

    # an int is a COUNT of randomly chosen rows
    counted = damage(full, frac=0, rows=3, seed=0)
    assert int(np.isnan(counted).all(axis=1).sum()) == 3
    assert np.array_equal(counted, damage(full, frac=0, rows=3, seed=0),
                          equal_nan=True)


def test_row_frac():
    full = _walk(rows=20, cols=4)
    assert int(np.isnan(damage(full, frac=0, row_frac=0.25, seed=0)
                        ).all(axis=1).sum()) == 5


def test_rows_and_scattered_damage_together():
    # the projectile tutorial's shape: an occlusion band plus scattered NaNs
    full = _walk(rows=40, cols=3)
    damaged, mask = damage(full, frac=0.1, rows=slice(15, 20), seed=42,
                           return_mask=True)
    assert int(np.isnan(damaged).all(axis=1).sum()) == 5
    intact = 40 * 3 - 5 * 3
    assert int(mask.sum()) == 5 * 3 + round(0.1 * intact)
    assert np.array_equal(mask, np.isnan(damaged))


def test_rows_out_of_bounds_raises():
    full = _walk(rows=10, cols=3)
    with pytest.raises(IndexError):
        damage(full, frac=0, rows=[10])
    with pytest.raises(IndexError):
        damage(full, frac=0, rows=[-11])
    with pytest.raises(IndexError):
        damage(full, frac=0, rows=99)
    with pytest.raises(IndexError):
        damage(full, frac=0, rows=np.ones(3, dtype=bool))


def test_invalid_arguments():
    full = _walk(rows=10, cols=3)
    with pytest.raises(ValueError, match='frac='):
        damage(full, frac=1.5)
    with pytest.raises(ValueError, match='row_frac='):
        damage(full, row_frac=2.0)
    with pytest.raises(ValueError, match='not both'):
        damage(full, rows=[1], row_frac=0.1)
    with pytest.raises(TypeError, match='numeric'):
        damage(np.array([['a', 'b'], ['c', 'd']]))
    with pytest.raises(ValueError, match='dimension'):
        damage(np.zeros((2, 3, 4)))


def test_dataframe_keeps_index_and_columns_and_becomes_float():
    index = pd.date_range('2020-01-01', periods=12)
    frame = pd.DataFrame(np.arange(36).reshape(12, 3), index=index,
                         columns=['x', 'y', 'z'])
    damaged, mask = damage(frame, frac=0.2, seed=0, return_mask=True)

    assert isinstance(damaged, pd.DataFrame)
    assert damaged.index.equals(index)
    assert list(damaged.columns) == ['x', 'y', 'z']
    assert set(damaged.dtypes) == {np.dtype(float)}
    assert isinstance(mask, pd.DataFrame)
    assert mask.index.equals(index) and list(mask.columns) == ['x', 'y', 'z']
    assert np.array_equal(mask.to_numpy(), damaged.isna().to_numpy())
    # the caller's frame is untouched, dtype included
    assert not frame.isna().to_numpy().any()
    assert set(frame.dtypes) == {np.dtype(int)}


def test_series_input():
    series = pd.Series(np.arange(20.0), name='height')
    damaged = damage(series, frac=0.25, seed=0)
    assert isinstance(damaged, pd.Series)
    assert damaged.name == 'height'
    assert int(damaged.isna().sum()) == 5
    assert not series.isna().any()


def test_read_only_fortran_ordered_input_is_written_through():
    """The pandas trap the projectile tutorial wrote out by hand."""
    values = np.asfortranarray(np.arange(60.0).reshape(15, 4))
    frame = pd.DataFrame(values, columns=list('abcd'))
    raw = frame.to_numpy()
    # the conditions that make a naive .ravel() write vanish
    assert raw.flags.f_contiguous and not raw.flags.writeable

    damaged = damage(frame, frac=0.2, seed=0)
    assert int(damaged.isna().to_numpy().sum()) == round(0.2 * 60)
    assert not frame.isna().to_numpy().any()

    read_only = np.arange(60.0).reshape(15, 4, order='F')
    read_only.setflags(write=False)
    assert int(np.isnan(damage(read_only, frac=0.2, seed=0)).sum()) == 12
    assert np.isfinite(read_only).all()


def test_already_missing_cells_are_not_damaged_twice():
    full = _walk(rows=20, cols=4)
    full[0, 0] = np.nan
    damaged, mask = damage(full, frac=0.25, seed=0, return_mask=True)
    assert not mask[0, 0]
    intact = full.size - 1
    assert int(mask.sum()) == round(0.25 * intact)
    assert int(np.isnan(damaged).sum()) == round(0.25 * intact) + 1


def test_list_input_draws_independently_and_reproducibly():
    datasets = [_walk(rows=30, cols=4, seed=0), _walk(rows=30, cols=4, seed=1)]
    first = damage(datasets, frac=0.2, seed=7)
    second = damage(datasets, frac=0.2, seed=7)

    assert isinstance(first, list) and len(first) == 2
    assert all(np.array_equal(a, b, equal_nan=True)
               for a, b in zip(first, second))
    # independent draws: the two masks differ
    assert not np.array_equal(np.isnan(first[0]), np.isnan(first[1]))
    assert all(not np.isnan(d).any() for d in datasets)

    damaged, masks = damage(datasets, frac=0.2, seed=7, return_mask=True)
    assert len(masks) == 2
    assert all(np.array_equal(m, np.isnan(d))
               for m, d in zip(masks, damaged))


def test_mixed_list_of_arrays_and_frames():
    frame = pd.DataFrame(np.arange(40.0).reshape(10, 4), columns=list('abcd'))
    out = damage([np.arange(40.0).reshape(10, 4), frame], frac=0.25, seed=0)
    assert isinstance(out[0], np.ndarray) and isinstance(out[1], pd.DataFrame)
    assert list(out[1].columns) == ['a', 'b', 'c', 'd']


def test_damaged_data_round_trips_through_impute():
    full = _walk(rows=120, cols=6)
    damaged = damage(full, frac=0.05, seed=0)
    assert np.isnan(damaged).any()

    filled = np.asarray(hyp.impute(damaged))
    assert not np.isnan(filled).any()
    observed = ~np.isnan(damaged)
    assert np.allclose(filled[observed], damaged[observed])
