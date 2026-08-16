"""`build_hierarchy_traces` is the ONE owner of final-trace construction.

Before 1.1, `build_multiindex_styles` (multiindex.py:197-229) both appended
the per-level means and styled them. Any second builder would append every
mean twice -- `test_every_expected_mean_appears_exactly_once` is the
regression test for exactly that.
"""
import matplotlib
matplotlib.use("Agg")

import warnings

import numpy as np
import pandas as pd
import pytest

from hypertools.plot.hierarchy import FinalTraces, build_hierarchy_traces
from hypertools.plot.multiindex import build_multiindex_styles, expand_multiindex

COL_META = {'n_levels': 2, 'axis': 'columns', 'level_names': ['Market', 'Sector'],
            'leaf_keys': [('M', 'Tech'), ('M', 'Fin'), ('M', 'Energy')]}
ROW_META = {'n_levels': 3, 'axis': 'rows', 'level_names': ['grp', 'cond', 'subj'],
            'leaf_keys': [('X', 'A', 'S0'), ('X', 'A', 'S1'),
                          ('X', 'B', 'S0'), ('X', 'B', 'S1'),
                          ('Y', 'A', 'S0'), ('Y', 'A', 'S1'),
                          ('Y', 'B', 'S0'), ('Y', 'B', 'S1')]}


def _leaves(n, rows=5, cols=2, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, cols)) for _ in range(n)]


def test_leaves_come_first_then_means_shallowest_last():
    ft = build_hierarchy_traces(_leaves(3), COL_META)
    assert isinstance(ft, FinalTraces)
    assert ft.is_mean == [False, False, False, True]
    assert ft.level_idx == [1, 1, 1, 0]
    assert ft.keys[-1] == ('M',)


def test_every_expected_mean_appears_exactly_once():
    """THE F1 regression test. A 3-level tree has 4 (grp, cond) means and 2
    grp means; each must be built once and only once."""
    ft = build_hierarchy_traces(_leaves(8), ROW_META)
    assert len(ft.arrays) == 8 + 4 + 2
    mean_keys = [k for k, m in zip(ft.keys, ft.is_mean) if m]
    assert mean_keys == [('X', 'A'), ('X', 'B'), ('Y', 'A'), ('Y', 'B'),
                         ('X',), ('Y',)]
    assert len(mean_keys) == len(set(mean_keys))


def test_two_level_hierarchy_builds_no_mean():
    """n_levels == 1: one leaf per group, and NO aggregate level exists."""
    meta = {'n_levels': 1, 'axis': 'columns', 'level_names': ['Group'],
            'leaf_keys': [('A',), ('B',), ('C',)]}
    ft = build_hierarchy_traces(_leaves(3), meta)
    assert len(ft.arrays) == 3
    assert ft.is_mean == [False, False, False]
    assert ft.level_idx == [0, 0, 0]


def test_means_equal_numpy_mean_of_their_members():
    leaves = _leaves(3)
    ft = build_hierarchy_traces(leaves, COL_META)
    assert np.array_equal(ft.arrays[3], np.mean(np.stack(leaves), axis=0))


def test_unequal_length_members_are_truncated_to_the_overlap():
    leaves = _leaves(3)
    leaves[2] = leaves[2][:3]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ft = build_hierarchy_traces(leaves, COL_META)
    assert ft.arrays[3].shape == (3, 2)
    expected = np.mean(np.stack([leaf[:3] for leaf in leaves]), axis=0)
    assert np.array_equal(ft.arrays[3], expected)


def test_unequal_length_warning_is_emitted_exactly_once():
    """One underlying issue, one warning -- the dedup multiindex.py:189-196
    already documents, now owned here."""
    leaves = _leaves(8)
    leaves[0] = leaves[0][:3]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        build_hierarchy_traces(leaves, ROW_META)
    unequal = [w for w in caught if 'unequal-length' in str(w.message)]
    assert len(unequal) == 1


def test_aux_arrays_are_co_truncated_with_the_data():
    """Contract 6: hue must never drift out of step with its trace."""
    leaves = _leaves(3)
    leaves[2] = leaves[2][:3]
    aux = [np.arange(5.0), np.arange(5.0), np.arange(3.0)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ft = build_hierarchy_traces(leaves, COL_META, aux=aux)
    for arr, a in zip(ft.arrays, ft.aux):
        assert len(arr) == len(a)
    assert len(ft.aux[3]) == 3


def test_mean_aux_is_the_mean_of_its_members_aux():
    aux = [np.array([1.0, 2, 3, 4, 5]), np.array([3.0, 4, 5, 6, 7]),
           np.array([5.0, 6, 7, 8, 9])]
    ft = build_hierarchy_traces(_leaves(3), COL_META, aux=aux)
    assert np.allclose(ft.aux[3], np.mean(np.stack(aux), axis=0))


def test_aux_is_none_when_no_aux_is_supplied():
    ft = build_hierarchy_traces(_leaves(3), COL_META)
    assert ft.aux is None


@pytest.mark.parametrize('n_aux', [1, 2, 4], ids=['too_few', 'one_short',
                                                  'too_many'])
def test_too_few_or_too_many_aux_arrays_raise_the_named_error(n_aux):
    """Contract 6 promises `assert_consistent`'s NAMED error for aux, and
    the count has to be checked before the mean loop reads it. Measured
    before this check: 1 aux for 3 leaves raised a bare `IndexError: list
    index out of range` from the mean loop, and 4 raised nothing at all --
    it returned a `FinalTraces` whose `aux` outnumbered its `arrays`."""
    with pytest.raises(ValueError, match='leaf/aux mismatch'):
        build_hierarchy_traces(_leaves(3), COL_META,
                               aux=[np.arange(5.0)] * n_aux)


def test_aux_count_error_reports_both_counts():
    with pytest.raises(ValueError, match=r'3 leaf array\(s\) but 1 aux'):
        build_hierarchy_traces(_leaves(3), COL_META, aux=[np.arange(5.0)])


def test_a_returned_FinalTraces_is_self_consistent_on_aux():
    """The `assert_consistent(aux=...)` call site Contract 6 names: every
    trace, leaves AND derived means, has exactly one aux entry."""
    ft = build_hierarchy_traces(_leaves(3), COL_META,
                                aux=[np.arange(5.0)] * 3)
    assert len(ft.aux) == len(ft.arrays) == 4
    ft.assert_consistent(aux=ft.aux)


def test_the_unequal_length_warning_blames_the_CALLER():
    """Every sibling hierarchy warning uses `external_stacklevel()`; this
    one did not, so it was attributed to `plot/hierarchy.py`'s own
    `warnings.warn` line, which tells a caller nothing about which of their
    frames is ragged. Same attribution idiom as
    `tests/test_h1_validation_warnings.py:246`."""
    leaves = _leaves(3)
    leaves[2] = leaves[2][:3]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        build_hierarchy_traces(leaves, COL_META)
    unequal = [w for w in caught if 'unequal-length' in str(w.message)]
    assert len(unequal) == 1
    assert unequal[0].filename == __file__, \
        f'blamed {unequal[0].filename}, not the caller'


def test_assert_consistent_names_the_offending_sequence():
    ft = build_hierarchy_traces(_leaves(3), COL_META)
    with pytest.raises(ValueError, match='forecasts'):
        ft.assert_consistent(forecasts=[None])


def test_assert_consistent_reports_both_lengths():
    ft = build_hierarchy_traces(_leaves(3), COL_META)
    with pytest.raises(ValueError, match='4.*1|1.*4'):
        ft.assert_consistent(forecasts=[None])


def test_assert_consistent_passes_on_matching_lengths():
    ft = build_hierarchy_traces(_leaves(3), COL_META)
    ft.assert_consistent(forecasts=[None] * 4, hues=[None] * 4)


def test_the_legacy_wrapper_appends_each_mean_exactly_once():
    """`build_multiindex_styles` keeps its (arrays, style) contract, but now
    delegates: no second mean-construction path exists."""
    leaves = _leaves(8)
    arrays, style = build_multiindex_styles(leaves, ROW_META)
    assert len(arrays) == 8 + 4 + 2 == len(style['linewidths'])
    assert style['linewidths'] == [1.0] * 8 + [2.0] * 4 + [3.0] * 2


def test_final_trace_arrays_are_plain_ndarrays_even_for_row_dataframe_leaves():
    """Contract 11 must hold BY CONSTRUCTION, not by discipline.

    `expand_multiindex` hands back real `DataFrame`s whose INDEX is still the
    full row MultiIndex, and those are a measured fixed point (D2): re-expanding
    one returns exactly itself. If such a leaf survived into `ft.arrays` it would
    reach `hyp.predict` in Task 8, be re-detected as hierarchical, and regroup
    forever. `build_hierarchy_traces` therefore coerces with `np.asarray`, which
    drops the index and column labels so there is nothing left to re-detect.

    This test exists because an earlier revision stated the ndarray contract in
    prose only -- the implementation still said `arrays = list(leaf_arrays)`,
    which preserves DataFrames, and nothing pinned it.
    """
    idx = pd.MultiIndex.from_tuples(
        [(c, s) for c in ('condA', 'condB') for s in ('S0', 'S1')
         for _ in range(5)], names=['cond', 'subj'])
    frame = pd.DataFrame(np.random.default_rng(0).normal(size=(20, 3)),
                         index=idx, columns=['x', 'y', 'z'])

    leaves, meta = expand_multiindex(frame)
    # the premise: the leaves really are hierarchy-carrying DataFrames
    assert all(isinstance(leaf, pd.DataFrame) for leaf in leaves)
    assert all(isinstance(leaf.index, pd.MultiIndex) for leaf in leaves)

    ft = build_hierarchy_traces(leaves, meta)

    # every trace -- leaves AND derived means -- is a plain ndarray
    assert all(isinstance(arr, np.ndarray) for arr in ft.arrays)
    assert not any(isinstance(arr, pd.DataFrame) for arr in ft.arrays)
    assert all(not isinstance(arr, np.ndarray) or arr.ndim == 2
               for arr in ft.arrays)
    # the means specifically (they are built here, not passed in)
    assert all(isinstance(ft.arrays[i], np.ndarray)
               for i, m in enumerate(ft.is_mean) if m)

    # and the caller's frames are untouched
    assert isinstance(frame.index, pd.MultiIndex)
    assert frame.columns.tolist() == ['x', 'y', 'z']
    assert all(isinstance(leaf, pd.DataFrame) for leaf in leaves)
    assert all(isinstance(leaf.index, pd.MultiIndex) for leaf in leaves)
