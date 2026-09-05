#!/usr/bin/env python
"""Tests for `hypertools.tools.stack` (GH #285).

`stack` is the ``pd.concat({(group, member): df}, axis=1)`` +
``columns.names = [...]`` incantation from `docs/tutorials/hierarchy.ipynb`
(cell 5), plus the group means that `examples/animate_market_sectors.py`
(``np.mean(aligned, axis=0)``), `examples/plot_align.py` and
`examples/save_movie.py` compute by hand. The first test pins that the frame
it builds is the tutorial's frame, cell for cell.
"""

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.core.hierarchy import group_columns
from hypertools.tools import stack

FEATURES = [f'feature {j + 1}' for j in range(6)]


def _subjects(n=3, rows=40, cols=6, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((rows, cols)) for _ in range(n)]


def _tutorial_frame(subjects):
    """docs/tutorials/hierarchy.ipynb, cell 5 -- the hand-built version."""
    frame = pd.concat(
        {('listeners', f'subject {i + 1}'):
         pd.DataFrame(data, columns=[f'feature {j + 1}'
                                     for j in range(data.shape[1])])
         for i, data in enumerate(subjects)}, axis=1)
    frame.columns.names = ['Group', 'Subject', 'Feature']
    return frame


def _group(frame, key):
    """The columns under one hierarchy key, as an array."""
    columns = [c for c in frame.columns if c[:len(key)] == key]
    return frame.loc[:, columns].to_numpy()


def test_matches_the_hierarchy_tutorials_hand_built_frame():
    subjects = _subjects()
    built = stack({'listeners': subjects},
                  names=[f'subject {i + 1}' for i in range(3)],
                  level_names=['Group', 'Subject', 'Feature'])
    hand = _tutorial_frame(subjects)

    assert list(built.columns) == list(hand.columns)
    assert list(built.columns.names) == ['Group', 'Subject', 'Feature']
    assert built.equals(hand)
    # the nested-dict spelling gives the identical frame
    nested = stack({'listeners': {f'subject {i + 1}': data
                                  for i, data in enumerate(subjects)}},
                   level_names=['Group', 'Subject', 'Feature'])
    assert nested.equals(hand)


def test_the_built_frame_plots_as_the_tutorial_does():
    subjects = _subjects()
    built = stack({'listeners': subjects},
                  names=[f'subject {i + 1}' for i in range(3)],
                  level_names=['Group', 'Subject', 'Feature'])

    bundle = hyp.plot(built, show=False, return_model=True)
    meta = bundle['trace_metadata']
    # three leaves plus the group mean, in hypertools' order
    assert len(bundle['trace_data']) == 4
    assert meta['keys'] == [('listeners', 'subject 1'),
                            ('listeners', 'subject 2'),
                            ('listeners', 'subject 3'),
                            ('listeners',)]
    assert meta['is_mean'] == [False, False, False, True]
    assert meta['level_idx'] == [1, 1, 1, 0]
    assert meta['axis'] == 'columns'
    assert meta['level_names'] == ['Group', 'Subject']
    assert len(bundle['fig'].axes[0].lines) == 4

    reference = hyp.plot(_tutorial_frame(subjects), show=False,
                         return_model=True)
    assert all(np.allclose(a, b) for a, b in
               zip(bundle['trace_data'], reference['trace_data']))


def test_group_columns_reads_the_built_frame():
    frame = stack({'listeners': {'subject 1': np.zeros((10, 6)),
                                 'subject 2': np.ones((10, 6))},
                   'speakers': {'subject 1': np.zeros((10, 6)),
                                'subject 2': np.ones((10, 6))}},
                  level_names=['Group', 'Subject', 'Feature'])
    leaves, meta = group_columns(frame)
    assert meta['leaf_keys'] == [('listeners', 'subject 1'),
                                 ('listeners', 'subject 2'),
                                 ('speakers', 'subject 1'),
                                 ('speakers', 'subject 2')]
    assert meta['n_levels'] == 2
    assert meta['level_names'] == ['Group', 'Subject']
    assert meta['axis'] == 'columns'
    assert list(leaves[0].columns) == FEATURES
    assert leaves[0].columns.name == 'Feature'


def test_single_level_hierarchy():
    frame = stack({'Tech': np.zeros((10, 3)), 'Energy': np.ones((10, 3))},
                  level_names=['Sector', 'Measure'])
    assert frame.columns.nlevels == 2
    bundle = hyp.plot(frame, show=False, return_model=True)
    # one grouping level: two leaves, and NO derived mean
    assert bundle['trace_metadata']['keys'] == [('Tech',), ('Energy',)]
    assert bundle['trace_metadata']['is_mean'] == [False, False]


def test_list_input_and_default_labels():
    frame = stack([np.zeros((5, 2)), np.ones((5, 2))])
    assert list(frame.columns) == [
        ('dataset 1', 'feature 1'), ('dataset 1', 'feature 2'),
        ('dataset 2', 'feature 1'), ('dataset 2', 'feature 2')]
    assert list(frame.columns.names) == ['level 0', 'feature']

    named = stack([np.zeros((5, 2)), np.ones((5, 2))],
                  names=['left', 'right'], level_names=['Side', 'Feature'])
    assert named.columns.get_level_values('Side').unique().tolist() == [
        'left', 'right']


def test_names_apply_to_every_positional_group():
    frame = stack({'listeners': _subjects(2, rows=5, cols=2),
                   'speakers': _subjects(2, rows=5, cols=2, seed=1)},
                  names=['a', 'b'], level_names=['Group', 'Who', 'Feature'])
    assert frame.columns.get_level_values('Who').unique().tolist() == ['a', 'b']
    with pytest.raises(ValueError, match='names='):
        stack({'g': [np.zeros((5, 2))] * 3}, names=['a', 'b'])


def test_dataframe_and_series_leaves():
    index = pd.date_range('2020-01-01', periods=6)
    left = pd.DataFrame(np.zeros((6, 2)), index=index, columns=['x', 'y'])
    right = pd.DataFrame(np.ones((6, 2)), index=index, columns=['x', 'y'])
    frame = stack({'a': left, 'b': right}, level_names=['G', 'Feature'])
    assert frame.index.equals(index)
    assert frame.columns.get_level_values('Feature').unique().tolist() == [
        'x', 'y']

    series = stack({'a': pd.Series(np.zeros(6), name='v', index=index),
                    'b': pd.Series(np.ones(6), name='v', index=index)},
                   level_names=['G', 'Feature'])
    assert list(series.columns) == [('a', 'v'), ('b', 'v')]


def test_differing_indexes_fall_back_to_a_range_index():
    left = pd.DataFrame(np.zeros((4, 2)), index=[10, 11, 12, 13],
                        columns=['x', 'y'])
    right = pd.DataFrame(np.ones((4, 2)), index=[0, 1, 2, 3],
                         columns=['x', 'y'])
    frame = stack({'a': left, 'b': right}, level_names=['G', 'Feature'])
    assert isinstance(frame.index, pd.RangeIndex)
    assert not frame.isna().to_numpy().any()


def test_permuted_features_are_matched_by_name():
    rng = np.random.default_rng(0)
    left = pd.DataFrame(rng.standard_normal((5, 3)), columns=['x', 'y', 'z'])
    right = left[['z', 'x', 'y']] + 1
    frame = stack({'a': left, 'b': right}, level_names=['G', 'Feature'])
    assert [c[1] for c in frame.columns] == ['x', 'y', 'z'] * 2
    assert np.allclose(_group(frame, ('b',)), left.to_numpy() + 1)


def test_aggregate_mean_on_a_single_level_hierarchy():
    frame = stack({'Tech': np.zeros((8, 3)), 'Energy': np.ones((8, 3))},
                  level_names=['Sector', 'Measure'], aggregate='mean')
    assert frame.columns.get_level_values('Sector').unique().tolist() == [
        'Tech', 'Energy', 'mean']
    assert np.allclose(_group(frame, ('mean',)), 0.5)

    bundle = hyp.plot(frame, show=False, return_model=True)
    assert bundle['trace_metadata']['keys'] == [
        ('Tech',), ('Energy',), ('mean',)]


def test_aggregate_appends_a_mean_at_every_level():
    subjects = {'listeners': {f'subject {i + 1}': np.full((6, 2), float(i))
                              for i in range(3)},
                'speakers': {f'subject {i + 1}': np.full((6, 2), 10.0 + i)
                             for i in range(2)}}
    frame = stack(subjects, level_names=['Group', 'Subject', 'Feature'],
                  aggregate='mean')

    keys = list(dict.fromkeys(c[:2] for c in frame.columns))
    # leaves first, then the deepest-level means, then the top-level one --
    # the order hypertools' own hierarchy code appends its means in
    assert keys[-3:] == [('listeners', 'mean'), ('speakers', 'mean'),
                         ('mean', 'mean')]
    assert np.allclose(_group(frame, ('listeners', 'mean')), 1.0)
    assert np.allclose(_group(frame, ('speakers', 'mean')), 10.5)
    assert np.allclose(_group(frame, ('mean', 'mean')),
                       np.mean([0, 1, 2, 10, 11]))


def test_aggregate_accepts_a_callable_and_a_name():
    frames = {'a': np.array([[0.0, 0.0]] * 4), 'b': np.array([[4.0, 4.0]] * 4)}
    median = stack(frames, level_names=['G', 'F'], aggregate='median')
    assert median.columns.get_level_values('G').unique().tolist() == [
        'a', 'b', 'median']

    def spread(members, axis=0):
        return members.max(axis=axis) - members.min(axis=axis)

    ranged = stack(frames, level_names=['G', 'F'], aggregate=spread)
    assert np.allclose(_group(ranged, ('spread',)), 4.0)


def test_aggregate_errors():
    frames = {'a': np.zeros((4, 2)), 'mean': np.ones((4, 2))}
    with pytest.raises(ValueError, match='already exists'):
        stack(frames, aggregate='mean')
    with pytest.raises(ValueError, match='aggregate='):
        stack({'a': np.zeros((4, 2))}, aggregate='mode')
    with pytest.raises(TypeError, match='aggregate='):
        stack({'a': np.zeros((4, 2))}, aggregate=3)
    with pytest.raises(ValueError, match='aggregate='):
        stack({'a': np.zeros((4, 2)), 'b': np.ones((4, 2))},
              aggregate=lambda members, axis=0: members[0][:2])


def test_input_validation():
    with pytest.raises(ValueError, match='same depth'):
        stack({'a': {'b': np.zeros((3, 2))}, 'c': np.zeros((3, 2))})
    with pytest.raises(ValueError, match='same number of rows'):
        stack({'a': np.zeros((3, 2)), 'b': np.zeros((4, 2))})
    with pytest.raises(ValueError, match='same features'):
        stack({'a': pd.DataFrame(np.zeros((3, 2)), columns=['x', 'y']),
               'b': pd.DataFrame(np.zeros((3, 2)), columns=['x', 'z'])})
    with pytest.raises(ValueError, match='same features'):
        stack({'a': np.zeros((3, 2)), 'b': np.zeros((3, 3))})
    with pytest.raises(ValueError, match='level_names='):
        stack({'a': np.zeros((3, 2))}, level_names=['only one'])
    with pytest.raises(ValueError, match='no datasets'):
        stack({})
    with pytest.raises(TypeError, match='dict'):
        stack(pd.DataFrame(np.zeros((3, 2))))
    with pytest.raises(ValueError, match='column MultiIndex'):
        stack({'a': stack({'x': np.zeros((3, 2))}),
               'b': stack({'y': np.zeros((3, 2))})})


def test_stacked_frame_forecasts_per_group():
    rng = np.random.default_rng(0)
    frame = stack({'Tech': np.cumsum(rng.standard_normal((30, 3)), axis=0),
                   'Energy': np.cumsum(rng.standard_normal((30, 3)), axis=0)},
                  level_names=['Sector', 'Measure'])
    forecasts = hyp.predict(frame, model='Kalman', t=5)
    assert len(forecasts) == 2
    assert all(np.asarray(f).shape == (5, 3) for f in forecasts)
