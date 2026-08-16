# -*- coding: utf-8 -*-
"""Two things a hierarchy user can be surprised by, pinned.

**Group order.** Feature correspondence across a column hierarchy's groups
is nominal, so permuting the columns WITHIN a group cannot move a
trajectory. The order of the GROUPS is a different question and is not
neutralised: groups become datasets, `reduce` row-stacks every dataset and
fits ONE model on the stack, so group order is row order in that stack --
and the default `IncrementalPCA` fits by `partial_fit` over successive
minibatches, so its fit depends on that order. Measured here: permuting the
sector BLOCKS of a 40x(4x5) frame moves every trace by ~1.7% of the plotted
range under `IncrementalPCA` and not at all (1e-14) under `PCA`, and the
identical asymmetry exists for a plain list of datasets, which is where it
comes from.

That behaviour is DOCUMENTED rather than fixed (a canonical group order
would move every hierarchy figure that already exists, and changing the
default reducer would move all of them), so these tests pin it: if it ever
changes, it must change deliberately, and the guide and docstring must
change with it.

**Re-applying the bundled pipeline.** `hyp.plot(df, return_model=True)`
fits its bundled `Pipeline` on the GROUPS of a column-hierarchical frame,
each as wide as one group. Re-applying it to that same frame used to raise
scikit-learn's "X has 20 features, but IncrementalPCA is expecting 5
features as input" -- and, when the reduce stage was a no-op because each
group already had <= ndims columns, to silently return the UNGROUPED,
unreduced frame. The pipeline now records the grouping and reproduces it.
"""

import pickle
import warnings

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.core.hierarchy import group_columns
from hypertools.core.pipeline import Pipeline

SECTORS = ('Tech', 'Energy', 'Health', 'Fin')
MEASURES = ('return', 'volatility', 'momentum', 'spread', 'turnover')


def _wide_frame(seed=0):
    """40 observations x 4 sector blocks x 5 shared measurements."""
    columns = pd.MultiIndex.from_tuples(
        [('Market', sector, measure)
         for sector in SECTORS for measure in MEASURES],
        names=['Market', 'Sector', 'Measure'])
    values = np.cumsum(
        np.random.default_rng(seed).standard_normal((40, len(columns))),
        axis=0)
    return pd.DataFrame(values, columns=columns)


def _drawn(frame, **kwargs):
    """{trace key: plotted trajectory} for one hierarchical plot.

    Keyed by hierarchy key, not by position, so a permutation that also
    REORDERS the traces is still compared trace-for-trace.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bundle = hyp.plot(frame, show=False, return_model=True, **kwargs)
    keys = [tuple(key) for key in bundle['trace_metadata']['keys']]
    return dict(zip(keys, [np.asarray(a) for a in bundle['trace_data']]))


def _max_shift(before, after):
    assert set(before) == set(after), 'the two plots drew different groups'
    return max(float(np.abs(before[key] - after[key]).max())
               for key in before)


def _plotted_range(traces):
    stacked = np.vstack(list(traces.values()))
    return float((stacked.max(axis=0) - stacked.min(axis=0)).max())


# --------------------------------------------------------------------------
# group order
# --------------------------------------------------------------------------

@pytest.mark.parametrize('reducer', ['IncrementalPCA', 'PCA', 'TruncatedSVD',
                                     'FactorAnalysis'])
def test_a_within_group_column_permutation_never_moves_a_trace(reducer):
    """Nominal correspondence, restated as the contrast for the test below."""
    frame = _wide_frame()
    permuted = frame[[('Market', sector, measure)
                      for sector in SECTORS
                      for measure in ('turnover', 'return', 'spread',
                                      'momentum', 'volatility')]]
    before = _drawn(frame, reduce=reducer)
    shift = _max_shift(before, _drawn(permuted, reduce=reducer))
    assert shift < 1e-9, (
        f'{reducer}: a within-group column permutation moved a trace by '
        f'{shift:.3e}; feature correspondence across groups is by NAME')


def test_a_between_group_block_permutation_moves_the_default_reducer():
    """PINNED, not endorsed: block order reaches the shared reduction.

    Permuting whole BLOCKS is a relabelling-free permutation of identical
    data, yet the default reducer's fit depends on the row order of the
    stacked datasets, so every leaf and every mean moves. Documented in
    `hyp.plot`'s `x` entry and docs/hierarchy.rst ("Group order, which is a
    different question"). If this assertion ever fails, the behaviour
    changed -- update BOTH of those, and this test, deliberately.
    """
    frame = _wide_frame()
    reordered = frame[[('Market', sector, measure)
                       for sector in ('Fin', 'Health', 'Energy', 'Tech')
                       for measure in MEASURES]]

    before = _drawn(frame)                      # default IncrementalPCA
    shift = _max_shift(before, _drawn(reordered))
    fraction = shift / _plotted_range(before)
    assert 1e-3 < fraction < 0.25, (
        f'block order moved the default-reducer figure by {fraction:.2%} of '
        'the plotted range; the documented measurement is ~1.7%')


def test_an_order_invariant_reducer_is_the_documented_remedy():
    """`reduce='PCA'` is what the docs tell the user to pass, so pin it."""
    frame = _wide_frame()
    reordered = frame[[('Market', sector, measure)
                       for sector in ('Fin', 'Health', 'Energy', 'Tech')
                       for measure in MEASURES]]
    shift = _max_shift(_drawn(frame, reduce='PCA'),
                       _drawn(reordered, reduce='PCA'))
    assert shift < 1e-9, (
        f'PCA moved a trace by {shift:.3e} under a block permutation; the '
        'guide recommends it precisely because it does not')


def test_the_same_order_dependence_exists_for_a_plain_list_of_datasets():
    """The cause is the shared reduction space, not the hierarchy.

    This is what makes "fix it for hierarchies only" the wrong shape of
    fix: the identical asymmetry is reachable without any MultiIndex.
    """
    rng = np.random.default_rng(0)
    datasets = [np.cumsum(rng.standard_normal((40, 5)), axis=0)
                for _ in range(4)]

    def xform(order, reducer):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            bundle = hyp.plot([datasets[i] for i in order], show=False,
                              return_model=True, reduce=reducer)
        return {i: np.asarray(a)
                for i, a in zip(order, bundle['xform_data'])}

    forward = xform([0, 1, 2, 3], 'IncrementalPCA')
    reverse = xform([3, 2, 1, 0], 'IncrementalPCA')
    assert max(float(np.abs(forward[i] - reverse[i]).max())
               for i in forward) > 1e-3

    forward = xform([0, 1, 2, 3], 'PCA')
    reverse = xform([3, 2, 1, 0], 'PCA')
    assert max(float(np.abs(forward[i] - reverse[i]).max())
               for i in forward) < 1e-9


def test_the_group_order_caveat_is_documented_where_a_user_will_meet_it():
    """A pinned surprise that is not written down is just a surprise."""
    assert 'Group order, which is a different question' in _guide()
    assert "reduce='PCA'" in _guide()

    entry = hyp.plot.__doc__
    assert 'The order of the' in entry and 'GROUPS' in entry
    assert 'partial_fit' in entry


def _guide():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(os.path.dirname(here), 'docs/hierarchy.rst'),
              encoding='utf-8') as handle:
        return handle.read()


# --------------------------------------------------------------------------
# re-applying the bundled pipeline
# --------------------------------------------------------------------------

def _bundle(frame, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return hyp.plot(frame, show=False, return_model=True, **kwargs)


def test_the_bundled_pipeline_re_applies_to_the_frame_that_produced_it():
    """The round trip that already worked for a flat frame and for a list.

    Before the fix this raised, from inside scikit-learn: "X has 20
    features, but IncrementalPCA is expecting 5 features as input".
    """
    frame = _wide_frame()
    bundle = _bundle(frame)
    out = bundle['pipeline'].transform(frame)

    assert isinstance(out, list) and len(out) == len(SECTORS)
    for reference, produced in zip(bundle['xform_data'], out):
        assert np.allclose(np.asarray(reference), np.asarray(produced))


def test_a_no_op_reduce_stage_never_returns_the_ungrouped_frame():
    """The silently-wrong half: groups already <= ndims wide.

    `reduce` fits no model when every dataset already has <= ndims columns,
    so the reduce stage passes its input straight through -- which used to
    mean `transform(df)` handed back the caller's own (40, 12) hierarchical
    frame, ungrouped and unreduced, with no error to notice.
    """
    columns = pd.MultiIndex.from_tuples(
        [('Market', sector, measure)
         for sector in SECTORS for measure in ('a', 'b', 'c')],
        names=['Market', 'Sector', 'Measure'])
    frame = pd.DataFrame(
        np.cumsum(np.random.default_rng(1).standard_normal((40, 12)), axis=0),
        columns=columns)

    bundle = _bundle(frame)
    out = bundle['pipeline'].transform(frame)

    assert isinstance(out, list), (
        f'transform returned a {type(out).__name__}, not one entry per group')
    assert [np.asarray(a).shape for a in out] == [(40, 3)] * len(SECTORS)
    leaves, _meta = group_columns(frame)
    for leaf, produced in zip(leaves, out):
        assert np.allclose(leaf.to_numpy(), np.asarray(produced))


def test_the_bundled_pipeline_matches_features_by_name():
    """`transform` is nominal in the same sense the figure is.

    `group_columns` matches a frame's groups to its FIRST group's order, so
    without matching against the fit-time labels a within-group permutation
    -- harmless to the figure -- silently moved every re-applied trace.
    """
    frame = _wide_frame()
    shuffled = frame[[('Market', sector, measure)
                      for sector in SECTORS
                      for measure in ('turnover', 'return', 'spread',
                                      'momentum', 'volatility')]]
    pipeline = _bundle(frame)['pipeline']
    for straight, permuted in zip(pipeline.transform(frame),
                                  pipeline.transform(shuffled)):
        assert np.allclose(np.asarray(straight), np.asarray(permuted))


def test_a_flattened_frame_is_refused_by_name_not_passed_through():
    frame = _wide_frame()
    pipeline = _bundle(frame)['pipeline']
    flat = frame.copy()
    flat.columns = ['_'.join(label) for label in frame.columns]

    with pytest.raises(ValueError) as excinfo:
        pipeline.transform(flat)
    message = str(excinfo.value)
    assert '5-feature groups' in message
    assert '[20] feature(s) per dataset' in message
    assert "df.columns.map('_'.join)" in message


def test_a_frame_naming_other_measurements_is_refused_by_name():
    frame = _wide_frame()
    pipeline = _bundle(frame)['pipeline']
    other = pd.DataFrame(
        np.random.default_rng(2).standard_normal((10, 10)),
        columns=pd.MultiIndex.from_tuples(
            [('Market', sector, measure)
             for sector in ('Tech', 'Energy')
             for measure in ('return', 'volatility', 'momentum', 'spread',
                             'drawdown')],
            names=['Market', 'Sector', 'Measure']))

    with pytest.raises(ValueError) as excinfo:
        pipeline.transform(other)
    message = str(excinfo.value)
    assert "missing ['turnover']" in message
    assert "unexpected ['drawdown']" in message


def test_a_list_of_per_group_arrays_is_accepted_unchanged():
    """A list is already grouped, so the hook must not group it again."""
    frame = _wide_frame()
    bundle = _bundle(frame)
    leaves, _meta = group_columns(frame)
    out = bundle['pipeline'].transform([leaf.to_numpy() for leaf in leaves])
    for reference, produced in zip(bundle['xform_data'], out):
        assert np.allclose(np.asarray(reference), np.asarray(produced))


def test_the_bundled_pipeline_still_pickles():
    """`hyp.save`/reuse-later must survive the extra recorded state
    (2026-07 audit F21-002 keeps this pipeline picklable)."""
    frame = _wide_frame()
    bundle = _bundle(frame)
    revived = pickle.loads(pickle.dumps(bundle['pipeline']))
    assert revived.input_hierarchy == bundle['pipeline'].input_hierarchy
    for reference, produced in zip(bundle['xform_data'],
                                   revived.transform(frame)):
        assert np.allclose(np.asarray(reference), np.asarray(produced))


def test_nothing_is_recorded_for_non_hierarchical_input():
    """Only the column-hierarchy path records anything, so only it changes."""
    rng = np.random.default_rng(3)
    flat = pd.DataFrame(np.cumsum(rng.standard_normal((40, 20)), axis=0))
    assert _bundle(flat)['pipeline'].input_hierarchy is None

    arrays = [np.cumsum(rng.standard_normal((40, 20)), axis=0)
              for _ in range(3)]
    assert _bundle(arrays)['pipeline'].input_hierarchy is None

    rows = pd.MultiIndex.from_product([['A', 'B'], range(20)],
                                      names=['group', 'obs'])
    row_hierarchy = pd.DataFrame(
        np.cumsum(rng.standard_normal((40, 8)), axis=0), index=rows,
        columns=list('abcdefgh'))
    row_bundle = _bundle(row_hierarchy)
    # a ROW hierarchy's leaves keep the frame's own width, so its pipeline
    # already round-tripped and nothing needed recording
    assert row_bundle['pipeline'].input_hierarchy is None
    assert np.asarray(row_bundle['pipeline'].transform(row_hierarchy)).shape \
        == (40, 3)


def test_a_hierarchy_pipeline_can_be_reused_through_plot():
    """The documented `pipeline=` reuse path keeps working on this input."""
    frame = _wide_frame()
    pipeline = _bundle(frame)['pipeline']
    other = _wide_frame(seed=7)
    reused = _bundle(other, pipeline=pipeline)
    assert len(reused['xform_data']) == len(SECTORS)
    for straight, through_plot in zip(pipeline.transform(other),
                                      reused['xform_data']):
        assert np.allclose(np.asarray(straight), np.asarray(through_plot))


@pytest.mark.parametrize('spec, error, fragment', [
    ('columns', TypeError, 'must be a dict or None'),
    ({'axis': 'rows', 'n_features': 3}, ValueError, "must be 'columns'"),
    ({'axis': 'columns'}, ValueError, 'must be a positive int'),
    ({'axis': 'columns', 'n_features': 0}, ValueError, 'positive int'),
    ({'axis': 'columns', 'n_features': 3, 'feature_correspondence': 'nearest'},
     ValueError, "must be 'name' or 'position'"),
    ({'axis': 'columns', 'n_features': 3, 'feature_labels': ['a', 'b']},
     ValueError, 'one entry per feature'),
])
def test_a_malformed_input_hierarchy_is_rejected_at_construction(
        spec, error, fragment):
    """Validated where it is written, not a session later on transform."""
    with pytest.raises(error) as excinfo:
        Pipeline(['PCA'], input_hierarchy=spec)
    assert fragment in str(excinfo.value)


# --------------------------------------------------------------------------
# a CALLER-SUPPLIED pipeline is bundled under the same promise
# --------------------------------------------------------------------------
# The recording above covered only the pipeline `plot()` builds for itself.
# A `pipeline=` the caller passed in is handed straight back in the bundle,
# so `return_model`'s promise ("calling bundle['pipeline'].transform(df) on
# a column-hierarchical frame groups it first") covered it too -- but with
# no `input_hierarchy` attached it still raised the exact pre-1.1.0 error
# that promise names: "X has 15 features, but IncrementalPCA is expecting 5
# features as input".

def _user_pipeline(frame, **kwargs):
    """A pipeline the USER fitted, on the frame's groups, outside plot()."""
    leaves, _meta = group_columns(frame)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _xform, pipeline = hyp.analyze(
            [leaf.to_numpy() for leaf in leaves], reduce='IncrementalPCA',
            ndims=3, return_model=True, **kwargs)
    return pipeline


def test_a_caller_supplied_pipeline_is_bundled_with_the_grouping_recorded():
    """The reported repro: `bundle['pipeline'].transform(df)` raised."""
    frame = _wide_frame()
    pipeline = _user_pipeline(frame)
    assert pipeline.input_hierarchy is None

    bundle = _bundle(frame, pipeline=pipeline)
    # the docstring's identity promise is unchanged -- it is recorded IN
    # PLACE, on that same object
    assert bundle['pipeline'] is pipeline
    assert pipeline.input_hierarchy == {
        'axis': 'columns',
        'n_features': len(MEASURES),
        'feature_correspondence': 'name',
        'feature_labels': list(MEASURES),
    }

    out = bundle['pipeline'].transform(frame)
    assert isinstance(out, list) and len(out) == len(SECTORS)
    for reference, produced in zip(bundle['xform_data'], out):
        assert np.allclose(np.asarray(reference), np.asarray(produced))


def test_a_caller_supplied_pipelines_own_hierarchy_record_is_left_alone():
    """An `input_hierarchy` the caller's pipeline already carries belongs to
    its OWN fit, so plot() must not overwrite it with the frame it happened
    to be applied to here. (Its `n_features` cannot disagree: a mismatch
    would have raised in `_regroup_hierarchical_input` during the analyze()
    call that drew the figure, long before the bundle was built.)"""
    frame = _wide_frame()
    pipeline = _user_pipeline(frame)
    # written directly, as a pipeline restored from an earlier session would
    # carry it -- positional correspondence, and no recorded feature labels
    own = {'axis': 'columns', 'n_features': len(MEASURES),
           'feature_correspondence': 'position', 'feature_labels': None}
    pipeline.input_hierarchy = own

    assert _bundle(frame, pipeline=pipeline)['pipeline'].input_hierarchy \
        == own


def test_nothing_is_recorded_on_a_caller_supplied_pipeline_off_hierarchy():
    """Only the column-hierarchy path records anything, on this branch too."""
    frame = _wide_frame()
    pipeline = _user_pipeline(frame)
    arrays = [np.cumsum(np.random.default_rng(s).standard_normal((40, 5)),
                        axis=0) for s in range(len(SECTORS))]
    assert _bundle(arrays, pipeline=pipeline)['pipeline'].input_hierarchy \
        is None
