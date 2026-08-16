"""Hierarchical forecasting: grouping, return shape, model ownership.

Column hierarchy -> one forecast per group (innermost level = features).
Row hierarchy    -> one forecast per group of the OUTER levels, with the
                    innermost level kept as each group's time index.
"""
import copy

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.predict.kalman import Kalman


def col_frame(T=120, seed=0):
    rng = np.random.default_rng(seed)
    tuples = [('Market', sector, m)
              for sector in ('Tech', 'Energy')
              for m in ('return', 'volatility', 'momentum')]
    cols = pd.MultiIndex.from_tuples(tuples,
                                     names=['Market', 'Sector', 'Measure'])
    return pd.DataFrame(rng.normal(size=(T, 6)).cumsum(axis=0) + 100.0, columns=cols)


def row_frame(T=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([['Tech', 'Energy'], range(T)],
                                     names=['Sector', 'day'])
    return pd.DataFrame(rng.normal(size=(2 * T, 3)).cumsum(axis=0), index=idx)


def dated_row_frame(T=60, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.date_range('2020-01-01', periods=T)
    idx = pd.MultiIndex.from_product([['Tech', 'Energy'], days],
                                     names=['Sector', 'date'])
    return pd.DataFrame(rng.normal(size=(2 * T, 3)).cumsum(axis=0), index=idx)


# --- grouping and shape -----------------------------------------------------

def test_column_hierarchy_forecasts_each_group():
    out = hyp.predict(col_frame(), model='Kalman', t=1)
    assert isinstance(out, list) and len(out) == 2
    assert all(np.asarray(g).shape == (1, 3) for g in out)


def test_row_hierarchy_groups_by_every_level_above_time():
    """(Sector, day) -> 2 sector forecasts, NOT 2*T one-row leaves."""
    out = hyp.predict(row_frame(), model='Kalman', t=2)
    assert isinstance(out, list) and len(out) == 2
    assert all(np.asarray(g).shape == (2, 3) for g in out)


def test_row_hierarchy_forecast_keeps_a_datetime_index():
    out = hyp.predict(dated_row_frame(), model='Kalman', t=3)
    for g in out:
        assert isinstance(g.index, pd.DatetimeIndex)
        assert g.index[0] > pd.Timestamp('2020-02-29')


def test_future_timestamp_horizon_on_a_row_hierarchy():
    """reset_index(drop=True) would make this impossible."""
    out = hyp.predict(dated_row_frame(), model='Kalman',
                      t=pd.Timestamp('2020-03-05'))
    assert len(out) == 2
    for g in out:
        assert g.index[-1] == pd.Timestamp('2020-03-05')
        assert np.asarray(g).shape[1] == 3


def test_at_or_before_timestamp_truncates_each_group():
    out = hyp.predict(dated_row_frame(), model='Kalman',
                      t=pd.Timestamp('2020-02-01'))
    assert len(out) == 2
    for g in out:
        assert g.index[-1] <= pd.Timestamp('2020-02-01')
        assert len(g) == 32


def test_unsorted_times_warn_naming_the_group():
    days = pd.date_range('2020-01-01', periods=40)
    perm = np.random.default_rng(0).permutation(40)
    idx = pd.MultiIndex.from_arrays(
        [['Tech'] * 40 + ['Energy'] * 40, list(days[perm]) + list(days)],
        names=['Sector', 'date'])
    df = pd.DataFrame(np.random.default_rng(1).normal(size=(80, 3)).cumsum(0),
                      index=idx)
    # STRENGTHENED vs the plan (which asserted only 'not sorted in ascending
    # order'): that substring is already in the UNPREFIXED warning
    # `predict/common.py`'s monotonicity check has raised since 1.0, so the
    # plan's version passed the moment the TypeError stopped -- it did not
    # test F8's "with a group name prepended" at all. Assert the prefix.
    with pytest.warns(UserWarning, match='not sorted in ascending order') as rec:
        hyp.predict(df, model='Kalman', t=1)
    unsorted = [str(w.message) for w in rec
                if 'not sorted in ascending order' in str(w.message)]
    assert len(unsorted) == 1, \
        f'exactly the Tech group is unsorted; got {unsorted}'
    assert unsorted[0].startswith("group ('Tech',): "), unsorted[0]


def test_duplicate_times_raise_naming_the_group():
    days = pd.date_range('2020-01-01', periods=20)
    idx = pd.MultiIndex.from_arrays(
        [['Tech'] * 40 + ['Energy'] * 40,
         list(days) * 2 + list(pd.date_range('2020-01-01', periods=40))],
        names=['Sector', 'date'])
    df = pd.DataFrame(np.random.default_rng(1).normal(size=(80, 3)).cumsum(0),
                      index=idx)
    # the repeated Tech index is also NON-monotonic, so this group warns on
    # its way to the error. Asserting the warning (STRENGTHENED vs the plan,
    # which asserted only the raise) pins that the per-group re-emission
    # happens BEFORE the re-raise -- `warnings.catch_warnings` swallows
    # whatever is not re-emitted, so a naive implementation loses it
    # silently -- and keeps the suite's zero-unasserted-warnings property.
    with pytest.warns(UserWarning, match='not sorted in ascending order'):
        with pytest.raises(ValueError, match="Tech"):
            hyp.predict(df, model='Kalman', t=1)


def test_group_forecast_matches_forecasting_that_group_alone():
    df = col_frame()
    grouped = hyp.predict(df, model='Kalman', t=1)
    alone = hyp.predict(df['Market']['Tech'], model='Kalman', t=1)
    assert np.allclose(np.asarray(grouped[0]), np.asarray(alone),
                       rtol=1e-6, atol=1e-6)


def test_grouped_leaves_are_non_hierarchical_so_the_recursion_terminates(
        monkeypatch):
    """The recursion guard, made OBSERVABLE rather than inferred (Revision
    note (v6) D1/D2). `predict()` recurses with `predict(group, ...)`, so a
    leaf still carrying its grouping levels is re-detected by the same
    `nlevels >= 2` predicate and regrouped without bound -- measured on v5's
    `sub.T` leaves. Both core helpers are WRAPPED here: they still run (this
    OBSERVES, it does not substitute), recording each leaf's axis index and
    their own call counts. Patching `hypertools.core.hierarchy` rather than
    the predict module is deliberate -- `predict()` imports them inside the
    function, so the name is looked up on the source module at call time.
    A test that merely 'does not hang' would not be adequate: the counts and
    the leaf indices are asserted explicitly.
    """
    import hypertools.core.hierarchy as hier

    real_columns, real_rows = hier.group_columns, hier.group_rows_for_forecast
    col_calls, row_calls, seen_cols, seen_rows = [], [], [], []

    def observing_columns(df):
        leaves, meta = real_columns(df)
        col_calls.append(df.columns.nlevels)
        seen_cols.extend(leaf.columns for leaf in leaves)
        return leaves, meta

    def observing_rows(df):
        groups, keys = real_rows(df)
        row_calls.append(df.index.nlevels)
        seen_rows.extend(group.index for group in groups)
        return groups, keys

    monkeypatch.setattr(hier, 'group_columns', observing_columns)
    monkeypatch.setattr(hier, 'group_rows_for_forecast', observing_rows)

    col_out = hyp.predict(col_frame(), model='Kalman', t=1)
    row_out = hyp.predict(row_frame(), model='Kalman', t=2)

    assert len(col_calls) == 1, \
        f'group_columns ran {len(col_calls)}x: the leaves were regrouped'
    assert len(row_calls) == 1, \
        f'group_rows_for_forecast ran {len(row_calls)}x: leaves regrouped'
    assert len(seen_cols) == 2 and len(seen_rows) == 2
    assert all(not isinstance(cols, pd.MultiIndex) for cols in seen_cols)
    assert all(not isinstance(idx, pd.MultiIndex) for idx in seen_rows)
    assert len(col_out) == 2
    assert all(np.asarray(f).shape == (1, 3) for f in col_out)
    assert len(row_out) == 2
    assert all(np.asarray(f).shape == (2, 3) for f in row_out)


def test_duplicate_innermost_names_forecast_by_occurrence():
    """DECIDED (Revision note (v6) D3, refined by (v8) F2): flattening onto
    the feature axis can collide two innermost labels WITHIN a group; they
    are kept, and matched across groups by (label, occurrence). Measured:
    2 groups, widths [3, 3], one (1, 3) forecast each -- nothing merged,
    nothing dropped."""
    cols = pd.MultiIndex.from_tuples(
        [('Rig', 'North', s) for s in ('temp', 'temp', 'flow')]
        + [('Rig', 'South', s) for s in ('temp', 'temp', 'flow')],
        names=['Rig', 'Well', 'Sensor'])
    df = pd.DataFrame(
        np.random.default_rng(0).normal(size=(120, 6)).cumsum(axis=0) + 100.0,
        columns=cols)
    out = hyp.predict(df, model='Kalman', t=1)
    assert len(out) == 2
    assert all(np.asarray(f).shape == (1, 3) for f in out)


def test_flat_frame_return_type_is_unchanged():
    flat = pd.DataFrame(np.random.default_rng(0).normal(size=(80, 4)).cumsum(0))
    out = hyp.predict(flat, model='Kalman', t=3)
    assert not isinstance(out, list)
    assert np.asarray(out).shape == (3, 4)


def test_horizon_is_respected_per_group():
    out = hyp.predict(col_frame(), model='Kalman', t=5)
    assert all(np.asarray(g).shape[0] == 5 for g in out)


def test_groups_come_back_in_input_order():
    df = col_frame()
    out = hyp.predict(df, model='Kalman', t=1)
    alone = hyp.predict(df['Market']['Tech'], model='Kalman', t=1)
    assert np.allclose(np.asarray(out[0]), np.asarray(alone))


# --- model ownership --------------------------------------------------------

def test_return_model_yields_parallel_sequences():
    forecasts, models = hyp.predict(col_frame(), model='Kalman', t=1,
                                    return_model=True)
    assert len(forecasts) == len(models) == 2
    assert all(np.asarray(f).shape == (1, 3) for f in forecasts)


def test_returned_models_are_distinct_objects_per_group():
    _, models = hyp.predict(col_frame(), model='Kalman', t=1, return_model=True)
    assert models[0] is not models[1]


def test_an_unfitted_instance_is_not_mutated_across_groups():
    """A REAL instance, not the string 'Kalman'. Fitting group 0 must not
    push group 1 onto the predict_new path, nor fit the caller's object."""
    shared = Kalman()
    assert not shared.is_fitted
    forecasts, models = hyp.predict(col_frame(), model=shared, t=1,
                                    return_model=True)
    assert not shared.is_fitted, 'the caller\'s instance was mutated'
    assert models[0] is not shared and models[1] is not shared
    independent = hyp.predict(col_frame()['Market']['Energy'],
                              model='Kalman', t=1)
    assert np.allclose(np.asarray(forecasts[1]), np.asarray(independent),
                       rtol=1e-6, atol=1e-6)


def test_a_fitted_instance_is_reused_not_refitted():
    """predict.py:245-249 promises fitted-model REUSE; hierarchical input
    must honour it, applying the same learned parameters to each group."""
    df = col_frame()
    _, fitted = hyp.predict(df['Market']['Tech'], model='Kalman', t=1,
                            return_model=True)
    assert fitted.is_fitted
    grouped = hyp.predict(df, model=fitted, t=1)
    reference = copy.deepcopy(fitted)
    per_group = [np.asarray(hyp.predict(df['Market'][s],
                                        model=copy.deepcopy(reference), t=1))
                 for s in ('Tech', 'Energy')]
    for got, want in zip(grouped, per_group):
        assert np.allclose(np.asarray(got), want, rtol=1e-6, atol=1e-6)


def test_a_fitted_instance_is_not_mutated_across_groups():
    df = col_frame()
    _, fitted = hyp.predict(df['Market']['Tech'], model='Kalman', t=1,
                            return_model=True)
    before = copy.deepcopy(fitted)
    _, models = hyp.predict(df, model=fitted, t=1, return_model=True)
    assert models[0] is not fitted and models[1] is not fitted
    assert np.allclose(np.asarray(hyp.predict(df['Market']['Tech'],
                                              model=before, t=1)),
                       np.asarray(hyp.predict(df['Market']['Tech'],
                                              model=fitted, t=1)))


def test_a_class_or_dict_spec_fits_one_model_per_group():
    """ADDED (not in the plan's 22): the ownership rule says a name, a CLASS
    or a DICT spec is passed through un-copied because it is stateless, and
    that each group is fitted independently from it. Only the `str` branch
    was pinned, so a blanket `copy.deepcopy(model)` -- or a spec accidentally
    shared as one constructed instance -- passed every prescribed test."""
    df = col_frame()
    alone = [np.asarray(hyp.predict(df['Market'][s], model='Kalman', t=1))
             for s in ('Tech', 'Energy')]
    for spec in (Kalman, {'model': 'Kalman', 'kwargs': {}}):
        forecasts, models = hyp.predict(df, model=spec, t=1,
                                        return_model=True)
        assert models[0] is not models[1], f'{spec!r} shared one model'
        assert all(m.is_fitted for m in models)
        for got, want in zip(forecasts, alone):
            assert np.allclose(np.asarray(got), want, rtol=1e-6, atol=1e-6), \
                f'{spec!r} did not fit each group independently'


def test_a_per_group_warning_keeps_its_category():
    """ADDED (not in the plan's 22): the per-group re-emission has to pass
    `w.category` through. Nothing pinned it, and `warnings.warn(msg)` alone
    would silently turn every non-UserWarning (here the deprecated
    ``{'model': ..., 'params': {...}}`` spec's DeprecationWarning) into a
    UserWarning -- exactly the visibility bug external_stacklevel() exists
    to prevent."""
    with pytest.warns(DeprecationWarning) as rec:
        out = hyp.predict(col_frame(), model={'model': 'Kalman', 'params': {}},
                          t=1)
    assert len(out) == 2
    messages = [str(w.message) for w in rec
                if issubclass(w.category, DeprecationWarning)]
    assert len(messages) == 2, f'one per group expected; got {messages}'
    assert all(m.startswith('group (') for m in messages), messages
    assert all('deprecated' in m for m in messages), messages


def test_grouping_does_not_mutate_the_callers_frame():
    """ADDED (not in the plan's 22): Contract 11 says grouping never mutates
    the caller's frame (`group_columns` copies before flattening each leaf's
    columns), but nothing pinned that end-to-end through `hyp.predict`."""
    df = col_frame()
    before = df.copy(deep=True)
    hyp.predict(df, model='Kalman', t=1)
    pd.testing.assert_frame_equal(df, before)
    assert isinstance(df.columns, pd.MultiIndex)
    assert df.columns.names == ['Market', 'Sector', 'Measure']

    rows = dated_row_frame()
    rows_before = rows.copy(deep=True)
    hyp.predict(rows, model='Kalman', t=1)
    pd.testing.assert_frame_equal(rows, rows_before)
    assert rows.index.nlevels == 2


def test_returned_models_can_be_reused_on_new_data():
    df = col_frame()
    _, models = hyp.predict(df, model='Kalman', t=1, return_model=True)
    again = hyp.predict(df['Market']['Tech'], model=models[0], t=1)
    assert np.asarray(again).shape == (1, 3)


# --- rejections -------------------------------------------------------------

def test_dual_axis_frame_is_rejected():
    idx = pd.MultiIndex.from_product([['a', 'b'], range(30)])
    cols = pd.MultiIndex.from_tuples([('M', 'T'), ('M', 'E')])
    df = pd.DataFrame(np.zeros((60, 2)), index=idx, columns=cols)
    with pytest.raises(ValueError, match='both a row and a column MultiIndex'):
        hyp.predict(df, model='Kalman', t=1)


def test_hierarchical_frame_in_a_list_is_rejected():
    """Today this raises TypeError deep inside pandas; 1.1 says why."""
    with pytest.raises(ValueError, match='hyp.predict'):
        hyp.predict([row_frame()], model='Kalman', t=1)


def test_group_with_too_little_history_raises_naming_the_group():
    idx = pd.MultiIndex.from_tuples([('Tech', 0), ('Energy', 0)],
                                    names=['Sector', 'day'])
    df = pd.DataFrame(np.zeros((2, 3)), index=idx)
    with pytest.raises(ValueError, match='Tech|Energy'):
        hyp.predict(df, model='Kalman', t=1)
