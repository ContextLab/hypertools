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
    # STRENGTHENED again (review of the Task 7 commit): `match="Tech"` is
    # satisfied by ANY ValueError whose text contains the group name, so it
    # could not tell the intended duplicate-timestamp rejection from an
    # over-broad check that also rejects legitimate frames. Assert the
    # message's SUBSTANCE as well as the group name.
    with pytest.warns(UserWarning, match='not sorted in ascending order'):
        with pytest.raises(ValueError,
                           match=r"group \('Tech',\).*duplicated entr.*ill-defined"):
            hyp.predict(df, model='Kalman', t=1)


# The two tests below are FLAT, not hierarchical, and live here on purpose:
# the duplicate-timestamp rejection above is implemented in `resolve_t`, which
# every input reaches, so this task changed flat behaviour too. Pinning both
# sides of that scope beside the hierarchical test keeps the decision (and its
# limits) visible to whoever revisits the check.

def test_a_flat_frame_with_duplicated_timestamps_is_rejected():
    """COMPATIBILITY CHANGE, pinned deliberately rather than left implicit:
    measured at ea5d9b5e (before Task 7), this frame forecast fine, returning
    (1, 3) with only the monotonicity warning. It now raises, because the
    horizon is as ill-defined here as it is inside a group."""
    idx = pd.DatetimeIndex(
        sorted(list(pd.date_range('2020-01-01', periods=5)) * 2))
    df = pd.DataFrame(
        np.random.default_rng(0).normal(size=(10, 3)).cumsum(0), index=idx)
    with pytest.raises(ValueError, match=r'duplicated entr.*ill-defined'):
        hyp.predict(df, model='Kalman', t=1)


def test_a_flat_frame_with_a_duplicated_integer_index_still_forecasts():
    """The OTHER side of that scope (*Decisions (resolved)* #4: "legitimate
    integer-indexed panels are not rejected"). `pd.concat([run_a, run_b])`
    yields a 0..n-1, 0..n-1 index; nothing about the horizon is ambiguous
    there (step 1, continue from the last row), so 1.0's behaviour stands."""
    rng = np.random.default_rng(0)
    stacked = pd.concat([pd.DataFrame(rng.normal(size=(30, 3)).cumsum(0)),
                         pd.DataFrame(rng.normal(size=(30, 3)).cumsum(0))])
    # the stacked index is non-monotonic, which warns (unchanged since 1.0)
    with pytest.warns(UserWarning, match='not sorted in ascending order'):
        out = hyp.predict(stacked, model='Kalman', t=1)
    assert np.asarray(out).shape == (1, 3)


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
    """A no-regression guard, NOT a feature test: it passes on the parent
    commit too (measured in review of the Task 7 commit -- of this module's
    25 tests it was the only one that did). STRENGTHENED to earn its place
    by pinning the flat/hierarchical CONTRAST, which nothing else did: flat
    `return_model=True` yields ONE ``(forecast, model)`` pair, where the
    hierarchical path yields two parallel LISTS
    (`test_return_model_yields_parallel_sequences`). A flat frame that
    accidentally took the grouping branch would return a 1-element list and
    still satisfy the original three assertions' shapes."""
    flat = pd.DataFrame(np.random.default_rng(0).normal(size=(80, 4)).cumsum(0))
    out = hyp.predict(flat, model='Kalman', t=3)
    assert not isinstance(out, list)
    assert np.asarray(out).shape == (3, 4)

    forecast, model = hyp.predict(flat, model='Kalman', t=3, return_model=True)
    assert not isinstance(forecast, list) and not isinstance(model, list)
    assert np.asarray(forecast).shape == (3, 4)
    assert isinstance(model, Kalman) and model.is_fitted
    assert np.allclose(np.asarray(forecast), np.asarray(out),
                       rtol=1e-6, atol=1e-6)


def test_horizon_is_respected_per_group():
    out = hyp.predict(col_frame(), model='Kalman', t=5)
    assert all(np.asarray(g).shape[0] == 5 for g in out)


def test_groups_come_back_in_input_order():
    """STRENGTHENED (review of the Task 7/8 commits): as written this was
    `test_group_forecast_matches_forecasting_that_group_alone` again with
    looser tolerances -- it checked out[0] only, so it said nothing about
    ORDER. `col_frame`'s sectors are ('Tech', 'Energy'), whose SORTED order
    is the reverse, so a `groupby(sort=True)`-style implementation returns
    the same two forecasts transposed; pin both positions AND the negative,
    or the pairing is satisfied by any permutation."""
    df = col_frame()
    out = hyp.predict(df, model='Kalman', t=1)
    alone = [np.asarray(hyp.predict(df['Market'][s], model='Kalman', t=1))
             for s in ('Tech', 'Energy')]
    assert not np.allclose(alone[0], alone[1]), \
        'the two groups must forecast differently for order to be observable'
    assert len(out) == 2
    for got, want in zip(out, alone):
        assert np.allclose(np.asarray(got), want, rtol=1e-6, atol=1e-6)
    assert not np.allclose(np.asarray(out[0]), alone[1]), \
        'the groups came back in sorted (Energy, Tech) order'


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
    shared as one constructed instance -- passed every prescribed test.

    EXTENDED (review of the Task 7 commit): a dict spec is NOT always
    stateless. `{'model': <instance>}` is an accepted spec form (the same one
    `tests/test_pipeline_analyze_hardening.py:92` uses for `cluster=`), and
    passing it through un-copied broke all three ownership promises at once:
    the caller's instance was fitted, every group shared it, and groups 2..n
    fell onto `predict_new` instead of being fitted independently."""
    df = col_frame()
    alone = [np.asarray(hyp.predict(df['Market'][s], model='Kalman', t=1))
             for s in ('Tech', 'Energy')]
    caller_instance = Kalman()
    for spec in (Kalman, {'model': 'Kalman', 'kwargs': {}},
                 {'model': caller_instance}):
        forecasts, models = hyp.predict(df, model=spec, t=1,
                                        return_model=True)
        assert models[0] is not models[1], f'{spec!r} shared one model'
        assert all(m.is_fitted for m in models)
        for got, want in zip(forecasts, alone):
            assert np.allclose(np.asarray(got), want, rtol=1e-6, atol=1e-6), \
                f'{spec!r} did not fit each group independently'
    assert not caller_instance.is_fitted, \
        "the caller's instance was fitted through the dict spec"
    assert all(m is not caller_instance for m in models)


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


def test_warnings_survive_a_group_that_fails_with_a_non_valueerror():
    """ADDED (review of the Task 7 commit): `warnings.catch_warnings(record=
    True)` SUPPRESSES what it records, so anything not re-emitted is lost.
    The re-emission ran after the `with` block and only `ValueError` was
    caught, so a group that warned and then failed with a TypeError (here the
    deprecated ``params`` spec, warning on its way to `Kalman(bogus=1)`'s
    TypeError) silently dropped the warning that the FLAT path emits."""
    spec = {'model': 'Kalman', 'params': {'bogus': 1}}
    with pytest.warns(DeprecationWarning) as rec:
        with pytest.raises(TypeError, match='bogus'):
            hyp.predict(col_frame(), model=dict(spec), t=1)
    messages = [str(w.message) for w in rec
                if issubclass(w.category, DeprecationWarning)]
    assert len(messages) == 1, f'the failing group warns once; got {messages}'
    assert messages[0].startswith("group ('Market', 'Tech'): "), messages[0]

    # the flat control: the same spec on one group's frame warns identically
    with pytest.warns(DeprecationWarning, match='deprecated'):
        with pytest.raises(TypeError, match='bogus'):
            hyp.predict(col_frame()['Market']['Tech'], model=dict(spec), t=1)


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


@pytest.mark.parametrize('frame_of', [col_frame, row_frame],
                         ids=['column-axis', 'row-axis'])
@pytest.mark.parametrize('kwargs, pattern', [
    ({'model': 'Kalman', 't': 0}, r'must be >= 1; got 0'),
    ({'model': 'Kalman', 't': None}, r'must be a positive integer'),
    ({'model': 'Kalman', 't': 1.5}, r'not a float'),
    ({'model': 'Kalmann', 't': 1}, r"unknown predict model 'Kalmann'"),
    ({'model': {'kwargs': {}}, 't': 1}, r"must include a 'model' key"),
])
def test_whole_call_argument_errors_are_not_blamed_on_a_group(frame_of,
                                                              kwargs, pattern):
    """`t` and `model=` describe the WHOLE CALL, so their errors must not
    carry a group's name.

    Every check on them ran inside the per-group recursion, so a caller who
    typed `t=0` was told "group ('Tech',): t (forecast horizon) must be >= 1"
    -- measured on all four spellings before the fix -- and went looking for
    a problem in that group's data. The prefix is now reserved for failures
    that really are one group's (too little history, duplicated times), as
    `test_group_with_too_little_history_raises_naming_the_group` still pins.
    """
    with pytest.raises(ValueError, match=pattern) as excinfo:
        hyp.predict(frame_of(), **kwargs)
    assert not str(excinfo.value).startswith('group '), \
        f'a whole-call mistake was blamed on a group: {excinfo.value}'


def test_only_valueerrors_are_prefixed_with_the_group_key():
    """The DOCSTRING's promise, pinned against what the loop actually does.

    It read "Per-group errors are re-raised prefixed with the group's key"
    without qualification, but the loop catches `ValueError` alone (see the
    `except ValueError` in `predict`'s group loop), so a TypeError escaped
    with a message indistinguishable from the flat path's -- doc and code
    disagreed. Both halves are asserted here so they cannot drift apart
    again: whichever way a future change goes, this test moves with it."""
    doc = hyp.predict.__doc__
    assert 'per-group `ValueError`' in doc, \
        'the docstring must scope the group-key promise to ValueError'
    assert 'Other exception types propagate unchanged' in doc

    # the promised half: a per-group ValueError IS named
    idx = pd.MultiIndex.from_tuples([('Tech', 0), ('Energy', 0)],
                                    names=['Sector', 'day'])
    with pytest.raises(ValueError) as caught:
        hyp.predict(pd.DataFrame(np.zeros((2, 3)), index=idx),
                    model='Kalman', t=1)
    assert str(caught.value).startswith('group ('), caught.value

    # the excluded half: a TypeError raised while a group is being forecast
    # (`Kalman(bogus=3)`) reaches the caller exactly as the flat path raises
    # it. That is the right message HERE -- a bad constructor kwarg is the
    # caller's, not a group's -- but it is not the prefixed one, which is
    # precisely what the unqualified docstring claimed.
    with pytest.raises(TypeError, match='bogus') as typed:
        hyp.predict(col_frame(), model='Kalman', t=1, bogus=3)
    assert not str(typed.value).startswith('group ')
    with pytest.raises(TypeError, match='bogus') as flat:
        hyp.predict(col_frame()['Market']['Tech'], model='Kalman', t=1,
                    bogus=3)
    assert str(typed.value) == str(flat.value)
