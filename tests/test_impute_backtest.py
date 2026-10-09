# -*- coding: utf-8 -*-
"""`hyp.impute(x, model=[...], truth=full)`: imputer comparison (GH #285).

Replaces the hand-rolled masked per-axis RMSE of
``docs/tutorials/projectile_kalman.ipynb`` cell 9 and the
scattered-vs-occluded imputer comparison of its cell 13: imputers are
scored on the DAMAGED CELLS ONLY, per axis, against the recorded truth.

Real imputers (PPCA, KNNImputer, Kalman, SimpleImputer) on real (seeded)
damage; "a perfect fill scores 0" is exercised on a constant column, where
the column-mean fill is exactly right by construction.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.impute.common import Imputer


@pytest.mark.parametrize('wrapped', [False, True])
def test_scoring_does_not_fit_the_callers_instance(wrapped):
    from hypertools.impute import SimpleImputer
    truth = pd.DataFrame({'x': [1., 2., 3., 4.]})
    damaged = truth.copy()
    damaged.iloc[1, 0] = np.nan
    model = SimpleImputer()
    spec = {'model': model} if wrapped else model
    expected = hyp.impute(damaged, model='SimpleImputer', truth=truth)
    actual = hyp.impute(damaged, model=spec, truth=truth)
    pd.testing.assert_frame_equal(actual, expected)
    assert not model.is_fitted


@pytest.mark.parametrize('wrapped', [False, True])
def test_scoring_refuses_an_imputer_that_has_seen_the_truth(wrapped):
    from hypertools.impute import SimpleImputer
    truth = pd.DataFrame({'x': [1., 2., 3., 4.]})
    model = SimpleImputer().fit(truth)
    damaged = truth.copy()
    damaged.iloc[1, 0] = np.nan
    spec = {'model': model} if wrapped else model
    with pytest.raises(ValueError, match='truth=.*unfitted'):
        hyp.impute(damaged, model=spec, truth=truth)


def _arc(n=40, seed=0):
    """A smooth, projectile-like trajectory (the tutorial's setting)."""
    t = np.linspace(0, 2, n)
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'x_ft': 3.0 * t + 0.01 * rng.standard_normal(n),
                         'y_ft': 1.5 * t + 0.01 * rng.standard_normal(n),
                         'z_ft': 6.0 + 10.0 * t - 5.0 * t ** 2})


def _damage(truth, occlusion=slice(15, 20), n_scatter=12, seed=1):
    """NaN out a band of whole rows plus scattered single cells.

    Mirrors what the parallel `hyp.tools.damage` helper produces (a
    NaN-damaged copy); this module only needs the NaN mask, so it does not
    depend on that helper.
    """
    rng = np.random.default_rng(seed)
    values = truth.to_numpy(dtype=float).copy()
    values[occlusion, :] = np.nan
    intact = np.flatnonzero(~np.isnan(values))
    values.flat[rng.choice(intact, size=n_scatter, replace=False)] = np.nan
    return pd.DataFrame(values, index=truth.index, columns=truth.columns)


# --- the multi-model list form -------------------------------------------

def test_list_of_models_returns_one_imputation_per_model():
    truth = _arc()
    damaged = _damage(truth)
    with pytest.warns(UserWarning, match='PPCA cannot fill'):
        # PPCA leaves fully-missing rows NaN (GH #169); the list form warns
        # exactly as the single-model call does
        out = hyp.impute(damaged, model=['PPCA', 'KNNImputer'])
    assert isinstance(out, dict)
    assert list(out) == ['PPCA', 'KNNImputer']
    for filled in out.values():
        assert isinstance(filled, pd.DataFrame)
        assert filled.shape == damaged.shape
    assert not np.isnan(out['KNNImputer'].to_numpy()).any()
    single = hyp.impute(damaged, model='KNNImputer')
    assert np.allclose(out['KNNImputer'].to_numpy(), single.to_numpy())
    assert not np.allclose(out['PPCA'].to_numpy(), out['KNNImputer'].to_numpy())


def test_mapping_form_names_the_imputers():
    damaged = _damage(_arc())
    out = hyp.impute(damaged, model={'1-NN': {'model': 'KNNImputer',
                                              'kwargs': {'n_neighbors': 1}},
                                     '9-NN': {'model': 'KNNImputer',
                                              'kwargs': {'n_neighbors': 9}}})
    assert list(out) == ['1-NN', '9-NN']
    assert not np.allclose(out['1-NN'].to_numpy(), out['9-NN'].to_numpy())


def test_list_form_with_return_model_gives_parallel_dicts():
    damaged = _damage(_arc())
    with pytest.warns(UserWarning, match='PPCA cannot fill'):
        filled, models = hyp.impute(damaged, model=['PPCA', 'KNNImputer'],
                                    return_model=True)
    assert list(filled) == list(models) == ['PPCA', 'KNNImputer']
    assert all(isinstance(m, Imputer) and m.is_fitted for m in models.values())


def test_repeated_models_are_numbered():
    damaged = _damage(_arc())
    out = hyp.impute(damaged, model=['KNNImputer', 'KNNImputer'])
    assert list(out) == ['KNNImputer', 'KNNImputer (2)']


# --- scoring against the truth -------------------------------------------

def test_truth_scores_only_the_damaged_cells():
    truth = _arc()
    damaged = _damage(truth)
    scores = hyp.impute(damaged, model=['Kalman', 'KNNImputer'], truth=truth)
    assert list(scores.index) == ['Kalman', 'KNNImputer', 'mean']
    assert list(scores.columns) == ['MAE', 'RMSE', 'MAPE', 'n', 'unscored']
    expected_n = int(np.isnan(damaged.to_numpy()).sum())
    assert (scores['n'] == expected_n).all()
    assert scores.attrs['baseline'] == 'mean'
    assert scores.attrs['best'] in ('Kalman', 'KNNImputer')


def test_observed_cells_never_enter_the_score():
    # every imputer passes observed cells through untouched, so scoring
    # them would dilute the comparison: a fill that is exactly wrong
    # everywhere it was allowed to guess must score >> 0
    truth = _arc()
    damaged = _damage(truth)
    scores = hyp.impute(damaged, model='SimpleImputer', truth=truth)
    n_missing = int(np.isnan(damaged.to_numpy()).sum())
    assert scores.loc['SimpleImputer', 'n'] == n_missing
    assert n_missing < damaged.size  # most cells are observed and unscored


def test_a_perfect_fill_scores_zero():
    # a constant column's observed mean IS its missing value
    truth = pd.DataFrame({'flat': np.full(30, 7.0),
                          'ramp': np.arange(30, dtype=float)})
    damaged = truth.copy()
    damaged.iloc[[4, 11, 22], 0] = np.nan
    damaged.iloc[[5, 12], 1] = np.nan
    scores = hyp.impute(damaged, model='SimpleImputer', truth=truth,
                        per_column=True)
    assert scores.loc[('SimpleImputer', 'flat'), 'MAE'] == pytest.approx(0.0)
    assert scores.loc[('SimpleImputer', 'flat'), 'RMSE'] == pytest.approx(0.0)
    assert scores.loc[('SimpleImputer', 'flat'), 'MAPE'] == pytest.approx(0.0)
    assert scores.loc[('mean', 'flat'), 'MAE'] == pytest.approx(0.0)
    assert scores.loc[('SimpleImputer', 'ramp'), 'MAE'] > 0


def test_kalman_beats_the_column_mean_across_an_occlusion():
    # the tutorial's point: only a temporal model bridges fully-missing rows
    truth = _arc()
    damaged = _damage(truth)
    occluded = np.zeros(truth.shape, dtype=bool)
    occluded[15:20, :] = True
    scores = hyp.impute(damaged, model='Kalman', truth=truth, mask=occluded)
    assert scores.loc['Kalman', 'MAE'] < scores.loc['mean', 'MAE']
    assert scores.attrs['beats_baseline'] is True
    assert scores.loc['Kalman', 'n'] == 15  # 5 frames x 3 axes


def test_mask_partitions_the_damaged_cells():
    truth = _arc()
    damaged = _damage(truth)
    occluded = np.zeros(truth.shape, dtype=bool)
    occluded[15:20, :] = True
    all_scores = hyp.impute(damaged, model='Kalman', truth=truth)
    band = hyp.impute(damaged, model='Kalman', truth=truth, mask=occluded)
    scattered = hyp.impute(damaged, model='Kalman', truth=truth,
                           mask=~occluded)
    assert band.loc['Kalman', 'n'] + scattered.loc['Kalman', 'n'] == \
        all_scores.loc['Kalman', 'n']


def test_per_column_is_the_per_axis_table():
    truth = _arc()
    damaged = _damage(truth)
    wide = hyp.impute(damaged, model=['Kalman'], truth=truth, metrics='rmse')
    long = hyp.impute(damaged, model=['Kalman'], truth=truth, metrics='rmse',
                      per_column=True)
    assert list(long.index.names) == ['model', 'column']
    assert list(long.loc['Kalman'].index) == ['x_ft', 'y_ft', 'z_ft']
    assert long.loc['Kalman', 'RMSE'].mean() == pytest.approx(
        wide.loc['Kalman', 'RMSE'])
    assert long.loc['Kalman', 'n'].sum() == wide.loc['Kalman', 'n']


def test_return_imputed_hands_back_what_was_scored():
    truth = _arc()
    damaged = _damage(truth)
    scores, filled = hyp.impute(damaged, model=['Kalman'], truth=truth,
                                return_imputed=True)
    assert set(filled) == {'Kalman', 'mean', 'truth'}
    assert filled['truth'].shape == truth.shape
    mask = np.isnan(damaged.to_numpy())
    err = np.abs(filled['Kalman'].to_numpy() - truth.to_numpy())
    per_column = [err[mask[:, j], j].mean() for j in range(truth.shape[1])]
    assert scores.loc['Kalman', 'MAE'] == pytest.approx(np.mean(per_column))
    # the baseline fill really is the observed column mean
    means = np.nanmean(damaged.to_numpy(), axis=0)
    assert np.allclose(filled['mean'].to_numpy()[mask],
                       np.broadcast_to(means, mask.shape)[mask])


def test_numpy_truth_and_array_input():
    truth = _arc()
    damaged = _damage(truth)
    with pytest.warns(UserWarning):
        scores = hyp.impute(damaged.to_numpy(), model='PPCA',
                            truth=truth.to_numpy())
    assert list(scores.index) == ['PPCA', 'mean']
    # PPCA cannot fill the 5 fully-missing rows (GH #169): those cells are
    # counted as `unscored` rather than quietly shrinking `n`
    n_damaged = int(np.isnan(damaged.to_numpy()).sum())
    assert scores.loc['PPCA', 'n'] + scores.loc['PPCA', 'unscored'] == n_damaged
    assert scores.loc['PPCA', 'unscored'] == 15  # 5 occluded rows x 3 axes
    assert scores.loc['mean', 'n'] == n_damaged
    assert scores.loc['mean', 'unscored'] == 0


def test_unfilled_cells_are_counted_and_warned_about():
    # PPCA structurally cannot fill a fully-missing row (GH #169). Those
    # cells must not quietly vanish from its score: a model graded on the
    # easy half of the damage would look better than one graded on all of it
    truth = _arc()
    damaged = _damage(truth)
    with pytest.warns(UserWarning, match='not directly comparable'):
        scores = hyp.impute(damaged, model=['PPCA', 'Kalman'], truth=truth)
    assert scores.loc['PPCA', 'unscored'] == 15
    assert scores.loc['Kalman', 'unscored'] == 0
    assert scores.loc['PPCA', 'n'] < scores.loc['Kalman', 'n']


def test_list_of_datasets_adds_a_dataset_level():
    a, b = _arc(seed=0), _arc(seed=2)
    da, db = _damage(a, seed=1), _damage(b, seed=3)
    wide = hyp.impute([da, db], model=['Kalman'], truth=[a, b])
    long = hyp.impute([da, db], model=['Kalman'], truth=[a, b],
                      per_column=True)
    assert list(long.index.names) == ['model', 'dataset', 'column']
    assert long.loc['Kalman', 'MAE'].mean() == pytest.approx(
        wide.loc['Kalman', 'MAE'])
    assert wide.loc['Kalman', 'n'] == int(np.isnan(da.to_numpy()).sum()
                                          + np.isnan(db.to_numpy()).sum())


def test_mape_is_nan_safe_when_truth_has_zeros():
    truth = pd.DataFrame({'x': np.arange(-10.0, 10.0)})
    damaged = truth.copy()
    damaged.iloc[[9, 10, 11], 0] = np.nan  # includes the exact 0.0 entry
    scores = hyp.impute(damaged, model='SimpleImputer', truth=truth)
    assert np.isfinite(scores['MAPE']).all()


def test_accepts_damage_from_the_tools_helper():
    """`hyp.tools.damage` is the intended source of the damaged input; the
    scoring path needs nothing from it but the NaNs it leaves behind."""
    damage = getattr(hyp.tools, 'damage', None)
    if damage is None:  # pragma: no cover - helper lands in a parallel change
        pytest.skip('hyp.tools.damage is not available in this build')
    truth = _arc()
    damaged = damage(truth, frac=0.15, seed=3)
    scores = hyp.impute(damaged, model=['Kalman', 'KNNImputer'], truth=truth)
    assert list(scores.index) == ['Kalman', 'KNNImputer', 'mean']
    assert scores.loc['Kalman', 'n'] == int(
        np.isnan(np.asarray(damaged, dtype=float)).sum())


# --- errors ---------------------------------------------------------------

def test_truth_shape_mismatch_raises():
    truth = _arc()
    damaged = _damage(truth)
    with pytest.raises(ValueError, match='must match cell for cell'):
        hyp.impute(damaged, model='PPCA', truth=truth.iloc[:-3])


def test_mask_without_truth_raises():
    damaged = _damage(_arc())
    with pytest.raises(ValueError, match='mask= only applies'):
        hyp.impute(damaged, model='PPCA', mask=np.ones(damaged.shape, bool))


def test_return_imputed_without_truth_raises():
    damaged = _damage(_arc())
    with pytest.raises(ValueError, match='only applies to a scored'):
        hyp.impute(damaged, model='PPCA', return_imputed=True)


def test_return_model_with_truth_raises():
    truth = _arc()
    with pytest.raises(ValueError, match='not supported with truth'):
        hyp.impute(_damage(truth), model='PPCA', truth=truth,
                   return_model=True)


def test_nothing_to_score_raises():
    truth = _arc()
    damaged = _damage(truth)
    empty_mask = np.zeros(truth.shape, dtype=bool)
    with pytest.raises(ValueError, match='nothing to score'):
        hyp.impute(damaged, model='PPCA', truth=truth, mask=empty_mask)


def test_reserved_baseline_name_raises():
    truth = _arc()
    with pytest.raises(ValueError, match='reserved'):
        hyp.impute(_damage(truth), model={'mean': 'PPCA'}, truth=truth)


def test_unknown_metric_raises():
    truth = _arc()
    with pytest.raises(ValueError, match='unknown metric'):
        hyp.impute(_damage(truth), model='PPCA', truth=truth, metrics='r2')


# --- 1.1 release review: metrics / warning attribution -------------------

def test_repeated_metric_is_rejected_by_name():
    # used to reach build_scores and die with "float() argument must be
    # ... not 'Series'"
    truth = _arc()
    damaged = _damage(truth)
    with pytest.raises(ValueError, match="metric 'mae' is listed more than once"):
        hyp.impute(damaged, model='KNNImputer', truth=truth, metrics=['mae', 'mae'])
    with pytest.raises(ValueError, match="metric 'MAE' is listed more than once"):
        hyp.impute(damaged, model='KNNImputer', truth=truth, metrics=['mae', 'MAE'])


def test_unscored_warning_points_at_the_caller():
    import os
    import hypertools
    package_dir = os.path.dirname(os.path.abspath(hypertools.__file__))
    truth = _arc()
    damaged = _damage(truth)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        scores = hyp.impute(damaged, model=['PPCA', 'Kalman'], truth=truth)
    assert scores.loc['PPCA', 'unscored'] == 15
    unscored = [w for w in caught if 'not directly comparable' in str(w.message)]
    assert len(unscored) == 1
    assert unscored[0].filename == __file__
    assert not unscored[0].filename.startswith(package_dir + os.sep)


# --- incomplete models never win (release red-team 2026-10-09) -------------

def _outlier_arc():
    """A trajectory whose occluded band holds large values: a model that
    skips the band is graded only on the easy scattered cells."""
    t = np.linspace(0, 2, 40)
    truth = pd.DataFrame({'x': 3 * t, 'y': 1.5 * t, 'z': 6 + 10 * t - 5 * t * t})
    truth.iloc[15:20] += 100
    return truth, hyp.damage(truth, rows=slice(15, 20), frac=.1, seed=1)


def test_incomplete_model_never_wins():
    # PPCA cannot fill the 5 occluded rows, so its MAE (0.94 over the 10
    # scattered cells) covers none of the hard ones; SimpleImputer filled
    # all 25 (MAE 63.7). The partial score used to be ranked first.
    truth, damaged = _outlier_arc()
    np.random.seed(0)
    with pytest.warns(UserWarning, match='excluded from the ranking'):
        scores = hyp.impute(damaged, model=['PPCA', 'SimpleImputer'],
                            truth=truth)
    assert scores.loc['PPCA', 'n'] == 10
    assert scores.loc['PPCA', 'unscored'] == 15
    assert scores.loc['SimpleImputer', 'unscored'] == 0
    # the descriptive row is kept, and it IS the smaller number
    assert scores.loc['PPCA', 'MAE'] < scores.loc['SimpleImputer', 'MAE']
    assert scores.attrs['best'] == 'SimpleImputer'
    assert scores.attrs['best_score'] == pytest.approx(
        scores.loc['SimpleImputer', 'MAE'])
    assert scores.attrs['incomplete'] == ['PPCA']
    # SimpleImputer IS the column-mean baseline: equal, so not strictly below
    assert scores.attrs['beats_baseline'] is False


def test_complete_comparison_reports_no_incomplete_models():
    truth = _arc()
    damaged = _damage(truth)
    scores = hyp.impute(damaged, model=['Kalman', 'KNNImputer'], truth=truth)
    assert scores.attrs['incomplete'] == []
    assert scores.attrs['best'] in ('Kalman', 'KNNImputer')
    assert isinstance(scores.attrs['beats_baseline'], bool)


def test_no_complete_model_gives_no_verdict():
    truth, damaged = _outlier_arc()
    np.random.seed(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        scores = hyp.impute(damaged, model='PPCA', truth=truth)
    assert scores.loc['PPCA', 'unscored'] == 15
    assert np.isfinite(scores.loc['PPCA', 'MAE'])  # row still described
    assert scores.attrs['best'] is None
    assert np.isnan(scores.attrs['best_score'])
    assert scores.attrs['beats_baseline'] is None
    assert scores.attrs['incomplete'] == ['PPCA']
    messages = [str(w.message) for w in caught]
    assert any('no model produced every' in m for m in messages)


def test_per_column_verdict_ranks_complete_models_only():
    truth, damaged = _outlier_arc()
    np.random.seed(0)
    with pytest.warns(UserWarning, match='excluded from the ranking'):
        long = hyp.impute(damaged, model=['PPCA', 'SimpleImputer'],
                          truth=truth, per_column=True)
    assert list(long.index.names) == ['model', 'column']
    assert (long.loc['PPCA', 'unscored'] == 5).all()
    assert long.attrs['best'] == 'SimpleImputer'
    assert long.attrs['incomplete'] == ['PPCA']


def test_model_incomplete_on_one_dataset_is_incomplete_for_the_verdict():
    # dataset 0 has scattered damage only (PPCA fills all of it); dataset 1
    # has an occluded band PPCA cannot fill. The verdict averages both, so
    # PPCA's coverage is judged over both.
    truth, occluded = _outlier_arc()
    scattered = hyp.damage(truth, frac=.1, seed=2)
    np.random.seed(0)
    with pytest.warns(UserWarning, match='excluded from the ranking'):
        scores = hyp.impute([scattered, occluded],
                            model=['PPCA', 'SimpleImputer'],
                            truth=[truth, truth])
    np.random.seed(0)
    with pytest.warns(UserWarning):
        long = hyp.impute([scattered, occluded],
                          model=['PPCA', 'SimpleImputer'],
                          truth=[truth, truth], per_column=True)
    assert long.loc[('PPCA', 0), 'unscored'].sum() == 0
    assert long.loc[('PPCA', 1), 'unscored'].sum() == 15
    assert scores.loc['PPCA', 'unscored'] == 15
    assert scores.attrs['incomplete'] == ['PPCA']
    assert scores.attrs['best'] == 'SimpleImputer'
    assert long.attrs['best'] == 'SimpleImputer'
    assert long.attrs['incomplete'] == ['PPCA']


# --- truth=/mask= are aligned by label (release red-team 2026-10-09) -------

def _labelled():
    truth = pd.DataFrame({'a': [1., 2., 3., 4.], 'b': [10., 20., 30., 40.]},
                         index=['r0', 'r1', 'r2', 'r3'])
    damaged = truth.copy()
    damaged.iloc[1, 0] = np.nan
    damaged.iloc[2, 1] = np.nan
    return truth, damaged


def _mae(damaged, truth, **kwargs):
    return float(hyp.impute(damaged, model='SimpleImputer', truth=truth,
                            **kwargs).loc['SimpleImputer', 'MAE'])


def test_truth_is_aligned_to_the_data_by_label():
    truth, damaged = _labelled()
    expected = _mae(damaged, truth)
    assert expected == pytest.approx(11 / 3)          # 3.667
    # reordered columns used to give 18.833, reversed rows 1.833
    assert _mae(damaged, truth[['b', 'a']]) == pytest.approx(expected)
    assert _mae(damaged, truth.iloc[::-1]) == pytest.approx(expected)
    assert _mae(damaged, truth.iloc[::-1][['b', 'a']]) == pytest.approx(expected)


def test_returned_truth_is_in_the_data_order():
    truth, damaged = _labelled()
    _, out = hyp.impute(damaged, model='SimpleImputer',
                        truth=truth.iloc[::-1][['b', 'a']],
                        return_imputed=True)
    pd.testing.assert_frame_equal(out['truth'], truth)


def test_default_integer_labels_are_labels_too():
    # a frame's default 0..n-1 index is still a set of row labels: a
    # reversed truth frame carries them, and is put back in the data's order
    truth, damaged = _labelled()
    truth, damaged = truth.reset_index(drop=True), damaged.reset_index(drop=True)
    assert _mae(damaged, truth.iloc[::-1]) == pytest.approx(11 / 3)


def test_mask_is_aligned_to_the_data_by_label():
    truth, damaged = _labelled()
    mask = pd.DataFrame(False, index=truth.index, columns=truth.columns)
    mask.loc['r1', 'a'] = True                 # score only the (r1, a) cell
    expected = hyp.impute(damaged, model='SimpleImputer', truth=truth,
                          mask=mask)
    assert expected.loc['SimpleImputer', 'n'] == 1
    assert expected.loc['SimpleImputer', 'MAE'] == pytest.approx(2 / 3)
    shuffled = hyp.impute(damaged, model='SimpleImputer', truth=truth,
                          mask=mask.iloc[::-1][['b', 'a']])
    pd.testing.assert_frame_equal(shuffled, expected)


@pytest.mark.parametrize('what', ['truth', 'mask'])
def test_mismatched_labels_are_rejected(what):
    truth, damaged = _labelled()
    mask = pd.DataFrame(True, index=truth.index, columns=truth.columns)
    given = {'truth': truth, 'mask': mask}
    given[what] = given[what].rename(columns={'b': 'c'})
    with pytest.raises(ValueError) as err:
        hyp.impute(damaged, model='SimpleImputer', **given)
    message = str(err.value)
    assert what in message and 'column' in message
    assert "'c'" in message and "'b'" in message
    given = {'truth': truth, 'mask': mask}
    given[what] = given[what].rename(index={'r3': 'r9'})
    with pytest.raises(ValueError) as err:
        hyp.impute(damaged, model='SimpleImputer', **given)
    message = str(err.value)
    assert what in message and 'row' in message
    assert "'r9'" in message and "'r3'" in message


def test_duplicated_labels_that_need_aligning_are_rejected():
    truth, damaged = _labelled()
    dup_truth = truth.rename(index={'r1': 'r0'}).iloc[::-1]
    with pytest.raises(ValueError, match=r"truth.*row.*duplicated.*'r0'"):
        hyp.impute(damaged, model='SimpleImputer', truth=dup_truth)
    dup_data = damaged.rename(columns={'b': 'a'})
    with pytest.raises(ValueError, match=r"column.*duplicated.*'a'"):
        hyp.impute(dup_data, model='SimpleImputer', truth=truth)


def test_identically_labelled_duplicates_stay_positional():
    # repeated labels in the SAME order on both sides need no alignment
    truth, damaged = _labelled()
    relabel = {'r1': 'r0', 'r3': 'r2'}
    assert _mae(damaged.rename(index=relabel),
                truth.rename(index=relabel)) == pytest.approx(11 / 3)


def test_bare_arrays_stay_positional():
    truth, damaged = _labelled()
    expected = 11 / 3
    # array / array, labelled data / array truth, array data / labelled truth
    assert _mae(damaged.to_numpy(), truth.to_numpy()) == pytest.approx(expected)
    assert _mae(damaged, truth.to_numpy()) == pytest.approx(expected)
    assert _mae(damaged.to_numpy(), truth) == pytest.approx(expected)
    # an array carries no labels, so a reordered one IS scored as given
    assert _mae(damaged, truth[['b', 'a']].to_numpy()) == pytest.approx(113 / 6)
    with pytest.raises(ValueError, match='must match cell for cell'):
        hyp.impute(damaged, model='SimpleImputer', truth=truth.to_numpy()[:3])


def test_frames_with_different_kinds_of_labels_are_rejected():
    # data indexed by dates, truth by the default 0..n-1: both are labelled
    # frames whose row labels differ. Guessing "positional" could mis-score
    # silently, so it is an error that names the fix.
    truth, damaged = _labelled()
    dates = pd.date_range('2026-01-01', periods=4)
    dated = damaged.set_axis(dates, axis=0)
    with pytest.raises(ValueError, match=r'(?s)truth.*row labels.*to_numpy'):
        hyp.impute(dated, model='SimpleImputer',
                   truth=truth.reset_index(drop=True))
    assert _mae(dated, truth.to_numpy()) == pytest.approx(11 / 3)
    assert _mae(dated, truth.set_axis(dates, axis=0)) == pytest.approx(11 / 3)


def test_each_dataset_is_aligned_independently():
    truth, damaged = _labelled()
    other = truth.rename(index=lambda r: r.upper()) * 2
    other_damaged = other.copy()
    other_damaged.iloc[0, 1] = np.nan
    expected = hyp.impute([damaged, other_damaged], model='SimpleImputer',
                          truth=[truth, other], per_column=True)
    shuffled = hyp.impute([damaged, other_damaged], model='SimpleImputer',
                          truth=[truth[['b', 'a']], other.iloc[::-1]],
                          per_column=True)
    pd.testing.assert_frame_equal(shuffled, expected)
    with pytest.raises(ValueError, match=r'(?s)truth\[1\].*row'):
        hyp.impute([damaged, other_damaged], model='SimpleImputer',
                   truth=[truth, truth])


def test_polars_truth_is_aligned_by_column_name():
    pl = pytest.importorskip('polars')
    truth, damaged = _labelled()
    # polars frames have column names but no row labels: columns are
    # aligned by name, rows compared by position
    swapped = pl.from_pandas(truth[['b', 'a']].reset_index(drop=True))
    assert _mae(damaged, swapped) == pytest.approx(11 / 3)
    plain = damaged.reset_index(drop=True)
    assert _mae(pl.from_pandas(plain), swapped) == pytest.approx(11 / 3)
    with pytest.raises(ValueError, match=r"(?s)truth.*column.*'c'"):
        hyp.impute(damaged, model='SimpleImputer',
                   truth=swapped.rename({'b': 'c'}))


def test_series_truth_is_aligned_by_row_label():
    truth = pd.DataFrame({'x': [1., 2., 3., 40.]}, index=list('abcd'))
    damaged = truth.copy()
    damaged.iloc[3, 0] = np.nan
    expected = _mae(damaged, truth)
    assert expected == pytest.approx(38.)
    assert _mae(damaged, truth['x'].iloc[::-1]) == pytest.approx(expected)
