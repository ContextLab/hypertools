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
import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.impute.common import Imputer


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
