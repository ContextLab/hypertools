import pandas as pd
import pytest

from hypertools.impute.impute import impute, IMPUTERS
from hypertools.impute.sklearn_imputers import KNNImputer

from .conftest import make_df_with_nans


IMPUTER_NAMES = ('PPCA', 'SimpleImputer', 'KNNImputer', 'IterativeImputer', 'Kalman')


@pytest.mark.parametrize('name', IMPUTER_NAMES)
def test_all_imputer_names_resolve(name):
    if name == 'Kalman':
        pytest.importorskip('pykalman')
    _, missing, _ = make_df_with_nans()
    out = impute(missing, model=name)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == missing.shape


def test_imputer_names_match_registry():
    assert {m.__name__ for m in IMPUTERS} == set(IMPUTER_NAMES)


def test_default_model_is_ppca():
    _, missing, _ = make_df_with_nans()
    out = impute(missing)
    assert not out.isna().any().any()


def test_dict_params_form():
    _, missing, _ = make_df_with_nans()
    out = impute(missing, model={'model': 'KNNImputer', 'params': {'n_neighbors': 3}})
    assert not out.isna().any().any()


def test_dict_args_kwargs_form():
    _, missing, _ = make_df_with_nans()
    out = impute(missing, model={'model': 'KNNImputer', 'args': [], 'kwargs': {'n_neighbors': 3}})
    assert not out.isna().any().any()


def test_class_form():
    _, missing, _ = make_df_with_nans()
    out = impute(missing, model=KNNImputer, n_neighbors=3)
    assert not out.isna().any().any()


def test_instance_form():
    _, missing, _ = make_df_with_nans()
    out = impute(missing, model=KNNImputer(n_neighbors=3))
    assert not out.isna().any().any()


def test_unknown_model_name_lists_options():
    _, missing, _ = make_df_with_nans()
    with pytest.raises(ValueError) as exc_info:
        impute(missing, model='NotARealImputer')
    message = str(exc_info.value)
    assert 'NotARealImputer' in message
    for name in IMPUTER_NAMES:
        assert name in message


def test_list_in_list_out():
    _, missing1, _ = make_df_with_nans(n=50, seed=1)
    _, missing2, _ = make_df_with_nans(n=70, seed=2)
    out = impute([missing1, missing2], model='SimpleImputer')
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0].shape[0] == 50
    assert out[1].shape[0] == 70
    assert not any(o.isna().any().any() for o in out)


# --- return_model round trip: no re-fitting on new data ---------------------

def test_return_model_roundtrip_knn_no_refit(monkeypatch):
    from sklearn.impute import KNNImputer as _SKKNNImputer

    _, missing, _ = make_df_with_nans(n=60, seed=3)
    filled_a, fitted = impute(missing, model='KNNImputer', return_model=True, n_neighbors=3)
    assert isinstance(filled_a, pd.DataFrame)
    original_fitted_imputer = fitted.models_['imputer']

    def _boom(self, *args, **kwargs):
        raise AssertionError('fit() must not be called during transform-reuse (no re-fitting)')

    monkeypatch.setattr(_SKKNNImputer, 'fit', _boom)

    _, missing_b, _ = make_df_with_nans(n=40, seed=4)
    filled_b = impute(missing_b, model=fitted)

    assert isinstance(filled_b, pd.DataFrame)
    assert filled_b.shape == missing_b.shape
    assert not filled_b.isna().any().any()
    # learned parameters are the SAME object -- never rebuilt/re-fit
    assert fitted.models_['imputer'] is original_fitted_imputer


def test_return_model_roundtrip_ppca_reuse_no_refit():
    _, missing, _ = make_df_with_nans(n=100, ncols=10, seed=5, n_missing=15)
    filled_a, fitted = impute(missing, model='PPCA', return_model=True)
    assert isinstance(filled_a, pd.DataFrame)
    original_ppca = fitted.models_['ppca']

    _, missing_b, _ = make_df_with_nans(n=100, ncols=10, seed=6, n_missing=10)
    filled_b = impute(missing_b, model=fitted)

    assert isinstance(filled_b, pd.DataFrame)
    assert filled_b.shape == missing_b.shape
    # learned PPCA model object is the SAME -- never rebuilt/re-fit
    assert fitted.models_['ppca'] is original_ppca


def test_return_model_roundtrip_kalman_no_reestimation(monkeypatch):
    pytest.importorskip('pykalman')
    from pykalman import KalmanFilter

    _, missing, _ = make_df_with_nans(n=70, ncols=3, seed=7, n_missing=8)
    filled_a, fitted = impute(missing, model='Kalman', return_model=True, n_iter=3)
    original_kf = fitted.models_['kf']

    def _boom(self, *args, **kwargs):
        raise AssertionError('em() must not be called during transform-reuse (no re-estimation)')

    monkeypatch.setattr(KalmanFilter, 'em', _boom)

    _, missing_b, _ = make_df_with_nans(n=50, ncols=3, seed=8, n_missing=6)
    filled_b = impute(missing_b, model=fitted)

    assert isinstance(filled_b, pd.DataFrame)
    assert filled_b.shape == missing_b.shape
    assert not filled_b.isna().any().any()
    assert fitted.models_['kf'] is original_kf
