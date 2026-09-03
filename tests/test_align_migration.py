"""Tests for the 1.0-pattern migration of hypertools.align (round17 Task 4):
exporting the class-based dispatcher as `hyp.align`, legacy compatibility
(`align='hyper'`/`'SRM'`/etc., `n_iter=`, the deprecated `align=` kwarg
alias, and the legacy `{'model', 'params'}` dict spec), the fit/transform
split on `Aligner` (GH #227 shape validation, never-refit reuse), and
`return_model`/cross-module kwargs. All data is real (small) numeric
arrays/DataFrames -- no mocks.
"""
import contextlib
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

import hypertools as hyp
from hypertools.align.align import align as aligner, ALIGNERS
from hypertools.align.hyperalign import HyperAlign
from hypertools.align.null import NullAlign
from hypertools.align.srm import (SharedResponseModel,
                                   DeterministicSharedResponseModel,
                                   RobustSharedResponseModel)
from hypertools.core.pipeline import Pipeline
from hypertools.tools.align import align as legacy_align


def _rng():
    return np.random.RandomState(0)


def _rotated_pair(seed=0):
    rng = np.random.RandomState(seed)
    base = rng.rand(20, 3)
    rot, _ = np.linalg.qr(rng.rand(3, 3))
    return base, base @ rot


# --- exported dispatcher --------------------------------------------------

def test_hyp_align_is_the_class_based_dispatcher():
    assert hyp.align.__module__ == 'hypertools.align.align'


def test_hyp_align_default_signature():
    import inspect
    sig = inspect.signature(hyp.align)
    for name in ('model', 'return_model', 'manip', 'normalize', 'reduce',
                 'ndims', 'cluster', 'format_data'):
        assert name in sig.parameters
    assert sig.parameters['model'].default == 'HyperAlign'
    assert sig.parameters['return_model'].default is False
    assert sig.parameters['format_data'].default is True


def test_basic_alignment_by_name():
    d1, d2 = _rotated_pair()
    out = hyp.align([pd.DataFrame(d1), pd.DataFrame(d2)], model='HyperAlign')
    assert isinstance(out, list) and len(out) == 2
    assert np.allclose(np.asarray(out[0]), np.asarray(out[1]), rtol=1)


# --- legacy compat: string aliases ('hyper'/'SRM'/etc.) -------------------

@pytest.mark.parametrize('legacy_name,canonical', [
    ('hyper', 'HyperAlign'),
    ('SRM', 'SharedResponseModel'),
])
def test_legacy_string_aliases_resolve_correctly(legacy_name, canonical):
    d1, d2 = _rotated_pair()
    # only 'hyper' is a DEPRECATED alias (it warns); 'SRM' is a supported
    # short name and must stay silent
    if legacy_name == 'hyper':
        ctx = pytest.warns(DeprecationWarning,
                           match="'hyper' is a deprecated alias")
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        out, model = hyp.align([d1, d2], model=legacy_name, return_model=True)
    assert type(model).__name__ == canonical
    assert isinstance(out, list) and len(out) == 2


def test_legacy_n_iter_passthrough_to_hyperalign():
    d1, d2 = _rotated_pair()
    # the legacy 'hyper' alias is exercised deliberately; assert its
    # deprecation notice fires
    with pytest.warns(DeprecationWarning,
                      match="'hyper' is a deprecated alias"):
        out, model = hyp.align([d1, d2], model='hyper', n_iter=3,
                               return_model=True)
    assert model.kwargs['n_iter'] == 3


# --- deprecated align= kwarg alias for model= ------------------------------

def test_deprecated_align_kwarg_alias_warns_and_works():
    d1, d2 = _rotated_pair()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = hyp.align([d1, d2], align='hyper')
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert isinstance(out, list) and len(out) == 2


def test_align_kwarg_alongside_nondefault_model_raises():
    d1, d2 = _rotated_pair()
    with pytest.raises(ValueError, match='both model='):
        hyp.align([d1, d2], model='SRM', align='hyper')


def test_hyp_plot_align_hyper_still_works():
    d1, d2 = _rotated_pair()
    fig = hyp.plot([d1, d2], align='hyper', show=False)
    assert fig is not None


# --- legacy {'model', 'params'} dict spec (and canonical 'kwargs') --------

def test_hyp_plot_align_canonical_dict_with_kwargs_key():
    d1, d2 = _rotated_pair()
    fig = hyp.plot([d1, d2],
                    align={'model': 'HyperAlign', 'kwargs': {'n_iter': 3}},
                    show=False)
    assert fig is not None


def test_hyp_plot_align_legacy_dict_with_params_key():
    d1, d2 = _rotated_pair()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        fig = hyp.plot([d1, d2],
                        align={'model': 'hyper', 'params': {'n_iter': 3}},
                        show=False)
    assert fig is not None


def test_align_dispatcher_dict_spec_with_kwargs_key():
    d1, d2 = _rotated_pair()
    out, model = hyp.align([d1, d2],
                            model={'model': 'HyperAlign', 'kwargs': {'n_iter': 3}},
                            return_model=True)
    assert model.kwargs['n_iter'] == 3


def test_align_dispatcher_legacy_dict_spec_warns():
    d1, d2 = _rotated_pair()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out, model = hyp.align([d1, d2],
                                model={'model': 'HyperAlign', 'params': {'n_iter': 3}},
                                return_model=True)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert model.kwargs['n_iter'] == 3


# --- Aligner.transform: fit/transform split + GH #227 shape validation ----

def test_aligner_transform_fits_on_half_and_applies_to_held_out_half():
    """Real numeric fit/transform split: fit HyperAlign on the first half of
    3 rotated+noisy copies of a common trajectory, transform the held-out
    second halves, and confirm alignment measurably increases cross-dataset
    correlation relative to the (unaligned) held-out data."""
    t = np.linspace(0, 4 * np.pi, 60)
    common = np.stack([np.sin(t), np.cos(t), t / 10], axis=1)

    def rotated_noisy(seed):
        r = np.random.RandomState(seed)
        rot, _ = np.linalg.qr(r.randn(3, 3))
        noise = 0.05 * r.randn(*common.shape)
        return (common + noise) @ rot

    datasets = [rotated_noisy(s) for s in (1, 2, 3)]
    half = 30
    train = [pd.DataFrame(d[:half]) for d in datasets]
    held_out = [pd.DataFrame(d[half:], index=range(half, 2 * half)) for d in datasets]

    model = HyperAlign(n_iter=10)
    model.fit(train)
    aligned_held_out = model.transform(held_out)

    def mean_pairwise_corr(dsets):
        arrs = [np.asarray(d).flatten() for d in dsets]
        corrs = [np.corrcoef(arrs[i], arrs[j])[0, 1]
                 for i in range(len(arrs)) for j in range(i + 1, len(arrs))]
        return np.mean(corrs)

    unaligned_corr = mean_pairwise_corr(held_out)
    aligned_corr = mean_pairwise_corr(aligned_held_out)
    assert aligned_corr > unaligned_corr + 0.3


# --- SRM family: held-out transform (FINDING 1, GH #227) -------------------

def _shared_trajectory_datasets(n_datasets=3, n_rows=90, n_cols=4, noise=0.05):
    """3 rotated+noisy copies (real numeric data, no mocks) of a common
    multi-dimensional trajectory, one per dataset."""
    t = np.linspace(0, 4 * np.pi, n_rows)
    common = np.stack([np.sin(t), np.cos(t), t / 10, np.sin(2 * t)], axis=1)[:, :n_cols]
    out = []
    for seed in range(1, n_datasets + 1):
        r = np.random.RandomState(seed)
        rot, _ = np.linalg.qr(r.randn(n_cols, n_cols))
        out.append((common + noise * r.randn(*common.shape)) @ rot)
    return out


def _mean_pairwise_corr(dsets):
    arrs = [np.asarray(d).flatten() for d in dsets]
    corrs = [np.corrcoef(arrs[i], arrs[j])[0, 1]
             for i in range(len(arrs)) for j in range(i + 1, len(arrs))]
    return np.mean(corrs)


@pytest.mark.parametrize('cls', [SharedResponseModel, DeterministicSharedResponseModel,
                                  RobustSharedResponseModel])
def test_srm_family_transform_applies_to_held_out_data_with_different_row_count(cls):
    """FINDING 1 regression: fit on the first 40 rows of 3 rotated+noisy
    copies of a shared trajectory, transform held-out data with a DIFFERENT
    row count than fit-time. Must not raise (previously raised ValueError:
    'Shape of passed values is (N, k), indices imply (M, k)'), must return
    one row per held-out row, and must be measurably more correlated across
    datasets than the raw (unaligned) held-out data."""
    datasets = _shared_trajectory_datasets(n_rows=90)
    train_n, held_out_n = 40, 25
    train = [pd.DataFrame(d[:train_n]) for d in datasets]
    held_out = [pd.DataFrame(d[train_n:train_n + held_out_n], index=range(held_out_n))
                for d in datasets]

    model = cls(features=3)
    model.fit(train)
    aligned_held_out = model.transform(held_out)

    assert len(aligned_held_out) == len(held_out)
    for a, h in zip(aligned_held_out, held_out):
        assert np.asarray(a).shape[0] == np.asarray(h).shape[0] == held_out_n

    unaligned_corr = _mean_pairwise_corr(held_out)
    aligned_corr = _mean_pairwise_corr(aligned_held_out)
    assert aligned_corr > unaligned_corr + 0.1


@pytest.mark.parametrize('cls', [SharedResponseModel, DeterministicSharedResponseModel,
                                  RobustSharedResponseModel])
def test_srm_family_transform_preserves_new_data_custom_index(cls):
    """FINDING 1 regression (silent-mislabeling case): when the held-out
    row count happens to MATCH fit-time, the output index must still be
    derived from the held-out (new) data's own index, not silently reused
    from the fit-time `indices`."""
    datasets = _shared_trajectory_datasets(n_rows=80)
    train_n = 40
    train = [pd.DataFrame(d[:train_n]) for d in datasets]
    # same row COUNT as fit-time, but a custom, disjoint index
    custom_index = pd.RangeIndex(1000, 1000 + train_n)
    held_out = [pd.DataFrame(d[train_n:train_n + train_n], index=custom_index)
                for d in datasets]

    model = cls(features=3)
    model.fit(train)
    aligned_held_out = model.transform(held_out)

    # `trim_and_pad` preserves the first dataset's index ORDER (F12-align-001
    # fix), so the output index must equal `custom_index` exactly -- both in
    # membership (the GH #227 regression: fit-time indices being reused) and
    # in order.
    for a in aligned_held_out:
        assert list(a.index) == list(custom_index)


def test_aligner_transform_before_fit_raises_not_fitted():
    m = HyperAlign()
    with pytest.raises(NotFittedError):
        m.transform([pd.DataFrame(_rng().rand(5, 2))])


def test_aligner_transform_replays_fit_data_when_no_argument_given():
    d1, d2 = _rotated_pair()
    m = NullAlign()
    m.fit([pd.DataFrame(d1), pd.DataFrame(d2)])
    out = m.transform()
    assert len(out) == 2 and np.asarray(out[0]).shape == (20, 3)


def test_aligner_transform_raises_on_dataset_count_mismatch():
    d1 = pd.DataFrame(_rng().rand(10, 3))
    d2 = pd.DataFrame(_rng().rand(10, 3))
    m = HyperAlign()
    m.fit([d1, d2])
    with pytest.raises(ValueError, match=r'2 dataset'):
        m.transform([d1])


def test_aligner_transform_raises_on_column_count_mismatch():
    d1 = pd.DataFrame(_rng().rand(10, 3))
    d2 = pd.DataFrame(_rng().rand(10, 3))
    d_bad = pd.DataFrame(_rng().rand(10, 5))
    m = HyperAlign()
    m.fit([d1, d2])
    with pytest.raises(ValueError, match=r'column'):
        m.transform([d1, d_bad])


# --- raw numpy new_data must not crash dw.unstack (round17 wave2 HIGH) ----

@pytest.mark.parametrize('model_name', ['HyperAlign', 'SharedResponseModel', 'Procrustes'])
def test_aligner_transform_accepts_raw_numpy_held_out_data(model_name):
    """Regression: `Aligner.transform` must coerce raw numpy array `new_data`
    (not wrapped in DataFrames) to DataFrame(s) before `dw.unstack` -- calling
    `.transform(...)` directly (bypassing the funnel/format_data path that
    `align()` applies before `fit`) with a list of raw numpy arrays used to
    raise `Exception: Unsupported datatype: <class 'list'>`. Held-out data
    has a DIFFERENT row count than fit-time data."""
    datasets = _shared_trajectory_datasets(n_rows=90)
    train_n, held_out_n = 40, 25
    train = [d[:train_n] for d in datasets]  # raw numpy, not DataFrames
    held_out = [d[train_n:train_n + held_out_n] for d in datasets]  # raw numpy

    _, model = aligner(train, model=model_name, return_model=True)
    aligned_held_out = model.transform(held_out)

    assert len(aligned_held_out) == len(held_out)
    for a, h in zip(aligned_held_out, held_out):
        assert np.asarray(a).shape[0] == np.asarray(h).shape[0] == held_out_n

    unaligned_corr = _mean_pairwise_corr(held_out)
    aligned_corr = _mean_pairwise_corr(aligned_held_out)
    assert aligned_corr > unaligned_corr + 0.1


def test_aligner_transform_bare_single_array_wraps_and_validates_count():
    """A single bare 2D array (not a list) passed to `transform` is wrapped
    as a 1-dataset input (mirroring `fit`'s own single-vs-list handling) and
    validated against the fit-time dataset count -- fitting on 3 datasets
    then transforming a single raw array must raise ValueError (dataset-count
    mismatch), not silently succeed or crash inside `dw.unstack`."""
    datasets = _shared_trajectory_datasets(n_datasets=3, n_rows=30)
    model = HyperAlign()
    model.fit([pd.DataFrame(d) for d in datasets])
    with pytest.raises(ValueError, match=r'3 dataset'):
        model.transform(datasets[0])


def test_aligner_is_fitted_property():
    m = HyperAlign()
    assert m.is_fitted is False
    m.fit([pd.DataFrame(_rng().rand(6, 2)), pd.DataFrame(_rng().rand(6, 2))])
    assert m.is_fitted is True


# --- return_model: single-stage and cross-module pipeline -----------------

def test_return_model_single_stage_returns_fitted_aligner():
    d1, d2 = _rotated_pair()
    out, model = hyp.align([d1, d2], model='HyperAlign', return_model=True)
    assert isinstance(model, HyperAlign)
    assert model.is_fitted


def test_return_model_cross_kwargs_returns_pipeline():
    # NB: cluster-stage parameters must be passed via the cluster stage's
    # dict spec. This test previously passed a bare `n_clusters=2`, which
    # was silently swallowed by HyperAlign's constructor and NEVER reached
    # KMeans (verified: the fitted cluster step used the default); unknown
    # align-model kwargs now raise TypeError (X2-error-quality-003).
    d1, d2 = _rotated_pair()
    out, model = hyp.align([d1, d2], model='HyperAlign',
                            cluster={'model': 'KMeans',
                                     'kwargs': {'n_clusters': 2}},
                            return_model=True)
    assert isinstance(model, Pipeline)


# --- fitted-model reuse: never refit (poison-pill) -------------------------

def test_fitted_aligner_passed_as_model_is_reused_never_refit():
    """Poison-pill: monkeypatch the fitted model's `fit` to raise, then
    confirm passing it back in as `model=` does NOT call `fit` again."""
    d1, d2 = _rotated_pair()
    fitted = HyperAlign(n_iter=3)
    fitted.fit([pd.DataFrame(d1), pd.DataFrame(d2)])

    def _poison(*args, **kwargs):
        raise AssertionError('fit() must not be called on an already-fitted '
                              'Aligner passed back in as model=')

    fitted.fit = _poison

    out, model = hyp.align([d1, d2], model=fitted, return_model=True)
    assert model is fitted
    assert isinstance(out, list) and len(out) == 2


def test_fitted_aligner_reused_produces_same_result_as_direct_transform():
    d1, d2 = _rotated_pair()
    d3, d4 = _rotated_pair(seed=1)
    fitted = HyperAlign(n_iter=5)
    fitted.fit([pd.DataFrame(d1), pd.DataFrame(d2)])

    direct = fitted.transform([pd.DataFrame(d3), pd.DataFrame(d4)])
    via_dispatcher = hyp.align([d3, d4], model=fitted)

    for a, b in zip(direct, via_dispatcher):
        assert np.allclose(np.asarray(a), np.asarray(b))


# --- registry / basic sanity -----------------------------------------------

def test_aligners_registry_unchanged():
    names = {cls.__name__ for cls in ALIGNERS}
    assert names == {'HyperAlign', 'SharedResponseModel',
                      'DeterministicSharedResponseModel',
                      'RobustSharedResponseModel', 'Procrustes', 'NullAlign'}


def test_model_none_returns_data_unchanged():
    d1, d2 = _rotated_pair()
    out, model = hyp.align([d1, d2], model=None, return_model=True)
    assert model is None


# --- legacy shim (hypertools.tools.align) still works unchanged -----------

def test_legacy_shim_still_exported_and_callable():
    d1, d2 = _rotated_pair()
    out = legacy_align([d1, d2], align='hyper')
    assert isinstance(out, list) and len(out) == 2
    assert all(isinstance(o, np.ndarray) for o in out)


def test_legacy_shim_srm_alias():
    d1, d2 = _rotated_pair()
    out = legacy_align([d1, d2], align='SRM')
    assert isinstance(out, list) and len(out) == 2
