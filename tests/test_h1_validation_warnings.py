"""Batch H1-validation-warnings (release-1.0 audit, final polish wave).

Real-call regression tests for the last verified-still-present minor
findings: eager kwarg validation (SRM features, describe max_dims/backend,
apply_model ndims, cluster n_clusters, plot rotations/zoom/transform/fmt/
legend/hue, align 3-D input), warning hygiene (legacy-dict deprecation
attribution, UMAP import noise, ARIMA narrow suppression, HyperAnimation
deletion, Pandas4Warning at the manip stacked-apply site), and API
consistency (cluster plain-int labels, predict-bundle forecast rows,
'hyper' alias deprecation, unknown-model family guidance).
"""
import gc
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

import hypertools as hyp


def _rng(seed=0):
    return np.random.default_rng(seed)


def _walk(seed=0, n=30, d=4):
    return np.cumsum(_rng(seed).standard_normal((n, d)), axis=0)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


# --- item 1: SRM features= validation (X2-019) -----------------------------

@pytest.mark.parametrize('bad', [-1, 0, 2.5, 'three', True])
def test_srm_features_validated(bad):
    x, y = _walk(0), _walk(1)
    with pytest.raises(ValueError, match='features must be a positive'):
        hyp.align([x, y], model='SRM', features=bad)


def test_srm_features_valid_int_still_works():
    x, y = _walk(0), _walk(1)
    aligned = hyp.align([x, y], model='SRM', features=3)
    assert [np.asarray(a).shape for a in aligned] == [(30, 3), (30, 3)]


# --- item 2: describe() validation (X2-017) --------------------------------

@pytest.mark.parametrize('bad', [0, -3, 2, 2.5, True])
def test_describe_max_dims_validated(bad):
    with pytest.raises(ValueError, match='max_dims must be an integer >= 3'):
        hyp.describe(_walk(), max_dims=bad, show=False)


def test_describe_backend_validated_even_without_show():
    with pytest.raises(ValueError, match='backend must be one of'):
        hyp.describe(_walk(), max_dims=4, show=False, backend='bogus')


def test_describe_one_column_data_raises():
    with pytest.raises(ValueError, match='at least 2 features'):
        hyp.describe(_rng().random((20, 1)), show=False)


def test_describe_two_column_default_evaluates_dim_two():
    result = hyp.describe(_rng().random((20, 2)), show=False)
    # the old default (min(shape) == 2) silently produced empty results
    assert len(result['average']) == 1
    assert result['individual'][0]


# --- item 3: apply_model ndims parity with reduce (X2-018) -----------------

def test_apply_model_ndims_zero_raises_like_reduce():
    x = _walk()
    with pytest.raises(ValueError, match='ndims must be >= 1'):
        hyp.apply_model(x, 'PCA', ndims=0)
    with pytest.raises(ValueError, match='ndims must be a positive integer'):
        hyp.apply_model(x, 'PCA', ndims=2.5)
    assert hyp.apply_model(x, 'PCA', ndims=2).shape == (30, 2)


# --- items 4-6: plot kwarg validation (X2-014/-015/-016) -------------------

def test_plot_rotations_string_raises_eagerly():
    with pytest.raises(ValueError, match='rotations must be a number'):
        hyp.plot(_walk(), animate=True, rotations='two', duration=0.2,
                 frame_rate=5, show=False)


def test_plot_rotations_bad_list_entry_raises():
    with pytest.raises(ValueError, match='non-negative numbers'):
        hyp.plot([_walk(0), _walk(1)], '.', animate='morph',
                 rotations=[1, -2, 1], duration=0.2, frame_rate=5,
                 show=False)


@pytest.mark.parametrize('bad', [-1, 0, 'big', None])
def test_plot_zoom_validated(bad):
    with pytest.raises(ValueError, match='zoom must be a positive number'):
        hyp.plot(_walk(), animate=True, zoom=bad, duration=0.2,
                 frame_rate=5, show=False)


def test_plot_transform_non_data_raises_typed_error():
    with pytest.raises(TypeError, match='transform= must be'):
        hyp.plot(_walk(), transform='banana', show=False)


def test_plot_transform_real_data_still_works():
    fig = hyp.plot(_walk(), transform=[_rng().random((30, 3))], show=False)
    assert fig is not None


def test_plot_fmt_non_string_raises_typed_error():
    with pytest.raises(TypeError, match='fmt must be'):
        hyp.plot(_walk(), fmt=123, show=False)
    with pytest.raises(TypeError, match='fmt must be'):
        hyp.plot(_walk(), fmt=['-', 7], show=False)


def test_plot_legend_scalar_int_raises_typed_error():
    with pytest.raises(TypeError, match='legend= must be'):
        hyp.plot(_walk(), legend=7, show=False)


def test_plot_scalar_hue_warns_but_still_draws():
    with pytest.warns(UserWarning, match='single scalar value'):
        fig = hyp.plot(_walk(), 'o', hue='notacolumn', show=False)
    assert fig is not None


# --- item 7: cluster n_clusters pre-validation (X2-012) --------------------

@pytest.mark.parametrize('bad', [0, -2, 2.5, 'three', True])
def test_cluster_n_clusters_validated(bad):
    with pytest.raises(ValueError, match='n_clusters must be an integer'):
        hyp.cluster(_walk(), n_clusters=bad)


# --- item 8: reduce dict-spec unknown model (X7-008 residue) ---------------

def test_reduce_dict_unknown_model_clear_error():
    with pytest.raises(ValueError, match="unknown reduce model 'Bogus'"):
        hyp.reduce(_walk(), reduce={'model': 'Bogus', 'kwargs': {}})


# --- item 9: align rejects 3-D input (X3-004) ------------------------------

def test_align_rejects_3d_stack():
    bad = [_rng(i).standard_normal((2, 10, 3)) for i in range(2)]
    with pytest.raises(ValueError, match='align expects 2-D'):
        hyp.align(bad)


def test_align_rejects_single_3d_array():
    with pytest.raises(ValueError, match='align expects 2-D'):
        hyp.align(_rng().standard_normal((2, 10, 3)))


# --- item 10: model='hyper' DeprecationWarning (X1-020) --------------------

def test_align_hyper_alias_deprecation_warns():
    x, y = _walk(0), _walk(1)
    with pytest.warns(DeprecationWarning, match="deprecated alias for "
                                                "'HyperAlign'"):
        hyp.align([x, y], model='hyper')


def test_align_hyper_alias_deprecation_in_dict_spec():
    x, y = _walk(0), _walk(1)
    with pytest.warns(DeprecationWarning, match="deprecated alias"):
        hyp.align([x, y], model={'model': 'hyper', 'kwargs': {}})


def test_classic_align_shim_and_plot_stage_kwarg_do_not_warn():
    # 'hyper' is deprecated only as hyp.align's model= spelling; the
    # classic hyp.tools.align API (where align='hyper' is the documented
    # DEFAULT) and the plot/analyze align= stage kwarg route through the
    # classic shim and must stay quiet
    x, y = _walk(0), _walk(1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        hyp.tools.align([x, y])
        fig = hyp.plot([x, y], align='hyper', show=False)
        plt.close(fig)
    assert not [m for m in w if issubclass(m.category, DeprecationWarning)]


def test_align_canonical_name_does_not_warn():
    x, y = _walk(0), _walk(1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        hyp.align([x, y], model='HyperAlign')
    assert not [x for x in w if issubclass(x.category, DeprecationWarning)]


# --- item 11: cluster returns plain Python ints (X1-022) -------------------

def test_cluster_labels_are_plain_python_ints():
    labels = hyp.cluster(_walk(), n_clusters=2, random_state=0)
    assert isinstance(labels, list)
    assert all(type(v) is int for v in labels)


# --- item 12: predict-bundle forecasts match hyp.predict (X1-016) ----------

def test_plot_bundle_forecasts_match_hyp_predict_rows():
    tt = np.linspace(0, 6 * np.pi, 150)
    traj = np.column_stack([np.sin(tt), np.cos(tt), tt / 10])
    t = 25
    standalone = hyp.predict(traj, model='Kalman', t=t)
    bundle = hyp.plot(traj, predict='Kalman', t=t, show=False,
                      return_model=True)
    plt.close(bundle['fig'])
    fc = np.asarray(bundle['predict']['forecasts'][0])
    assert standalone.shape == (t, 3)
    assert fc.shape == (t, 3)
    # the drawn overlay is smoothed like any line (PCHIP-densified beyond the
    # raw t+1 vertices), while the returned forecast stays exactly t rows
    ax = bundle['fig'].axes[0]
    assert len(ax.lines[-1].get_xdata()) > t + 1


# --- item 13: legacy-dict deprecation attributed to user code (X4-008) -----

def test_legacy_dict_deprecation_attributed_to_caller_through_plot():
    d = np.vstack([_rng(i).standard_normal((30, 5)) + i * 3
                   for i in range(2)])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        fig = hyp.plot(d, cluster={'model': 'KMeans',
                                   'params': {'n_clusters': 2}}, show=False)
        plt.close(fig)
    dep = [x for x in w if issubclass(x.category, DeprecationWarning)
           and 'params' in str(x.message)]
    assert dep, 'expected the legacy-dict DeprecationWarning'
    # attributed to THIS test file (user code), not a hypertools frame --
    # that is what makes it visible under Python's default filters
    assert all(x.filename == __file__ for x in dep), \
        [x.filename for x in dep]


def test_legacy_dict_deprecation_attributed_to_caller_through_analyze():
    d = _walk()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        hyp.analyze(d, reduce={'model': 'PCA',
                               'params': {'n_components': 2}})
    dep = [x for x in w if issubclass(x.category, DeprecationWarning)
           and 'params' in str(x.message)]
    assert dep and all(x.filename == __file__ for x in dep)


# --- item 16: discarded HyperAnimation no longer scolds (X4-012) -----------

def test_deleted_unrendered_animation_does_not_warn():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        ani = hyp.plot(_walk(), animate=True, duration=0.5, frame_rate=5,
                       show=False)
        del ani
        gc.collect()
    assert not [x for x in w
                if 'deleted without rendering' in str(x.message)]


def test_hyper_animation_save_still_works_after_del_guard(tmp_path):
    ani = hyp.plot(_walk(), animate=True, duration=0.5, frame_rate=5,
                   show=False)
    out = tmp_path / 'anim.gif'
    ani.save(str(out))
    assert out.exists() and out.stat().st_size > 0


# --- item 17: ARIMA suppression narrowed (X4-015) --------------------------

def test_arima_fit_remains_warning_free_on_routine_noise():
    # a short noisy series triggers statsmodels' routine starting-parameter
    # and convergence warnings; those specific ones must stay suppressed
    x = np.cumsum(_rng().standard_normal((25, 2)), axis=0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        fc = hyp.predict(x, model='ARIMA', t=3)
    assert fc.shape == (3, 2)
    leaked = [str(x.message) for x in w
              if 'Non-invertible' in str(x.message)
              or 'Non-stationary' in str(x.message)
              or type(x.message).__name__ in ('ConvergenceWarning',
                                              'ValueWarning')]
    assert not leaked, leaked


def test_arima_fitter_does_not_swallow_unrelated_user_warnings():
    # the suppression must be NARROW: an unrelated UserWarning raised during
    # fit still propagates. Simulate by checking the filter registrations
    # directly (real statsmodels fits do not raise arbitrary warnings on
    # demand): inside the fitter's catch_warnings block, an unrelated
    # message must not match any 'ignore' filter added by the fitter.
    from hypertools.predict.arima import _import_statsmodels_warnings
    categories, messages = _import_statsmodels_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        for category in categories:
            warnings.filterwarnings('ignore', category=category)
        for message in messages:
            warnings.filterwarnings('ignore', message=message,
                                    category=UserWarning)
        warnings.warn('a genuine unrelated data-quality problem',
                      UserWarning)
        warnings.warn('Non-invertible starting MA parameters found.',
                      UserWarning)
    kept = [str(x.message) for x in w]
    assert kept == ['a genuine unrelated data-quality problem']


# --- item 19: apply_model unknown-model family guidance (X1-015) -----------

def test_apply_model_unknown_family_error_points_at_dispatchers():
    with pytest.raises(ValueError) as err:
        hyp.apply_model(_walk(), 'ZScore')
    msg = str(err.value)
    assert 'REDUCE and CLUSTER' in msg
    assert 'hyp.manip' in msg and 'hyp.predict' in msg \
        and 'hyp.impute' in msg and 'hyp.align' in msg


# --- item 20: PPCA rank-deficient data -> clear error (G1) -----------------

def test_ppca_rank_deficient_collinear_data_clear_error():
    # perfectly collinear columns (rank 1) with missing entries: the EM's
    # covariance updates go singular, which used to surface as a raw
    # numpy.linalg.LinAlgError from inside the vendored model
    rng = _rng(1)
    col = rng.standard_normal((40, 1))
    x = np.tile(col, (1, 5))
    x[rng.random(x.shape) < 0.1] = np.nan
    with pytest.raises(ValueError, match='RANK-DEFICIENT') as err:
        hyp.impute(x, model='PPCA', random_state=0)
    assert 'jitter' in str(err.value)


def test_ppca_full_rank_data_still_imputes():
    rng = _rng(2)
    x = rng.standard_normal((60, 5))
    x[rng.random(x.shape) < 0.1] = np.nan
    out = np.asarray(hyp.impute(x, model='PPCA', random_state=0))
    assert not np.isnan(out).any()


# --- item 21: Pandas4Warning filtered at the manip stacked-apply site ------

def test_multi_dataset_manip_emits_no_pandas_copy_deprecation():
    a, b = _rng(0).standard_normal((30, 4)), _rng(1).standard_normal((30, 4))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        out = hyp.manip([a, b], 'ZScore')
    assert len(out) == 2
    assert not [x for x in w if 'copy keyword' in str(x.message)]


# --- item 14/15: warning hygiene spot checks -------------------------------

def test_reduce_umap_first_use_no_tensorflow_import_warning():
    pytest.importorskip('umap')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        out = hyp.reduce(_rng().standard_normal((60, 10)), 'UMAP', ndims=2)
    assert out.shape == (60, 2)
    assert not [x for x in w
                if 'Tensorflow not installed' in str(x.message)]


def test_dispatcher_userwarning_attributed_to_caller():
    # X4-009 sweep spot check: cluster()'s ndims-passthrough warning must
    # point at user code, not a hypertools frame
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        hyp.cluster(_walk(), n_clusters=2, ndims=3, random_state=0)
    hits = [x for x in w if 'passthrough to the reduce stage'
            in str(x.message)]
    assert hits and all(x.filename == __file__ for x in hits)
