"""Flat model parameters in a dict spec raise instead of being dropped.

A dict model spec is ``{'model': ..., 'args': [...], 'kwargs': {...}}``
(or the legacy ``{'model': ..., 'params': {...}}``). A FLAT key such as
``{'model': 'PCA', 'whiten': True}`` used to be ignored without a word by
every dispatcher, so the model silently ran with its defaults. d3fefd63
made `hyp.cluster` raise for it; the same check now covers `hyp.reduce`
(and its streaming path), `hyp.manip`, `hyp.align` (and the classic
`hypertools.tools.align`), `hyp.impute`, `hyp.Pipeline`, `hyp.apply_model`
and `hypertools.tools.text2mat` (1.1 review). `hyp.predict` specs are
exempt: they carry flat ``t``/``horizon``/``block`` keys by design.

Each block below first shows, with a real model, that the parameter in
question changes the result when it is nested under 'kwargs' (so a dropped
flat key is a real, silent change of results), then that the flat form
raises a ValueError naming the key and spelling out the corrected spec.

Three sibling silent drops found alongside are covered too: the outer
``**kwargs`` of `hyp.manip`/`hyp.align` next to a dict spec (now merged
into its 'kwargs'), constructor parameters handed to an already-built
aligner (now a UserWarning), and a dict spec's positional 'args' on the
streaming-reduce and text2mat paths (now passed to the constructor).
"""
import numpy as np
import pandas as pd
import pytest

import hypertools as hyp


def _assert_names_and_suggests(err, key, kwargs_repr):
    msg = str(err.value)
    assert f"'{key}'" in msg
    assert f"'kwargs': {kwargs_repr}" in msg
    assert 'unrecognized top-level key' in msg


# --- reduce ----------------------------------------------------------------

def _reduce_data():
    rng = np.random.default_rng(0)
    # features with very different variances, so whitening is visible
    return rng.standard_normal((120, 6)) * np.array([10., 5., 2., 1., .5, .1])


def test_reduce_nested_whiten_changes_result():
    x = _reduce_data()
    plain = hyp.reduce(x, reduce={'model': 'PCA'}, ndims=2)
    white = hyp.reduce(x, reduce={'model': 'PCA', 'kwargs': {'whiten': True}},
                       ndims=2)
    assert not np.allclose(plain, white)
    np.testing.assert_allclose(np.std(white, axis=0, ddof=1), 1.0, rtol=1e-6)


@pytest.mark.parametrize('via', ['reduce', 'reduce-model', 'analyze', 'plot',
                                 'stream', 'describe'])
def test_reduce_flat_spec_raises(via):
    x = _reduce_data()
    spec = {'model': 'PCA', 'whiten': True}
    with pytest.raises(ValueError) as err:
        if via == 'reduce':
            hyp.reduce(x, reduce=spec, ndims=2)
        elif via == 'reduce-model':
            hyp.reduce(x, model=spec, ndims=2)
        elif via == 'analyze':
            hyp.analyze(x, reduce=spec, ndims=2)
        elif via == 'plot':
            hyp.plot(x, '.', reduce=spec, ndims=2, show=False)
        elif via == 'stream':
            hyp.plot(iter(x), reduce=spec, ndims=2, stream_init=60,
                     stream_max=80, show=False)
        else:
            hyp.describe(x, reduce=spec, max_dims=3, show=False)
    _assert_names_and_suggests(err, 'whiten', "{'whiten': True}")


def test_reduce_flat_spec_suggestion_merges_legacy_params():
    x = _reduce_data()
    with pytest.raises(ValueError) as err:
        hyp.reduce(x, reduce={'model': 'PCA', 'params': {'whiten': True},
                              'svd_solver': 'full'}, ndims=2)
    _assert_names_and_suggests(err, 'svd_solver',
                               "{'whiten': True, 'svd_solver': 'full'}")


def test_reduce_stream_nested_whiten_is_honored():
    x = _reduce_data()
    fig = hyp.plot(iter(x), reduce={'model': 'PCA',
                                    'kwargs': {'whiten': True}},
                   ndims=2, stream_init=60, stream_max=80, show=False)
    assert fig.stream_info['reduce_model'].whiten is True


def test_reduce_stream_spec_args_are_honored():
    # the streaming path also dropped a spec's positional 'args', so
    # {'model': 'PCA', 'args': [2]} fit ndims (here 3) components
    x = _reduce_data()
    fig = hyp.plot(iter(x), reduce={'model': 'PCA', 'args': [2]}, ndims=3,
                   stream_init=60, stream_max=80, show=False)
    assert fig.stream_info['reduce_model'].n_components == 2
    assert [np.shape(d) for d in fig.stream_info['xform_data']] == [(80, 2)]


def test_reduce_stream_instance_spec_parameters_warn():
    # an already-constructed estimator inside a streaming dict spec is used
    # as-is; its spec parameters used to vanish without a word
    from sklearn.decomposition import PCA
    x = _reduce_data()
    with pytest.warns(UserWarning, match='already-constructed PCA'):
        fig = hyp.plot(iter(x), reduce={'model': PCA(n_components=2),
                                        'kwargs': {'whiten': True}},
                       ndims=2, stream_init=60, stream_max=80, show=False)
    assert fig.stream_info['reduce_model'].whiten is False


# --- manip -----------------------------------------------------------------

def _manip_data():
    rng = np.random.default_rng(1)
    return np.cumsum(rng.standard_normal((80, 3)), axis=0)


def test_manip_nested_kernel_width_changes_result():
    x = _manip_data()
    default = np.asarray(hyp.manip(x, model={'model': 'Smooth'}))
    wide = np.asarray(hyp.manip(x, model={'model': 'Smooth',
                                          'kwargs': {'kernel_width': 31}}))
    assert not np.allclose(default, wide)


@pytest.mark.parametrize('via', ['manip', 'manip-list', 'manip-cross',
                                 'analyze', 'plot'])
def test_manip_flat_spec_raises(via):
    x = _manip_data()
    spec = {'model': 'Smooth', 'kernel_width': 31}
    with pytest.raises(ValueError) as err:
        if via == 'manip':
            hyp.manip(x, model=spec)
        elif via == 'manip-list':
            hyp.manip(x, model=[spec, 'ZScore'])
        elif via == 'manip-cross':
            hyp.manip(x, model=spec, normalize='across')
        elif via == 'analyze':
            hyp.analyze(x, manip=spec)
        else:
            hyp.plot(x, manip=spec, show=False)
    _assert_names_and_suggests(err, 'kernel_width', "{'kernel_width': 31}")


@pytest.mark.parametrize('cross', [False, True])
def test_manip_outer_kwargs_join_a_dict_spec(cross):
    # hyp.manip(x, model={'model': 'Smooth'}, kernel_width=31) used to drop
    # kernel_width silently (only a bare name/class received **kwargs)
    x = _manip_data()
    extra = {'normalize': 'across'} if cross else {}
    nested = np.asarray(hyp.manip(x, model={'model': 'Smooth',
                                            'kwargs': {'kernel_width': 31}},
                                  **extra))
    outer = np.asarray(hyp.manip(x, model={'model': 'Smooth'},
                                 kernel_width=31, **extra))
    np.testing.assert_allclose(outer, nested)
    # and the outer keyword wins over the spec's own value
    wins = np.asarray(hyp.manip(x, model={'model': 'Smooth',
                                          'kwargs': {'kernel_width': 5}},
                                kernel_width=31, **extra))
    np.testing.assert_allclose(wins, nested)


# --- align -----------------------------------------------------------------

def _align_data():
    rng = np.random.default_rng(2)
    base = np.cumsum(rng.standard_normal((60, 4)), axis=0)
    rotations = [np.linalg.qr(rng.standard_normal((4, 4)))[0]
                 for _ in range(3)]
    return [base @ r + 0.3 * rng.standard_normal((60, 4)) for r in rotations]


def test_align_nested_n_iter_changes_result():
    data = _align_data()
    default = hyp.align(data, model={'model': 'HyperAlign'})
    zero = hyp.align(data, model={'model': 'HyperAlign',
                                  'kwargs': {'n_iter': 0}})
    assert not all(np.allclose(np.asarray(a), np.asarray(b))
                   for a, b in zip(default, zero))


@pytest.mark.parametrize('via', ['align', 'align-cross', 'analyze', 'plot',
                                 'classic'])
def test_align_flat_spec_raises(via):
    data = _align_data()
    spec = {'model': 'HyperAlign', 'n_iter': 0}
    with pytest.raises(ValueError) as err:
        if via == 'align':
            hyp.align(data, model=spec)
        elif via == 'align-cross':
            hyp.align(data, model=spec, reduce='PCA', ndims=2)
        elif via == 'analyze':
            hyp.analyze(data, align=spec)
        elif via == 'plot':
            hyp.plot(data, align=spec, show=False)
        else:
            from hypertools.tools import align as classic_align
            classic_align(data, align={'model': 'hyper', 'n_iter': 0})
    _assert_names_and_suggests(err, 'n_iter', "{'n_iter': 0}")


def _same(a, b):
    return all(np.allclose(np.asarray(i), np.asarray(j))
               for i, j in zip(a, b))


def test_align_outer_kwargs_join_a_dict_spec():
    # hyp.align(data, model={'model': 'HyperAlign'}, n_iter=0) used to drop
    # n_iter silently (only a bare name/class received **kwargs)
    data = _align_data()
    nested = hyp.align(data, model={'model': 'HyperAlign',
                                    'kwargs': {'n_iter': 0}})
    outer = hyp.align(data, model={'model': 'HyperAlign'}, n_iter=0)
    assert _same(outer, nested)
    wins = hyp.align(data, model={'model': 'HyperAlign',
                                  'kwargs': {'n_iter': 10}}, n_iter=0)
    assert _same(wins, nested)


def test_align_instance_ignores_parameters_with_a_warning():
    # an already-constructed aligner cannot take constructor parameters;
    # they used to be dropped without a word, in both spellings
    from hypertools.align import HyperAlign
    data = _align_data()
    expected = hyp.align(data, model=HyperAlign(n_iter=2))
    with pytest.warns(UserWarning, match='already-constructed HyperAlign'):
        out = hyp.align(data, model={'model': HyperAlign(n_iter=2),
                                     'kwargs': {'n_iter': 0}})
    assert _same(out, expected)
    with pytest.warns(UserWarning, match=r"ignoring keyword argument\(s\) "
                                         r"\['n_iter'\]"):
        out = hyp.align(data, model=HyperAlign(n_iter=2), n_iter=0)
    assert _same(out, expected)


# --- impute ----------------------------------------------------------------

def _impute_data():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((60, 4))
    x[:, 1] += 2 * x[:, 0]
    x[rng.random(x.shape) < 0.15] = np.nan
    return pd.DataFrame(x, columns=list('abcd'))


def test_impute_nested_n_neighbors_changes_result():
    x = _impute_data()
    default = hyp.impute(x, model={'model': 'KNNImputer'})
    one = hyp.impute(x, model={'model': 'KNNImputer',
                               'kwargs': {'n_neighbors': 1}})
    assert not np.allclose(np.asarray(default), np.asarray(one))


@pytest.mark.parametrize('via', ['impute', 'impute-truth', 'format_data',
                                 'plot'])
def test_impute_flat_spec_raises(via):
    x = _impute_data()
    spec = {'model': 'KNNImputer', 'n_neighbors': 1}
    with pytest.raises(ValueError) as err:
        if via == 'impute':
            hyp.impute(x, model=spec)
        elif via == 'impute-truth':
            truth = x.fillna(0.0)
            hyp.impute(x, model=spec, truth=truth)
        elif via == 'format_data':
            from hypertools.tools import format_data
            format_data(x, impute=spec)
        else:
            hyp.plot(x, impute=spec, show=False)
    _assert_names_and_suggests(err, 'n_neighbors', "{'n_neighbors': 1}")


def test_impute_name_mapping_is_not_a_spec():
    # a dict with none of the spec keys is a NAME -> spec mapping (GH #285),
    # so its keys are imputer names, never "unrecognized" spec keys
    x = _impute_data()
    out = hyp.impute(x, model={'1-NN': {'model': 'KNNImputer',
                                        'kwargs': {'n_neighbors': 1}}})
    assert list(out) == ['1-NN']


# --- Pipeline / apply_model ------------------------------------------------

def test_pipeline_nested_whiten_changes_result():
    x = _reduce_data()
    plain = hyp.Pipeline([{'model': 'PCA', 'kwargs': {'n_components': 2}}])
    white = hyp.Pipeline([{'model': 'PCA',
                           'kwargs': {'n_components': 2, 'whiten': True}}])
    assert not np.allclose(plain.fit_transform(x), white.fit_transform(x))


@pytest.mark.parametrize('step', [
    {'model': 'PCA', 'kwargs': {'n_components': 2}, 'whiten': True},
    ('pca', {'model': 'PCA', 'kwargs': {'n_components': 2}, 'whiten': True}),
])
def test_pipeline_flat_spec_raises(step):
    with pytest.raises(ValueError) as err:
        hyp.Pipeline([step])
    _assert_names_and_suggests(err, 'whiten',
                               "{'n_components': 2, 'whiten': True}")


def test_apply_model_nested_whiten_changes_result():
    x = _reduce_data()
    plain = hyp.apply_model(x, {'model': 'PCA'}, ndims=2)
    white = hyp.apply_model(x, {'model': 'PCA', 'kwargs': {'whiten': True}},
                            ndims=2)
    assert not np.allclose(plain, white)


@pytest.mark.parametrize('as_list', [False, True])
def test_apply_model_flat_spec_raises(as_list):
    x = _reduce_data()
    spec = {'model': 'PCA', 'whiten': True}
    with pytest.raises(ValueError) as err:
        hyp.apply_model(x, [spec] if as_list else spec, ndims=2)
    _assert_names_and_suggests(err, 'whiten', "{'whiten': True}")


# --- text2mat --------------------------------------------------------------

_TEXTS = [
    'the cat sat on the mat with another cat',
    'dogs and cats are friendly household pets',
    'the stock market fell sharply on monday morning',
    'investors sold shares as the market dropped',
    'a quiet cat naps in the warm afternoon sun',
    'bond yields rose while the market slid lower',
]

# NMF on a six-document toy corpus hits its iteration cap; that is fixture
# noise, not what these tests are about
_nmf_noise = pytest.mark.filterwarnings(
    'ignore::sklearn.exceptions.ConvergenceWarning')


@_nmf_noise
def test_text2mat_nested_vectorizer_kwargs_change_result():
    from hypertools.tools import text2mat
    a = text2mat(_TEXTS, vectorizer={'model': 'CountVectorizer'},
                 semantic={'model': 'NMF', 'kwargs': {'n_components': 2,
                                                      'random_state': 0}})
    b = text2mat(_TEXTS, vectorizer={'model': 'CountVectorizer',
                                     'kwargs': {'max_features': 3}},
                 semantic={'model': 'NMF', 'kwargs': {'n_components': 2,
                                                      'random_state': 0}})
    assert not np.allclose(np.vstack(a), np.vstack(b))


@_nmf_noise
def test_text2mat_spec_args_are_honored():
    # a dict spec's positional 'args' were dropped too: NMF's first
    # positional parameter is n_components, so args=[2] must give 2 columns
    # (the registry default is 20)
    from hypertools.tools import text2mat
    out = text2mat(_TEXTS, vectorizer='CountVectorizer',
                   semantic={'model': 'NMF', 'args': [2],
                             'kwargs': {'random_state': 0,
                                        'max_iter': 1000}})
    assert [np.shape(o) for o in out] == [(len(_TEXTS), 2)]


@pytest.mark.parametrize('which', ['vectorizer', 'semantic'])
def test_text2mat_flat_spec_raises(which):
    from hypertools.tools import text2mat
    vec = {'model': 'CountVectorizer'}
    sem = {'model': 'NMF', 'kwargs': {'n_components': 2}}
    if which == 'vectorizer':
        vec = {'model': 'CountVectorizer', 'max_features': 3}
        key, kw = 'max_features', "{'max_features': 3}"
    else:
        sem = {'model': 'NMF', 'kwargs': {'n_components': 2},
               'random_state': 0}
        key, kw = 'random_state', "{'n_components': 2, 'random_state': 0}"
    with pytest.raises(ValueError) as err:
        text2mat(_TEXTS, vectorizer=vec, semantic=sem)
    _assert_names_and_suggests(err, key, kw)


# --- a misspelled 'model' keeps the dispatcher's own error ------------------

@pytest.mark.parametrize('call', [
    lambda x: hyp.reduce(x, reduce={'mode': 'PCA'}, ndims=2),
    lambda x: hyp.apply_model(x, {'mode': 'PCA'}),
    lambda x: hyp.impute(x, model={'mode': 'PPCA', 'kwargs': {}}),
    # these two used to leak a bare KeyError: 'model'
    lambda x: hyp.Pipeline([{'mode': 'PCA'}]),
    lambda x: hyp.tools.text2mat(_TEXTS,
                                 vectorizer={'mode': 'CountVectorizer'}),
])
def test_missing_model_key_error_wins_over_the_flat_key_check(call):
    # {'mode': 'PCA'} is a typo of 'model', not a flat parameter: the
    # dispatcher's "'model' key" error is the right diagnosis
    with pytest.raises(ValueError, match="'model' key"):
        call(_reduce_data())


# --- predict is exempt -----------------------------------------------------

def test_predict_spec_is_not_checked_here():
    # predict specs legitimately carry flat keys (t/horizon/block in plot's
    # predict= spec); the dispatcher's own spec path is untouched
    x = np.sin(np.arange(60) / 5.0).reshape(-1, 1)
    fc = hyp.predict(x, model={'model': 'AutoRegressor',
                               'kwargs': {'model': 'Ridge', 'lags': 5}}, t=3)
    assert np.asarray(fc).shape == (3, 1)
