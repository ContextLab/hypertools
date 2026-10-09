# -*- coding: utf-8 -*-
"""`sklearn.base.clone(hyp.Pipeline(...))` works (1.1 release red-team).

`hyp.Pipeline` subclasses scikit-learn's `BaseEstimator` (so it has
`get_params`/`set_params`), but its constructor resolves and names its
steps, so scikit-learn's default clone -- which rebuilds the object from
`get_params()` and requires the constructor to store each argument
untouched -- raised ``RuntimeError: Cannot clone object ... as the
constructor either does not set or modifies parameter steps`` (also in
1.0.0).

Real scikit-learn models and real hypertools stages throughout.
"""
import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

import hypertools as hyp
from hypertools.core.pipeline import build_pipeline


def _x(seed=0, shape=(50, 6)):
    return np.random.default_rng(seed).normal(size=shape)


def test_clone_of_the_reported_pipeline_works():
    pipe = hyp.Pipeline([('pca', PCA(n_components=2))])
    copy = clone(pipe)
    assert isinstance(copy, hyp.Pipeline)
    assert [name for name, _ in copy.steps] == ['pca']
    assert isinstance(copy.named_steps['pca'], PCA)
    assert copy.named_steps['pca'].n_components == 2
    assert copy.named_steps['pca'] is not pipe.named_steps['pca']


def test_clone_of_a_fitted_pipeline_is_unfitted_and_independent():
    x = _x()
    pipe = hyp.Pipeline([('scale', StandardScaler()),
                         ('pca', PCA(n_components=2))])
    out = pipe.fit_transform(x)
    assert pipe.is_fitted
    copy = clone(pipe)
    assert not copy.is_fitted
    assert not hasattr(copy.named_steps['pca'], 'components_')
    assert not hasattr(copy.named_steps['scale'], 'mean_')
    # the original is untouched by cloning, and by fitting the clone
    other = _x(seed=1)
    copy_out = copy.fit_transform(other)
    assert copy.is_fitted and pipe.is_fitted
    assert np.allclose(pipe.transform(x), out)
    assert not np.allclose(copy.named_steps['pca'].components_,
                           pipe.named_steps['pca'].components_)
    # the clone behaves like a freshly built pipeline
    fresh = hyp.Pipeline([('scale', StandardScaler()),
                          ('pca', PCA(n_components=2))])
    assert np.allclose(copy_out, fresh.fit_transform(other))


def test_clone_keeps_step_names_order_and_settings():
    pipe = hyp.Pipeline(['ZScore', ('reduce', PCA(n_components=3, whiten=True)),
                         {'model': 'PCA', 'kwargs': {'n_components': 2}}])
    copy = clone(pipe)
    assert [n for n, _ in copy.steps] == [n for n, _ in pipe.steps]
    assert [type(m) for _, m in copy.steps] == [type(m) for _, m in pipe.steps]
    assert copy.named_steps['reduce'].get_params() == \
        pipe.named_steps['reduce'].get_params()
    x = _x()
    assert np.allclose(np.asarray(copy.fit_transform(x)),
                       np.asarray(pipe.fit_transform(x)))


def test_clone_of_a_nested_pipeline():
    inner = hyp.Pipeline([('pca', PCA(n_components=3))])
    pipe = hyp.Pipeline([('inner', inner), ('last', PCA(n_components=2))])
    x = _x()
    pipe.fit(x)
    copy = clone(pipe)
    assert not copy.is_fitted
    assert isinstance(copy.named_steps['inner'], hyp.Pipeline)
    assert copy.named_steps['inner'] is not inner
    assert not copy.named_steps['inner'].is_fitted
    assert np.allclose(copy.fit_transform(x), pipe.transform(x))


def test_clone_of_a_dispatcher_pipeline_is_unfitted_and_refits():
    """The Pipeline `return_model=True` hands back for a multi-stage call."""
    x = _x()
    reduced, pipe = hyp.reduce(x, reduce='PCA', ndims=2, normalize='across',
                               return_model=True)
    assert isinstance(pipe, hyp.Pipeline) and pipe.is_fitted
    copy = clone(pipe)
    assert [n for n, _ in copy.steps] == [n for n, _ in pipe.steps]
    assert not copy.is_fitted
    with pytest.raises(NotFittedError):
        copy.transform(x)
    assert np.allclose(np.asarray(copy.fit_transform(x)), np.asarray(reduced))
    # fitting another clone on other data leaves the original's fit alone
    clone(pipe).fit(_x(seed=5))
    assert np.allclose(np.asarray(pipe.transform(x)), np.asarray(reduced))


def test_clone_of_a_dispatcher_pipeline_does_not_share_a_model_instance():
    x = _x()
    model = PCA(n_components=2)
    pipe = build_pipeline(normalize='across', reduce=model)
    expected = np.asarray(pipe.fit_transform(x))
    copy = clone(pipe)
    copy.fit(_x(seed=7))
    # the caller's instance was fitted by `pipe` on x, and the clone's fit
    # on other data did not overwrite it
    assert np.allclose(np.asarray(pipe.transform(x)), expected)
    assert np.allclose(model.transform(
        np.asarray(hyp.normalize(x, normalize='across'))), expected)


def test_clone_of_a_pipeline_reusing_a_fitted_reducer_refits_it():
    """A dispatcher pipeline whose reduce stage REUSES a fitted `Reducer`:
    the clone's stage is that model unfitted, so it is fit on the clone's
    own data."""
    x, other = _x(), _x(seed=9)
    _, fitted = hyp.reduce(x, reduce='PCA', ndims=3, return_model=True)
    pipe = build_pipeline(normalize='across', reduce=fitted, ndims=3)
    pipe.fit(x)
    copy = clone(pipe)
    assert not copy.is_fitted
    out = np.asarray(copy.fit_transform(other))
    fresh = build_pipeline(normalize='across', reduce='PCA', ndims=3)
    assert np.allclose(out, np.asarray(fresh.fit_transform(other)))
    # the reused model was not refit by the clone
    assert np.allclose(np.asarray(hyp.reduce(x, reduce=fitted)),
                       np.asarray(hyp.reduce(x, reduce='PCA', ndims=3)))


def test_apply_model_returns_one_fitted_pipeline_per_dataset():
    """`hyp.apply_model(..., stack=False)` clones the model for each
    dataset. While a `Pipeline` could not be cloned it silently reused the
    ONE object, so every returned "model" was the same pipeline, holding
    only the last dataset's fit."""
    data = [_x(seed=s, shape=(40, 6)) for s in (1, 2, 3)]
    pipe = hyp.Pipeline([('pca', PCA(n_components=2))])
    reduced, models = hyp.apply_model(data, pipe, stack=False,
                                      return_model=True)
    assert len({id(m) for m in models}) == 3
    assert not pipe.is_fitted
    for dataset, out, model in zip(data, reduced, models):
        assert model.is_fitted
        assert np.allclose(np.asarray(out),
                           PCA(n_components=2).fit_transform(dataset))
        # each returned model reproduces ITS dataset's result
        assert np.allclose(np.asarray(model.transform(dataset)),
                           np.asarray(out))


def test_set_params_rejects_unknown_and_nested_names():
    pipe = hyp.Pipeline([('pca', PCA(n_components=2))])
    with pytest.raises(ValueError, match='named_steps'):
        pipe.set_params(pca__n_components=3)
    with pytest.raises(ValueError, match="'bogus'"):
        pipe.set_params(bogus=1)
    assert pipe.named_steps['pca'].n_components == 2


def test_clone_keeps_input_hierarchy_as_an_independent_copy():
    hierarchy = {'axis': 'columns', 'n_features': 3}
    pipe = hyp.Pipeline([('pca', PCA(n_components=2))],
                        input_hierarchy=hierarchy)
    copy = clone(pipe)
    assert copy.input_hierarchy == pipe.input_hierarchy
    assert copy.input_hierarchy is not pipe.input_hierarchy


def test_clone_of_an_empty_pipeline():
    copy = clone(hyp.Pipeline([]))
    assert copy.steps == []


def test_get_params_set_params_round_trip():
    pipe = hyp.Pipeline([('scale', StandardScaler()),
                         ('pca', PCA(n_components=2))])
    params = pipe.get_params(deep=False)
    assert set(params) == {'steps', 'input_hierarchy'}
    assert params['steps'] is pipe.steps
    assert pipe.set_params(**params) is pipe
    assert pipe.get_params(deep=False)['steps'] == params['steps']
    x = _x()
    fresh = hyp.Pipeline([('scale', StandardScaler()),
                          ('pca', PCA(n_components=2))])
    assert np.allclose(pipe.fit_transform(x), fresh.fit_transform(x))


def test_set_params_steps_are_resolved_and_named_like_the_constructor():
    x = _x()
    pipe = hyp.Pipeline([('pca', PCA(n_components=2))])
    pipe.fit(x)
    pipe.set_params(steps=['ZScore', {'model': 'PCA',
                                      'kwargs': {'n_components': 3}}])
    assert [n for n, _ in pipe.steps] == ['zscore', 'pca']
    assert isinstance(pipe.named_steps['pca'], PCA)
    # new steps: the pipeline's own earlier fit no longer counts
    assert not pipe.is_fitted
    assert np.asarray(pipe.fit_transform(x)).shape == (50, 3)
    with pytest.raises(TypeError, match='steps must be a list'):
        pipe.set_params(steps='PCA')
    with pytest.raises(ValueError, match='unique'):
        pipe.set_params(steps=[('a', 'PCA'), ('a', 'ZScore')])


def test_clone_then_pickle_round_trip():
    x = _x()
    _, pipe = hyp.reduce(x, reduce='PCA', ndims=2, normalize='across',
                         return_model=True)
    # round trip of an object built in this test (trusted bytes)
    copy = pickle.loads(pickle.dumps(clone(pipe)))
    assert not copy.is_fitted
    assert np.asarray(copy.fit_transform(x)).shape == (50, 2)
