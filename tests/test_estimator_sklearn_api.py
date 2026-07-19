"""Every model handed back by a hypertools dispatcher (``return_model=True``)
must honor the scikit-learn estimator protocol: ``get_params()`` /
``set_params()`` must work (no ``AttributeError``), and ``clone()`` must
round-trip (except the public ``Pipeline``, whose ``__init__`` resolves specs
to instances -- a documented clone limitation).

Regression coverage for QC 2026-07: ``hyp.align(..., model='Procrustes',
return_model=True)`` returned an aligner whose ``get_params()`` raised
``AttributeError: 'Procrustes' object has no attribute 'target'`` because the
aligner ``__init__`` folded its declared parameters into a single
``self.kwargs`` dict instead of storing each as its own attribute. The sweep
that found it also caught the same class of bug in ``Normalizer`` (was not a
``BaseEstimator`` at all) and a ``clone()`` round-trip failure in
``Reducer``/``Clusterer`` (they copied ``params`` in ``__init__``).

All data is real (small) numeric arrays -- no mocks.
"""
import numpy as np
import pytest
from sklearn.base import BaseEstimator, clone

import hypertools as hyp


def _rng():
    return np.random.default_rng(0)


def _single():
    return _rng().normal(size=(60, 6))


def _list():
    r = _rng()
    return [r.normal(size=(30, 5)), r.normal(size=(30, 5))]


# --- Jeremy's exact reported case --------------------------------------

def test_procrustes_return_model_get_params_does_not_crash():
    x = np.cumsum(_rng().normal(size=(100, 3)), axis=0)
    y = np.cumsum(_rng().normal(size=(100, 3)), axis=0)
    _, xform = hyp.align([x, y], model='Procrustes', return_model=True)
    params = xform.get_params()  # used to raise AttributeError: no attribute 'target'
    # every declared Procrustes __init__ parameter is present and readable
    for name in ('target', 'scaling', 'reflection', 'reduction', 'oblique',
                 'oblique_rcond', 'index'):
        assert name in params
        assert getattr(xform, name) == params[name]


# --- get_params / set_params / clone across every returned model -------

def _returned_models():
    """(label, fitted_model, expects_clone) for each dispatcher."""
    x, xl = _single(), _list()
    xi = x.copy()
    xi[_rng().random(x.shape) < 0.1] = np.nan
    cases = []
    _, m = hyp.reduce(x, reduce='PCA', ndims=2, return_model=True)
    cases.append(('reduce/PCA', m, True))
    _, m = hyp.cluster(x, cluster='KMeans', n_clusters=3, return_model=True)
    cases.append(('cluster/KMeans', m, True))
    for am in ('Procrustes', 'HyperAlign', 'SharedResponseModel'):
        _, m = hyp.align(xl, model=am, return_model=True)
        cases.append((f'align/{am}', m, True))
    _, m = hyp.manip(x, model='ZScore', return_model=True)
    cases.append(('manip/ZScore', m, True))
    _, m = hyp.normalize(x, normalize='across', return_model=True)
    cases.append(('normalize/across', m, True))
    _, m = hyp.predict(np.cumsum(x[:, :2], axis=0), model='Kalman', t=5,
                       return_model=True)
    cases.append(('predict/Kalman', m, True))
    _, m = hyp.impute(xi, model='PPCA', return_model=True)
    cases.append(('impute/PPCA', m, True))
    return cases


@pytest.mark.parametrize('label,model,expects_clone', _returned_models(),
                         ids=lambda v: v if isinstance(v, str) else '')
def test_returned_model_is_sklearn_estimator(label, model, expects_clone):
    assert isinstance(model, BaseEstimator), f'{label} is not a BaseEstimator'
    params = model.get_params()          # must not raise
    assert isinstance(params, dict)
    # get_params keys are all readable as attributes (the sklearn contract)
    for name in model._get_param_names():
        getattr(model, name)             # must not raise AttributeError
    if expects_clone:
        clone(model)                     # must not raise RuntimeError


def test_normalizer_is_baseestimator_with_get_params():
    _, m = hyp.normalize(_single(), normalize='within', return_model=True)
    assert isinstance(m, BaseEstimator)
    assert m.get_params() == {'normalize': 'within'}


def test_reducer_clone_round_trips_after_params_verbatim_fix():
    _, m = hyp.reduce(_single(), reduce='PCA', ndims=2, return_model=True)
    fresh = clone(m)                     # used to raise RuntimeError
    assert fresh.get_params()['params'] == m.get_params()['params']


def test_clusterer_clone_round_trips_after_params_verbatim_fix():
    _, m = hyp.cluster(_single(), cluster='KMeans', n_clusters=4,
                       return_model=True)
    clone(m)                             # used to raise RuntimeError


# --- set_params actually changes what fit/transform use (aligners) ------

def test_aligner_set_params_propagates_to_fit_kwargs():
    xl = _list()
    _, m = hyp.align(xl, model='Procrustes', return_model=True)
    fresh = clone(m)
    fresh.set_params(scaling=False)
    assert fresh.scaling is False
    # the kwargs property (what fit/transform forward to the fitter) reflects it
    assert fresh.kwargs['scaling'] is False
