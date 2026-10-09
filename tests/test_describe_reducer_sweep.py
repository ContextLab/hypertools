# -*- coding: utf-8 -*-
"""`hyp.describe` sweeps the DIMENSIONALITY of whatever reducer it is given.

1.1 release red-team: a configured reducer instance (``PCA(n_components=2)``),
a fitted reducer from ``hyp.reduce(..., return_model=True)``, and a dict spec
whose kwargs pin ``n_components`` were all reduced to that ONE dimensionality
at every sweep point, while the curve labelled the points 2..max_dims-1 -- a
flat, false sweep.

Contract under test:

- an UNFITTED instance is cloned for each sweep point with ``n_components``
  set to that point (other settings kept; the caller's instance untouched);
- a dict spec's pinned ``n_components`` is overridden by the sweep;
- a FITTED model, and an instance with no ``n_components`` parameter, raise
  a ``ValueError`` that says what to pass instead;
- no "Unequal values passed to dims and n_components" warnings.

Real scikit-learn models and real `hyp.describe` calls throughout.
"""
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from sklearn.decomposition import PCA, FastICA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import hypertools as hyp  # noqa: E402


def _x():
    return np.random.default_rng(8).normal(size=(60, 6))


def _curve(reduce, **kwargs):
    kwargs.setdefault('max_dims', 6)
    return hyp.describe(_x(), reduce=reduce, show=False, **kwargs)['average']


def _strict(reduce, **kwargs):
    """The curve, with every warning turned into an error."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        return _curve(reduce, **kwargs)


def test_reference_curve_by_name():
    curve = _curve('PCA')
    assert np.allclose(curve, [0.739, 0.852, 0.930, 0.974], atol=5e-4)
    assert np.all(np.diff(curve) > 0)


def test_unfitted_instance_sweeps_like_the_name():
    assert np.allclose(_strict(PCA(n_components=2)), _curve('PCA'))


def test_unfitted_instance_without_a_dimensionality_sweeps_like_the_name():
    assert np.allclose(_strict(PCA()), _curve('PCA'))


def test_caller_instance_is_not_mutated_or_fitted():
    model = PCA(n_components=2, whiten=True)
    _curve(model)
    assert model.n_components == 2
    assert model.whiten is True
    assert not hasattr(model, 'components_')


def test_instance_keeps_its_other_settings():
    """The sweep changes ONLY the dimensionality: a whitened PCA gives a
    different curve from the default one, and the same curve as the dict
    spec that asks for whitening."""
    whitened = _strict(PCA(n_components=2, whiten=True))
    spec = _curve({'model': 'PCA', 'kwargs': {'whiten': True}})
    assert np.allclose(whitened, spec)
    assert not np.allclose(whitened, _curve('PCA'))


def test_bare_class_sweeps_like_the_name():
    assert np.allclose(_strict(PCA), _curve('PCA'))


@pytest.mark.parametrize('spec', [
    {'model': 'PCA', 'kwargs': {'n_components': 2}},
    {'model': PCA, 'kwargs': {'n_components': 2}},
    {'model': PCA(n_components=2)},
], ids=['name', 'class', 'instance'])
def test_dict_spec_pinned_dimensionality_is_overridden_by_the_sweep(spec):
    assert np.allclose(_strict(spec), _curve('PCA'))


def test_dict_spec_is_not_mutated():
    spec = {'model': 'PCA', 'kwargs': {'n_components': 2, 'whiten': True}}
    _curve(spec)
    assert spec == {'model': 'PCA',
                    'kwargs': {'n_components': 2, 'whiten': True}}


def test_legacy_params_spec_pinned_dimensionality_is_overridden():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        # the legacy form's own DeprecationWarning is unrelated to the sweep
        warnings.simplefilter('ignore', DeprecationWarning)
        curve = _curve({'model': 'PCA', 'params': {'n_components': 2}})
    assert np.allclose(curve, _curve('PCA'))


def test_seeded_instance_sweeps_reproducibly():
    a = _curve(FastICA(n_components=2, random_state=0, max_iter=1000))
    b = _curve(FastICA(n_components=2, random_state=0, max_iter=1000))
    assert a == b
    assert len(set(np.round(a, 6))) == len(a)


def test_fitted_reducer_from_return_model_is_rejected():
    x = _x()
    _, fitted = hyp.reduce(x, reduce='PCA', ndims=2, return_model=True)
    with pytest.raises(ValueError, match='already fitted') as err:
        hyp.describe(x, reduce=fitted, max_dims=6, show=False)
    message = str(err.value)
    assert "'kwargs'" in message and 'unfitted' in message
    assert 'Reducer' in message


def test_fitted_sklearn_instance_is_rejected():
    x = _x()
    model = PCA(n_components=2).fit(x)
    with pytest.raises(ValueError, match='already fitted'):
        hyp.describe(x, reduce=model, max_dims=6, show=False)
    with pytest.raises(ValueError, match='already fitted'):
        hyp.describe(x, reduce={'model': model}, max_dims=6, show=False)


def test_fitted_pipeline_from_return_model_is_rejected():
    x = _x()
    _, pipeline = hyp.reduce(x, reduce='PCA', ndims=2, normalize='across',
                             return_model=True)
    assert isinstance(pipeline, hyp.Pipeline)
    with pytest.raises(ValueError, match='already fitted'):
        hyp.describe(x, reduce=pipeline, max_dims=6, show=False)


@pytest.mark.parametrize('make', [
    StandardScaler,
    lambda: hyp.Pipeline([('pca', PCA(n_components=2))]),
], ids=['StandardScaler', 'unfitted-Pipeline'])
def test_instance_whose_dimensionality_cannot_be_set_is_rejected(make):
    with pytest.raises(ValueError, match='n_components') as err:
        hyp.describe(_x(), reduce=make(), max_dims=6, show=False)
    message = str(err.value)
    assert 'cannot' in message and "'kwargs'" in message


def test_rejection_happens_before_any_figure_is_drawn():
    before = plt.get_fignums()
    with pytest.raises(ValueError, match='already fitted'):
        hyp.describe(_x(), reduce=PCA(n_components=2).fit(_x()), max_dims=6)
    assert plt.get_fignums() == before


def test_multiple_datasets_sweep_with_an_instance():
    rng = np.random.default_rng(3)
    data = [rng.normal(size=(40, 6)), rng.normal(size=(40, 6))]
    by_instance = hyp.describe(data, reduce=PCA(n_components=2), max_dims=6,
                               show=False)
    by_name = hyp.describe(data, reduce='PCA', max_dims=6, show=False)
    for key in ('average', 'pooled'):
        assert np.allclose(by_instance[key], by_name[key])
        assert len(set(np.round(by_instance[key], 6))) == 4
    assert np.allclose(by_instance['individual'], by_name['individual'])


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_instance_curve_is_what_both_backends_draw(backend):
    result = hyp.describe(_x(), reduce=PCA(n_components=2), max_dims=6,
                          show=False, backend=backend)
    expected = _curve('PCA')
    if backend == 'plotly':
        import plotly.graph_objects as go
        fig = result['fig']
        assert isinstance(fig, go.Figure)
        xs, ys = list(fig.data[0].x), list(fig.data[0].y)
    else:
        assert isinstance(result['fig'], plt.Figure)
        line = result['fig'].axes[0].lines[0]
        xs, ys = list(line.get_xdata()), list(line.get_ydata())
    assert xs == [2, 3, 4, 5]
    assert np.allclose(ys, expected)
