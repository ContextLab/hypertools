"""``return_model=True`` reuses the pipeline the figure was drawn with
instead of fitting a second one (1.1 feature-tour report, section 9.8: an
Isomap panel grid warned twice per reducer). Real reducers; the fit count
is observed through sklearn's own connected-components warning."""

import warnings

import numpy as np

import hypertools as hyp


def _digits400():
    d = hyp.load('digits')
    return d.drop(columns='target').to_numpy()[:400]


def _isomap_fits(**kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        out = hyp.plot(_digits400(), '.', reduce='Isomap', show=False, **kw)
    return sum('connected components' in str(m.message) for m in w), out


def test_return_model_fits_the_reducer_once():
    n_plain, _ = _isomap_fits()
    n_bundle, bundle = _isomap_fits(return_model=True)
    assert n_plain == 1
    assert n_bundle == 1
    pipe = bundle['pipeline']
    assert pipe.is_fitted
    # the bundled pipeline reproduces the figure's analyzed data
    replay = np.asarray(pipe.transform(_digits400()))
    np.testing.assert_allclose(replay, np.asarray(bundle['xform_data'][0]),
                               rtol=1e-6, atol=1e-6)


def test_return_model_pipeline_keeps_the_cluster_stage():
    x = hyp.load('random_walk', n_samples=80, n_features=6, random_state=0)
    bundle = hyp.plot(x, '.', reduce='PCA', n_clusters=3, return_model=True,
                      show=False)
    pipe = bundle['pipeline']
    assert list(pipe.named_steps) == ['reduce', 'cluster']
    assert pipe.is_fitted
    labels = np.asarray(pipe.transform(x))
    assert labels.shape[0] == 80 and len(np.unique(labels)) == 3


def test_panels_with_return_model_fit_each_reducer_once():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        bundle = hyp.plot(_digits400(), '.', reduce=['PCA', 'Isomap'],
                          panels=True, return_model=True, show=False)
    assert sum('connected components' in str(m.message) for m in w) == 1
    assert bundle['panels'] == (1, 2)
    assert all(m['pipeline'].is_fitted for m in bundle['panel_models'])
