"""`hyp.plot(x, pipeline=p)` replays hand-built `hyp.Pipeline`s, not only
the dispatcher pipelines `hyp.analyze`/`hyp.plot` build.

Before this, a pipeline whose steps were raw scikit-learn estimators (e.g.
``hyp.Pipeline([('reduce', PCA(n_components=3))])``, fitted or not) died in
`Pipeline._reraise_with_list_hint`'s "does not accept a list of datasets"
TypeError (plot always hands the pipeline a LIST of datasets), an unfitted
pipeline died in NotFittedError, a pipeline assembled from an already-fitted
PCA was not `is_fitted`, and a bare fitted `Reducer` handed in as pipeline=
crashed inside scikit-learn ("Found array with dim 3"). Real plot calls
throughout: MPLBACKEND=Agg, show=False.
"""
import os

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError

import hypertools as hyp
from hypertools.core.pipeline import build_pipeline

os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib  # noqa: E402

matplotlib.use('Agg')


def _data():
    rng = np.random.default_rng(0)
    return rng.standard_normal((60, 10)), rng.standard_normal((40, 10))


def _line_counts(fig):
    """Row count of every drawn 3-D line (pre-interpolation is not
    recoverable from the artist, so the bundle's xform_data carries the
    shape assertion; here we only check the figure drew something 3-D)."""
    ax = fig.axes[0]
    return [len(line.get_data_3d()[0]) for line in ax.lines]


def _bundle(data, pipeline):
    bundle = hyp.plot(data, pipeline=pipeline, show=False, return_model=True)
    assert set(bundle) >= {'fig', 'xform_data', 'pipeline'}
    assert bundle['fig'].axes[0].name == '3d'
    assert all(n > 0 for n in _line_counts(bundle['fig']))
    return bundle


# (a) the fitted Reducer that hyp.reduce(..., return_model=True) hands back
def test_bare_fitted_reducer_is_a_one_step_pipeline():
    x, y = _data()
    _, reducer = hyp.reduce(x, model='PCA', ndims=3, return_model=True)
    bundle = _bundle(y, reducer)
    assert [a.shape for a in bundle['xform_data']] == [(40, 3)]
    assert np.allclose(bundle['xform_data'][0], reducer.transform(y))
    assert isinstance(bundle['pipeline'], hyp.Pipeline)
    assert bundle['pipeline'].is_fitted
    assert bundle['pipeline'].steps[0][1] is reducer


# (b) a hand-built Pipeline around an UNFITTED raw sklearn PCA
def test_unfitted_raw_sklearn_pipeline_is_fit_on_x():
    x, y = _data()
    pipe = hyp.Pipeline([('reduce', PCA(n_components=3))])
    assert pipe.is_fitted is False
    bundle = _bundle(x, pipe)
    assert [a.shape for a in bundle['xform_data']] == [(60, 3)]
    assert bundle['pipeline'] is pipe
    assert pipe.is_fitted is True
    # the fit was on x: the very same PCA now projects y identically
    pca = pipe.named_steps['reduce']
    assert np.allclose(bundle['xform_data'][0], pca.transform(x))
    replay = _bundle(y, pipe)
    assert np.allclose(replay['xform_data'][0], pca.transform(y))


def test_unfitted_raw_sklearn_pipeline_fits_on_stacked_rows_of_a_list():
    x, y = _data()
    pipe = hyp.Pipeline([('reduce', PCA(n_components=3))])
    bundle = _bundle([x, y], pipe)
    assert [a.shape for a in bundle['xform_data']] == [(60, 3), (40, 3)]
    reference = PCA(n_components=3).fit(np.vstack([x, y]))
    assert np.allclose(np.abs(bundle['xform_data'][0]),
                       np.abs(reference.transform(x)))
    assert np.allclose(np.abs(bundle['xform_data'][1]),
                       np.abs(reference.transform(y)))


# (c) a hand-built Pipeline around an already-FITTED raw sklearn PCA
def test_fitted_raw_sklearn_pipeline_transforms_without_refit():
    x, y = _data()
    pca = PCA(n_components=3).fit(x)
    components_before = pca.components_.copy()
    pipe = hyp.Pipeline([('reduce', pca)])
    assert pipe.is_fitted is True, "a pipeline of fitted steps is fitted"
    assert np.allclose(pipe.transform(y), pca.transform(y))
    bundle = _bundle(y, pipe)
    assert [a.shape for a in bundle['xform_data']] == [(40, 3)]
    assert np.allclose(bundle['xform_data'][0], pca.transform(y))
    assert np.array_equal(pca.components_, components_before), "never refit"


def test_pipeline_fit_transformed_by_hand_replays():
    x, y = _data()
    pipe = hyp.Pipeline(['ZScore', {'model': 'PCA', 'kwargs': {'n_components': 3}}])
    pipe.fit_transform(x)
    bundle = _bundle(y, pipe)
    assert [a.shape for a in bundle['xform_data']] == [(40, 3)]
    assert np.allclose(bundle['xform_data'][0], pipe.transform(y))


# (d) hypertools' own dispatcher pipelines (build_pipeline / analyze / plot)
def test_dispatcher_pipeline_unfitted_and_fitted():
    x, y = _data()
    unfitted = build_pipeline(reduce='PCA', ndims=3)
    assert unfitted.is_fitted is False
    bundle = _bundle(x, unfitted)
    assert [a.shape for a in bundle['xform_data']] == [(60, 3)]
    assert unfitted.is_fitted is True

    _, fitted = hyp.analyze(x, reduce='PCA', ndims=3, return_model=True)
    bundle = _bundle(y, fitted)
    assert [a.shape for a in bundle['xform_data']] == [(40, 3)]
    assert np.allclose(bundle['xform_data'][0], np.asarray(fitted.transform(y)))

    plotted = hyp.plot(x, reduce='PCA', ndims=3, show=False, return_model=True)
    bundle = _bundle(y, plotted['pipeline'])
    assert [a.shape for a in bundle['xform_data']] == [(40, 3)]


def test_non_pipeline_object_raises_clear_type_error():
    x, _ = _data()
    with pytest.raises(TypeError, match=r"pipeline= expects a hypertools\.Pipeline"):
        hyp.plot(x, pipeline='PCA', show=False)


def test_pipeline_is_fitted_from_fitted_steps_only_when_all_are():
    x, _ = _data()
    fitted = PCA(n_components=3).fit(x)
    assert hyp.Pipeline([fitted]).is_fitted is True
    assert hyp.Pipeline([fitted, PCA(n_components=2)]).is_fitted is False
    with pytest.raises(NotFittedError):
        hyp.Pipeline([fitted, PCA(n_components=2)]).transform(x)


def test_pipeline_ending_in_labels_raises_clear_error():
    x, _ = _data()
    from sklearn.cluster import DBSCAN
    pipe = hyp.Pipeline([('reduce', PCA(n_components=3)), ('labels', DBSCAN())])
    with pytest.raises(ValueError, match=r"1-D result .* pass those as hue="):
        hyp.plot(x, pipeline=pipe, show=False)
