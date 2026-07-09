"""Pipeline / analyze / apply_model robustness (QC 2026-07 release hunt).

- analyze(cluster=...) returned raw cluster LABELS, contradicting its documented
  "returns the transformed data" contract;
- normalize=False + another stage + return_model produced a Pipeline that raised
  NotFittedError on .transform (build_pipeline treated False as a real spec);
- a hard clusterer (DBSCAN, ...) in a Pipeline crashed (no fit_transform);
- apply_model(model='KMeans', ndims=3) crashed (n_components forced on KMeans).

Real data, no mocks.
"""
import numpy as np
import pytest

import hypertools as hyp
from hypertools import Pipeline
from hypertools.tools.analyze import analyze


def _rng():
    return np.random.default_rng(0)


def _two():
    r = _rng()
    return [r.normal(size=(30, 8)), r.normal(size=(30, 8))]


# --- C1: analyze returns transformed data, not labels ------------------

def test_analyze_cluster_returns_transformed_data_not_labels():
    out, model = analyze(_two(), normalize='across', reduce='PCA', ndims=2,
                         cluster='KMeans', return_model=True)
    assert isinstance(out, list) and len(out) == 2
    assert all(np.asarray(o).shape == (30, 2) for o in out)
    assert [n for n, _ in model.steps] == ['normalize', 'reduce', 'cluster']
    labels = model.named_steps['cluster'].transform(
        np.vstack([np.asarray(o) for o in out]))
    assert len(np.asarray(labels)) == 60


# --- C2: normalize=False + return_model + reuse ------------------------

def test_normalize_false_pipeline_transform_reuse():
    x = _two()
    _, model = analyze(x, normalize=False, reduce='PCA', ndims=3,
                       return_model=True)
    assert [n for n, _ in model.steps] == ['reduce']  # normalize=False skipped
    out = model.transform([x[0]])  # must not raise NotFittedError
    assert out is not None


# --- C3: hard clusterers in a Pipeline ---------------------------------

@pytest.mark.parametrize('clf', ['DBSCAN', 'AgglomerativeClustering', 'MeanShift'])
def test_hard_clusterer_in_pipeline_returns_labels(clf):
    x = _rng().normal(size=(40, 6))
    out = np.asarray(Pipeline([clf]).fit_transform(x))
    assert out.shape[0] == 40  # per-observation labels, no crash


def test_manip_list_with_clusterer_does_not_crash():
    x = _rng().normal(size=(40, 6))
    assert np.asarray(hyp.manip(x, model=['ZScore', 'DBSCAN'])).shape[0] == 40


# --- C4: apply_model ndims guard ---------------------------------------

def test_apply_model_kmeans_with_ndims_does_not_crash():
    x = _rng().normal(size=(40, 6))
    out = hyp.apply_model(x, model='KMeans', ndims=3)  # n_components not forced
    assert np.asarray(out).shape[0] == 40


def test_apply_model_pca_ndims_still_reduces():
    x = _rng().normal(size=(40, 6))
    assert np.asarray(hyp.apply_model(x, model='PCA', ndims=3)).shape == (40, 3)
