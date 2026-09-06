"""Third-party reducers must not spam the user with warnings that are not
about the user's choices (1.1 feature-tour report, section 9.8)."""

import warnings

import numpy as np
import pytest

import hypertools as hyp


def _digits400():
    d = hyp.load('digits')
    return d.drop(columns='target').to_numpy()[:400]


def test_seeded_umap_does_not_warn_about_n_jobs():
    pytest.importorskip('umap')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        out = hyp.reduce(_digits400(), reduce='UMAP', ndims=3, random_state=0)
    assert np.asarray(out).shape == (400, 3)
    assert not [m for m in w if 'n_jobs' in str(m.message)]


def test_seeded_umap_honours_an_explicit_n_jobs():
    pytest.importorskip('umap')
    from hypertools.reduce.common import resolve_reducer
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        _, model = hyp.reduce(_digits400(), reduce={'model': 'UMAP', 'kwargs': {'n_jobs': 2}},
                              ndims=3, random_state=0, return_model=True)
    # umap itself overrides to 1 and says so: that warning is the user's
    # own n_jobs= choice and must still reach them
    assert [m for m in w if 'n_jobs' in str(m.message)]
    assert isinstance(model.model, resolve_reducer('UMAP'))   # Reducer wrapper


def test_isomap_graph_completion_does_not_leak_sparse_efficiency_warnings():
    from scipy.sparse import SparseEfficiencyWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        out = hyp.reduce(_digits400(), reduce='Isomap', ndims=3)
    assert np.asarray(out).shape == (400, 3)
    assert not [m for m in w if issubclass(m.category, SparseEfficiencyWarning)]
    # sklearn's own data warning (too few neighbours for one connected
    # graph) is about the user's data and still reaches them, once
    assert sum('connected components' in str(m.message) for m in w) == 1
