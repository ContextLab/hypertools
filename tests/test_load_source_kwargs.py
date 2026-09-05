# -*- coding: utf-8 -*-
"""``hypertools.load`` plumbs source-specific keyword arguments through to
the resolvers added for GH #285 (synthetic datasets, web sources, the URL
cache, and Hugging Face label decoding).

Before this change, ``hyp.load('random_walk', n_samples=40)`` raised
``TypeError`` -- ``load()``'s signature had no ``**kwargs`` at all, so a
caller could only reach these knobs by calling
``hypertools.io.sources.synthetic_dataset``/``load_source`` directly. Now
``load()`` accepts ``cache``/``offline``/``decode_labels`` and an open
``**source_kwargs`` for the synthetic (step 6) and web-prefix (step 7)
resolvers, and rejects a keyword that no resolver on the matched path
actually accepts (a passthrough of already-loaded data, or a misspelled
keyword against a source that takes none) with a ``TypeError`` naming it.

Real calls throughout -- no mocks. The Hugging Face and URL-download cases
are skipped (never xfailed/passed silently) on a transient network error,
matching ``tests/test_load_538_kaggle.py`` / ``tests/test_load_hf_classlabel.py``.
"""

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use('Agg')

import hypertools as hyp                                        # noqa: E402
from hypertools.io.sources import (HypertoolsOfflineError,      # noqa: E402
                                   cached_url_path, url_cache_dir)
from tests._netskip import skip_on_transient_network            # noqa: E402

# a small (42 KB), stable CSV in this repository, served raw over https --
# the same file tests/test_load_url_cache.py uses for the same reason.
CSV_URL = ('https://raw.githubusercontent.com/ContextLab/hypertools/'
           'master/docs/tutorials/stock_closes_cached.csv')
# never fetched -- offline=True must reject this on a cache miss alone,
# without opening any connection, so the URL need not resolve to anything
UNCACHED_URL = 'https://example.com/hypertools-test-never-cached.csv'


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Point the URL cache at a temp directory so this test never writes
    to the user's real ~/hypertools_data."""
    path = tmp_path / 'urlcache'
    monkeypatch.setenv('HYPERTOOLS_URL_CACHE', str(path))
    assert url_cache_dir() == path
    return path


# --------------------------------------------------------------- synthetic

def test_random_walk_shape_and_reproducibility():
    a = hyp.load('random_walk', n_samples=40, n_features=5, random_state=0)
    b = hyp.load('random_walk', n_samples=40, n_features=5, random_state=0)
    assert isinstance(a, np.ndarray)
    assert a.shape == (40, 5)
    assert np.array_equal(a, b)


def test_n_datasets_returns_a_list():
    walks = hyp.load('helix', n_datasets=3, n_samples=10, random_state=1)
    assert isinstance(walks, list) and len(walks) == 3
    assert all(w.shape == (10, 3) for w in walks)


def test_blobs_has_a_target_column():
    df = hyp.load('blobs', n_samples=30, centers=3, random_state=1)
    assert isinstance(df, pd.DataFrame)
    assert 'target' in df.columns
    assert df['target'].nunique() == 3
    assert len(df) == 30


def test_synthetic_names_resolve_before_the_seaborn_lookup():
    """hyp.load resolves a registered synthetic name right after the
    scikit-learn lookup (io/load.py:_resolve), before seaborn_dataset.
    Real observable: none of the synthetic names is also a seaborn or
    scikit-learn dataset name (so the order can never shadow a real
    download), and a synthetic load completes with no network use --
    it returns identical arrays when run twice, seeded, in well under the
    time a seaborn name listing takes."""
    import time
    from hypertools.io.sources import SYNTHETIC_DATASETS, SKLEARN_DATASETS
    assert not set(SYNTHETIC_DATASETS) & set(SKLEARN_DATASETS)
    t0 = time.perf_counter()
    a = hyp.load('random_walk', n_samples=5, n_features=2, random_state=0)
    b = hyp.load('random_walk', n_samples=5, n_features=2, random_state=0)
    assert a.shape == (5, 2) and np.array_equal(np.asarray(a), np.asarray(b))
    assert time.perf_counter() - t0 < 5.0


# ------------------------------------------------------------- URL cache

def test_cache_true_writes_under_the_cache_dir_and_offline_reads_it(cache_dir):
    path = cached_url_path(CSV_URL)
    assert not path.exists()

    with skip_on_transient_network(f'downloading {CSV_URL}'):
        first = hyp.load(CSV_URL, cache=True)
    assert isinstance(first, pd.DataFrame)
    assert path.is_file()

    # a second, OFFLINE call must read the cache and never touch the
    # network: the cached file is not rewritten (same mtime, no .part
    # sidecar appears) and an UNCACHED url under offline=True raises
    # instead of downloading -- real observables, no patched network.
    mtime = path.stat().st_mtime_ns
    second = hyp.load(CSV_URL, offline=True)
    pd.testing.assert_frame_equal(first, second)
    assert path.stat().st_mtime_ns == mtime
    assert not list(path.parent.glob('*.part'))
    with pytest.raises(HypertoolsOfflineError):
        hyp.load(CSV_URL.replace('.csv', '.never-cached.csv'), offline=True)


def test_offline_without_cache_raises(cache_dir):
    with pytest.raises(HypertoolsOfflineError):
        hyp.load(UNCACHED_URL, offline=True)


# -------------------------------------------------------- Hugging Face

DATASET = 'fancyzhx/ag_news'
SPLIT = 'test[:20]'


def test_decode_labels_false_keeps_ints_on_a_tiny_hf_dataset():
    pytest.importorskip('datasets')
    with skip_on_transient_network(f'loading {DATASET} through hyp.load'):
        raw = hyp.load(DATASET, split=SPLIT, decode_labels=False)
        decoded = hyp.load(DATASET, split=SPLIT)
    assert raw['label'].dtype.kind == 'i'
    assert decoded['label'].map(type).eq(str).all()


# --------------------------------------------------------- misuse -> TypeError

def test_passthrough_dataframe_with_source_kwarg_raises_typeerror():
    df = pd.DataFrame(np.zeros((5, 3)))
    with pytest.raises(TypeError, match=r'n_samples'):
        hyp.load(df, n_samples=5)


def test_passthrough_ndarray_with_source_kwarg_raises_typeerror():
    with pytest.raises(TypeError, match=r'random_state'):
        hyp.load(np.zeros((5, 3)), random_state=0)


def test_misspelled_kwarg_on_a_sklearn_dataset_raises_typeerror_naming_it():
    with pytest.raises(TypeError, match=r'reduc\b'):
        hyp.load('iris', reduc='PCA')


def test_misspelled_kwarg_on_a_built_in_dataset_raises_typeerror_naming_it():
    with pytest.raises(TypeError, match=r'n_samples'):
        hyp.load('spiral', n_samples=10)


def test_unknown_kwarg_on_a_synthetic_dataset_raises_from_the_generator():
    # a typo'd kwarg to the generator itself (not a hyp.load-level typo)
    # still raises TypeError, just from further down the call chain
    with pytest.raises(TypeError):
        hyp.load('random_walk', n_smaples=5)
