# -*- coding: utf-8 -*-
"""A non-streaming Hugging Face load decodes ClassLabel columns (GH #285).

``Dataset.to_pandas()`` returns ``ClassLabel`` features as bare integer
codes, which are meaningless without the dataset's feature spec -- the
hugging_face_embeddings tutorial worked around it with
``news.features['label'].names``. ``hyp.load()`` now decodes those columns
to their string names by default, matching what the dataset itself
advertises; ``decode_labels=False`` keeps the raw codes.

Real loads of a real (tiny) slice of a real Hub dataset -- no mocks --
skipped only when the Hub is transiently unreachable.
"""

import os
import subprocess
import sys

import pandas as pd
import pytest

import matplotlib
matplotlib.use('Agg')

import hypertools as hyp                                       # noqa: E402
from hypertools.io.sources import _load_hf                     # noqa: E402
from tests._netskip import skip_on_transient_network           # noqa: E402

# a 4-class news-topic dataset; 20 rows of its test split is a few KB
DATASET = 'fancyzhx/ag_news'
SPLIT = 'test[:20]'
LABELS = ['World', 'Sports', 'Business', 'Sci/Tech']


@pytest.fixture(scope='module')
def datasets_module():
    return pytest.importorskip('datasets')


def test_class_labels_are_decoded_by_default(datasets_module):
    with skip_on_transient_network(f'loading {DATASET}'):
        df = _load_hf(DATASET, split=SPLIT)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 20
    assert df['label'].map(type).eq(str).all()
    assert set(df['label']).issubset(LABELS)
    assert 'text' in df.columns


def test_decode_labels_false_keeps_the_integer_codes(datasets_module):
    with skip_on_transient_network(f'loading {DATASET}'):
        raw = _load_hf(DATASET, split=SPLIT, decode_labels=False)
    assert raw['label'].dtype.kind == 'i'
    assert set(raw['label']).issubset(range(len(LABELS)))


def test_decoded_names_match_the_datasets_own_feature_spec(datasets_module):
    with skip_on_transient_network(f'loading {DATASET}'):
        decoded = _load_hf(DATASET, split=SPLIT)
        raw = _load_hf(DATASET, split=SPLIT, decode_labels=False)
        names = datasets_module.load_dataset(
            DATASET, split=SPLIT).features['label'].names
    assert names == LABELS
    expected = [names[code] for code in raw['label']]
    assert list(decoded['label']) == expected


def test_hyp_load_decodes_labels(datasets_module):
    # the public path: no extra arguments needed
    with skip_on_transient_network(f'loading {DATASET} through hyp.load'):
        df = hyp.load(DATASET, split=SPLIT)
    assert set(df['label']).issubset(LABELS)


def test_non_classlabel_columns_are_untouched(datasets_module):
    with skip_on_transient_network(f'loading {DATASET}'):
        df = _load_hf(DATASET, split=SPLIT)
        raw = _load_hf(DATASET, split=SPLIT, decode_labels=False)
    assert list(df['text']) == list(raw['text'])


def test_decoded_labels_work_as_a_hue(datasets_module):
    import matplotlib.pyplot as plt
    with skip_on_transient_network(f'loading {DATASET}'):
        df = hyp.load(DATASET, split=SPLIT)
    # (generated directly rather than via hyp.load: threading generator
    # kwargs through hyp.load needs the load() signature change reported
    # with this work)
    from hypertools.io.sources import synthetic_dataset
    points = synthetic_dataset('blobs', n_samples=20, random_state=0)
    fig = hyp.plot(points[['dim_0', 'dim_1']], hue=list(df['label']),
                   show=False)
    assert fig is not None
    plt.close('all')


def test_hf_env_defaults_are_set_before_the_datasets_import(datasets_module):
    # HF_HUB_DISABLE_PROGRESS_BARS / HF_HUB_VERBOSITY / TOKENIZERS_PARALLELISM
    # are read at import time by huggingface_hub and tokenizers, so _load_hf
    # must set them BEFORE importing datasets. Checked in a FRESH subprocess
    # (this process may already have them set) that reports whether the
    # values were in place by the time huggingface_hub was imported.
    script = (
        'import os, sys\n'
        'for var in ("HF_HUB_DISABLE_PROGRESS_BARS", "HF_HUB_VERBOSITY",\n'
        '            "TOKENIZERS_PARALLELISM"):\n'
        '    os.environ.pop(var, None)\n'
        'import hypertools.io.sources as s\n'
        'assert "huggingface_hub" not in sys.modules\n'
        f'df = s._load_hf({DATASET!r}, split={SPLIT!r})\n'
        'import huggingface_hub.utils as u\n'
        'print(os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"),\n'
        '      os.environ.get("HF_HUB_VERBOSITY"),\n'
        '      os.environ.get("TOKENIZERS_PARALLELISM"),\n'
        '      u.are_progress_bars_disabled(), len(df))\n')
    env = {k: v for k, v in os.environ.items()
           if k not in ('HF_HUB_DISABLE_PROGRESS_BARS', 'HF_HUB_VERBOSITY',
                        'TOKENIZERS_PARALLELISM')}
    with skip_on_transient_network(f'loading {DATASET} in a subprocess'):
        proc = subprocess.run([sys.executable, '-c', script], env=env,
                              capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip()[-2000:])
    assert proc.stdout.split() == ['1', 'error', 'false', 'True', '20']
