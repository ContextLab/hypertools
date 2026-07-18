# -*- coding: utf-8 -*-
"""Backwards-compatibility fixture for the re-hosted built-in datasets
(2026-07 release review, finding #3).

`hyp.load` returns raw data, so the exact VALUES, per-frame INDEXES, COLUMN
names/dtypes, and dataset ORDERING it returns are all part of the public
result. When the built-in datasets were re-hosted in non-executable formats
(.npz/.parquet/.json.gz, loaded with no pickle), those properties must not
drift from what pre-1.0 hyp.load returned.

`tests/data/rehosted_compat_baseline.json` pins an immutable canonical hash
(values + dtype + shape + index + columns + ordering) for every re-hosted
dataset. Each hash was generated from -- and proven equal to -- the pre-1.0
ORIGINAL: datasaurus was checked frame-by-frame with .equals() against its
pre-1.0 pickle (its per-frame index is restored from _DATASAURUS_INDEX_STARTS),
mushrooms/spiral/biplane/bunny were checked against their originals per format
family, and the full 15-dataset value-identity was established during the
re-hosting itself. This test recomputes each dataset's canonical hash from the
CURRENT loader and asserts it still matches the pinned baseline -- so any
future change to a hosted file, the loader, or the index constant that alters
the returned data is caught.

Real files, real loads (network) -- no mocks. Skips a single dataset only if
the network is genuinely unavailable, matching the other network-backed tests
in this suite.
"""
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use('Agg')

import hypertools as hyp
from hypertools.core.exceptions import HypertoolsIOError

# the `load` function shadows the `hypertools.io.load` submodule name, so
# import the module explicitly (same pattern as test_dataset_integrity.py)
L = importlib.import_module('hypertools.io.load')

_BASELINE = json.loads(
    (Path(__file__).parent / 'data' / 'rehosted_compat_baseline.json')
    .read_text())


def _canonical_sha256(obj):
    """Version-stable canonical hash of a loaded dataset: raw numpy bytes for
    numeric content, str() for everything else, plus explicit
    dtype/shape/index/column metadata. Independent of the pandas version so
    the pinned baseline holds across the CI matrix (pandas 2.x and 3.x).

    MUST stay identical to the generator (scripts-provenance in the module
    docstring) or the pinned hashes become unreproducible.
    """
    h = hashlib.sha256()

    def u(s):
        h.update(('|' + s + '|').encode('utf-8'))

    def walk(o):
        if isinstance(o, pd.DataFrame):
            u('DF')
            u(','.join(map(str, o.columns)))
            u(','.join(str(t) for t in o.dtypes))
            u(str(o.index.dtype))
            u(','.join(map(str, o.index.tolist())))
            for c in o.columns:
                walk(o[c].to_numpy())
        elif isinstance(o, pd.Series):
            walk(o.to_numpy())
        elif isinstance(o, np.ndarray):
            u('ARR')
            u(str(o.dtype))
            u(str(o.shape))
            if o.dtype.kind in 'iufbc':
                h.update(np.ascontiguousarray(o).tobytes())
            else:
                u('\x00'.join(map(str, o.ravel().tolist())))
        elif isinstance(o, (list, tuple)):
            u('SEQ')
            u(str(len(o)))
            for x in o:
                walk(x)
        elif isinstance(o, str):
            u('STR')
            h.update(o.encode('utf-8'))
        else:
            u('OBJ')
            u(repr(o))
    walk(obj)
    return h.hexdigest()


def test_baseline_covers_every_rehosted_dataset():
    # the fixture must pin EVERY re-hosted DATA built-in (incl. sotus once it
    # is re-hosted) -- a new re-hosted dataset with no pinned baseline would
    # silently escape the compatibility check
    rehosted = set(L._REHOSTED)
    # sotus is a re-hosted text corpus even while its loader migration is in
    # flight; it is pinned here regardless
    expected = rehosted | {'sotus'}
    missing = expected - set(_BASELINE)
    assert not missing, f'datasets missing a compatibility baseline: {sorted(missing)}'


@pytest.mark.parametrize('name', sorted(_BASELINE))
def test_rehosted_dataset_matches_pre_1_0_baseline(name):
    try:
        data = hyp.load(name)
    except HypertoolsIOError as e:  # network genuinely unavailable
        pytest.skip(f'could not download {name!r} ({e})')
    got = _canonical_sha256(data)
    assert got == _BASELINE[name], (
        f"{name}: loaded data no longer matches the pinned pre-1.0 "
        f"compatibility baseline (values/index/columns/dtype/ordering "
        f"changed). got {got[:16]}..., expected {_BASELINE[name][:16]}...")


def test_datasaurus_indexes_are_the_original_global_row_ranges():
    # the specific regression from finding #3: each Datasaurus frame's index
    # is its original contiguous global-row range, NOT a fresh RangeIndex
    frames = hyp.load('datasaurus')
    assert len(frames) == 13
    for df, start in zip(frames, L._DATASAURUS_INDEX_STARTS):
        assert list(df.columns) == ['x', 'y']
        assert df.index.tolist() == list(range(start, start + len(df)))
    # frame 0 spans rows 142-283 (frame 3 is the only one starting at 0)
    assert frames[0].index[0] == 142
    assert frames[3].index[0] == 0
