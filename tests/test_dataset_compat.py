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
import os
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

_DATA = Path(__file__).parent / 'data'
_BASELINE = json.loads((_DATA / 'rehosted_compat_baseline.json').read_text())

# The RELEASE GATE (finding #2, 2026-07 review): on ordinary cross-platform
# CI a dataset that can't be downloaded (outage, rate limit, expired URL) is
# skipped so unrelated PRs aren't blocked. But a release run must PROVE the
# hosted artifacts are actually reachable and correct -- otherwise a green run
# could mean "every dataset check silently skipped". Setting
# HYPERTOOLS_REQUIRE_DATASETS=1 (the dedicated `dataset-gate` CI job does)
# turns any download/load failure into a HARD failure instead of a skip.
REQUIRE_DATASETS = os.environ.get('HYPERTOOLS_REQUIRE_DATASETS') == '1'


def _canonical_sha256(obj):
    """Version-stable canonical hash of a loaded dataset, capturing VALUES +
    per-column/index LOGICAL type (numeric vs bool vs text) + columns +
    ordering. Deliberately NORMALIZED so the pinned baseline holds across the
    CI matrix's pandas versions (2.x and 3.x): pandas 3 loads a parquet's text
    columns as the new ``str`` dtype where pandas 2 gives ``object`` -- the
    VALUES are identical, only the dtype *label* differs, so text/object/
    category/string columns are all hashed as one logical "text" type on their
    str values rather than on the version-specific dtype string.

    scripts/gen_rehosted_compat_baseline.py imports THIS function to generate
    the committed baseline, so the two can never drift.
    """
    h = hashlib.sha256()

    def u(s):
        h.update(('|' + s + '|').encode('utf-8'))

    def hash_1d(values, dtype):
        # one column or an index: numeric -> raw bytes (dtype-kind stable
        # across pandas versions); everything else -> normalized text values
        arr = np.asarray(values)
        if arr.dtype.kind in 'iuf':          # int / unsigned / float
            u('num:' + arr.dtype.kind + str(arr.dtype.itemsize))
            h.update(np.ascontiguousarray(arr).tobytes())
        elif arr.dtype.kind == 'b':          # bool
            u('bool')
            h.update(np.ascontiguousarray(arr.astype('u1')).tobytes())
        elif arr.dtype.kind == 'c':          # complex
            u('complex')
            h.update(np.ascontiguousarray(arr).tobytes())
        else:                                # object / str / category / etc.
            u('text')
            u('\x00'.join('' if v is None else str(v)
                          for v in np.asarray(values, dtype=object).tolist()))
        _ = dtype  # dtype label deliberately excluded (version-sensitive)

    def walk(o):
        if isinstance(o, pd.DataFrame):
            u('DF')
            u(','.join(map(str, o.columns)))
            u('IDX')
            hash_1d(o.index.to_numpy(), o.index.dtype)
            for c in o.columns:
                u('COL')
                hash_1d(o[c].to_numpy(), o[c].dtype)
        elif isinstance(o, pd.Series):
            u('SERIES')
            hash_1d(o.to_numpy(), o.dtype)
        elif isinstance(o, np.ndarray):
            u('ARR')
            u(str(o.shape))
            hash_1d(o.ravel(), o.dtype)
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
    except HypertoolsIOError as e:
        if REQUIRE_DATASETS:
            raise            # release gate: a load failure is NOT skippable
        pytest.skip(f'could not download {name!r} ({e})')
    got = _canonical_sha256(data)
    assert got == _BASELINE[name], (
        f"{name}: loaded data no longer matches the pinned pre-1.0 "
        f"compatibility baseline (values/index/columns/dtype/ordering "
        f"changed). got {got[:16]}..., expected {_BASELINE[name][:16]}...")


@pytest.mark.skipif(
    not REQUIRE_DATASETS,
    reason='release gate; set HYPERTOOLS_REQUIRE_DATASETS=1 (the dataset-gate '
           'CI job) to require every hosted dataset to actually load')
def test_release_gate_every_dataset_loads_and_matches():
    # finding #2: a dedicated release gate that CANNOT pass by skipping. It
    # loads every re-hosted dataset (a download/load failure raises here), so
    # a systemic outage/rate-limit/expired-URL turns the run RED instead of a
    # deceptively-green all-skipped run, and it reports the exact count.
    checked = []
    for name in sorted(_BASELINE):
        data = hyp.load(name)          # raises -> hard fail, never skipped
        assert _canonical_sha256(data) == _BASELINE[name], name
        checked.append(name)
    assert checked == sorted(_BASELINE)
    print(f'\nRELEASE GATE: {len(checked)} re-hosted datasets downloaded and '
          f'validated against the pinned baseline: {", ".join(checked)}')


def test_baseline_matches_frozen_legacy_provenance():
    # finding #3: the compat baseline is generated FROM the current loader, so
    # on its own it only prevents future drift. This test anchors it to
    # INDEPENDENT, frozen evidence: rehosted_legacy_provenance.json records,
    # for every dataset, the canonical hash of what PRE-1.0 hyp.load returned
    # -- computed directly from the retired legacy artifacts (each pinned by
    # its own SHA-256; see scripts/gen_legacy_provenance.py). If a regression
    # is ever blessed by regenerating the baseline, the baseline will no
    # longer match this frozen legacy hash and this test fails.
    prov = json.loads((_DATA / 'rehosted_legacy_provenance.json').read_text())
    missing = (set(L._REHOSTED) | {'sotus'}) - set(prov)
    assert not missing, f'datasets missing legacy provenance: {sorted(missing)}'
    mism = {n: (prov[n]['legacy_canonical_hash'], _BASELINE[n])
            for n in prov
            if prov[n]['legacy_canonical_hash'] != _BASELINE.get(n)}
    assert not mism, (
        'the compatibility baseline no longer matches the frozen pre-1.0 '
        f'legacy evidence (dataset: (legacy_hash, baseline_hash)): {mism}')


def test_conversion_manifest_matches_loader_pins():
    # the committed conversion record (retired artifact -> non-executable
    # converted file, with roundtrip_ok) must agree with the loader's pinned
    # SHA-256 for every re-hosted dataset, so it can't drift from what ships
    man = json.loads((_DATA / 'rehosted_conversion_manifest.json').read_text())
    for name in sorted(set(L._REHOSTED) | {'sotus'}):
        assert name in man, f'{name} missing from the conversion manifest'
        assert man[name]['converted_sha256'] == L._EXAMPLE_DATA_SHA256[name], (
            f'{name}: conversion manifest converted_sha256 != loader pin')
        assert man[name].get('roundtrip_ok') is True, (
            f'{name}: conversion manifest does not record a verified roundtrip')


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
