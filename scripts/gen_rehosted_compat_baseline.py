#!/usr/bin/env python
"""Regenerate tests/data/rehosted_compat_baseline.json (release review #3).

Pins a NORMALIZED canonical hash (values + logical column/index type +
columns + ordering) of the CURRENT loader output for every re-hosted built-in
dataset. The hash function is imported from tests/test_dataset_compat.py so
the committed baseline and the test that checks it can never drift.

Each pinned value was proven equal to the pre-1.0 ORIGINAL when it was first
created: datasaurus was checked frame-by-frame with .equals() against its
pre-1.0 pickle (its index is restored from _DATASAURUS_INDEX_STARTS), and
mushrooms/spiral/biplane/bunny were checked against their originals per format
family. Run this only to RE-pin after an intentional, separately-verified
change to a hosted dataset.

Usage:  MPLBACKEND=Agg python scripts/gen_rehosted_compat_baseline.py
"""
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ.setdefault('MPLBACKEND', 'Agg')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'tests'))

import hypertools as hyp
import hypertools.io.load as _load_mod  # noqa: F401  (module, not the fn)
from test_dataset_compat import _canonical_sha256

import importlib
L = importlib.import_module('hypertools.io.load')

# Overwriting the baseline BLESSES whatever the current loader returns, so a
# regression re-run here would silently pin the regression. Require explicit
# acknowledgement, and point at the independent legacy-equivalence guard
# (finding #3, 2026-07 review).
if '--force' not in sys.argv:
    sys.exit(
        'refusing to overwrite tests/data/rehosted_compat_baseline.json '
        'without --force.\n'
        'This re-pins the baseline to the CURRENT loader output. Only do this '
        'for an intentional, separately-verified change, and re-run\n'
        '  python scripts/gen_legacy_provenance.py\n'
        'afterwards so tests/test_dataset_compat.py'
        '::test_baseline_matches_frozen_legacy_provenance still proves\n'
        'equivalence to the pre-1.0 originals. Re-run with --force to proceed.')

NAMES = sorted(set(L._REHOSTED) | {'sotus'})

baseline = {}
for name in NAMES:
    baseline[name] = _canonical_sha256(hyp.load(name))
    print(f'{name}: {baseline[name]}')

out = os.path.join(REPO, 'tests', 'data', 'rehosted_compat_baseline.json')
with open(out, 'w') as f:
    json.dump(baseline, f, indent=2, sort_keys=True)
    f.write('\n')
print('\nwrote', out)
