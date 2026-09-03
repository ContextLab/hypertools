#!/usr/bin/env python
"""Regenerate tests/data/rehosted_legacy_provenance.json (release review #3).

Independent, machine-readable evidence that each re-hosted built-in dataset
reproduces the PRE-1.0 original -- anchored to the legacy artifacts
themselves, NOT to the current loader (which is what the compatibility
baseline is generated from). For every independently-verifiable dataset it
records: the pre-1.0 source, the retired artifact's SHA-256, and the canonical
hash of what pre-1.0 hyp.load returned.

    !!! SECURITY -- READ BEFORE RUNNING !!!
    This script downloads the RETIRED LEGACY PICKLE artifacts and unpickles
    them, which executes arbitrary code embedded in a pickle. It is gated:
    each download is verified against the immutable EXPECTED_LEGACY_SHA256
    hashes below BEFORE it is ever deserialized, and any mismatch aborts
    without unpickling. Even so, run this ONLY in a disposable, isolated
    environment (throwaway VM/container), never as part of package runtime.
    The expected hashes below are the trust anchor and are NEVER overwritten
    by a normal run -- editing them is a reviewed change.

Network + ~260 MB of downloads (the legacy nips pickle alone is ~170 MB);
run once in isolation, review the diff, commit.
Usage:  MPLBACKEND=Agg python scripts/gen_legacy_provenance.py
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import warnings

warnings.filterwarnings('ignore')
os.environ.setdefault('MPLBACKEND', 'Agg')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'tests'))

import importlib  # noqa: E402  (after the sys.path insert for tests/ helpers)

import requests  # noqa: E402  (after the sys.path insert for tests/ helpers)

import hypertools as hyp                       # noqa: E402,F401
# the `load` function shadows the `hypertools.io.load` submodule name
L = importlib.import_module('hypertools.io.load')  # noqa: E402
from hypertools.datageometry import DataGeometry  # noqa: E402
from hypertools.io.sources import parse_drive_interstitial  # noqa: E402
import pandas as pd                            # noqa: E402
from pathlib import Path                        # noqa: E402
from test_dataset_compat import _canonical_sha256  # noqa: E402

# pre-1.0 registry (from d3dff5e3^ hypertools/io/load.py): the retired
# originals every 1.0 re-hosted DATA dataset must still reproduce.
LEGACY_SOURCE = {
    'weights': '1ZXLao5Rxkr45KUMkv08Y1eAedTkpivsd',
    'weights_avg': '1gfI1WB7QqogdYgdclqznhUfxsrhobueO',
    'weights_sample': '1ub-xlYW1D_ASzbLcALcPJuhHUxRwHdIs',
    'spiral': '1nHAusn2VsQinJk35xvJSd7CtWPC1uOwK',
    'mushrooms': '12hmCIZp1tyUoPRHwpiAsm1GDBxiJS8ji',
    'wiki': '1NUqm3svfu2rrFH04xmLbOh0u5WyTe9mh',
    'nips': '1FV7xT2hVgZ1sXfMvAdP1jRsK_dWhp49I',
    'bunny': 'https://www.dropbox.com/s/7d9vo9idqk1hn31/bunny.pkl?dl=1',
    'cube': 'https://www.dropbox.com/s/tkrwe2m4maxl83j/cube.pkl?dl=1',
    'dragon': 'https://www.dropbox.com/s/6w84icbvzh5oilr/dragon.pkl?dl=1',
    'sphere': 'https://www.dropbox.com/s/wp8suye6oh4ze3u/sphere.pkl?dl=1',
    'teapot': 'https://www.dropbox.com/s/f3jj18h3ge2gns6/teapot.pkl?dl=1',
    'vase': 'https://www.dropbox.com/s/prquc7ov18zguuu/vase.pkl?dl=1',
    'biplane': 'https://www.dropbox.com/s/4b9y9ouvjpjbj6x/biplane.pkl?dl=1',
    'datasaurus': 'https://www.dropbox.com/s/6wxjyw8p052a5t9/datasaurus.pkl?dl=1',
}

# IMMUTABLE TRUST ANCHOR (finding #1): the SHA-256 of each retired legacy
# artifact, established by review. A freshly downloaded artifact is verified
# against this BEFORE it is unpickled; a mismatch aborts. Regeneration NEVER
# rewrites these -- changing one is a deliberate, reviewed edit.
EXPECTED_LEGACY_SHA256 = {
    'weights': '695f50f48328f7b9f5741c89854b07f0c4989c4275f929caa76e95af2c92a7ff',
    'weights_avg': '52be2d02d2c5754adbb58e68f86d2c2da2b7a339162f1d2e0c7e3b987ffde06f',
    'weights_sample': 'eaf67c631e9cc8207c70ad1c93c6c022298a6e57f946ef39e24299c9c1bf3f8d',
    'spiral': '7ca728d2972cb0271b3c68693aa7ec744962f8499043120eeefc6b755591f94c',
    'mushrooms': 'b3abdaf8ae1597eeb95c1f1bc6cff6c38d02c9dff99a66ebafed6dc168d2c8cf',
    'wiki': '722d20a286edfad607904123d7756b95fb49e72e037af5d091422c994c4893be',
    'nips': 'e240532dab310652bb489b4f0880af9f681652708dfe60ac3d6ff4e4ee4aaffc',
    'bunny': '7a43745c17834d54bb9dc10b7c286b4f23a4a1c437f8419d53dbe2eaf6ece663',
    'cube': 'ca43191a3c77ce90d449a9cd327a53aaa7bd55032c7de06567c175d6524a02c1',
    'dragon': 'dbfdbbc077f3884251a7140ee030eaf29cff915448d68e3afd96780e5cf79434',
    'sphere': '8dae53277e2f15a57b3ca00299b6e7b980dcde6524c17350ad3b0cc3b3e0688f',
    'teapot': 'c195e6221ad369b274d5f531b98a763c8fe03efadfc5d582011b3148fbf35973',
    'vase': 'b1ef3da871ae93f1a661cc432cc70a2b662cc98748173b44457f838aee493e0f',
    'biplane': 'f5e5661c2eea7a03f30229d6df5546bdd1a9df9e578c865dcefee983801fc814',
    'datasaurus': '7ce78b634ef299098c75445bfc8f28f3edf122b415cdcc179ffda11b2e0bd126',
}

# the two maps must describe the same 15 datasets -- guard against one drifting
# out of sync with the other (a source with no trust anchor, or vice versa)
assert set(LEGACY_SOURCE) == set(EXPECTED_LEGACY_SHA256), (
    'LEGACY_SOURCE and EXPECTED_LEGACY_SHA256 disagree: '
    f'{set(LEGACY_SOURCE) ^ set(EXPECTED_LEGACY_SHA256)}')


def fetch(src, attempts=5):
    """Download a legacy artifact (full URL or Google-Drive id), retrying
    through Drive's rate-limit HTML and virus-scan interstitial. Returns raw
    bytes; NOTHING is deserialized here."""
    last = None
    for i in range(attempts):
        if i:
            time.sleep(2 * 3 ** (i - 1))
        sess = requests.Session()
        try:
            if str(src).startswith('http'):
                r = sess.get(src, stream=True, timeout=180)
            else:
                r = sess.get(L.BASE_URL, params={'id': src}, stream=True,
                             timeout=180)
                if 'html' in r.headers.get('Content-Type', ''):
                    parsed = parse_drive_interstitial(
                        r.content.decode('utf-8', 'replace'))
                    if parsed:
                        url, params = parsed
                        r = sess.get(url, params=params, stream=True,
                                     timeout=180)
            r.raise_for_status()
            raw = r.content
            if raw[:1] == b'<':
                last = 'rate-limit HTML'
                continue
            return raw
        except Exception as e:                 # noqa: BLE001
            last = f'{type(e).__name__}: {e}'
    raise RuntimeError(f'could not fetch {src!r} ({last})')


def legacy_load(name, verified_bytes, workdir):
    """Reconstruct what pre-1.0 hyp.load returned for a DATA dataset from
    ALREADY-AUTHENTICATED bytes: tolerant unpickle -> (mushrooms
    dict->DataFrame) -> DataGeometry.get_data(). Only ever called after the
    caller has verified the bytes against EXPECTED_LEGACY_SHA256."""
    tmp = os.path.join(workdir, name)
    with open(tmp, 'wb') as f:
        f.write(verified_bytes)
    geo = L._unpickle_example(Path(tmp))
    if name == 'mushrooms' and isinstance(geo, DataGeometry):
        geo.data = pd.DataFrame(geo.data)
    return geo.get_data() if isinstance(geo, DataGeometry) else geo


BASELINE = json.loads(open(os.path.join(
    REPO, 'tests', 'data', 'rehosted_compat_baseline.json')).read())

manifest = {}
mismatches = []
# a private temp dir (finding #1: no predictable repo-relative temp files)
workdir = tempfile.mkdtemp(prefix='ht-legacy-provenance-')
try:
    for name, src in LEGACY_SOURCE.items():
        raw = fetch(src)
        got = hashlib.sha256(raw).hexdigest()
        expected = EXPECTED_LEGACY_SHA256[name]
        # AUTHENTICATE BEFORE DESERIALIZE (finding #1)
        if got != expected:
            raise SystemExit(
                f'ABORT: {name} legacy artifact SHA-256 mismatch\n'
                f'  expected {expected}\n  got      {got}\n'
                'The retired URL served bytes that do not match the reviewed '
                'trust anchor; refusing to unpickle. Investigate before '
                'touching EXPECTED_LEGACY_SHA256.')
        obj = legacy_load(name, raw, workdir)   # only reached when authentic
        ch = _canonical_sha256(obj)
        ok = (ch == BASELINE[name])
        manifest[name] = {
            'legacy_source': src,
            'legacy_artifact_sha256': expected,   # the trust anchor, verified
            'legacy_canonical_hash': ch,
            'independent_evidence': True,
        }
        print(f'{name:15s} sha OK  legacy={ch[:16]}  baseline={BASELINE[name][:16]}'
              f'  match={ok}  ({len(raw)//1024} KB)')
        if not ok:
            mismatches.append(name)
finally:
    shutil.rmtree(workdir, ignore_errors=True)

# sotus: DOCUMENTED EVIDENCE EXCEPTION (finding #3). Its pre-1.0 origin was the
# datawrangler zoo corpus (removed in 1.0), for which there is no reliably
# recoverable, hypertools-hosted legacy artifact (the local datawrangler cache
# is an online-only 0-byte placeholder). The re-hosted json.gz was verified
# round-trip-identical to the datawrangler-loaded corpus AT MIGRATION TIME
# (scripts-era build step), but that is NOT independently re-derivable here, so
# we record it honestly as an exception rather than computing a circular
# "legacy" hash from the current loader.
manifest['sotus'] = {
    'legacy_source': 'datawrangler-zoo:sotus',
    'legacy_artifact_sha256': None,
    'legacy_canonical_hash': None,
    'independent_evidence': False,
    'note': ('pre-1.0 origin was the datawrangler zoo corpus (removed in 1.0); '
             'no reliably recoverable legacy artifact, so equivalence rests on '
             'the migration-time round-trip check, not an independent frozen '
             'artifact. Not claimed as independent proof.'),
}
print("sotus           EXCEPTION (documented; no independent legacy artifact)")

if mismatches:
    raise SystemExit(f'MISMATCH vs baseline: {mismatches} -- do NOT commit')

out = os.path.join(REPO, 'tests', 'data', 'rehosted_legacy_provenance.json')
with open(out, 'w') as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
    f.write('\n')
print('\n15 datasets independently reproduce the pre-1.0 original; '
      'sotus recorded as a documented exception. wrote', out)
