#!/usr/bin/env python
"""Generate tests/data/rehosted_legacy_provenance.json (release review #3).

Independent, machine-readable evidence that each re-hosted built-in dataset
reproduces the PRE-1.0 original -- anchored to the legacy artifacts
themselves, NOT to the current loader (which is what the compatibility
baseline is generated from). For every re-hosted dataset it records:

  legacy_source          the pre-1.0 download URL / Google-Drive id
  legacy_artifact_sha256 SHA-256 of the retired original file (pins it)
  legacy_canonical_hash  canonical hash of what pre-1.0 hyp.load returned,
                         obtained by re-running the pre-1.0 extraction
                         (tolerant unpickle -> DataGeometry.get_data(), with
                         mushrooms' dict->DataFrame step) on that artifact

A committed test asserts legacy_canonical_hash == the current baseline, so
regenerating the baseline after a regression can no longer silently bless it:
the frozen legacy hash would no longer match.

Network + ~85 MB of downloads; run once, review the diff, commit.
Usage:  MPLBACKEND=Agg python scripts/gen_legacy_provenance.py
"""
import hashlib
import io
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
os.environ.setdefault('MPLBACKEND', 'Agg')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'tests'))

import importlib

import requests

import hypertools as hyp                       # noqa: E402
# the `load` function shadows the `hypertools.io.load` submodule name
L = importlib.import_module('hypertools.io.load')  # noqa: E402
from hypertools.datageometry import DataGeometry  # noqa: E402
from hypertools.io.sources import parse_drive_interstitial  # noqa: E402
import pandas as pd                            # noqa: E402
from test_dataset_compat import _canonical_sha256  # noqa: E402

# pre-1.0 registry (from d3dff5e3^ hypertools/io/load.py): the retired
# originals every 1.0 re-hosted DATA dataset must still reproduce.
LEGACY = {
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


def fetch(src, attempts=5):
    """Download a legacy artifact (full URL or Google-Drive id), retrying
    through Drive's rate-limit HTML and virus-scan interstitial."""
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


def legacy_load(name, raw):
    """Exactly what pre-1.0 hyp.load returned for a DATA dataset: tolerant
    unpickle -> (mushrooms dict->DataFrame) -> DataGeometry.get_data()."""
    tmp = os.path.join(REPO, f'.legacy_{name}.tmp')
    with open(tmp, 'wb') as f:
        f.write(raw)
    try:
        geo = L._unpickle_example(__import__('pathlib').Path(tmp))
    finally:
        os.unlink(tmp)
    if name == 'mushrooms' and isinstance(geo, DataGeometry):
        geo.data = pd.DataFrame(geo.data)
    return geo.get_data() if isinstance(geo, DataGeometry) else geo


BASELINE = json.loads(open(os.path.join(
    REPO, 'tests', 'data', 'rehosted_compat_baseline.json')).read())

manifest = {}
mismatches = []
for name, src in LEGACY.items():
    raw = fetch(src)
    art_sha = hashlib.sha256(raw).hexdigest()
    obj = legacy_load(name, raw)
    ch = _canonical_sha256(obj)
    ok = (ch == BASELINE[name])
    manifest[name] = {
        'legacy_source': src,
        'legacy_artifact_sha256': art_sha,
        'legacy_canonical_hash': ch,
    }
    print(f'{name:15s} legacy={ch[:16]}  baseline={BASELINE[name][:16]}  '
          f'match={ok}  ({len(raw)//1024} KB)')
    if not ok:
        mismatches.append(name)

# sotus: its legacy source IS the datawrangler zoo corpus (there was never a
# hypertools-hosted pickle for it), so anchor its evidence to that corpus as
# currently loaded -- the re-hosted json.gz must reproduce it.
sotus_obj = hyp.load('sotus')
manifest['sotus'] = {
    'legacy_source': 'datawrangler-zoo:sotus',
    'legacy_artifact_sha256': None,
    'legacy_canonical_hash': _canonical_sha256(sotus_obj),
}
print(f"{'sotus':15s} legacy={manifest['sotus']['legacy_canonical_hash'][:16]}"
      f"  baseline={BASELINE['sotus'][:16]}  "
      f"match={manifest['sotus']['legacy_canonical_hash'] == BASELINE['sotus']}")
if manifest['sotus']['legacy_canonical_hash'] != BASELINE['sotus']:
    mismatches.append('sotus')

if mismatches:
    raise SystemExit(f'MISMATCH vs baseline: {mismatches} -- do NOT commit')

out = os.path.join(REPO, 'tests', 'data', 'rehosted_legacy_provenance.json')
with open(out, 'w') as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
    f.write('\n')
print('\nall', len(manifest), 'datasets reproduce the pre-1.0 original; wrote', out)
