# -*- coding: utf-8 -*-
"""HF progress-bar / tokenizer env vars are set BEFORE the lazy
`sentence_transformers` import (GH #285).

Before this fix, three tutorial notebooks (`docs/tutorials/
wikipedia_embeddings.ipynb`, `hugging_face_embeddings.ipynb`,
`conversation_trajectories.ipynb`) and two scripts
(`scripts/execute_tutorial.py`, `scripts/render_bluesky_clips.py`) each set
`HF_HUB_DISABLE_PROGRESS_BARS`, `HF_HUB_VERBOSITY`, and
`TOKENIZERS_PARALLELISM` by hand, with a comment explaining that
`huggingface_hub`/`transformers`/`sentence_transformers` read these
variables AT IMPORT TIME, so the import had to be moved below the
`os.environ[...] = ...` lines in every one of those cells/scripts.

Now (`hypertools/tools/text2mat.py`, `_HFTextModel.transform`): the three
variables are set via `os.environ.setdefault(...)` immediately before the
lazy `sentence_transformers` import, so a plain `hyp.plot(docs,
vectorizer=...)` needs no caller-side preamble, and an explicit caller
setting is never overwritten (`setdefault`).

Every test here makes a REAL `hyp.plot()` call with a REAL
sentence-transformers model (downloaded/cached on first use); nothing is
mocked. Each runs in a FRESH subprocess so that no earlier import in this
test session (or an earlier test module) has already set these variables
or already imported `sentence_transformers`/`transformers` -- the whole
point being to observe what happens the first time, from a clean process,
which only a subprocess can show. The `text` extra is optional and CI's
dev install leaves it out, so the module skips without
`sentence_transformers` (same `importorskip` convention as
tests/test_text2mat_hf_defaults.py).
"""

import os
import subprocess
import sys

import pytest

pytest.importorskip('sentence_transformers')

HF_MODEL = 'all-MiniLM-L6-v2'
ENV_VARS = ('HF_HUB_DISABLE_PROGRESS_BARS', 'HF_HUB_VERBOSITY',
            'TOKENIZERS_PARALLELISM')
EXPECTED = {
    'HF_HUB_DISABLE_PROGRESS_BARS': '1',
    'HF_HUB_VERBOSITY': 'error',
    'TOKENIZERS_PARALLELISM': 'false',
}

# A generous ceiling for a subprocess that imports torch/transformers and
# downloads/loads a cached MiniLM model.
TIMEOUT_S = 120

_SCRIPT = """
import os
import sys

import matplotlib
matplotlib.use('Agg')

import hypertools as hyp

docs = [
    'the cat sat on the mat',
    'dogs chase cats around the yard',
    'stock markets fell sharply today',
    'the central bank raised interest rates again',
]
fig = hyp.plot(docs, vectorizer={model!r}, show=False)
assert fig is not None

# Report the post-import environment on stdout (stderr is reserved for
# capturing any progress-bar chatter the test asserts against).
for var in {env_vars!r}:
    print(f'{{var}}={{os.environ.get(var, "<UNSET>")}}')
"""


def _run_subprocess(env):
    env = dict(env)
    env['MPLBACKEND'] = 'Agg'
    script = _SCRIPT.format(model=HF_MODEL, env_vars=ENV_VARS)
    result = subprocess.run(
        [sys.executable, '-c', script],
        env=env, capture_output=True, text=True, timeout=TIMEOUT_S)
    assert result.returncode == 0, (
        f'subprocess failed (rc={result.returncode}):\\n'
        f'--- stdout ---\\n{result.stdout}\\n'
        f'--- stderr ---\\n{result.stderr}')
    reported = {}
    for line in result.stdout.splitlines():
        if '=' in line and line.split('=', 1)[0] in ENV_VARS:
            k, v = line.split('=', 1)
            reported[k] = v
    return reported, result.stderr


def test_env_vars_set_before_import_with_no_progress_bar_output():
    env = {k: v for k, v in os.environ.items() if k not in ENV_VARS}
    reported, stderr = _run_subprocess(env)
    assert reported == EXPECTED, (
        f'expected {EXPECTED}, got {reported}\\nstderr:\\n{stderr}')
    # No tqdm/hub progress-bar chatter reached stderr: carriage returns and
    # the "it/s" throughput suffix are tqdm's signature, whether the bar is
    # a bare tqdm render or the ipywidgets/hub variant's fallback text.
    assert '\\r' not in stderr, f'stderr contained a carriage return:\\n{stderr!r}'
    assert 'it/s' not in stderr, f'stderr contained tqdm output:\\n{stderr!r}'


def test_preset_tokenizers_parallelism_survives():
    # A caller's own explicit setting must win over the library default
    # (`os.environ.setdefault` never overwrites an existing value).
    env = {k: v for k, v in os.environ.items()
           if k not in ('HF_HUB_DISABLE_PROGRESS_BARS', 'HF_HUB_VERBOSITY')}
    env['TOKENIZERS_PARALLELISM'] = 'true'
    reported, stderr = _run_subprocess(env)
    expected = dict(EXPECTED, TOKENIZERS_PARALLELISM='true')
    assert reported == expected, (
        f'expected {expected}, got {reported}\\nstderr:\\n{stderr}')
