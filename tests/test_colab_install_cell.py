"""Unit tests for scripts/add_colab_install_cell.py.

The script keeps every committed docs notebook's Colab install cell pointed at
the right hypertools build: the matching GitHub branch on a dev branch, the
plain PyPI spec on ``master``. The 2026-07 release review found the original
could only ADD an install cell, never RE-TARGET an existing one (its guard
skipped any cell already containing "pip install"), so a stale
``@dev-1.0-refactor`` target could never be migrated. These tests pin the
re-targeting behavior that fixes that.
"""

import ast
import importlib.util
import json
import os
import pathlib

import pytest

_SCRIPT = (pathlib.Path(__file__).resolve().parent.parent
           / 'scripts' / 'add_colab_install_cell.py')


def _load():
    spec = importlib.util.spec_from_file_location('add_colab_install_cell',
                                                  _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


acic = _load()

_URL = 'git+https://github.com/ContextLab/hypertools.git'
_BRANCH_LINE = (
    '%pip install -q "hypertools[interactive] @ '
    + _URL + '@dev-1.0-refactor"')


def test_hyp_spec_branch_vs_master():
    assert acic.hyp_spec('interactive', 'dev-1.0') == (
        'hypertools[interactive] @ ' + _URL + '@dev-1.0')
    # master -> plain PyPI spec, no git url
    assert acic.hyp_spec('interactive', 'master') == 'hypertools[interactive]'
    assert '@' not in acic.hyp_spec('interactive', 'master')


def test_retarget_branch_to_branch_preserves_extras():
    for extras in ('interactive', 'interactive,lsl', 'interactive,predict',
                   'predict'):
        line = f'%pip install -q "hypertools[{extras}] @ {_URL}@dev-1.0-refactor"'
        out = acic.retarget_text(line, 'dev-1.0')
        assert out == f'%pip install -q "hypertools[{extras}] @ {_URL}@dev-1.0"'
        # extras survived exactly
        assert f'hypertools[{extras}]' in out


def test_retarget_to_master_drops_git_ref():
    out = acic.retarget_text(_BRANCH_LINE, 'master')
    assert out == '%pip install -q "hypertools[interactive]"'
    assert 'git+' not in out and '@' not in out


def test_retarget_to_master_cleans_the_preview_note():
    # release review, GAP #1: the master migration must also strip the
    # "(<x> preview) ... On release this becomes ..." note, or the released
    # notebooks ship saying "preview".
    cell = ('# Install hypertools (dev-1.0 preview) -- run this first on Colab.\n'
            '# On release this becomes: %pip install hypertools\n'
            f'    {_BRANCH_LINE}')
    out = acic.retarget_text(cell, 'master')
    assert 'preview' not in out
    assert 'On release this becomes' not in out
    assert '# Install hypertools (run this first on Colab)' in out
    assert acic.retarget_text(out, 'master') == out       # idempotent


def test_retarget_updates_preview_note_token_on_branch():
    note = '# Install hypertools (dev-1.0-refactor preview) -- run this first'
    assert acic.retarget_text(note, 'dev-1.0') == (
        '# Install hypertools (dev-1.0 preview) -- run this first')


def test_retarget_leaves_unrelated_installs_untouched():
    # a tutorial's own extra deps and a bare PyPI fallback must not be rewritten
    for line in ('%pip install -q convokit',
                 '%pip install -q sentence-transformers',
                 '%pip install -q "hypertools[predict]"'):
        assert acic.retarget_text(line, 'dev-1.0') == line
        assert acic.retarget_text(line, 'master') == line


def test_retarget_is_idempotent():
    once = acic.retarget_text(_BRANCH_LINE, 'dev-1.0')
    twice = acic.retarget_text(once, 'dev-1.0')
    assert once == twice
    m_once = acic.retarget_text(_BRANCH_LINE, 'master')
    assert acic.retarget_text(m_once, 'master') == m_once


def test_retarget_notebook_preserves_other_cells_and_guards():
    nb = {
        'cells': [
            {'cell_type': 'code', 'source': [
                "import importlib.util\n",
                "if importlib.util.find_spec('hypertools') is None:\n",
                f'    {_BRANCH_LINE}\n',
                "if importlib.util.find_spec('convokit') is None:\n",
                "    %pip install -q convokit"]},
            {'cell_type': 'markdown', 'source': ['# heading']},
            {'cell_type': 'code', 'source': ["import hypertools as hyp"]},
        ]
    }
    changed = acic.retarget_notebook(nb, 'dev-1.0')
    assert changed is True
    cell0 = ''.join(nb['cells'][0]['source'])
    assert '@dev-1.0"' in cell0 and 'dev-1.0-refactor' not in cell0
    # the convokit guard and the import cell are untouched
    assert "find_spec('convokit')" in cell0
    assert '%pip install -q convokit' in cell0
    assert ''.join(nb['cells'][2]['source']) == 'import hypertools as hyp'
    # second pass is a no-op
    assert acic.retarget_notebook(nb, 'dev-1.0') is False


def _conf_install_cell(branch, monkeypatch):
    """Call docs/conf.py's `_install_notebook_cell()` for a forced branch,
    AST-extracted so importing the whole Sphinx config (with its extensions) is
    not required. Returns the generated first-cell text."""
    conf = pathlib.Path(__file__).resolve().parent.parent / 'docs' / 'conf.py'
    tree = ast.parse(conf.read_text(encoding='utf-8'))
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef)
               and n.name == '_install_notebook_cell'), None)
    assert fn is not None, 'docs/conf.py has no _install_notebook_cell()'
    src = ast.get_source_segment(conf.read_text(encoding='utf-8'), fn)
    ns = {'os': os, '__file__': str(conf)}
    exec(src, ns)                                    # noqa: S102 (trusted repo file)
    monkeypatch.setenv('READTHEDOCS_GIT_IDENTIFIER', branch)
    return ns['_install_notebook_cell']()


def test_is_release_ref_master_and_version_tags_only():
    # release review: a vX.Y.Z tag build must be RELEASE form (PyPI), not a
    # `@v1.0.0`-from-GitHub preview; dev branches stay preview.
    for ref in ('master', 'v1.0.0', 'v0.8.2', 'v10.20.30'):
        assert acic._is_release_ref(ref), ref
    for ref in ('dev-1.0', 'dev-1.0-refactor', 'HEAD', 'feature/x', '',
                'v1.0', 'v1.0.0rc1', '1.0.0'):
        assert not acic._is_release_ref(ref), ref


@pytest.mark.parametrize('branch', ['master', 'dev-1.0', 'v1.0.0'])
def test_conf_py_gallery_generator_matches_the_script(branch, monkeypatch):
    # docs/conf.py generates the gallery notebooks' install cell; the script
    # does the same for the hand-authored tutorials. They are "kept in sync" by
    # hand -- lock that so the two can't drift (release review).
    conf_cell = _conf_install_cell(branch, monkeypatch)
    # the actual install magic, NOT the "# On release this becomes: %pip
    # install ..." comment line (which also contains 'pip install')
    conf_pip = next(ln for ln in conf_cell.splitlines()
                    if ln.strip().startswith('%pip install'))
    _, script_pip = acic.install_lines(branch)
    assert conf_pip.strip() == script_pip.strip(), (
        f'docs/conf.py and scripts/add_colab_install_cell.py disagree on the '
        f'{branch} install line:\n  conf.py: {conf_pip.strip()}\n  script : '
        f'{script_pip.strip()}')


def test_process_real_tutorial_roundtrip_is_stable(tmp_path):
    """Running the real retarget on a copy of a tracked tutorial is a stable,
    valid-JSON, non-ASCII-preserving operation (no \\uXXXX churn)."""
    src = (pathlib.Path(__file__).resolve().parent.parent
           / 'docs' / 'tutorials' / 'analyze.ipynb')
    original = src.read_text(encoding='utf-8')
    nb = json.loads(original)
    acic.retarget_notebook(nb, 'dev-1.0')          # already dev-1.0 -> no-op
    dst = tmp_path / 'analyze.ipynb'
    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')
    # valid JSON, and literal UTF-8 preserved (curly quotes etc. not escaped)
    round_tripped = dst.read_text(encoding='utf-8')
    assert json.loads(round_tripped) == nb
    assert '\\u2019' not in round_tripped     # no escaped apostrophes
