"""Unit tests for scripts/check_release_notebooks.py.

The docs-clean CI job uses this to verify that every GENERATED gallery notebook
installs the released PyPI package. These tests pin the classification so the
guarantee ("every published notebook installs hypertools[...], no branch
install, no preview note") is covered independently of the workflow YAML.
"""

import importlib.util
import json
import pathlib

_SCRIPT = (pathlib.Path(__file__).resolve().parent.parent
           / 'scripts' / 'check_release_notebooks.py')


def _load():
    spec = importlib.util.spec_from_file_location('check_release_notebooks',
                                                  _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


crn = _load()

_PYPI = '# Install hypertools (run this first on Colab)\n%pip install -q "hypertools[interactive]"'
_BRANCH = '%pip install -q "hypertools[interactive] @ git+https://github.com/ContextLab/hypertools.git@dev-1.0"'
_PREVIEW = ('# Install hypertools (dev-1.0 preview) -- run this first on Colab.\n'
            '# On release this becomes: %pip install hypertools\n'
            '%pip install -q "hypertools[interactive]"')


def _nb(*code_sources):
    return {'cells': [{'cell_type': 'code', 'source': [s]} for s in code_sources],
            'metadata': {}, 'nbformat': 4, 'nbformat_minor': 5}


def _write(tmp_path, name, nb):
    p = tmp_path / name
    p.write_text(json.dumps(nb), encoding='utf-8')
    return str(p)


def test_good_pypi_notebook_passes(tmp_path):
    p = _write(tmp_path, 'good.ipynb', _nb('%matplotlib inline', _PYPI))
    assert crn.classify_notebooks([p]) == ([], [], [])


def test_branch_install_flagged(tmp_path):
    p = _write(tmp_path, 'branch.ipynb', _nb(_BRANCH))
    missing, branch, stale = crn.classify_notebooks([p])
    assert branch == ['branch.ipynb'] and not missing and not stale


def test_preview_note_flagged_even_with_pypi_install(tmp_path):
    p = _write(tmp_path, 'prev.ipynb', _nb(_PREVIEW))
    missing, branch, stale = crn.classify_notebooks([p])
    assert stale == ['prev.ipynb'] and not missing and not branch


def test_no_hypertools_install_flagged_missing(tmp_path):
    p = _write(tmp_path, 'none.ipynb', _nb('%pip install -q numpy', 'import hypertools'))
    missing, branch, stale = crn.classify_notebooks([p])
    assert missing == ['none.ipynb'] and not branch and not stale


def test_non_bracket_install_flagged_missing(tmp_path):
    # a hypertools install that is NOT the hypertools[...] extras form
    p = _write(tmp_path, 'bare.ipynb', _nb('%pip install -q hypertools'))
    missing, branch, stale = crn.classify_notebooks([p])
    assert missing == ['bare.ipynb']


def test_mixed_set_each_lands_in_its_bucket(tmp_path):
    good = _write(tmp_path, 'good.ipynb', _nb(_PYPI))
    branch = _write(tmp_path, 'branch.ipynb', _nb(_BRANCH))
    prev = _write(tmp_path, 'prev.ipynb', _nb(_PREVIEW))
    none = _write(tmp_path, 'none.ipynb', _nb('import numpy'))
    m, b, s = crn.classify_notebooks([good, branch, prev, none])
    assert m == ['none.ipynb'] and b == ['branch.ipynb'] and s == ['prev.ipynb']


def test_main_ok_and_min_count(tmp_path):
    _write(tmp_path, 'good.ipynb', _nb(_PYPI))
    assert crn.main([str(tmp_path)]) == 0
    assert crn.main(['--min', '2', str(tmp_path)]) == 1        # only 1 present


def test_main_fails_on_branch_install(tmp_path):
    _write(tmp_path, 'branch.ipynb', _nb(_BRANCH))
    assert crn.main([str(tmp_path)]) == 1
