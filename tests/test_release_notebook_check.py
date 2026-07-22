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


def test_valid_plus_second_malformed_install_is_rejected(tmp_path):
    # release review edge case: a good hypertools[...] install AND a second
    # bare `pip install hypertools` in the same notebook must NOT pass -- EVERY
    # install must be valid, so per-LINE inspection flags the malformed one.
    p = _write(tmp_path, 'mixed.ipynb', _nb(_PYPI, '%pip install -q hypertools'))
    missing, branch, stale = crn.classify_notebooks([p])
    assert missing == ['mixed.ipynb'] and not branch and not stale


def test_valid_plus_second_branch_install_is_rejected(tmp_path):
    p = _write(tmp_path, 'mixed2.ipynb', _nb(_PYPI, _BRANCH))
    missing, branch, stale = crn.classify_notebooks([p])
    assert branch == ['mixed2.ipynb'] and not missing and not stale


def test_mixed_set_each_lands_in_its_bucket(tmp_path):
    good = _write(tmp_path, 'good.ipynb', _nb(_PYPI))
    branch = _write(tmp_path, 'branch.ipynb', _nb(_BRANCH))
    prev = _write(tmp_path, 'prev.ipynb', _nb(_PREVIEW))
    none = _write(tmp_path, 'none.ipynb', _nb('import numpy'))
    m, b, s = crn.classify_notebooks([good, branch, prev, none])
    assert m == ['none.ipynb'] and b == ['branch.ipynb'] and s == ['prev.ipynb']


def test_alternate_install_spellings_are_detected(tmp_path):
    # release review, HIGH bypass: non-`pip install` spellings must NOT evade the
    # check. A branch install written as !pip3 / pipx / pip<TAB> is still a
    # branch install.
    for i, cmd in enumerate((
            '!pip3 install -q "hypertools[interactive] @ git+https://github.com/ContextLab/hypertools.git@dev-1.0"',
            '!pipx install "hypertools[interactive] @ git+https://github.com/ContextLab/hypertools.git@dev-1.0"',
            '%pip\tinstall -q "hypertools[interactive] @ git+https://github.com/ContextLab/hypertools.git@dev-1.0"')):
        p = _write(tmp_path, f'alt{i}.ipynb', _nb(cmd))
        missing, branch, stale = crn.classify_notebooks([p])
        assert branch == [f'alt{i}.ipynb'], (cmd, missing, branch, stale)


def test_documentation_string_is_not_a_false_positive(tmp_path):
    # release review, MEDIUM false-positive: a code cell that merely DOCUMENTS a
    # source install inside a string must not be mistaken for an executed install
    # and block an otherwise-valid release.
    nb = _nb(
        _PYPI,
        'print("to try the dev build: '
        'pip install git+https://github.com/ContextLab/hypertools.git@main")')
    p = _write(tmp_path, 'doc.ipynb', nb)
    assert crn.classify_notebooks([p]) == ([], [], [])


def test_comment_install_line_is_not_detected(tmp_path):
    nb = _nb('# to install: %pip install "hypertools[interactive]"\n' + _PYPI)
    p = _write(tmp_path, 'cmt.ipynb', nb)
    assert crn.classify_notebooks([p]) == ([], [], [])


def test_main_ok_and_min_count(tmp_path):
    _write(tmp_path, 'good.ipynb', _nb(_PYPI))
    assert crn.main([str(tmp_path)]) == 0
    assert crn.main(['--min', '2', str(tmp_path)]) == 1        # only 1 present


def test_main_fails_on_branch_install(tmp_path):
    _write(tmp_path, 'branch.ipynb', _nb(_BRANCH))
    assert crn.main([str(tmp_path)]) == 1
