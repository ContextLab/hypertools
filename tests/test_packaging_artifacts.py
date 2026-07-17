"""Packaging regression tests (audit findings X5-packaging-001/-002,
X7-code-org-rest-004, and final-wave X5-004/-005/-010).

Builds REAL wheel + sdist artifacts (PEP 517, via `python -m build
--no-isolation`) into a temp directory and inspects their contents:

- hypertools/core/config.ini must ship in both artifacts (without it,
  ``get_default_options()`` was silently empty in every pip install)
- no stray virtualenv content (a local ``hypertools-dev/`` venv used to be
  swept into both artifacts as a namespace package, installing
  ``hypertools-dev/bin/activate_this.py`` into users' site-packages)
- the wheel's top_level.txt must list only ``hypertools``
- the sdist must ship the FULL runnable tests tree, subdirectories and
  data files included (X5-004: only the flat tests/test_*.py files
  shipped, so ``pytest`` failed collection from an unpacked sdist)
- the wheel METADATA long-description must contain no relative markdown
  links (X5-005: 8 ``](images/...)`` links rendered broken on PyPI)
- the license must ship as a PEP 639 SPDX expression with the LICENSE
  file in the wheel, and the build must emit no license deprecation
  warning (X5-010: the ``license = { text = ... }`` table form is
  deprecated)

Defaults must also load path-independently (not relative to the cwd).
"""
import os
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def built_artifacts(tmp_path_factory):
    """(wheel path, sdist path, combined build stdout+stderr)."""
    outdir = tmp_path_factory.mktemp('dist')
    result = subprocess.run(
        [sys.executable, '-m', 'build', '--no-isolation',
         '--outdir', str(outdir), str(REPO_ROOT)],
        capture_output=True, text=True, timeout=580,
        env=dict(os.environ, MPLBACKEND='Agg'))
    assert result.returncode == 0, \
        f'build failed:\n{result.stdout}\n{result.stderr}'
    wheels = list(outdir.glob('*.whl'))
    sdists = list(outdir.glob('*.tar.gz'))
    assert len(wheels) == 1 and len(sdists) == 1, list(outdir.iterdir())
    return wheels[0], sdists[0], result.stdout + '\n' + result.stderr


def test_wheel_ships_config_ini(built_artifacts):
    wheel, _, _ = built_artifacts
    names = zipfile.ZipFile(wheel).namelist()
    assert 'hypertools/core/config.ini' in names


def test_sdist_ships_config_ini(built_artifacts):
    _, sdist, _ = built_artifacts
    names = tarfile.open(sdist).getnames()
    assert any(n.endswith('hypertools/core/config.ini') for n in names)


def test_no_stray_venv_content_in_wheel(built_artifacts):
    wheel, _, _ = built_artifacts
    names = zipfile.ZipFile(wheel).namelist()
    offenders = [n for n in names
                 if 'hypertools-dev' in n or 'activate_this' in n]
    assert offenders == []


def test_no_stray_venv_content_in_sdist(built_artifacts):
    _, sdist, _ = built_artifacts
    names = tarfile.open(sdist).getnames()
    offenders = [n for n in names
                 if 'hypertools-dev' in n or 'activate_this' in n]
    assert offenders == []


def test_wheel_top_level_lists_only_hypertools(built_artifacts):
    wheel, _, _ = built_artifacts
    zf = zipfile.ZipFile(wheel)
    top_level = [n for n in zf.namelist() if n.endswith('top_level.txt')]
    assert len(top_level) == 1
    contents = zf.read(top_level[0]).decode().split()
    assert contents == ['hypertools']


# ---- X5-004: the sdist tests/ tree must be runnable, not a flat shard ----

def test_sdist_ships_full_tests_tree(built_artifacts):
    """Only the flat tests/test_*.py files used to ship; the per-module
    subdirectories, data files, matplotlibrc, and the impute conftest were
    all dropped, so ``pytest`` could not run from an unpacked sdist."""
    _, sdist, _ = built_artifacts
    names = tarfile.open(sdist).getnames()
    for sub in ('align', 'cluster', 'core', 'external', 'impute', 'io',
                'manip', 'plot', 'predict', 'reduce'):
        assert any(f'/tests/{sub}/' in n and n.endswith('.py')
                   for n in names), f'tests/{sub}/ missing from sdist'
    assert any(n.endswith('tests/impute/conftest.py') for n in names), \
        'tests/impute/conftest.py missing from sdist'
    assert any(n.endswith('tests/matplotlibrc') for n in names), \
        'tests/matplotlibrc missing from sdist'
    assert any('/tests/data/' in n for n in names), \
        'tests/data/ missing from sdist'


def test_sdist_tests_tree_carries_no_caches(built_artifacts):
    _, sdist, _ = built_artifacts
    names = tarfile.open(sdist).getnames()
    offenders = [n for n in names
                 if '__pycache__' in n or '.pytest_cache' in n
                 or n.endswith(('.pyc', '.pyo'))]
    assert offenders == []


# ---- X5-005: PyPI-rendered description must not use relative links ----

def _wheel_metadata(wheel):
    zf = zipfile.ZipFile(wheel)
    meta = [n for n in zf.namelist()
            if re.fullmatch(r'[^/]+\.dist-info/METADATA', n)]
    assert len(meta) == 1, meta
    return zf.read(meta[0]).decode('utf-8')


def test_wheel_metadata_has_no_relative_links(built_artifacts):
    """The METADATA long-description (the readme) is rendered on PyPI,
    where relative ``](images/...)`` links 404 (8 shipped broken)."""
    wheel, _, _ = built_artifacts
    metadata = _wheel_metadata(wheel)
    relative = re.findall(r'\]\((?!https?://|#|mailto:)[^)]+\)', metadata)
    assert relative == [], f'relative links in wheel METADATA: {relative}'
    # the readme images must arrive as absolute raw.githubusercontent links
    assert ('](https://raw.githubusercontent.com/ContextLab/hypertools/'
            in metadata)


# ---- X5-010: PEP 639 SPDX license, no deprecated table form ----

def test_wheel_license_is_spdx_expression(built_artifacts):
    wheel, _, _ = built_artifacts
    assert 'License-Expression: MIT' in _wheel_metadata(wheel)
    names = zipfile.ZipFile(wheel).namelist()
    assert any(re.fullmatch(r'[^/]+\.dist-info/licenses/LICENSE', n)
               for n in names), 'LICENSE file missing from wheel dist-info'


def test_build_emits_no_license_deprecation_warning(built_artifacts):
    """The old ``license = { text = "MIT" }`` TOML-table form made
    setuptools emit a SetuptoolsDeprecationWarning on every build."""
    _, _, build_log = built_artifacts
    offenders = [line for line in build_log.splitlines()
                 if 'license' in line.lower()
                 and ('deprecat' in line.lower() or 'Warning' in line)]
    assert offenders == [], offenders


def test_default_options_load_path_independently(tmp_path):
    """get_default_options() must not depend on the process cwd (it should
    resolve the bundled config.ini through the package itself)."""
    code = (
        "from hypertools.core.configurator import get_default_options\n"
        "opts = get_default_options()\n"
        "assert opts['reduce'] != {}, 'reduce defaults empty'\n"
        "assert opts['cluster']['n_clusters'] == 3, opts['cluster']\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, '-c', code],
                            cwd=str(tmp_path), capture_output=True,
                            text=True, timeout=300,
                            env=dict(os.environ, MPLBACKEND='Agg'))
    assert result.returncode == 0, result.stderr
    assert 'OK' in result.stdout
