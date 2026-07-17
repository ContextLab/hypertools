"""Packaging regression tests (audit findings X5-packaging-001/-002,
X7-code-org-rest-004).

Builds REAL wheel + sdist artifacts (PEP 517, via `python -m build
--no-isolation`) into a temp directory and inspects their contents:

- hypertools/core/config.ini must ship in both artifacts (without it,
  ``get_default_options()`` was silently empty in every pip install)
- no stray virtualenv content (a local ``hypertools-dev/`` venv used to be
  swept into both artifacts as a namespace package, installing
  ``hypertools-dev/bin/activate_this.py`` into users' site-packages)
- the wheel's top_level.txt must list only ``hypertools``

Defaults must also load path-independently (not relative to the cwd).
"""
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def built_artifacts(tmp_path_factory):
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
    return wheels[0], sdists[0]


def test_wheel_ships_config_ini(built_artifacts):
    wheel, _ = built_artifacts
    names = zipfile.ZipFile(wheel).namelist()
    assert 'hypertools/core/config.ini' in names


def test_sdist_ships_config_ini(built_artifacts):
    _, sdist = built_artifacts
    names = tarfile.open(sdist).getnames()
    assert any(n.endswith('hypertools/core/config.ini') for n in names)


def test_no_stray_venv_content_in_wheel(built_artifacts):
    wheel, _ = built_artifacts
    names = zipfile.ZipFile(wheel).namelist()
    offenders = [n for n in names
                 if 'hypertools-dev' in n or 'activate_this' in n]
    assert offenders == []


def test_no_stray_venv_content_in_sdist(built_artifacts):
    _, sdist = built_artifacts
    names = tarfile.open(sdist).getnames()
    offenders = [n for n in names
                 if 'hypertools-dev' in n or 'activate_this' in n]
    assert offenders == []


def test_wheel_top_level_lists_only_hypertools(built_artifacts):
    wheel, _ = built_artifacts
    zf = zipfile.ZipFile(wheel)
    top_level = [n for n in zf.namelist() if n.endswith('top_level.txt')]
    assert len(top_level) == 1
    contents = zf.read(top_level[0]).decode().split()
    assert contents == ['hypertools']


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
