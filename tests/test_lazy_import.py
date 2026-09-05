"""``hypertools._shared.lazy_import``: optional dependencies are installed on
demand, from the ONE declaration of the extras in pyproject.toml.
"""

import os
import shutil
import subprocess
import re
import sys

import pytest

try:
    import tomllib                      # Python 3.11+
except ImportError:                     # 3.10: parse the one table we need
    tomllib = None

from hypertools._shared import lazy_import as L
from tests._netskip import skip_on_transient_network

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pyproject_extras():
    path = os.path.join(REPO, 'pyproject.toml')
    if tomllib is not None:
        with open(path, 'rb') as f:
            return tomllib.load(f)['project']['optional-dependencies']
    # Python 3.10 has no tomllib: read the [project.optional-dependencies]
    # table by hand. Its values are `name = ["spec", ...]` lists; comments and
    # bracketed extras such as "pydata-wrangler[hf]" both contain `]`, so
    # strip comments first and then walk each list tracking quotes and depth.
    with open(path, encoding='utf-8') as f:
        text = f.read()
    table = text.split('[project.optional-dependencies]', 1)[1]
    table = re.sub(r'#[^\n]*', '', table)
    extras = {}
    for m in re.finditer(r'^([\w-]+)\s*=\s*\[', table, re.M):
        i, depth, in_str = m.end(), 1, False
        while depth:
            ch = table[i]
            if ch == '"':
                in_str = not in_str
            elif not in_str and ch == '[':
                depth += 1
            elif not in_str and ch == ']':
                depth -= 1
            i += 1
        extras[m.group(1)] = re.findall(r'"([^"]+)"', table[m.end():i - 1])
        if re.match(r'\s*\n\[', table[i:]):          # next table starts
            break
    return extras


def test_extra_requirements_are_read_from_pyproject_not_a_second_list():
    declared = _pyproject_extras()
    for extra in sorted(set(L.EXTRA_FOR_MODULE.values())):
        assert L.extra_requirements(extra) == declared[extra], extra


def test_every_mapped_module_points_at_a_declared_extra():
    declared = _pyproject_extras()
    for module, extra in L.EXTRA_FOR_MODULE.items():
        assert extra in declared, (module, extra)
    with pytest.raises(ValueError, match='no optional extra'):
        L.extra_requirements('no-such-extra')


def test_lazy_import_returns_the_module_when_it_is_installed():
    pytest.importorskip('plotly')
    import plotly
    assert L.lazy_import('plotly') is plotly
    assert L.lazy_import('plotly.io').__name__ == 'plotly.io'


def test_disabled_auto_install_fails_with_the_manual_command(monkeypatch):
    monkeypatch.setenv('HYPERTOOLS_AUTO_INSTALL', '0')
    with pytest.raises(ImportError) as info:
        L.lazy_import('hypertools_no_such_module_xyz', purpose='a test', extra='kaggle')
    assert 'pip install "hypertools[kaggle]"' in str(info.value)
    assert 'a test' in str(info.value)


def test_a_module_no_extra_provides_fails_without_installing_anything():
    with pytest.raises(ImportError, match='declares no extra'):
        L.lazy_import('hypertools_no_such_module_xyz')


def test_lazy_import_installs_a_missing_package_into_a_fresh_interpreter(tmp_path):
    """A REAL install: a throwaway venv that lacks `tomli` imports it through
    lazy_import, which pip-installs it first (explicit requirements, since the
    venv has no hypertools metadata)."""
    venv = tmp_path / 'venv'
    subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True)
    py = venv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    shutil.copy(L.__file__, tmp_path / 'lazy_import.py')
    code = ("import lazy_import as L; m = L.lazy_import('tomli', purpose='a test', "
            "requirements=['tomli']); print('imported', m.__name__)")
    with skip_on_transient_network('pip install into a throwaway venv'):
        out = subprocess.run([str(py), '-c', code], cwd=tmp_path, capture_output=True,
                             text=True, timeout=600)
        if out.returncode != 0 and ('Connection' in out.stderr or 'Temporary failure' in out.stderr
                                    or 'Read timed out' in out.stderr):
            pytest.skip(f'transient network error installing tomli: {out.stderr[-200:]}')
    assert out.returncode == 0, out.stderr[-800:]
    assert 'hypertools: installing tomli (needed for a test)' in out.stdout
    assert 'imported tomli' in out.stdout


def test_ensure_kaleido_chrome_leaves_plotly_able_to_render():
    pytest.importorskip('plotly')
    pytest.importorskip('kaleido')
    import plotly.graph_objects as go
    import plotly.io as pio
    L.ensure_kaleido_chrome()
    assert len(pio.to_image(go.Figure(), format='png')) > 1000
    assert L._kaleido_ready
