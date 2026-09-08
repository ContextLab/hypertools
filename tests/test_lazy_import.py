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


# --- every optional import in the library goes through lazy_import ----------

#: modules that import an optional package WITHOUT calling lazy_import for it
#: in the same file, each with the reason that is correct
_LAZY_IMPORT_EXEMPT = {
    # a type check that must never install anything: `import plotly` inside
    # try/except returns False for "not a plotly figure" when plotly is absent
    'hypertools/plot/plot.py': {'plotly'},
    # imports plotly only on the `resolve_backend(backend) == 'plotly'` branch,
    # and resolve_backend() installs the [interactive] extra on demand first
    'hypertools/reduce/describe.py': {'plotly'},
    # a subprocess the plotly backend spawns only after ensure_kaleido_chrome()
    # (which lazy-imports kaleido) succeeded in the parent, so plotly is there
    'hypertools/plot/_kaleido_export_worker.py': {'plotly'},
}


def _optional_imports(source):
    """Top-level names of EXTRA_FOR_MODULE that `source` imports."""
    import ast
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        for name in names:
            top = name.split('.')[0]
            if top in L.EXTRA_FOR_MODULE:
                found.add(top)
    return found


def test_every_optional_import_in_the_library_goes_through_lazy_import():
    """The on-demand installer only helps where the code asks for it: a plain
    `import plotly` in a module that never calls `lazy_import('plotly')` (or
    `ensure_kaleido_chrome()`, which lazy-imports kaleido) would raise
    ImportError before the extra could be installed. Scan every library
    module for imports of the packages EXTRA_FOR_MODULE maps and require the
    same file to install them, unless it is exempt above for a stated
    reason. (Release review 2026-09-07: the audit of leftover install
    instructions asked whether every site really installs on demand.)"""
    pkg = os.path.join(REPO, 'hypertools')
    missing = []
    seen_exempt = set()
    for root, _dirs, files in os.walk(pkg):
        for fn in files:
            if not fn.endswith('.py'):
                continue
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, REPO).replace(os.sep, '/')
            if rel == 'hypertools/_shared/lazy_import.py':
                continue
            with open(path, encoding='utf-8') as f:
                source = f.read()
            for top in sorted(_optional_imports(source)):
                if top in _LAZY_IMPORT_EXEMPT.get(rel, ()):
                    seen_exempt.add((rel, top))
                    continue
                installs = (re.search(r"lazy_import\(\s*['\"]" + top + r"\b", source)
                            or (top == 'kaleido'
                                and 'ensure_kaleido_chrome(' in source))
                if not installs:
                    missing.append((rel, top))
    assert not missing, missing
    # every exemption still describes a real import (no stale entries)
    declared = {(rel, top) for rel, tops in _LAZY_IMPORT_EXEMPT.items()
                for top in tops}
    assert seen_exempt == declared, declared - seen_exempt


def test_the_optional_import_scan_sees_a_plain_import():
    assert _optional_imports('import plotly.graph_objects as go') == {'plotly'}
    assert _optional_imports('from skimage import measure') == {'skimage'}
    # sklearn.datasets is not the `datasets` package
    assert _optional_imports('from sklearn import datasets') == set()
    assert _optional_imports('from .common import Forecaster') == set()


# --- hyp.set_autoinstall: the public switch -----------------------------------

@pytest.fixture
def _restore_autoinstall(monkeypatch):
    """Leave the module-level setting as this test found it."""
    monkeypatch.setattr(L, '_AUTO_INSTALL', L._AUTO_INSTALL)
    monkeypatch.delenv('HYPERTOOLS_AUTO_INSTALL', raising=False)


def test_set_autoinstall_is_public_and_mirrors_set_interactive_backend(_restore_autoinstall):
    import hypertools as hyp
    assert hyp.set_autoinstall is L.set_autoinstall
    assert 'set_autoinstall' in hyp.__all__
    assert L.auto_install_enabled() is True                # the default
    handle = hyp.set_autoinstall(False)                    # called directly
    assert handle.enabled is False and repr(handle) == 'set_autoinstall(False)'
    assert L.auto_install_enabled() is False
    with hyp.set_autoinstall(True) as inner:               # context manager
        assert inner.enabled is True
        assert L.auto_install_enabled() is True
    assert L.auto_install_enabled() is False               # restored
    hyp.set_autoinstall()                                  # default: on
    assert L.auto_install_enabled() is True


def test_set_autoinstall_overrides_the_environment_variable(_restore_autoinstall, monkeypatch):
    import hypertools as hyp
    monkeypatch.setenv('HYPERTOOLS_AUTO_INSTALL', '0')
    assert L.auto_install_enabled() is False               # env sets the start
    with hyp.set_autoinstall(True):
        assert L.auto_install_enabled() is True            # the call wins
    assert L.auto_install_enabled() is False


def test_set_autoinstall_rejects_non_booleans(_restore_autoinstall):
    import hypertools as hyp
    for bad in (1, 'off', None):
        with pytest.raises(TypeError, match='True or False'):
            hyp.set_autoinstall(bad)
    assert L.auto_install_enabled() is True                # nothing changed


def test_set_autoinstall_off_fails_with_the_manual_command_without_pip(_restore_autoinstall, monkeypatch):
    """With installation off, a missing module raises at once with the manual
    command and the way back on; pip is never run (a `subprocess.run` that
    reached pip would install a real package, so the test watches for the
    call by replacing the module's runner with one that fails the test)."""
    import hypertools as hyp

    def _no_pip(*a, **k):
        raise AssertionError('pip must not run with autoinstall off')
    monkeypatch.setattr(L, '_pip_install', _no_pip)
    with hyp.set_autoinstall(False), \
            pytest.raises(ImportError, match=r'hypertools\[kaggle\].*set_autoinstall\(True\)') as info:
        L.lazy_import('hypertools_no_such_module_xyz', purpose='a test', extra='kaggle')
    assert 'automatic installation is off' in str(info.value)
