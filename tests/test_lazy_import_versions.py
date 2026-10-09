"""``lazy_import`` checks the VERSION of an installed optional extra, not only
that it imports (release red-team review 2026-10-09).

The reviewer's environment: real plotly 5.24.1 (below the declared
``plotly>=6.1.1``) beside kaleido 1.3.0. ``lazy_import`` returned whatever
imported, so a static export failed with plotly's own text -- "Image export
using the "kaleido" engine requires the kaleido package" -- although kaleido
was installed and the real cause was the old plotly. A resolved
``pip install "hypertools[interactive]"`` never produces that pair; an
environment with an older plotly already in it (a notebook image) does.

Everything here runs against REAL installed distributions: a requirement the
installed version satisfies, a deliberately higher one (``numpy>=999``) for
the too-old branch, a real pip upgrade inside a throwaway venv, and (under
the ``bigdata`` marker, deselected by default) the reviewer's plotly 5.24.1
environment itself.
"""

import os
import shutil
import subprocess
import sys
from importlib import metadata

import pytest

from hypertools._shared import lazy_import as L
from tests._netskip import skip_on_transient_network

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def _restore_autoinstall(monkeypatch):
    """Leave the module-level setting as this test found it."""
    monkeypatch.setattr(L, '_AUTO_INSTALL_SCOPES', list(L._AUTO_INSTALL_SCOPES))
    monkeypatch.setattr(L, '_AUTO_INSTALL_BASELINE', [None, L._AUTO_INSTALL_BASELINE[1]])
    monkeypatch.delenv('HYPERTOOLS_AUTO_INSTALL', raising=False)


# --- the check itself, on real installed distributions ------------------------

def test_a_satisfied_requirement_is_not_reported():
    assert L.outdated_requirements(['numpy>=1.0']) == []
    assert L.outdated_requirements([f'numpy>={metadata.version("numpy")}']) == []
    assert L.outdated_requirements([f'numpy=={metadata.version("numpy")}']) == []


def test_a_requirement_above_the_installed_version_is_reported():
    have = metadata.version('numpy')
    assert L.outdated_requirements(['numpy>=999']) == [('numpy>=999', 'numpy', have)]
    # every requirement is checked, and only the failing ones come back
    assert L.outdated_requirements(['scipy>=1.0', 'numpy>=999', 'pandas>999']) == [
        ('numpy>=999', 'numpy', have),
        ('pandas>999', 'pandas', metadata.version('pandas'))]


def test_distribution_names_that_differ_from_import_names_do_not_alarm():
    # scikit-learn imports as sklearn, pillow as PIL; the check goes by the
    # DISTRIBUTION name the requirement declares, in any spelling pip accepts
    assert L.outdated_requirements(
        ['scikit-learn>=0.1', 'scikit_learn>=0.1', 'Scikit.Learn>=0.1',
         'pillow>=1', 'pydata-wrangler>=0.1', 'pydata-wrangler[hf]>=0.1']) == []
    have = metadata.version('scikit-learn')
    assert L.outdated_requirements(['scikit_learn>=999']) == [
        ('scikit_learn>=999', 'scikit_learn', have)]
    assert L.outdated_requirements(['pydata-wrangler[hf]>=999']) == [
        ('pydata-wrangler[hf]>=999', 'pydata-wrangler',
         metadata.version('pydata-wrangler'))]


def test_a_distribution_that_is_not_installed_is_not_too_old():
    # "missing" is the import's business (it is installed on demand), not
    # the version check's
    assert L.outdated_requirements(['hypertools-no-such-dist-xyz>=3']) == []


def test_an_upper_bound_or_an_unparseable_requirement_is_not_too_old():
    # only a FLOOR can make an installed version "too old"
    assert L.outdated_requirements(['numpy<0.1', 'numpy!=' + metadata.version('numpy')]) == []
    assert L.outdated_requirements(['this is not a requirement >= 3']) == []
    # a marker that does not apply to this interpreter
    assert L.outdated_requirements(['numpy>=999; python_version < "3.0"']) == []


@pytest.mark.parametrize('installed, requirement, ok', [
    ('5.24.1', 'plotly>=6.1.1', False),            # the reviewer's pair
    ('6.1.0', 'plotly>=6.1.1', False),
    ('6.1.1', 'plotly>=6.1.1', True),
    ('6.3.0', 'plotly>=6.1.1', True),
    ('0.2.1', 'kaleido>=1.0', False),
    ('1.0.0', 'kaleido>=1.0', True),
    # a development / pre-release / local build OF the floor version is that
    # version for this purpose
    ('6.1.1.dev0', 'plotly>=6.1.1', True),
    ('6.1.1rc1', 'plotly>=6.1.1', True),
    ('6.1.1.dev3+g1a2b3c4', 'plotly>=6.1.1', True),
    ('7.0.0a1', 'plotly>=6.1.1', True),
    ('6.1.1.post1', 'plotly>=6.1.1', True),
    ('6.1.0rc2', 'plotly>=6.1.1', False),
    # "unknown version" placeholders (a source tree with no tag, a broken
    # build) carry no information: never an alarm
    ('0+unknown', 'plotly>=6.1.1', True),
    ('0.0.0', 'plotly>=6.1.1', True),
    ('0.0.0.dev0+g1a2b3c4', 'plotly>=6.1.1', True),
    ('not-a-version', 'plotly>=6.1.1', True),
    ('', 'plotly>=6.1.1', True),
    # other floor-setting operators
    ('2.0', 'x>2.0', False),
    ('2.1', 'x>2.0', True),
    ('2.3', 'x~=2.4', False),
    ('2.5', 'x~=2.4', True),
    ('2.3', 'x==2.4', False),
    ('2.4', 'x==2.4', True),
    ('2.3', 'x==2.*', True),                       # no single floor to compare
    ('1.0', 'x>=2.0,<3', False),
    ('9.0', 'x>=2.0,<3', True),                    # too NEW is not too old
    ('5.0', 'x', True),
])
def test_floor_comparison(installed, requirement, ok):
    assert L._meets_floor(installed, requirement) is ok


def test_every_extra_installed_in_this_environment_meets_its_declared_floor():
    """The development environment is a correctly resolved install: the
    check must be silent for every extra hypertools declares."""
    for extra in sorted(set(L.EXTRA_FOR_MODULE.values())):
        assert L.outdated_requirements(L.extra_requirements(extra)) == [], extra


# --- lazy_import: an installed-but-too-old requirement ------------------------

def test_too_old_with_installation_off_names_versions_and_the_command(_restore_autoinstall, capsys):
    import hypertools as hyp
    have = metadata.version('numpy')
    with hyp.set_autoinstall(False), pytest.raises(ImportError) as info:
        L.lazy_import('numpy', purpose='a test', extra='interactive',
                      requirements=['numpy>=999'])
    msg = str(info.value)
    assert f'numpy {have} is installed' in msg
    assert 'numpy>=999' in msg
    assert 'a test' in msg
    assert L.install_command('interactive') in msg
    assert 'pip install "hypertools[interactive]"' in msg
    assert 'automatic installation is off' in msg
    assert 'set_autoinstall(True)' in msg
    assert 'not installed' not in msg              # it IS installed: say so
    out = capsys.readouterr()
    assert out.out == '' and out.err == ''         # no notice: pip never ran


def test_too_old_without_an_extra_quotes_the_requirements_in_the_command(_restore_autoinstall):
    import hypertools as hyp
    with hyp.set_autoinstall(False), pytest.raises(ImportError) as info:
        L.lazy_import('numpy', requirements=['numpy>=999'])
    # quoted: an unquoted `>=999` is a shell redirection
    assert '`pip install "numpy>=999"`' in str(info.value)


def test_too_old_and_already_imported_is_not_upgraded_in_place(_restore_autoinstall, capsys):
    """numpy is imported in this process. A running interpreter cannot swap
    an imported package, so with installation ON nothing is installed
    either: the error says so and asks for a restart."""
    import numpy
    import hypertools as hyp
    have = metadata.version('numpy')
    hyp.set_autoinstall(True)
    with pytest.raises(ImportError) as info:
        L.lazy_import('numpy', purpose='a test', extra='interactive',
                      requirements=['numpy>=999'])
    msg = str(info.value)
    assert f'numpy {have} is installed' in msg and 'numpy>=999' in msg
    assert 'already imported' in msg
    assert 'restart' in msg.lower()
    assert 'pip install "hypertools[interactive]"' in msg
    out = capsys.readouterr()
    assert 'hypertools: upgrading' not in out.out   # pip never ran
    assert 'hypertools: installing' not in out.out
    assert metadata.version('numpy') == have
    assert sys.modules['numpy'] is numpy


def test_every_requirement_of_the_extra_is_checked_not_only_the_module(_restore_autoinstall):
    """The feature needs the extra's packages TOGETHER (plotly with kaleido):
    asking for one module reports another requirement that is too old."""
    import hypertools as hyp
    with hyp.set_autoinstall(False), pytest.raises(ImportError) as info:
        L.lazy_import('scipy', purpose='a test',
                      requirements=['scipy>=1.0', 'numpy>=999'])
    msg = str(info.value)
    assert f'numpy {metadata.version("numpy")} is installed' in msg
    assert 'scipy' not in msg.split('`')[0]         # scipy itself is fine


def test_a_too_old_result_is_never_cached(_restore_autoinstall):
    import hypertools as hyp
    for _ in range(2):
        with hyp.set_autoinstall(False), pytest.raises(ImportError, match='numpy>=999'):
            L.lazy_import('numpy', requirements=['numpy>=999'])


def test_a_verified_extra_is_checked_once_per_process():
    """The fast path: after one successful check the (module, extra,
    requirements) key is remembered and the import returns directly."""
    pytest.importorskip('plotly')
    import plotly
    assert L.lazy_import('plotly') is plotly
    key = ('plotly', 'interactive', None)
    assert key in L._VERSIONS_VERIFIED
    assert L.lazy_import('plotly.io').__name__ == 'plotly.io'
    assert L.lazy_import('plotly') is plotly
    # explicit requirements are their own key
    import numpy
    assert L.lazy_import('numpy', requirements=['numpy>=1.0']) is numpy
    assert ('numpy', None, ('numpy>=1.0',)) in L._VERSIONS_VERIFIED


def test_a_module_no_extra_declares_is_imported_without_a_version_check():
    import json
    assert L.lazy_import('json') is json


# --- the plotly backend asks lazy_import BEFORE plotly is imported -------------

def _run(code, **kw):
    env = dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get('PYTHONPATH', ''))
    return subprocess.run([sys.executable, '-c', code], capture_output=True,
                          text=True, timeout=600, env=env, **kw)


def test_the_plotly_figure_type_check_does_not_import_plotly():
    """An upgrade is only possible while plotly is NOT yet imported, so
    nothing may import it ahead of `lazy_import('plotly')`. The type check
    `_is_plotly_figure` used to (`from plotly.basedatatypes import ...`)."""
    out = _run("import sys\n"
               "import hypertools as hyp\n"
               "from hypertools.plot.plot import _is_plotly_figure\n"
               "assert not _is_plotly_figure(object())\n"
               "assert 'plotly' not in sys.modules, 'plotly was imported'\n"
               "import plotly.graph_objects as go\n"
               "assert _is_plotly_figure(go.Figure())\n"
               "print('ok')\n")
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == 'ok'


def test_resolve_backend_checks_the_version_before_importing_plotly():
    """`resolve_backend('plotly')` used to `import plotly` itself and call
    lazy_import only when that failed, so an installed old plotly was
    imported unchecked. Now the extra is verified first, in a fresh
    interpreter where plotly is not imported yet."""
    pytest.importorskip('plotly')
    out = _run("import sys\n"
               "import hypertools as hyp\n"
               "from hypertools._shared import lazy_import as L\n"
               "from hypertools.plot.plotly_backend import resolve_backend\n"
               "assert 'plotly' not in sys.modules, 'plotly imported at import time'\n"
               "assert not L._VERSIONS_VERIFIED\n"
               "assert resolve_backend('plotly') == 'plotly'\n"
               "assert ('plotly', 'interactive', None) in L._VERSIONS_VERIFIED\n"
               "assert 'plotly' in sys.modules\n"
               "hyp.set_interactive_backend('plotly')\n"
               "assert resolve_backend('auto') == 'plotly'\n"
               "print('ok')\n")
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == 'ok'


# --- a REAL upgrade, in a throwaway interpreter --------------------------------

def _venv_python(venv):
    return venv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def _pip_failed_on_network(out):
    return out.returncode != 0 and any(
        s in out.stderr for s in ('Connection', 'Temporary failure',
                                  'Read timed out', 'Could not resolve'))


def test_lazy_import_upgrades_a_too_old_package_in_a_fresh_interpreter(tmp_path):
    """A throwaway venv holds a REAL old `tomli` (2.0.0). Three fresh
    interpreters, each a real process:

    1. without `packaging` the check cannot run and must not alarm: the old
       module is returned as before;
    2. with `packaging`, and tomli already imported: nothing is installed,
       the error asks for a restart, and 2.0.0 is still on disk;
    3. with `packaging`, tomli not yet imported: pip upgrades it, the notice
       names the installed version and the requirement, and the NEW module
       is what is imported.
    """
    venv = tmp_path / 'venv'
    subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True)
    py = _venv_python(venv)
    shutil.copy(L.__file__, tmp_path / 'lazy_import.py')

    def pip(*args):
        with skip_on_transient_network('pip install into a throwaway venv'):
            out = subprocess.run([str(py), '-m', 'pip', 'install', '-q', *args],
                                 capture_output=True, text=True, timeout=600)
            if _pip_failed_on_network(out):
                pytest.skip(f'transient network error: {out.stderr[-200:]}')
        assert out.returncode == 0, out.stderr[-800:]

    def run(code):
        return subprocess.run([str(py), '-c', code], cwd=tmp_path,
                              capture_output=True, text=True, timeout=600)

    version = "from importlib import metadata; print('on disk', metadata.version('tomli'))"
    call = ("L.lazy_import('tomli', purpose='a test', "
            "requirements=['tomli>=2.0.1'])")

    pip('tomli==2.0.0')
    # 1. no `packaging` in this interpreter: no check, no alarm
    out = run("import importlib.util as u; assert u.find_spec('packaging') is None\n"
              f"import lazy_import as L; m = {call}\n"
              "print('imported', m.__name__); " + version)
    assert out.returncode == 0, out.stderr[-800:]
    assert 'imported tomli' in out.stdout and 'on disk 2.0.0' in out.stdout
    assert 'hypertools:' not in out.stdout

    pip('packaging')
    # 2. already imported: not upgraded in place
    out = run("import tomli, lazy_import as L\n"
              "try:\n"
              f"    {call}\n"
              "except ImportError as e:\n"
              "    print('ERROR', e)\n" + version)
    assert out.returncode == 0, out.stderr[-800:]
    assert 'ERROR tomli 2.0.0 is installed' in out.stdout
    assert 'tomli>=2.0.1' in out.stdout
    assert 'already imported' in out.stdout and 'restart' in out.stdout.lower()
    assert '`pip install "tomli>=2.0.1"`' in out.stdout
    assert 'hypertools: upgrading' not in out.stdout
    assert 'on disk 2.0.0' in out.stdout

    # 2b. installation off: the policy error, nothing installed
    out = run("import os; os.environ['HYPERTOOLS_AUTO_INSTALL'] = '0'\n"
              "import lazy_import as L\n"
              "try:\n"
              f"    {call}\n"
              "except ImportError as e:\n"
              "    print('ERROR', e)\n" + version)
    assert out.returncode == 0, out.stderr[-800:]
    assert 'ERROR tomli 2.0.0 is installed' in out.stdout
    assert 'automatic installation is off' in out.stdout
    assert 'restart' not in out.stdout.lower()     # not imported: no restart needed
    assert 'on disk 2.0.0' in out.stdout

    # 3. a real upgrade, then the new module
    with skip_on_transient_network('pip upgrade inside a throwaway venv'):
        out = run(f"import lazy_import as L; m = {call}\n"
                  "import sys; print('imported', m.__name__)\n" + version)
        if _pip_failed_on_network(out):
            pytest.skip(f'transient network error: {out.stderr[-200:]}')
    assert out.returncode == 0, out.stderr[-800:]
    assert ('hypertools: upgrading tomli 2.0.0 to tomli>=2.0.1 '
            '(needed for a test) ...') in out.stdout
    assert 'imported tomli' in out.stdout
    assert 'on disk 2.0.0' not in out.stdout
    from packaging.version import Version
    on_disk = out.stdout.split('on disk ')[1].split()[0]
    assert Version(on_disk) >= Version('2.0.1')


# --- the reviewer's environment: plotly 5.24.1 + kaleido 1.3.0 -----------------

_PROBE = r'''
import importlib.metadata as md, json, os, sys
import numpy as np
mode, out = sys.argv[1], sys.argv[2]
result = {'plotly_before': md.version('plotly'), 'kaleido': md.version('kaleido')}
if mode == 'preimported':
    import plotly                               # the reviewer's probe did this
import hypertools as hyp
result['source'] = hyp.__file__
hyp.set_autoinstall(mode != 'off')
try:
    hyp.plot(np.arange(30.).reshape(10, 3), reduce=None, backend='plotly',
             show=False, save_path=out)
    result.update(status='success', size=os.path.getsize(out))
except Exception as exc:
    result.update(status='error', exception=type(exc).__name__, message=str(exc))
result['plotly_after'] = md.version('plotly')
print('RESULT ' + json.dumps(result))
'''

_RENDER_PROBE = r'''
import plotly
from hypertools._shared import lazy_import as L
try:
    print('RENDER', L._kaleido_can_render())
except Exception as exc:
    print('RENDER-ERROR', type(exc).__name__, str(exc))
'''


@pytest.mark.bigdata
def test_old_plotly_environment_is_diagnosed_and_then_upgraded(tmp_path):
    """The reviewer's environment, built for real: this checkout installed
    into a throwaway venv beside plotly 5.24.1 and kaleido 1.3.0, then a
    plotly static export through `hyp.plot` in fresh subprocesses.

    Before the fix every mode failed with plotly's "Image export using the
    "kaleido" engine requires the kaleido package". Deselected by default
    (it resolves and installs the full dependency set, and the last step
    may download Chrome for kaleido): `pytest -m bigdata`.
    """
    import json
    venv = tmp_path / 'venv'
    uv = shutil.which('uv')
    packages = ['-e', REPO, 'plotly==5.24.1', 'kaleido==1.3.0']
    with skip_on_transient_network('building the old-plotly environment'):
        if uv:
            subprocess.run([uv, 'venv', '-q', '--seed', '--python', sys.executable,
                            str(venv)], check=True, timeout=600)
            build = subprocess.run(
                [uv, 'pip', 'install', '-q', '--python', str(_venv_python(venv)),
                 *packages], capture_output=True, text=True, timeout=1800)
        else:
            subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True)
            build = subprocess.run(
                [str(_venv_python(venv)), '-m', 'pip', 'install', '-q', *packages],
                capture_output=True, text=True, timeout=1800)
        if _pip_failed_on_network(build) or 'error sending request' in build.stderr:
            pytest.skip(f'transient network error: {build.stderr[-300:]}')
    assert build.returncode == 0, build.stderr[-2000:]
    py = _venv_python(venv)
    probe = tmp_path / 'probe.py'
    probe.write_text(_PROBE)
    env = {k: v for k, v in os.environ.items()
           if k not in ('PYTHONPATH', 'HYPERTOOLS_AUTO_INSTALL')}

    def run(mode):
        out = subprocess.run([str(py), str(probe), mode, str(tmp_path / f'{mode}.png')],
                             capture_output=True, text=True, timeout=1800,
                             cwd=tmp_path, env=env)
        lines = [ln for ln in out.stdout.splitlines() if ln.startswith('RESULT ')]
        assert lines, (out.stdout[-1000:], out.stderr[-2000:])
        return json.loads(lines[-1][7:]), out

    upgrade = 'pip install "hypertools[interactive]"'

    # installation off: the real cause, the versions, and the command
    res, _ = run('off')
    assert os.path.samefile(os.path.dirname(os.path.dirname(res['source'])), REPO)
    assert res['plotly_before'] == '5.24.1' and res['kaleido'] == '1.3.0'
    assert res['status'] == 'error' and res['exception'] == 'ImportError', res
    assert 'plotly 5.24.1 is installed' in res['message'], res
    assert 'plotly>=6.1.1' in res['message'] and upgrade in res['message'], res
    assert 'automatic installation is off' in res['message'], res
    assert 'requires the kaleido package' not in res['message'], res
    assert res['plotly_after'] == '5.24.1'

    # installation on, plotly 5.24.1 already imported by the caller: nothing
    # is swapped under the running interpreter
    res, out = run('preimported')
    assert res['status'] == 'error' and res['exception'] == 'ImportError', res
    assert 'plotly 5.24.1 is installed' in res['message'], res
    assert 'already imported' in res['message'] and upgrade in res['message'], res
    assert 'restart' in res['message'].lower(), res
    assert 'requires the kaleido package' not in res['message'], res
    assert 'hypertools: upgrading' not in out.stdout
    assert res['plotly_after'] == '5.24.1'

    # the low-level render check names the pair, not "kaleido is missing"
    low = subprocess.run([str(py), '-c', _RENDER_PROBE], capture_output=True,
                         text=True, timeout=600, cwd=tmp_path, env=env)
    assert 'RENDER-ERROR HypertoolsIOError' in low.stdout, (low.stdout, low.stderr[-1500:])
    assert 'plotly 5.24.1' in low.stdout and 'kaleido 1.3.0' in low.stdout, low.stdout
    assert upgrade in low.stdout, low.stdout

    # installation on, fresh interpreter: upgraded, then the export works
    with skip_on_transient_network('upgrading plotly in the throwaway venv'):
        res, out = run('on')
        if res['status'] == 'error' and 'automatically failed' in res['message'] \
                and _pip_failed_on_network(
                    subprocess.CompletedProcess([], 1, out.stdout, out.stderr)):
            pytest.skip(f'transient network error: {out.stderr[-300:]}')
    assert ('hypertools: upgrading plotly 5.24.1 to plotly>=6.1.1 '
            '(needed for the plotly backend) ...') in out.stdout, out.stdout[-1500:]
    assert res['status'] == 'success', res
    assert res['size'] > 1000
    from packaging.version import Version
    assert Version(res['plotly_after']) >= Version('6.1.1'), res
