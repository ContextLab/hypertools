"""Guard the Colab install cells in the committed tutorial notebooks.

Two layers (2026-07 release review, notebook-install finding):

* ALWAYS ON -- the tracked notebooks must not drift back to the defunct
  ``dev-1.0-refactor`` branch (whose head is no longer an ancestor of the
  release line, so it installs code missing recent fixes), and every hypertools
  GitHub-branch install must point at ONE consistent branch. This would have
  caught the stale-branch bug and prevents it regressing on any branch.

* RELEASE GATE (``HYPERTOOLS_REQUIRE_RELEASE=1``; the dedicated
  ``release-gate`` CI job sets it on master/tag builds) -- every
  hypertools install must be the plain PyPI spec, i.e. NO ``git+`` / ``@<branch>``
  preview install may survive into a release. This cannot pass by skipping.
"""

import glob
import json
import os
import re
import subprocess

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TUT_DIR = os.path.join(_REPO, 'docs', 'tutorials')

# docs/ is not shipped in the wheel/sdist (MANIFEST grafts only tests/), so when
# the suite runs from an installed package these notebook checks have nothing to
# inspect -- skip the whole module rather than fail.
pytestmark = pytest.mark.skipif(
    not os.path.isdir(_TUT_DIR),
    reason='requires a source checkout (docs/tutorials/ absent in wheel/sdist)')

# a line that actually EXECUTES a package install (shell/magic prefix or a bare
# tool token at line start), so it matches %pip/!pip/!pip3/pipx/pip<TAB>install/
# uv pip/conda but NOT a comment or a documentation string like
# `print("pip install git+...")`. Keyed on the literal `pip install` before, it
# both missed `!pip3 install ...@branch` and false-flagged such print strings
# (release review). Kept identical to scripts/check_release_notebooks.py.
_INSTALL_LINE_RE = re.compile(
    r'^[%!]?\s*(?:pip[0-9]*|pipx|uv\s+pip|conda|mamba|python[0-9.]*\s+-m\s+pip)'
    r'\s+install\b', re.IGNORECASE)
# a hypertools GitHub-branch install spec, capturing the branch
_HYP_BRANCH_RE = re.compile(
    r'hypertools\[[^\]]*\]\s*@\s*'
    r'git\+https://github\.com/ContextLab/hypertools\.git@([\w./\-]+)')

REQUIRE_RELEASE = os.environ.get('HYPERTOOLS_REQUIRE_RELEASE') == '1'


def _git_ls(pattern):
    try:
        out = subprocess.run(['git', 'ls-files', pattern],
                             capture_output=True, text=True, cwd=_REPO,
                             timeout=30).stdout.split()
        return [os.path.join(_REPO, p) for p in out]
    except Exception:
        return []


def _tracked_tutorials():
    """The hand-authored tutorial notebooks (always git-tracked)."""
    got = _git_ls('docs/tutorials/*.ipynb')
    return got or sorted(glob.glob(os.path.join(_TUT_DIR, '*.ipynb')))


def _tracked_published_notebooks():
    """Every git-tracked published notebook. The hand-authored tutorials are tracked:
    docs/auto_examples/*.ipynb are GITIGNORED and regenerated at build time
    from docs/conf.py's branch-aware install cell, so they are not shipped and
    do not exist in a bare checkout (the release-gate CI job runs on a bare
    checkout). The GENERATED gallery is release-gated separately in the
    docs-clean CI job. Scanning the union keeps this gate correct if a gallery
    notebook is ever committed. (2026-07 release review, finding #2.)"""
    got = _git_ls('docs/tutorials/*.ipynb') + _git_ls('docs/auto_examples/*.ipynb')
    if got:
        return got
    # no .git -> source-archive fallback (both dirs, if present)
    out = []
    for d in (_TUT_DIR, os.path.join(_REPO, 'docs', 'auto_examples')):
        out += sorted(glob.glob(os.path.join(d, '*.ipynb')))
    return out


def _hyp_install_lines(path):
    """Every code-cell line that pip-installs hypertools, across all cells."""
    with open(path, encoding='utf-8') as f:
        nb = json.load(f)
    lines = []
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        for line in ''.join(cell.get('source', [])).splitlines():
            if 'hypertools' in line and _INSTALL_LINE_RE.match(line.lstrip()):
                lines.append(line)
    return lines


def test_there_are_tracked_published_notebooks():
    # guards against the scan silently passing because it found nothing
    expected = set('align analyze animate_forecast cluster conversation_shape conversation_trajectories hierarchy hugging_face_embeddings io lsl_streaming manip market_sectors modern_sklearn_dynamics morph_shapes_zoo normalize painting_embeddings pipelines plot projectile_kalman reduce stock_forecasting streaming_data text weather_decades wikipedia_embeddings'.split())
    assert {os.path.splitext(os.path.basename(p))[0] for p in _tracked_tutorials()} == expected
    # the published-notebook union is at least the tutorials
    assert len(_tracked_published_notebooks()) >= len(_tracked_tutorials())


def test_no_notebook_installs_the_defunct_refactor_branch():
    offenders = []
    branches = set()
    for path in _tracked_published_notebooks():
        for line in _hyp_install_lines(path):
            for br in _HYP_BRANCH_RE.findall(line):
                branches.add(br)
                if br == 'dev-1.0-refactor':
                    offenders.append((os.path.basename(path), line.strip()))
    assert not offenders, (
        'tutorial notebooks still install the defunct dev-1.0-refactor '
        f'branch (run scripts/add_colab_install_cell.py): {offenders}')
    # and every branch install points at ONE branch (no stale-vs-current mix)
    assert len(branches) <= 1, (
        f'tutorial notebooks install from mixed branches {sorted(branches)}; '
        'run scripts/add_colab_install_cell.py to unify them')


@pytest.mark.skipif(
    not REQUIRE_RELEASE,
    reason='release gate; set HYPERTOOLS_REQUIRE_RELEASE=1 (the '
           'release-gate CI job does on master/tag builds)')
def test_release_gate_no_branch_installs_in_published_notebooks():
    # finding: a release must not ship notebooks that install a GitHub branch
    # (they would install code that omits release fixes, or 404 once the branch
    # is deleted post-merge). Every hypertools install must be the PyPI spec.
    offenders = []
    for path in _tracked_published_notebooks():
        for line in _hyp_install_lines(path):
            if 'git+' in line or _HYP_BRANCH_RE.search(line):
                offenders.append((os.path.basename(path), line.strip()))
    assert not offenders, (
        'RELEASE GATE: published notebooks still contain preview/branch '
        'hypertools installs; run `python scripts/add_colab_install_cell.py` '
        f'on master before creating the release commit: {offenders}')


@pytest.mark.skipif(
    not REQUIRE_RELEASE,
    reason='release gate; set HYPERTOOLS_REQUIRE_RELEASE=1 (the '
           'release-gate CI job does on master/tag builds)')
def test_release_gate_no_preview_note_in_published_notebooks():
    # release review, GAP #1: the migration flips the install LINE but must also
    # strip the standard "(<x> preview) / On release this becomes ..." note, or
    # the released notebooks ship saying "preview". Scans install cells only.
    offenders = []
    for path in _tracked_published_notebooks():
        with open(path, encoding='utf-8') as f:
            nb = json.load(f)
        for cell in nb.get('cells', []):
            if cell.get('cell_type') != 'code':
                continue
            src = ''.join(cell.get('source', []))
            if 'pip install' not in src:
                continue
            for marker in ('On release this becomes', ' preview)'):
                if marker in src:
                    offenders.append((os.path.basename(path), marker))
    assert not offenders, (
        'RELEASE GATE: published notebooks still carry a preview install note; '
        'run `python scripts/add_colab_install_cell.py` on master: '
        f'{offenders}')


# --- an install cell never ships output --------------------------------------

def _install_cells(path):
    with open(path, encoding='utf-8') as f:
        nb = json.load(f)
    return [c for c in nb.get('cells', []) if c.get('cell_type') == 'code'
            and any(_INSTALL_LINE_RE.match(ln.lstrip())
                    for ln in ''.join(c.get('source', [])).splitlines())]


def test_no_published_install_cell_carries_output():
    """scripts/execute_tutorial.py skips the Colab install cell, and a skipped
    cell keeps whatever the file had: projectile_kalman and streaming_data
    shipped a pip upgrade notice naming a local interpreter path from the
    1.0.0 run that executed it (found 2026-09-07). The cell did not run in
    the published execution, so it has nothing to show."""
    offenders = []
    for path in _tracked_published_notebooks():
        for cell in _install_cells(path):
            if cell.get('outputs') or cell.get('execution_count') is not None:
                offenders.append(os.path.relpath(path, _REPO))
    assert not offenders, offenders


def test_execute_tutorial_drops_the_outputs_of_the_cell_it_skips(tmp_path):
    """`skip_install_cells` (the in-memory step `execute()` runs before
    nbclient) tags the install cell and clears its stored output;
    `restore_install_cells` removes only the tag it added."""
    import importlib.util
    import nbformat
    spec = importlib.util.spec_from_file_location(
        'execute_tutorial', os.path.join(_REPO, 'scripts', 'execute_tutorial.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    nb = nbformat.v4.new_notebook()
    install = nbformat.v4.new_code_cell(
        '%pip install -q "hypertools[interactive]"', execution_count=1,
        outputs=[nbformat.v4.new_output('stream', name='stdout',
                                        text='[notice] A new release of pip')])
    install.metadata['tags'] = ['keep-me']
    work = nbformat.v4.new_code_cell('import hypertools', execution_count=2,
                                     outputs=[nbformat.v4.new_output(
                                         'stream', name='stdout', text='hi')])
    nb.cells = [install, work]
    skipped = mod.skip_install_cells(nb)
    assert skipped == [install]
    assert install.metadata['tags'] == ['keep-me', mod.SKIP_TAG]
    assert install.outputs == [] and install.execution_count is None
    assert work.outputs and work.execution_count == 2       # untouched
    mod.restore_install_cells(skipped)
    assert install.metadata['tags'] == ['keep-me']
    path = tmp_path / 'nb.ipynb'
    nbformat.write(nb, path)
    assert _install_cells(path)[0]['outputs'] == []


def test_tutorial_installers_enforce_version_and_preserve_prerequisites():
    for path in _tracked_tutorials():
        with open(path, encoding='utf-8') as handle:
            nb=json.load(handle)
        installers=[c for c in nb['cells'] if 'hypertools-install' in c.get('metadata',{}).get('tags',[])]
        assert len(installers)==1, path
        source=''.join(installers[0]['source'])
        assert "Version('1.1.0')" in source and '>=1.1.0' in source, path
        assert 'will not replace your checkout' in source, path
        assert 'pip install -q convokit' not in source and 'pip install -q py7zr' not in source


def test_executor_keeps_setup_and_independent_install_cells():
    import importlib.util
    import nbformat
    spec=importlib.util.spec_from_file_location('execute_tutorial',os.path.join(_REPO,'scripts','execute_tutorial.py'))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config=nbformat.v4.new_code_cell("SETTINGS = {}\n# Optional: pip install extras")
    prerequisite=nbformat.v4.new_code_cell('%pip install convokit')
    install=nbformat.v4.new_code_cell("subprocess.check_call([sys.executable,'-m','pip','install',spec])",metadata={'tags':['hypertools-install']})
    nb=nbformat.v4.new_notebook(cells=[config,prerequisite,install])
    skipped=module.skip_install_cells(nb)
    assert skipped==[install]
    assert 'skip-execution' not in config.metadata.get('tags',[])
    assert 'skip-execution' not in prerequisite.metadata.get('tags',[])


def _load_executor():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'execute_tutorial', os.path.join(_REPO, 'scripts', 'execute_tutorial.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_executor_scrubs_the_kernel_cell_path_from_warnings():
    """A warning raised by a cell names the kernel's per-session temp file
    (plot.ipynb stored `/var/folders/<id>/T/ipykernel_21956/2889100357.py:14:
    UserWarning: ...`, found 2026-09-11); the executor rewrites it to
    `<cell>`, on every platform's spelling, and still rewrites the home dir."""
    import nbformat
    module = _load_executor()
    home = '/Users/someone'
    texts = [
        '/var/folders/tp/qtzc39jx5w556wl5w3dj21wr0000gn/T/ipykernel_21956/'
        '2889100357.py:14: UserWarning: Missing data\n',
        '/tmp/ipykernel_77/123.py:3: UserWarning: x\n',
        'C:\\Users\\someone\\AppData\\Local\\Temp\\ipykernel_5\\99.py:1: W\n',
        '/Users/someone/hypertools/hypertools/predict/common.py:416: UserWarning\n',
    ]
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        'x', outputs=[nbformat.v4.new_output('stream', name='stderr', text=t)
                      for t in texts])])
    assert module.scrub_home(nb, home=home) == len(texts)
    got = [o['text'] for o in nb.cells[0].outputs]
    assert got[:3] == ['<cell>:14: UserWarning: Missing data\n',
                       '<cell>:3: UserWarning: x\n', '<cell>:1: W\n']
    assert got[3] == '~/hypertools/hypertools/predict/common.py:416: UserWarning\n'


def test_executor_quiets_liblsl_info_logging(tmp_path):
    """liblsl logs `api_config.cpp ... INFO| Loaded default config` to stderr
    on first load, and lsl_streaming.ipynb stored two such lines (2026-09-10).
    Measured on the real liblsl in a subprocess: the line appears with no
    config (the control) and not under the config the executor installs.
    Only a StreamInfo is built -- nothing is advertised on the network."""
    import sys
    pytest.importorskip('pylsl')
    module = _load_executor()
    probe = ("import pylsl; pylsl.StreamInfo('hyp-cfg-probe', 'HYPCFGPROBE', 1, "
             "source_id='hyp-cfg-probe')")
    env = {k: v for k, v in os.environ.items() if k != 'LSLAPICFG'}
    control = subprocess.run([sys.executable, '-c', probe], env=env,
                             capture_output=True, text=True, timeout=120)
    assert control.returncode == 0, control.stderr
    assert 'INFO|' in control.stderr, 'control run: liblsl no longer logs INFO'
    path = module.quiet_liblsl_config(str(tmp_path), environ=env)
    assert env['LSLAPICFG'] == path and os.path.exists(path)
    quiet = subprocess.run([sys.executable, '-c', probe], env=env,
                           capture_output=True, text=True, timeout=120)
    assert quiet.returncode == 0, quiet.stderr
    assert 'INFO|' not in quiet.stderr, quiet.stderr


def test_executor_keeps_a_callers_liblsl_config(tmp_path):
    module = _load_executor()
    env = {'LSLAPICFG': '/somewhere/else.cfg'}
    assert module.quiet_liblsl_config(str(tmp_path), environ=env) == '/somewhere/else.cfg'
    assert env == {'LSLAPICFG': '/somewhere/else.cfg'}
    assert not os.listdir(tmp_path)
