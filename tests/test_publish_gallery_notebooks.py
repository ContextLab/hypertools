"""Tests for scripts/publish_gallery_notebooks.py.

Covers the pure layout + ref validation + manifest, AND a real end-to-end
``publish()`` round-trip against a local bare git remote (the git push is a
MANUAL release step -- there is no CI job for it -- so it is exercised here
rather than assumed)."""

import importlib.util
import json
import pathlib
import shutil
import subprocess

import pytest

_SCRIPT = (pathlib.Path(__file__).resolve().parent.parent
           / 'scripts' / 'publish_gallery_notebooks.py')


def _load():
    spec = importlib.util.spec_from_file_location('publish_gallery_notebooks',
                                                  _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pgn = _load()


def test_target_paths_lays_out_under_ref_auto_examples():
    nbs = ['/build/docs/auto_examples/plot_basic.ipynb',
           '/build/docs/auto_examples/animate_spin.ipynb']
    got = pgn.target_paths(nbs, 'v1.0.0')
    assert got == {
        '/build/docs/auto_examples/plot_basic.ipynb':
            'v1.0.0/auto_examples/plot_basic.ipynb',
        '/build/docs/auto_examples/animate_spin.ipynb':
            'v1.0.0/auto_examples/animate_spin.ipynb',
    }


def test_target_paths_accepts_branch_and_tag_refs():
    for ref in ('v1.0.0', 'dev-1.0', 'master'):
        out = pgn.target_paths(['/x/plot_basic.ipynb'], ref)
        assert out['/x/plot_basic.ipynb'] == f'{ref}/auto_examples/plot_basic.ipynb'


def test_target_paths_ignores_non_ipynb():
    assert pgn.target_paths(['/x/readme.txt', '/x/a.ipynb'], 'v1.0.0') == {
        '/x/a.ipynb': 'v1.0.0/auto_examples/a.ipynb'}


@pytest.mark.parametrize('bad', ['../evil', 'a/../b', '', '-x', '/abs',
                                 'a b', 'a;b'])
def test_target_paths_rejects_unsafe_refs(bad):
    with pytest.raises(ValueError):
        pgn.target_paths(['/x/a.ipynb'], bad)


# --------------------------------------------------------------- manifest

def test_manifest_content_records_full_sorted_inventory():
    nbs = ['/b/auto_examples/plot_basic.ipynb',
           '/b/auto_examples/animate_spin.ipynb',
           '/b/auto_examples/readme.txt']           # non-ipynb ignored
    m = pgn.manifest_content(nbs, 'v1.0.0', 'abc123')
    assert m == {'ref': 'v1.0.0', 'source_commit': 'abc123', 'count': 2,
                 'notebooks': ['animate_spin', 'plot_basic']}


def test_manifest_count_always_equals_listed_notebooks():
    # written from the actual files, so count can never over-claim the set
    for n in (0, 1, 5):
        nbs = [f'/x/nb_{i}.ipynb' for i in range(n)]
        m = pgn.manifest_content(nbs, 'v1.0.0')
        assert m['count'] == len(m['notebooks']) == n
        assert m['source_commit'] is None


# --------------------------------------------------------- real publish()

def _git(cwd, *args):
    subprocess.run(['git', *args], cwd=cwd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@pytest.mark.parametrize('empty_default', [False, True],
                         ids=['nonempty-default', 'empty-default'])
def test_publish_round_trip_to_bare_remote(tmp_path, monkeypatch, empty_default):
    """publish(--push) bootstraps the orphan docs-notebooks branch, lays the
    notebooks out under <ref>/auto_examples/, writes a matching manifest, and
    does not leak the default branch's files into the orphan. Exercises both
    the non-empty default branch (real-world) and the empty-default edge that
    the --ignore-unmatch bootstrap guards."""
    if shutil.which('git') is None:
        pytest.skip('git not available')
    for k in ('AUTHOR', 'COMMITTER'):
        monkeypatch.setenv(f'GIT_{k}_NAME', 'test')
        monkeypatch.setenv(f'GIT_{k}_EMAIL', 'test@example.com')

    bare = tmp_path / 'remote.git'
    bare.mkdir()
    _git(bare, 'init', '--bare', '-b', 'master', str(bare))
    if not empty_default:
        src = tmp_path / 'src'
        src.mkdir()
        _git(src, 'init', '-b', 'master', str(src))
        (src / 'readme.md').write_text('hi')
        _git(src, 'add', '-A')
        _git(src, 'commit', '-m', 'init')
        _git(src, 'remote', 'add', 'origin', str(bare))
        _git(src, 'push', 'origin', 'master')

    nbdir = tmp_path / 'nbs'
    nbdir.mkdir()
    stems = ['plot_align', 'plot_basic', 'plot_clusters']
    for stem in stems:
        (nbdir / f'{stem}.ipynb').write_text('{"cells": []}')

    rc = pgn.publish(str(nbdir), 'v1.0.0', push=True, remote=str(bare))
    assert rc == 0

    chk = tmp_path / 'chk'
    _git(tmp_path, 'clone', '--branch', 'docs-notebooks', str(bare), str(chk))
    for stem in stems:
        assert (chk / 'v1.0.0' / 'auto_examples' / f'{stem}.ipynb').is_file()
    # the default branch's files must not survive into the orphan branch
    assert not (chk / 'readme.md').exists()
    manifest = json.loads((chk / 'v1.0.0' / 'manifest.json').read_text())
    assert manifest['ref'] == 'v1.0.0'
    assert manifest['count'] == 3
    assert manifest['notebooks'] == stems


def test_publish_is_idempotent_and_replaces_stale_notebooks(tmp_path, monkeypatch):
    """A second publish of a changed set replaces the ref's notebooks wholesale
    (removed examples don't linger) and rewrites the manifest to match."""
    if shutil.which('git') is None:
        pytest.skip('git not available')
    for k in ('AUTHOR', 'COMMITTER'):
        monkeypatch.setenv(f'GIT_{k}_NAME', 'test')
        monkeypatch.setenv(f'GIT_{k}_EMAIL', 'test@example.com')
    bare = tmp_path / 'remote.git'
    bare.mkdir()
    _git(bare, 'init', '--bare', '-b', 'master', str(bare))

    nbdir = tmp_path / 'nbs'
    nbdir.mkdir()
    for stem in ('plot_basic', 'old_example'):
        (nbdir / f'{stem}.ipynb').write_text('{"cells": []}')
    assert pgn.publish(str(nbdir), 'v1.0.0', push=True, remote=str(bare)) == 0

    # second publish: drop old_example, add plot_new
    (nbdir / 'old_example.ipynb').unlink()
    (nbdir / 'plot_new.ipynb').write_text('{"cells": []}')
    assert pgn.publish(str(nbdir), 'v1.0.0', push=True, remote=str(bare)) == 0

    chk = tmp_path / 'chk'
    _git(tmp_path, 'clone', '--branch', 'docs-notebooks', str(bare), str(chk))
    ae = chk / 'v1.0.0' / 'auto_examples'
    assert (ae / 'plot_new.ipynb').is_file()
    assert (ae / 'plot_basic.ipynb').is_file()
    assert not (ae / 'old_example.ipynb').exists()   # stale example removed
    manifest = json.loads((chk / 'v1.0.0' / 'manifest.json').read_text())
    assert manifest['notebooks'] == ['plot_basic', 'plot_new']
    assert manifest['count'] == 2
