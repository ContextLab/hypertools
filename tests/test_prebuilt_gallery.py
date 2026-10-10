"""Tests for the pre-built gallery that Read the Docs reuses:
``scripts/publish_prebuilt_gallery.py`` (publish) and
``docs/fetch_prebuilt_gallery.py`` (fetch).

Read the Docs ends a build after 15 minutes and executing the gallery takes
longer, so its build fetches the gallery the release pipeline published and
sphinx-gallery skips every example whose source md5 is unchanged. Publishing
is a MANUAL release step with no CI job, so the round trip is exercised here
against a real local bare git remote.
"""

import importlib.util
import json
import pathlib
import subprocess
import warnings

import pytest

_REPO = pathlib.Path(__file__).resolve().parent.parent


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, _REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fpg = _load('fetch_prebuilt_gallery', 'docs/fetch_prebuilt_gallery.py')
ppg = _load('publish_prebuilt_gallery', 'scripts/publish_prebuilt_gallery.py')
glf = _load('_gallery_log_filter', 'docs/_gallery_log_filter.py')

pytestmark = pytest.mark.skipif(
    not (_REPO / 'examples').is_dir() or not (_REPO / 'pyproject.toml').is_file(),
    reason='requires a source checkout (examples/ and pyproject.toml)')


def _git(args, cwd):
    subprocess.run(['git'] + args, cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def git_identity(monkeypatch):
    # CI runners have no git identity configured; commits need one
    for var, value in (('GIT_AUTHOR_NAME', 'hypertools tests'),
                       ('GIT_AUTHOR_EMAIL', 'tests@example.invalid'),
                       ('GIT_COMMITTER_NAME', 'hypertools tests'),
                       ('GIT_COMMITTER_EMAIL', 'tests@example.invalid')):
        monkeypatch.setenv(var, value)


def _checkout(tmp_path, name='work', examples=None):
    """A small committed checkout shaped like this repository."""
    root = tmp_path / name
    (root / 'examples').mkdir(parents=True)
    (root / 'docs').mkdir()
    (root / 'pyproject.toml').write_text(
        '[project]\nname = "hypertools"\nversion = "9.8.7"\n',
        encoding='utf-8')
    (root / '.gitignore').write_text('docs/auto_examples/\n', encoding='utf-8')
    for stem, body in (examples or {'plot_a': 'print("a")\n',
                                    'plot_b': 'print("b")\n'}).items():
        (root / 'examples' / f'{stem}.py').write_text(body, encoding='utf-8')
    _git(['init', '--quiet', '--initial-branch', 'master'], root)
    _git(['add', '-A'], root)
    _git(['commit', '--quiet', '-m', 'checkout'], root)
    return root


def _build_gallery(root, stems=None):
    """What a finished sphinx-gallery run leaves for each example: the copied
    source, its md5, the page and an image."""
    gallery = root / 'docs' / 'auto_examples'
    (gallery / 'images').mkdir(parents=True, exist_ok=True)
    for src in sorted((root / 'examples').glob('*.py')):
        if stems is not None and src.stem not in stems:
            continue
        (gallery / src.name).write_text(src.read_text(encoding='utf-8'),
                                        encoding='utf-8')
        (gallery / (src.name + '.md5')).write_text(fpg.source_md5(src),
                                                   encoding='utf-8')
        (gallery / f'{src.stem}.rst').write_text(f'{src.stem}\n=====\n',
                                                 encoding='utf-8')
        (gallery / 'images' / f'sphx_glr_{src.stem}_001.png').write_bytes(
            b'\x89PNG\r\n\x1a\n' + src.stem.encode())
    return gallery


def _bare_remote(tmp_path):
    remote = tmp_path / 'remote.git'
    _git(['init', '--quiet', '--bare', str(remote)], tmp_path)
    return remote


def test_branch_name_is_per_version_and_rejects_unsafe_versions():
    assert fpg.branch_for('1.1.0') == 'docs-gallery-v1.1.0'
    assert fpg.branch_for('1.2.0rc1') == 'docs-gallery-v1.2.0rc1'
    for bad in ('', None, '../x', '1..2', 'v1.0', '1.0 --upload-pack=x'):
        with pytest.raises(ValueError):
            fpg.branch_for(bad)


def test_project_version_reads_this_checkouts_pyproject():
    import tomllib
    with open(_REPO / 'pyproject.toml', 'rb') as f:
        want = tomllib.load(f)['project']['version']
    assert fpg.project_version(_REPO) == want


def test_source_md5_is_the_md5_sphinx_gallery_compares():
    # the whole scheme rests on this hash equalling the one sphinx-gallery
    # computes when it decides whether to execute an example
    utils = pytest.importorskip('sphinx_gallery.utils')
    examples = sorted((_REPO / 'examples').glob('*.py'))
    assert len(examples) > 40
    for src in examples:
        assert fpg.source_md5(src) == utils.get_md5sum(src, mode='t'), src.name


def test_source_md5_ignores_line_ending_differences(tmp_path):
    # the gallery is built on macOS and reused on Linux; a Windows checkout
    # must hash the same too
    unix, dos = tmp_path / 'u.py', tmp_path / 'd.py'
    unix.write_bytes(b'x = 1\ny = 2\n')
    dos.write_bytes(b'x = 1\r\ny = 2\r\n')
    assert fpg.source_md5(unix) == fpg.source_md5(dos)


def test_stale_examples_are_the_ones_sphinx_would_execute(tmp_path,
                                                          git_identity):
    root = _checkout(tmp_path)
    gallery = _build_gallery(root)
    assert fpg.stale_examples(root / 'examples', gallery) == []
    # an edited example, and one the gallery never built
    (root / 'examples' / 'plot_a.py').write_text('print("changed")\n',
                                                 encoding='utf-8')
    (root / 'examples' / 'plot_new.py').write_text('print("new")\n',
                                                   encoding='utf-8')
    assert fpg.stale_examples(root / 'examples', gallery) == [
        'plot_a.py', 'plot_new.py']


def test_publish_then_fetch_round_trip(tmp_path, git_identity, capsys):
    root = _checkout(tmp_path)
    gallery = _build_gallery(root)
    remote = _bare_remote(tmp_path)

    assert ppg.publish(repo_root=str(root), push=True,
                       remote=str(remote)) == 0
    head = ppg.head_commit(str(root))
    shown = subprocess.run(
        ['git', 'show', 'docs-gallery-v9.8.7:manifest.json'], cwd=remote,
        check=True, capture_output=True, text=True).stdout
    manifest = json.loads(shown)
    assert manifest['source_commit'] == head
    assert manifest['version'] == '9.8.7'
    assert manifest['examples'] == fpg.example_md5s(root / 'examples')

    # a second checkout of the same commit, as Read the Docs would have
    fresh = tmp_path / 'fresh'
    _git(['clone', '--quiet', str(root), str(fresh)], tmp_path)
    assert not (fresh / 'docs' / 'auto_examples').exists()
    status, stale = fpg.fetch(str(fresh), remote=str(remote))
    assert (status, stale) == ('fetched', [])
    for path in sorted(p.relative_to(gallery) for p in gallery.rglob('*')
                       if p.is_file()):
        assert ((fresh / 'docs' / 'auto_examples' / path).read_bytes()
                == (gallery / path).read_bytes()), path
    assert not (fresh / 'docs' / 'auto_examples' / 'manifest.json').exists()
    assert '2 of 2 examples are current' in capsys.readouterr().out


def test_republishing_replaces_the_branch_instead_of_growing_it(
        tmp_path, git_identity):
    root = _checkout(tmp_path)
    _build_gallery(root)
    remote = _bare_remote(tmp_path)
    for _ in range(2):
        assert ppg.publish(repo_root=str(root), push=True,
                           remote=str(remote)) == 0
    count = subprocess.run(
        ['git', 'rev-list', '--count', 'docs-gallery-v9.8.7'], cwd=remote,
        check=True, capture_output=True, text=True).stdout.strip()
    assert count == '1'


def test_fetch_reports_examples_changed_since_publishing(tmp_path,
                                                         git_identity):
    root = _checkout(tmp_path)
    _build_gallery(root)
    remote = _bare_remote(tmp_path)
    assert ppg.publish(repo_root=str(root), push=True,
                       remote=str(remote)) == 0
    fresh = tmp_path / 'fresh'
    _git(['clone', '--quiet', str(root), str(fresh)], tmp_path)
    (fresh / 'examples' / 'plot_b.py').write_text('print("later")\n',
                                                  encoding='utf-8')
    assert fpg.fetch(str(fresh), remote=str(remote)) == ('fetched',
                                                         ['plot_b.py'])
    assert fpg.main(['--repo-root', str(fresh), '--remote', str(remote),
                     '--require']) == 1          # present, but stale


def test_fetch_without_a_published_gallery_lets_the_build_go_on(
        tmp_path, git_identity, capsys):
    root = _checkout(tmp_path)
    remote = _bare_remote(tmp_path)
    assert fpg.fetch(str(root), remote=str(remote)) == ('missing', None)
    assert not (root / 'docs' / 'auto_examples').exists()
    assert 'sphinx will execute every example' in capsys.readouterr().out
    assert fpg.main(['--repo-root', str(root), '--remote', str(remote)]) == 0
    assert fpg.main(['--repo-root', str(root), '--remote', str(remote),
                     '--require']) == 1


def test_fetch_leaves_an_existing_gallery_alone(tmp_path, git_identity):
    root = _checkout(tmp_path)
    gallery = _build_gallery(root)
    marker = gallery / 'plot_a.rst'
    marker.write_text('local build\n', encoding='utf-8')
    remote = _bare_remote(tmp_path)
    assert fpg.fetch(str(root), remote=str(remote)) == ('present', [])
    assert marker.read_text(encoding='utf-8') == 'local build\n'


def test_publish_refuses_an_incomplete_gallery(tmp_path, git_identity,
                                               capsys):
    root = _checkout(tmp_path)
    _build_gallery(root, stems={'plot_a'})
    remote = _bare_remote(tmp_path)
    assert ppg.publish(repo_root=str(root), push=True,
                       remote=str(remote)) == 1
    assert 'plot_b.py' in capsys.readouterr().err
    refs = subprocess.run(['git', 'for-each-ref'], cwd=remote, check=True,
                          capture_output=True, text=True).stdout
    assert refs == ''


def test_publish_refuses_uncommitted_changes(tmp_path, git_identity, capsys):
    root = _checkout(tmp_path)
    _build_gallery(root)
    (root / 'pyproject.toml').write_text(
        '[project]\nname = "hypertools"\nversion = "9.8.7"\n# edited\n',
        encoding='utf-8')
    assert ppg.publish(repo_root=str(root), push=False) == 1
    assert 'uncommitted changes' in capsys.readouterr().err


def test_publish_refuses_pages_naming_the_build_machines_path(
        tmp_path, git_identity, capsys):
    root = _checkout(tmp_path)
    gallery = _build_gallery(root)
    (gallery / 'plot_a.rst').write_text(
        f'    {root}/examples/plot_a.py:3: UserWarning: careful\n',
        encoding='utf-8')
    assert ppg.publish(repo_root=str(root), push=False) == 1
    assert 'plot_a.rst' in capsys.readouterr().err


def test_publish_without_push_leaves_the_remote_untouched(tmp_path,
                                                          git_identity,
                                                          capsys):
    root = _checkout(tmp_path)
    _build_gallery(root)
    remote = _bare_remote(tmp_path)
    assert ppg.publish(repo_root=str(root), push=False,
                       remote=str(remote)) == 0
    assert 'no push performed' in capsys.readouterr().out
    refs = subprocess.run(['git', 'for-each-ref'], cwd=remote, check=True,
                          capture_output=True, text=True).stdout
    assert refs == ''


def test_warnings_name_files_by_their_repository_path(tmp_path):
    # a warning an example raises is printed on its gallery page; it must not
    # carry the build machine's path to the checkout
    script = tmp_path / 'examples' / 'plot_w.py'
    script.parent.mkdir()
    script.write_text('import warnings\nwarnings.warn("careful")\n',
                      encoding='utf-8')
    original = warnings.formatwarning
    try:
        fmt = glf.relative_warning_paths(str(tmp_path))
        assert glf.relative_warning_paths(str(tmp_path)) is fmt   # idempotent
        text = warnings.formatwarning('careful', UserWarning, str(script), 2)
        assert text.startswith('examples/plot_w.py:2: UserWarning: careful')
        assert str(tmp_path) not in text
        # a file outside the checkout keeps its path
        other = warnings.formatwarning('careful', UserWarning,
                                       '/somewhere/else.py', 7)
        assert other.startswith('/somewhere/else.py:7: UserWarning: careful')
    finally:
        warnings.formatwarning = original


def test_a_raised_warning_is_printed_with_the_repository_path(tmp_path):
    # end to end through Python's own warning machinery, in a real process
    import sys
    script = tmp_path / 'examples' / 'plot_w.py'
    script.parent.mkdir()
    script.write_text('import warnings\nwarnings.warn("careful")\n',
                      encoding='utf-8')
    driver = (
        'import importlib.util, runpy, sys\n'
        f'spec = importlib.util.spec_from_file_location("g", {str(_REPO / "docs" / "_gallery_log_filter.py")!r})\n'
        'g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)\n'
        f'g.relative_warning_paths({str(tmp_path)!r})\n'
        f'runpy.run_path({str(script)!r})\n')
    done = subprocess.run([sys.executable, '-c', driver], capture_output=True,
                          text=True, check=True)
    assert 'examples/plot_w.py:2: UserWarning: careful' in done.stderr
    assert str(tmp_path) not in done.stderr


def test_readthedocs_fetches_the_gallery_before_building():
    text = (_REPO / '.readthedocs.yaml').read_text(encoding='utf-8')
    pre = text.index('pre_build:')
    post = text.index('post_build:')
    assert pre < text.index('python docs/fetch_prebuilt_gallery.py') < post


def test_a_gallery_built_here_names_no_local_path():
    gallery = _REPO / 'docs' / 'auto_examples'
    if not gallery.is_dir() or not any(gallery.glob('*.rst')):
        pytest.skip('no built gallery in docs/auto_examples')
    assert ppg.files_naming(str(gallery), str(_REPO)) == []


def _push_tree_as_gallery(tmp_path, remote, build):
    """Publish a hand-made tree to the gallery branch, bypassing the
    publisher's checks -- what someone pushing to the branch directly could
    do. ``build(gallery_dir)`` fills ``auto_examples/``."""
    work = tmp_path / 'handmade'
    (work / 'auto_examples').mkdir(parents=True)
    build(work / 'auto_examples')
    (work / 'manifest.json').write_text('{"source_commit": "0"}\n',
                                        encoding='utf-8')
    _git(['init', '--quiet', '--initial-branch', 'docs-gallery-v9.8.7'], work)
    _git(['add', '-A'], work)
    _git(['commit', '--quiet', '-m', 'handmade'], work)
    _git(['push', '--quiet', '--force', str(remote),
          'HEAD:refs/heads/docs-gallery-v9.8.7'], work)


def test_fetch_rejects_a_gallery_holding_a_symlink(tmp_path, git_identity,
                                                   capsys):
    # copying or reading a symlink would pull a file from the build machine
    # into the published site
    secret = tmp_path / 'secret.txt'
    secret.write_text('build machine secret\n', encoding='utf-8')
    root = _checkout(tmp_path)
    remote = _bare_remote(tmp_path)

    def build(gallery):
        (gallery / 'plot_a.rst').write_text('plot_a\n======\n',
                                            encoding='utf-8')
        try:
            (gallery / 'leak.rst').symlink_to(secret)
        except (OSError, NotImplementedError):
            pytest.skip('this platform cannot create a symlink here')

    _push_tree_as_gallery(tmp_path, remote, build)
    assert fpg.fetch(str(root), remote=str(remote)) == ('rejected', None)
    assert not (root / 'docs' / 'auto_examples').exists()
    out = capsys.readouterr().out
    assert 'REJECTED' in out and 'leak.rst' in out
    assert fpg.main(['--repo-root', str(root), '--remote', str(remote),
                     '--require']) == 1


@pytest.mark.parametrize('page', [
    '.. include:: /etc/passwd\n',
    '.. literalinclude:: ../../secrets.env\n',
    '.. raw:: html\n    :file: /home/docs/.netrc\n',
    '.. image:: ../../../private.png\n',
    ':download:`notes <../../.git/config>`\n',
    '.. csv-table::\n   :file: C:\\Users\\docs\\keys.csv\n',
])
def test_fetch_rejects_a_page_that_reads_outside_the_gallery(
        tmp_path, git_identity, page):
    root = _checkout(tmp_path)
    remote = _bare_remote(tmp_path)
    _push_tree_as_gallery(
        tmp_path, remote,
        lambda gallery: (gallery / 'plot_a.rst').write_text(
            'plot_a\n======\n\n' + page, encoding='utf-8'))
    assert fpg.fetch(str(root), remote=str(remote)) == ('rejected', None)
    assert not (root / 'docs' / 'auto_examples').exists()


def test_the_constructs_a_real_gallery_page_uses_are_accepted(tmp_path):
    gallery = tmp_path / 'auto_examples'
    (gallery / 'images').mkdir(parents=True)
    (gallery / 'plot_a.rst').write_text(
        '.. image-sg:: /auto_examples/images/sphx_glr_plot_a_001.png\n'
        '   :srcset: /auto_examples/images/sphx_glr_plot_a_001.png, '
        '/auto_examples/images/sphx_glr_plot_a_001_2_00x.png 2.00x\n\n'
        '.. video:: /auto_examples/images/sphx_glr_plot_a_001.mp4\n\n'
        '.. raw:: html\n    :file: images/sphx_glr_plot_a_002.html\n\n'
        '.. raw:: html\n\n    <div></div>\n\n'
        ':download:`Download zipped: plot_a.zip <plot_a.zip>`\n\n'
        '.. image:: https://example.org/badge.svg\n',
        encoding='utf-8')
    assert fpg.unsafe_entries(str(gallery)) == []


def test_publish_refuses_a_gallery_the_fetch_step_would_reject(
        tmp_path, git_identity, capsys):
    root = _checkout(tmp_path)
    gallery = _build_gallery(root)
    (gallery / 'plot_a.rst').write_text(
        'plot_a\n======\n\n.. include:: /etc/hosts\n', encoding='utf-8')
    assert ppg.publish(repo_root=str(root), push=False) == 1
    assert 'outside the gallery' in capsys.readouterr().err


def test_a_gallery_built_here_passes_the_fetch_safety_check():
    gallery = _REPO / 'docs' / 'auto_examples'
    if not gallery.is_dir() or not any(gallery.glob('*.rst')):
        pytest.skip('no built gallery in docs/auto_examples')
    assert fpg.unsafe_entries(str(gallery)) == []
