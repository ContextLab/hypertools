"""The rendered API reference shows no leaked reStructuredText markup.

Builds the API pages for real -- ``docs/api.rst`` through Sphinx with the
same autodoc + numpydoc + autosummary chain ``docs/conf.py`` uses (no
gallery, no notebooks, the built-in theme) -- and runs
``scripts/scan_api_markup_leaks.py`` over the HTML. The scanner reports
literal backticks, ``:role:`` syntax and ``**`` pairs in body text,
numpydoc items that are really wrapped prose, and signature defaults with
an opaque repr.

A second build of a small module whose docstrings contain each of those
mistakes proves the scanner still sees them.
"""
import importlib.util
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip('sphinx')
pytest.importorskip('numpydoc')

REPO = pathlib.Path(__file__).resolve().parents[1]
SCANNER_PATH = REPO / 'scripts' / 'scan_api_markup_leaks.py'

CONF = '''
import sys
sys.path.insert(0, {root!r})
project = 'hypertools'
extensions = ['sphinx.ext.autodoc', 'numpydoc', 'sphinx.ext.autosummary']
numpydoc_class_members_toctree = False
autosummary_generate = True
html_theme = 'basic'
'''


def _scanner():
    spec = importlib.util.spec_from_file_location('scan_api_markup_leaks',
                                                  SCANNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build(source_dir, out_dir):
    env = dict(os.environ, MPLBACKEND='Agg')
    done = subprocess.run(
        [sys.executable, '-m', 'sphinx', '-q', '-b', 'html',
         '-d', str(out_dir.parent / 'doctrees'),
         str(source_dir), str(out_dir)],
        capture_output=True, text=True, env=env, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    return done


def _format(results):
    return '\n'.join(f'{page} line {line}: [{kind}] {text}'
                     for page, hits in results.items()
                     for line, kind, text in hits)


def test_api_reference_has_no_leaked_markup(tmp_path):
    source = tmp_path / 'src'
    source.mkdir()
    (source / 'conf.py').write_text(CONF.format(root=str(REPO)),
                                    encoding='utf-8')
    api = (REPO / 'docs' / 'api.rst').read_text(encoding='utf-8')
    # api.rst is an :orphan: page in the real site; here it is the root
    (source / 'index.rst').write_text(api, encoding='utf-8')
    out = tmp_path / 'html'
    _build(source, out)

    pages = sorted(p.name for p in out.glob('hypertools.*.html'))
    # every autosummary entry got a page (plot, predict, the LSL classes...)
    for expected in ('hypertools.plot.html', 'hypertools.predict.html',
                     'hypertools.impute.html',
                     'hypertools.io.LSLStream.html',
                     'hypertools.text_windows.html'):
        assert expected in pages, pages
    assert len(pages) >= 40, pages

    scanner = _scanner()
    results = {page: scanner.scan_file(str(out / page))
               for page in pages + ['index.html']}
    leaks = {page: hits for page, hits in results.items() if hits}
    assert not leaks, _format(leaks)

    # the pages really carry the rendered docstrings the scan is about
    plot_page = (out / 'hypertools.plot.html').read_text(encoding='utf-8')
    assert 'slow_warning_seconds' in plot_page
    assert 'morph_samples' in plot_page


LEAKY_MODULE = '''
_SENTINEL = object()


def leaky(a, b=None, c=_SENTINEL):
    """Do a thing.

    Parameters
    ----------
    a : int
        About ~`a` seconds, in `unit`s.
    b (``mode='x'`` only) : int
        **Bold with ``literal`` inside.** See :func:`leaky`x and
        un**bold**ed text.

    Returns
    -------
    the result (and the fitted model if asked for it). Lists in,
    lists out: one dataset in gives one result back out again
    """


def clean(a, **kwargs):
    """Do a thing.

    Parameters
    ----------
    a : int
        About `a` seconds, i.e. ``grid**3`` cells.
    **kwargs
        Passed through to :func:`clean`.

    Returns
    -------
    result : DataFrame or list of DataFrame
        The result.
    """
'''


def test_scanner_flags_each_kind_of_leak(tmp_path):
    source = tmp_path / 'src'
    source.mkdir()
    (source / 'leakymod.py').write_text(LEAKY_MODULE, encoding='utf-8')
    (source / 'conf.py').write_text(CONF.format(root=str(source)),
                                    encoding='utf-8')
    (source / 'index.rst').write_text(textwrap.dedent('''\
        Leaks
        =====

        .. currentmodule:: leakymod

        .. autosummary::
          :toctree:

          leaky
          clean
        '''), encoding='utf-8')
    out = tmp_path / 'html'
    _build(source, out)

    scanner = _scanner()
    kinds = {kind for _, kind, _ in
             scanner.scan_file(str(out / 'leakymod.leaky.html'))}
    assert 'backtick' in kinds
    assert 'role' in kinds
    assert 'strong-markers' in kinds
    assert 'opaque-default' in kinds
    assert any(kind.startswith('prose-term') for kind in kinds), kinds

    assert scanner.scan_file(str(out / 'leakymod.clean.html')) == []
