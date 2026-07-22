"""Unit tests for scripts/publish_gallery_notebooks.py (the pure layout + ref
validation; the git push itself is exercised by the publish-gallery-notebooks
CI job, not here)."""

import importlib.util
import pathlib

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
