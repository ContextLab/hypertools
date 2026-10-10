"""The animated showcase notebooks must display their animation.

Six gallery examples build the animation inside ``if __name__ ==
'__main__':`` and end with ``fig = anim.figure``. The docs page shows the
animation because the docs build captures it, but the generated notebook
displayed nothing (reported from Colab right after 1.1.0). The docs build now
appends a final ``anim`` cell (docs/_gallery_notebooks.py).
"""
import importlib.util
import json
import pathlib

import pytest

_REPO = pathlib.Path(__file__).resolve().parent.parent
_SHOWCASES = ('animate_morph_zoo', 'animate_conversation',
              'animate_painting_embeddings', 'animate_forecast',
              'animate_market_sectors', 'animate_weather_decades')

pytestmark = pytest.mark.skipif(
    not (_REPO / 'docs' / '_gallery_notebooks.py').is_file(),
    reason='requires a source checkout (docs/)')


def _load():
    spec = importlib.util.spec_from_file_location(
        '_gallery_notebooks', _REPO / 'docs' / '_gallery_notebooks.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gnb = _load() if (_REPO / 'docs' / '_gallery_notebooks.py').is_file() else None


def _notebook(*code, indent=2):
    cells = [{'cell_type': 'markdown', 'metadata': {}, 'source': ['# Title']}]
    cells += [{'cell_type': 'code', 'execution_count': None,
               'metadata': {'collapsed': False}, 'outputs': [],
               'source': src.splitlines(keepends=True)} for src in code]
    return json.dumps({'cells': cells, 'metadata': {}, 'nbformat': 4,
                       'nbformat_minor': 0}, indent=indent)


_UNSHOWN = ("import hypertools as hyp\n\nif __name__ == '__main__':\n"
            "    anim = construct_artifact(shapes)\n    fig = anim.figure\n")


def test_a_notebook_ending_with_the_unshown_animation_gets_a_playback_cell(
        tmp_path):
    path = tmp_path / 'animate_x.ipynb'
    path.write_text(_notebook('%pip install -q "hypertools[interactive]"',
                              _UNSHOWN), encoding='utf-8')
    assert gnb.add_inline_playback(str(path)) is True
    nb = json.loads(path.read_text(encoding='utf-8'))
    last = nb['cells'][-1]
    assert last['cell_type'] == 'code' and last['outputs'] == []
    lines = [ln for ln in ''.join(last['source']).splitlines()
             if ln.strip() and not ln.lstrip().startswith('#')]
    assert lines == ['anim']        # a bare expression: the cell's value
    # the earlier cells are untouched
    assert ''.join(nb['cells'][-2]['source']) == _UNSHOWN


def test_adding_the_cell_twice_changes_nothing(tmp_path):
    path = tmp_path / 'animate_x.ipynb'
    path.write_text(_notebook(_UNSHOWN), encoding='utf-8')
    assert gnb.add_inline_playback(str(path)) is True
    once = path.read_bytes()
    assert gnb.add_inline_playback(str(path)) is False
    assert path.read_bytes() == once


@pytest.mark.parametrize('source', [
    "import hypertools as hyp\nhyp.plot(data, animate=True)\n",   # shows itself
    "fig = anim.figure\n",                    # top level: not the pattern
    "    fig = anim.figure\nprint('saved')\n",   # something follows it
])
def test_other_notebooks_are_left_alone(tmp_path, source):
    path = tmp_path / 'plot_x.ipynb'
    path.write_text(_notebook(source), encoding='utf-8')
    before = path.read_bytes()
    assert gnb.add_inline_playback(str(path)) is False
    assert path.read_bytes() == before


@pytest.mark.parametrize('indent', [1, 2])
def test_the_files_json_layout_is_kept(tmp_path, indent):
    path = tmp_path / 'animate_x.ipynb'
    path.write_text(_notebook(_UNSHOWN, indent=indent), encoding='utf-8')
    gnb.add_inline_playback(str(path))
    text = path.read_text(encoding='utf-8')
    assert text == json.dumps(json.loads(text), indent=indent)


@pytest.mark.parametrize('stem', _SHOWCASES)
def test_the_showcase_examples_still_end_the_way_the_cell_is_keyed_on(stem):
    # if an example stops ending with `fig = anim.figure`, its notebook
    # silently loses the playback cell: fail here instead
    source = (_REPO / 'examples' / f'{stem}.py').read_text(encoding='utf-8')
    nb = {'cells': [{'cell_type': 'code', 'source': source}]}
    assert gnb.needs_inline_playback(nb), stem


def test_the_playback_cell_really_displays_a_video():
    # the cell relies on HyperAnimation._repr_html_; exercise it for real
    import numpy as np
    import hypertools as hyp
    rng = np.random.default_rng(0)
    anim = hyp.plot([rng.standard_normal((30, 3)) for _ in range(2)],
                    animate='morph', duration=1, frame_rate=5, show=False,
                    backend='matplotlib')
    html = anim._repr_html_()
    assert html and ('<video' in html or 'animation' in html.lower())


def test_a_gallery_built_here_shows_every_showcase_animation():
    gallery = _REPO / 'docs' / 'auto_examples'
    built = [gallery / f'{stem}.ipynb' for stem in _SHOWCASES]
    if not all(p.is_file() for p in built):
        pytest.skip('no built gallery in docs/auto_examples')
    for path in built:
        nb = json.loads(path.read_text(encoding='utf-8'))
        if gnb.needs_inline_playback(nb):
            pytest.skip('gallery was built before the playback cell existed; '
                        'rebuild the docs')
        code = [c for c in nb['cells'] if c['cell_type'] == 'code']
        assert ''.join(code[-1]['source']).rstrip().endswith('anim'), path.name
