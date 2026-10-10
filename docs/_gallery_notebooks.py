"""Post-processing for the notebooks sphinx-gallery generates.

Kept in its own SIDE-EFFECT-FREE module (standard library only) so
tests/test_gallery_notebook_playback.py can unit-test it.

Six animated showcase examples build their animation inside
``if __name__ == '__main__':`` and end with ``fig = anim.figure``, so that
importing the example (the test suite does) builds nothing. The docs page
still shows the animation, because the docs build captures it. A notebook
shows nothing: an assignment inside an ``if`` block is not a cell value.
``add_inline_playback`` appends a final cell holding ``anim``, which Jupyter
and Colab render as an inline video through ``HyperAnimation._repr_html_``.
"""
import copy
import json
import re

PLAYBACK_SOURCE = (
    '# Play the animation inline. The whole clip is rendered first, which\n'
    '# takes from a few seconds to a few minutes, depending on its length.\n'
    'anim')
_ENDS_WITH_UNSHOWN_ANIMATION = re.compile(r'(?m)^[ \t]+fig = anim\.figure[ \t]*$')


def needs_inline_playback(nb):
    """True when the notebook's last code cell ends by binding
    ``fig = anim.figure`` inside a block, i.e. it built an animation and
    displayed nothing."""
    code = [c for c in nb.get('cells', []) if c.get('cell_type') == 'code']
    if not code:
        return False
    source = code[-1]['source']
    source = ''.join(source) if isinstance(source, list) else source
    lines = [ln for ln in source.splitlines() if ln.strip()]
    return bool(lines) and bool(_ENDS_WITH_UNSHOWN_ANIMATION.match(lines[-1]))


def add_inline_playback(path):
    """Append the playback cell to the notebook at ``path`` if it needs one.
    Returns True when the file was changed. Idempotent: once the cell is
    there, the last code cell no longer ends with the assignment."""
    with open(path, encoding='utf-8') as f:
        raw = f.read()
    nb = json.loads(raw)
    if not needs_inline_playback(nb):
        return False
    template = [c for c in nb['cells'] if c['cell_type'] == 'code'][-1]
    cell = copy.deepcopy(template)
    cell['outputs'] = []
    cell['execution_count'] = None
    cell['source'] = PLAYBACK_SOURCE.splitlines(keepends=True)
    nb['cells'].append(cell)
    # keep the file's own JSON layout so the change is one appended cell
    tail = '\n' if raw.endswith('\n') else ''
    original = json.loads(raw)
    layout = next(((i, a) for i in (1, 2, None) for a in (False, True)
                   if json.dumps(original, indent=i, ensure_ascii=a) + tail
                   == raw), (1, False))
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(nb, indent=layout[0], ensure_ascii=layout[1])
                + tail)
    return True
