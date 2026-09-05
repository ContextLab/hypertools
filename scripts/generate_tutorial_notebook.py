"""Regenerate a launch tutorial notebook from its example script.

Usage: .venv/bin/python scripts/generate_tutorial_notebook.py [stem ...]
       (no stems = every SPECS entry), then execute with
       scripts/execute_tutorial.py.

Each notebook is derived from ONE example under examples/ (the Plan 4
loader/builder scripts): the module docstring becomes the intro, the code is
split into sections at the symbols named in SPECS (a section starts at the
comment block directly above its first symbol), the ``__main__`` block
becomes the "load and build" cell, and a save cell writes the mp4 the final
markdown cell embeds. The Colab install cell and the notebook metadata are
carried over byte-identical from the notebook being replaced, so the
release-time flip of the install cell keeps working on the regenerated file.
A stem whose notebook does not exist yet (a NEW spec) takes both from
``TEMPLATE`` instead, so its install cell matches the hand-written tutorials.

Why a script and not a hand edit: the five notebooks were regenerated five
times on 2026-09-03 alone, and a hand-split drifts from the example it
documents (a helper renamed in the script but not the notebook was the
reason for this file). ``tests/test_examples_are_native.py`` checks the pair
stays in step.
"""
import ast
import json
import os
import re
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, 'examples')
TUTORIALS = os.path.join(ROOT, 'docs', 'tutorials')
#: the install cell and metadata a brand-new notebook starts from
TEMPLATE = os.path.join(TUTORIALS, 'cluster.ipynb')
DRAW_LAST = '_ = anim.draw_frame(anim.n_frames - 1)   # the fully revealed frame\n'

#: notebook stem -> (example module, mp4 dpi, [(heading, prose, first symbol)])
#: The dpi is the docs artefact's resolution: 70 for the two long clips
#: (market 1200 frames, weather 2400 frames -- at 100 dpi they were 15 and
#: 27 MB, too heavy for an autoplaying tutorial page), 100 for the rest.
#: The first section always starts right after the docstring; its symbol is
#: therefore ignored and given as None. `__main__` names the guard block.
SPECS = {
    'market_sectors': ('animate_market_sectors', 100, [
        ('## 1. Imports, a disk cache, and the universe',
         'Six sectors of four or five tickers each -- unequal on purpose -- '
         'with fixed sector colours, the animation length, a whitened PCA so '
         "a sector's common growth does not flatten its path onto one line, "
         'and the colormap that tints the date red or green.', None),
        ('## 2. Prices and share counts, with a synthetic fallback',
         'Yahoo Finance supplies adjusted AND unadjusted daily closes (an '
         'explicit date window: `range=max` silently degrades to quarterly '
         "bars); the SEC's XBRL API supplies reported shares outstanding, "
         'which are not split-adjusted, so market cap multiplies them by the '
         'unadjusted close. Both are cached on disk. These are the only '
         'functions that touch the network; `HYPERTOOLS_OFFLINE` makes them '
         'refuse rather than degrade, which is how the test-suite proves the '
         'import path fetches nothing.', 'Market'),
        ('## 3. Growth curves per sector, market-cap weights, the basket\'s '
         'return',
         'What is plotted is each stock\'s cumulative log return since the '
         'first month (a growth curve), one matrix per sector. Each sector\'s '
         'share of the basket\'s capitalisation, month by month, becomes the '
         'colour weights of the market path, and the cap-weighted trailing '
         'twelve-month return tints the title. `load_market` is the real '
         'path; `fixture_data` is the same payload from the seeded basket.',
         'assemble'),
        ('## 4. Reduce per sector, hyperalign, draw seven paths',
         'Three library calls: `hyp.reduce` per sector (its own stocks, its '
         'own space), `hyp.align(..., align=\'HyperAlign\')` into one shared '
         'space, and `hyp.plot` on the six aligned paths plus their mean, '
         'coloured through the mixture hue. The `on_frame` hook only sets '
         'the title: the date under the head, tinted by the basket\'s '
         'return.', 'construct_artifact'),
    ]),
    'weather_decades': ('animate_weather_decades', 100, [
        ('## 1. Imports and a disk cache',
         'The temperature matrix, the city coordinates and the coastlines '
         'are fetched once and cached in the system temp directory.', None),
        ('## 2. Fetch the paper\'s archive and the coastlines, with fallbacks',
         'The fetchers are the only code in this notebook that touches the '
         'network. If the archive is unavailable a seeded synthetic set of '
         'twenty seasonal series (opposite hemispheric phase, slow warming '
         'drift, synthetic coordinates) takes its place; if only the '
         'coastline file is unavailable the map panel says so and draws the '
         'dots on a bare longitude/latitude frame.', 'Weather'),
        ('## 3. One call, three panels',
         'Each keyword of the `hyp.plot` call is a stage, run in the '
         "library's canonical order: `manip='Smooth'`, `normalize='across'`, "
         'the default reduction to three dimensions, then a two-minute '
         'animated reveal with one camera orbit. The map and the '
         'mean-temperature panels are added to the same figure and kept in '
         'step with the head of the path by one `on_frame` hook, which also '
         'sets the figure-wide month/year title.', 'construct_artifact'),
    ]),
    'painting_embeddings': ('animate_painting_embeddings', 100, [
        ('## 1. Imports, a disk cache, and the five paintings',
         'The descriptions are bundled inline with each painting\'s artist, '
         'year, Wikimedia file and a hand-picked fallback colour. The '
         'layout constants place the text column and the thumbnails.',
         None),
        ('## 2. Windows, canvases and colours',
         'Each description is cut into overlapping ten-word windows. The '
         'canvas is downloaded once (the only network access here) and '
         '`image_palette` picks its most salient legible colour; offline, '
         'the fallback colour and a flat swatch stand in. `fixture_data` '
         'takes every colour from the one committed thumbnail and embeds '
         'with TF-IDF, so no test fetches a canvas or a model.', 'Paintings'),
        ('## 3. One call, plus the annotated column',
         'Raw text in, five clouds out, spun by `animate=\'spin\'`. '
         'Everything after the call is annotation of the same figure: the '
         'name with artist and year, the embedded description, and a '
         'thumbnail centred on it.', 'construct_artifact'),
    ]),
    'conversation_shape': ('animate_conversation', 100, [
        ('## 1. Imports, the speakers, and the turns',
         'The Mad Tea-Party turns, spoken text only, with each speaker\'s '
         'colour, the recency-fade constants, the title wrap width and the '
         'tail length.', None),
        ('## 2. Windows and the payload',
         'Each turn becomes a list of sliding word windows; the payload '
         'carries the windows, the speakers, the spoken lines (for the '
         'titles) and the vectorizer. `fixture_data` embeds with TF-IDF so '
         'no test downloads a model.', 'Conversation'),
        ('## 3. The recency fade and the title hooks',
         '`recency_fade` fades earlier turns on the public `on_frame` hook; '
         '`speaker_title` tints the title with the current speaker\'s colour '
         'on every frame; `make_room_for_title` grows the figure so a '
         'two-line title clears the box.', 'turn_alpha'),
        ('## 4. One call',
         'Raw dialogue in, one disjoint trajectory per turn, coloured by '
         'speaker, revealed one turn at a time over thirty seconds and two '
         'camera rotations.', 'construct_artifact'),
    ]),
    'morph_shapes_zoo': ('animate_morph_zoo', 100, [
        ('## 1. Imports and the sampling constants',
         'The zoo\'s five shapes, the per-shape point cap, the cube shrink '
         'and the title styling the hook re-applies each frame.', None),
        ('## 2. The shapes zoo, sampled and loop-closed, with parametric '
         'stand-ins',
         '`hyp.load` fetches each shape once into `~/hypertools_data`; the '
         'clouds are normalised into the unit cube, sampled, and the first '
         'is repeated at the end so a looping player never hard-cuts. On a '
         'cold cache with no network, five parametric clouds stand in.',
         'Shapes'),
        ('## 3. One morph call, with restyled titles',
         'The per-segment `rotations` list sets each hold and transition\'s '
         'screen time; `title=` names the shape while it holds. The '
         '`on_frame` hook re-applies the larger, lowered title style on '
         'every frame.', 'construct_artifact'),
    ]),
    'animate_forecast': ('animate_forecast', 100, [
        ('## 1. Imports, the regions, and the clip constants',
         'Three regions of six cities each, every region spanning both '
         'hemispheres so its year is a loop rather than a line; the five-year '
         'window, the twelve-month horizon and the frame budget.', None),
        ('## 2. The archive, with a synthetic fallback',
         'The fetcher is the only code in this notebook that touches the '
         'network: the paper\'s temperature archive, cached once. If it is '
         'unavailable a seeded synthetic set of three seasonal regions takes '
         'its place, and `fixture_data` is that same payload, so no test '
         'fetches anything.', 'Climate'),
        ('## 3. One call: an animated forecast with a fading fan',
         '`predict=\'Kalman\'` with `animate=True` refits on every distinct '
         'revealed history and re-anchors the forecast on the last revealed '
         'month; `forecast_trail=True` keeps the earlier fits as a fading '
         'fan. `forecast_hue=`, `forecast_palette=` and `forecast_fmt=` '
         'restyle only the forecasts, and `slow_warning_seconds=None` '
         'silences the long-schedule notice for a wait that is known.',
         'construct_artifact'),
    ]),
}


def docstring_to_markdown(doc):
    """RST-flavoured module docstring -> markdown intro cell."""
    lines = doc.strip('\n').splitlines()
    # the title is the line between two ==== rules
    if lines and set(lines[0]) == {'='}:
        title, body = lines[1].strip(), lines[3:]
    else:
        title, body = lines[0].strip(), lines[1:]
    text = '\n'.join(body).strip('\n')
    text = re.sub(r'``([^`]+)``', r'`\1`', text)
    text = re.sub(r'`([^`<]+?) <([^>]+)>`_', r'[\1](\2)', text)
    return f'# {title}\n\n{text}\n'


def section_start(lines, node):
    """First line of the comment block directly above `node` (0-based)."""
    start = node.lineno - 1
    while start > 0 and lines[start - 1].startswith('#'):
        start -= 1
    return start


def split_sections(source, spec):
    """(heading, prose, code) per section, plus the dedented main body."""
    tree = ast.parse(source)
    lines = source.splitlines()
    doc = ast.get_docstring(tree)
    top = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            top[node.name] = node
        elif isinstance(node, ast.If):
            top['__main__'] = node
    body_start = tree.body[0].end_lineno       # line after the docstring
    while lines[body_start].startswith('#') or not lines[body_start].strip():
        body_start += 1                         # the source/licence comments
    cuts = [body_start] + [section_start(lines, top[sym])
                           for _h, _p, sym in spec[1:]]
    cuts.append(section_start(lines, top['__main__']))
    sections = []
    for (heading, prose, _sym), a, b in zip(spec, cuts, cuts[1:]):
        code = '\n'.join(lines[a:b]).strip('\n') + '\n'
        sections.append((heading, prose, code))
    main = top['__main__']
    main_body = textwrap.dedent('\n'.join(lines[main.lineno:main.end_lineno]))
    return doc, sections, main_body.strip('\n') + '\n'


def cell(kind, text, execution_count=None):
    src = text.splitlines(keepends=True)
    c = {'cell_type': kind, 'metadata': {}, 'source': src}
    if kind == 'code':
        c.update(execution_count=execution_count, outputs=[])
    return c


def build(stem):
    module, dpi, spec = SPECS[stem]
    path = os.path.join(TUTORIALS, stem + '.ipynb')
    with open(path if os.path.exists(path) else TEMPLATE) as handle:
        old = json.load(handle)
    install = old['cells'][0]
    assert 'pip install' in ''.join(install['source']), path
    install = {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
               'outputs': [], 'source': install['source']}
    with open(os.path.join(EXAMPLES, module + '.py')) as handle:
        source = handle.read()
    doc, sections, main_body = split_sections(source, spec)
    cells = [install, cell('markdown', docstring_to_markdown(doc))]
    for heading, prose, code in sections:
        cells.append(cell('markdown', f'{heading}\n\n{prose}\n'))
        cells.append(cell('code', code))
    n = len(sections) + 1
    cells.append(cell('markdown', f'## {n}. Load the data and build the '
                                  'animation\n'))
    cells.append(cell('code', main_body + DRAW_LAST))
    cells.append(cell('markdown', f'## {n + 1}. Save the animation\n'))
    cells.append(cell('code', f"anim.save('{stem}.mp4', dpi={dpi})\n"
                              f"print('saved {stem}.mp4')\n"))
    title = docstring_to_markdown(doc).splitlines()[0][2:]
    # an mp4, not a GIF: the rebuilt clips run one to two minutes (1200-2400
    # frames), which a GIF cannot carry at any useful size. The <video> tag
    # is what the page plays; the markdown link beneath it is what makes
    # nbsphinx copy the file into the build (it copies linked local files,
    # not raw-HTML sources), and gives GitHub's notebook viewer something.
    cells.append(cell('markdown',
                      f'<video controls loop muted autoplay playsinline '
                      f'src="{stem}.mp4" title="{title}" '
                      f'style="max-width: 100%"></video>\n\n'
                      f'[Download the clip]({stem}.mp4)\n'))
    notebook = {'cells': cells, 'metadata': old['metadata'],
                'nbformat': old['nbformat'], 'nbformat_minor': old['nbformat_minor']}
    with open(path, 'w') as handle:
        json.dump(notebook, handle, indent=1, ensure_ascii=False)
        handle.write('\n')
    print(f'{stem}: {len(cells)} cells from examples/{module}.py')


if __name__ == '__main__':
    for stem in sys.argv[1:] or SPECS:
        build(stem)
