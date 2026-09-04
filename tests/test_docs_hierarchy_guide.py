"""The hierarchy guide exists, is reachable, and is TRUE.

Two kinds of assertion live here.

Structural ones pin the sections the maintainer review named (F22) and the
links that make the page reachable -- a guide that is written but never
linked is not documentation, and an .rst in no toctree is also a Sphinx
warning against a zero-warning build standard.

The one that matters most is `test_every_doctest_in_the_guide_runs`. Every
example on the page is a real, executed `hyp.plot`/`hyp.predict` call whose
printed output -- trace counts, hierarchy keys, forecast shapes, and the
verbatim text of six error/warning messages -- is compared against what the
library actually produces. A guide is the first place a user meets this
feature, so a stale example is worse than no example: this test is what
makes the page fail loudly instead of aging quietly.
"""
import json
import os
import re

import matplotlib

# the guide draws real figures; never bring up a GUI backend under pytest
matplotlib.use('Agg')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUIDE = os.path.join(REPO, 'docs/hierarchy.rst')


def _read(rel):
    with open(os.path.join(REPO, rel), encoding='utf-8') as handle:
        return handle.read()


REQUIRED_SECTIONS = [
    'Row versus column semantics',
    'Plotting versus forecasting',
    'Hue over a hierarchy',
    'Mean trace construction',
    'Limitations',
    'Dual-axis and list inputs',
    'Return shapes',
    'Fitted model behaviour',
    'Backend parity',
    'Feature names and duplicates',
]


def test_the_guide_page_exists():
    assert os.path.exists(GUIDE)


def test_the_guide_is_in_the_index_toctree():
    index = _read('docs/index.rst')
    toctree = index.split('.. toctree::')[1]
    assert re.search(r'^\s+hierarchy\s*$', toctree, re.M)


def test_the_guide_covers_every_required_section():
    guide = _read('docs/hierarchy.rst')
    missing = [s for s in REQUIRED_SECTIONS if s not in guide]
    assert not missing, f'hierarchy.rst is missing sections: {missing}'


def test_the_comparison_table_is_in_the_guide():
    """The row-plot vs row-forecast divergence, visibly -- including the
    shape rule that decides whether `plot(..., predict=)` works at all
    (Contract 10)."""
    guide = _read('docs/hierarchy.rst')
    assert 'Row MultiIndex, plot' in guide
    assert 'Row MultiIndex, predict' in guide
    assert 'Column MultiIndex' in guide
    assert 'full tuple' in guide
    assert 'at least 2 rows' in guide


def test_the_comparison_table_carries_the_predict_column():
    """Deferred to this task by Task 8 Step 7: the table gains the
    `plot(..., predict=)` rule, whose answer differs on all three rows."""
    guide = _read('docs/hierarchy.rst')
    table = guide.split('.. list-table:: How each axis is read')[1]
    table = table.split('\n\n\n')[0]
    assert '``plot(..., predict=)``' in table
    # the row axis needs every leaf AND every mean; the column axis needs
    # only the frame, since grouping never shortens a trace
    assert 'only when every leaf and mean has at least 2 rows' in table
    assert 'whenever the frame has at least 2 rows' in table
    assert 'n/a' in table, 'hyp.predict has no plot(predict=) to describe'


def test_api_rst_links_the_guide_from_both_sections():
    """`api.rst` is the reference page; both dispatchers that grew a
    hierarchy path have to point at the guide, not just one."""
    api = _read('docs/api.rst')
    assert api.count(':doc:`hierarchy`') >= 2, 'link it from Plot AND Predict'
    predict_section, plot_section = api.split('\nPlot\n')
    predict_section = predict_section.split('\nPredict\n')[1]
    assert ':doc:`hierarchy`' in predict_section, 'Predict section has no link'
    assert ':doc:`hierarchy`' in plot_section, 'Plot section has no link'


def test_tutorials_rst_links_the_guide():
    tut = _read('docs/tutorials.rst')
    assert ':doc:`hierarchy`' in tut


def _notebook_code(rel):
    """Every code cell's source of a notebook, concatenated."""
    cells = json.loads(_read(rel))['cells']
    return '\n'.join(''.join(c['source'])
                     for c in cells if c['cell_type'] == 'code')


def test_the_market_section_describes_the_notebook_it_links():
    """`tutorials.rst` must state something true about the market page.

    History, because the guard has flipped twice: Plan 2 Task 10 deferred a
    hierarchy framing while the notebook still drew "one moving path"; Plan
    4 rebuilt it around a column MultiIndex (six tiled panels); round 2
    (2026-09-03) rebuilt it again as six sectors reduced separately,
    hyperaligned into one space and drawn with their mean -- no MultiIndex
    at all. So the section is checked against what the notebook actually
    does: if the notebook builds a MultiIndex, the section must say
    hierarchy; otherwise it must name the hyperalignment and the market
    mean, and must not claim tiled panels or a hierarchy.
    """
    code = _notebook_code('docs/tutorials/market_sectors.ipynb')
    is_hierarchical = bool(re.search(
        r'MultiIndex\.from_(tuples|product|arrays)', code))
    tut = _read('docs/tutorials.rst')
    section = tut.split('six sectors, one space')[1].split('.. toctree::')[0]
    if is_hierarchical:
        assert 'hierarch' in section.lower(), (
            'market_sectors.ipynb builds a MultiIndex; retitle its '
            'docs/tutorials.rst section to the hierarchy framing')
    else:
        assert "align='hyper'" in code, (
            'the market notebook neither builds a hierarchy nor hyperaligns; '
            'this test does not know what its section should say')
        assert 'hyperalign' in section and 'mean' in section, (
            'the section must name the hyperalignment and the market mean')
        assert 'tiled' not in section and 'MultiIndex' not in section.split(
            '(For the column-MultiIndex route')[0], (
            'the section still describes the retired tiled/hierarchy design')
