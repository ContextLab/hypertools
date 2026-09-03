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
import doctest
import json
import os
import re
import warnings

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


def test_the_market_section_is_reframed_once_its_notebook_is_hierarchical():
    """Task 10 Step 4 asks for the market section to drop "one moving path"
    for a hierarchy framing. Measured at the time of writing, that notebook
    contains ZERO MultiIndex constructions -- the rewrite around a
    ``(Market, Sector, Ticker)`` column hierarchy is **Plan 4 Task 2 Step
    5**, which has not landed. Describing it as a hierarchy today would make
    `tutorials.rst` state something false about the page it links, so the
    section keeps its accurate title until the notebook changes.

    This is the guard that keeps the deferral honest rather than silent: the
    moment the notebook really is hierarchical, the old framing becomes wrong
    and this test fails, forcing the reframing then.
    """
    is_hierarchical = bool(re.search(
        r'MultiIndex\.from_(tuples|product|arrays)',
        _notebook_code('docs/tutorials/market_sectors.ipynb')))
    tut = _read('docs/tutorials.rst')
    if is_hierarchical:
        assert 'one moving path' not in tut, (
            'market_sectors.ipynb is now hierarchical (Plan 4 Task 2 landed);'
            ' retitle its docs/tutorials.rst section to the hierarchy framing'
            ' and add the sectors-as-leaves / market-mean / price-hue /'
            ' per-trace-forecast synopsis (Plan 2 Task 10 Step 4).')
    else:
        assert 'one moving path' in tut, (
            'the notebook still plots a single trajectory, so the section'
            ' title should still say so')


def test_index_rst_distinguishes_row_and_column_semantics():
    index = _read('docs/index.rst')
    assert 'row MultiIndex' in index and 'column MultiIndex' in index


def test_pipeline_order_documents_the_hierarchy_branch():
    po = _read('docs/pipeline_order.rst')
    assert 'hierarchy' in po.lower()
    assert 'expansion' in po.lower()
    assert 'mean trace' in po.lower()


def test_pipeline_order_alt_text_describes_the_regenerated_diagram():
    """The SVG grew a side branch; an unchanged :alt: describing the old
    diagram would be a documentation defect of its own (Step 5)."""
    po = _read('docs/pipeline_order.rst')
    alt = po.split(':alt:')[1].split('\n\n')[0].lower()
    assert 'hierarchy' in alt and 'branch' in alt


def test_every_doctest_in_the_guide_runs():
    """Run every example on the page against the real library.

    Warnings are silenced only for the DURATION of the run: several examples
    deliberately exercise warning-emitting behaviour (the unequal-length
    truncation, the row-hierarchy list flattening), and the ones whose text
    matters capture and PRINT it, so the assertion on that text is made by
    doctest itself rather than lost here.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        results = doctest.testfile(
            GUIDE, module_relative=False, verbose=False,
            optionflags=doctest.NORMALIZE_WHITESPACE)
    assert results.attempted > 100, (
        f'only {results.attempted} examples ran; the guide should be worked '
        'end to end')
    assert results.failed == 0, (
        f'{results.failed} of {results.attempted} examples in '
        'docs/hierarchy.rst no longer match what hypertools does -- rerun '
        'them and fix the GUIDE (or the code), never the expectation')
