"""Guards for defects found by browsing the BUILT documentation site (2026-10).

Each test exercises the real helper on real inputs: the gallery scraper renders
real ``hyp.plot(..., animate=True)`` animations through sphinx-gallery's own
writer, and the HTML post-processors run on real files written to ``tmp_path``.
"""
import importlib.util
import os
import re

import pytest

DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs'))


def _load(name):
    path = os.path.join(DOCS_DIR, name + '.py')
    if not os.path.exists(path):
        pytest.skip(f'docs/{name}.py not present')
    spec = importlib.util.spec_from_file_location('_hyp_docs_' + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _custom_css_rules():
    """(selector, body) pairs of docs/_static/custom.css, comments removed."""
    path = os.path.join(DOCS_DIR, '_static', 'custom.css')
    if not os.path.exists(path):
        pytest.skip('docs/_static/custom.css not present')
    with open(path, encoding='utf-8') as f:
        css = re.sub(r'/\*.*?\*/', '', f.read(), flags=re.S)
    return [(sel.strip(), body) for sel, body in
            re.findall(r'([^{}]+)\{([^{}]*)\}', css)]


# -- 1. gallery cards: the example page must be reachable from its card -----

def test_gallery_card_link_to_the_example_page_is_not_hidden():
    """sphinx-gallery's card markup carries exactly one link to the example
    page: ``<p><a class="reference internal" href="X.html">``. custom.css
    used to ``display: none`` that paragraph, which left the card with only
    the thumbnail's Colab link and no way to reach the example page."""
    for selector, body in _custom_css_rules():
        for sel in selector.split(','):
            if re.fullmatch(r'\.sphx-glr-thumbcontainer\s+p(\s+a)?', sel.strip()):
                assert not re.search(r'display\s*:\s*none', body), (
                    f'{sel.strip()!r} hides the only link to the example page')


def test_gallery_card_colours_follow_the_theme():
    """A hard-coded colour on the card title / tooltip is illegible in one of
    furo's two themes (``#333`` on the dark background); they must use the
    theme's CSS variables."""
    offenders = []
    for selector, body in _custom_css_rules():
        if 'sphx-glr-thumb' not in selector:
            continue
        for prop, value in re.findall(r'([\w-]+)\s*:\s*([^;]+);', body):
            if prop in ('color', 'background', 'background-color',
                        'border', 'border-color'):
                if re.search(r'#[0-9a-fA-F]{3,8}\b|rgba?\(|\b(white|black)\b',
                             value) and 'var(' not in value:
                    offenders.append((selector, prop, value.strip()))
    assert not offenders, offenders


def test_api_page_titles_keep_their_case():
    """The sitewide ``text-transform: lowercase`` heading style must not
    apply to the autosummary pages' titles: ``hypertools.io.LSLStream``
    rendered as ``hypertools.io.lslstream``, a name that does not exist."""
    rules = _custom_css_rules()
    assert any('text-transform' in body and 'lowercase' in body
               for _, body in rules), 'the lowercase heading style is gone'
    exempt = [sel for sel, body in rules
              if re.search(r'text-transform\s*:\s*none', body)]
    assert any('dl.py' in sel and 'h1' in sel for sel in exempt), exempt


# -- 2. each animation is rendered exactly once ------------------------------

def _scrape_blocks(tmp_path, make_blocks):
    """Run ``make_blocks`` (a generator yielding after each example "block"
    has bound its names into the shared namespace it is handed) through the
    gallery's real scraper, the way sphinx-gallery calls it once per block.
    Returns the number of ``.. video::`` directives emitted per block."""
    pytest.importorskip('sphinx_gallery')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    if not FFMpegWriter.isAvailable():
        pytest.skip('ffmpeg not available')
    from sphinx_gallery.gen_gallery import DEFAULT_GALLERY_CONF
    from sphinx_gallery.scrapers import ImagePathIterator

    scrapers = _load('_gallery_scrapers')
    gallery_conf = dict(DEFAULT_GALLERY_CONF)
    gallery_conf.update(src_dir=str(tmp_path), builder_name='html',
                        matplotlib_animations=(True, 'mp4'))
    (tmp_path / 'images').mkdir()
    block_vars = {
        'example_globals': {},
        'src_file': str(tmp_path / 'example.py'),
        'image_path_iterator': ImagePathIterator(
            str(tmp_path / 'images' / 'sphx_glr_example_{0:03}.png')),
    }
    counts = []
    plt.close('all')
    try:
        for _ in make_blocks(block_vars['example_globals']):
            rst = scrapers.matplotlib_and_hyperanimation_scraper(
                ('code', '', 1), block_vars, gallery_conf)
            counts.append(rst.count('.. video::'))
            # sphinx-gallery closes every pyplot figure after each block
            assert plt.get_fignums() == []
    finally:
        plt.close('all')
    return counts


def _data():
    import numpy as np
    return np.cumsum(np.random.default_rng(0).standard_normal((12, 4)), axis=0)


def test_managed_animations_are_scraped_once_per_block(tmp_path):
    """examples/animate.py: two blocks, each ``fig, ani = hyp.plot(...,
    animate=True)`` on a pyplot-managed figure. The built page showed FIVE
    videos (2 + 3): the matplotlib scraper rendered the block's animation and
    closed its figure, so the HyperAnimation scraper then judged the same
    animation "unmanaged" and rendered it again -- and re-rendered every
    earlier block's animation still alive in the example's namespace."""
    import hypertools as hyp

    def blocks(ns):
        ns['fig'], ns['ani'] = hyp.plot(_data(), animate=True,
                                        backend='matplotlib', duration=1)
        yield
        ns['fig'], ns['ani_mds'] = hyp.plot(_data(), animate=True,
                                            backend='matplotlib', duration=1)
        yield
        yield                       # a later block that plots nothing

    assert _scrape_blocks(tmp_path, blocks) == [1, 1, 0]


def test_unmanaged_show_false_animation_is_still_scraped_once(tmp_path):
    """The reason the HyperAnimation scraper exists: ``show=False`` leaves the
    figure unmanaged by pyplot, so the matplotlib scraper never sees it. It
    must still be rendered -- once, in the block that created it."""
    import hypertools as hyp

    def blocks(ns):
        ns['ani'] = hyp.plot(_data(), animate=True, backend='matplotlib',
                             duration=1, show=False)
        yield
        yield

    assert _scrape_blocks(tmp_path, blocks) == [1, 0]


# -- 5. tutorial pages: no dead plotly import, no second MathJax -------------

NOTEBOOK_PLOTLY_OUTPUT = '''<html><head>
<script defer="defer" src="https://cdn.jsdelivr.net/npm/mathjax@4/tex-mml-chtml.js"></script>
</head><body>
<div class="output_area rendered_html docutils container">
        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        </script>
        <script type="module">import "https://cdn.plot.ly/plotly-3.6.0.min"</script>
        </div>
<div class="output_area rendered_html docutils container">
<div style="height:480px; width:640px;">            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script>if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script>window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-3.6.0.min.js" integrity="sha256-abc" crossorigin="anonymous"></script>                <div id="f1" class="plotly-graph-div" style="height:100%; width:100%;"></div>            <script>window.PLOTLYENV=window.PLOTLYENV || {};</script></div>
</div></body></html>
'''


def test_notebook_plotly_scripts_are_cleaned(tmp_path):
    """tutorials/plot.html logged a 403 and a MathJax TypeError: plotly's
    notebook renderer stores (a) a bare ES-module import of
    ``cdn.plot.ly/plotly-X.min`` (no ``.js``; the CDN answers 403) and (b) a
    MathJax 2.7.5 <script> in every figure, which collides with the MathJax 4
    the page itself loads. The figure's own ``plotly-X.min.js`` <script> must
    survive -- it is what actually draws the figure."""
    mod = _load('post_build')
    page = tmp_path / 'tutorials' / 'plot.html'
    page.parent.mkdir()
    page.write_text(NOTEBOOK_PLOTLY_OUTPUT, encoding='utf-8')
    untouched = tmp_path / 'tutorials' / 'other.html'
    untouched.write_text('<html><body><p>no plotly</p></body></html>',
                         encoding='utf-8')

    assert mod.clean_notebook_plotly_scripts(str(tmp_path)) == 1
    html = page.read_text(encoding='utf-8')
    assert 'plotly-3.6.0.min"' not in html
    assert 'mathjax/2.7.5' not in html
    assert html.count('src="https://cdn.plot.ly/plotly-3.6.0.min.js"') == 1
    assert 'class="plotly-graph-div"' in html
    assert 'mathjax@4/tex-mml-chtml.js' in html      # the page's own MathJax
    assert untouched.read_text(encoding='utf-8') == (
        '<html><body><p>no plotly</p></body></html>')
    # idempotent
    assert mod.clean_notebook_plotly_scripts(str(tmp_path)) == 0


# -- 6. gallery tooltips: no literal RST markup ------------------------------

@pytest.mark.parametrize('raw, clean', [
    ('Anything (``fig.savefig(...)``, grabbing fig.axes[0]) works.',
     'Anything (fig.savefig(...), grabbing fig.axes[0]) works.'),
    ('Every stage dispatcher (`hyp.manip`, hyp.normalize) accepts it.',
     'Every stage dispatcher (hyp.manip, hyp.normalize) accepts it.'),
    ('Its vectorizer=/``semantic=`` specs; .mov/``.avi`` too.',
     'Its vectorizer=/semantic= specs; .mov/.avi too.'),
    ('See :func:`hypertools.plot` and :class:`~hypertools.Pipeline`.',
     'See hypertools.plot and Pipeline.'),
    ('(``hyp.load(&#x27;digits&#x27;)``) holds 8x8 images',
     '(hyp.load(&#x27;digits&#x27;)) holds 8x8 images'),
    ('No markup at all: 5 &lt; 6 &amp; a - b.',
     'No markup at all: 5 &lt; 6 &amp; a - b.'),
])
def test_tooltip_markup_is_stripped(raw, clean):
    assert _load('post_build').strip_rst_markup(raw) == clean


def test_gallery_tooltips_are_cleaned_in_the_built_index(tmp_path):
    mod = _load('post_build')
    index = tmp_path / 'index.html'
    card = ('<div class="sphx-glr-thumbcontainer" tooltip="{tip}">'
            '<img alt="" src="../_images/sphx_glr_x_thumb.png" />\n'
            '<p><a class="reference internal" href="x.html">'
            '<span class="doc">X</span></a></p>\n'
            '  <div class="sphx-glr-thumbnail-title">X</div>\n</div>')
    body = '<p>Prose keeps its <code>``literal``</code> text.</p>'
    index.write_text(
        card.format(tip='Pass (``animate=True``) to `hyp.plot`.')
        + card.format(tip='Nothing to clean here.') + body, encoding='utf-8')

    assert mod.clean_gallery_tooltips(str(index)) == 1
    html = index.read_text(encoding='utf-8')
    assert 'tooltip="Pass (animate=True) to hyp.plot."' in html
    assert 'tooltip="Nothing to clean here."' in html
    assert body in html                 # only tooltip attributes are touched
    assert mod.clean_gallery_tooltips(str(index)) == 0
