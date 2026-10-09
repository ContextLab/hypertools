"""Image scrapers the sphinx-gallery build uses (see ``image_scrapers`` in
conf.py).

Kept in its own module, free of conf.py's import-time side effects, so
tests/test_docs_site_fixes.py can drive the scraper the way sphinx-gallery
does (once per code block) and count what it emits.
"""
import weakref

#: every animation already handled for the running build. Weak, so an
#: animation that an example rebinds and drops is forgotten with it (an
#: ``id()`` set would mistake a new animation at a recycled address for an
#: old one and skip it).
_SCRAPED = weakref.WeakSet()


def matplotlib_and_hyperanimation_scraper(block, block_vars, gallery_conf,
                                          **kwargs):
    """sphinx-gallery's matplotlib scraper, plus the animations it misses.

    sphinx-gallery's matplotlib scraper walks ``plt.get_fignums()`` and pairs
    each MANAGED figure with any ``matplotlib.animation.Animation`` in the
    example's namespace. A ``hyp.plot(..., show=False)`` animation leaves its
    figure unmanaged by pyplot, so the launch examples (which bind the
    ``HyperAnimation`` wrapper the call returns) rendered NOTHING in the
    gallery -- measured 2026-09-03: their pages carried no image block at
    all. This scraper therefore also renders, through sphinx-gallery's own
    animation writer, every ``HyperAnimation`` (or bare ``Animation``) in the
    namespace that the matplotlib scraper did not.

    Each animation is rendered exactly ONCE. Two things used to break that
    (built site, 2026-10: examples/animate.py showed five videos for its two
    ``hyp.plot`` calls) when this ran as a separate scraper AFTER the
    matplotlib one:

    * the matplotlib scraper closes every pyplot figure when it finishes, so
      by then the animation it had just rendered no longer looked "managed"
      and was rendered a second time;
    * an example's namespace persists across its code blocks, so every
      animation bound by an earlier block was rendered again in each later
      block.

    So the set of managed figures is read BEFORE the matplotlib scraper runs,
    and every animation handled (by either path) is remembered.
    """
    from pathlib import PurePosixPath
    import matplotlib.pyplot as plt
    from matplotlib.animation import Animation
    from sphinx_gallery.scrapers import _anim_rst, matplotlib_scraper
    from hypertools.plot.hyper_animation import HyperAnimation

    managed = {plt.figure(num) for num in plt.get_fignums()}
    pending = []
    for value in list(block_vars['example_globals'].values()):
        ani = value.animation if isinstance(value, HyperAnimation) else value
        if not isinstance(ani, Animation) or ani in _SCRAPED:
            continue
        _SCRAPED.add(ani)
        if ani._fig not in managed:     # else the matplotlib scraper has it
            pending.append(ani)

    rst = [matplotlib_scraper(block, block_vars, gallery_conf, **kwargs)]
    for ani in pending:
        image_path = PurePosixPath(next(block_vars['image_path_iterator']))
        rst.append(_anim_rst(ani, image_path, gallery_conf))
    return '\n'.join(part for part in rst if part)
