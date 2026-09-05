"""Negative tick labels on caller-supplied axes render without a missing-glyph
warning.

The bundled Noto Sans lacks U+2212 MINUS SIGN. hypertools' own axes carry
the full font stack (per-glyph fallback reaches DejaVu Sans), but an ``ax=``
created outside hypertools' rc context has tick labels on the ``sans-serif``
alias, which matplotlib resolves to a single font -- so hypertools' layout
pass warned ``Glyph 8722 (\\N{MINUS SIGN}) missing from font(s) Noto Sans``
for every negative tick (1.1 feature tour on Colab, 2026-09-04).
"""

import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import hypertools as hyp

matplotlib.use('Agg')


def _negative_cloud():
    return np.random.default_rng(0).standard_normal((60, 3)) * 5   # ticks span negatives


def _glyph_warnings(w):
    return [str(x.message) for x in w if 'Glyph' in str(x.message)]


def test_caller_axes_with_hue_and_title_do_not_warn_about_the_minus_glyph():
    labels = ['a'] * 30 + ['b'] * 30
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
        for ax in axes:
            hyp.plot(_negative_cloud(), '.', ax=ax, hue=labels, title='panel', show=False)
        fig.canvas.draw()
    assert _glyph_warnings(w) == []
    plt.close(fig)


def test_hypertools_own_figure_with_negative_ticks_does_not_warn_either():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        fig = hyp.plot(_negative_cloud(), '.', hue=np.linspace(-3, 3, 60), title='t', show=False)
        fig.canvas.draw()
    assert _glyph_warnings(w) == []
    plt.close(fig)


def test_two_panels_with_names_and_align_do_not_warn_about_the_minus_glyph():
    """The align tutorial's case: a second hyp.plot(ax=) call measures the
    first panel's tick labels, formatted earlier with the Unicode minus."""
    base = _negative_cloud()
    rng = np.random.default_rng(1)
    copies = []
    for scale in (1.0, 0.5, 2.0):
        rotation, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        copies.append(scale * base @ rotation)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
        hyp.plot(copies, ['-', '--', ':'], reduce=None, ax=axes[0],
                 names=['a', 'b', 'c'], title='before', show=False)
        hyp.plot(copies, ['-', '--', ':'], reduce=None, ax=axes[1], align='HyperAlign',
                 names=['a', 'b', 'c'], title='after', show=False)
        fig.tight_layout()
        fig.canvas.draw()
    assert _glyph_warnings(w) == []
    plt.close(fig)
