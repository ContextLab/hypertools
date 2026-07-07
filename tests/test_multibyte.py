# -*- coding: utf-8 -*-
"""Tests for multibyte (e.g. Japanese/CJK) character support (GH #205):

- `hypertools.plot.fonts.find_covering_font`: auto-detects an installed
  font whose character map covers a given set of (possibly non-ASCII)
  strings.
- `hypertools.plot.fonts.resolve_font`: resolves the user-facing `font=`
  kwarg (None/auto, a family name string, a font file path string, or a
  `FontProperties` instance) into a `FontProperties` or `None`.
- `hyp.plot(..., font=...)`: threads the resolved font onto every text
  surface the matplotlib backend draws (point `labels=`, legend, colorbar
  tick labels/axis label, title), so multibyte text renders as real
  glyphs instead of silent "tofu" (empty boxes).

Every assertion here renders a REAL figure on a REAL (Agg) canvas -- no
mocks. Tests that specifically need a CJK-covering font installed are
gated on `covering_font_available` (this machine has Hiragino Sans;
CI provisions a covering font separately -- see the F2 follow-up noted in
notes/issues-to-close-on-merge.md).
"""

import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pytest

import hypertools as hyp
from hypertools.plot.fonts import find_covering_font, resolve_font

JP_LABELS_A = ['いち', 'に', 'さん']
JP_LABELS_B = ['よん', 'ご', 'ろく']

covering_font_available = find_covering_font(['い']) is not None
requires_covering_font = pytest.mark.skipif(
    not covering_font_available,
    reason="no installed font covers this test's CJK characters (CI "
           "provisions one separately; this machine's Hiragino Sans/Noto "
           "Sans/etc. covers it locally)",
)


def _random_points(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 3))


def _rgb_buffer(fig):
    """Rasterize `fig` on a fresh Agg canvas and return its RGB pixels."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


def _missing_glyph_warnings(fig):
    """Draw `fig` on a fresh Agg canvas and return any 'missing from
    font' `UserWarning`s raised during that draw. matplotlib emits these
    at actual RASTERIZATION time (one per distinct missing glyph), not at
    Text-creation time, so the capture must wrap `canvas.draw()` -- not
    the `hyp.plot(...)` call that built the figure."""
    canvas = FigureCanvasAgg(fig)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        canvas.draw()
    return [str(w.message) for w in caught
            if 'missing from font' in str(w.message).lower()]


def _rightmost_leftmost_inked(fig):
    """The rightmost/leftmost inked (non-background) pixel column, and
    the canvas width -- reused from `tests/test_colorbar.py`'s technique
    for proving a legend/colorbar is not clipped off a figure edge."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())[..., :3]
    inked_cols = np.where((buf < 245).any(axis=(0, 2)))[0]
    assert len(inked_cols), "nothing rendered at all"
    return int(inked_cols.max()), int(inked_cols.min()), buf.shape[1]


def _labeled_plot(labels, **kwargs):
    x = _random_points(len(labels))
    return hyp.plot(x, '.', labels=labels, show=False, **kwargs)


# --------------------------------------------------------- covering font


def test_covering_font_available_helper_is_a_bool():
    # sanity: the module-level gate itself must be a real bool derived
    # from a real font scan, not e.g. always-True from a broken skip.
    assert isinstance(covering_font_available, bool)


def test_find_covering_font_returns_none_for_ascii_only():
    assert find_covering_font(['hello', 'world']) is None
    assert find_covering_font([None, 'abc', ['def', None]]) is None


@requires_covering_font
def test_find_covering_font_covers_every_requested_codepoint():
    fp = find_covering_font(JP_LABELS_A)
    assert fp is not None
    from matplotlib.ft2font import FT2Font
    ft = FT2Font(fp.get_file())
    codepoints = {ord(ch) for s in JP_LABELS_A for ch in s if ord(ch) > 127}
    assert all(ft.get_char_index(cp) != 0 for cp in codepoints)


def test_find_covering_font_warns_once_when_nothing_covers():
    # An absurd/unassigned-plane codepoint that no real installed font
    # covers -- exercises the "no covering font" warning path
    # deterministically, independent of what fonts happen to be
    # installed on the machine running this test.
    exotic = chr(0x10FFFD)  # last valid Unicode codepoint (Plane 16, PUA)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = find_covering_font([exotic])
    assert result is None
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1
    msg = str(user_warnings[0].message)
    assert 'font' in msg.lower()
    assert 'font=' in msg or 'Noto Sans CJK' in msg


# ------------------------------------------------------------- resolve_font


def test_resolve_font_none_ascii_only_returns_none():
    assert resolve_font(None, ['plain ascii text']) is None


@requires_covering_font
def test_resolve_font_none_non_ascii_returns_covering_font():
    fp = resolve_font(None, JP_LABELS_A)
    assert fp is not None


def test_resolve_font_fontproperties_passthrough():
    from matplotlib.font_manager import FontProperties
    fp = FontProperties(family='DejaVu Sans')
    assert resolve_font(fp, ['anything']) is fp


def test_resolve_font_invalid_family_raises_value_error():
    with pytest.raises(ValueError, match='not a recognized'):
        resolve_font('TotallyBogusFontFamilyXYZ123', ['abc'])


def test_resolve_font_nonexistent_path_raises_value_error():
    with pytest.raises(ValueError, match='no such file exists'):
        resolve_font('/no/such/path/font.ttf', ['abc'])


def test_resolve_font_invalid_type_raises_value_error():
    with pytest.raises(ValueError, match='font='):
        resolve_font(12345, ['abc'])


# ------------------------------------------------------------ no-tofu proof
#
# (a) no 'missing from font' warning at draw time, for labels/legend/
#     colorbar; (b) rendering the SAME plot with two different (but
#     same-shaped) Japanese label sets must produce DIFFERENT rasterized
#     pixels -- tofu boxes for different strings render identically, so
#     this catches silent tofu even if no warning fires.

@requires_covering_font
def test_labels_no_missing_glyph_warnings():
    fig = _labeled_plot(JP_LABELS_A)
    assert _missing_glyph_warnings(fig) == []
    plt.close(fig)


@requires_covering_font
def test_labels_different_text_renders_different_pixels():
    fig_a = _labeled_plot(JP_LABELS_A)
    fig_b = _labeled_plot(JP_LABELS_B)
    buf_a = _rgb_buffer(fig_a)
    buf_b = _rgb_buffer(fig_b)
    assert buf_a.shape == buf_b.shape
    assert not np.array_equal(buf_a, buf_b), (
        "plots with different Japanese labels rendered IDENTICAL pixels "
        "-- this is exactly what silent tofu (identical empty boxes for "
        "different strings) looks like")
    plt.close(fig_a)
    plt.close(fig_b)


@requires_covering_font
def test_legend_no_missing_glyph_warnings():
    x = [_random_points(15, seed=1), _random_points(15, seed=2)]
    fig = hyp.plot(x, legend=['いろは', 'にほへ'], show=False)
    assert _missing_glyph_warnings(fig) == []
    plt.close(fig)


@requires_covering_font
def test_colorbar_no_missing_glyph_warnings():
    x = [_random_points(15, seed=1), _random_points(15, seed=2)]
    fig = hyp.plot(x, legend=['いろは', 'にほへ'], colorbar=True, show=False)
    assert _missing_glyph_warnings(fig) == []
    plt.close(fig)


@requires_covering_font
def test_colorbar_custom_label_no_missing_glyph_warnings():
    x = [_random_points(15, seed=1), _random_points(15, seed=2)]
    fig = hyp.plot(x, legend=['あ', 'い'],
                   colorbar={'label': '色の凡例'}, show=False)
    assert _missing_glyph_warnings(fig) == []
    plt.close(fig)


@requires_covering_font
def test_title_no_missing_glyph_warnings():
    x = _random_points(20)
    fig = hyp.plot(x, title='日本語のタイトル', show=False)
    assert _missing_glyph_warnings(fig) == []
    plt.close(fig)


# ----------------------------------------------------------- font= kwarg forms

@requires_covering_font
def test_font_kwarg_family_name_string_no_missing_glyph_warnings():
    fp = find_covering_font(JP_LABELS_A)
    fig = _labeled_plot(JP_LABELS_A, font=fp.get_name())
    assert _missing_glyph_warnings(fig) == []
    plt.close(fig)


@requires_covering_font
def test_font_kwarg_file_path_string_no_missing_glyph_warnings():
    fp = find_covering_font(JP_LABELS_A)
    fig = _labeled_plot(JP_LABELS_A, font=fp.get_file())
    assert _missing_glyph_warnings(fig) == []
    plt.close(fig)


@requires_covering_font
def test_font_kwarg_fontproperties_instance_no_missing_glyph_warnings():
    fp = find_covering_font(JP_LABELS_A)
    fig = _labeled_plot(JP_LABELS_A, font=fp)
    assert _missing_glyph_warnings(fig) == []
    plt.close(fig)


def test_font_kwarg_invalid_family_raises_value_error():
    with pytest.raises(ValueError, match='not a recognized'):
        hyp.plot(_random_points(3), '.', labels=['a', 'b', 'c'],
                 font='TotallyBogusFontXYZ123', show=False)


# --------------------------------------------------------- ASCII regression
#
# ASCII-only plots must render EXACTLY as before this feature: `font=`
# resolves to None (no override), and hypertools' pre-existing font
# choices for labels (hardcoded `family="serif"`) and legend (unset --
# rcParams' default family) are unchanged.

def test_ascii_only_resolve_font_returns_none():
    assert resolve_font(None, ['abc', ['def', None], 'ghi']) is None


def test_ascii_only_labels_keep_historical_serif_family():
    fig = _labeled_plot(['a', 'b', 'c'])
    ax = fig.axes[0]
    label_texts = [t for t in ax.texts if t.get_text() in ('a', 'b', 'c')]
    assert label_texts, "no label Text artists found"
    for t in label_texts:
        assert t.get_fontfamily() == ['serif']
        assert t.get_fontproperties().get_file() is None
    plt.close(fig)


def test_ascii_only_legend_keeps_rcparams_default_family():
    x = [_random_points(10, seed=1), _random_points(10, seed=2)]
    fig = hyp.plot(x, legend=['group a', 'group b'], show=False)
    ax = fig.axes[0]
    legend = ax.get_legend()
    default_family = plt.rcParams['font.family']
    for t in legend.get_texts():
        assert t.get_fontfamily() == default_family
        assert t.get_fontproperties().get_file() is None
    plt.close(fig)


# ------------------------------------------------------- legend fit (CJK)

@requires_covering_font
def test_legend_with_long_japanese_labels_fully_inside_canvas():
    g1 = _random_points(30, seed=1)
    g2 = _random_points(30, seed=2)
    g3 = _random_points(30, seed=3)
    long_jp_labels = [
        'とても長いグループラベルあいうえお',
        'もう少し長いラベルかきくけこ',
        'いちばん長いラベルさしすせそたちつてと',
    ]
    fig = hyp.plot([g1, g2, g3], legend=long_jp_labels, show=False)
    rightmost, leftmost, w_px = _rightmost_leftmost_inked(fig)
    assert rightmost < w_px - 1, "Japanese legend clipped off the right edge"
    assert _missing_glyph_warnings(fig) == []
    plt.close(fig)


# ------------------------------------------------------- animated (CJK)

@requires_covering_font
def test_animated_cjk_legend_no_missing_glyph_warnings_across_frames():
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.standard_normal((30, 3)), axis=0)
    hue = ['あ'] * 10 + ['い'] * 10 + ['う'] * 10
    fig, ani = hyp.plot(walk, hue=hue, animate=True, duration=1,
                        frame_rate=5, legend=True, show=False)
    all_warnings = []
    for frame_idx in (0, 1, 2):
        ani._draw_frame(frame_idx)
        all_warnings.extend(_missing_glyph_warnings(fig))
    assert all_warnings == [], (
        f"missing-glyph warning(s) during animated frames: {all_warnings}")
    plt.close(fig)
