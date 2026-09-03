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

import importlib.util
import json
import os
import subprocess
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pytest

import hypertools as hyp
from hypertools.plot.fonts import (_codepoints_uncovered_by_stack,
                                   find_covering_font, resolve_font,
                                   sans_serif_stack)

JP_LABELS_A = ['いち', 'に', 'さん']
JP_LABELS_B = ['よん', 'ご', 'ろく']

# "can hypertools render CJK here?" -- i.e. does the Noto-first fallback stack
# (bundled Noto Sans + installed pan-CJK families + DejaVu) cover a CJK glyph?
# (Since the stack now supplies CJK per-glyph, find_covering_font(CJK) returns
# None on a machine whose stack already covers it -- so it can no longer be
# used as this gate.)
covering_font_available = not _codepoints_uncovered_by_stack({ord('い')})
requires_covering_font = pytest.mark.skipif(
    not covering_font_available,
    reason="no installed font covers this test's CJK characters (CI "
           "provisions one separately; this machine's Hiragino Sans/Noto "
           "Sans/etc. covers it locally)",
)


def _installed_cjk_fontproperties():
    """A `FontProperties` for SOME installed font covering CJK 'い'.

    Auto-detection (`resolve_font(None, ...)`) now returns None for CJK when the
    fallback STACK already covers it, so tests that need a concrete CJK font --
    e.g. to pass an explicit ``font=`` -- get one directly here. Only called
    under `requires_covering_font`, so a covering font is known to exist."""
    from matplotlib.font_manager import FontProperties
    from hypertools.plot.fonts import _ordered_font_entries, _font_covers
    for entry in _ordered_font_entries():
        if _font_covers(entry.fname, {ord('い')}):
            return FontProperties(fname=entry.fname)
    return None


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


def test_ci_has_covering_font():
    # GH #205 (F2 follow-up): every `requires_covering_font`-gated test in
    # this file SKIPS (doesn't fail) when no installed font covers the
    # test's CJK characters -- correct for a random contributor's laptop
    # that may not have a CJK font installed, but on CI this must be a
    # hard FAILURE instead: if it silently skips there, the CJK code path
    # (find_covering_font/resolve_font, the no-tofu proofs, the font=
    # kwarg forms, the legend-fit and animated-frame checks) gets ZERO
    # coverage on every PR, forever, with a green checkmark. The CI
    # workflow (.github/workflows/test.yml) installs fonts-noto-cjk on
    # ubuntu-latest and rebuilds matplotlib's font cache afterward
    # specifically so this never fires; macOS/windows runners ship
    # Hiragino Sans / Yu Gothic-family fonts out of the box.
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        pytest.skip("only meaningful on CI (GITHUB_ACTIONS=true); a "
                    "missing covering font on a local machine is expected "
                    "and handled by requires_covering_font's skip")
    assert covering_font_available, (
        "no installed font covers a basic CJK test string on this CI "
        "runner -- every requires_covering_font-gated test in "
        "tests/test_multibyte.py just silently skipped. Check that "
        ".github/workflows/test.yml's font-provisioning step (fonts-noto-cjk "
        "install + fc-cache + matplotlib font-cache rebuild) ran, and that "
        "it ran BEFORE this test's font scan (matplotlib caches its font "
        "list after first scan -- a stale cache from before the font "
        "install would hide the new font from find_covering_font)."
    )


def test_find_covering_font_returns_none_for_ascii_only():
    assert find_covering_font(['hello', 'world']) is None
    assert find_covering_font([None, 'abc', ['def', None]]) is None


@requires_covering_font
def test_find_covering_font_returns_none_when_stack_already_covers_cjk():
    # NEW contract (maintainer font review): when the Noto-first fallback stack
    # already covers the text (CJK included, on a machine whose CJK family is in
    # the stack), find_covering_font returns None -- no override, so the primary
    # face stays the bundled Noto Sans. It only returns a font for a real GAP.
    assert not _codepoints_uncovered_by_stack(
        {ord(ch) for s in JP_LABELS_A for ch in s if ord(ch) > 127})
    assert find_covering_font(JP_LABELS_A) is None


def test_find_covering_font_warns_once_when_nothing_covers():
    # Unicode NONCHARACTERS (U+FDD0, U+10FFFE) are guaranteed by the
    # standard never to be assigned and are excluded from font cmaps --
    # unlike the Private Use Area codepoint this test originally used
    # (U+10FFFD), which pan-Unicode fonts on CI runners genuinely map,
    # making find_covering_font correctly return a font instead of None.
    # Requiring BOTH noncharacters (different planes) makes a false
    # positive from a permissive font practically impossible.
    exotic = chr(0xFDD0) + chr(0x10FFFE)
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
def test_resolve_font_none_cjk_returns_none_stack_covers_it():
    # auto-detect returns None because the fallback stack renders CJK per-glyph
    # (Noto stays primary) -- and the text still renders with no tofu (proved by
    # the no-missing-glyph-warnings tests below)
    assert resolve_font(None, JP_LABELS_A) is None


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
    fp = _installed_cjk_fontproperties()
    fig = _labeled_plot(JP_LABELS_A, font=fp.get_name())
    assert _missing_glyph_warnings(fig) == []
    plt.close(fig)


@requires_covering_font
def test_font_kwarg_file_path_string_no_missing_glyph_warnings():
    fp = _installed_cjk_fontproperties()
    fig = _labeled_plot(JP_LABELS_A, font=fp.get_file())
    assert _missing_glyph_warnings(fig) == []
    plt.close(fig)


@requires_covering_font
def test_font_kwarg_fontproperties_instance_no_missing_glyph_warnings():
    fp = _installed_cjk_fontproperties()
    fig = _labeled_plot(JP_LABELS_A, font=fp)
    assert _missing_glyph_warnings(fig) == []
    plt.close(fig)


def test_font_kwarg_invalid_family_raises_value_error():
    with pytest.raises(ValueError, match='not a recognized'):
        hyp.plot(_random_points(3), '.', labels=['a', 'b', 'c'],
                 font='TotallyBogusFontXYZ123', show=False)


# --------------------------------------------------------- ASCII regression
#
# ASCII-only plots resolve `font=` to None (no per-artist override) and draw
# every text surface in hypertools' bundled sans-serif fallback STACK.
#
# NOTE: point labels previously hardcoded `family="serif"`. That was changed
# deliberately (maintainer font review): it clashed with the sans-serif used
# everywhere else, and -- because a hardcoded generic resolves through
# matplotlib's stock serif list rather than hypertools' stack -- a label
# character the serif faces happened to lack (e.g. U+2726 '✦') rendered as
# "tofu" even when an installed font had a glyph for it.

def test_ascii_only_resolve_font_returns_none():
    assert resolve_font(None, ['abc', ['def', None], 'ghi']) is None


def test_ascii_only_labels_inherit_the_sans_stack():
    fig = _labeled_plot(['a', 'b', 'c'])
    ax = fig.axes[0]
    label_texts = [t for t in ax.texts if t.get_text() in ('a', 'b', 'c')]
    assert label_texts, "no label Text artists found"
    for t in label_texts:
        assert t.get_fontfamily() != ['serif'], "labels must not force serif"
        assert t.get_fontfamily() == sans_serif_stack()
        assert t.get_fontproperties().get_file() is None
    plt.close(fig)


def test_ascii_only_legend_inherits_the_sans_stack():
    x = [_random_points(10, seed=1), _random_points(10, seed=2)]
    fig = hyp.plot(x, legend=['group a', 'group b'], show=False)
    ax = fig.axes[0]
    legend = ax.get_legend()
    for t in legend.get_texts():
        assert t.get_fontfamily() == sans_serif_stack()
        assert t.get_fontproperties().get_file() is None
    plt.close(fig)


def test_plot_does_not_leak_font_rcparams_into_global_state():
    # the stack is applied inside a scoped rc_context -- plotting must not
    # change the font the user's own (non-hypertools) figures render with
    before = list(plt.rcParams['font.family'])
    before_sans = list(plt.rcParams['font.sans-serif'])
    plt.close(_labeled_plot(['a', 'b', 'c']))
    assert list(plt.rcParams['font.family']) == before
    assert list(plt.rcParams['font.sans-serif']) == before_sans


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


# --------------------------------------------------------- plotly backend
#
# GH #205 F2: the plotly backend now honors `font=` too (matplotlib was
# F1). Plotly text surfaces only take a FAMILY NAME (not a font file), set
# once at `layout.font.family` and inherited by every text surface
# hypertools creates (legend trace names, colorbar title/ticktext, plot
# title) unless that surface hardcodes its own family (none do, after
# this change -- the plot title used to; see plotly_backend.py).
#
# GH #205 F3: plotly now ALSO draws point `labels=` annotations, at parity
# with matplotlib's `annotate_plot` (`layout.scene.annotations` for 3-D,
# `layout.annotations` for 2-D) -- see tests/test_plotly_labels.py for the
# full label-to-point mapping / mismatched-count / animate parity suite.
# This module only covers the multibyte (CJK) angle: exact string survival
# and the kaleido anti-tofu pixel-diff check, below.

_RENDER_PLOTLY_SCRIPT = os.path.join(
    os.path.dirname(__file__), '..', 'scripts', 'render_multibyte_plotly.py')
_KALEIDO_TIMEOUT_S = 120

# Imported from the render script rather than restated, so the two cannot
# drift into disagreeing about which exit code means "no usable browser".
_render_spec = importlib.util.spec_from_file_location(
    '_render_multibyte_plotly', _RENDER_PLOTLY_SCRIPT)
_render_mod = importlib.util.module_from_spec(_render_spec)
_render_spec.loader.exec_module(_render_mod)
_NO_BROWSER_EXIT = _render_mod.NO_BROWSER_EXIT
_BROWSER_PATH_ENV = _render_mod.BROWSER_PATH_ENV
_is_browser_lifecycle_error = _render_mod.is_browser_lifecycle_error


def _plotly_plot(legend, **kwargs):
    data = [_random_points(10, seed=i) for i in range(len(legend))]
    return hyp.plot(data, legend=legend, backend='plotly', show=False,
                    **kwargs)


def test_plotly_labels_kwarg_is_drawn_as_scene_annotations():
    # replaces the old F2 placeholder (`_kwarg_is_accepted_but_not_drawn`):
    # plotly now really draws `labels=`, one `layout.scene.annotations`
    # entry per non-None label, in the same order as the (single-dataset)
    # input points.
    x = _random_points(3)
    fig = hyp.plot(x, '.', labels=['a', 'b', 'c'], backend='plotly',
                   show=False)
    assert fig is not None
    annotations = fig.layout.scene.annotations
    assert [a.text for a in annotations] == ['a', 'b', 'c']
    # still no `text` on the data trace itself -- annotations are a
    # separate layout-level artist, exactly like matplotlib's `ax.annotate`
    # calls are separate from the `ax.plot` trace
    assert all(getattr(tr, 'text', None) is None for tr in fig.data)


@requires_covering_font
def test_plotly_japanese_labels_exact_string_equality():
    import importlib
    stack = importlib.import_module(
        'hypertools.plot.plotly_backend')._PLOTLY_SANS_STACK
    x = _random_points(3)
    fig = hyp.plot(x, '.', labels=JP_LABELS_A, backend='plotly', show=False)
    annotations = fig.layout.scene.annotations
    assert [a.text for a in annotations] == JP_LABELS_A
    # CJK no longer prepends an auto-detected family; the browser resolves the
    # default CSS stack (which lists pan-CJK fallbacks after the Latin faces)
    # per glyph. The layout and every annotation carry that default stack.
    assert fig.layout.font.family == stack
    for a in annotations:
        assert a.font.family == stack


@requires_covering_font
def test_plotly_legend_names_exact_string_equality():
    fig = _plotly_plot(JP_LABELS_A[:2])
    names = [tr.name for tr in fig.data if tr.name]
    assert names == JP_LABELS_A[:2]


@requires_covering_font
def test_plotly_title_exact_string_equality():
    fig = hyp.plot(_random_points(20), title='日本語のタイトル',
                   backend='plotly', show=False)
    assert fig.layout.title.text == '日本語のタイトル'


@requires_covering_font
def test_plotly_colorbar_label_and_ticktext_exact_string_equality():
    fig = hyp.plot([_random_points(10, seed=1), _random_points(10, seed=2)],
                   legend=['あ', 'い'], colorbar={'label': '色の凡例'},
                   backend='plotly', show=False)
    cb = fig.data[-1].marker.colorbar
    assert cb.title.text == '色の凡例'
    assert list(cb.ticktext) == ['あ', 'い']


@requires_covering_font
def test_plotly_layout_font_uses_default_stack_for_auto_cjk():
    # auto-detected CJK adds no override now -- plotly uses its default CSS
    # stack, which already lists pan-CJK fallbacks for the browser to resolve
    # per glyph (and keeps Noto Sans preferred)
    import importlib
    stack = importlib.import_module(
        'hypertools.plot.plotly_backend')._PLOTLY_SANS_STACK
    fig = _plotly_plot(JP_LABELS_A[:2])
    assert fig.layout.font.family == stack
    assert 'Noto Sans' in fig.layout.font.family


@requires_covering_font
def test_plotly_layout_font_family_matches_explicit_font_kwarg():
    name = _installed_cjk_fontproperties().get_name()
    # ASCII-only text, but font= given explicitly -- must still be honored,
    # prepended to the default stack
    fig = _plotly_plot(['group a', 'group b'], font=name)
    assert fig.layout.font.family is not None
    assert name in fig.layout.font.family


def test_plotly_ascii_only_uses_the_default_sans_stack():
    # Changed deliberately (maintainer font review): layout.font.family used to
    # be left unset for ASCII-only text, so plotly fell back to its OWN default
    # face and looked nothing like the matplotlib backend. It now always
    # carries hypertools' sans stack -- a CSS stack, which a browser resolves
    # PER GLYPH, so the pan-CJK entries keep mixed-script text rendering too.
    fig = _plotly_plot(['group a', 'group b'])
    family = fig.layout.font.family
    assert family is not None
    assert 'Noto Sans' in family          # the bundled face leads the stack
    assert family.strip().endswith('sans-serif')   # guaranteed final fallback


def test_plotly_font_kwarg_invalid_family_raises_value_error():
    with pytest.raises(ValueError, match='not a recognized'):
        hyp.plot(_random_points(3), legend=None, backend='plotly',
                 font='TotallyBogusFontXYZ123', show=False)


# ---------------------------------------------- plotly pixel-level (kaleido)
#
# static kaleido export is exercised directly elsewhere in this repo (e.g.
# tests/test_marker_parity.py calls fig.write_image in-process with no
# issue) -- unlike the 6 known deadlock-prone ANIMATED/SVG plotly export
# tests deselected in test_animation_export.py/test_round3.py. Still, this
# spawns a real Chromium subprocess via kaleido, so these checks run the
# render in a SEPARATE process (scripts/render_multibyte_plotly.py) via
# subprocess.run(..., timeout=...), which can actually kill a wedged
# child -- and skip (not fail) the test if that timeout fires.

def _render_plotly_png(legend, title, out_path, labels=None):
    try:
        result = subprocess.run(
            [sys.executable, _RENDER_PLOTLY_SCRIPT, json.dumps(legend),
             title, out_path, json.dumps(labels)],
            timeout=_KALEIDO_TIMEOUT_S, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        pytest.skip(
            f"kaleido/Chromium render did not complete within "
            f"{_KALEIDO_TIMEOUT_S}s -- treating this as an environment "
            "hang, not a test failure (see the known deadlock-prone "
            "plotly export tests deselected in "
            "test_animation_export.py/test_round3.py)")
        return
    if result.returncode == _NO_BROWSER_EXIT:
        # The render subprocess reports "no usable browser here" with its own
        # reserved exit code, raised only for the three browser-lifecycle
        # exception types kaleido itself exports plus plotly's no-Chrome
        # RuntimeError. A Chrome that dies during startup is a fact about the
        # machine, not about hypertools -- but the discrimination is narrow on
        # purpose: EVERY other non-zero exit is still a hard failure below, so
        # a real rendering defect cannot hide behind this skip.
        pytest.skip(
            "kaleido could not drive a browser in this environment "
            f"(render script exit {_NO_BROWSER_EXIT}):\n{result.stderr[-600:]}")
        return
    assert result.returncode == 0, (
        f"render_multibyte_plotly.py failed (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")


@requires_covering_font
def test_plotly_legend_different_cjk_text_renders_different_pixels(tmp_path):
    from PIL import Image

    png_a = str(tmp_path / 'jp_a.png')
    png_b = str(tmp_path / 'jp_b.png')
    png_ascii = str(tmp_path / 'ascii.png')
    _render_plotly_png(JP_LABELS_A[:2], '', png_a)
    _render_plotly_png(JP_LABELS_B[:2], '', png_b)
    _render_plotly_png(['group a', 'group b'], '', png_ascii)

    buf_a = np.asarray(Image.open(png_a).convert('RGB'))
    buf_b = np.asarray(Image.open(png_b).convert('RGB'))
    buf_ascii = np.asarray(Image.open(png_ascii).convert('RGB'))

    assert buf_a.shape == buf_b.shape == buf_ascii.shape
    assert not np.array_equal(buf_a, buf_b), (
        "plotly renders with different Japanese legend text produced "
        "IDENTICAL pixels -- exactly what silent tofu (identical empty "
        "boxes for different strings) looks like")
    assert not np.array_equal(buf_a, buf_ascii), (
        "Japanese-legend render is pixel-identical to the ASCII-legend "
        "render -- the Japanese text isn't actually being drawn")
    assert not np.array_equal(buf_b, buf_ascii), (
        "Japanese-legend render is pixel-identical to the ASCII-legend "
        "render -- the Japanese text isn't actually being drawn")


@requires_covering_font
def test_plotly_point_labels_different_cjk_text_renders_different_pixels(
        tmp_path):
    # GH #205 F3: same anti-tofu proof as the legend check above, but for
    # `labels=` point annotations specifically -- two different single
    # Japanese point labels (one non-None entry each, out of the render
    # script's 15-point single dataset) must NOT render to identical
    # pixels, and must differ from an unlabeled (labels=None) render.
    from PIL import Image

    png_a = str(tmp_path / 'jp_label_a.png')
    png_b = str(tmp_path / 'jp_label_b.png')
    png_none = str(tmp_path / 'no_label.png')
    labels_a = [[JP_LABELS_A[0]] + [None] * 14]
    labels_b = [[JP_LABELS_B[0]] + [None] * 14]
    _render_plotly_png(['group a'], '', png_a, labels=labels_a)
    _render_plotly_png(['group a'], '', png_b, labels=labels_b)
    _render_plotly_png(['group a'], '', png_none, labels=None)

    buf_a = np.asarray(Image.open(png_a).convert('RGB'))
    buf_b = np.asarray(Image.open(png_b).convert('RGB'))
    buf_none = np.asarray(Image.open(png_none).convert('RGB'))

    assert buf_a.shape == buf_b.shape == buf_none.shape
    assert not np.array_equal(buf_a, buf_b), (
        "plotly renders with different Japanese point labels produced "
        "IDENTICAL pixels -- exactly what silent tofu (identical empty "
        "boxes for different strings) looks like")
    assert not np.array_equal(buf_a, buf_none), (
        "Japanese point-label render is pixel-identical to the unlabeled "
        "render -- the label isn't actually being drawn")
    assert not np.array_equal(buf_b, buf_none), (
        "Japanese point-label render is pixel-identical to the unlabeled "
        "render -- the label isn't actually being drawn")


# ------------------------------------------- bundled font + fallback stack
# hypertools vendors one small sans-serif face (Noto Sans, SIL OFL 1.1, under
# hypertools/external/fonts) so plots look the same on every platform instead
# of inheriting whatever the machine happens to have, and builds a per-glyph
# FALLBACK STACK around it so mixed-script text renders completely.

def test_bundled_font_file_is_shipped_and_registered():
    from hypertools.plot.fonts import bundled_font_files, installed_families
    files = bundled_font_files()
    assert files, "no bundled font file found"
    assert any(f.endswith('NotoSans-Regular.ttf') for f in files)
    for f in files:
        assert os.path.getsize(f) > 0
    # the OFL requires the license to travel with the font
    license_path = os.path.join(os.path.dirname(files[0]), 'OFL.txt')
    assert os.path.isfile(license_path)
    assert 'SIL Open Font License' in open(license_path, encoding='utf-8').read()
    assert 'Noto Sans' in installed_families()


def test_sans_serif_stack_is_installed_only_and_anchored():
    from hypertools.plot.fonts import installed_families, sans_serif_stack
    stack = sans_serif_stack()
    assert stack, "stack must never be empty"
    assert stack[0] == 'Noto Sans', "bundled face should lead"
    # DejaVu Sans (shipped with matplotlib) always anchors the end, so the
    # stack can never resolve to nothing
    assert stack[-1] == 'DejaVu Sans'
    # every entry must be installed -- naming a missing family makes
    # matplotlib log 'findfont: ... not found' for every text artist
    available = installed_families()
    assert all(fam in available for fam in stack)
    assert len(stack) == len(set(stack)), "no duplicate families"


def test_sans_serif_stack_puts_an_explicit_font_first():
    from hypertools.plot.fonts import sans_serif_stack
    stack = sans_serif_stack('DejaVu Sans')
    assert stack[0] == 'DejaVu Sans'
    assert len(stack) == len(set(stack))


def test_renderable_mixed_script_text_does_not_warn():
    # the covering-font warning used to fire whenever no SINGLE font covered
    # all the text -- which, with per-glyph fallback, is the common case for
    # ordinary accented/Greek text that renders perfectly. It must now warn
    # only about characters NOTHING in the stack can draw.
    from hypertools.plot.fonts import _codepoints_uncovered_by_stack
    text = 'café Ω≈π'
    codepoints = {ord(c) for c in text if ord(c) > 127}
    assert _codepoints_uncovered_by_stack(codepoints) == set()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        resolve_font(None, [text])
    assert not [x for x in w if 'tofu' in str(x.message)]


def test_truly_uncovered_codepoint_is_still_detected():
    # a private-use codepoint no real font maps -- the warning path must
    # still catch genuine gaps rather than being disabled outright
    from hypertools.plot.fonts import _codepoints_uncovered_by_stack
    assert _codepoints_uncovered_by_stack({0xE000}) == {0xE000}


# ------------------------------------- stable primary face (maintainer review)
# The bundled Noto Sans must stay the PRIMARY family for text the fallback stack
# already covers -- a stray accent, Greek letter, or math symbol must NOT swap
# the whole plot onto some other installed platform font (the pre-review bug:
# "Cafe" rendered in Noto Sans but "Café" switched everything to Hiragino).

@pytest.mark.parametrize('text', [
    'plain ascii',                       # ASCII
    'Café résumé naïve',                 # accented Latin
    'Ω α β γ Δ Σ π',                     # Greek
    '± × ÷ ≈ ≤ ≥ ∑ ∫ √ ∞',               # common math symbols
    'Å µm ° © ® ™ € £',                  # misc symbols
    'Cyrillic Привет',                   # Cyrillic
])
def test_covered_text_keeps_noto_sans_primary(text):
    # resolve_font returns None (no override) whenever the stack covers the text
    assert resolve_font(None, [text]) is None
    # ... so every drawn text surface keeps Noto Sans as its first family
    fig = hyp.plot(_random_points(3), '.', labels=['a', 'b', 'c'],
                   title=text, show=False)
    assert fig.axes[0].title.get_fontfamily()[0] == 'Noto Sans'
    for t in fig.axes[0].texts:
        assert t.get_fontfamily()[0] == 'Noto Sans'
    plt.close(fig)


def test_covered_text_renders_without_missing_glyph_warnings():
    # stable primary face must not come at the cost of tofu: accented/Greek/math
    # still render (from Noto, or DejaVu via per-glyph fallback)
    fig = hyp.plot(_random_points(3), '.', labels=['a', 'b', 'c'],
                   title='Café Ω ± ∑ Å µ °', show=False)
    assert _missing_glyph_warnings(fig) == []
    plt.close(fig)


def test_sans_serif_stack_first_leads_but_extra_is_only_a_fallback():
    # the core of the fix: an EXPLICIT font= (`first`) leads the stack, but an
    # AUTO-detected gap font (`extra`) is a fallback only -- Noto stays primary.
    # DejaVu Sans ships with matplotlib, so it is installed on EVERY platform
    # (unlike Arial/Helvetica, which are macOS/Windows-only and would be
    # filtered out of the stack on Linux CI).
    assert sans_serif_stack(first='DejaVu Sans')[0] == 'DejaVu Sans'
    assert sans_serif_stack(extra='DejaVu Sans')[0] == 'Noto Sans'


def test_auto_gap_font_is_appended_not_made_primary(monkeypatch):
    # simulate a real coverage gap: force resolve_font(None, ...) to return a
    # covering font (as it would for a script no stack family has) and verify
    # plot() adds it as a FALLBACK, leaving Noto Sans primary on the artists.
    import importlib
    from matplotlib.font_manager import FontProperties
    plotmod = importlib.import_module('hypertools.plot.plot')
    dejavu = FontProperties(family='DejaVu Sans')
    monkeypatch.setattr(plotmod, 'resolve_font', lambda font, texts: (
        dejavu if font is None else None))
    fig = hyp.plot(_random_points(3), '.', labels=['a', 'b', 'c'],
                   title='gap', show=False)
    # the artist keeps the in-context stack: Noto Sans PRIMARY, the gap filler
    # (DejaVu) present only as a later fallback (rcParams itself is restored on
    # rc_context exit, so it is not inspected here)
    family = fig.axes[0].title.get_fontfamily()
    assert family[0] == 'Noto Sans'
    assert 'DejaVu Sans' in family
    assert family.index('Noto Sans') < family.index('DejaVu Sans')
    plt.close(fig)


# ------------------------------------- cross-backend gap-font propagation
# When the curated stack cannot render a character but another INSTALLED font
# can, matplotlib appends that font to its rcParams stack. plotly has no such
# stack, so the discovered family must be handed to it explicitly and appended
# near the END of its CSS stack -- otherwise plotly silently renders tofu for a
# character matplotlib renders fine (maintainer font review).

def test_auto_gap_font_propagates_to_plotly_as_a_fallback(monkeypatch):
    import importlib
    from matplotlib.font_manager import FontProperties
    plotmod = importlib.import_module('hypertools.plot.plot')
    dejavu = FontProperties(family='DejaVu Sans')
    monkeypatch.setattr(plotmod, 'resolve_font',
                        lambda font, texts: dejavu if font is None else None)
    fig = hyp.plot(_random_points(4), '.', labels=['a', 'b', 'c', 'd'],
                   title='gap', backend='plotly', show=False)
    fam = fig.layout.font.family
    assert 'DejaVu Sans' in fam
    # appended AFTER the primary Latin faces (Noto leads) and BEFORE the
    # generic sans-serif tail, so it supplies only missing glyphs
    assert fam.index('Noto Sans') < fam.index('DejaVu Sans') < fam.rindex(
        'sans-serif')
    # every text surface (here the point annotations) inherits the same stack
    anns = fig.layout.scene.annotations
    assert anns
    assert all('DejaVu Sans' in a.font.family for a in anns)


def test_explicit_font_leads_plotly_stack_not_appended():
    # explicit font= is the caller's choice -> it must LEAD the plotly CSS
    # stack, unlike an auto gap font which is only a trailing fallback
    fig = hyp.plot(_random_points(4), '.', title='x', backend='plotly',
                   font='DejaVu Sans', show=False)
    assert fig.layout.font.family.startswith('"DejaVu Sans"')


def test_default_plotly_stack_unchanged_without_a_gap():
    fig = hyp.plot(_random_points(4), '.', title='plain', backend='plotly',
                   show=False)
    import importlib
    stack = importlib.import_module(
        'hypertools.plot.plotly_backend')._PLOTLY_SANS_STACK
    assert fig.layout.font.family == stack


# ------------------------------------- plotly CSS stack builder (helper unit)
# `_plotly_font_family` replaced a fragile string `.replace('sans-serif', ...)`
# and is the single source of truth for the default stack, the explicit-lead
# case, and the trailing gap-fallback case. It also de-duplicates a family that
# is already curated, so an explicit/extra face already in the stack stays tidy
# (maintainer font review, non-blocking cleanup).

def _plotly_font_helper():
    import importlib
    return importlib.import_module(
        'hypertools.plot.plotly_backend')._plotly_font_family


def test_plotly_font_family_default_equals_the_module_constant():
    import importlib
    pb = importlib.import_module('hypertools.plot.plotly_backend')
    assert pb._plotly_font_family() == pb._PLOTLY_SANS_STACK
    # the generic CSS family is always the bare, final token
    assert pb._PLOTLY_SANS_STACK.rsplit(',', 1)[-1].strip() == 'sans-serif'


def test_plotly_font_family_explicit_leads_and_is_quoted():
    fam = _plotly_font_helper()(explicit='Roboto')
    assert fam.startswith('"Roboto"')
    assert fam.index('Roboto') < fam.index('Noto Sans')


def test_plotly_font_family_extra_trails_before_generic_tail():
    fam = _plotly_font_helper()(extra='DejaVu Sans')
    assert fam.index('Noto Sans') < fam.index('DejaVu Sans') < fam.rindex(
        'sans-serif')


def test_plotly_font_family_dedups_family_already_in_stack():
    build = _plotly_font_helper()
    # an explicit face that is already curated is not repeated ...
    assert build(explicit='Arial').count('"Arial"') == 1
    # ... nor is a gap face that happens to collide with a curated one
    assert build(extra='Helvetica').count('"Helvetica"') == 1


def test_plotly_public_path_dedups_explicit_family_already_in_stack():
    # driving the real plot() path with font='Noto Sans' (already the primary
    # curated face) must not duplicate it in the layout stack
    fig = hyp.plot(_random_points(4), '.', title='x', backend='plotly',
                   font='Noto Sans', show=False)
    assert fig.layout.font.family.count('"Noto Sans"') == 1
    assert fig.layout.font.family.startswith('"Noto Sans"')


# --------------------------------------- the no-browser skip is discriminating
#
# The pixel checks above skip when the render subprocess reports that no
# browser could be driven. A skip that fires too eagerly is worse than no
# skip at all, so both halves of that discrimination are pinned here with
# REAL exceptions -- never a stub.
#
# Every test in this section runs WITHOUT a working browser, deliberately:
# the machines where the skip's accuracy actually matters are the ones that
# cannot start Chrome, and a check that needs Chrome in order to prove
# "Chrome is not what failed" is exactly backwards. That is why the
# "not the browser" half calls `is_browser_lifecycle_error` directly on a
# real exception object instead of driving the subprocess: with the
# subprocess, a bad output path is only reached AFTER the browser renders,
# so on a Chrome-less machine the run exits NO_BROWSER for the right reason
# and the check fails for the wrong one.

def _run_render(out_path, env=None):
    full = dict(os.environ)
    full.update(env or {})
    return subprocess.run(
        [sys.executable, _RENDER_PLOTLY_SCRIPT, json.dumps(['a', 'b']), '',
         out_path],
        timeout=_KALEIDO_TIMEOUT_S, capture_output=True, text=True, env=full)


@requires_covering_font
def test_render_script_exits_NO_BROWSER_when_the_browser_will_not_launch(
        tmp_path):
    """`/bin/echo` is a real executable that is not a browser, so kaleido
    genuinely raises `BrowserFailedError: the browser seemed to close
    immediately after starting` -- the same class of failure as a managed
    Chrome dying during startup. That must exit with the reserved code, not
    with a generic non-zero the caller would (correctly) treat as a defect."""
    result = _run_render(str(tmp_path / 'nope.png'),
                         {_BROWSER_PATH_ENV: '/bin/echo'})
    assert result.returncode == _NO_BROWSER_EXIT, (
        f'expected exit {_NO_BROWSER_EXIT}, got {result.returncode}\n'
        f'stderr:\n{result.stderr[-800:]}')
    assert 'NO_BROWSER:' in result.stderr


@requires_covering_font
def test_render_script_exits_NO_BROWSER_when_CHROME_CANNOT_BE_FOUND(tmp_path):
    """The other no-browser shape, and the one no `except` clause can catch
    by type: plotly swallows kaleido's `ChromeNotFoundError` and re-raises a
    PLAIN `RuntimeError` (`plotly/io/_kaleido.py:411`), so it is recognised
    by plotly's own message constant.

    Driven for real, and without launching anything: `BROWSER_PATH` is
    choreographer's documented override (`choreographer/utils/_which.py:91`)
    and pointing it at a path that is not a file makes `Chromium.__init__`
    raise `ChromeNotFoundError` (`chromium.py:177`). `HOME` moves with it
    because choreographer looks for its OWN downloaded Chrome first, before
    ever consulting `BROWSER_PATH` (`chromium.py:83`), and that download
    lives under the home directory.
    """
    result = _run_render(
        str(tmp_path / 'out.png'),
        {'BROWSER_PATH': str(tmp_path / 'not-a-browser'),
         'HOME': str(tmp_path / 'home')})
    assert result.returncode == _NO_BROWSER_EXIT, (
        f'expected exit {_NO_BROWSER_EXIT}, got {result.returncode}\n'
        f'stderr:\n{result.stderr[-800:]}')
    assert 'NO_BROWSER: RuntimeError' in result.stderr


def test_a_real_OSError_from_the_render_is_NOT_the_browser(tmp_path):
    """The half that matters: a failure that is not the browser's must never
    be laundered into the skip. Without this, widening the predicate by one
    exception type would silently turn every pixel check into a skip.

    The `OSError` is a real one, caught from a real failed write into a
    directory that does not exist -- the same call the render performs, and
    the same exception it would raise.
    """
    try:
        with open(tmp_path / 'no_such_dir' / 'x.png', 'wb'):
            pass
    except OSError as err:
        caught = err
    else:
        raise AssertionError('writing into a missing directory must fail')
    assert not _is_browser_lifecycle_error(caught), (
        f'{caught!r} is a hypertools/kaleido failure, not a missing browser')


def test_a_real_RuntimeError_that_is_NOT_plotly_s_is_NOT_the_browser():
    """`ChromeNotFoundError` reaches us as a bare `RuntimeError`, so the
    predicate has to inspect a message -- which makes "some OTHER
    `RuntimeError`" the exact failure mode to pin. This one is real (the
    interpreter's own, from a generator that lets `StopIteration` escape),
    not a constructed stand-in."""
    def leaks_stop_iteration():
        yield next(iter([]))

    try:
        next(leaks_stop_iteration())
    except RuntimeError as err:
        caught = err
    else:
        raise AssertionError('a generator raising StopIteration must become '
                             'a RuntimeError (PEP 479)')
    assert not _is_browser_lifecycle_error(caught), (
        f'{caught!r} is unrelated to the browser: {caught}')
