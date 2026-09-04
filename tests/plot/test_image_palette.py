"""Palette extraction from an image, and the `palette='image:<path>'` spelling.

The ordering rule is the point: `examples/animate_painting_embeddings.py:138-140`
picked `km.cluster_centers_[np.argmax(counts)]` -- the LARGEST cluster -- which
in a painting is the background. Measured on the synthetic image below, that
rule returns the beige (0.784, 0.769, 0.737); this module pins the vivid red
(0.863, 0.078, 0.078) as the FIRST colour instead.

No network: every image is written to `tmp_path` and read back.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from PIL import Image
from matplotlib.colors import to_rgb

import hypertools as hyp
from hypertools.plot.colors import (IMAGE_PALETTE_N, continuous_colormap,
                                    get_palette_colors, image_palette)

BEIGE = (0.784, 0.769, 0.737)
VIVID = (0.863, 0.078, 0.078)


def _png(tmp_path, arr, name):
    path = tmp_path / name
    Image.fromarray(arr.astype(np.uint8)).save(path)
    return str(path)


def painting_png(tmp_path, name='painting.png'):
    """90% muted beige 'canvas', 10% vivid red 'subject'."""
    arr = np.zeros((100, 100, 3), np.uint8)
    arr[:, :] = (200, 196, 188)
    arr[:10, :] = (220, 20, 20)
    return _png(tmp_path, arr, name)


def grey_png(tmp_path, name='grey.png'):
    arr = np.zeros((100, 100, 3), np.uint8)
    arr[:, :] = (30, 30, 30)
    arr[:20, :] = (200, 200, 200)
    return _png(tmp_path, arr, name)


def six_png(tmp_path, name='six.png'):
    arr = np.zeros((120, 120, 3), np.uint8)
    for i, c in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255),
                           (255, 255, 0), (255, 0, 255), (0, 255, 255)]):
        arr[i * 20:(i + 1) * 20, :] = c
    return _png(tmp_path, arr, name)


def nine_png(tmp_path, name='nine.png'):
    """NINE genuinely distinct bands, so 'not capped at six' tests the cap
    rather than the interpolation fallback."""
    arr = np.zeros((180, 180, 3), np.uint8)
    for i, c in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255),
                           (255, 255, 0), (255, 0, 255), (0, 255, 255),
                           (255, 128, 0), (128, 0, 255), (0, 128, 128)]):
        arr[i * 20:(i + 1) * 20, :] = c
    return _png(tmp_path, arr, name)


def one_colour_png(tmp_path, name='one.png'):
    arr = np.full((60, 60, 3), 200, np.uint8)
    return _png(tmp_path, arr, name)


def _ax(fig):
    return [a for a in fig.axes if hasattr(a, 'zaxis')][0]


# --- the extraction itself ---------------------------------------------------

def test_returns_rgb_floats_in_the_unit_range(tmp_path):
    pal = image_palette(painting_png(tmp_path))
    assert pal.ndim == 2 and pal.shape[1] == 3
    assert pal.dtype == np.float64
    assert pal.min() >= 0.0 and pal.max() <= 1.0


def test_a_vivid_minority_colour_beats_the_muted_background(tmp_path):
    """THE regression test. Largest-cluster ordering returns the beige."""
    pal = image_palette(painting_png(tmp_path))
    assert pal[0] == pytest.approx(VIVID, abs=0.02)


def test_the_background_is_kept_but_demoted(tmp_path):
    """Not discarded -- just not first. A palette should still describe the
    whole canvas."""
    pal = image_palette(painting_png(tmp_path))
    assert any(np.allclose(c, BEIGE, atol=0.02) for c in pal)
    assert not np.allclose(pal[0], BEIGE, atol=0.02)


def test_a_greyscale_image_falls_back_to_population_order(tmp_path):
    """With no chroma anywhere, `frac * chroma` is all zeros and 'largest'
    IS the right answer: the 80% dark tone leads."""
    pal = image_palette(grey_png(tmp_path))
    assert pal[0] == pytest.approx((0.118, 0.118, 0.118), abs=0.02)


def test_n_colors_is_an_upper_bound_and_colours_are_distinct(tmp_path):
    pal = image_palette(six_png(tmp_path), n_colors=6)
    assert len(pal) == 6
    assert len(np.unique(np.round(pal, 3), axis=0)) == 6
    assert len(image_palette(six_png(tmp_path), n_colors=3)) == 3


def test_an_image_with_fewer_unique_colours_returns_fewer(tmp_path):
    """Two unique pixel colours cannot yield six clusters; asking for six
    must NOT raise or emit sklearn's ConvergenceWarning."""
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        pal = image_palette(painting_png(tmp_path), n_colors=6)
    assert len(pal) == 2
    assert not [w for w in caught if 'ConvergenceWarning' in type(w.message).__name__]


def test_accepts_a_pil_image_and_a_numpy_array(tmp_path):
    arr = np.zeros((100, 100, 3), np.uint8)
    arr[:, :] = (200, 196, 188)
    arr[:10, :] = (220, 20, 20)
    from_path = image_palette(painting_png(tmp_path))
    assert image_palette(arr)[0] == pytest.approx(from_path[0], abs=0.02)
    assert image_palette(Image.fromarray(arr))[0] == pytest.approx(
        from_path[0], abs=0.02)


def test_a_float_array_in_unit_range_is_accepted(tmp_path):
    arr = np.zeros((100, 100, 3), float)
    arr[:, :] = (200 / 255, 196 / 255, 188 / 255)
    arr[:10, :] = (220 / 255, 20 / 255, 20 / 255)
    assert image_palette(arr)[0] == pytest.approx(VIVID, abs=0.02)


def test_extraction_is_deterministic(tmp_path):
    path = painting_png(tmp_path)
    assert np.allclose(image_palette(path), image_palette(path))


def test_a_missing_file_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match='no_such_canvas.jpg'):
        image_palette(str(tmp_path / 'no_such_canvas.jpg'))


def test_n_colors_must_be_a_positive_integer(tmp_path):
    with pytest.raises(ValueError, match='positive integer'):
        image_palette(painting_png(tmp_path), n_colors=0)


# --- the `palette='image:<path>'` spelling ------------------------------------

def test_palette_string_resolves_through_get_palette_colors(tmp_path):
    """One interception in _get_palette must serve every palette consumer."""
    path = painting_png(tmp_path)
    resolved = get_palette_colors(f'image:{path}', 2)
    assert resolved[0] == pytest.approx(VIVID, abs=0.02)


def test_palette_string_colours_a_categorical_hue(tmp_path):
    """Reads ax.LINES, not ax.collections. A `fmt='.'` plot draws `Line2D`
    into `ax.lines`; the only collections on a 3-D axes are pane/grid
    artists whose facecolor array is EMPTY. v2 harvested those instead, so
    the filter emptied the list and `np.vstack([])` raised `need at least
    one array to concatenate` -- against a CORRECT implementation. Measured
    while auditing: the line colors were exactly right the whole time."""
    path = painting_png(tmp_path)
    rng = np.random.default_rng(0)
    ds = [rng.normal(size=(10, 4)) for _ in range(2)]
    fig = hyp.plot(ds, '.', hue=['a'] * 10 + ['b'] * 10,
                   palette=f'image:{path}', show=False)
    drawn = [to_rgb(ln.get_color()) for ln in _ax(fig).lines]
    assert drawn, 'no line artists were drawn'
    assert any(np.allclose(c, VIVID, atol=0.02) for c in drawn)


def test_a_categorical_hue_is_not_capped_at_six_categories(tmp_path):
    """`IMAGE_PALETTE_N = 6` is the CONTINUOUS anchor count, not a limit on
    categories. With a fixed count this raised `palette= supplies 6 color(s)
    but 9 are required`. Uses a NINE-colour image, so this tests the cap and
    not the interpolation fallback."""
    path = nine_png(tmp_path)
    rng = np.random.default_rng(0)
    labels = [c for c in 'abcdefghi' for _ in range(4)]
    fig = hyp.plot([rng.normal(size=(36, 4))], '.', hue=labels,
                   palette=f'image:{path}', show=False)
    drawn = {to_rgb(ln.get_color()) for ln in _ax(fig).lines}
    assert len(drawn) == 9, f'expected 9 distinct colours, got {len(drawn)}'


def test_an_image_with_too_few_colours_interpolates_rather_than_repeats(
        tmp_path):
    """Cycling would give two categories the SAME colour -- the ambiguity
    the short-list error exists to prevent (the `raise ValueError` arm of
    `_get_palette`'s `len(colors) < n_colors` branch). A caller
    cannot add colours to an image, so the anchors are blended up instead.
    `painting_png` is genuinely two-tone, so 5 categories need 3 blended."""
    path = painting_png(tmp_path)
    rng = np.random.default_rng(0)
    labels = [c for c in 'abcde' for _ in range(4)]
    fig = hyp.plot([rng.normal(size=(20, 4))], '.', hue=labels,
                   palette=f'image:{path}', show=False)
    drawn = [to_rgb(ln.get_color()) for ln in _ax(fig).lines]
    assert len({tuple(np.round(c, 6)) for c in drawn}) == 5, (
        'repeated colours would make two categories indistinguishable')
    assert np.allclose(drawn[0], VIVID, atol=0.02), (
        'the most salient anchor must survive interpolation, and lead')


def test_a_single_colour_image_raises_rather_than_inventing_colours(
        tmp_path):
    """The one case interpolation cannot honestly serve."""
    path = one_colour_png(tmp_path)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match='single dominant color'):
        hyp.plot([rng.normal(size=(20, 4))], '.',
                 hue=[c for c in 'abcde' for _ in range(4)],
                 palette=f'image:{path}', show=False)


def test_palette_string_blends_anchors_for_a_continuous_hue(tmp_path):
    """A short list + a continuous hue is seaborn blend_palette semantics
    (the `continuous` arm of `_get_palette`'s `len(colors) < n_colors` branch), so an image palette gives a gradient between its
    extracted anchors -- no error about 'too few colors'."""
    path = six_png(tmp_path)
    cmap = continuous_colormap(f'image:{path}', n_bins=100)
    assert cmap.N == 100
    assert len(np.unique(np.round(cmap(np.linspace(0, 1, 100))[:, :3], 3),
                         axis=0)) > IMAGE_PALETTE_N


def test_palette_string_with_a_missing_file_names_the_file(tmp_path):
    rng = np.random.default_rng(0)
    ds = [rng.normal(size=(10, 4))]
    with pytest.raises(FileNotFoundError, match='gone.png'):
        hyp.plot(ds, '.', hue=np.arange(10),
                 palette=f"image:{tmp_path / 'gone.png'}", show=False)


def test_plotly_backend_accepts_an_image_palette(tmp_path):
    """Backend parity: the interception is in colors.py, above both backends."""
    pytest.importorskip('plotly')
    path = painting_png(tmp_path)
    rng = np.random.default_rng(0)
    ds = [rng.normal(size=(10, 4)) for _ in range(2)]
    hyp.set_interactive_backend('plotly')
    try:
        fig = hyp.plot(ds, '.', hue=['a'] * 10 + ['b'] * 10,
                       palette=f'image:{path}', show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    assert len(fig.data) >= 2
