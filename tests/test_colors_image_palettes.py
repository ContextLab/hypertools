#!/usr/bin/env python
"""Luminance bounds on `image_palette`, and per-dataset `palette=` lists
(GH #285).

Every image here is a REAL PNG written with PIL into tmp_path and read back
through the same `_image_pixels` -> k-means path a user's painting goes
through; nothing is mocked. The two-tone fixtures are built so k-means'
answer is exact (k = the number of distinct colors), which is what lets the
assertions name colors rather than tolerances.
"""

import numpy as np
import pytest
from matplotlib.colors import to_rgb

from hypertools.plot.colors import (DEFAULT_PALETTE, IMAGE_PALETTE_N,
                                    LUMINANCE_WEIGHTS, dataset_colors,
                                    dataset_palettes, get_palette_colors,
                                    image_palette, luminance, mat2colors,
                                    palette_lead_color)

# the painting example's own selection, reproduced verbatim so a later
# conversion of examples/animate_painting_embeddings.py can be checked
# against it (examples/animate_painting_embeddings.py:126-127, 202)
LUMA = np.array([0.2126, 0.7152, 0.0722])
MAX_LUMINANCE = 0.6

PALE_YELLOW = (250, 250, 180)     # large, chromatic, and far too bright
DEEP_BLUE = (20, 30, 120)         # smaller, less salient, legible
MUTED_GREY = (200, 200, 205)      # large and almost achromatic
VIVID_RED = (220, 20, 30)         # small but vivid


def write_png(tmp_path, name, blocks):
    """A PNG made of horizontal bands: `blocks` is [(rgb, n_rows), ...]."""
    from PIL import Image

    rows = []
    for rgb, n_rows in blocks:
        rows.append(np.tile(np.asarray(rgb, dtype=np.uint8), (n_rows, 40, 1)))
    arr = np.concatenate(rows, axis=0)
    path = tmp_path / name
    Image.fromarray(arr).save(path)
    return str(path)


@pytest.fixture
def bright_and_dark(tmp_path):
    """A "Great Wave" in miniature: the salient color is unusably bright."""
    return write_png(tmp_path, 'wave.png',
                     [(PALE_YELLOW, 64), (DEEP_BLUE, 16)])


@pytest.fixture
def muted_and_vivid(tmp_path):
    return write_png(tmp_path, 'canvas.png',
                     [(MUTED_GREY, 70), (VIVID_RED, 10)])


# --------------------------------------------------------------- luminance
def test_luminance_formula_matches_the_documented_weights():
    assert luminance('white') == pytest.approx(1.0)
    assert luminance('black') == pytest.approx(0.0)
    assert np.allclose(LUMINANCE_WEIGHTS, LUMA)
    for spec in ('#E4572E', 'C0', (0.2, 0.4, 0.6), (0.2, 0.4, 0.6, 0.5)):
        assert luminance(spec) == pytest.approx(
            float(np.asarray(to_rgb(spec)) @ LUMA))


def test_luminance_of_a_sequence_is_one_value_per_color():
    values = luminance([(1.0, 1.0, 1.0), 'black', '#E4572E'])
    assert values.shape == (3,)
    assert values[0] == pytest.approx(1.0)
    assert values[1] == pytest.approx(0.0)


def test_luminance_rejects_a_non_color():
    with pytest.raises(ValueError, match='luminance'):
        luminance(['not-a-color'])


# ---------------------------------------------------- image_palette bounds
def test_salience_ordering_is_unchanged(muted_and_vivid):
    # the pre-existing contract: a small vivid region beats a large muted
    # one (ordering by cluster SIZE would return the background)
    palette = image_palette(muted_and_vivid)
    assert len(palette) == 2
    assert np.allclose(palette[0], np.asarray(VIVID_RED) / 255, atol=0.01)


def test_max_luminance_steps_past_the_too_bright_salient_color(
        bright_and_dark):
    unfiltered = image_palette(bright_and_dark)
    assert luminance(unfiltered[0]) > MAX_LUMINANCE      # the pale sky leads
    filtered = image_palette(bright_and_dark, max_luminance=MAX_LUMINANCE)
    assert np.allclose(filtered[0], np.asarray(DEEP_BLUE) / 255, atol=0.01)


def test_max_luminance_reproduces_the_painting_examples_hand_selection(
        bright_and_dark):
    # examples/animate_painting_embeddings.py:200-202, verbatim
    full = image_palette(bright_and_dark)
    by_hand = next(tuple(c) for c in full
                   if float(c @ LUMA) <= MAX_LUMINANCE)
    by_kwarg = tuple(image_palette(bright_and_dark,
                                   max_luminance=MAX_LUMINANCE)[0])
    # the SAME entry of the same salience ordering: the kwarg steps past
    # the brighter, more salient cluster exactly as the hand-written
    # `next(...)` did (allclose, not ==, only because two k-means fits of
    # one image agree to ~1e-15 rather than bit for bit)
    assert next(i for i, c in enumerate(full)
                if float(c @ LUMA) <= MAX_LUMINANCE) == 1
    assert np.allclose(by_kwarg, by_hand, atol=1e-9)


def test_filtering_happens_after_ordering_not_before(bright_and_dark):
    # the filter must SUBSET the salience-ordered list, never re-cluster or
    # re-rank it: every kept color keeps its relative order
    full = image_palette(bright_and_dark, n_colors=IMAGE_PALETTE_N)
    kept = image_palette(bright_and_dark, n_colors=IMAGE_PALETTE_N,
                         max_luminance=MAX_LUMINANCE)
    expected = np.asarray([c for c in full if luminance(c) <= MAX_LUMINANCE])
    assert len(kept) == len(expected)
    # allclose, not ==: two k-means fits of the same image agree to ~1e-15,
    # not bit for bit (threaded reductions sum in whatever order they win)
    assert np.allclose(kept, expected)


def test_min_luminance_drops_the_dark_colors(bright_and_dark):
    kept = image_palette(bright_and_dark, min_luminance=0.5)
    assert len(kept) == 1
    assert np.allclose(kept[0], np.asarray(PALE_YELLOW) / 255, atol=0.01)


def test_both_bounds_together(bright_and_dark):
    kept = image_palette(bright_and_dark, min_luminance=0.05,
                         max_luminance=0.6)
    assert np.allclose(kept[0], np.asarray(DEEP_BLUE) / 255, atol=0.01)


def test_bounds_excluding_every_color_raise_and_report_the_luminances(
        bright_and_dark):
    with pytest.raises(ValueError) as excinfo:
        image_palette(bright_and_dark, max_luminance=0.01)
    message = str(excinfo.value)
    assert 'luminance' in message
    assert 'measured' in message
    # the measured values are in the message, so the caller can see how far
    # off the bound was
    assert '0.1' in message or '0.13' in message


def test_no_bounds_returns_every_color(bright_and_dark):
    assert len(image_palette(bright_and_dark)) == 2


@pytest.mark.parametrize('kwargs', [
    {'max_luminance': 'bright'},
    {'min_luminance': -0.1},
    {'max_luminance': 1.5},
    {'max_luminance': True},
])
def test_invalid_luminance_bounds_raise(bright_and_dark, kwargs):
    with pytest.raises(ValueError, match='luminance'):
        image_palette(bright_and_dark, **kwargs)


def test_min_above_max_raises(bright_and_dark):
    with pytest.raises(ValueError, match='greater than'):
        image_palette(bright_and_dark, min_luminance=0.8, max_luminance=0.2)


# ------------------------------------------ 'image:<path>?<options>' specs
def test_image_spec_carries_the_luminance_bound(bright_and_dark):
    spec = f'image:{bright_and_dark}?max_luminance=0.6'
    colors = get_palette_colors(spec, 1)
    assert np.allclose(colors[0], np.asarray(DEEP_BLUE) / 255, atol=0.01)


def test_image_spec_options_reach_mat2colors(bright_and_dark):
    colors = mat2colors(['a'], palette=f'image:{bright_and_dark}'
                                      '?max_luminance=0.6')
    assert np.allclose(colors[0], np.asarray(DEEP_BLUE) / 255, atol=0.01)


def test_image_spec_n_colors_option(bright_and_dark):
    spec = f'image:{bright_and_dark}?n_colors=1'
    # n_colors=1 -> k-means with k=1 -> the image's average color, which is
    # neither of the two bands
    average = get_palette_colors(spec, 1)[0]
    assert not np.allclose(average, np.asarray(DEEP_BLUE) / 255, atol=0.05)
    assert not np.allclose(average, np.asarray(PALE_YELLOW) / 255, atol=0.05)


def test_a_real_path_containing_a_question_mark_wins(tmp_path):
    try:
        path = write_png(tmp_path, 'wave?2.png',
                         [(PALE_YELLOW, 64), (DEEP_BLUE, 16)])
    except OSError:
        pytest.skip("this filesystem cannot hold '?' in a filename")
    # the file EXISTS, so the text is a path, not a path plus options
    assert len(image_palette(path)) == 2
    assert len(get_palette_colors(f'image:{path}', 2)) == 2


def test_unknown_option_key_is_treated_as_part_of_the_path(tmp_path):
    missing = str(tmp_path / 'nope.png?sharpness=3')
    with pytest.raises(FileNotFoundError) as excinfo:
        image_palette(missing)
    # the error names what the user actually typed
    assert 'sharpness=3' in str(excinfo.value)


def test_unreadable_option_value_says_which_option(bright_and_dark):
    with pytest.raises(ValueError, match='max_luminance'):
        get_palette_colors(f'image:{bright_and_dark}?max_luminance=dim', 2)


# ------------------------------------------------- per-dataset palette list
def test_a_list_of_colors_is_still_a_list_of_colors():
    # the historical meaning, unchanged: every entry is a color
    assert dataset_palettes(['red', '#00ff00', (0, 0, 1)], 3) is None
    assert dataset_palettes([(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)], 2) is None
    assert dataset_palettes(np.array([[1.0, 0, 0], [0, 0, 1.0]]), 2) is None


def test_non_list_palettes_are_never_per_dataset():
    import matplotlib

    for palette in ('hls', 'image:whatever.png', None, {'a': 'red'},
                    matplotlib.colormaps['viridis']):
        assert dataset_palettes(palette, 3) is None


def test_a_list_with_an_image_entry_is_per_dataset(bright_and_dark,
                                                   muted_and_vivid):
    specs = dataset_palettes([f'image:{bright_and_dark}',
                              f'image:{muted_and_vivid}'], 2)
    assert specs == [f'image:{bright_and_dark}', f'image:{muted_and_vivid}']


def test_a_list_of_palette_names_is_per_dataset():
    assert dataset_palettes(['viridis', 'magma'], 2) == ['viridis', 'magma']


def test_a_mixed_list_is_per_dataset_and_a_color_entry_is_a_one_color_palette(
        bright_and_dark):
    specs = dataset_palettes([f'image:{bright_and_dark}', 'red'], 2)
    assert specs[1] == 'red'
    assert palette_lead_color(specs[1]) == (1.0, 0.0, 0.0)


def test_nested_color_lists_are_per_dataset():
    specs = dataset_palettes([['red', 'lime'], ['blue', 'black']], 2)
    assert specs == [['red', 'lime'], ['blue', 'black']]
    assert palette_lead_color(specs[0]) == (1.0, 0.0, 0.0)


def test_a_single_entry_broadcasts_to_every_dataset(bright_and_dark):
    specs = dataset_palettes([f'image:{bright_and_dark}'], 3)
    assert specs == [f'image:{bright_and_dark}'] * 3


def test_wrong_length_raises_and_names_both_readings(bright_and_dark):
    with pytest.raises(ValueError) as excinfo:
        dataset_palettes([f'image:{bright_and_dark}', 'viridis', 'magma'], 2)
    message = str(excinfo.value)
    assert '3 per-dataset palettes' in message
    assert '2 dataset' in message


def test_an_entry_that_is_neither_a_color_nor_a_palette_raises():
    with pytest.raises(ValueError) as excinfo:
        dataset_palettes(['viridis', 'not-a-thing-at-all'], 2)
    assert 'not-a-thing-at-all' in str(excinfo.value)


def test_per_dataset_list_reaching_a_single_palette_path_says_so(
        bright_and_dark):
    with pytest.raises(ValueError, match='PER-DATASET'):
        mat2colors(['a', 'b'],
                   palette=[f'image:{bright_and_dark}', 'viridis'])


# ------------------------------------------------------- lead / per-dataset
def test_palette_lead_color_of_an_image_is_the_salient_color_not_the_mean(
        muted_and_vivid):
    lead = palette_lead_color(f'image:{muted_and_vivid}')
    assert np.allclose(lead, np.asarray(VIVID_RED) / 255, atol=0.01)
    # ... and NOT what asking the palette machinery for one color gives,
    # which is k-means with k=1: the image's average
    assert not np.allclose(lead, get_palette_colors(
        f'image:{muted_and_vivid}', 1)[0], atol=0.05)


def test_palette_lead_color_honors_a_luminance_bound(bright_and_dark):
    lead = palette_lead_color(f'image:{bright_and_dark}?max_luminance=0.6')
    assert np.allclose(lead, np.asarray(DEEP_BLUE) / 255, atol=0.01)


def test_palette_lead_color_of_names_lists_and_colormaps():
    import matplotlib

    assert palette_lead_color('red') == (1.0, 0.0, 0.0)
    assert palette_lead_color(['lime', 'red']) == (0.0, 1.0, 0.0)
    assert np.allclose(palette_lead_color('hls'),
                       get_palette_colors('hls', 1)[0])
    assert np.allclose(palette_lead_color(matplotlib.colormaps['viridis']),
                       get_palette_colors(matplotlib.colormaps['viridis'],
                                          1)[0])


def test_dataset_colors_gives_each_dataset_its_own_image(bright_and_dark,
                                                         muted_and_vivid):
    colors = dataset_colors([f'image:{bright_and_dark}?max_luminance=0.6',
                             f'image:{muted_and_vivid}'], 2)
    assert colors.shape == (2, 3)
    assert np.allclose(colors[0], np.asarray(DEEP_BLUE) / 255, atol=0.01)
    assert np.allclose(colors[1], np.asarray(VIVID_RED) / 255, atol=0.01)


def test_dataset_colors_falls_back_to_todays_colors():
    for palette in ('hls', 'Set2', DEFAULT_PALETTE):
        assert np.allclose(dataset_colors(palette, 4),
                           get_palette_colors(palette, 4)), palette
    assert np.allclose(dataset_colors(['red', 'lime', 'blue'], 3),
                       [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
