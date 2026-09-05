#!/usr/bin/env python
"""`palette=` as a {category: color} dict (GH #285).

The point of the dict form is that the caller stops having to compute the
category order by hand so a LIST palette lines up with the right category
(`examples/animate_conversation.py` did exactly that). These tests pin the
mapping itself -- explicit keys, the documented default-palette fallback for
categories the dict does not name, and the errors -- and check that a list /
name / Colormap palette still resolves to exactly what it did before.
"""

import numpy as np
import pytest
from matplotlib.colors import to_rgb

from hypertools.plot.colors import (DEFAULT_PALETTE, get_palette_colors,
                                    mat2colors, resolve_category_colors)


def test_dict_maps_each_category_explicitly():
    mapping = resolve_category_colors(
        {'Alice': '#E4572E', 'Bob': 'C0', 'Carol': (0.0, 0.5, 0.0)},
        ['Alice', 'Bob', 'Carol'])
    assert mapping == {'Alice': to_rgb('#E4572E'), 'Bob': to_rgb('C0'),
                       'Carol': (0.0, 0.5, 0.0)}


def test_mapping_is_ordered_by_category_order_not_dict_order():
    # the dict is written in one order, the categories drawn in another;
    # the returned mapping follows the CATEGORIES (this is what makes it
    # safe to zip against per-category artists)
    mapping = resolve_category_colors({'Bob': 'blue', 'Alice': 'red'},
                                      ['Alice', 'Bob'])
    assert list(mapping) == ['Alice', 'Bob']
    assert mapping['Alice'] == to_rgb('red')


def test_missing_categories_take_their_own_default_palette_slot():
    cats = ['Alice', 'Bob', 'Carol', 'Dave']
    default = get_palette_colors(DEFAULT_PALETTE, len(cats))
    mapping = resolve_category_colors({'Bob': 'red'}, cats)
    assert mapping['Bob'] == to_rgb('red')
    # every other category keeps EXACTLY the color it would have had with
    # no dict at all -- naming a subset shifts nothing else
    for i, cat in enumerate(cats):
        if cat != 'Bob':
            assert np.allclose(mapping[cat], default[i]), cat


def test_empty_dict_is_the_default_palette():
    cats = ['a', 'b', 'c']
    assert np.allclose(list(resolve_category_colors({}, cats).values()),
                       get_palette_colors(DEFAULT_PALETTE, len(cats)))


def test_default_palette_argument_is_honored():
    mapping = resolve_category_colors({'a': 'red'}, ['a', 'b'],
                                      default_palette=['red', 'lime'])
    assert mapping == {'a': (1.0, 0.0, 0.0), 'b': (0.0, 1.0, 0.0)}


def test_unknown_key_raises_and_lists_the_categories_seen():
    with pytest.raises(ValueError) as excinfo:
        resolve_category_colors({'Alice': 'red', 'Alicia': 'blue'},
                                ['Alice', 'Bob'])
    message = str(excinfo.value)
    assert 'Alicia' in message
    assert "'Alice', 'Bob'" in message      # the categories actually seen
    assert 'Alice' in message


def test_several_unknown_keys_are_all_reported():
    with pytest.raises(ValueError) as excinfo:
        resolve_category_colors({'x': 'red', 'y': 'blue'}, ['a'])
    message = str(excinfo.value)
    assert 'categories' in message
    assert "'x'" in message and "'y'" in message


def test_non_color_value_names_the_offending_key():
    with pytest.raises(ValueError, match=r"palette\['a'\]"):
        resolve_category_colors({'a': 'not-a-color'}, ['a'])


def test_integer_and_boolean_categories():
    mapping = resolve_category_colors({1: 'red', 2: 'blue'}, [1, 2])
    assert mapping == {1: (1.0, 0.0, 0.0), 2: (0.0, 0.0, 1.0)}
    bools = resolve_category_colors({True: 'red', False: 'blue'},
                                    [True, False])
    assert bools[True] == (1.0, 0.0, 0.0)
    assert bools[False] == (0.0, 0.0, 1.0)


def test_tuple_categories():
    cats = [('a', 1), ('b', 2)]
    mapping = resolve_category_colors({('a', 1): 'red'}, cats)
    assert mapping[('a', 1)] == (1.0, 0.0, 0.0)
    assert np.allclose(mapping[('b', 2)],
                       get_palette_colors(DEFAULT_PALETTE, 2)[1])


def test_integer_keys_reach_stringified_categories():
    # plot() stringifies integer hue categories for its legend; a dict
    # keyed by the ORIGINAL integers still has to find them
    mapping = resolve_category_colors({0: 'red', 1: 'blue'}, ['0', '1'])
    assert mapping == {'0': (1.0, 0.0, 0.0), '1': (0.0, 0.0, 1.0)}


def test_string_keys_reach_integer_categories():
    mapping = resolve_category_colors({'0': 'red'}, [0, 1])
    assert mapping[0] == (1.0, 0.0, 0.0)


def test_exact_key_wins_over_string_match():
    mapping = resolve_category_colors({0: 'red', '0': 'blue'}, [0, '0'])
    assert mapping[0] == (1.0, 0.0, 0.0)
    assert mapping['0'] == (0.0, 0.0, 1.0)


def test_duplicate_categories_collapse_in_first_appearance_order():
    mapping = resolve_category_colors({'b': 'red'}, ['b', 'a', 'b', 'a'])
    assert list(mapping) == ['b', 'a']


def test_empty_categories_is_empty_mapping():
    assert resolve_category_colors({'anything': 'red'}, []) == {}
    assert resolve_category_colors('hls', []) == {}


def test_list_name_and_colormap_palettes_are_unchanged():
    import matplotlib

    cats = ['a', 'b', 'c']
    for palette in ('hls', 'Set2', ['red', 'lime', 'blue'],
                    matplotlib.colormaps['viridis']):
        expected = get_palette_colors(palette, len(cats))
        got = resolve_category_colors(palette, cats)
        assert np.allclose(list(got.values()), expected), palette


def test_short_color_list_still_raises_for_categories():
    with pytest.raises(ValueError, match='2 color'):
        resolve_category_colors(['red', 'blue'], ['a', 'b', 'c'])


def test_mat2colors_accepts_a_dict_palette():
    colors = mat2colors(['a', 'b', 'a', 'c'],
                        palette={'a': 'red', 'b': 'blue'})
    assert np.allclose(colors[0], (1.0, 0.0, 0.0))
    assert np.allclose(colors[2], (1.0, 0.0, 0.0))
    assert np.allclose(colors[1], (0.0, 0.0, 1.0))
    # 'c' was not named: it keeps its default-palette slot (index 2 of the
    # three categories in first-appearance order)
    assert np.allclose(colors[3], get_palette_colors(DEFAULT_PALETTE, 3)[2])


def test_mat2colors_categorical_is_bitwise_unchanged_without_a_dict():
    labels = ['x', 'y', 'z', 'x']
    base = get_palette_colors('hls', 3)
    expected = np.asarray([base[0], base[1], base[2], base[0]])
    assert np.allclose(mat2colors(labels, palette='hls'), expected)


def test_dict_palette_rejected_on_a_continuous_hue():
    with pytest.raises(ValueError, match='CATEGORICAL'):
        mat2colors([0.0, 1.0, 2.0], palette={'a': 'red'})


def test_dict_palette_rejected_on_a_matrix_hue():
    with pytest.raises(ValueError, match='CATEGORICAL'):
        mat2colors(np.array([[0.5, 0.5], [1.0, 0.0]]),
                   palette={'a': 'red', 'b': 'blue'})


def test_dict_palette_rejected_by_get_palette_colors():
    with pytest.raises(ValueError, match='CATEGORICAL'):
        get_palette_colors({'a': 'red'}, 2)
