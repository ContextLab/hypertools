# -*- coding: utf-8 -*-

import numpy as np
import pytest

from hypertools.plot.colors import mat2colors, colors2groups


def test_mat2colors_categorical():
    colors = mat2colors(['a', 'b', 'a', 'c'])
    assert colors.shape == (4, 3)
    # same category -> same color; different categories -> different colors
    assert np.array_equal(colors[0], colors[2])
    assert not np.array_equal(colors[0], colors[1])
    assert not np.array_equal(colors[1], colors[3])


def test_mat2colors_continuous():
    colors = mat2colors(np.linspace(0, 1, 50))
    assert colors.shape == (50, 3)
    assert colors.min() >= 0 and colors.max() <= 1
    # endpoints of a continuous ramp should differ
    assert not np.array_equal(colors[0], colors[-1])


def test_mat2colors_column_vector_matches_1d():
    vals = np.linspace(0, 1, 20)
    assert np.allclose(mat2colors(vals), mat2colors(vals.reshape(-1, 1)))


def test_mat2colors_proportion_blend():
    # pure memberships recover the base component colors; a 50/50 mix lands
    # between them
    props = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    colors = mat2colors(props, palette='hls')
    import seaborn as sns
    base = np.asarray(sns.color_palette('hls', 2))
    assert np.allclose(colors[0], base[0], atol=1e-8)
    assert np.allclose(colors[1], base[1], atol=1e-8)
    assert np.allclose(colors[2], base.mean(axis=0), atol=1e-8)


def test_mat2colors_unnormalized_rows():
    # rows that don't sum to 1 (e.g. NMF loadings) are normalized internally
    m = np.array([[2.0, 0.0], [0.0, 4.0]])
    colors = mat2colors(m)
    assert np.allclose(colors, mat2colors(np.array([[1.0, 0.0], [0.0, 1.0]])))


def test_mat2colors_custom_palette_list():
    palette = [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    colors = mat2colors(np.array([[1.0, 0.0], [0.0, 1.0]]), palette=palette)
    assert np.allclose(colors[0], palette[0])
    assert np.allclose(colors[1], palette[1])


def test_mat2colors_rejects_3d():
    with pytest.raises(ValueError):
        mat2colors(np.zeros((2, 2, 2)))


def test_colors2groups_grouping():
    colors = np.array([[1.0, 0.0, 0.0]] * 5 + [[0.0, 0.0, 1.0]] * 5)
    group_ids, group_colors = colors2groups(colors)
    assert len(group_ids) == 10
    assert len(group_colors) == 2
    # ids are hashable and usable as grouping keys
    assert group_ids[0] == group_ids[4]
    assert group_ids[0] != group_ids[5]
    # group colors approximate member colors
    assert np.allclose(group_colors[group_ids[0]], (1, 0, 0), atol=0.15)


def test_colors2groups_resolution_bounds_group_count():
    rng = np.random.default_rng(0)
    colors = rng.random((500, 3))
    group_ids, group_colors = colors2groups(colors, res=3)
    assert len(group_colors) <= 27
