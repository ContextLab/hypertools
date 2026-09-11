# -*- coding: utf-8 -*-
"""A 3-D plotly `density=` volume stays inside the cube.

1.1 release review (L2): the KDE grid behind the 3-D `go.Volume` is padded
past the data, so it reached beyond the scene's [-1, 1] range (x from -1.30
to 1.26 on the reviewer's case) and its translucent shells were drawn over
the cube's edges, which rendered stippled -- only ~1698 of 3899 cube pixels
survived. The grid is now clipped to the cube.

Kaleido-rendered pixels: every dark pixel the figure draws without its
volume must still be drawn with it.
"""

import io

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

import hypertools as hyp

go = pytest.importorskip('plotly.graph_objects')


def _fig(**kw):
    points = np.random.default_rng(0).normal(size=(70, 3))
    return hyp.plot(points, fmt='o', markersize=2, reduce=None, ndims=3,
                    density={'grid': 15, 'alpha': .15, 'levels': 2},
                    show=False, backend='plotly', **kw)


def _dark(fig):
    from PIL import Image
    img = Image.open(io.BytesIO(fig.to_image(format='png')))
    return np.asarray(img.convert('L')) < 110


def test_volume_grid_stays_inside_the_scene_range():
    fig = _fig()
    vol, = [t for t in fig.data if t.type == 'volume']
    lim = fig.layout.scene.xaxis.range[1]
    for axis in (vol.x, vol.y, vol.z):
        a = np.asarray(axis, dtype=float)
        assert a.min() >= -lim - 1e-12 and a.max() <= lim + 1e-12


def test_cube_edges_survive_the_volume():
    fig = _fig()
    bare = go.Figure(layout=fig.layout)
    for t in fig.data:
        if t.type != 'volume':
            bare.add_trace(t)
    ref = _dark(bare)
    kept = _dark(fig) & ref
    assert ref.sum() > 1000
    assert kept.sum() >= 0.98 * ref.sum(), (int(kept.sum()), int(ref.sum()))


def test_pooled_density_is_clipped_too():
    rng = np.random.default_rng(1)
    data = [rng.normal(size=(40, 3)), rng.normal(size=(40, 3)) + 2]
    fig = hyp.plot(data, fmt='o', reduce=None, ndims=3,
                   density={'per_group': False, 'grid': 15}, show=False,
                   backend='plotly')
    vol, = [t for t in fig.data if t.type == 'volume']
    lim = fig.layout.scene.xaxis.range[1]
    assert np.abs(np.asarray(vol.x, dtype=float)).max() <= lim + 1e-12


def test_volume_still_shows_its_glow():
    """Clipping must not blank the density: the volume still draws a
    visible share of the scene."""
    fig = _fig()
    bare = go.Figure(layout=fig.layout)
    for t in fig.data:
        if t.type != 'volume':
            bare.add_trace(t)
    from PIL import Image
    a = np.asarray(Image.open(io.BytesIO(fig.to_image(format='png')))
                   .convert('RGB')).astype(int)
    b = np.asarray(Image.open(io.BytesIO(bare.to_image(format='png')))
                   .convert('RGB')).astype(int)
    changed = (np.abs(a - b).sum(axis=2) > 12).sum()
    assert changed > 2000
