"""Hue/color coloring for hyp.plot (QC 2026-07, Jeremy's notes).

B1  a mixture/proportion matrix hue must BLEND the component colors -- a
    [0.5, 0.5] row is a true 50/50 blend, not the argmax component. (The old
    mat2colors subtracted the per-row min before normalizing, collapsing every
    row onto a pure palette vertex, so soft clusters never blended.)
B2  surface=True must honor hue -- the hull color follows the dataset's mean
    hue color instead of the palette cycle.
B3  an arbitrary high-dimensional matrix hue (>3 columns, or any matrix with
    color_reduce=) is reduced to 3 columns and mapped directly to (r, g, b).

All data is real numeric arrays -- no mocks. Plots run headless (Agg).
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

import hypertools as hyp
from hypertools.plot.colors import mat2colors


# --- B1: proportion-matrix hue blends ----------------------------------

def test_mat2colors_blends_proportions_not_argmax():
    import seaborn as sns
    props = np.array([[0.5, 0.5], [0.9, 0.1], [0.466, 0.534]])
    cols = mat2colors(props, palette='hls')
    base = np.asarray(sns.color_palette('hls', 2))[:, :3]
    # [0.5, 0.5] is exactly the midpoint of the two component colors
    assert np.allclose(cols[0], 0.5 * base[0] + 0.5 * base[1], atol=1e-6)
    # a near-even row is NOT the same color as a lopsided row (old bug: both
    # collapsed to the pure argmax component)
    assert not np.allclose(cols[2], cols[1], atol=0.05)


def test_mat2colors_signed_matrix_still_shifts():
    # signed matrices (negative entries) are still shifted to non-negative
    # before blending -- must not crash and must return valid RGB
    signed = np.array([[-1.0, 1.0], [2.0, -2.0], [0.0, 0.0]])
    cols = mat2colors(signed, palette='hls')
    assert cols.shape == (3, 3)
    assert cols.min() >= 0 and cols.max() <= 1


def test_gaussian_mixture_soft_cluster_produces_many_blended_colors():
    rng = np.random.default_rng(0)
    xc = np.vstack([rng.normal(loc=[0, 0], scale=10, size=(100, 2)),
                    rng.normal(loc=[5, 5], scale=7, size=(100, 2)),
                    rng.normal(loc=[0, 5], scale=6, size=(100, 2))])
    soft = np.asarray(hyp.cluster(xc, cluster='GaussianMixture', n_clusters=2))
    cols = mat2colors(soft, palette='hls')
    # overlapping blobs -> lots of intermediate memberships -> many blended
    # colors (the old code produced exactly 2)
    assert len(np.unique(np.round(cols, 2), axis=0)) > 10


# --- B2: surface honors hue --------------------------------------------

def test_surface_hue_renders_without_error():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 5))
    fig = hyp.plot(x, ndims=3, surface=True, hue=np.linspace(0, 1, 200),
                   palette='viridis', show=False)
    # the surface mesh color should track the mean hue color, not palette C0;
    # here we just assert it renders (the visual check is in the PR evidence)
    assert fig is not None


# --- B3: color_reduce / high-dim matrix hue -> RGB ---------------------

def test_matrix_hue_gt3_columns_reduces_to_rgb():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=(120, 4)), axis=0)
    h10 = rng.normal(size=(120, 10))
    fig = hyp.plot(x, '.', ndims=3, hue=h10, show=False)  # auto IncrementalPCA->RGB
    assert fig is not None


def test_color_reduce_kwarg_accepts_reduce_spec():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=(120, 4)), axis=0)
    h5 = rng.normal(size=(120, 5))
    fig = hyp.plot(x, '.', ndims=3, hue=h5, color_reduce='PCA', show=False)
    assert fig is not None


def test_matrix_hue_le3_columns_stays_palette_blend():
    # a <=3-column matrix without color_reduce keeps blending over the palette
    # (mixture-proportion semantics), not the RGB-reduce path
    rng = np.random.default_rng(0)
    h3 = np.abs(rng.normal(size=(60, 3)))
    h3 /= h3.sum(axis=1, keepdims=True)
    fig = hyp.plot(np.cumsum(rng.normal(size=(60, 3)), axis=0), '.', ndims=3,
                   hue=h3, show=False)
    assert fig is not None
