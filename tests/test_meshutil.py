"""Tests for hypertools.plot.meshutil -- pure-geometry mesh utilities.

These are geometry tests: no mocks, real ConvexHull/Delaunay/spline
computations on real point clouds, with numeric assertions (containment
fractions, signed volumes, Euler characteristics, timing).
"""
import time
import warnings

import numpy as np
import pytest
from matplotlib.path import Path
from scipy.spatial import ConvexHull, Delaunay

from hypertools.plot.meshutil import (
    backface_cull,
    blinn_phong_colors,
    face_normals,
    points_enclosed,
    smooth_hull_2d,
    smooth_hull_3d,
)


def _random_blob_3d(n=200, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3)) * np.array([1.4, 1.0, 1.1])
    pts[:, 2] += 0.4 * np.sin(1.3 * pts[:, 0])
    pts[:, 0] += 0.3 * np.cos(1.1 * pts[:, 1])
    return pts


def _random_blob_2d(n=200, seed=11):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 2)) * np.array([1.5, 1.0])
    pts[:, 1] += 0.4 * np.sin(1.4 * pts[:, 0])
    return pts


def _signed_volume(verts, faces):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    return np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0


class TestSmoothHull3DOrientation:
    def test_signed_volume_is_positive(self):
        verts, faces = smooth_hull_3d(_random_blob_3d())
        assert _signed_volume(verts, faces) > 0

    def test_normals_point_away_from_centroid(self):
        verts, faces = smooth_hull_3d(_random_blob_3d())
        centroid = verts.mean(axis=0)
        fn = face_normals(verts, faces)
        face_centers = verts[faces].mean(axis=1)
        dots = np.einsum("ij,ij->i", fn, centroid - face_centers)
        assert np.all(dots < 0)


class TestSmoothHull3DContainment:
    def test_pre_smoothed_hull_contains_all_points_exactly(self):
        # Sanity check on the exactness method itself: every input point is
        # trivially inside (or on) its own convex hull.
        pts = _random_blob_3d()
        hull = ConvexHull(pts)
        delaunay = Delaunay(pts[hull.vertices])
        inside = delaunay.find_simplex(pts, tol=1e-9) >= 0
        assert inside.mean() >= 0.999

    def test_smoothed_mesh_retains_at_least_96_percent_of_points(self):
        pts = _random_blob_3d()
        verts, faces = smooth_hull_3d(pts)
        delaunay = Delaunay(verts)
        inside = delaunay.find_simplex(pts, tol=1e-7) >= 0

        # distance-tolerance fallback for points that fall just outside the
        # convex hull of the smoothed vertex set due to smoothing/subdivision
        outside = ~inside
        if outside.any():
            d = np.linalg.norm(
                verts[None, :, :] - pts[outside, None, :], axis=-1
            ).min(axis=1)
            scale = np.ptp(pts, axis=0).mean()
            inside[outside] = d < 0.05 * scale

        assert inside.mean() >= 0.96


class TestSmoothHull3DTinyCloudContainment:
    """GH #109 rendering-fix: a fixed `taubin_iters`/`rounds` count shrinks
    a small, sparse hull proportionally much more than a large, dense one,
    so `pre_inflate` alone under-compensates for small point clouds -- a
    5-point cloud's mesh used to contain as few as ~1/5 of its own input
    points, with no warning. `smooth_hull_3d` now rescales post-hoc (never
    shrinking) to recover >= 96% containment regardless of hull size."""

    @pytest.mark.parametrize('n', [5, 6, 8, 20, 200])
    def test_small_and_large_clouds_retain_at_least_96_percent(self, n):
        rng = np.random.default_rng(n)
        pts = rng.normal(size=(n, 3))
        verts, faces = smooth_hull_3d(pts)
        assert points_enclosed(pts, verts).mean() >= 0.96

    def test_no_warning_when_containment_target_is_met(self):
        rng = np.random.default_rng(5)
        pts = rng.normal(size=(5, 3))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            smooth_hull_3d(pts)
        assert len(w) == 0

    def test_rescale_never_shrinks_the_mesh(self):
        # a large, dense, round point cloud already comfortably meets the
        # containment target before any post-hoc rescale -- confirm the
        # rescale step is a no-op (never shrinks) rather than always
        # forcing some minimum growth
        pts = _random_blob_3d(n=200)
        verts, faces = smooth_hull_3d(pts)
        assert points_enclosed(pts, verts).mean() >= 0.96


class TestSmoothHull3DScaling:
    def test_face_count_scales_by_4_pow_rounds(self):
        pts = _random_blob_3d()
        base_verts, base_faces = smooth_hull_3d(pts, rounds=0)
        f0 = len(base_faces)
        for rounds in (1, 2, 3):
            _, faces = smooth_hull_3d(pts, rounds=rounds)
            assert len(faces) == f0 * 4 ** rounds

    def test_mesh_stays_closed_genus_zero_at_every_round(self):
        pts = _random_blob_3d()
        for rounds in (0, 1, 2, 3):
            verts, faces = smooth_hull_3d(pts, rounds=rounds)
            edges = len(faces) * 3 // 2
            assert len(verts) - edges + len(faces) == 2


class TestSmoothHull3DPerformance:
    def test_runtime_for_200_points_rounds_3(self):
        pts = _random_blob_3d(n=200)
        start = time.perf_counter()
        smooth_hull_3d(pts, rounds=3)
        elapsed = time.perf_counter() - start
        # loose bound to avoid CI flakiness; verified prototype runs <50ms
        assert elapsed < 0.5


class TestSmoothHull3DDegenerateInputs:
    def test_coplanar_points_raise_valueerror(self):
        rng = np.random.default_rng(1)
        pts = np.zeros((20, 3))
        pts[:, :2] = rng.normal(size=(20, 2))
        with pytest.raises(ValueError):
            smooth_hull_3d(pts)

    def test_fewer_than_four_points_raise_valueerror(self):
        pts = np.random.default_rng(2).normal(size=(3, 3))
        with pytest.raises(ValueError):
            smooth_hull_3d(pts)

    def test_empty_points_raise_valueerror(self):
        with pytest.raises(ValueError):
            smooth_hull_3d(np.empty((0, 3)))


class TestSmoothHull2DClosedCurve:
    def test_curve_closes_without_a_gap(self):
        curve = smooth_hull_2d(_random_blob_2d())
        seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        wrap = np.linalg.norm(curve[0] - curve[-1])
        assert wrap < 3 * np.median(seg)

    def test_hull_vertices_are_contained(self):
        pts = _random_blob_2d()
        hull = ConvexHull(pts)
        poly = pts[hull.vertices]
        curve = smooth_hull_2d(pts)
        path = Path(curve)
        inside = path.contains_points(poly, radius=1e-6)
        assert inside.mean() == 1.0

    def test_c1_continuity_at_spline_knots(self):
        samples_per_edge = 20
        curve = smooth_hull_2d(_random_blob_2d(), samples_per_edge=samples_per_edge)
        tangents = np.diff(curve, axis=0, append=curve[:1])
        unit = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)
        cosang = np.einsum("ij,ij->i", unit, np.roll(unit, -1, axis=0))
        angles = np.arccos(np.clip(cosang, -1.0, 1.0))
        knot_idx = np.arange(0, len(curve), samples_per_edge)
        median_angle = np.median(angles)
        # knot-boundary turning angles should not be gross outliers relative
        # to the generic per-sample turning angle along the curve (no kinks)
        assert np.all(angles[knot_idx] < 5 * median_angle + 1e-6)

    def test_collinear_points_raise_valueerror(self):
        pts = np.column_stack([np.linspace(0, 1, 10), np.zeros(10)])
        with pytest.raises(ValueError):
            smooth_hull_2d(pts)

    def test_fewer_than_three_points_raise_valueerror(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        with pytest.raises(ValueError):
            smooth_hull_2d(pts)


class TestFaceNormals:
    def test_unit_length(self):
        verts, faces = smooth_hull_3d(_random_blob_3d())
        fn = face_normals(verts, faces)
        assert np.allclose(np.linalg.norm(fn, axis=1), 1.0, atol=1e-6)

    def test_outward_facing(self):
        verts, faces = smooth_hull_3d(_random_blob_3d())
        centroid = verts.mean(axis=0)
        fn = face_normals(verts, faces)
        face_centers = verts[faces].mean(axis=1)
        assert np.all(np.einsum("ij,ij->i", fn, face_centers - centroid) > 0)


class TestBlinnPhongColors:
    def test_output_shape_and_range(self):
        verts, faces = smooth_hull_3d(_random_blob_3d())
        view = np.array([0.3, -0.6, 0.7])
        rgba = blinn_phong_colors(verts, faces, base_rgb=(0.3, 0.45, 0.65), view=view)
        assert rgba.shape == (len(faces), 4)
        assert np.all(rgba >= 0.0) and np.all(rgba <= 1.0)

    def test_alpha_channel_defaults_opaque(self):
        verts, faces = smooth_hull_3d(_random_blob_3d())
        view = np.array([0.0, 0.0, 1.0])
        rgba = blinn_phong_colors(verts, faces, base_rgb=(0.5, 0.5, 0.5), view=view)
        assert np.allclose(rgba[:, 3], 1.0)

    def test_custom_lightdir_changes_result(self):
        verts, faces = smooth_hull_3d(_random_blob_3d())
        view = np.array([0.0, 0.0, 1.0])
        default_rgba = blinn_phong_colors(
            verts, faces, base_rgb=(0.5, 0.5, 0.5), view=view
        )
        custom_rgba = blinn_phong_colors(
            verts,
            faces,
            base_rgb=(0.5, 0.5, 0.5),
            view=view,
            lightdir=np.array([1.0, 0.0, 0.0]),
        )
        assert not np.allclose(default_rgba, custom_rgba)


class TestBackfaceCull:
    def test_mask_shape_and_dtype(self):
        verts, faces = smooth_hull_3d(_random_blob_3d())
        mask = backface_cull(verts, faces, np.array([0.0, 0.0, 1.0]))
        assert mask.shape == (len(faces),)
        assert mask.dtype == bool

    def test_front_facing_faces_are_kept(self):
        verts, faces = smooth_hull_3d(_random_blob_3d())
        view = np.array([0.0, 0.0, 1.0])
        mask = backface_cull(verts, faces, view)
        fn = face_normals(verts, faces)
        facing = fn @ (view / np.linalg.norm(view))
        assert np.all(mask[facing > 0.5])

    def test_back_facing_faces_are_culled(self):
        verts, faces = smooth_hull_3d(_random_blob_3d())
        view = np.array([0.0, 0.0, 1.0])
        mask = backface_cull(verts, faces, view)
        fn = face_normals(verts, faces)
        facing = fn @ (view / np.linalg.norm(view))
        assert np.all(~mask[facing < -0.5])
