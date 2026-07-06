# -*- coding: utf-8 -*-
"""``animate='morph'`` (maintainer request, 2026-07-06): Hungarian-matched
point-cloud morphs between datasets, on both rendering backends, plus the
per-segment `rotations` list.

Covers `hypertools.plot.morph` (schedule math, Hungarian matching, easing,
color interpolation, azimuth math) in isolation, `plot.py`'s validation
(`animate=` scalar/list forms, `rotations=` list-vs-morph coupling), and
both backends' actual frame-by-frame wiring.
"""

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import hypertools as hyp
from hypertools.plot import morph


def _poly3d_verts(coll):
    """Extract a Poly3DCollection's raw vertex coordinates as an (n, 3)
    array, version-robustly (see tests/test_surface.py's identical
    helper -- duplicated here to avoid a cross-test-module import)."""
    vec = getattr(coll, '_vec', None)
    if vec is not None:
        return np.asarray(vec)[:3].T
    faces = getattr(coll, '_faces', None)
    if faces is not None:
        verts = np.asarray(faces, dtype=float).reshape(-1, 3)
        invalid = getattr(coll, '_invalid_vertices', False)
        if invalid is not False and np.ndim(invalid):
            verts = verts[~np.asarray(invalid).reshape(-1)]
        return verts
    segs = getattr(coll, '_segments3d', None)
    if segs is not None:
        return np.vstack([np.asarray(s, dtype=float)[:, :3] for s in segs])
    raise AttributeError('cannot locate 3-D vertices on this mpl version')


def _blobs(n=40, d=3, k=3, seed=0, spacing=6.0):
    """`k` well-separated Gaussian blobs of `n` points each, in `d` dims."""
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((n, d)) + spacing * i for i in range(k)]


# ---------------------------------------------------------------------------
# hypertools.plot.morph: pure math, no plotting
# ---------------------------------------------------------------------------

class TestSmoothstep:
    def test_endpoints(self):
        assert morph.smoothstep(0.0) == pytest.approx(0.0)
        assert morph.smoothstep(1.0) == pytest.approx(1.0)

    def test_midpoint(self):
        assert morph.smoothstep(0.5) == pytest.approx(0.5)

    def test_clips_outside_unit_interval(self):
        assert morph.smoothstep(-1.0) == pytest.approx(0.0)
        assert morph.smoothstep(2.0) == pytest.approx(1.0)


class TestSampleAndMatchClouds:
    def test_requires_at_least_two_clouds(self):
        with pytest.raises(ValueError, match="at least 2"):
            morph.sample_and_match_clouds([np.zeros((5, 3))])

    def test_known_optimal_hungarian_assignment(self):
        """2 clouds, 2 points each, where the optimal (minimum total
        distance) assignment is unambiguous and known by construction:
        cloud b's points are the EXACT reverse order of cloud a's."""
        a = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
        b = np.array([[10.0, 10.0, 10.0], [0.0, 0.0, 0.0]])
        sampled = morph.sample_and_match_clouds([a, b], morph_samples=2, seed=0)
        # after matching, sampled[1] row i must be b's point NEAREST to
        # sampled[0] row i, i.e. sampled[1] == a (reordered from b)
        np.testing.assert_allclose(sorted(sampled[0].tolist()),
                                   sorted(a.tolist()))
        np.testing.assert_allclose(sampled[1], sampled[0])

    def test_three_cloud_chain_matches_pairwise(self):
        """Chained matching: for every consecutive pair, the REALIZED
        (diagonal, i.e. row i <-> row i) assignment cost must equal the
        independently-recomputed Hungarian-optimal cost for that exact
        pair of point sets -- true for every link in the chain, not just
        the first."""
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial.distance import cdist

        clouds = _blobs(n=20, k=3, seed=1)
        sampled = morph.sample_and_match_clouds(clouds, morph_samples=20, seed=1)
        for k in range(2):
            cost = cdist(sampled[k], sampled[k + 1])
            realized = cost[np.arange(len(cost)), np.arange(len(cost))].sum()
            row, col = linear_sum_assignment(cost)
            assert realized == pytest.approx(cost[row, col].sum())

    def test_samples_equal_count_from_every_cloud(self):
        clouds = [np.zeros((10, 3)), np.zeros((30, 3)), np.zeros((15, 3))]
        sampled = morph.sample_and_match_clouds(clouds, morph_samples=8, seed=0)
        assert all(s.shape[0] == 8 for s in sampled)

    def test_default_cap_is_min_count_capped_at_1000(self):
        clouds = [np.zeros((1500, 3)), np.zeros((1200, 3))]
        sampled = morph.sample_and_match_clouds(clouds, seed=0)
        assert all(s.shape[0] == 1000 for s in sampled)

    def test_morph_samples_larger_than_cloud_is_capped_not_padded(self):
        clouds = [np.zeros((5, 3)), np.zeros((20, 3))]
        sampled = morph.sample_and_match_clouds(clouds, morph_samples=50, seed=0)
        assert all(s.shape[0] == 5 for s in sampled)


class TestSegmentFrameCounts:
    def test_length_is_2n_minus_1(self):
        for n in range(2, 6):
            assert len(morph.segment_frame_counts(n, 100)) == 2 * n - 1

    def test_sums_to_total_frames_when_evenly_divisible(self):
        counts = morph.segment_frame_counts(3, 100)
        assert sum(counts) == 100
        assert counts == [20, 20, 20, 20, 20]

    def test_remainder_goes_to_earliest_segments(self):
        counts = morph.segment_frame_counts(3, 102)  # 102 / 5 = 20 rem 2
        assert sum(counts) == 102
        assert counts == [21, 21, 20, 20, 20]

    def test_floor_when_fewer_frames_than_segments(self):
        counts = morph.segment_frame_counts(5, 3)  # 9 segments, only 3 frames
        assert len(counts) == 9
        assert sum(counts) == 9  # bumped up to at least 1 per segment
        assert all(c == 1 for c in counts)

    def test_requires_at_least_two_datasets(self):
        with pytest.raises(ValueError, match="at least 2"):
            morph.segment_frame_counts(1, 100)


class TestFrameToSegment:
    def test_maps_frame_zero_to_first_segment(self):
        counts = [4, 4, 4, 4, 4]
        assert morph.frame_to_segment(counts, 0) == (0, 0, 4)

    def test_maps_segment_boundary(self):
        counts = [4, 4, 4, 4, 4]
        assert morph.frame_to_segment(counts, 4) == (1, 0, 4)
        assert morph.frame_to_segment(counts, 8) == (2, 0, 4)

    def test_maps_last_frame_of_a_segment(self):
        counts = [4, 4, 4, 4, 4]
        assert morph.frame_to_segment(counts, 3) == (0, 3, 4)

    def test_uneven_counts(self):
        counts = [21, 21, 20, 20, 20]
        assert morph.frame_to_segment(counts, 21) == (1, 0, 21)
        assert morph.frame_to_segment(counts, 42) == (2, 0, 20)


class TestMorphPositionsAndColor:
    def setup_method(self):
        self.sampled = [np.zeros((5, 3)), np.ones((5, 3)) * 10,
                        np.ones((5, 3)) * 20]
        self.colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]

    def test_hold_segment_returns_dataset_unchanged(self):
        for step in (0, 3, 9):
            pts = morph.morph_positions(self.sampled, 0, step, 10)
            np.testing.assert_array_equal(pts, self.sampled[0])
        for step in (0, 3, 9):
            pts = morph.morph_positions(self.sampled, 2, step, 10)
            np.testing.assert_array_equal(pts, self.sampled[1])

    def test_morph_segment_endpoints_exact(self):
        pts0 = morph.morph_positions(self.sampled, 1, 0, 10)
        np.testing.assert_allclose(pts0, self.sampled[0])
        pts1 = morph.morph_positions(self.sampled, 1, 9, 10)
        np.testing.assert_allclose(pts1, self.sampled[1])

    def test_morph_segment_midpoint_matches_smoothstep_formula(self):
        step, n_steps = 3, 10
        t = morph.smoothstep(step / (n_steps - 1))
        expected = (1 - t) * self.sampled[0] + t * self.sampled[1]
        pts = morph.morph_positions(self.sampled, 1, step, n_steps)
        np.testing.assert_allclose(pts, expected)

    def test_color_hold_is_solid_dataset_color(self):
        c = morph.morph_color(self.colors, 2, 5, 10)
        assert c == self.colors[1]

    def test_color_morph_endpoints(self):
        c0 = morph.morph_color(self.colors, 1, 0, 10)
        c1 = morph.morph_color(self.colors, 1, 9, 10)
        np.testing.assert_allclose(c0, self.colors[0])
        np.testing.assert_allclose(c1, self.colors[1])

    def test_interpolate_color_linear(self):
        c = morph.interpolate_color((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.25)
        np.testing.assert_allclose(c, (0.25, 0.25, 0.25))


class TestResolveMorphRotationsAndAzimuths:
    def test_scalar_passes_through(self):
        assert morph.resolve_morph_rotations(2.0, 3) == 2.0

    def test_valid_list_length(self):
        result = morph.resolve_morph_rotations([1, 0.25, 2, 0.25, 1], 3)
        assert result == [1.0, 0.25, 2.0, 0.25, 1.0]

    def test_wrong_length_raises_naming_expected_length(self):
        with pytest.raises(ValueError, match="exactly 5"):
            morph.resolve_morph_rotations([1, 2], 3)

    def test_scalar_azimuths_match_uniform_pacing(self):
        counts = [5, 5, 5, 5, 5]
        azims = morph.segment_azimuths(counts, 2.0, -60)
        total = sum(counts)
        expected = [-60 + 360.0 * 2.0 * k / total for k in range(total)]
        np.testing.assert_allclose(azims, expected)

    def test_list_azimuths_cumulative_at_segment_boundaries(self):
        counts = [4, 4, 4, 4, 4]
        rotations = [1, 0.25, 2, 0.25, 1]
        azims = morph.segment_azimuths(counts, rotations, -60)
        assert azims[0] == pytest.approx(-60.0)
        # start of segment 1 = azim0 + 360 * rotations[0]
        assert azims[4] == pytest.approx(-60 + 360.0 * 1)
        # start of segment 2 = azim0 + 360 * sum(rotations[:2])
        assert azims[8] == pytest.approx(-60 + 360.0 * (1 + 0.25))
        # start of segment 3
        assert azims[12] == pytest.approx(-60 + 360.0 * (1 + 0.25 + 2))
        # start of segment 4 (final)
        assert azims[16] == pytest.approx(
            -60 + 360.0 * (1 + 0.25 + 2 + 0.25))

    def test_list_azimuths_within_segment_spread_uniformly(self):
        counts = [4, 4]
        rotations = [1, 0.5]
        azims = morph.segment_azimuths(counts, rotations, 0)
        # segment 0: 0, 90, 180, 270 (360 * 1 spread over 4 steps)
        np.testing.assert_allclose(azims[:4], [0, 90, 180, 270])


# ---------------------------------------------------------------------------
# plot.py validation
# ---------------------------------------------------------------------------

class TestAnimateMorphValidation:
    def test_scalar_morph_requires_two_datasets(self):
        with pytest.raises(ValueError, match="at least 2"):
            hyp.plot(_blobs(k=1), '.', animate='morph', duration=1,
                     frame_rate=5, show=False)

    def test_list_form_requires_two_tagged(self):
        with pytest.raises(ValueError, match="at least 2"):
            hyp.plot(_blobs(k=3), '.', animate=['morph', None, None],
                     duration=1, frame_rate=5, show=False)

    def test_list_wrong_length_raises(self):
        with pytest.raises(ValueError, match="2 entries"):
            hyp.plot(_blobs(k=3), '.', animate=['morph', 'morph'],
                     duration=1, frame_rate=5, show=False)

    def test_list_invalid_entry_raises(self):
        with pytest.raises(ValueError, match="'spin'"):
            hyp.plot(_blobs(k=3), '.', animate=['morph', 'spin', None],
                     duration=1, frame_rate=5, show=False)

    def test_rotations_list_requires_morph(self):
        with pytest.raises(ValueError, match="only supported with"):
            hyp.plot(_blobs(k=3), '.', animate=True,
                     rotations=[1, 0.25, 2, 0.25, 1], duration=1,
                     frame_rate=5, show=False)

    def test_rotations_list_wrong_length_for_morph(self):
        with pytest.raises(ValueError, match="exactly 5"):
            hyp.plot(_blobs(k=3), '.', animate='morph', rotations=[1, 2],
                     duration=1, frame_rate=5, show=False)

    def test_morph_2d_raises_not_implemented(self):
        data = [b[:, :2] for b in _blobs(k=3)]
        with pytest.raises(NotImplementedError, match="3-D"):
            hyp.plot(data, '.', animate='morph', ndims=2, duration=1,
                     frame_rate=5, show=False)


# ---------------------------------------------------------------------------
# matplotlib backend: actual frame wiring
# ---------------------------------------------------------------------------

class TestMplMorphAnimation:
    def _build(self, k=3, animate='morph', **kwargs):
        data = _blobs(n=30, k=k, seed=2)
        kwargs.setdefault('duration', 2)
        kwargs.setdefault('frame_rate', 10)
        return hyp.plot(data, '.', animate=animate, show=False, **kwargs)

    def test_hold_frame_matches_sampled_cloud_exactly(self):
        fig, ani = self._build()
        morph_state = ani._args[0]
        frame_counts = morph_state['frame_counts']
        assert len(frame_counts) == 5  # 3 datasets -> 5 segments

        ani._func(0, *ani._args)  # frame 0: hold on dataset 0
        artist = morph_state['artist']
        xs, ys, zs = artist.get_data_3d()
        expected = morph_state['sampled'][0]
        np.testing.assert_allclose(xs, expected[:, 0])
        np.testing.assert_allclose(ys, expected[:, 1])
        np.testing.assert_allclose(zs, expected[:, 2])

    def test_mid_morph_frame_matches_manual_smoothstep(self):
        fig, ani = self._build()
        morph_state = ani._args[0]
        frame_counts = morph_state['frame_counts']
        # first frame of segment 1 (the first morph) is at index frame_counts[0]
        mid_frame = frame_counts[0] + frame_counts[1] // 2
        seg_idx, step, n_steps = morph.frame_to_segment(frame_counts, mid_frame)
        assert seg_idx == 1
        expected = morph.morph_positions(morph_state['sampled'], seg_idx,
                                         step, n_steps)

        ani._func(mid_frame, *ani._args)
        xs, ys, zs = morph_state['artist'].get_data_3d()
        np.testing.assert_allclose(xs, expected[:, 0])
        np.testing.assert_allclose(ys, expected[:, 1])
        np.testing.assert_allclose(zs, expected[:, 2])

    def test_color_interpolates_during_morph_segment(self):
        fig, ani = self._build()
        morph_state = ani._args[0]
        frame_counts = morph_state['frame_counts']
        mid_frame = frame_counts[0] + frame_counts[1] // 2
        seg_idx, step, n_steps = morph.frame_to_segment(frame_counts, mid_frame)
        expected_color = morph.morph_color(morph_state['colors'], seg_idx,
                                           step, n_steps)

        ani._func(mid_frame, *ani._args)
        import matplotlib.colors as mcolors
        actual_color = mcolors.to_rgb(morph_state['artist'].get_color())
        np.testing.assert_allclose(actual_color, expected_color, atol=1e-6)

    def test_rotations_list_azimuth_boundaries(self):
        fig, ani = self._build(rotations=[1, 0.25, 2, 0.25, 1])
        morph_state = ani._args[0]
        frame_counts = morph_state['frame_counts']
        ax = fig.axes[0]

        boundaries = [0, frame_counts[0],
                     frame_counts[0] + frame_counts[1],
                     sum(frame_counts[:3]), sum(frame_counts[:4])]
        rotations_cum = [0, 1, 1.25, 3.25, 3.5]
        for boundary, cum in zip(boundaries, rotations_cum):
            ani._func(boundary, *ani._args)
            expected_azim = -60 + 360.0 * cum
            assert ax.azim == pytest.approx(expected_azim, abs=1e-6)

    def test_static_untagged_dataset_present_every_frame(self):
        fig, ani = self._build(k=4, animate=['morph', 'morph', 'morph', None])
        ax = fig.axes[0]
        static_line = ax.lines[3]
        assert static_line.get_visible()
        before = static_line.get_data_3d()
        for k in [0, 5, 15, 19]:
            ani._func(k, *ani._args)
            after = static_line.get_data_3d()
            np.testing.assert_allclose(before, after)
            assert static_line.get_visible()

    def test_morph_tagged_lines_hidden(self):
        fig, ani = self._build(k=4, animate=['morph', 'morph', 'morph', None])
        ax = fig.axes[0]
        for i in range(3):
            assert not ax.lines[i].get_visible()

    def test_trails_ignored_with_warning_for_morph(self):
        with pytest.warns(UserWarning, match="morph"):
            self._build(animate='morph', chemtrails=True)

    def test_surface_morph_mesh_changes_across_frames(self):
        fig, ani = self._build(surface=True)
        ax = fig.axes[0]
        morph_state = ani._args[0]
        frame_counts = morph_state['frame_counts']

        ani._func(0, *ani._args)
        colls0 = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        verts0 = np.vstack([_poly3d_verts(c) for c in colls0])

        mid = frame_counts[0] + frame_counts[1] // 2
        ani._func(mid, *ani._args)
        colls1 = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        verts1 = np.vstack([_poly3d_verts(c) for c in colls1])

        assert verts0.shape != verts1.shape or not np.allclose(verts0, verts1)


# ---------------------------------------------------------------------------
# plotly backend: frame counts, camera eyes, static datasets, surface
# ---------------------------------------------------------------------------

class TestPlotlyMorphAnimation:
    def _build(self, k=3, animate='morph', **kwargs):
        data = _blobs(n=30, k=k, seed=3)
        kwargs.setdefault('duration', 2)
        kwargs.setdefault('frame_rate', 10)
        return hyp.plot(data, '.', animate=animate, backend='plotly',
                        show=False, **kwargs)

    def test_frame_count_matches_duration_and_frame_rate(self):
        fig = self._build()
        assert len(fig.frames) == 20  # duration=2 * frame_rate=10

    def test_camera_eye_rotates_with_rotations_list(self):
        from hypertools.plot.plotly_backend import _camera_eye, _anim_zoom_r

        fig = self._build(rotations=[1, 0.25, 2, 0.25, 1])
        counts = morph.segment_frame_counts(3, 20)
        azims = morph.segment_azimuths(counts, [1, 0.25, 2, 0.25, 1], -60)

        for k in [0, counts[0], sum(counts[:2]), 19]:
            expected_eye = _camera_eye(10, azims[k], r=_anim_zoom_r(1))
            actual_eye = fig.frames[k].layout.scene.camera.eye
            assert actual_eye.x == pytest.approx(expected_eye['x'], abs=1e-6)
            assert actual_eye.y == pytest.approx(expected_eye['y'], abs=1e-6)
            assert actual_eye.z == pytest.approx(expected_eye['z'], abs=1e-6)

    def test_static_untagged_dataset_trace_untouched_by_frames(self):
        fig = self._build(k=4, animate=['morph', 'morph', 'morph', None])
        # the static dataset's own trace is trace 0 (built first, in the
        # main per-dataset loop, before the morph traces are appended) --
        # no frame should ever reference it.
        for frame in fig.frames:
            assert 0 not in frame.traces
        assert fig.data[0].visible in (True, None)

    def test_surface_morph_mesh_changes_across_frames(self):
        fig = self._build(surface=True)
        mesh_traces = [i for i, t in enumerate(fig.data) if t.type == 'mesh3d']
        assert len(mesh_traces) == 1

        frame0_mesh = fig.frames[0].data[
            [t.type for t in fig.frames[0].data].index('mesh3d')]
        mid = len(fig.frames) // 2
        mid_mesh = fig.frames[mid].data[
            [t.type for t in fig.frames[mid].data].index('mesh3d')]
        x0, xm = np.asarray(frame0_mesh.x), np.asarray(mid_mesh.x)
        assert x0.shape != xm.shape or not np.allclose(x0, xm)

    def test_trails_ignored_with_warning_for_morph(self):
        with pytest.warns(UserWarning, match="morph"):
            self._build(animate='morph', chemtrails=True)
