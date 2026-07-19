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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import hypertools as hyp
from hypertools.plot import morph
from hypertools.plot.matplotlib_backend import _anim_box_zoom


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
        sampled, dup_masks = morph.sample_and_match_clouds(
            [a, b], morph_samples=2, seed=0)
        # after matching, sampled[1] row i must be b's point NEAREST to
        # sampled[0] row i, i.e. sampled[1] == a (reordered from b)
        np.testing.assert_allclose(sorted(sampled[0].tolist()),
                                   sorted(a.tolist()))
        np.testing.assert_allclose(sampled[1], sampled[0])
        # both clouds already have 2 points (the target count) -- no
        # duplication needed for either.
        assert not dup_masks[0].any()
        assert not dup_masks[1].any()

    def test_three_cloud_chain_matches_pairwise(self):
        """Chained matching: for every consecutive pair, the REALIZED
        (diagonal, i.e. row i <-> row i) assignment cost must equal the
        independently-recomputed Hungarian-optimal cost for that exact
        pair of point sets -- true for every link in the chain, not just
        the first."""
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial.distance import cdist

        clouds = _blobs(n=20, k=3, seed=1)
        sampled, dup_masks = morph.sample_and_match_clouds(
            clouds, morph_samples=20, seed=1)
        for k in range(2):
            cost = cdist(sampled[k], sampled[k + 1])
            realized = cost[np.arange(len(cost)), np.arange(len(cost))].sum()
            row, col = linear_sum_assignment(cost)
            assert realized == pytest.approx(cost[row, col].sum())

    def test_samples_equal_count_from_every_cloud(self):
        """`morph_samples` caps EVERY cloud at 8 points before duplication;
        all 3 clouds here (10/30/15) exceed the cap, so all are downsampled
        to exactly 8 -- an all-equal-post-cap case, so no duplication is
        needed and every dup mask is all-False."""
        clouds = [np.zeros((10, 3)), np.zeros((30, 3)), np.zeros((15, 3))]
        sampled, dup_masks = morph.sample_and_match_clouds(
            clouds, morph_samples=8, seed=0)
        assert all(s.shape[0] == 8 for s in sampled)
        assert all(not m.any() for m in dup_masks)

    def test_default_is_uncapped_target_is_largest_cloud(self):
        """Full-sample morphs (maintainer request, 2026-07-06 follow-up):
        the DEFAULT (`morph_samples=None`) no longer shrinks every cloud
        to the smallest one capped at 1000 -- it applies NO cap at all, so
        the target count is simply the largest cloud's own point count,
        and every smaller cloud is padded (duplicated) up to it."""
        clouds = [np.zeros((1500, 3)), np.zeros((1200, 3))]
        sampled, dup_masks = morph.sample_and_match_clouds(clouds, seed=0)
        assert all(s.shape[0] == 1500 for s in sampled)
        assert not dup_masks[0].any()
        assert dup_masks[1].sum() == 300

    def test_morph_samples_caps_before_duplication(self):
        """A `morph_samples` cap LARGER than every cloud has no downsampling
        effect on either cloud, but the smaller cloud is still padded
        (duplicated) up to the larger cloud's own (uncapped) size --
        `morph_samples` only ever caps from above, it never prevents the
        padding-to-largest behavior."""
        clouds = [np.zeros((5, 3)), np.zeros((20, 3))]
        sampled, dup_masks = morph.sample_and_match_clouds(
            clouds, morph_samples=50, seed=0)
        assert all(s.shape[0] == 20 for s in sampled)
        assert dup_masks[0].sum() == 15  # 20 - 5
        assert not dup_masks[1].any()


class TestFullSampleDuplication:
    """Numerical evidence for the full-sample-morph algorithm (maintainer
    request, 2026-07-06 follow-up): 3 datasets of UNEQUAL size (100, 250,
    40) -- the largest (250) is the target `n`; the other two are padded
    up to it by duplicating their own points."""

    def _clouds(self, seed=0):
        rng = np.random.default_rng(seed)
        return [rng.standard_normal((100, 3)),
                rng.standard_normal((250, 3)),
                rng.standard_normal((40, 3))]

    def test_target_n_is_largest_dataset_and_dup_counts_are_exact(self):
        clouds = self._clouds()
        sampled, dup_masks = morph.sample_and_match_clouds(clouds, seed=0)
        assert all(s.shape[0] == 250 for s in sampled)
        assert dup_masks[0].sum() == 150   # 250 - 100
        assert dup_masks[1].sum() == 0     # the largest: no duplicates
        assert dup_masks[2].sum() == 210   # 250 - 40
        assert all(m.shape == (250,) for m in dup_masks)

    def test_every_original_point_present_exactly_once_among_non_dups(self):
        """Set equality: the non-duplicate rows of `sampled[k]` are EXACTLY
        the original dataset's points, each appearing exactly once (no
        original point dropped, none appearing twice among non-dups)."""
        clouds = self._clouds()
        sampled, dup_masks = morph.sample_and_match_clouds(clouds, seed=0)
        for k, original in enumerate(clouds):
            non_dup = sampled[k][~dup_masks[k]]
            assert non_dup.shape[0] == original.shape[0]
            # exact set equality (row order may differ after Hungarian
            # matching for k >= 1, so sort both by their raw bytes)
            a = sorted(map(tuple, non_dup.tolist()))
            b = sorted(map(tuple, original.tolist()))
            assert a == b

    def test_duplicates_are_exact_copies_of_the_same_datasets_own_points(self):
        clouds = self._clouds()
        sampled, dup_masks = morph.sample_and_match_clouds(clouds, seed=0)
        for k, original in enumerate(clouds):
            dup_rows = sampled[k][dup_masks[k]]
            original_set = set(map(tuple, original.tolist()))
            for row in dup_rows.tolist():
                assert tuple(row) in original_set

    def test_n_greater_than_2m_uses_replacement_and_masks_are_correct(self):
        """Edge case: a 10-point dataset paired against a 25-point dataset
        (n = 25 > 2 * 10): 15 duplicates are needed from only 10 real
        points, so sampling MUST be done with replacement (some real
        points get copied more than once) -- this must not raise, and the
        resulting mask/set-equality guarantees above must still hold."""
        rng = np.random.default_rng(3)
        small = rng.standard_normal((10, 3))
        big = rng.standard_normal((25, 3))
        sampled, dup_masks = morph.sample_and_match_clouds(
            [small, big], seed=3)
        assert sampled[0].shape[0] == 25
        assert dup_masks[0].sum() == 15  # 25 - 10, need > m -> replacement
        assert not dup_masks[1].any()
        non_dup = sampled[0][~dup_masks[0]]
        assert sorted(map(tuple, non_dup.tolist())) == sorted(
            map(tuple, small.tolist()))
        dup_rows = sampled[0][dup_masks[0]]
        small_set = set(map(tuple, small.tolist()))
        for row in dup_rows.tolist():
            assert tuple(row) in small_set


class TestMorphVisibleMask:
    def test_none_dup_masks_hides_nothing(self):
        assert morph.morph_visible_mask(None, 0) is None
        assert morph.morph_visible_mask(None, 1) is None

    def test_hold_segment_returns_that_datasets_own_mask(self):
        masks = [np.array([True, False]), np.array([False, False]),
                 np.array([False, True])]
        np.testing.assert_array_equal(
            morph.morph_visible_mask(masks, 0), masks[0])
        np.testing.assert_array_equal(
            morph.morph_visible_mask(masks, 2), masks[1])
        np.testing.assert_array_equal(
            morph.morph_visible_mask(masks, 4), masks[2])

    def test_morph_segment_hides_nothing(self):
        masks = [np.array([True, False]), np.array([False, True])]
        assert morph.morph_visible_mask(masks, 1) is None


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


class TestConstantRotationSpeed:
    """Maintainer request (2026-07-06): "the rotation speed should always
    be constant -- so more rotations means more time spent on that part
    of the animation." A LIST `rotations` must allocate each segment's
    frame count PROPORTIONAL to its own rotation count, so degrees/frame
    is identical everywhere, rather than splitting frames evenly (which
    made higher-rotation segments spin faster)."""

    def _deg_per_frame(self, frame_counts, rotations, azim0=0):
        azims = morph.segment_azimuths(frame_counts, rotations, azim0)
        return np.diff(azims)

    def test_proportional_split_exact_ratios(self):
        # 2 datasets -> 3 segments; rotations=[2, 1, 1] -> frame shares
        # 2/4, 1/4, 1/4 of the total, exactly (evenly divisible case).
        counts = morph.segment_frame_counts(2, 80, [2.0, 1.0, 1.0])
        assert counts == [40, 20, 20]
        assert sum(counts) == 80

    def test_proportional_split_largest_remainder(self):
        # Not evenly divisible: shares must still sum EXACTLY to the
        # total, largest-fractional-remainder segments getting the extra
        # frame(s).
        counts = morph.segment_frame_counts(2, 81, [2.0, 1.0, 1.0])
        assert sum(counts) == 81
        # 81 * 2/4 = 40.5, 81 * 1/4 = 20.25 (x2) -> floors [40, 20, 20],
        # remainder 1 goes to the largest fractional share (segment 0).
        assert counts == [41, 20, 20]

    def test_degrees_per_frame_constant_across_all_frames(self):
        rotations = [2.0, 1.0, 1.0]
        counts = morph.segment_frame_counts(2, 80, rotations)
        deltas = self._deg_per_frame(counts, rotations)
        expected = 360.0 * sum(rotations) / sum(counts)
        np.testing.assert_allclose(deltas, expected, atol=1e-6)

    def test_degrees_per_frame_constant_five_segment_case(self):
        # The maintainer's own example: 3 datasets -> 5 segments. 180
        # frames divides evenly by every weight here (sum=4.5), so the
        # largest-remainder split needs no rounding at all.
        rotations = [1, 0.25, 2, 0.25, 1]
        counts = morph.segment_frame_counts(3, 180, rotations)
        assert counts == [40, 10, 80, 10, 40]
        deltas = self._deg_per_frame(counts, rotations)
        expected = 360.0 * sum(rotations) / sum(counts)
        np.testing.assert_allclose(deltas, expected, atol=1e-6)

    def test_zero_rotation_segment_gets_floor_share_and_stays_still(self):
        # 22 frames divides evenly given weights [1, 0.1, 1] (sum 2.1) once
        # the middle segment is bumped to its 2-frame minimum, so segments
        # 0 and 2 come out exactly equal (no rounding tie to break).
        rotations = [1, 0, 1]
        counts = morph.segment_frame_counts(2, 22, rotations)
        assert counts == [10, 2, 10]
        assert counts[1] >= 2  # min-2-frames floor
        # segments 0 and 2 (equal, nonzero rotation) get equal frame
        # counts and thus equal (constant) angular speed...
        assert counts[0] == counts[2]
        azims = morph.segment_azimuths(counts, rotations, -60)
        seg0_delta = np.diff(azims[:counts[0]])
        seg2_start = counts[0] + counts[1]
        seg2_delta = np.diff(azims[seg2_start:seg2_start + counts[2]])
        np.testing.assert_allclose(seg0_delta, seg2_delta, atol=1e-6)
        # ...while the zero-rotation middle segment's azimuth never moves.
        seg1_delta = np.diff(azims[counts[0]:counts[0] + counts[1]])
        np.testing.assert_allclose(seg1_delta, 0.0, atol=1e-9)

    def test_every_segment_at_least_two_frames(self):
        counts = morph.segment_frame_counts(3, 21, [5, 0, 0, 0, 5])
        assert all(c >= 2 for c in counts)
        assert sum(counts) == 21

    def test_scalar_rotations_frame_counts_unchanged(self):
        """Regression: scalar `rotations` must still split frames EVENLY
        -- scalar behavior is constant speed by construction (equal
        segment durations) and must not change with this fix."""
        assert morph.segment_frame_counts(3, 100, 2.0) == [20, 20, 20, 20, 20]
        assert morph.segment_frame_counts(3, 100, None) == [20, 20, 20, 20, 20]
        assert (morph.segment_frame_counts(3, 100, 2.0)
               == morph.segment_frame_counts(3, 100))

    def test_scalar_rotations_azimuth_step_constant(self):
        counts = morph.segment_frame_counts(3, 100, 2.0)
        azims = morph.segment_azimuths(counts, 2.0, -60)
        deltas = np.diff(azims)
        expected = 360.0 * 2.0 / sum(counts)
        np.testing.assert_allclose(deltas, expected, atol=1e-6)

    def test_morph_schedule_single_call_matches_pieces(self):
        frame_counts, rotations_resolved, azimuths = morph.morph_schedule(
            3, 200, [1, 0.25, 2, 0.25, 1], -60)
        expected_rotations = morph.resolve_morph_rotations(
            [1, 0.25, 2, 0.25, 1], 3)
        expected_counts = morph.segment_frame_counts(
            3, 200, expected_rotations)
        expected_azimuths = morph.segment_azimuths(
            expected_counts, expected_rotations, -60)
        assert rotations_resolved == expected_rotations
        assert frame_counts == expected_counts
        np.testing.assert_allclose(azimuths, expected_azimuths)


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

    def test_morph_2d_now_works(self):
        # round17 #9 (GH #123): animate='morph' now supports 2-D data,
        # same as every other animate style -- redefines the old 3-D-only
        # contract (previously a NotImplementedError here). Mirrors
        # `TestMplMorphAnimation.test_hold_frame_matches_sampled_cloud_exactly`
        # below, but for a plain (non-Axes3D) 2-D figure.
        data = [b[:, :2] for b in _blobs(k=3, seed=2)]
        fig, ani = hyp.plot(data, '.', animate='morph', ndims=2, duration=1,
                            frame_rate=5, show=False)
        ax = fig.axes[0]
        assert not hasattr(ax, "get_proj")  # a plain (2-D) Axes, not Axes3D

        morph_state = ani._args[0]
        frame_counts = morph_state['frame_counts']
        assert len(frame_counts) == 5  # 3 datasets -> 5 segments

        ani._func(0, *ani._args)  # frame 0: hold on dataset 0
        artist = morph_state['artist']
        xs, ys = artist.get_data()
        expected = morph_state['sampled'][0]
        np.testing.assert_allclose(xs, expected[:, 0])
        np.testing.assert_allclose(ys, expected[:, 1])
        plt.close(fig)

    def test_morph_1d_still_raises_not_implemented(self):
        data = [b[:, :1] for b in _blobs(k=3)]
        with pytest.raises(NotImplementedError, match="2-D or 3-D"):
            hyp.plot(data, '.', animate='morph', ndims=1, duration=1,
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

    def test_rotations_list_constant_rotation_speed(self):
        """Maintainer request (2026-07-06): constant rotation speed --
        segment duration proportional to rotation count. Sampling `ax.azim`
        at every frame across the whole mpl animation, the per-frame delta
        must be identical everywhere (within a tiny rounding tolerance at
        segment boundaries from the largest-remainder integer split). Uses
        180 frames (duration=18 * frame_rate=10), which divides the
        [1, 0.25, 2, 0.25, 1] weights (sum 4.5) with no rounding at all."""
        rotations = [1, 0.25, 2, 0.25, 1]
        fig, ani = self._build(rotations=rotations, duration=18, frame_rate=10)
        morph_state = ani._args[0]
        frame_counts = morph_state['frame_counts']
        total_frames = sum(frame_counts)
        ax = fig.axes[0]

        azims = []
        for k in range(total_frames):
            ani._func(k, *ani._args)
            azims.append(ax.azim)
        deltas = np.diff(azims)
        expected = 360.0 * sum(rotations) / total_frames
        np.testing.assert_allclose(deltas, expected, atol=0.5)

    def test_static_untagged_dataset_present_every_frame(self):
        """M4 visual-review fix: with mixed tagging, the untagged (static
        backdrop) dataset's Line3D must be initialized with -- and keep --
        its FULL point count at every frame, not just the first point.
        `update_morph` never touches untagged lines, so whatever they are
        initialized with is what stays on screen for the whole animation;
        before the fix they were initialized (like every other dataset's
        line) with only `dat[0:1, ...]`, i.e. a single point, so the
        static backdrop was invisible in practice (a 1-point "cloud")."""
        n_points = 30
        fig, ani = self._build(k=4, animate=['morph', 'morph', 'morph', None],
                               legend=True)
        fig.canvas.draw()
        ax = fig.axes[0]
        static_line = ax.lines[3]
        assert static_line.get_visible()

        morph_state = ani._args[0]
        total_frames = sum(morph_state['frame_counts'])
        frames = [0, total_frames // 2, total_frames - 1]

        reference = None
        for k in frames:
            ani._func(k, *ani._args)
            xs, ys, zs = static_line.get_data_3d()
            assert len(xs) == n_points, (
                f"frame {k}: untagged dataset's Line3D has {len(xs)} "
                f"points, expected the full {n_points}"
            )
            current = np.stack([xs, ys, zs], axis=1)
            if reference is None:
                reference = current
            else:
                np.testing.assert_allclose(reference, current)
            assert static_line.get_visible()

        # legend entries: untagged datasets keep their own legend entry
        # (this must survive the fix -- only the DATA of the line changes,
        # never its label/visibility bookkeeping)
        legend = ax.get_legend()
        assert legend is not None
        legend_labels = [t.get_text() for t in legend.get_texts()]
        assert len(legend_labels) == 4
        assert '4' in legend_labels

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


class TestMplFullSampleMorphVisibility:
    """Numerical evidence #4 (maintainer spec): with 3 UNEQUAL-size
    datasets (100, 250, 40), the target count `n` is 250 (the largest);
    the DRAWN artist must show only each dataset's own `m_k` true points
    during that dataset's hold, and all `n` points (both endpoints'
    duplicates included) during a mid-morph frame."""

    def _build(self):
        rng = np.random.default_rng(11)
        data = [rng.standard_normal((100, 3)),
                rng.standard_normal((250, 3)) + 8.0,
                rng.standard_normal((40, 3)) + 16.0]
        fig, ani = hyp.plot(data, '.', animate='morph', show=False,
                            duration=2, frame_rate=10)
        return fig, ani

    def test_hold_frame_visible_count_equals_own_dataset_size(self):
        fig, ani = self._build()
        morph_state = ani._args[0]
        frame_counts = morph_state['frame_counts']
        assert len(frame_counts) == 5  # 3 datasets -> 5 segments

        hold_frames = [0, sum(frame_counts[:2]), sum(frame_counts[:4])]
        sizes = [100, 250, 40]
        for frame, m in zip(hold_frames, sizes):
            ani._func(frame, *ani._args)
            xs, ys, zs = morph_state['artist'].get_data_3d()
            assert len(xs) == m, f"hold frame {frame}: {len(xs)} != {m}"

    def test_mid_morph_frame_visible_count_equals_n(self):
        fig, ani = self._build()
        morph_state = ani._args[0]
        frame_counts = morph_state['frame_counts']
        mid = frame_counts[0] + frame_counts[1] // 2
        ani._func(mid, *ani._args)
        xs, ys, zs = morph_state['artist'].get_data_3d()
        assert len(xs) == 250  # n = the largest dataset's own count


class TestAlphaCompositingEquivalence:
    """Numerical evidence #5 (the maintainer's STATED REASON for hiding
    duplicated points at holds): rasterizing a hold frame with alpha<1
    markers must be PIXEL-IDENTICAL (tiny AA tolerance) to rasterizing a
    plain static `hyp.plot` of that dataset's own true points, same
    style/camera. Uses dataset 0, which `sample_and_match_clouds` never
    reorders (it's the first cloud in the chain), so its hold-frame
    points are the LITERAL, identically-ordered `data[0]` array."""

    def test_hold_frame_matches_plain_plot_pixel_for_pixel(self):
        rng = np.random.default_rng(12)
        data = [rng.standard_normal((100, 3)),
                rng.standard_normal((250, 3)) + 8.0,
                rng.standard_normal((40, 3)) + 16.0]
        # a bare color STRING (not an RGB tuple) avoids the ambiguity of a
        # 3-element tuple being broadcast as "one color per dataset" for
        # the 3-dataset morph call below (an RGB tuple happens to have the
        # same length as the dataset count here).
        color = 'crimson'
        zoom = 1

        fig, ani = hyp.plot(data, '.', animate='morph', show=False,
                            duration=2, frame_rate=10, color=color,
                            markersize=6, zoom=zoom)
        morph_state = ani._args[0]
        morph_state['artist'].set_alpha(0.5)
        ani._func(0, *ani._args)  # frame 0: hold on dataset 0, dups hidden
        if not hasattr(fig.canvas, 'buffer_rgba'):  # CI backend fallback
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            FigureCanvasAgg(fig)
        fig.canvas.draw()
        anim_rgba = np.asarray(fig.canvas.buffer_rgba()).copy()

        # Plain STATIC plot of the SAME 3-dataset list (so the normalize/
        # reduce/align pipeline -- which runs identically regardless of
        # `animate=` -- produces IDENTICAL processed coordinates for
        # dataset 0 as the morph call above), with datasets 1/2's own
        # lines hidden so only dataset 0's true points are visible -- an
        # independent, from-scratch render of "just plot dataset 0's own
        # points, plain." The static and animated 3-D code paths differ
        # slightly in axes positioning/box-aspect zoom by default, so both
        # are pinned to the SAME values here (matching what
        # `animate_plot3D` sets up internally) rather than relying on
        # coincidental defaults.
        fig2 = hyp.plot(data, '.', show=False, color=color, markersize=6)
        ax2 = fig2.axes[0]
        lines2 = ax2.get_lines()
        lines2[1].set_visible(False)
        lines2[2].set_visible(False)
        lines2[0].set_alpha(0.5)
        ax2.set_position([0.0, 0.0, 1.0, 1.0])
        ax2.set_box_aspect(None, zoom=_anim_box_zoom(zoom))
        fig2.set_size_inches(fig.get_size_inches())
        fig2.set_dpi(fig.dpi)
        # static show=False figures get closed (GH #148); on matplotlib 3.11
        # (CI) close resets the canvas to FigureCanvasBase, which cannot
        # rasterize -- attach a real Agg canvas explicitly before drawing
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        FigureCanvasAgg(fig2)
        fig2.canvas.draw()
        plain_rgba = np.asarray(fig2.canvas.buffer_rgba()).copy()

        assert anim_rgba.shape == plain_rgba.shape
        match = (anim_rgba == plain_rgba).mean()
        assert match >= 0.999, f"only {match:.4%} of pixels matched"

        plt.close(fig)
        plt.close(fig2)


# ---------------------------------------------------------------------------
# M3b box-containment fix: the axes cube/limits (mpl) and scene ranges
# (plotly) must be sized from the SAME `sampled` clouds `update_morph`/
# `_add_animation` actually draw each frame (plus their union), not from
# each dataset's full-order cloud. A cube-corner-like cloud's convex hull
# is degenerate enough (many coplanar points) that smooth_hull_3d's
# ConvexHull/Taubin pipeline is not invariant to input row order, so sizing
# from a differently-ordered copy of the exact same points under-covers
# the actual per-frame mesh -- confirmed visually in
# docs/images/v1.0-seven-features/surface_morph_frames.png (the "hold:
# cube" panel's hull spilled past the drawn wireframe box) and empirically
# (the same 400 cube-shaped points, reordered, gave a mesh whose max
# |vertex| exceeded the containment margin computed from the original
# order).
# ---------------------------------------------------------------------------

def _cube_corners():
    """The 8 corners of the [-1, 1]^3 cube -- the degenerate, many-
    coplanar-point shape that exposed the box-containment bug (this is
    exactly the failing case: a cloud that spans the full [-1, 1] cube)."""
    return np.array([[sx, sy, sz]
                     for sx in (-1.0, 1.0)
                     for sy in (-1.0, 1.0)
                     for sz in (-1.0, 1.0)])


def _sphere_cloud(n=64, seed=7):
    """A different, non-degenerate blob on the unit ball -- the morph
    partner for `_cube_corners`."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * rng.uniform(0.3, 1.0, size=(n, 1))


class TestBoxContainmentUnionHull:
    """Regression for the M3b box-containment bug: the cube-corner cloud
    is dataset 0 of a 2-dataset morph (hold: cube-corners -> morph ->
    hold: sphere-blob); both hold frames and mid-morph frames must keep
    every surface vertex inside the drawn box on both backends."""

    def _data(self):
        return [_cube_corners(), _sphere_cloud()]

    def test_mpl_hold_and_mid_morph_within_axes_cube(self):
        fig, ani = hyp.plot(self._data(), '.', animate='morph',
                            surface=True, duration=2, frame_rate=10,
                            show=False)
        ax = fig.axes[0]
        morph_state = ani._args[0]
        cube_scale = ani._args[1]
        frame_counts = morph_state['frame_counts']
        assert len(frame_counts) == 3  # 2 datasets -> 3 segments

        xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
        assert xlim == pytest.approx((-cube_scale, cube_scale))
        assert ylim == pytest.approx((-cube_scale, cube_scale))
        assert zlim == pytest.approx((-cube_scale, cube_scale))

        mid_morph = frame_counts[0] + frame_counts[1] // 2
        hold_cube, hold_sphere = 0, frame_counts[0] + frame_counts[1]
        for frame in [hold_cube, mid_morph, hold_sphere]:
            ani._func(frame, *ani._args)

            colls = [c for c in ax.collections
                    if isinstance(c, Poly3DCollection)]
            assert colls, f"expected a surface mesh at frame {frame}"
            for c in colls:
                verts = _poly3d_verts(c)
                if len(verts) == 0:
                    continue
                assert verts[:, 0].min() >= xlim[0] - 1e-6
                assert verts[:, 0].max() <= xlim[1] + 1e-6
                assert verts[:, 1].min() >= ylim[0] - 1e-6
                assert verts[:, 1].max() <= ylim[1] + 1e-6
                assert verts[:, 2].min() >= zlim[0] - 1e-6
                assert verts[:, 2].max() <= zlim[1] + 1e-6

            # the drawn wireframe cube itself is sized to the SAME bound
            planes = ani._func.planes
            assert planes
            for plane in planes:
                seg_pts = np.vstack([np.asarray(s, dtype=float)
                                     for s in plane._segments3d])
                assert np.abs(seg_pts).max() == pytest.approx(cube_scale)

    def test_plotly_mesh_within_scene_ranges_every_frame(self):
        fig = hyp.plot(self._data(), '.', animate='morph', surface=True,
                       backend='plotly', duration=2, frame_rate=10,
                       show=False)
        xr = fig.layout.scene.xaxis.range
        yr = fig.layout.scene.yaxis.range
        zr = fig.layout.scene.zaxis.range

        mesh_seen = False
        for frame in fig.frames:
            for trace in frame.data:
                if trace.type != 'mesh3d':
                    continue
                x, y, z = (np.asarray(trace.x), np.asarray(trace.y),
                          np.asarray(trace.z))
                if x.size == 0:
                    continue
                mesh_seen = True
                assert x.min() >= xr[0] - 1e-6 and x.max() <= xr[1] + 1e-6
                assert y.min() >= yr[0] - 1e-6 and y.max() <= yr[1] + 1e-6
                assert z.min() >= zr[0] - 1e-6 and z.max() <= zr[1] + 1e-6
        assert mesh_seen, "expected at least one frame with a mesh3d trace"


# ---------------------------------------------------------------------------
# M4 review fix: surface=True + animate='morph' must size the axes box from
# the sampled+union meshes only (M3b machinery) -- NEVER from a morph-tagged
# dataset's full raw cloud, which was both redundant with that sizing and a
# documented OOM/hang cliff on large (tens-of-thousands-point) clouds, since
# `smooth_hull_3d` (especially its `points_enclosed` Delaunay containment
# check) scales with the FULL point count, not the much smaller
# `morph_samples` cap.
# ---------------------------------------------------------------------------

class TestMorphSurfaceSizingSkipsFullCloud:
    def test_large_cloud_completes_quickly_and_never_builds_full_mesh(
        self, monkeypatch
    ):
        """A 20,000-point Gaussian blob morphing against a small second
        dataset, with surface=True, must (a) finish setting up the
        animation quickly (well under the ~minutes-to-hang cliff a full-
        cloud ConvexHull/smooth/containment pass on 20k points can trigger)
        and (b) never call `build_mesh_3d` with anywhere near 20,000
        points -- every call must be capped at `morph_samples` (or the
        small union of two sampled clouds), proving the axes-box sizing
        never touches the raw cloud."""
        import time
        from hypertools.plot import matplotlib_backend as mb

        rng = np.random.default_rng(0)
        big_cheap_blob = rng.standard_normal((20_000, 3))
        small_blob = rng.standard_normal((30, 3)) + 6.0
        morph_samples = 200

        seen_point_counts = []
        real_build_mesh_3d = mb.build_mesh_3d

        def _spy_build_mesh_3d(points, *args, **kwargs):
            seen_point_counts.append(len(points))
            return real_build_mesh_3d(points, *args, **kwargs)

        monkeypatch.setattr(mb, "build_mesh_3d", _spy_build_mesh_3d)

        start = time.monotonic()
        fig, ani = hyp.plot(
            [big_cheap_blob, small_blob], '.', animate='morph',
            surface=True, morph_samples=morph_samples,
            duration=1, frame_rate=5, show=False,
        )
        elapsed = time.monotonic() - start

        assert elapsed < 20, (
            f"surface=True + animate='morph' setup took {elapsed:.1f}s "
            "for a 20,000-point cloud (expected <20s) -- likely rebuilding "
            "a full-cloud mesh solely for axes-box sizing"
        )
        assert seen_point_counts, "expected at least one build_mesh_3d call"
        # the largest possible legitimate call is the union of both sampled
        # clouds (<= 2 * morph_samples); 20,000 would blow well past this
        assert max(seen_point_counts) <= 2 * morph_samples, (
            f"build_mesh_3d was called with up to {max(seen_point_counts)} "
            f"points (> 2 * morph_samples={morph_samples}) -- the full "
            "20,000-point raw cloud was built for sizing instead of only "
            "the sampled/union meshes"
        )


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
        counts, _, azims = morph.morph_schedule(
            3, 20, [1, 0.25, 2, 0.25, 1], -60)

        for k in [0, counts[0], sum(counts[:2]), 19]:
            expected_eye = _camera_eye(10, azims[k], r=_anim_zoom_r(1))
            actual_eye = fig.frames[k].layout.scene.camera.eye
            assert actual_eye.x == pytest.approx(expected_eye['x'], abs=1e-6)
            assert actual_eye.y == pytest.approx(expected_eye['y'], abs=1e-6)
            assert actual_eye.z == pytest.approx(expected_eye['z'], abs=1e-6)

    def test_camera_eye_angular_deltas_equal_across_frames(self):
        """Maintainer request (2026-07-06): constant rotation speed. The
        camera eye's angular position (recovered via atan2 of its x/y,
        which tracks azimuth on the unit circle at a fixed elevation/zoom)
        must advance by the SAME angle every single frame across the whole
        plotly animation, not just at the sampled boundary frames. Uses
        180 frames (duration=18 * frame_rate=10), which divides the
        [1, 0.25, 2, 0.25, 1] weights (sum 4.5) with no rounding at all."""
        rotations = [1, 0.25, 2, 0.25, 1]
        fig = self._build(rotations=rotations, duration=18, frame_rate=10)
        eyes = [f.layout.scene.camera.eye for f in fig.frames]
        angles = np.unwrap([np.arctan2(e.y, e.x) for e in eyes])
        deltas = np.degrees(np.diff(angles))
        expected = 360.0 * sum(rotations) / len(fig.frames)
        np.testing.assert_allclose(deltas, expected, atol=0.5)

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


# ---------------------------------------------------------------------------
# Numerical evidence #4/#6 (maintainer spec), plotly backend: full-sample
# morphs with 3 UNEQUAL-size datasets -- each frame's Scatter3d trace must
# carry exactly `m_k` points at dataset `k`'s own hold and exactly `n`
# points (the largest dataset's count) mid-morph.
# ---------------------------------------------------------------------------

class TestPlotlyFullSampleMorphVisibility:
    def _build(self):
        rng = np.random.default_rng(13)
        data = [rng.standard_normal((100, 3)),
                rng.standard_normal((250, 3)) + 8.0,
                rng.standard_normal((40, 3)) + 16.0]
        return hyp.plot(data, '.', animate='morph', backend='plotly',
                        show=False, duration=2, frame_rate=10)

    def test_frame_point_counts_match_hold_and_mid_morph_expectations(self):
        fig = self._build()
        frame_counts, _, _ = morph.morph_schedule(3, len(fig.frames), 1, -60)
        assert len(frame_counts) == 5

        hold_frames_and_sizes = [
            (0, 100),
            (sum(frame_counts[:2]), 250),
            (sum(frame_counts[:4]), 40),
        ]
        for frame_idx, m in hold_frames_and_sizes:
            trace = fig.frames[frame_idx].data[0]
            assert len(trace.x) == m, (
                f"hold frame {frame_idx}: {len(trace.x)} points, expected {m}"
            )

        mid = frame_counts[0] + frame_counts[1] // 2
        mid_trace = fig.frames[mid].data[0]
        assert len(mid_trace.x) == 250  # n = the largest dataset's count

    def test_morph_frame_includes_duplicate_points(self):
        """A mid-morph frame must include ALL `n` points -- i.e. it must
        NOT match either endpoint dataset's own (smaller) true-point
        count, proving duplicates are shown (not filtered) while
        traveling."""
        fig = self._build()
        frame_counts, _, _ = morph.morph_schedule(3, len(fig.frames), 1, -60)
        mid = frame_counts[0] + frame_counts[1] // 2
        mid_trace = fig.frames[mid].data[0]
        assert len(mid_trace.x) == 250
        assert len(mid_trace.x) not in (100, 40)
