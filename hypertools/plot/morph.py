#!/usr/bin/env python
"""``animate='morph'``: Hungarian-matched point-cloud morphs between
datasets (maintainer request, 2026-07-06).

Lifted from ``examples/plot_shape_morph.py`` (the original hand-rolled
demo) into a reusable, backend-agnostic module: sample an equal number of
points from each morphing dataset, chain-match consecutive clouds with the
Hungarian algorithm (``scipy.optimize.linear_sum_assignment`` on
``scipy.spatial.distance.cdist``) so each point travels the shortest total
distance to its partner in the next cloud, then ease between clouds with
smoothstep interpolation on a hold/morph/hold/... frame schedule ([hold_1,
morph_1->2, hold_2, ..., hold_N] -- ``2*N - 1`` segments for ``N``
datasets). Both ``hypertools.plot.matplotlib_backend`` and
``hypertools.plot.plotly_backend`` build their ``animate='morph'`` frames
from these same helpers, so the two backends stay in lockstep.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

__all__ = [
    "smoothstep",
    "sample_and_match_clouds",
    "segment_frame_counts",
    "frame_to_segment",
    "morph_positions",
    "interpolate_color",
    "morph_color",
    "resolve_morph_rotations",
    "segment_azimuths",
]


def smoothstep(t):
    """Smoothstep easing: ``3t^2 - 2t^3``, clipped to ``t in [0, 1]``.
    Flat (zero-slope) at both endpoints -- morphs ease in and out rather
    than moving at a constant rate."""
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def sample_and_match_clouds(clouds, morph_samples=None, seed=0):
    """Sample an equal-sized subset of points from every cloud in `clouds`
    (no replacement, seeded) and chain-match consecutive clouds with the
    Hungarian algorithm, exactly as ``examples/plot_shape_morph.py`` did by
    hand.

    Parameters
    ----------
    clouds : sequence of (n_i, d) array-like
        One point cloud per morphing dataset, in morph order. Must contain
        at least 2 clouds.
    morph_samples : int or None, optional
        The number of points sampled from EVERY cloud (all clouds are
        sampled down to the SAME count, so points can be matched 1-to-1).
        Default (``None``): ``min(smallest cloud's point count, 1000)`` --
        capped at 1000 so the Hungarian assignment's ``O(n^3)`` cost stays
        tractable for large datasets. If given, the effective count is
        ``min(morph_samples, smallest cloud's point count)`` -- a count
        larger than the smallest cloud is silently capped, never padded.
    seed : int, optional
        Seed for the sampling RNG (``numpy.random.default_rng``), default 0
        -- deterministic and reproducible across calls.

    Returns
    -------
    list of (n_points, d) ndarray
        One sampled+matched cloud per input, same length as `clouds`.
        ``sampled[0]`` is sampled but unmatched (nothing precedes it);
        every subsequent ``sampled[k]`` is REORDERED so row ``i`` is the
        optimal (minimum total travel distance) partner of
        ``sampled[k - 1][i]``.
    """
    if len(clouds) < 2:
        raise ValueError(
            f"sample_and_match_clouds needs at least 2 clouds; got "
            f"{len(clouds)}"
        )
    clouds = [np.atleast_2d(np.asarray(c, dtype=np.float64)) for c in clouds]
    min_count = min(c.shape[0] for c in clouds)
    cap = 1000 if morph_samples is None else int(morph_samples)
    n_points = max(1, min(min_count, cap))

    rng = np.random.default_rng(seed)
    sampled = [
        c[rng.choice(c.shape[0], size=n_points, replace=False)]
        for c in clouds
    ]

    for i in range(len(sampled) - 1):
        cost = cdist(sampled[i], sampled[i + 1])
        _, col_ind = linear_sum_assignment(cost)
        sampled[i + 1] = sampled[i + 1][col_ind]
    return sampled


def segment_frame_counts(n_datasets, total_frames):
    """Split `total_frames` as evenly as possible across the ``2 *
    n_datasets - 1`` hold/morph segments (``[hold_1, morph_1->2, hold_2,
    ..., hold_N]``), any remainder going to the earliest segments.

    Returns a list of ``2 * n_datasets - 1`` positive ints summing to
    exactly ``max(2 * n_datasets - 1, total_frames)`` (there is always at
    least 1 frame per segment, even if `total_frames` is smaller than the
    segment count).
    """
    if n_datasets < 2:
        raise ValueError(
            f"animate='morph' needs at least 2 datasets to morph between; "
            f"got {n_datasets}"
        )
    n_segments = 2 * n_datasets - 1
    total_frames = max(n_segments, int(total_frames))
    base, rem = divmod(total_frames, n_segments)
    return [base + (1 if i < rem else 0) for i in range(n_segments)]


def frame_to_segment(frame_counts, frame):
    """Map a global 0-indexed animation `frame` to ``(segment_idx,
    step, n_steps)``: which segment it falls in, its LOCAL step within
    that segment, and that segment's total frame count. Frames at or past
    the end of the schedule clamp to the final step of the final segment.
    """
    remaining = int(frame)
    for i, n in enumerate(frame_counts):
        if remaining < n:
            return i, remaining, n
        remaining -= n
    last = len(frame_counts) - 1
    return last, frame_counts[last] - 1, frame_counts[last]


def morph_positions(sampled, seg_idx, step, n_steps):
    """Point positions for segment `seg_idx` at local `step` (of `n_steps`
    total in that segment).

    Even `seg_idx` (0, 2, 4, ...) are HOLDS: the corresponding dataset
    ``sampled[seg_idx // 2]`` is returned unchanged (matches every frame).
    Odd `seg_idx` are MORPHS: ``sampled[k]`` eases into ``sampled[k + 1]``
    (``k = seg_idx // 2``) via :func:`smoothstep`, ``t=0`` exactly
    reproducing ``sampled[k]`` and ``t=1`` exactly reproducing
    ``sampled[k + 1]``.
    """
    if seg_idx % 2 == 0:
        return sampled[seg_idx // 2]
    k = seg_idx // 2
    t = float(smoothstep(step / max(1, n_steps - 1)))
    return (1.0 - t) * sampled[k] + t * sampled[k + 1]


def interpolate_color(color_a, color_b, t):
    """Linear (unclipped) RGB interpolation between two ``(r, g, b)``
    colors (each component in ``[0, 1]``): ``t=0`` -> `color_a`, ``t=1`` ->
    `color_b`."""
    a = np.asarray(color_a, dtype=np.float64)
    b = np.asarray(color_b, dtype=np.float64)
    return tuple((1.0 - t) * a + t * b)


def morph_color(colors, seg_idx, step, n_steps):
    """Drawn color for segment `seg_idx` at local `step` (of `n_steps`),
    on the SAME schedule as :func:`morph_positions`: holds are the
    corresponding dataset's own solid color; morphs RGB-lerp (smoothstep-
    eased, matching the position easing) between the two datasets'
    colors."""
    if seg_idx % 2 == 0:
        return tuple(colors[seg_idx // 2])
    k = seg_idx // 2
    t = float(smoothstep(step / max(1, n_steps - 1)))
    return interpolate_color(colors[k], colors[k + 1], t)


def resolve_morph_rotations(rotations, n_datasets):
    """Validate `rotations` for ``animate='morph'`` with `n_datasets`
    morphing datasets.

    A scalar is returned unchanged (a single float): the TOTAL number of
    camera rotations spread uniformly over the whole animation, exactly
    like every other ``animate`` style (see
    :func:`hypertools.plot.morph.segment_azimuths`).

    A list/tuple must have exactly ``2 * n_datasets - 1`` entries (one per
    hold/morph segment: ``[hold_1, morph_1->2, hold_2, ..., hold_N]``) --
    each segment's OWN number of rotations, spread uniformly over that
    segment's own frames.

    Raises
    ------
    ValueError
        A list/tuple was given with the wrong length (names the expected
        length in the message).
    """
    if isinstance(rotations, (list, tuple)):
        n_segments = 2 * n_datasets - 1
        if len(rotations) != n_segments:
            raise ValueError(
                f"rotations list has {len(rotations)} entries but "
                f"animate='morph' with {n_datasets} morphing datasets "
                f"needs exactly {n_segments} (2 * n_datasets - 1: "
                "[hold_1, morph_1->2, hold_2, ..., hold_N])"
            )
        return [float(r) for r in rotations]
    return float(rotations)


def segment_azimuths(frame_counts, rotations, azim0):
    """Per-GLOBAL-frame camera azimuth (degrees) for ``animate='morph'``,
    one entry per frame across the whole ``sum(frame_counts)``-frame
    animation.

    If `rotations` is a scalar, this is IDENTICAL to every other
    ``animate`` style's pacing: `rotations` total turns spread uniformly
    over the whole animation (``azim0 + 360 * rotations * frame /
    total_frames``), ignoring segment boundaries entirely.

    If `rotations` is a list (length ``len(frame_counts)``, validated by
    :func:`resolve_morph_rotations`), segment `k`'s `rotations[k]` turns
    are spread uniformly over THAT segment's own ``frame_counts[k]``
    frames, and each segment's starting angle continues exactly where the
    previous segment's full (unswept-remainder) rotation would have
    landed -- i.e. segment `k` starts at ``azim0 + 360 *
    sum(rotations[:k])`` -- so the camera never jumps at a segment
    boundary.
    """
    total_frames = sum(frame_counts)
    if not isinstance(rotations, (list, tuple)):
        return [azim0 + 360.0 * rotations * k / total_frames
                for k in range(total_frames)]

    if len(rotations) != len(frame_counts):
        raise ValueError(
            f"rotations list has {len(rotations)} entries but "
            f"frame_counts has {len(frame_counts)} segments"
        )
    azims = []
    current = float(azim0)
    for n_frames, rot in zip(frame_counts, rotations):
        m = max(1, n_frames)
        for step in range(n_frames):
            azims.append(current + 360.0 * rot * step / m)
        current += 360.0 * rot
    return azims
