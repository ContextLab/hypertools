#!/usr/bin/env python
"""``animate='morph'``: Hungarian-matched point-cloud morphs between
datasets (maintainer request, 2026-07-06).

Lifted from the shape-morph gallery example (now
``examples/animate_morph_zoo.py``; the original hand-rolled
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

Full-sample morphs (maintainer request, 2026-07-06 follow-up): earlier
versions of this module sampled every dataset down to the SMALLEST
morphing dataset's point count so clouds could be matched 1-to-1. Every
dataset now keeps its FULL point count instead: the target count ``n`` is
the LARGEST (post-`morph_samples`-cap) dataset's size, and any dataset
with ``m < n`` points is padded up to ``n`` by duplicating ``n - m`` of
its OWN points (chosen at random, seeded). No real data point is dropped
by the padding step itself. Whether one is dropped EARLIER, by sampling,
is the caller's documented choice: with an explicit ``morph_samples=``,
or with ``simplify=True`` (the default) over clouds larger than
``plot.MORPH_SAMPLES_REQUIRED_ABOVE`` = 2000 points, each cloud is first
downsampled to that cap -- silently, because an uncapped Hungarian match
over ~30k-point clouds does not finish (measured: still running after 10
minutes; capped at 2000 it renders in 8.2 s). With ``simplify=False`` and
no ``morph_samples=``, the original guarantee holds absolutely at any
size: every dataset keeps its FULL point count, and an intractable morph
raises rather than approximating. The duplicated rows are tracked per
dataset (see
:func:`sample_and_match_clouds`'s ``dup_masks`` return value) and hidden
during that dataset's own HOLD frames (see :func:`morph_visible_mask`) so
alpha-compositing a hold frame looks identical to a plain plot of that
dataset's true points; they are shown, like every other point, during
MORPH frames.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

__all__ = [
    "smoothstep",
    "sample_and_match_clouds",
    "morph_visible_mask",
    "segment_frame_counts",
    "frame_to_segment",
    "morph_positions",
    "interpolate_color",
    "morph_color",
    "morph_alpha",
    "resolve_morph_rotations",
    "segment_azimuths",
    "morph_schedule",
    "ZERO_ROTATION_FLOOR",
    "MORPH_SURFACE_SIZING_MARGIN",
]

#: Constant-rotation-speed fix (maintainer request, 2026-07-06): when a
#: LIST of per-segment `rotations` is given to ``animate='morph'``, each
#: segment's screen time is made PROPORTIONAL to its own rotation count so
#: the camera's angular speed (degrees/frame) is identical everywhere --
#: more rotations means more time spent on that part of the animation,
#: never faster spinning. A segment with `rotations[k] == 0` would
#: otherwise get ZERO proportional share (and thus collapse to the
#: minimum/instant), so its EFFECTIVE weight for frame allocation is
#: floored at `ZERO_ROTATION_FLOOR` turns -- it still gets exactly the
#: screen time a 0.1-rotation segment would, purely so it stays visible;
#: its actual camera motion is still governed by its real `rotations[k]`
#: (0, i.e. the camera does not move during that segment). See
#: :func:`segment_frame_counts`.
ZERO_ROTATION_FLOOR = 0.1

#: Extra box-containment safety margin (maintainer request, 2026-07-06
#: follow-up: full-sample duplication) for ``surface=True`` + ``animate=
#: 'morph'`` sizing ONLY, multiplied on top of `surface_cube_scale`'s own
#: (2%) margin. `matplotlib_backend.animate_plot3D`/`plotly_backend
#: ._add_animation` size the axes box/scene ranges once, up front, from
#: meshes built from the two ENDPOINT `sampled` clouds plus their flat
#: union (see the "M3b" sizing notes in both backends) -- a cheap,
#: normally-safe bound, since every interpolated mid-morph point is a
#: convex combination of union points. But `smooth_hull_3d`'s Taubin-
#: smoothing + containment-regrow pipeline is NOT simply monotonic in its
#: input point set: a mid-morph cloud built from a HEAVILY duplicated
#: dataset (e.g. a small cloud padded far past its own size, see
#: `sample_and_match_clouds`) can, after its own smoothing pass, need MORE
#: containment growth than either endpoint's or the union's mesh did --
#: verified empirically (a 2-dataset morph, an 8-point cube-corner cloud
#: padded up to a 64-point target against a genuine 64-point sphere cloud,
#: `surface=True`): the endpoint+union sizing bound alone under-covered
#: the worst actual mid-morph frame by up to ~9%. This fixed extra margin
#: is a generous, empirically-validated buffer over that observed
#: worst case.
MORPH_SURFACE_SIZING_MARGIN = 1.2


def smoothstep(t):
    """Smoothstep easing: ``3t^2 - 2t^3``, clipped to ``t in [0, 1]``.
    Flat (zero-slope) at both endpoints -- morphs ease in and out rather
    than moving at a constant rate."""
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def sample_and_match_clouds(clouds, morph_samples=None, seed=0):
    """Pad every cloud in `clouds` up to the LARGEST (post-`morph_samples`-
    cap) cloud's point count by duplicating points at random (seeded), then
    chain-match consecutive clouds with the Hungarian algorithm.

    Maintainer request (2026-07-06): earlier versions of this function
    SHRANK every cloud down to the smallest morphing dataset's point count.
    Every dataset now keeps its own FULL point count instead -- no real
    point is ever dropped. This works by duplicating points on the smaller
    datasets: the target count ``n`` is the LARGEST dataset's size (after
    the optional `morph_samples` cap below), and a dataset with ``m < n``
    points is padded with ``n - m`` extra copies of its OWN points, chosen
    at random. If ``n - m > m`` (more duplicates are needed than the
    dataset has real points -- e.g. a 10-point dataset padded up to a
    25-point target), duplicates are drawn WITH replacement (some real
    points get copied more than once); otherwise they are drawn WITHOUT
    replacement (every duplicate is a distinct real point, just also
    appearing once more).

    Parameters
    ----------
    clouds : sequence of (n_i, d) array-like
        One point cloud per morphing dataset, in morph order. Must contain
        at least 2 clouds.
    morph_samples : int or None, optional
        An OPTIONAL cap on cloud size, applied BEFORE the duplication
        logic: any cloud larger than `morph_samples` is first downsampled
        (without replacement, seeded) to exactly `morph_samples` points.
        Default (``None``): no cap -- the target count is simply the
        largest cloud's own (real) point count. Since the Hungarian
        assignment's cost is roughly ``O(n^3)``, `morph_samples` is
        recommended for clouds larger than ~2000 points (e.g.
        ``morph_samples=1000``) to keep matching tractable; the default,
        uncapped behavior can be slow -- or exhaust memory -- for very
        large datasets.
    seed : int, optional
        Seed for the sampling RNG (``numpy.random.default_rng``), default 0
        -- deterministic and reproducible across calls.

    Returns
    -------
    (sampled, dup_masks) : (list of (n, d) ndarray, list of (n,) bool ndarray)
        `sampled`: one padded+matched cloud per input, same length as
        `clouds`, each with exactly `n` rows (`n` = the largest capped
        cloud's point count). ``sampled[0]`` is padded but unmatched
        (nothing precedes it); every subsequent ``sampled[k]`` is
        REORDERED so row ``i`` is the optimal (minimum total travel
        distance) partner of ``sampled[k - 1][i]`` -- this reordering
        never changes which points are duplicates, only their row
        position (`dup_masks` is permuted identically).
        `dup_masks`: one ``bool`` mask of length `n` per input, aligned
        row-for-row with `sampled`. ``dup_masks[k][i] is True`` iff
        ``sampled[k][i]`` is a DUPLICATE of another row in ``sampled[k]``
        (i.e. not that dataset's own original data) -- exactly ``n -
        m_k`` entries are `True`, where ``m_k`` is dataset `k`'s own
        (capped) point count. All-`False` for any dataset whose own count
        already equals `n` (the largest dataset has no duplicates at
        all).
    """
    if len(clouds) < 2:
        raise ValueError(
            f"sample_and_match_clouds needs at least 2 clouds; got "
            f"{len(clouds)}"
        )
    clouds = [np.atleast_2d(np.asarray(c, dtype=np.float64)) for c in clouds]
    rng = np.random.default_rng(seed)

    cap = None if morph_samples is None else int(morph_samples)
    capped = []
    for c in clouds:
        if cap is not None and c.shape[0] > cap:
            idx = rng.choice(c.shape[0], size=cap, replace=False)
            capped.append(c[idx])
        else:
            capped.append(c)

    n_points = max(1, max(c.shape[0] for c in capped))

    full = []
    dup_masks = []
    for c in capped:
        m = c.shape[0]
        mask = np.zeros(n_points, dtype=bool)
        if m >= n_points:
            full.append(c.copy())
        else:
            need = n_points - m
            replace = need > m
            dup_idx = rng.choice(m, size=need, replace=replace)
            full.append(np.vstack([c, c[dup_idx]]))
            mask[m:] = True
        dup_masks.append(mask)

    for i in range(len(full) - 1):
        cost = cdist(full[i], full[i + 1])
        _, col_ind = linear_sum_assignment(cost)
        full[i + 1] = full[i + 1][col_ind]
        dup_masks[i + 1] = dup_masks[i + 1][col_ind]
    return full, dup_masks


def morph_visible_mask(dup_masks, seg_idx):
    """Boolean mask (length `n`, or ``None``) of which points to HIDE for
    schedule segment `seg_idx`, given `dup_masks` (`sample_and_match_clouds`'s
    second return value, or ``None``).

    On a HOLD segment (`seg_idx` even), the currently-held dataset's own
    duplicated rows (``dup_masks[seg_idx // 2]``) are hidden -- so drawing
    only the non-hidden rows reproduces a plain plot of that dataset's true
    (non-duplicated) points exactly, which is what makes alpha-compositing
    (e.g. semi-transparent markers) behave correctly at a hold. On a MORPH
    segment (odd), nothing is hidden: every one of the `n` points --
    including both endpoint datasets' duplicates -- is visible, since a
    morph is a continuous interpolation between the FULL `n`-point clouds.

    Returns ``None`` (meaning "hide nothing") if `dup_masks` is ``None``.
    """
    if dup_masks is None:
        return None
    if seg_idx % 2 == 0:
        return dup_masks[seg_idx // 2]
    return None


def _largest_remainder_alloc(weights, total):
    """Apportion the integer `total` across `weights` proportionally
    (Hamilton/largest-remainder method): each share is
    ``total * weights[i] / sum(weights)``, floored, then the leftover
    units (``total - sum(floors)``) go one-by-one to the segments with
    the largest fractional remainder (ties broken by earliest index).
    Sums to exactly `total`."""
    n = len(weights)
    wsum = sum(weights)
    raw = [total * w / wsum for w in weights]
    counts = [int(np.floor(r)) for r in raw]
    remainder = total - sum(counts)
    order = sorted(range(n), key=lambda i: (-(raw[i] - counts[i]), i))
    for i in order[:remainder]:
        counts[i] += 1
    return counts


def _proportional_frame_counts(rotations, total_frames):
    """Largest-remainder proportional split of `total_frames` across
    ``len(rotations)`` segments, weighted by each segment's EFFECTIVE
    rotation count (``max(r, ZERO_ROTATION_FLOOR)``), then topped up so
    every segment has at least 2 frames (taking the surplus from
    whichever segment currently has the most, one frame at a time --
    total stays exactly `total_frames`)."""
    n = len(rotations)
    total_frames = max(2 * n, int(total_frames))
    weights = [max(float(r), ZERO_ROTATION_FLOOR) for r in rotations]
    counts = _largest_remainder_alloc(weights, total_frames)

    while True:
        short = [i for i, c in enumerate(counts) if c < 2]
        if not short:
            break
        for i in short:
            j = max(range(n), key=lambda idx: counts[idx])
            counts[j] -= 1
            counts[i] += 1
    return counts


def segment_frame_counts(n_datasets, total_frames, rotations=None):
    """Split `total_frames` across the ``2 * n_datasets - 1`` hold/morph
    segments (``[hold_1, morph_1->2, hold_2, ..., hold_N]``).

    `rotations` is `resolve_morph_rotations`'s already-resolved return
    value (or ``None``):

    - Omitted, ``None``, or a SCALAR: split EVENLY, any remainder going to
      the earliest segments (unchanged since `animate='morph'`'s first
      release -- a scalar already means "spread uniformly over the whole
      animation in TIME", so equal segment durations already give
      constant angular speed by construction). Returns a list of ``2 *
      n_datasets - 1`` positive ints summing to exactly ``max(2 *
      n_datasets - 1, total_frames)`` (at least 1 frame per segment).

    - A LIST/tuple (length ``2 * n_datasets - 1``, one entry per
      hold/morph segment): split PROPORTIONALLY to each segment's own
      rotation count instead, via the largest-remainder method, so every
      segment plays at the SAME angular speed (degrees/frame) -- segment
      `k` gets ``total_frames * effective_r_k / sum(effective_r)`` frames,
      where ``effective_r_k = max(rotations[k], ZERO_ROTATION_FLOOR)``
      (see module docstring): a zero-rotation segment still gets the
      screen time a `ZERO_ROTATION_FLOOR`-rotation segment would, so it
      stays visible instead of collapsing to an instant. Every segment
      gets at least 2 frames regardless (`total_frames` is bumped up to
      ``2 * (2 * n_datasets - 1)`` if needed).

    Raises
    ------
    ValueError
        `n_datasets` < 2, or a `rotations` list/tuple with the wrong
        length (names the expected length in the message).
    """
    if n_datasets < 2:
        raise ValueError(
            f"animate='morph' needs at least 2 datasets to morph between; "
            f"got {n_datasets}"
        )
    n_segments = 2 * n_datasets - 1
    if isinstance(rotations, (list, tuple)):
        if len(rotations) != n_segments:
            raise ValueError(
                f"rotations list has {len(rotations)} entries but "
                f"animate='morph' with {n_datasets} morphing datasets "
                f"needs exactly {n_segments} (2 * n_datasets - 1: "
                "[hold_1, morph_1->2, hold_2, ..., hold_N])"
            )
        return _proportional_frame_counts(rotations, total_frames)
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


def morph_alpha(alphas, seg_idx, step, n_steps):
    """Drawn alpha (opacity) for segment `seg_idx` at local `step` (of
    `n_steps`), on the SAME schedule as :func:`morph_positions` and
    :func:`morph_color` (GH #284: ``alpha=`` reaches the traveling cloud).

    `alphas` holds one entry per morphing dataset, in morph order: the
    dataset's own ``alpha=`` (a float in ``[0, 1]``) or ``None`` when none
    was given. Returns ``None`` -- "leave the artist at its default" --
    when every entry is ``None``, so an animation that never asked for an
    alpha is drawn exactly as before. Otherwise a HOLD (even `seg_idx`)
    is the held dataset's own alpha, and a MORPH (odd) eases (smoothstep,
    matching the position/color easing) from the departing dataset's
    alpha to the arriving one's, an unset entry counting as opaque
    (``1.0``) -- the same rule every other animation style applies to a
    per-dataset ``alpha=`` list, restated for one artist that stands in
    for several datasets in turn.
    """
    if alphas is None or all(a is None for a in alphas):
        return None
    vals = [1.0 if a is None else float(a) for a in alphas]
    if seg_idx % 2 == 0:
        return vals[seg_idx // 2]
    k = seg_idx // 2
    t = float(smoothstep(step / max(1, n_steps - 1)))
    # `a + t * (b - a)` (not `(1 - t) * a + t * b`) so a scalar `alpha=`
    # (every entry equal) stays EXACTLY that value on every transition
    # frame instead of drifting by a float ulp.
    return vals[k] + t * (vals[k + 1] - vals[k])


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


def morph_schedule(n_datasets, total_frames, rotations, azim0):
    """Compute the ENTIRE ``animate='morph'`` per-frame schedule exactly
    ONCE: resolve `rotations` (:func:`resolve_morph_rotations`), allocate
    `frame_counts` (:func:`segment_frame_counts` -- equal for a scalar,
    proportional-to-rotation for a list, so angular speed stays constant
    either way), and expand the per-GLOBAL-frame azimuth track
    (:func:`segment_azimuths`).

    Both ``hypertools.plot.matplotlib_backend`` and
    ``hypertools.plot.plotly_backend`` call this SAME function for
    ``animate='morph'`` -- neither ever reassembles the schedule from the
    pieces itself -- so the two backends can never drift out of sync.

    Returns
    -------
    (frame_counts, rotations_resolved, azimuths)
        `frame_counts`: list of ``2 * n_datasets - 1`` ints summing to
        (approximately, see :func:`segment_frame_counts`) `total_frames`.
        `rotations_resolved`: `resolve_morph_rotations`'s return value.
        `azimuths`: list of ``sum(frame_counts)`` per-frame azimuths.
    """
    rotations_resolved = resolve_morph_rotations(rotations, n_datasets)
    frame_counts = segment_frame_counts(n_datasets, total_frames,
                                        rotations_resolved)
    azimuths = segment_azimuths(frame_counts, rotations_resolved, azim0)
    return frame_counts, rotations_resolved, azimuths
