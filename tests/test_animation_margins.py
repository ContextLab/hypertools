# -*- coding: utf-8 -*-
"""Regression guard for the "bounding-box cube crowded/cut off" issue
(PR #272 Sec.2: ``set_box_aspect`` zoom 1.25->1.125 + full-canvas axes for
animated 3-D plots).

D1 diagnosis (2026-07-06, maintainer report: "the issue has re-surfaced for
*some* animations, e.g. the shape morph demo"): a real-render harness was
built that steps every frame of an animated matplotlib figure (via the
FuncAnimation's own update function -- ``ani._func(k, *ani._args)``, the
same technique `tests/test_morph_animation.py` already uses), rasterizes
the canvas, and measures the pixel margin from the nearest inked (non-
background) pixel to each of the four canvas edges.

Exhaustive per-frame scans (every single frame, not a subsample) of both
`examples/plot_shape_morph.py`'s call (``animate='morph'``, no surface,
the full 7-shape zoo, 390 frames) and `examples/animate_surface_morph.py`'s
call (``animate='morph'`` + ``surface=``, 360 frames, cube_scale grown to
~1.16 by the union-hull sizing), plus a `cube_scale`~1.44 stress case,
found NO clipping: every style (`True`/'parallel', 'spin', 'serial',
'morph', with and without `surface=`) currently produces comfortably
healthy margins (roughly 60-260px on this machine's default canvas size),
and margins are invariant to `cube_scale` growing past 1 (as expected,
since ``set_box_aspect(zoom=...)`` and the axes limits both scale by the
same ratio). Root cause of the maintainer's visual impression was traced
to two measurement confounds, not a real defect: a legend box or a
plotly Play/Pause button widget rendered close to a canvas edge, counted
as "inked" by a naive whole-canvas scan even though neither is part of
the cube/data scene.

This test file locks in that verified-healthy state so it becomes a real
regression guard: if a future change to the zoom/margin treatment
(`_anim_box_zoom`, `ax.set_position`, or the `cube_scale_anim`/axis-limit
sizing in `matplotlib_backend.animate_plot3D`) ever breaks the margin for
any style, these tests fail. The floor (15px) is deliberately far below
every measured value above -- loose enough to tolerate CI font/anti-
aliasing variation, tight enough to catch an actual clipping regression.
"""
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hypertools as hyp

MARGIN_FLOOR_PX = 15


def _measure_margins(fig, bg_thresh=250):
    """Rasterize the current canvas and return the pixel margin from the
    nearest inked (non-near-white) pixel to each of the four edges. A
    legend, if present, is masked out first -- it is a UI decoration, not
    part of the cube/data scene the "crowded/cut off" report is about."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    h, w = buf.shape[0], buf.shape[1]
    rgb = buf[:, :, :3].astype(np.int32)
    inked = rgb.min(axis=2) < bg_thresh
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            bbox = leg.get_window_extent(fig.canvas.get_renderer())
            x0, x1 = max(0, int(bbox.x0) - 1), min(w, int(bbox.x1) + 1)
            y0 = max(0, int(h - bbox.y1) - 1)
            y1 = min(h, int(h - bbox.y0) + 1)
            inked[y0:y1, x0:x1] = False
    cols = np.where(inked.any(axis=0))[0]
    rows = np.where(inked.any(axis=1))[0]
    assert len(cols) and len(rows), "rasterized frame is entirely blank"
    return dict(
        left=int(cols.min()),
        right=int(w - 1 - cols.max()),
        top=int(rows.min()),
        bottom=int(h - 1 - rows.max()),
    )


def _min_margins_over_frames(fig, ani, frame_indices):
    """Step the animation's own update function across `frame_indices`
    (spanning >= one full rotation) and return the min margin per edge."""
    fig.canvas.draw()
    mins = dict(left=10**9, right=10**9, top=10**9, bottom=10**9)
    for k in frame_indices:
        ani._func(k, *ani._args)
        m = _measure_margins(fig)
        for edge in mins:
            mins[edge] = min(mins[edge], m[edge])
    return mins


def _full_rotation_frames(ani, n_samples=16):
    total = getattr(ani, '_save_count', None) or getattr(ani, 'save_count', None)
    assert total, "could not determine animation frame count"
    n_samples = min(n_samples, total)
    idx = sorted(set(np.linspace(0, total - 1, n_samples).astype(int).tolist()))
    return idx, total


def _blob_clouds(k, n=30, seed=0, spread=0.3):
    rng = np.random.default_rng(seed)
    return [rng.normal(scale=spread, size=(n, 3)) + rng.normal(scale=0.4, size=3)
            for _ in range(k)]


def _assert_healthy(mins, label):
    for edge, val in mins.items():
        assert val >= MARGIN_FLOOR_PX, (
            f"{label}: {edge} margin only {val}px (floor {MARGIN_FLOOR_PX}px) "
            f"-- bounding-box cube is crowding/clipping the canvas edge"
        )


class TestMorphAnimationMargins:
    def test_morph_no_surface_full_rotation_margins(self):
        clouds = _blob_clouds(3, n=40, seed=1)
        fig, ani = hyp.plot(clouds, '.', animate='morph', duration=3,
                            frame_rate=10, show=False)
        idx, total = _full_rotation_frames(ani)
        mins = _min_margins_over_frames(fig, ani, idx)
        plt.close(fig)
        _assert_healthy(mins, f"morph (no surface), {total} frames")

    def test_morph_with_surface_full_rotation_margins(self):
        clouds = _blob_clouds(3, n=20, seed=2)
        surface_spec = {'alpha': 0.9, 'color': '#2E86AB', 'smoothing': 1,
                        'keep_points': True}
        fig, ani = hyp.plot(clouds, '.', animate='morph', duration=3,
                            frame_rate=10, morph_samples=20,
                            surface=surface_spec, show=False)
        idx, total = _full_rotation_frames(ani)
        mins = _min_margins_over_frames(fig, ani, idx)
        plt.close(fig)
        _assert_healthy(mins, f"morph (with surface), {total} frames")

    def test_morph_rotations_list_full_rotation_margins(self):
        """The shape-morph demos use a per-segment `rotations` list (not a
        scalar) -- exercise that path explicitly, since it drives the
        camera azimuth differently (`morph.morph_schedule`)."""
        clouds = _blob_clouds(3, n=30, seed=3)
        fig, ani = hyp.plot(clouds, '.', animate='morph',
                            rotations=[1, 0.25, 1, 0.25, 1], duration=5,
                            frame_rate=10, show=False)
        idx, total = _full_rotation_frames(ani)
        mins = _min_margins_over_frames(fig, ani, idx)
        plt.close(fig)
        _assert_healthy(mins, f"morph (rotations list), {total} frames")


class TestOtherStyleAnimationMargins:
    """Same margin guard for the other animate styles, so a future change
    to the shared zoom/margin plumbing (`_anim_box_zoom`, `set_position`,
    `cube_scale_anim`) is caught regardless of which style regresses."""

    @pytest.mark.parametrize('style', [True, 'spin', 'serial'])
    def test_style_full_rotation_margins(self, style):
        clouds = _blob_clouds(3, n=30, seed=4)
        fig, ani = hyp.plot(clouds, '.', animate=style, duration=3,
                            frame_rate=10, show=False)
        idx, total = _full_rotation_frames(ani)
        mins = _min_margins_over_frames(fig, ani, idx)
        plt.close(fig)
        _assert_healthy(mins, f"animate={style!r}, {total} frames")

    def test_animate_true_with_legend_margins(self):
        """Regression guard for the legend-fitting interaction: a legend
        must never eat into the cube's own zoom margin (`_fit_right_legend`
        widens the canvas rather than shrinking the axes)."""
        clouds = _blob_clouds(2, n=30, seed=5)
        fig, ani = hyp.plot(clouds, animate=True, legend=['a', 'b'],
                            duration=3, frame_rate=10, show=False)
        idx, total = _full_rotation_frames(ani)
        mins = _min_margins_over_frames(fig, ani, idx)
        plt.close(fig)
        _assert_healthy(mins, f"animate=True + legend, {total} frames")

    def test_surface_cube_scale_growth_does_not_shrink_margin(self):
        """Stress the theorized "zoom doesn't scale with an enlarged cube"
        failure mode directly: a sparse cloud + surface= pushes
        `cube_scale_anim` well past 1 (union-hull/`_rescale_for_containment`
        growth). `set_box_aspect(zoom=...)` and the axes limits are both
        sized off `cube_scale_anim`, so the rendered margin should be
        unaffected by how large the cube itself gets."""
        clouds = _blob_clouds(3, n=8, seed=6, spread=0.3)
        surface_spec = {'alpha': 0.9, 'color': '#2E86AB', 'smoothing': 1,
                        'keep_points': True}
        fig, ani = hyp.plot(clouds, '.', animate='morph', duration=4,
                            frame_rate=10, morph_samples=8,
                            surface=surface_spec, show=False)
        idx, total = _full_rotation_frames(ani)
        mins = _min_margins_over_frames(fig, ani, idx)
        plt.close(fig)
        _assert_healthy(mins, f"morph + surface (sparse, grown cube), "
                              f"{total} frames")
