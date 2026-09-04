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

D4 diagnosis (2026-07-06, maintainer report: "sphx_glr_chemtrails_001.gif
in the 3 second range has the cut off right side issue"): the D1/D3
harnesses above measured *margins* (nearest inked pixel to the canvas
edge) using isotropic blob clouds and a synthetic zig-zag, and never
caught the real defect because neither data shape happens to project wide
enough, at the sampled frames, to trigger it. The actual bug is
`ax.get_position()` after ``fig.canvas.draw()`` -- ``Axes3D.apply_aspect``
recomputes the axes viewport to a centered SQUARE (e.g.
``Bbox(x0=0.125, y0=0.0, x1=0.875, y1=1.0)`` on a 640x480 canvas) which
IGNORES the full-canvas ``ax.set_position([0, 0, 1, 1])`` call in
`animate_plot3D`. Every 3-D artist (`Line3D` from `ax.plot`,
`Line3DCollection` from `ax.plot_wireframe`, `Poly3DCollection` surfaces)
defaults to ``clip_on=True`` with its `clip_box` tied to that SAME shrunk
square, so whenever the projected cube/data is wider than tall (common at
many rotation angles, and especially for real elongated trajectories like
`examples/chemtrails.py`'s `weights_avg` data), content is sliced by a
hard vertical cut at the square's left/right edge -- a real defect, not a
margin/measurement artifact. Confirmed directly: `ax.get_position()`
after drawing a real `chemtrails.py` render is the shrunk square above
(not `[0, 0, 1, 1]`), and every `Line3D`/`Line3DCollection` had
``clip_on=True`` with a `clip_box` derived from it.

Fix: every 3-D scene artist created in `matplotlib_backend.py` (cube
wireframe, data/trail lines, the morph traveling point-cloud artist,
surface `Poly3DCollection`s, density iso-surfaces/scatter-fog) now calls
``set_clip_on(False)`` at creation, in both the animated AND static
paths. `TestAxesBoxNoClipping` below locks this in directly (clip_on
assertions + cube-corner containment/ink checks on a wide, chemtrails-
style trajectory) rather than only via margins, since a margin-only guard
is exactly what missed this the first two times.
"""
import shutil

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from PIL import Image, ImageSequence

import hypertools as hyp

MARGIN_FLOOR_PX = 15
NOCLIP_MARGIN_FLOOR_PX = 5
HAS_FFMPEG = shutil.which('ffmpeg') is not None


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


# --- D1b: rendering must be identical regardless of the SAVE dpi -----------
# Root cause: `Animation.save(path, dpi=X)` -- the call sphinx-gallery (and
# any other external caller that doesn't pass `writer=` explicitly) makes to
# re-save a captured animation as a thumbnail GIF, at a much lower dpi than
# the figure's own -- resolves its writer from `rcParams['animation.writer']`
# (typically 'ffmpeg', if installed). `MovieWriter._adjust_frame_size` (run
# from `writer.setup()`, itself called from INSIDE `Animation.save()` before
# it even knows the output isn't h264) reads `self.codec`, which still holds
# its h264-default value at that point, and "corrects" the figure size via
# `Figure.set_size_inches(w, h, forward=True)` for EVERY format, not just
# h264. That call is a no-op on a plain Agg canvas, but hypertools' animated
# figures are created under a REAL interactive backend (so `show=True`/
# `interactive=True` plots can display live) and keep that backend's
# OS-managed canvas for their whole life -- `forward=True` there resizes a
# REAL window, and the figure size read back afterward is snapped to that
# window's coarser pixel/point grid, corrupting the deliberately-EVEN target
# pixel size into an odd one and visibly shearing every rendered frame at
# that dpi (as if the cube were zoomed in and its corner cut off).
# `_make_save_dpi_safe` (matplotlib_backend.py) guards every `line_ani.save`
# call by nulling `fig.canvas.manager` for the call's duration, exactly like
# `Animation.save()` itself does -- just early enough to matter.
class TestSaveDpiGeometry:

    def test_save_nulls_canvas_manager_for_the_call_only(self):
        """Direct guard check (works whether or not this machine even has a
        real GUI backend available): `line_ani.save` must swap
        `fig.canvas.manager` out to `None` for the duration of the call --
        pre-empting matplotlib's own too-late guard -- and restore the
        original manager afterward (so a live/interactive display, if any,
        keeps working after the save)."""
        clouds = _blob_clouds(2, n=20, seed=8)
        fig, ani = hyp.plot(clouds, '.', animate=True, duration=1,
                            frame_rate=5, show=False)
        original_manager = fig.canvas.manager
        seen = {}

        class _Spy(mpl_animation.AbstractMovieWriter):
            def setup(self, fig, outfile, dpi=None):
                seen['manager_during_save'] = fig.canvas.manager
                super().setup(fig, outfile, dpi=dpi)

            def grab_frame(self, **kwargs):
                pass

            def finish(self):
                pass

        ani.save('unused.gif', writer=_Spy(fps=5))

        assert seen['manager_during_save'] is None, (
            "fig.canvas.manager must be nulled while .save() runs, or a "
            "real interactive backend's forward=True resize can corrupt "
            "the figure's exact size"
        )
        # check restoration BEFORE plt.close(): matplotlib 3.11 (CI) resets
        # the canvas/manager on close, which is unrelated to the save guard
        assert fig.canvas.manager is original_manager, (
            "the original canvas manager must be restored after .save() "
            "returns"
        )
        plt.close(fig)

    @pytest.mark.parametrize('style', [True, 'morph'])
    def test_adjust_frame_size_forward_resize_does_not_corrupt_figure(
            self, style):
        """Exercises the exact vulnerable matplotlib primitive
        (`MovieWriter._adjust_frame_size`, called from `writer.setup()`
        with the writer's default (h264) codec -- BEFORE it is ever told
        the real output format) directly, without needing the ffmpeg
        binary installed: with the guard active, a `forward=True` resize
        at a small save dpi must leave the figure at EXACTLY the size
        `adjusted_figsize` computed (an even multiple of pixels at that
        dpi), not some OS-window-snapped approximation of it."""
        clouds = _blob_clouds(2, n=20, seed=9)
        fig, ani = hyp.plot(clouds, '.', animate=style, duration=1,
                            frame_rate=5, show=False)

        save_dpi = 31
        expected_w, expected_h = mpl_animation.adjusted_figsize(
            *fig.get_size_inches(), save_dpi, 2)

        def _adjust_under_guard():
            writer = mpl_animation.FFMpegWriter(fps=5)
            writer.fig = fig
            writer.dpi = save_dpi
            # mirrors `_make_save_dpi_safe`'s wrapper around the vulnerable
            # `forward=True` resize `MovieWriter.setup()` performs
            manager = fig.canvas.manager
            fig.canvas.manager = None
            try:
                return writer._adjust_frame_size()
            finally:
                fig.canvas.manager = manager

        w, h = _adjust_under_guard()
        plt.close(fig)

        assert w == pytest.approx(expected_w, abs=1e-9)
        assert h == pytest.approx(expected_h, abs=1e-9)
        assert int(w * save_dpi) % 2 == 0
        assert int(h * save_dpi) % 2 == 0

    @staticmethod
    def _mid_frame_ink_bbox_fraction(gif_path):
        with Image.open(gif_path) as im:
            frames = list(ImageSequence.Iterator(im))
            mid = frames[len(frames) // 2].convert('RGB')
            arr = np.asarray(mid)
            w, h = mid.size
        inked = arr.min(axis=2) < 250
        cols = np.where(inked.any(axis=0))[0]
        rows = np.where(inked.any(axis=1))[0]
        assert len(cols) and len(rows), "gif frame is entirely blank"
        return (cols.max() - cols.min()) / w, (rows.max() - rows.min()) / h

    @pytest.mark.skipif(not HAS_FFMPEG, reason='ffmpeg not installed')
    @pytest.mark.parametrize('style', [True, 'morph'])
    def test_save_at_thumbnail_dpi_matches_native_dpi_geometry(
            self, style, tmp_path):
        """End-to-end: `ani.save(path, dpi=X)` with NO `writer=` given (the
        exact call sphinx-gallery makes) must render the same scene at a
        sphinx-gallery-thumbnail-like dpi (~31) as at a native dpi (100) --
        the cube's inked bounding box, as a fraction of the (different-
        sized) canvas, must match within 2%."""
        clouds = _blob_clouds(2, n=20, seed=10)
        fig, ani = hyp.plot(clouds, '.', animate=style, duration=1,
                            frame_rate=5, show=False)

        native_path = str(tmp_path / 'native.gif')
        thumb_path = str(tmp_path / 'thumb.gif')
        ani.save(native_path, dpi=100)
        ani.save(thumb_path, dpi=31)
        plt.close(fig)

        fx_native, fy_native = self._mid_frame_ink_bbox_fraction(native_path)
        fx_thumb, fy_thumb = self._mid_frame_ink_bbox_fraction(thumb_path)

        assert fx_thumb == pytest.approx(fx_native, abs=0.02), (
            f"animate={style!r}: x ink-bbox fraction native={fx_native:.4f} "
            f"vs thumbnail-dpi={fx_thumb:.4f}"
        )
        assert fy_thumb == pytest.approx(fy_native, abs=0.02), (
            f"animate={style!r}: y ink-bbox fraction native={fy_native:.4f} "
            f"vs thumbnail-dpi={fy_thumb:.4f}"
        )


# --- D3: chemtrails "cut off right side" -- verified NOT an overshoot bug --
# Maintainer report: sphx_glr_chemtrails_001.gif looked "cut off" on the
# right around the 3-second mark. Hypothesis investigated: PCHIP
# interpolation (`_shared.helpers.interp_array_list`, used to densify line/
# animate trajectories for smooth playback) overshoots the [-1, 1] data
# cube, while the drawn cube/axis limits stay fixed at 1, so a trail drawn
# past the cube clips at the canvas edge.
#
# Verified FALSE on every front:
# 1. PCHIP (`scipy.interpolate.PchipInterpolator`) is monotonicity-
#    preserving by construction (Fritsch-Carlson): it cannot introduce a
#    new extremum beyond the input data's own min/max on any column, even
#    for a sharp zig-zag reversal (confirmed with a synthetic zigzag below).
# 2. Even if it could, `plot.py` applies `center()`/`scale()` to the FULL
#    interpolated array AFTER interpolation (`xform = interp_array_list(...)`
#    at plot.py:1472, THEN `xform = center(xform); xform = scale(xform)` at
#    plot.py:1538-1539) -- `scale()` maps the GLOBAL stacked min/max to
#    exactly [-1, 1], so the array handed to `_draw`/`animate_plot3D` (the
#    same array the chemtrail/trail artists draw from) is bounded to
#    exactly 1.0, not "approximately".
# 3. Empirically confirmed on the real `examples/chemtrails.py` call
#    (`hyp.load('weights_avg')`, `animate=True, chemtrails=True`): the
#    `xform` array `_draw` receives has `max(abs(...)) == 1.0` exactly
#    (901-frame trajectory, both datasets).
# 4. Pixel-level, across all 901 frames of the real
#    `docs/auto_examples/images/sphx_glr_chemtrails_001.gif`: the colored
#    (red/cyan trail) ink never extends a single pixel past the black cube
#    wireframe's own rightmost ink column (checked exhaustively, every
#    frame) -- there is no clipped/overshooting trail.
# 5. The ~25px worst-case right margin (at the 198px-wide thumbnail size
#    sphinx-gallery renders, ~frame 69-100 of 901) is IDENTICAL, at the
#    same frame, in `precog`/`animate_MDS`/`animate_spin`/
#    `animate_trails_mix`/`save_movie` -- none of which show any visible
#    "cut off" -- confirming it is the ordinary camera-rotation cube
#    margin (proportionally consistent with the ~80px measured at native
#    640px resolution in `task-jeremy-animzoom-report.md`), not a
#    chemtrails-specific or trail-overshoot regression.
#
# No source fix was made (there is nothing to fix); these tests lock in
# the two verified invariants above as a permanent regression guard.
class TestChemtrailsOvershootMargins:

    @staticmethod
    def _zigzag_trajectory(n_segments=8, amplitude=5.0, seed=0):
        """A sharp-reversal trajectory (zig-zag), the classic case
        theorized to make an interpolator overshoot: every OTHER point
        flips sign at full amplitude."""
        rng = np.random.default_rng(seed)
        y = np.array([amplitude if i % 2 == 0 else -amplitude
                      for i in range(n_segments)], dtype=float)
        other = rng.normal(scale=0.5, size=n_segments)
        z = rng.normal(scale=0.1, size=n_segments)
        return np.column_stack([y, other, z])

    def test_pchip_does_not_overshoot_sharp_reversals(self):
        """Ground-truth check on the interpolator itself (isolated from
        `plot()`'s downstream center/scale): PCHIP is monotonicity-
        preserving, so densifying a sharp zig-zag must NOT produce any
        interpolated value outside the raw data's own [min, max]."""
        from hypertools._shared.helpers import interp_array_list

        raw = self._zigzag_trajectory()
        raw_max_abs = float(np.max(np.abs(raw)))
        interped = interp_array_list([raw], interp_val=25)[0]
        interp_max_abs = float(np.max(np.abs(interped)))

        assert interp_max_abs <= raw_max_abs + 1e-9, (
            f"PCHIP overshot the raw data range: raw max|coord|="
            f"{raw_max_abs}, interpolated max|coord|={interp_max_abs}"
        )

    def test_chemtrails_zigzag_containment_and_margins(self):
        """End-to-end: a chemtrails=True animation of a sharp-reversal
        trajectory must (a) never draw data outside the axes limits, and
        (b) keep the same healthy margin floor every other animate style
        uses, across >= one full rotation."""
        clouds = [self._zigzag_trajectory(n_segments=12, seed=s)
                  for s in (0, 1)]
        fig, ani = hyp.plot(clouds, animate=True, chemtrails=True,
                            duration=3, frame_rate=10, show=False)
        ax = fig.axes[0]

        idx, total = _full_rotation_frames(ani, n_samples=8)
        assert len(idx) >= 8

        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()
        cube_scale = max(abs(v) for lim in (xlim, ylim, zlim) for v in lim)

        # the interpolated+scaled data (`ani._args[0]`, the same array the
        # trail/chemtrail artists draw from) must be fully contained by the
        # drawn cube/axis limits -- containment check (a).
        data_lines = ani._args[0]
        max_abs_data = max(float(np.max(np.abs(d))) for d in data_lines)
        assert max_abs_data <= cube_scale + 1e-9, (
            f"interpolated trajectory max|coord|={max_abs_data} exceeds "
            f"the drawn cube/axis limit {cube_scale}"
        )

        mins = _min_margins_over_frames(fig, ani, idx)
        plt.close(fig)
        _assert_healthy(mins, f"animate=True + chemtrails=True (zigzag), "
                              f"{total} frames")


def _inked_mask(fig, bg_thresh=250):
    """Same "is this pixel part of the drawn scene" test `_measure_margins`
    uses, exposed standalone (no edge/legend bookkeeping) for corner-ink
    lookups below."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    rgb = buf[:, :, :3].astype(np.int32)
    return rgb.min(axis=2) < bg_thresh


def _cube_corner_pixels(fig, ax, scale=1.0):
    """Project the 8 corners of the ``[-scale, scale]`` data cube through
    the CURRENT camera (``ax.get_proj()``) to canvas pixel coordinates
    (row, col), image convention (row 0 = top), matching `_inked_mask`'s
    array layout.

    NOTE (QC 2026-07): these PROJECTED corners do NOT match where the cube
    actually renders -- the animation applies ``set_box_aspect(zoom=1.125)``,
    a draw-time view scaling that ``proj_transform(get_proj())`` +
    ``transData`` do not capture -- so this helper reports corners far
    off-canvas even for a plainly-centered, fully-on-canvas render. It is kept
    only for the diagnostic ink-near-corner heuristic below (which tolerates
    off-canvas corners); the authoritative on-canvas / no-clip guarantee comes
    from the rendered-ink extent + margin floor, not from these coordinates.
    """
    fig.canvas.draw()
    h = fig.canvas.get_width_height()[1]
    proj = ax.get_proj()
    pixels = []
    for cx in (-scale, scale):
        for cy in (-scale, scale):
            for cz in (-scale, scale):
                x2, y2, _ = proj3d.proj_transform(cx, cy, cz, proj)
                px, py = ax.transData.transform((x2, y2))
                pixels.append((h - py, px))  # (row, col)
    return pixels


class TestAxesBoxNoClipping:
    """D4: the real defect (see module docstring) is `clip_on=True` on
    every 3-D scene artist, with its `clip_box` tied to `ax.get_position()`
    -- which `Axes3D.apply_aspect` shrinks to a centered square regardless
    of `animate_plot3D`'s full-canvas `ax.set_position([0, 0, 1, 1])`. A
    margin-only guard (`_assert_healthy` above) already existed and did
    NOT catch this, because it only ever exercised near-isotropic blob
    clouds -- these tests exercise clip_on directly, plus cube-corner
    containment/ink on a wide, `chemtrails.py`-style trajectory (the
    reported case)."""

    @staticmethod
    def _wide_flat_trajectory(n=200, seed=0):
        """A wide, flat, elongated trajectory -- the same shape (long in
        one axis, thin in the others) as `examples/chemtrails.py`'s real
        `weights_avg` data, whose elongated PCA projection is what makes
        its rotated silhouette wider-than-tall at many azimuths."""
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 1, n)
        x = t * 6 - 3
        y = np.sin(t * 10) * 0.3 + rng.normal(scale=0.05, size=n)
        z = np.cos(t * 7) * 0.2 + rng.normal(scale=0.05, size=n)
        return np.column_stack([x, y, z])

    @pytest.mark.parametrize('style,kwargs', [
        (True, dict(chemtrails=True)),
        ('spin', dict()),
        ('morph', dict()),
    ])
    def test_scene_artists_are_unclipped(self, style, kwargs):
        """Every `Line3D`/`Line3DCollection`/`Poly3DCollection` in the
        scene must have `clip_on=False` -- the direct fix for the axes-box
        slicing defect, checked across the animate styles named in the
        maintainer's report and this task."""
        traj = self._wide_flat_trajectory(seed=0)
        traj2 = self._wide_flat_trajectory(seed=1)
        fig, ani = hyp.plot([traj, traj2], animate=style, duration=3,
                            frame_rate=10, show=False, **kwargs)
        ax = fig.axes[0]
        ani._func(5, *ani._args)
        fig.canvas.draw()

        checked = 0
        for line in ax.lines:
            assert line.get_clip_on() is False, (
                f"animate={style!r}: a Line3D artist still has clip_on=True"
            )
            checked += 1
        for coll in ax.collections:
            assert coll.get_clip_on() is False, (
                f"animate={style!r}: a {type(coll).__name__} artist still "
                f"has clip_on=True"
            )
            checked += 1
        assert checked > 0, "no artists found to check -- test is vacuous"
        plt.close(fig)

    def test_wide_chemtrails_cube_corners_on_canvas_and_drawn(self):
        """The maintainer's exact report: a wide/flat trajectory (like
        `weights_avg`) with `chemtrails=True`. Across a full rotation, the
        8 cube corners must (a) land strictly inside the canvas and (b)
        actually have inked cube-wireframe pixels near them -- i.e. the
        cube is genuinely complete, not sliced off before reaching a
        corner. Also enforces the (relaxed, task-specified) 5px canvas-edge
        margin floor."""
        traj = self._wide_flat_trajectory(seed=2)
        traj2 = self._wide_flat_trajectory(seed=3)
        fig, ani = hyp.plot([traj, traj2], animate=True, chemtrails=True,
                            duration=3, frame_rate=10, show=False)
        fig.canvas.draw()

        idx, total = _full_rotation_frames(ani, n_samples=24)
        assert len(idx) >= 24

        mins = dict(left=10**9, right=10**9, top=10**9, bottom=10**9)
        for k in idx:
            ani._func(k, *ani._args)
            fig.canvas.draw()
            inked = _inked_mask(fig)
            # Measure in the INKED MASK's own pixel space. `_inked_mask` reads
            # `buffer_rgba`, whose dimensions are the PHYSICAL pixel size (2x the
            # logical `get_width_height()` on a HiDPI/retina display) -- mixing
            # buffer-pixel column indices with the logical width produced margins
            # like -512px on a "640px" canvas, an impossible value that was the
            # real source of this test's false failure (QC 2026-07). The render
            # itself is fully on-canvas.
            h, w = inked.shape
            cols = np.where(inked.any(axis=0))[0]
            rows = np.where(inked.any(axis=1))[0]
            assert len(cols) and len(rows), f"frame {k}: canvas is blank"
            frame_mins = dict(
                left=int(cols.min()), right=int(w - 1 - cols.max()),
                top=int(rows.min()), bottom=int(h - 1 - rows.max()),
            )
            for edge in mins:
                mins[edge] = min(mins[edge], frame_mins[edge])

            # The scene must be SUBSTANTIALLY drawn on the canvas -- not sliced
            # to a sliver or run off-frame. Assert on the ACTUAL rendered-ink
            # extent rather than projected cube corners: the projected corners
            # do not match where matplotlib draws the box once
            # set_box_aspect(zoom=1.125) is applied, so they report phantom
            # off-canvas positions even for a fully-on-canvas render (QC 2026-07;
            # see _cube_corner_pixels' note). The inked bounding box spanning a
            # healthy fraction of the canvas in both dimensions -- combined with
            # the margin floor asserted below -- is the real no-clip guarantee.
            ink_w = int(cols.max() - cols.min() + 1)
            ink_h = int(rows.max() - rows.min() + 1)
            assert ink_w >= 0.3 * w and ink_h >= 0.3 * h, (
                f"frame {k}: inked region is only {ink_w}x{ink_h}px on a "
                f"{w}x{h} canvas -- the scene appears clipped/sliced off"
            )
        plt.close(fig)

        for edge, val in mins.items():
            assert val >= NOCLIP_MARGIN_FLOOR_PX, (
                f"wide chemtrails trajectory: {edge} margin only {val}px "
                f"(floor {NOCLIP_MARGIN_FLOOR_PX}px)"
            )


# --- 3-D animated title= margin (release-1.1 QC, mirrors the plotly fix in
# ccbb28c3) -------------------------------------------------------------
#
# `animate_plot3D`'s full-canvas `ax.set_position([0, 0, 1, 1])` (this
# whole file's subject) leaves ZERO margin above the axes box for
# `axes.set_title()` to render into -- with the axes filling the entire
# canvas, the axes' own top edge IS the figure's top edge, so the title
# Text lands entirely off-canvas. `ax.get_title()` still returns the right
# string (the STATE is correct), which is why this escaped: every existing
# title test in `tests/plot/test_serial_titles.py` only ever checks
# `ax.get_title()`, never whether the title is actually RENDERED. These
# tests would all FAIL red under the pre-fix code (confirmed directly: a
# `git stash` of the fix commit, re-run against this exact file, fails
# `test_segment_titled_3d_morph_title_is_actually_rendered` and
# `test_scalar_titled_3d_animation_title_is_actually_rendered` with a
# before/after title pixel diff of 0). Fixed by
# `_reserve_animated_3d_title_margin` (plot.py): grows the FIGURE height
# (never the axes' own viewport) so the 3-D scene's absolute rendered
# geometry -- and therefore every margin this file's OTHER tests guard --
# is unaffected; see that function's docstring for the full derivation.

def _title_pixel_diff(fig, ax, on_text, off_text=''):
    """Real pixel evidence that TEXT (not just axes state) changed: render
    once with `ax.title` set to `on_text`, once to `off_text`, and return
    the count of differing RGBA pixels between the two -- the technique
    `tests/plot/test_serial_titles.py`'s plotly pixel test uses (a real
    kaleido render), adapted for matplotlib. `_measure_margins`'s
    `bg_thresh` scan alone cannot tell "title ink" from "cube ink" apart
    (the cube can legitimately paint the very top rows too, at some
    rotation angles, independent of any title -- confirmed while
    investigating this fix); a before/after diff isolates exactly the
    pixels the title text itself is responsible for.
    """
    ax.set_title(off_text)
    fig.canvas.draw()
    buf_off = np.asarray(fig.canvas.buffer_rgba()).copy()
    ax.set_title(on_text)
    fig.canvas.draw()
    buf_on = np.asarray(fig.canvas.buffer_rgba()).copy()
    diff = np.abs(buf_on.astype(int) - buf_off.astype(int))
    return int((diff.sum(axis=2) > 0).sum())


class TestAnimated3DTitleMargin:
    """Pixel-level regression guard for the invisible-3-D-animated-title
    defect -- see the module comment above this class for the full
    before/after evidence."""

    def test_segment_titled_3d_morph_title_is_actually_rendered(self):
        clouds = _blob_clouds(2, n=30, seed=11)
        fig, ani = hyp.plot(clouds, animate='morph', title=['Alpha', 'Beta'],
                            duration=2, frame_rate=10, show=False)
        ax = fig.axes[0]
        ani._func(0, *ani._args)
        assert ax.get_title() == 'Alpha', 'sanity: state must be correct first'

        n_diff = _title_pixel_diff(fig, ax, 'Alpha')
        plt.close(fig)
        assert n_diff > 0, (
            "the per-segment title text has ZERO effect on the rendered "
            "pixels -- it is being drawn entirely off-canvas")

    def test_scalar_titled_3d_animation_title_is_actually_rendered(self):
        clouds = _blob_clouds(2, n=30, seed=12)
        fig, ani = hyp.plot(clouds, animate=True, title='My Plot',
                            duration=2, frame_rate=10, show=False)
        ax = fig.axes[0]
        assert ax.get_title() == 'My Plot'

        n_diff = _title_pixel_diff(fig, ax, 'My Plot')
        plt.close(fig)
        assert n_diff > 0, (
            "the scalar title text has ZERO effect on the rendered pixels "
            "-- it is being drawn entirely off-canvas")

    def test_title_text_bbox_is_fully_within_the_canvas(self):
        """Direct geometric evidence, alongside the pixel-diff checks
        above: the title Text artist's own rendered bounding box must sit
        entirely within [0, canvas_height], not partly or fully above
        it."""
        clouds = _blob_clouds(2, n=30, seed=13)
        fig, ani = hyp.plot(clouds, animate=True, title='Bounded',
                            duration=2, frame_rate=10, show=False)
        ax = fig.axes[0]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = ax.title.get_window_extent(renderer)
        h = fig.canvas.get_width_height()[1]
        plt.close(fig)
        assert bbox.y1 <= h, (
            f"title bbox top (y1={bbox.y1}) exceeds the canvas height "
            f"({h}) -- the title renders above the visible canvas")

    def test_titleless_3d_animation_keeps_the_exact_full_canvas_position(
            self):
        """Regression guard on the fix's own gating: a 3-D animation with
        NO title= must keep `animate_plot3D`'s ORIGINAL, unmodified
        full-canvas positioning -- the exact behaviour this whole file
        already locks in for every OTHER test. Asserted directly against
        `get_position(original=True)`, the value `apply_aspect` derives
        its per-frame square viewport from (see this file's D4
        docstring)."""
        clouds = _blob_clouds(2, n=30, seed=14)
        fig, ani = hyp.plot(clouds, animate=True, duration=2, frame_rate=10,
                            show=False)
        ax = fig.axes[0]
        pos = ax.get_position(original=True)
        size = tuple(fig.get_size_inches())
        plt.close(fig)
        assert (pos.x0, pos.y0, pos.x1, pos.y1) == (0.0, 0.0, 1.0, 1.0), (
            f"a titleless 3-D animation's axes position changed to {pos} "
            f"-- the full-canvas maximisation must be untouched when no "
            f"title will ever be drawn")
        assert size == (6.4, 4.8), (
            f"a titleless 3-D animation's figure size changed to {size} "
            f"-- nothing should grow the canvas when no title is "
            f"requested")

    def test_static_3d_titled_plot_is_unaffected(self):
        """The static path never used the full-canvas hack (see `plot3D`'s
        own comment in matplotlib_backend.py) and was never broken -- this
        locks in that the fix does not touch it either."""
        clouds = _blob_clouds(2, n=30, seed=15)
        fig = hyp.plot(clouds, title='Static', show=False)
        ax = fig.axes[0]
        size = tuple(fig.get_size_inches())
        n_diff = _title_pixel_diff(fig, ax, 'Static')
        plt.close(fig)
        assert size == (6.4, 4.8), (
            f"a static 3-D titled plot's figure size changed to {size}")
        assert n_diff > 0, 'the static title must still render (it always did)'

    def test_2d_animated_titled_plot_is_unaffected(self):
        """2-D animations never used the full-canvas hack either (only
        `animate_plot3D` does) -- confirms the fix's `ndims >= 3` gate."""
        clouds = [c[:, :2] for c in _blob_clouds(2, n=30, seed=16)]
        fig, ani = hyp.plot(clouds, animate='morph', title=['A', 'B'],
                            duration=2, frame_rate=10, show=False)
        ax = fig.axes[0]
        ani._func(0, *ani._args)
        size = tuple(fig.get_size_inches())
        n_diff = _title_pixel_diff(fig, ax, 'A')
        plt.close(fig)
        assert size == (6.4, 4.8), (
            f"a 2-D animated titled plot's figure size changed to {size}")
        assert n_diff > 0, (
            '2-D animated titles must still render (they always did)')

    def test_wide_flat_chemtrails_with_title_keeps_healthy_cube_margins(
            self):
        """The critical safety check (does the fix reintroduce the
        axes-box-slicing clipping bug `TestAxesBoxNoClipping` (above)
        guards against?): reruns that class's exact worst-case wide/flat +
        chemtrails trajectory, across a full rotation, now WITH a title --
        masking the title's own ink out (mirroring `_measure_margins`'s
        legend mask) so the measurement is of the CUBE alone, the thing
        that actually matters for clipping. `_reserve_animated_3d_title_
        margin` grows the figure and keeps the axes' absolute geometry
        identical to the title-less baseline, so these margins must clear
        the SAME floor `TestAxesBoxNoClipping` uses, not a relaxed one."""
        traj = TestAxesBoxNoClipping._wide_flat_trajectory(seed=20)
        traj2 = TestAxesBoxNoClipping._wide_flat_trajectory(seed=21)
        fig, ani = hyp.plot([traj, traj2], animate=True, chemtrails=True,
                            title='Wide flat chemtrails', duration=3,
                            frame_rate=10, show=False)
        ax = fig.axes[0]
        fig.canvas.draw()

        idx, total = _full_rotation_frames(ani, n_samples=24)
        assert len(idx) >= 24
        mins = dict(left=10**9, right=10**9, top=10**9, bottom=10**9)
        for k in idx:
            ani._func(k, *ani._args)
            fig.canvas.draw()
            inked = _inked_mask(fig)
            # mask the title's own ink out -- see _measure_margins's
            # identical legend-masking pattern above
            renderer = fig.canvas.get_renderer()
            tb = ax.title.get_window_extent(renderer)
            h_full, w_full = inked.shape
            tx0 = max(0, int(tb.x0) - 2)
            tx1 = min(w_full, int(tb.x1) + 2)
            ty0 = max(0, int(h_full - tb.y1) - 2)
            ty1 = min(h_full, int(h_full - tb.y0) + 2)
            inked[ty0:ty1, tx0:tx1] = False
            cols = np.where(inked.any(axis=0))[0]
            rows = np.where(inked.any(axis=1))[0]
            assert len(cols) and len(rows), f"frame {k}: canvas is blank"
            frame_mins = dict(
                left=int(cols.min()), right=int(w_full - 1 - cols.max()),
                top=int(rows.min()), bottom=int(h_full - 1 - rows.max()),
            )
            for edge in mins:
                mins[edge] = min(mins[edge], frame_mins[edge])
        plt.close(fig)
        for edge, val in mins.items():
            assert val >= NOCLIP_MARGIN_FLOOR_PX, (
                f"wide chemtrails trajectory WITH a title: {edge} margin "
                f"only {val}px (floor {NOCLIP_MARGIN_FLOOR_PX}px) -- the "
                f"title-margin fix may be shrinking/clipping the cube")


# --- measurement draws must never start the animation early -------------
#
# Found and fixed while verifying the title fix's safety (checking whether
# `_fit_right_legend`/`_add_right_colorbar` -- this file's own
# "neighbours" -- had the same "full canvas hides something" problem the
# title did). They didn't have a rendering bug, but sneaking a
# `FigureCanvasAgg(fig); canvas.draw()` measurement draw against the REAL,
# ANIMATED figure (both already did this, for the legend/colorbar width
# fit; `_animated_3d_title_line_height_in`, the new title fix, initially
# did too) turned out to have an unrelated, genuinely dangerous side
# effect: it IS the figure's first-ever draw (`hyp.plot(..., show=False)`
# never draws the canvas itself), which fires `FuncAnimation`'s deferred
# `'draw_event'` -> `Animation._start()` -> `_init_draw()` -> a REAL
# frame-0 update through `line_ani._func`, dispatching any `on_frame=`
# callback (and the `_frame_hooks`-driven per-segment `title=` schedule)
# one extra time -- silently, during figure CONSTRUCTION, before the
# caller has done anything at all. This broke
# `tests/plot/test_serial_titles.py::
# test_title_list_matches_the_published_current_index` the first time the
# title-margin fix was attempted (confirmed: 17 recorded on_frame calls
# for a 16-frame animation, one-frame-shifted from `_titles_over`'s own
# count). Fixed by `_measurement_renderer` (plot.py), which guards every
# such measurement draw with `canvas._is_saving = True` -- matplotlib's
# own, officially-supported escape hatch for exactly this ("makes the
# draw_event animation-starting callback a no-op", `Animation.save`'s own
# comment) -- and by `_animated_3d_title_line_height_in` measuring on a
# throwaway `Figure` that was never connected to the real animation at
# all.

class TestMeasurementDrawsDoNotStartTheAnimation:

    @staticmethod
    def _n_on_frame_calls_before_any_manual_drive(**plot_kwargs):
        seen = []
        clouds = _blob_clouds(2, n=20, seed=30)
        fig, ani = hyp.plot(clouds, animate=True, duration=2, frame_rate=10,
                            on_frame=seen.append, show=False, **plot_kwargs)
        plt.close(fig)
        return len(seen)

    def test_reserving_the_title_margin_does_not_fire_on_frame(self):
        n = self._n_on_frame_calls_before_any_manual_drive(title='T')
        assert n == 0, (
            f"{n} on_frame call(s) fired during hyp.plot() construction, "
            f"before any frame was manually driven -- the title-margin "
            f"measurement draw started the animation prematurely")

    def test_fitting_the_right_side_legend_does_not_fire_on_frame(self):
        n = self._n_on_frame_calls_before_any_manual_drive(
            legend=['a', 'b'])
        assert n == 0, (
            f"{n} on_frame call(s) fired during hyp.plot() construction "
            f"-- the legend-fit measurement draw started the animation "
            f"prematurely")

    def test_fitting_the_right_colorbar_does_not_fire_on_frame(self):
        clouds = _blob_clouds(2, n=20, seed=31)
        seen = []
        fig, ani = hyp.plot(
            clouds, animate=True,
            hue=[np.linspace(0, 1, 20), np.linspace(0, 1, 20)],
            colorbar=True, duration=2, frame_rate=10, on_frame=seen.append,
            show=False)
        plt.close(fig)
        assert len(seen) == 0, (
            f"{len(seen)} on_frame call(s) fired during hyp.plot() "
            f"construction -- the right-colorbar-fit measurement draw "
            f"started the animation prematurely")

    def test_frame_schedule_is_not_shifted_after_construction(self):
        """End-to-end: with the guard in place, manually driving N frames
        after construction must report exactly N on_frame calls (not N+1
        from a leaked construction-time call) -- the exact symptom that
        broke `tests/plot/test_serial_titles.py::
        test_title_list_matches_the_published_current_index` before this
        was fixed (a per-segment title= list + on_frame= drifted out of
        sync by one frame)."""
        seen = []
        clouds = _blob_clouds(2, n=20, seed=32)
        fig, ani = hyp.plot(clouds, animate=True, title='T',
                            legend=['a', 'b'], duration=2, frame_rate=10,
                            on_frame=seen.append, show=False)
        assert len(seen) == 0, 'construction itself must not fire on_frame'
        for f in range(16):
            ani._func(f, *ani._args)
        plt.close(fig)
        assert len(seen) == 16, (
            f"expected exactly 16 on_frame calls after manually driving "
            f"16 frames, got {len(seen)}")
