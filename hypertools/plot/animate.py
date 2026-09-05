#!/usr/bin/env python
"""Animation save helpers for hypertools' plotting backend.

Extracted from `plot/plot.py`: given a matplotlib `FuncAnimation`, dispatch
to the appropriate writer based on the target file extension (svg/gif/png
or apng/mp4 and friends).
"""

import numpy as np
import matplotlib.animation as animation
from .._shared.animated_svg import combine_frames_svg


class HyperFuncAnimation(animation.FuncAnimation):
    """``matplotlib.animation.FuncAnimation`` whose ``_stop`` is idempotent.

    matplotlib's notebook backend (``nbAgg``, the backend hypertools selects
    in Colab and classic Jupyter) processes a figure's close event
    re-entrantly: ``FigureManagerNbAgg.destroy()`` calls ``clearup_closed()``,
    which fires ``close_event`` while the manager's own comm-close handler is
    still being processed, so every ``close_event`` callback runs twice.
    ``Animation._stop`` is one of them and is not idempotent: the second call
    finds ``event_source`` already ``None`` and raises ``AttributeError:
    'NoneType' object has no attribute 'remove_callback'``. Reproduced
    2026-09-04 with plain matplotlib 3.10.8 and 3.11.1 (no hypertools
    involved) and, on Colab, with every hypertools animation: the next
    static-plot cell failed in IPython's end-of-cell figure flush
    (``plt.close('all')``) and ``show=False`` animated plots failed inside
    ``plot()`` itself, at its own ``plt.close(fig)``. Skipping the repeat
    call is what ``_stop`` would do itself if it checked; nothing else about
    the animation changes (``isinstance(x, FuncAnimation)`` still holds).
    """

    def _stop(self, *args):
        if self.event_source is None:      # already stopped: re-entered close
            return
        super()._stop(*args)


# video containers the ffmpeg writer (h264) can mux into
_FFMPEG_EXTENSIONS = ('mp4', 'mov', 'avi', 'm4v', 'mkv')

#: x264 constant-rate-factor for video exports. 23 is x264's own default: a
#: quality target, so the file size follows the CONTENT (a 560-px line plot
#: on white encodes at a fraction of the bits a 1400-px one needs). Until
#: 1.1 every video was written at a fixed ``bitrate=1800`` kbit/s, which made
#: a two-minute clip 27 MB whatever its size or content (measured 2026-09-04:
#: the same weather animation was 27.2 MB at 1400x700 and 26.2 MB at
#: 980x490), and starved a large figure while over-spending on a small one.
VIDEO_CRF = 23


def _ffmpeg_quality_kwargs():
    """Writer kwargs for a quality-targeted (CRF) h264 encode, shared by
    ``_save_animation`` and the streaming recorder so both agree."""
    return dict(codec='h264', extra_args=['-crf', str(VIDEO_CRF)])
# every extension _save_animation understands, for error messages
_SUPPORTED_ANIMATION_EXTENSIONS = (
    '.gif, .png/.apng (animated PNG), .svg (animated vector graphics), '
    'and -- with FFmpeg installed -- '
    + '/'.join('.' + e for e in _FFMPEG_EXTENSIONS)
)


class _RealTimePillowWriter(animation.PillowWriter):
    """PillowWriter whose per-frame delays cumulatively round onto the
    format's timing grid, so total playback matches the requested duration.

    GIF stores per-frame delays in centiseconds; the stock PillowWriter
    writes a single ``int(1000 / fps)`` delay for every frame, so e.g. the
    default ``frame_rate=30`` (33.33 ms/frame) lands on the 10 ms grid as
    30 ms/frame and the whole animation plays ~10% fast (release-1.0 audit,
    F04-010: 27 s for a requested 30 s). Cumulative rounding instead
    alternates 30/40 ms so every frame's cumulative timestamp stays within
    half a grid step of exact -- total wall-clock error is bounded by
    ``grid_ms / 2`` regardless of length. ``grid_ms=10`` matches GIF;
    APNG delays are stored in (at least) milliseconds, so ``grid_ms=1``.
    """

    def __init__(self, *args, grid_ms=10, **kwargs):
        super().__init__(*args, **kwargs)
        self._grid_ms = max(1, int(grid_ms))

    def finish(self):
        """Write the buffered frames with cumulative-rounded per-frame
        durations (see class docstring) instead of PillowWriter's single
        truncated duration, so total playback time stays within grid_ms/2
        of the requested ``len(frames) / fps`` seconds."""
        per_frame_ms = 1000.0 / self.fps
        grid = self._grid_ms
        durations, prev = [], 0
        for i in range(1, len(self._frames) + 1):
            cum = int(round(i * per_frame_ms / grid)) * grid
            durations.append(cum - prev)
            prev = cum
        self._frames[0].save(
            self.outfile, save_all=True, append_images=self._frames[1:],
            duration=durations, loop=0)


def _save_animation(line_ani, save_path, frame_rate, dpi=None):
    """Save a matplotlib animation, choosing the writer by file extension.

    .gif and .png/.apng use PillowWriter (no ffmpeg required; Pillow writes
    animated PNGs when the extension is .png/.apng); .svg builds a
    SMIL-animated vector file; .mp4/.mov/.avi/.m4v/.mkv use the ffmpeg
    writer. Any other extension raises ``ValueError`` naming the supported
    formats (release-1.0 audit, F04-011/F05-013: unknown extensions -- and
    paths with NO extension -- previously fell through to ffmpeg and
    surfaced as a raw ``CalledProcessError`` dumping the ffmpeg command
    line).

    ``dpi`` is handed to the raster and video writers exactly as
    ``matplotlib.animation.Animation.save`` takes it; ``None`` keeps the
    figure's own dpi. The SVG writer is vector and ignores it.
    """
    # gif / apng / video writers save EVERY animation frame (no subsampling),
    # with per-frame delays that cumulatively track 1000/frame_rate ms (see
    # _RealTimePillowWriter), so an exported file plays in real time: its
    # save_count (== frame_rate * duration) frames total ~= duration seconds
    # within the format's timing grid. Only the vector (SVG) writer
    # subsamples, to bound file size. Do not subsample the raster/video
    # paths or playback would run too fast.
    import os
    save_path = os.fspath(save_path)  # pathlib.Path works too (F09-004)
    ext = os.path.splitext(save_path)[1].lower().lstrip('.')
    if ext == 'svg':
        _save_animated_svg(line_ani, save_path, frame_rate)
    elif ext == 'gif':
        line_ani.save(save_path, dpi=dpi,
                      writer=_RealTimePillowWriter(fps=frame_rate,
                                                   grid_ms=10))
    elif ext in ('png', 'apng'):
        # Pillow emits an animated PNG (APNG) for multi-frame PNG saves,
        # but only recognizes the .png extension -- write to a UNIQUE
        # temporary .png in the target directory and rename it onto the
        # requested name. (Release-1.0 audit, F09-002: writing to
        # `save_path[:-5] + '.png'` silently destroyed a pre-existing
        # sibling `.png` file whenever the caller asked for `.apng`.)
        import tempfile
        target_dir = os.path.dirname(os.path.abspath(save_path))
        fd, tmp_path = tempfile.mkstemp(suffix='.png', dir=target_dir)
        os.close(fd)
        try:
            line_ani.save(tmp_path, dpi=dpi,
                          writer=_RealTimePillowWriter(fps=frame_rate,
                                                       grid_ms=1))
            # mkstemp's private 0600 mode must not leak onto the saved
            # animation: preserve an existing target's mode / honor the
            # umask for new files (release-1.0 audit: security re-review
            # of F09-002; shared with hyp.save's atomic-write path)
            from ..io.save import _transfer_file_mode
            _transfer_file_mode(tmp_path, save_path)
            os.replace(tmp_path, save_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    elif ext in _FFMPEG_EXTENSIONS:
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=frame_rate, **_ffmpeg_quality_kwargs())
        line_ani.save(save_path, writer=writer, dpi=dpi)
    else:
        what = f"extension {'.' + ext!r}" if ext else "missing extension"
        raise ValueError(
            f"unsupported animation save format ({what}) for "
            f"{save_path!r}; supported extensions are "
            f"{_SUPPORTED_ANIMATION_EXTENSIONS}. (For a static image "
            "instead, pass animate=False.)"
        )


class _SVGFrameCollector(animation.AbstractMovieWriter):
    """Movie 'writer' that captures animation frames as SVG documents and
    assembles them into a single SMIL-animated SVG (vector output)."""

    def __init__(self, fps, max_frames=60):
        super().__init__(fps=fps)
        self.max_frames = max_frames
        self.frames = []
        self._seen = 0
        self._stride = 1

    def setup(self, fig, outfile, dpi=None):
        """Record the figure, output path, and DPI to use for captured frames."""
        self.fig = fig
        self.outfile = outfile
        self.dpi = dpi if dpi is not None else fig.dpi

    @property
    def frame_size(self):
        """`(width_px, height_px)` of a captured frame, derived from the figure size and DPI."""
        w, h = self.fig.get_size_inches()
        return int(w * self.dpi), int(h * self.dpi)

    def set_stride(self, total_frames):
        """Set the frame-capture stride so at most `self.max_frames` of `total_frames` are kept."""
        self._stride = max(1, int(np.ceil(total_frames / self.max_frames)))

    def grab_frame(self, **savefig_kwargs):
        """Capture the current figure as an SVG frame, honoring the stride set by `set_stride`."""
        if self._seen % self._stride == 0:
            import io
            buf = io.StringIO()
            self.fig.savefig(buf, format='svg', **savefig_kwargs)
            self.frames.append(buf.getvalue())
        self._seen += 1

    def finish(self):
        """Combine all captured SVG frames into one SMIL-animated SVG and write it to `self.outfile`."""
        duration = max(1.0, len(self.frames) * self._stride / self.fps)
        with open(self.outfile, 'w') as f:
            f.write(combine_frames_svg(self.frames, duration))


def _save_animated_svg(line_ani, save_path, frame_rate):
    """Save a matplotlib animation as a SMIL-animated (vector) SVG. Frames
    are subsampled to at most ~60 to keep file sizes reasonable."""
    collector = _SVGFrameCollector(fps=frame_rate)
    total = getattr(line_ani, '_save_count', None) or \
        getattr(line_ani, 'save_count', None) or 100
    collector.set_stride(total)
    line_ani.save(save_path, writer=collector)


#: How many frames `drawn_extent` samples when the caller does not say.
#: The drawn extent of an animated 3-D plot changes with the camera angle,
#: so it has to be measured over the orbit rather than on one frame; 12
#: samples cover a full turn's 90-degree symmetry period three times, and
#: cost 12 canvas renders rather than one per frame.
DRAWN_EXTENT_SAMPLES = 12


def drawn_extent(anim, frames=None, threshold=5):
    """The union bounding box of everything an animation DRAWS, measured
    from rendered pixels (GH #285).

    Parameters
    ----------
    anim : hypertools.plot.hyper_animation.HyperAnimation
        The animation to measure. Reached as ``anim.drawn_extent(...)``.
    frames : int, iterable of int, or None, optional
        Which frames to measure. ``None`` (default) samples
        `DRAWN_EXTENT_SAMPLES` frames evenly across the animation; an int
        `n` samples `n` frames evenly; an iterable measures exactly those
        frame indices.
    threshold : int, optional
        How far (0-255, per channel) a pixel must differ from the figure's
        own background colour to count as drawn. The default 5 treats
        anything below 250 on a white figure as ink -- faint antialiasing
        included.

    Returns
    -------
    matplotlib.transforms.Bbox
        In FIGURE fractions: ``(0, 0)`` is the bottom-left corner of the
        canvas and ``(1, 1)`` the top-right, y increasing UPWARD (the
        `Figure.transFigure` convention, not the top-down pixel one). Read
        the corners as ``bbox.p0`` / ``bbox.p1``, or use ``bbox.x0``,
        ``bbox.width`` and friends. Convert to axes fractions with
        ``bbox.transformed(ax.transAxes.inverted())``.

    Notes
    -----
    COST: one full canvas render per sampled frame, plus one pass over the
    rendered pixels. On a 640x504 figure that is a few milliseconds each;
    on a large figure with surfaces it is not cheap, which is why the
    default samples 12 frames rather than all of them.

    This DRAWS the sampled frames, so the figure is left showing the last
    one measured. Frames are idempotent and order-independent by contract
    (see `FrameContext`), so redraw whichever frame you want afterwards
    with ``anim.draw_frame(i)``.

    Measuring pixels rather than asking the artists is deliberate: an
    animated 3-D plot's wireframe box is re-projected every frame from the
    camera angle, and its drawn size has no artist-level accessor that
    accounts for the projection. The pixels are the one record of what was
    actually drawn.
    """
    import numpy as _np
    from matplotlib.transforms import Bbox
    from matplotlib.colors import to_rgb

    fig = anim.figure
    n_frames = anim.n_frames
    if frames is None:
        frames = DRAWN_EXTENT_SAMPLES
    if isinstance(frames, (int, _np.integer)) and not isinstance(frames, bool):
        count = max(1, min(int(frames), n_frames))
        frames = sorted({int(round(v)) for v in
                         _np.linspace(0, n_frames - 1, count)})
    else:
        frames = [int(f) for f in frames]
    if not frames:
        raise ValueError(
            "drawn_extent needs at least one frame to measure; got an empty "
            "frames= sequence.")

    bg = _np.array([round(c * 255) for c in to_rgb(fig.get_facecolor())])
    lo = _np.full(2, _np.inf)
    hi = _np.full(2, -_np.inf)
    for i in frames:
        anim.draw_frame(i)                    # validates the index for us
        fig.canvas.draw()
        rgb = _np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(int)
        ink = _np.abs(rgb - bg).max(-1) > threshold
        rows, cols = _np.nonzero(ink)
        if rows.size == 0:
            continue                          # a blank frame draws nothing
        n_rows, n_cols = ink.shape
        lo = _np.minimum(lo, [cols.min() / n_cols,
                              1 - (rows.max() + 1) / n_rows])
        hi = _np.maximum(hi, [(cols.max() + 1) / n_cols,
                              1 - rows.min() / n_rows])
    if not _np.isfinite(lo).all():
        raise ValueError(
            "drawn_extent found nothing drawn on any of the "
            f"{len(frames)} sampled frames: every pixel matched the "
            "figure's background. Raise threshold= only makes this "
            "stricter -- check that the animation actually draws something "
            "at these frame indices.")
    return Bbox([list(lo), list(hi)])


# public alias
save_animation = _save_animation
