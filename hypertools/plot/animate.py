#!/usr/bin/env python
"""Animation save helpers for hypertools' plotting backend.

Extracted from `plot/plot.py`: given a matplotlib `FuncAnimation`, dispatch
to the appropriate writer based on the target file extension (svg/gif/png
or apng/mp4 and friends).
"""

import numpy as np
import matplotlib.animation as animation
from .._shared.animated_svg import combine_frames_svg


# video containers the ffmpeg writer (h264) can mux into
_FFMPEG_EXTENSIONS = ('mp4', 'mov', 'avi', 'm4v', 'mkv')
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


def _save_animation(line_ani, save_path, frame_rate):
    """Save a matplotlib animation, choosing the writer by file extension.

    .gif and .png/.apng use PillowWriter (no ffmpeg required; Pillow writes
    animated PNGs when the extension is .png/.apng); .svg builds a
    SMIL-animated vector file; .mp4/.mov/.avi/.m4v/.mkv use the ffmpeg
    writer. Any other extension raises ``ValueError`` naming the supported
    formats (release-1.0 audit, F04-011/F05-013: unknown extensions -- and
    paths with NO extension -- previously fell through to ffmpeg and
    surfaced as a raw ``CalledProcessError`` dumping the ffmpeg command
    line).
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
        line_ani.save(save_path,
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
            line_ani.save(tmp_path,
                          writer=_RealTimePillowWriter(fps=frame_rate,
                                                       grid_ms=1))
            os.replace(tmp_path, save_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    elif ext in _FFMPEG_EXTENSIONS:
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=frame_rate, bitrate=1800)
        line_ani.save(save_path, writer=writer)
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


# public alias
save_animation = _save_animation
