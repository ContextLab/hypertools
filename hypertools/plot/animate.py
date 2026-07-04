#!/usr/bin/env python
"""Animation save helpers for hypertools' plotting backend.

Extracted from `plot/plot.py`: given a matplotlib `FuncAnimation`, dispatch
to the appropriate writer based on the target file extension (svg/gif/png
or apng/mp4 and friends).
"""

import numpy as np
import matplotlib.animation as animation
from .._shared.animated_svg import combine_frames_svg


def _save_animation(line_ani, save_path, frame_rate):
    """Save a matplotlib animation, choosing the writer by file extension.

    .gif and .png/.apng use PillowWriter (no ffmpeg required; Pillow writes
    animated PNGs when the extension is .png/.apng); .mp4/.mov/.avi and
    anything else use the ffmpeg writer, matching hypertools' historical
    behavior.
    """
    ext = save_path.lower().rsplit('.', 1)[-1]
    if ext == 'svg':
        _save_animated_svg(line_ani, save_path, frame_rate)
    elif ext == 'gif':
        line_ani.save(save_path, writer=animation.PillowWriter(fps=frame_rate))
    elif ext in ('png', 'apng'):
        # Pillow emits an animated PNG (APNG) for multi-frame PNG saves,
        # but only recognizes the .png extension -- write to .png and
        # rename if the caller asked for .apng
        import os
        target = save_path
        if ext == 'apng':
            target = save_path[:-5] + '.png'
        line_ani.save(target, writer=animation.PillowWriter(fps=frame_rate))
        if target != save_path:
            os.replace(target, save_path)
    else:
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=frame_rate, bitrate=1800)
        line_ani.save(save_path, writer=writer)


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
        self.fig = fig
        self.outfile = outfile
        self.dpi = dpi if dpi is not None else fig.dpi

    @property
    def frame_size(self):
        w, h = self.fig.get_size_inches()
        return int(w * self.dpi), int(h * self.dpi)

    def set_stride(self, total_frames):
        self._stride = max(1, int(np.ceil(total_frames / self.max_frames)))

    def grab_frame(self, **savefig_kwargs):
        if self._seen % self._stride == 0:
            import io
            buf = io.StringIO()
            self.fig.savefig(buf, format='svg', **savefig_kwargs)
            self.frames.append(buf.getvalue())
        self._seen += 1

    def finish(self):
        from .._shared.animated_svg import combine_frames_svg
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
