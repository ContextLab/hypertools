# -*- coding: utf-8 -*-
"""Subprocess entry point for plotly animation export.

Renders every frame of a plotly animation figure to an image file, one per
frame, so the parent (``plotly_backend._render_frames_via_subprocess``) can
enforce a HARD wall-clock timeout by killing this process -- and, with it, the
headless Chrome kaleido drives. A blocked native/browser call cannot be
interrupted or reclaimed from a Python thread, so this process boundary is the
only reliable timeout (GH #291 follow-up / Windows-CI kaleido hang).

Usage (invoked by the parent, not by users)::

    python -m hypertools.plot._kaleido_export_worker \
        FIG_JSON OUT_DIR EXT WIDTH HEIGHT

Reads the plotly figure (with its animation frames) from ``FIG_JSON`` and
writes ``OUT_DIR/000000.EXT`` .. ``OUT_DIR/{n-1:06d}.EXT`` -- each rendered to a
temporary ``.part`` file and renamed into place atomically, so a killed render
never leaves a half-written frame the parent could mistake for a complete one.
Exits non-zero (with a traceback on stderr) if any frame fails to render.
"""
import json
import os
import sys


def main(argv):
    """Render all frames of the figure named in `argv` to image files.

    `argv` is ``[fig_json_path, out_dir, ext, width, height]`` (the CLI
    arguments after the module name)."""
    fig_json_path, out_dir, ext = argv[0], argv[1], argv[2]
    width, height = int(argv[3]), int(argv[4])

    import plotly.graph_objects as go

    from .plotly_backend import _frame_snapshots, _shared_kaleido_session

    with open(fig_json_path, encoding='utf-8') as fh:
        fig = go.Figure(json.load(fh))

    # one shared headless-Chrome session for every frame (fast); if it wedges,
    # the parent kills this whole process, so no in-process recovery is needed
    with _shared_kaleido_session():
        for i, snapshot in enumerate(_frame_snapshots(fig)):
            img = snapshot.to_image(format=ext, width=width, height=height)
            part = os.path.join(out_dir, f'.{i:06d}.{ext}.part')
            with open(part, 'wb') as out:
                out.write(img)
            os.replace(part, os.path.join(out_dir, f'{i:06d}.{ext}'))


if __name__ == '__main__':
    main(sys.argv[1:])
