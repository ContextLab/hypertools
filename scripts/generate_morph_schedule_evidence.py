"""Regenerate the ``animate='morph'`` constant-rotation-speed evidence
grids (maintainer request, 2026-07-06 -- fix R1: "the rotation speed
should always be constant -- so more rotations means more time spent on
that part of the animation", i.e. segment duration proportional to
rotation count).

Drives the REAL `hyp.plot(..., animate='morph', rotations=[1, 0.25, 2,
0.25, 1])` code path for both backends -- the matplotlib `FuncAnimation`
frame updater and the plotly `go.Frame` list -- so the evidence reflects
exactly what a user gets, never a hand-rolled reimplementation of the
schedule. Checkpoint frame indices are recomputed from the ACTUAL (now
non-uniform, rotation-proportional) `frame_counts` the fix produces,
which are very different from the old equal-per-segment split.

Run from the repo root:
    .venv/bin/python scripts/generate_morph_schedule_evidence.py

Outputs:
    docs/images/v1.0-seven-features/morph_anim_mpl.png
    docs/images/v1.0-seven-features/morph_anim_plotly.png
"""

import copy
import io
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go

import hypertools as hyp
from hypertools.plot import morph as _morph

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'docs', 'images', 'v1.0-seven-features')
ROTATIONS = [1, 0.25, 2, 0.25, 1]
N_DATASETS = 3
DURATION, FRAME_RATE = 6, 30  # 180 frames -- divides [1, 0.25, 2, 0.25, 1]
                              # (sum 4.5) with NO rounding at all, so
                              # deg/frame comes out exactly constant
AZIM0 = -60


def _blobs(seed, n=150, spacing=6.0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((n, 3)) + spacing * i
            for i in range(N_DATASETS)]


def _checkpoints(frame_counts):
    """(label, frame_index, segment_index) for the 6 evidence panels --
    hold1 / mid-morph1 / hold2 / mid-morph2 / hold3 / final -- recomputed
    against the ACTUAL per-segment `frame_counts` (no longer uniform)."""
    c = frame_counts
    total = sum(c)
    checkpoints = [
        ('hold1', 0),
        ('mid-morph1', c[0] + c[1] // 2),
        ('hold2', c[0] + c[1]),
        ('mid-morph2', c[0] + c[1] + c[2] + c[3] // 2),
        ('hold3', c[0] + c[1] + c[2] + c[3]),
        ('final', total - 1),
    ]
    out = []
    for label, k in checkpoints:
        seg_idx, _, _ = _morph.frame_to_segment(c, k)
        out.append((label, k, seg_idx))
    return out


def _assemble_grid(panels, title, out_path):
    w, h = panels[0].size
    pad_top = 70
    grid = Image.new('RGB', (w * 3, h * 2 + pad_top), 'white')
    for i, im in enumerate(panels):
        x, y = (i % 3) * w, pad_top + (i // 3) * h
        grid.paste(im, (x, y))
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype(
            '/System/Library/Fonts/Supplemental/Arial.ttf', 24)
    except OSError:
        font = ImageFont.load_default()
    draw.text((20, 18), title, fill='black', font=font)
    grid.save(out_path)
    print(f'wrote {out_path} {grid.size}')


def generate_mpl():
    data = _blobs(seed=2)
    fig, ani = hyp.plot(data, '.', animate='morph', rotations=ROTATIONS,
                        duration=DURATION, frame_rate=FRAME_RATE, show=False)
    morph_state = ani._args[0]
    frame_counts = morph_state['frame_counts']
    azimuths = ani._args[2]
    checkpoints = _checkpoints(frame_counts)
    print('mpl frame_counts:', frame_counts)

    ax = fig.axes[0]
    panels = []
    for label, k, seg_idx in checkpoints:
        ani._func(k, *ani._args)
        azim = azimuths[k]
        ax.set_title(
            f"{label}: seg {seg_idx}, frame {k}, azim={azim:.1f}deg",
            fontsize=14)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        panels.append(Image.open(buf).convert('RGB'))
    plt.close(fig)

    title = (f"animate='morph' (matplotlib) -- rotations={ROTATIONS} -- "
             "constant rotation speed (fix R1)")
    _assemble_grid(panels, title,
                   os.path.join(OUT, 'morph_anim_mpl.png'))
    return frame_counts


def generate_plotly():
    data = _blobs(seed=3)
    fig = hyp.plot(data, '.', animate='morph', backend='plotly',
                   rotations=ROTATIONS, duration=DURATION,
                   frame_rate=FRAME_RATE, show=False)
    total_frames = DURATION * FRAME_RATE
    frame_counts, _, _ = _morph.morph_schedule(
        N_DATASETS, total_frames, ROTATIONS, AZIM0)
    checkpoints = _checkpoints(frame_counts)
    print('plotly frame_counts:', frame_counts)

    base_data = list(fig.data)
    panels = []
    for label, k, seg_idx in checkpoints:
        frame = fig.frames[k]
        data_k = copy.deepcopy(base_data)
        for trace_idx, new_trace in zip(frame.traces, frame.data):
            # R2 fix: MERGE the frame's (partial) trace update onto a copy
            # of the base trace, rather than replacing it wholesale --
            # `go.Frame` updates only specify what CHANGES per frame (e.g.
            # the morph trace's frame data omits `marker.size`, matching
            # real Plotly.js `animate()` semantics, which merges frame
            # attributes onto the current trace state instead of
            # discarding everything else). A wholesale replace silently
            # reverted `marker.size` to plotly's own default (much bigger
            # than hypertools' now-correctly-small morph dots), masking the
            # R2 marker-size fix in this evidence script's reconstructed
            # stills even though real animated playback is unaffected.
            merged = copy.deepcopy(data_k[trace_idx])
            merged.update(new_trace)
            data_k[trace_idx] = merged
        layout = copy.deepcopy(fig.layout)
        if frame.layout is not None and frame.layout.scene is not None:
            layout.scene.camera = frame.layout.scene.camera
        azim_deg = None
        eye = layout.scene.camera.eye
        azim_deg = np.degrees(np.arctan2(eye.y, eye.x))
        still = go.Figure(data=data_k, layout=layout)
        still.update_layout(
            showlegend=False, width=520, height=470,
            margin=dict(l=10, r=10, t=60, b=10),
            updatemenus=[],
            title=dict(text=(f"{label}: seg {seg_idx}, frame {k}, "
                             f"azim~={azim_deg:.1f}deg"), font=dict(size=13),
                       x=0.02, xanchor='left'))
        png = still.to_image(format='png', scale=2)
        panels.append(Image.open(io.BytesIO(png)).convert('RGB'))

    title = (f"animate='morph' (plotly) -- rotations={ROTATIONS} -- "
             "constant rotation speed (fix R1)")
    _assemble_grid(panels, title,
                   os.path.join(OUT, 'morph_anim_plotly.png'))
    return frame_counts


def main():
    os.makedirs(OUT, exist_ok=True)
    mpl_counts = generate_mpl()
    plotly_counts = generate_plotly()

    print()
    print('| segment | rotations | frames (mpl) | frames (plotly) | '
         'deg/frame (mpl) |')
    print('|-|-|-|-|-|')
    for i, r in enumerate(ROTATIONS):
        dpf = 360.0 * r / mpl_counts[i]
        print(f'| {i} | {r} | {mpl_counts[i]} | {plotly_counts[i]} | '
             f'{dpf:.3f} |')


if __name__ == '__main__':
    main()
