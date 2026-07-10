#!/usr/bin/env python
"""Regenerate the story-trajectories animation assets (GH #275, QC 2026-07).

Produces, under ``docs/``:
    images/v1.0-round17/story_trajectories.mp4   -- the spinning animation
    images/v1.0-round17/story_frame_{early,mid,late}.png -- three camera angles
    _static/thumbnails/sphx_glr_plot_story_trajectories_thumb.gif -- gallery gif

The pipeline is fully deterministic (IncrementalPCA + HyperAlign are both
deterministic; no random_state needed), so re-running reproduces the committed
assets. The gif is transcoded from the mp4 with ffmpeg (palettegen) if ffmpeg
is available; otherwise that step is skipped with a message.

Why this pipeline (see examples/plot_story_trajectories.py for the full write-up):
    * hyperalign in a LOW-dimensional (IncrementalPCA, ndims=10) space with
      n_iter=10 -- NOT the full 100-hub space -- then show the first 3 aligned
      dims; this tightens the subjects' within-timepoint dispersion (their
      spread around a shared centroid, normalized by cloud scale) ~18%
      (0.88 -> 0.73), so they move together.
    * IncrementalPCA (linear) keeps trajectories smooth: the largest per-step
      jump (normalized) is ~0.37 vs ~3.3 for the old UMAP embedding -- UMAP left
      them jumpy and poorly aligned.
"""
import os
import shutil
import subprocess
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import hypertools as hyp

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.normpath(os.path.join(HERE, '..', 'docs'))
IMG_DIR = os.path.join(DOCS, 'images', 'v1.0-round17')
THUMB = os.path.join(DOCS, '_static', 'thumbnails',
                     'sphx_glr_plot_story_trajectories_thumb.gif')

MANIP = [{'model': 'Smooth', 'kwargs': {'kernel_width': 40}},
         {'model': 'Resample', 'kwargs': {'n_samples': 600}},
         'ZScore']
ALIGN = {'model': 'HyperAlign', 'kwargs': {'n_iter': 10}}
NDIMS_ALIGN = 10
DURATION, FRAME_RATE, ZOOM = 9, 30, 1.5


def build_aligned():
    data = hyp.load('weights')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        aligned = hyp.analyze(data, manip=MANIP, reduce='IncrementalPCA',
                              ndims=NDIMS_ALIGN, align=ALIGN)
    return [np.asarray(a)[:, :3] for a in aligned]


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    aligned = build_aligned()
    n = len(aligned)
    colors = [(*plt.get_cmap('gist_rainbow')(i / (n - 1))[:3], 0.5)
              for i in range(n)]

    mp4 = os.path.join(IMG_DIR, 'story_trajectories.mp4')
    anim = hyp.plot(aligned, '-', color=colors, linewidth=1.2, animate='spin',
                    duration=DURATION, frame_rate=FRAME_RATE, zoom=ZOOM,
                    reduce=None, normalize=None, size=[6, 6], show=False)
    anim.save(mp4)
    print('wrote', mp4)

    # three camera-angle stills from the mp4
    if shutil.which('ffmpeg'):
        n_frames = DURATION * FRAME_RATE
        for frac, name in ((0.15, 'early'), (0.5, 'mid'), (0.85, 'late')):
            still = os.path.join(IMG_DIR, f'story_frame_{name}.png')
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', mp4,
                            '-vf', f'select=gte(n\\,{int(n_frames * frac)})',
                            '-vframes', '1', still], check=True)
            print('wrote', still)
        # gallery thumbnail gif (260px, 10fps, palettegen for quality)
        pal = os.path.join(IMG_DIR, '_palette.png')
        vf = 'fps=10,scale=260:-1:flags=lanczos'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', mp4,
                        '-vf', f'{vf},palettegen=stats_mode=diff', pal], check=True)
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', mp4,
                        '-i', pal, '-lavfi',
                        f'{vf}[x];[x][1:v]paletteuse=dither=bayer', THUMB],
                       check=True)
        os.remove(pal)
        print('wrote', THUMB)
    else:
        print('ffmpeg not found; skipped stills + thumbnail gif')


if __name__ == '__main__':
    main()
