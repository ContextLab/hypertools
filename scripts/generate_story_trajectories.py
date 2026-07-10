#!/usr/bin/env python
"""Regenerate the story-trajectories animation assets (GH #275, QC 2026-07).

Produces, under ``docs/``:
    images/v1.0-round17/story_trajectories.mp4   -- the window animation
    images/v1.0-round17/story_frame_{early,mid,late}.png -- three story moments
    _static/thumbnails/sphx_glr_plot_story_trajectories_thumb.gif -- gallery gif

The pipeline is fully deterministic (HyperAlign + IncrementalPCA are both
deterministic), so re-running reproduces the committed assets. The gif is
transcoded from the mp4 with ffmpeg (palettegen) when ffmpeg is available.

Why this pipeline (see examples/plot_story_trajectories.py for the full write-up):
    * ALIGN IN THE 100-HUB FEATURE SPACE FIRST, then reduce to 3-D. Hyperalignment
      needs room -- the full 100-hub space -- to rotate every subject's trajectory
      onto a shared response; reducing to 3-10 dims BEFORE aligning starves it and
      leaves the subjects a poorly-aligned tangle. Aligning in the hub space and
      THEN reducing tightens the subjects' within-timepoint dispersion (their
      spread around a shared centroid, / cloud scale) from ~0.73 to ~0.51.
    * WINDOW animation: a sliding opaque trail traverses each aligned trajectory,
      so you watch all 36 subjects move together through the story.
    * IncrementalPCA (linear) keeps the reduced trajectories smooth.
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
         {'model': 'Resample', 'kwargs': {'n_samples': 300}},
         'ZScore']
ALIGN = {'model': 'HyperAlign', 'kwargs': {'n_iter': 10}}
FRAME_RATE, ZOOM, FOCUSED, DURATION = 30, 1.5, 2.5, 9


def build_reduced():
    """manip -> ALIGN (in 100-hub space) -> reduce to 3-D. The order matters:
    aligning before reducing is what makes the subjects move together."""
    data = hyp.load('weights')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        manip_data = hyp.manip(data, model=MANIP)          # 100-hub space
        aligned = hyp.align(manip_data, align=ALIGN)       # align in 100-hub space
        reduced = hyp.reduce(aligned, reduce='IncrementalPCA', ndims=3)
    return [np.asarray(r)[:, :3] for r in reduced]


def _n_frames(mp4):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                          '-count_frames', '-show_entries', 'stream=nb_read_frames',
                          '-of', 'default=nokey=1:noprint_wrappers=1', mp4],
                         capture_output=True, text=True)
    try:
        return int(out.stdout.strip())
    except ValueError:
        return FRAME_RATE * DURATION


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    reduced = build_reduced()
    n = len(reduced)
    colors = [(*plt.get_cmap('gist_rainbow')(i / (n - 1))[:3], 0.9)
              for i in range(n)]

    mp4 = os.path.join(IMG_DIR, 'story_trajectories.mp4')
    anim = hyp.plot(reduced, '-', color=colors, linewidth=1.5, animate='window',
                    focused=FOCUSED, duration=DURATION, frame_rate=FRAME_RATE,
                    zoom=ZOOM, reduce=None, normalize=None, size=[6, 6], show=False)
    anim.save(mp4)
    print('wrote', mp4)

    if shutil.which('ffmpeg') and shutil.which('ffprobe'):
        nf = _n_frames(mp4)
        # three moments as the story unfolds (window slides start -> end)
        for frac, name in ((0.15, 'early'), (0.5, 'mid'), (0.85, 'late')):
            still = os.path.join(IMG_DIR, f'story_frame_{name}.png')
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', mp4,
                            '-vf', f'select=gte(n\\,{int(nf * frac)})',
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
        print('ffmpeg/ffprobe not found; skipped stills + thumbnail gif')


if __name__ == '__main__':
    main()
