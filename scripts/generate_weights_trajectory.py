"""The classic hypertools story-trajectories animation (readthedocs
hypertools.gif), reconstructed from the 2020 pieman_trajectory_demo notebook
(hypertools 0.6.2 + timecorr):

    smooth (gaussian, var=300) -> align x n_iter -> smooth -> UMAP -> animate

Run from the repo root:
    .venv/bin/python scripts/generate_weights_trajectory.py

Output: docs/images/v2.0-animations/weights_hyperaligned.gif
"""

import os
import subprocess

import numpy as np
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use('Agg')
import hypertools as hyp

N_ITER = 20
KERNEL_VAR = 300
ALIGN = 'SRM'
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'docs', 'images', 'v2.0-animations')


def smooth(datasets, var=KERNEL_VAR):
    """Gaussian temporal smoothing (timecorr-style, variance in timepoints)."""
    return [gaussian_filter1d(np.asarray(d, dtype=np.float64),
                              sigma=np.sqrt(var), axis=0)
            for d in datasets]


def main():
    data = hyp.load('weights').get_data()

    data = smooth(data)                                # round 1: raw data
    data = hyp.align(data, align=ALIGN, n_iter=N_ITER)  # repeated alignment
    data = smooth(data)                                # round 2: aligned data

    mp4 = os.path.join(OUT_DIR, '_weights_tmp.mp4')
    gif = os.path.join(OUT_DIR, 'weights_hyperaligned.gif')
    hyp.plot(data,
             # moderate UMAP neighborhoods (36) rope same-timepoint rows
             # across subjects together while preserving the trajectory's
             # dramatic bend/loop; the modern default (15) disperses the
             # bundle into a hairball, and large values (150+) over-globalize
             # the embedding and flatten the loop into a near-straight line
             reduce={'model': 'UMAP',
                     'params': {'n_neighbors': 36, 'min_dist': 0.1,
                                'random_state': 42}},
             # spin (not window) so the complete looping bundle stays on
             # screen and orbits in 3D -- the classic readthedocs gif style;
             # a rolling window would only ever show a tangled fragment
             animate='spin', duration=30, frame_rate=30,
             rotations=1, linewidth=3, zoom=2, size=[8, 6],
             save_path=mp4, show=False)
    # the dense spinning bundle (36 fat trajectories every frame) is large;
    # scale=340 + a 48-colour palette keeps it near the other animation gifs
    # (~7 MB) without visible loss, still 30 fps with no frame downsampling
    subprocess.run(
        ['ffmpeg', '-y', '-i', mp4, '-vf',
         'fps=30,scale=340:-1:flags=lanczos,split[s0][s1];'
         '[s0]palettegen=max_colors=48:stats_mode=diff[p];'
         '[s1][p]paletteuse=dither=none', gif],
        check=True, capture_output=True)
    os.remove(mp4)
    print(f'saved {gif} ({os.path.getsize(gif) // 1024}KB)')


if __name__ == '__main__':
    main()
