"""Shapes-zoo morphing animation (PR #270 round-4 demo).

Loops through the shapes zoo (bunny -> cube -> dragon -> sphere -> teapot ->
vase -> biplane): n points are sampled without replacement from each shape
(n = smallest shape's point count), consecutive shapes are matched with the
Hungarian algorithm (scipy.optimize.linear_sum_assignment), and each point
travels smoothly to its matched position across n_steps morph frames. The
sequence alternates hold segments (shape at rest) and morph segments, while
the camera completes 2 * n_shapes - 1 total rotations (one per segment).

Run from the repo root:
    .venv/bin/python scripts/generate_shape_morph.py

Output: docs/images/v2.0-animations/shapes_morph.gif
"""

import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import hypertools as hyp

SHAPES = ['bunny', 'cube', 'dragon', 'sphere', 'teapot', 'vase', 'biplane']
N_STEPS = 270         # frames per hold segment and per morph segment
FRAME_RATE = 30
SEED = 42
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'docs', 'images', 'v2.0-animations', 'shapes_morph.mp4')


def normalize_shape(points):
    """Center and scale a point cloud into the hypertools [-1, 1] cube."""
    points = np.asarray(points, dtype=np.float64)
    points = points - points.mean(axis=0)
    return points / np.abs(points).max()


def main():
    rng = np.random.default_rng(SEED)

    clouds = []
    for name in SHAPES:
        data = hyp.load(name)
        points = np.asarray(data, dtype=np.float64) \
            if not hasattr(data, 'values') else data.values.astype(np.float64)
        clouds.append(normalize_shape(points))

    n = min(len(c) for c in clouds)
    print(f'{len(SHAPES)} shapes; sampling n={n} points from each')
    sampled = [c[rng.choice(len(c), size=n, replace=False)] for c in clouds]

    # Hungarian matching: reorder each next shape so point i morphs to its
    # optimally assigned partner (minimum total travel distance)
    for i in range(len(sampled) - 1):
        cost = cdist(sampled[i], sampled[i + 1])
        _, col_ind = linear_sum_assignment(cost)
        sampled[i + 1] = sampled[i + 1][col_ind]
        print(f'  matched {SHAPES[i]} -> {SHAPES[i + 1]} '
              f'(mean travel {cost[np.arange(n), col_ind].mean():.3f})')

    # frame schedule: hold, morph, hold, morph, ... (2 * n_shapes - 1 segs)
    segments = []
    for i in range(len(sampled)):
        segments.append(('hold', sampled[i], sampled[i]))
        if i < len(sampled) - 1:
            segments.append(('morph', sampled[i], sampled[i + 1]))
    total_frames = len(segments) * N_STEPS
    rotations = len(segments)  # == 2 * len(SHAPES) - 1

    # hypertools-styled scene: draw the first cloud, then animate the artist
    geo = hyp.plot(sampled[0], 'k.', show=False)
    fig, ax = geo.fig, geo.ax
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    point_artist = ax.get_lines()[0]
    point_artist.set_markersize(1.5)

    def update(frame):
        seg_idx, step = divmod(frame, N_STEPS)
        kind, start, end = segments[seg_idx]
        t = step / max(1, N_STEPS - 1) if kind == 'morph' else 0.0
        # smoothstep easing for a natural morph
        t = t * t * (3 - 2 * t)
        pts = (1 - t) * start + t * end
        point_artist.set_data(pts[:, 0], pts[:, 1])
        point_artist.set_3d_properties(pts[:, 2])
        ax.view_init(elev=10,
                     azim=-60 + 360.0 * rotations * frame / total_frames)
        return (point_artist,)

    ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                  interval=1000 / FRAME_RATE, blit=False)
    Writer = animation.writers['ffmpeg']
    ani.save(OUT, writer=Writer(fps=FRAME_RATE, bitrate=1800))
    plt.close('all')
    print(f'saved {OUT} ({os.path.getsize(OUT) // 1024}KB, '
          f'{total_frames} frames, {rotations} rotations)')


if __name__ == '__main__':
    main()
