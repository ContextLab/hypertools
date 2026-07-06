# -*- coding: utf-8 -*-
"""
================================
Morphing through the shapes zoo
================================

HyperTools' "shapes zoo" (bunny, cube, dragon, sphere, teapot, vase,
biplane -- see the *A zoo of 3D shapes* example) can be morphed smoothly
from one point cloud to the next with the ``animate='morph'`` `hyp.plot`
style (PR #272, maintainer request 2026-07-06 -- see the `animate`/
`rotations`/`morph_samples` entries of the `hyp.plot` docstring for the
full spec). Under the hood, an equal-sized sample of points is drawn from
each shape, consecutive shapes are matched point-for-point with the
Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) so that each
point travels the shortest total distance to its partner in the next
shape, and the coordinates are eased between shapes frame by frame while
the camera spins around the scene -- exactly the hand-rolled recipe this
example used to implement itself before `animate='morph'` existed, now
built into the library behind a single `hyp.plot` call. `rotations` also
accepts a per-segment list for finer camera control: below, holds spin a
slow, easy-to-watch full rotation while each transition only spins a
brisk quarter-turn, so the camera visibly "steps" forward every time one
shape morphs into the next.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import numpy as np

import hypertools as hyp

# every point cloud in the shapes zoo, visited in this order
shapes = ['bunny', 'cube', 'dragon', 'sphere', 'teapot', 'vase', 'biplane']


def normalize_shape(points):
    """Center and scale a point cloud into the hypertools [-1, 1] cube."""
    points = np.asarray(points, dtype=np.float64)
    points = points - points.mean(axis=0)
    return points / np.abs(points).max()


clouds = [normalize_shape(hyp.load(shape)) for shape in shapes]

# frame schedule: hold, morph, hold, morph, ..., hold -- 2 * n_shapes - 1
# segments in all. `rotations` gives each segment its OWN camera-spin
# count: holds get a slow, easy-to-watch full rotation (1) while
# transitions get a brisk quarter-turn (0.25), so the camera visibly steps
# forward every time a shape morphs into the next one. 13 segments * 30
# frames/segment (1 sec/segment @ 30 fps) = 390 frames total, gallery-tractable.
rotations = [1, 0.25] * (len(shapes) - 1) + [1]

fig, ani = hyp.plot(clouds, fmt='.', color='k', markersize=1.5,
                    animate='morph', rotations=rotations,
                    duration=len(rotations), frame_rate=30)
