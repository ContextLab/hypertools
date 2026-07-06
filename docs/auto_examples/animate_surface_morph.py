# -*- coding: utf-8 -*-
"""
=====================================
Morphing hull surfaces through shapes
=====================================

Building on the *Morphing through the shapes zoo* example, HyperTools can
also render a smooth, lit convex-hull SURFACE around a moving point cloud
(the ``surface=`` `hyp.plot` kwarg -- see :mod:`hypertools.plot.meshutil`
and :mod:`hypertools.plot.surface`, GH #109) instead of just the raw
points. Combining `surface=` with `animate='morph'` (PR #272, maintainer
request 2026-07-06) recomputes the moving cloud's smoothed hull mesh from
scratch on every frame, shaded with a two-light Blinn-Phong model and
backface-culled for the current camera angle -- so the "blob" skin flows
continuously as the underlying points rearrange themselves, all from one
`hyp.plot` call. Camera rotation speed (degrees/frame) is always constant
across the whole animation, so a per-segment `rotations` list (below)
controls how much SCREEN TIME each hold/transition gets, never how fast
it spins. Since a convex hull cannot reproduce concave features,
holds on concave shapes like the bunny necessarily render as a smooth,
rounded blob; that loss of concavity is an expected trade-off of the
hull-surface approach, not a bug. Hulls hug the data tightly BY
CONSTRUCTION (each smoothing round pulls stray vertices back onto the
original hull surface, see :func:`hypertools.plot.meshutil.smooth_hull_3d`)
rather than via any fixed overshoot allowance, so the axes box never needs
a hand-computed fudge factor to contain the surface. A final, bounded,
grow-only rescale then guarantees at least 99% containment of the actual
points; for ordinary, reasonably-sampled clouds this rescale rarely does
more than nudge the mesh a few percent, and only grows large for very
sparse clouds (rule of thumb: fewer than ~10 points), where a coarse,
few-vertex hull loses proportionally more to smoothing and needs more
correction to recover that same 99% guarantee.

To keep the gallery build modest, only 5 of the 7 zoo shapes are used
(dropping the very high point-count dragon and biplane meshes), and each
shape is downsampled to 400 points before it's ever handed to `hyp.plot`
(a hull's shape is set by its extreme points, so fewer interior points
barely change the rendered surface). This matters for more than just the
per-frame ``ConvexHull`` cost: `hyp.plot` also builds one full hull per
morphing dataset up front (from whatever point cloud it's given) to size
the axes box safely for every frame, and a raw shapes-zoo cloud is dense
enough that its hull can itself have tens of thousands of faces --
downsampling first keeps both that one-time sizing pass and every
per-frame rebuild fast. The mesh smoothing is also capped at 2 rounds
rather than the library default of 3 (roughly 20ms/frame instead of
~100ms+, measured on this machine).
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import numpy as np

import hypertools as hyp

# a subset of the shapes zoo -- smaller point-cloud shapes are used so the
# per-frame convex-hull/smoothing pass stays fast (see module docstring)
shapes = ['bunny', 'cube', 'sphere', 'teapot', 'vase']

# a hull's shape is set by its extreme points, not its interior density, so
# downsampling every shape to 400 points barely changes the rendered
# surface while keeping both the one-time axes-sizing hull and every
# per-frame rebuilt hull cheap (see module docstring)
n_points = 400
rng = np.random.default_rng(42)


def normalize_shape(points):
    """Center and scale a point cloud into the hypertools [-1, 1] cube."""
    points = np.asarray(points, dtype=np.float64)
    points = points - points.mean(axis=0)
    return points / np.abs(points).max()


def load_shape(name):
    """Load and normalize one shapes-zoo cloud, downsampled to `n_points`
    (without replacement) so its hull stays cheap to compute."""
    points = normalize_shape(hyp.load(name))
    idx = rng.choice(len(points), size=min(n_points, len(points)),
                     replace=False)
    return points[idx]


clouds = [load_shape(shape) for shape in shapes]

# frame schedule: hold, morph, hold, morph, ..., hold -- 2 * n_shapes - 1 =
# 9 segments in all. Holds get a slow, easy-to-watch full rotation (1)
# while transitions get a brisk quarter-turn (0.25), so the camera visibly
# steps forward every time one shape morphs into the next. Camera speed
# stays CONSTANT (degrees/frame) across the whole animation, so each
# segment's SCREEN TIME is proportional to its own rotation count instead
# of split evenly: the 5 full-rotation holds get 60 frames each and the 4
# quarter-turn transitions get 15 frames each (360 frames total @ 30 fps,
# 12 sec, same total length as an equal-time split would have given).
rotations = [1, 0.25] * (len(shapes) - 1) + [1]

# surface look: a solid, lit blue-teal hull with 2 rounds of smoothing
# (rather than the library default of 3) to keep per-frame cost low; the
# points are kept only as a faint texture underneath the surface (set
# below), which does the visual work
surface_spec = {
    'alpha': 0.97,
    'color': '#2E86AB',
    'smoothing': 2,
    'keep_points': True,
}

fig, ani = hyp.plot(clouds, fmt='.', color='k', markersize=0.6,
                    animate='morph', rotations=rotations,
                    duration=12, frame_rate=30,
                    morph_samples=n_points, surface=surface_spec)

# fade the point layer to a subtle texture underneath the hull surface
fig.axes[0].get_lines()[0].set_alpha(0.25)
