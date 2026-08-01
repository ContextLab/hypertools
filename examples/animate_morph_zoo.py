# -*- coding: utf-8 -*-
"""
============================================
Morphing through the shapes zoo, with titles
============================================

One `hyp.plot(..., animate='morph')` call smoothly *morphs* a cloud of black
dots from one shape to the next, holding on each shape before flowing into the
following one. HyperTools ships a "shapes zoo" of classic 3-D point clouds
(``bunny``, ``cube``, ``sphere``, ``vase``, ...), each downloaded once and then
cached in ``~/hypertools_data`` -- so this example is fully offline and
deterministic after the first run.

The **title that tracks the current shape** comes straight from the library:
passing a list of per-shape names as ``title=`` to ``hyp.plot`` is enough. A
morph animation alternates "hold" segments (the camera slowly orbits a
finished shape) with "transition" segments (one shape flowing into the next);
``hyp.plot`` names the shape while holding and shows nothing mid-transition,
so the label never sits over a half-formed cloud.

To keep the gallery build quick, each shape is capped at 2000 points (the cap
the morph's point matching then runs on) and the zoo's five shapes are
morphed; the technique is identical for the full clouds.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import numpy as np

import hypertools as hyp


def normalize(points):
    """Center a point cloud and scale it into the hypertools [-1, 1] cube."""
    points = np.asarray(points, dtype=float)
    points = points - points.mean(axis=0)
    return points / np.abs(points).max()


# NOTE on the teapot: ``hyp.load('teapot')`` returns 1728 rows but only 301
# UNIQUE coordinates (ratio 0.174, measured 2026-07-26), where every other shape
# is essentially all-unique (bunny 35947/35947, vase 36022/36022, cube
# 30034/30246, sphere 29891/30135). Its segment therefore draws with a few
# hundred distinct dots rather than a couple of thousand and reads sparser than
# its neighbours. That is the shipped dataset, not a fault in this example.
SHAPES = ['bunny', 'cube', 'sphere', 'teapot', 'vase']
TITLES = ['Bunny', 'Cube', 'Sphere', 'Teapot', 'Vase']

# normalize() above is NOT redundant with hyp.plot: plot rescales every dataset
# with ONE shared affine, so clouds left in their own raw units would be drawn
# at wildly different sizes. The sampling below IS done by hand rather than
# left to morph_samples: the loop-closing repeat of the first cloud (see below)
# has to be the SAME sample, and morph_samples draws a fresh subset per
# dataset. The cap itself is what keeps the morph tractable: the one-to-one
# point matching is a Hungarian assignment costing roughly O(n^3), and the
# zoo's clouds have ~30k points each.
N = 2000
# the normalized cube reaches +/-1 on every axis, i.e. exactly the drawn axes
# box, so its frames read as noise in a wireframe rather than as a cube; shrink
# it to sit visibly inside. The other shapes still set the shared box.
CUBE_SCALE = 0.8
rng = np.random.default_rng(0)


def load(name):
    points = normalize(hyp.load(name))
    if name == 'cube':
        points = points * CUBE_SCALE
    idx = rng.choice(len(points), size=min(N, len(points)), replace=False)
    return points[idx]


clouds = [load(name) for name in SHAPES]
# close the loop: morph back to the FIRST shape so a looping player does not
# hard-cut from the last shape to the first. Reusing the same sampled array
# means the closing hold and the opening hold draw an identical point set.
clouds.append(clouds[0])
titles = TITLES + [TITLES[0]]

# [hold_1, morph_1->2, hold_2, ..., hold_N] for the 5 clouds = 9 segments.
# Camera speed is constant, so these ratios set each segment's SCREEN TIME: a
# full turn per shape, half a turn per transition. The first shape's hold is
# split in half across the two ends (they play back-to-back on repeat, giving
# one full hold), and the total, 8.0, is a whole number of turns, so the
# azimuth also wraps exactly at the loop point.
rotations = [0.75] + [0.5, 1.0] * (len(SHAPES) - 1) + [0.5, 0.75]
duration, fps = 12, 20

# THE hypertools call: black pixel-sized dots morphing through the zoo.
# title= names each shape while its hold plays and is left blank by
# hyp.plot itself during every transition -- no hand-rolled schedule needed.
fig, ani = hyp.plot(clouds, fmt='.', color='k', markersize=1.6,
                    animate='morph', rotations=rotations, morph_samples=N,
                    duration=duration, frame_rate=fps, size=(6, 6), show=False,
                    title=titles)
