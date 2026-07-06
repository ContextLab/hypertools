# -*- coding: utf-8 -*-
"""
=====================================
Morphing hull surfaces through shapes
=====================================

Building on the *Morphing through the shapes zoo* example, HyperTools can
also render a smooth, lit convex-hull SURFACE around a moving point cloud
(the ``surface=`` plot() kwarg -- see :mod:`hypertools.plot.meshutil` and
:mod:`hypertools.plot.surface`, GH #109) instead of just the raw points.
Here the same Hungarian-matched morph machinery drives a point cloud that
morphs between shapes in the zoo, but on every frame the cloud's smoothed
hull mesh is rebuilt from scratch, shaded with a two-light Blinn-Phong
model, and backface-culled for the current camera angle -- so the "blob"
skin flows continuously as the underlying points rearrange themselves.
Since a convex hull cannot reproduce concave features, holds on concave
shapes like the bunny necessarily render as a smooth, rounded blob; that
loss of concavity is an expected trade-off of the hull-surface approach,
not a bug.

To keep the gallery build modest, only 5 of the 7 zoo shapes are used
(dropping the very high point-count dragon and biplane meshes), each
morph samples only 400 points (a hull's shape is set by its extreme
points, so fewer interior points barely change the rendered surface but
noticeably speed up the per-frame ``ConvexHull`` call), and the mesh
smoothing is capped at 2 rounds rather than the library default of 3
(roughly 20ms/frame instead of ~100ms+, measured on this machine).
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import hypertools as hyp
from hypertools.plot.meshutil import blinn_phong_colors, backface_cull
from hypertools.plot.surface import (
    build_mesh_3d,
    mpl_lighting_kwargs,
    surface_cube_scale,
    view_vector,
)

# a subset of the shapes zoo -- smaller point-cloud shapes are used so the
# per-frame convex-hull/smoothing pass stays fast (see module docstring)
shapes = ['bunny', 'cube', 'sphere', 'teapot', 'vase']

# frames per hold segment and per morph segment; kept modest so the
# gallery build finishes quickly -- 9 segments * 40 frames = 360 frames
# at 30 frames/sec
n_steps = 40
frame_rate = 30
rotations = 2
rng = np.random.default_rng(42)

# a smooth hull surface is shaped by its extreme (hull) points, not its
# interior density, so downsampling to 400 points/shape barely changes the
# rendered surface while keeping ConvexHull + Taubin smoothing cheap
n_points_cap = 400

# surface look: a solid, lit blue-teal hull with 2 rounds of smoothing
# (rather than the library default of 3) to keep per-frame cost low
surface_spec = {
    'alpha': 0.97,
    'color': '#2E86AB',
    'lighting': {},
    'smoothing': 2,
    'pre_inflate': 1.15,
    'keep_points': True,
}


def normalize_shape(points):
    """Center and scale a point cloud into the hypertools [-1, 1] cube."""
    points = np.asarray(points, dtype=np.float64)
    points = points - points.mean(axis=0)
    return points / np.abs(points).max()


clouds = [normalize_shape(hyp.load(shape)) for shape in shapes]

# sample the same (capped) number of points from every cloud, without
# replacement, so morph frames can interpolate point-to-point
n_points = min(n_points_cap, min(len(cloud) for cloud in clouds))
sampled = [cloud[rng.choice(len(cloud), size=n_points, replace=False)]
          for cloud in clouds]

# Hungarian matching: reorder each shape so point i morphs to its optimal
# partner (minimum total travel distance) in the previous shape
for i in range(len(sampled) - 1):
    cost = cdist(sampled[i], sampled[i + 1])
    _, col_ind = linear_sum_assignment(cost)
    sampled[i + 1] = sampled[i + 1][col_ind]

# frame schedule: hold, morph, hold, morph, ... (2 * n_shapes - 1 segments)
segments = []
for i in range(len(sampled)):
    segments.append((sampled[i], sampled[i]))
    if i < len(sampled) - 1:
        segments.append((sampled[i], sampled[i + 1]))
total_frames = len(segments) * n_steps

# draw the first shape as tiny, faint dots (the surface does the visual
# work; the points are kept only as a subtle texture), then animate the
# point artist's coordinates and the hull surface together on every frame.
# Supplying our own axes keeps the figure registered with pyplot (hyp.plot's
# show=False otherwise deregisters it, GH #148), which sphinx-gallery needs
# in order to capture the FuncAnimation below as a video.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
hyp.plot(sampled[0], 'k.', show=False, ax=ax)
ax.set_position([0.0, 0.0, 1.0, 1.0])
point_artist = ax.get_lines()[0]
point_artist.set_markersize(0.6)
point_artist.set_alpha(0.25)

# the smoothed hull surface (pre_inflate=1.15 plus smoothing overshoot)
# bulges past the [-1, 1] cube hyp.plot() sized the axes to -- widen the
# view limits so it never clips against the plot bounds. GH #109 round 2:
# rather than a hand-picked fudge factor, compute the ACTUAL bound needed
# from every shape's own hull mesh (the morph interpolates between these,
# so their hulls -- not necessarily the in-between frames -- set the
# widest extent) via the same `surface_cube_scale` helper the library
# itself now uses to size its own axes cube/plotly scene range.
shape_meshes = [build_mesh_3d(cloud, surface_spec, quiet=True) for cloud in sampled]
cube_scale = surface_cube_scale(shape_meshes)
ax.set_xlim3d([-cube_scale, cube_scale])
ax.set_ylim3d([-cube_scale, cube_scale])
ax.set_zlim3d([-cube_scale, cube_scale])

base_rgb = to_rgb(surface_spec['color'])
light_kw = mpl_lighting_kwargs(surface_spec)
surface_coll = [None]


def draw_surface(points, elev, azim):
    """Rebuild and shade the hull surface for the current point cloud and
    camera angle, removing the previous frame's Poly3DCollection first."""
    if surface_coll[0] is not None:
        surface_coll[0].remove()
        surface_coll[0] = None
    mesh = build_mesh_3d(points, surface_spec, quiet=True)
    if mesh is None:
        return
    verts, faces = mesh
    v = view_vector(elev, azim)
    rgba = blinn_phong_colors(verts, faces, base_rgb, v, **light_kw)
    keep = backface_cull(verts, faces, v)
    rgba = rgba.copy()
    rgba[:, 3] = surface_spec['alpha']
    coll = Poly3DCollection(
        verts[faces[keep]], facecolors=rgba[keep], edgecolors="none",
        linewidths=0, shade=False, antialiaseds=False,
    )
    coll.set_label("_nolegend_")
    ax.add_collection3d(coll)
    surface_coll[0] = coll


def update(frame):
    seg_idx, step = divmod(frame, n_steps)
    start, end = segments[seg_idx]
    t = step / max(1, n_steps - 1)
    t = t * t * (3 - 2 * t)  # smoothstep easing (holds: start == end, so t
                             # has no effect and the shape simply sits still)
    points = (1 - t) * start + t * end
    point_artist.set_data(points[:, 0], points[:, 1])
    point_artist.set_3d_properties(points[:, 2])
    elev = 10
    azim = -60 + 360.0 * rotations * frame / total_frames
    ax.view_init(elev=elev, azim=azim)
    draw_surface(points, elev, azim)
    return (point_artist,)


ani = animation.FuncAnimation(fig, update, frames=total_frames,
                              interval=1000 / frame_rate, blit=False)
