# -*- coding: utf-8 -*-
"""
=====================================
Morphing hull surfaces through shapes
=====================================

Building on the *Morphing through the shapes zoo, with titles* example, this
one wraps the moving point cloud in a smooth, lit convex-hull SURFACE: the
``surface=`` kwarg of `hyp.plot` combined with ``animate='morph'``. The hull
mesh is recomputed from the travelling cloud on every frame, shaded with a
two-light Blinn-Phong model and backface-culled for the current camera
angle, so a blue-teal "skin" flows continuously from the bunny to the cube,
the sphere, the teapot and the vase as the points underneath rearrange
themselves -- all from one `hyp.plot` call. The points stay visible as a
faint black texture under the surface.

A convex hull cannot reproduce concave features, so the holds on concave
shapes (the bunny, the teapot's spout and handle) render as smooth, rounded
blobs; that loss of concavity is the expected trade-off of a hull surface,
not a bug. The hull hugs the data by construction: each smoothing round
pulls stray vertices back onto the original hull (see
:func:`hypertools.plot.meshutil.smooth_hull_3d`), and a final, bounded,
grow-only rescale guarantees that at least 99% of the points sit inside the
surface. For ordinary clouds that rescale only nudges the mesh by a few
percent; it grows large only for very sparse clouds (fewer than ~10 points),
where a coarse, few-vertex hull loses proportionally more to smoothing.

Camera speed (degrees per frame) is constant across the whole animation, so
the per-segment ``rotations`` list sets how much SCREEN TIME each hold and
transition gets, never how fast it spins: every hold gets a slow full turn
and every transition a brisk quarter-turn, so the camera visibly steps
forward each time one shape morphs into the next.

Each cloud is first centred and scaled into the [-1, 1] cube with
``hyp.manip(..., model='Normalize', mode='isotropic')`` -- one centroid and
one scalar per shape, so proportions are preserved -- because `hyp.plot`
draws every dataset in one shared frame. Five of the zoo's seven shapes are
used, and ``morph_samples=400`` caps each at 400 points before the morph's
point matching: a hull's shape is set by its extreme points, so fewer
interior points barely change the rendered surface while every per-frame
hull rebuild stays cheap. The mesh smoothing is capped at 2 rounds rather
than the default 3, roughly a 4x per-frame speedup (measured: 30 frames in
4.5 s against 19.3 s).
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

# use a pre-rendered gif of the morphing lit surface as this example's gallery
# thumbnail, as the other animated examples do. Without it, sphinx-gallery
# thumbnails the trailing static-figure snapshot as a frozen png.
# sphinx_gallery_thumbnail_path = '_static/thumbnails/sphx_glr_animate_surface_morph_thumb.gif'

import hypertools as hyp

# five of the zoo's seven shapes, each centred and scaled into the [-1, 1]
# cube with one shared centroid and scalar (mode='isotropic'), so every shape
# keeps its proportions inside hyp.plot's shared frame
shapes = ['bunny', 'cube', 'sphere', 'teapot', 'vase']
clouds = [hyp.manip(hyp.load(shape), model='Normalize', mode='isotropic',
                    min=-1, max=1) for shape in shapes]

# a hull's shape is set by its extreme points, not its interior density, so
# capping every shape at 400 points (morph_samples= below: hyp.plot samples
# without replacement from a seeded generator) barely changes the rendered
# surface while keeping every per-frame rebuilt hull cheap
n_points = 400

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

# fade the point layer to a subtle texture underneath the hull surface.
# NOTE: under animate='morph' the per-dataset lines are hidden and the
# VISIBLE traveling point cloud is a separate artist; alpha= on the hyp.plot
# call reaches only the hidden per-dataset lines (the visible cloud's alpha
# stays unset), so select the visible line and fade it directly
visible_lines = [ln for ln in fig.axes[0].get_lines() if ln.get_visible()]
visible_lines[-1].set_alpha(0.25)
