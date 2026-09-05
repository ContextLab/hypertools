# -*- coding: utf-8 -*-
"""
============================================
Morphing through the shapes zoo, with titles
============================================

One `hyp.plot(..., animate='morph')` call smoothly *morphs* a cloud of black
dots from one shape to the next, holding on each shape before flowing into the
following one. HyperTools ships a "shapes zoo" of seven classic 3-D point
clouds (``bunny``, ``cube``, ``dragon``, ``sphere``, ``teapot``, ``vase`` and
``biplane``), each downloaded once and then cached in ``~/hypertools_data``
-- so this example is fully offline and deterministic after the first run.
On a cold cache with no network it says so and morphs five parametric
stand-ins instead, so it always renders.

Each cloud arrives in its own units, and ``hyp.plot`` draws every dataset it
is given in one shared frame, so the clouds are first put on a common footing
with ``hyp.manip(..., model='Normalize', mode='isotropic', min=-1, max=1)``:
one centroid subtracted and one scalar divided out per cloud, so every shape
keeps its proportions and its largest extent just touches the [-1, 1] cube.

The **title that tracks the current shape** comes straight from the library:
passing a list of per-shape names as ``title=`` to ``hyp.plot`` is enough. A
morph animation alternates "hold" segments (the camera slowly orbits a
finished shape) with "transition" segments (one shape flowing into the next);
``hyp.plot`` names the shape while holding and shows nothing mid-transition,
so the label never sits over a half-formed cloud.

The library re-sets that title every frame at matplotlib's default size and
position, so restyling it is a one-line ``anim.on_frame`` hook: the callback
runs after the library's own updater and re-applies the current text at twice
the size, in bold, lowered to just above the point cloud. The hook assigns
the style on every frame (transitions keep their blank text), so any frame
can be drawn in any order and looks the same.

To keep the gallery build quick, each shape is capped at 2000 points (the cap
the morph's point matching then runs on) and five of the zoo's shapes are
morphed over a 30-second, 600-frame loop; the technique is identical for the
full clouds. Adding ``'dragon'`` and ``'biplane'`` to ``SHAPES`` morphs all
seven -- each extra shape adds a hold and a transition to the loop, so give
it a longer ``duration`` to keep every hold on screen as long.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import os
from typing import NamedTuple

import numpy as np

import hypertools as hyp

# NOTE on the teapot: ``hyp.load('teapot')`` returns 1728 rows but only 301
# UNIQUE coordinates (ratio 0.174, measured 2026-07-26), where every other shape
# is essentially all-unique (bunny 35947/35947, vase 36022/36022, cube
# 30034/30246, sphere 29891/30135). Its segment therefore draws with a few
# hundred distinct dots rather than a couple of thousand and reads sparser than
# its neighbours. That is the shipped dataset, not a fault in this example.
SHAPES = ['bunny', 'cube', 'sphere', 'teapot', 'vase']
# The isotropic Normalize in assemble() is NOT redundant with hyp.plot: plot
# rescales every dataset with ONE shared affine, so clouds left in their own
# raw units would be drawn at wildly different sizes. The sampling IS done by hand rather than left to
# morph_samples: the loop-closing repeat of the first cloud has to be the SAME
# sample, and morph_samples draws a fresh subset per dataset. The cap itself
# is what keeps the morph tractable: the one-to-one point matching is a
# Hungarian assignment costing roughly O(n^3), and the zoo's clouds have ~30k
# points each. Passing morph_samples=N as well makes the cap explicit and
# reproducible rather than a silent default.
N = 2000
# the normalized cube reaches +/-1 on every axis, i.e. exactly the drawn axes
# box, so its frames read as noise in a wireframe rather than as a cube; shrink
# it to sit visibly inside. The other shapes still set the shared box.
CUBE_SCALE = 0.8
# Title styling, re-applied every frame by an on_frame hook (see
# construct_artifact): the library's own per-segment updater calls
# ax.set_title(name) each frame, which resets the size to rcParams and the
# position to matplotlib's automatic spot above the axes box. Twice the
# default 'large' (~12 pt) size, bold, and lowered toward the cloud but kept
# clear of the axes box: measured over all 600 frames (title bbox bottom vs
# the highest projected box corner, 2026-09-03), 0.90 collided by 7 px at
# the box's near-top-corner azimuths and 0.93 clears them everywhere. The
# family is named explicitly because hypertools' bundled default (Noto Sans)
# ships only a Regular face, so ``fontweight='bold'`` alone silently falls
# back to regular (checked with font_manager.findfont, 2026-09-03); DejaVu
# Sans Bold ships inside matplotlib itself, so it is always available.
TITLE_FONTSIZE = 24
TITLE_FONTFAMILY = 'DejaVu Sans'
TITLE_Y = 0.93


class Shapes(NamedTuple):
    clouds: list                # sampled, normalized, loop-closed
    titles: list                # one per cloud
    source: str                 # which path produced them


def assemble(clouds, n, source, seed=0):
    """Normalize, shrink the cube, sample n points, and close the loop:
    morphing back to the FIRST shape means a looping player never hard-cuts
    from the last shape to the first, and reusing the same sampled array
    means the closing hold and the opening hold draw an identical point set."""
    rng = np.random.default_rng(seed)
    sampled = []
    for name, points in clouds.items():
        # one shared centre and scale per cloud (mode='isotropic'): the shape
        # keeps its proportions and its largest extent touches the cube
        points = np.asarray(hyp.manip(points, model='Normalize',
                                      mode='isotropic', min=-1, max=1),
                            dtype=float)
        points = points * (CUBE_SCALE if name == 'cube' else 1.0)
        sampled.append(points[rng.choice(len(points), size=min(n, len(points)),
                                         replace=False)])
    titles = [name.capitalize() for name in clouds]
    return Shapes(sampled + [sampled[0]], titles + [titles[0]], source)


# --- the data half: the ONLY code here that reaches the network -------------
def synthetic_shapes(n=N, seed=0):
    """Five parametric clouds standing in for the zoo when it cannot be
    fetched: the morph is the same technique on any point clouds."""
    rng = np.random.default_rng(seed)
    u, v = rng.uniform(0, 2 * np.pi, n), rng.uniform(-1, 1, n)
    ring = np.sqrt(1 - v ** 2)
    return {
        'sphere': np.column_stack([ring * np.cos(u), ring * np.sin(u), v]),
        'cube': rng.uniform(-1, 1, (n, 3)),
        'torus': np.column_stack([(1 + 0.4 * np.cos(np.pi * v)) * np.cos(u),
                                  (1 + 0.4 * np.cos(np.pi * v)) * np.sin(u),
                                  0.4 * np.sin(np.pi * v)]),
        'helix': np.column_stack([np.cos(3 * np.pi * v), np.sin(3 * np.pi * v), v]),
        'cone': np.column_stack([(1 - v) / 2 * np.cos(u), (1 - v) / 2 * np.sin(u), v]),
    }


def load_shapes(shapes=SHAPES, n=N):
    """The ONLY function here that may touch the network. ``hyp.load`` caches
    the zoo under ``~/hypertools_data``; on a cold cache with no network it
    raises, and this degrades to the parametric stand-ins and says so."""
    if os.environ.get('HYPERTOOLS_OFFLINE'):
        raise RuntimeError('HYPERTOOLS_OFFLINE is set: refusing to fetch')
    try:
        return assemble({name: hyp.load(name) for name in shapes}, n,
                        'the hypertools shapes zoo')
    except Exception as error:
        print(f'shapes zoo unavailable ({error!r}); using parametric stand-ins')
        return assemble(synthetic_shapes(n), n, 'parametric stand-ins (offline)')


def fixture_data():
    """The same payload from the parametric clouds. No network, no bytes."""
    return assemble(synthetic_shapes(), N, 'parametric stand-ins (fixture)')


# --- the figure half: no network, deterministic given its input -------------
def construct_artifact(data):
    """`data.clouds` / `data.titles` in, the animation out. Returns the
    HyperAnimation wrapper, never the unpacked pair."""
    # [hold_1, morph_1->2, hold_2, ..., hold_N]: with the loop-closing copy
    # there are 6 clouds, so 2*6 - 1 = 11 segments. Camera speed is constant,
    # so these ratios set each segment's SCREEN TIME: a full turn per shape,
    # half a turn per transition. The first shape's hold is split in half
    # across the two ends (they play back-to-back on repeat, giving one full
    # hold), and the total, 8.0, is a whole number of turns, so the azimuth
    # also wraps exactly at the loop point.
    rotations = [0.75] + [0.5, 1.0] * (len(data.clouds) - 2) + [0.5, 0.75]
    # THE hypertools call: black pixel-sized dots morphing through the zoo.
    # title= names each shape while its hold plays and is left blank by
    # hyp.plot itself during every transition -- no hand-rolled schedule.
    anim = hyp.plot(data.clouds, fmt='.', color='k', markersize=1.6,
                    animate='morph', rotations=rotations, morph_samples=N,
                    duration=30, frame_rate=20, size=(6, 6), show=False,
                    title=data.titles)

    def restyle_title(ctx):
        """Runs AFTER the library's title updater on every frame: re-apply
        whatever text it set (the shape's name in a hold, '' in a
        transition) at the larger, bolder, lowered style. Assigned every
        frame, never accumulated, so frames stay order-independent."""
        ctx.axes.set_title(ctx.axes.get_title(), fontsize=TITLE_FONTSIZE,
                           fontweight='bold', fontfamily=TITLE_FONTFAMILY,
                           y=TITLE_Y)

    anim.on_frame(restyle_title)
    return anim


if __name__ == '__main__':
    shapes = load_shapes()
    print(f'shapes: {len(shapes.clouds) - 1} clouds + the loop-closing copy, '
          f'{N} points each ({shapes.source})')
    anim = construct_artifact(shapes)
    fig = anim.figure
