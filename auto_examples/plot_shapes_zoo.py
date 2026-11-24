# -*- coding: utf-8 -*-
"""
=============================
A zoo of 3D shapes
=============================

HyperTools ships with a "shapes zoo" of classic 3D point clouds (they
download once and are then cached in `~/hypertools_data`).  This example
loads *every* shape in the zoo and displays each in its own panel, plotted
as small black dots, via one `hyp.plot` call with ``panels=``.  The dense
clouds (tens of thousands of points) use the ``,`` pixel marker; the teapot,
which is the 301 control points of the Utah teapot (stored as 1,728 rows),
is drawn with the larger ``.`` marker so that it reads next to them.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import hypertools as hyp

# every point cloud in the shapes zoo
shapes = ['bunny', 'cube', 'dragon', 'sphere', 'teapot', 'vase', 'biplane']

clouds = [hyp.load(shape) for shape in shapes]

# small black dots: ',' is matplotlib's single-pixel marker, ideal for
# dense point clouds. A sparse cloud all but vanishes at one pixel per
# point (the teapot has 1,728 rows, against 23,000 to 91,000 for the
# others), so a cloud of fewer than 5,000 points gets the '.' marker
# instead; markersize= sizes it and leaves the pixel marker alone.
# panels=(2, 4) lays the seven shapes into a 2x4 grid
# (the eighth, spare cell is hidden automatically); panel_fit='independent'
# fits each panel's own reduction separately, exactly as the equivalent
# per-panel hyp.plot(shape, ax=ax) loop did, since these clouds differ
# enough in scale that one shared fit would be dominated by the largest.
# The biplane loads as a DataFrame with columns x, y and z, which hypertools
# would write beside that one panel as axis labels; the empty labels keep
# all seven panels alike. No size= is needed: the figure is sized by the
# grid (3.2 in per panel each way, so 12.8 x 6.4 in here).
hyp.plot(clouds, [',' if len(cloud) >= 5000 else '.' for cloud in clouds],
         color='k', markersize=2.5, panels=(2, 4), panel_fit='independent',
         title=[shape.capitalize() for shape in shapes],
         xlabel='', ylabel='', zlabel='')
