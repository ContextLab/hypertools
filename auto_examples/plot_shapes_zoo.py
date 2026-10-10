# -*- coding: utf-8 -*-
"""
=============================
A zoo of 3D shapes
=============================

HyperTools ships with a "shapes zoo" of classic 3D point clouds (they
download once and are then cached in `~/hypertools_data`).  This example
loads *every* shape in the zoo and displays each in its own panel, plotted
as small black dots (the ``,`` pixel marker), via one `hyp.plot` call with
``panels=``.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import hypertools as hyp

# every point cloud in the shapes zoo
shapes = ['bunny', 'cube', 'dragon', 'sphere', 'teapot', 'vase', 'biplane']

# small black dots: ',' is matplotlib's single-pixel marker, ideal for
# dense point clouds. panels=(2, 4) lays the seven shapes into a 2x4 grid
# (the eighth, spare cell is hidden automatically); panel_fit='independent'
# fits each panel's own reduction separately, exactly as the equivalent
# per-panel hyp.plot(shape, ax=ax) loop did, since these clouds differ
# enough in scale that one shared fit would be dominated by the largest.
hyp.plot([hyp.load(shape) for shape in shapes], ',', color='k',
         panels=(2, 4), panel_fit='independent',
         title=[shape.capitalize() for shape in shapes], size=(12, 6))
