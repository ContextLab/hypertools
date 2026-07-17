# -*- coding: utf-8 -*-
"""
=========================================================
Animated trajectory plotted with multidimensional scaling
=========================================================

This is a trajectory of brain data plotted in 3D with multidimensional scaling.
"""

# sphinx_gallery_thumbnail_path = '_static/thumbnails/sphx_glr_animate_MDS_thumb.gif'

# Code source: Andrew Heusser
# License: MIT

# import hypertools
import hypertools as hyp

# load the data
data = hyp.load('weights_avg')

# plot
fig, ani = hyp.plot(data, animate=True, reduce='MDS')
