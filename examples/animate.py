# -*- coding: utf-8 -*-
"""
=============================
Animated plots
=============================

Timeseries plots can be animated by simply passing `animate=True` when
calling hyp.plot.
"""

# sphinx_gallery_thumbnail_path = '_static/thumbnails/sphx_glr_animate_thumb.gif'

# Code source: Andrew Heusser
# License: MIT

# import
import hypertools as hyp

# load example data
data = hyp.load('weights_avg')

# plot
fig, ani = hyp.plot(data, animate=True, legend=['first', 'second'])
