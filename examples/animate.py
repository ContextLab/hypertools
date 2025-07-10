# -*- coding: utf-8 -*-
"""
=============================
Animated plots
=============================

Timeseries plots can be animated by simply passing `animate=True` to the geo (
or when calling hyp.plot).
"""

# sphinx_gallery_thumbnail_path = '_static/thumbnails/sphx_glr_animate_thumb.gif'

# Code source: Andrew Heusser
# License: MIT

# import
import hypertools as hyp

# load example data
geo = hyp.load('weights_avg')

# plot
geo.plot(animate=True, legend=['first', 'second'])
