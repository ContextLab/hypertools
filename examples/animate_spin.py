# -*- coding: utf-8 -*-
"""
=============================
Create a rotating static plot
=============================

In addition to plotting dynamic timeseries data, the spin feature can be used to
visualize static data in an animated rotating plot.
"""

# sphinx_gallery_thumbnail_path = '_static/thumbnails/sphx_glr_animate_spin_thumb.gif'

# Code source: Andrew Heusser
# License: MIT

# import hypertools
import hypertools as hyp

# load the geo
geo = hyp.load('weights_sample')

# plot
geo.plot(fmt='.', animate='spin')
