# -*- coding: utf-8 -*-
"""
=============================
Animated trajectory plotted with multidimensional scaling
=============================

This is a trajectory of brain data plotted in 3D with multidimensional scaling.
"""

# Code source: Andrew Heusser
# License: MIT

# import hypertools
import hypertools as hyp

# load the geo
geo = hyp.load('weights_avg')

# plot
geo.plot(animate=True, reduce='MDS')
