# -*- coding: utf-8 -*-
"""
=============================
A 2D Plot
=============================

A 2D plot can be created by setting ndims=2.
"""

# Code source: Andrew Heusser
# License: MIT

# import
import hypertools as hyp

# load data
geo = hyp.load('weights_sample')

# plot
geo.plot(fmt='.', ndims=2)
