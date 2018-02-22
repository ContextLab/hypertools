# -*- coding: utf-8 -*-
"""
=============================
Generating a legend
=============================

An example of how to use the `legend` kwarg to generate a legend.
"""

# Code source: Andrew Heusser
# License: MIT

# import
import hypertools as hyp

# load example data
geo = hyp.load('weights_sample')

# plot
geo.plot(fmt='.', legend=['Group A', 'Group B', 'Group C'])
