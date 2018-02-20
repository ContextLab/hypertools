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
data, labels = hyp.load('weights_sample')

# plot
hyp.plot(data, '.', legend=['Group A', 'Group B', 'Group C'])
