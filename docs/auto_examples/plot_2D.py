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
<<<<<<< HEAD
data = hyp.load('weights_sample')
=======
data = hyp.tools.load('weights_sample')
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764

# plot
hyp.plot(data, '.', ndims=2)
