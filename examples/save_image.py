# -*- coding: utf-8 -*-
"""
=============================
Saving a plot
=============================

To save a plot, simply use the `save_path` kwarg, and specify where you want
the image to be saved, including the file extension (e.g. pdf)
"""

# Code source: Andrew Heusser
# License: MIT

# import
import hypertools as hyp
import numpy as np

# load example data
geo = hyp.load('weights_sample')

# plot
geo.plot(fmt='o', save_path='test-image.pdf')
