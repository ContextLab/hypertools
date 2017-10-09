# -*- coding: utf-8 -*-
"""
=============================
Saving a geo
=============================

To save a plot, simply use the `save_path` kwarg, and specify where you want
the image to be saved, including the file extension (e.g. pdf)
"""

# Code source: Andrew Heusser
# License: MIT

# import
import hypertools as hyp

# load some data
data = hyp.load('mushrooms')

# create a geo
geo = hyp.plot(data, show=False)


geo.save('test')

geo = hyp.load('test.geo')

hyp.plot(geo.transform(data), '.')
