# -*- coding: utf-8 -*-
"""
=============================
Grouping data by category
=============================

When plotting, its useful to have a way to color points by some category or
variable.  Hypertools does this using the `hue` kwarg, which takes a list
of string category labels or numerical values.  If text labels are passed, the
data is restructured according to those labels and plotted in different colors
according to your color palette.  If numerical values are passed, the values
are binned (default resolution: 100) and plotted according to your color
palette.
"""

# Code source: Andrew Heusser
# License: MIT

# import
import hypertools as hyp
import numpy as np

# load example data
data, labels = hyp.load('weights_sample')

# simulate groups
hue = labels

# plot
hyp.plot(data, '.', hue=hue)

# simulate random groups
hue=[]
for idx,i in enumerate(data):
    tmp=[]
    for iidx,ii in enumerate(i):
            tmp.append(int(np.random.randint(1000, size=1)))
    hue.append(tmp)

# plot
hyp.plot(data, '.', hue=hue)
