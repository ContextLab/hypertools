# -*- coding: utf-8 -*-
"""
=============================
Grouping data by category
=============================

When plotting, its useful to have a way to color points by some category or
variable.  Hypertools does this using the `group` kwarg, which takes a list
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
<<<<<<< HEAD
data = hyp.load('weights_sample')
=======
data = hyp.tools.load('weights_sample')
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764

# simulate groups
group = [['a' if idx % 2 == 0 else 'b' for idx, j in enumerate(i)] for i in data]

# plot
hyp.plot(data, '.', group=group)

# simulate random groups
group=[]
for idx,i in enumerate(data):
    tmp=[]
    for iidx,ii in enumerate(i):
            tmp.append(int(np.random.randint(1000, size=1)))
    group.append(tmp)

# plot
hyp.plot(data, '.', group=group)
