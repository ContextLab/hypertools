# -*- coding: utf-8 -*-
"""
=============================
Aligning matrices to a common space
=============================

In this example, we plot the trajectory of multivariate brain activity for
two groups of subjects that have been hyperaligned (Haxby et al, 2011).  First,
we use the align tool to project all subjects in the list to a common space.
Then we average the data into two groups, and plot.
"""

# Code source: Andrew Heusser
# License: MIT

# import
import hypertools as hyp
import numpy as np

# load example data
<<<<<<< HEAD
data = hyp.load('weights', align='hyper')
=======
data = hyp.tools.load('weights', align=True)
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764

# average into two groups
group1 = np.mean(data[:17], 0)
group2 = np.mean(data[18:], 0)

# plot
hyp.plot([group1[:100, :], group2[:100, :]])
