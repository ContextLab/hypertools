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

import hypertools as hyp
import scipy.io as sio
import numpy as np

data = hyp.tools.load('weights')
aligned_data = hyp.tools.align(data)

group1 = np.mean(aligned_data[:17],0)
group2 = np.mean(aligned_data[18:],0)

hyp.plot([group1[:100,:],group2[:100,:]])
