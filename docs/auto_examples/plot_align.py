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

data=sio.loadmat('sample_data/weights.mat')

w=data['weights'][0]
w = [i for i in w]
aligned_w = hyp.tools.align(w)

w1 = np.mean(aligned_w[:17],0)
w2 = np.mean(aligned_w[18:],0)

hyp.plot([w1[:100,:],w2[:100,:]])
