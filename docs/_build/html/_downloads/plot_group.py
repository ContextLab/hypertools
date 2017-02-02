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

import hypertools as hyp
import scipy.io as sio
import numpy as np

data=sio.loadmat('sample_data/weights.mat')
w=[i for i in data['weights'][0][0:3]]

group = [['a' if idx % 2 == 0 else 'b' for idx,j in enumerate(i)] for i in w]
hyp.plot(w,'o',group=group)

group=[]
for idx,i in enumerate(w):
    tmp=[]
    for iidx,ii in enumerate(i):
            tmp.append(int(np.random.randint(1000, size=1)))
    group.append(tmp)

hyp.plot(w,'o',group=group)
