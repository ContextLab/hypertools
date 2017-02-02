# -*- coding: utf-8 -*-
"""
=============================
Labeling your datapoints
=============================

This is an example of how to use the `label(s)` kwarg, which must be a list the
length of the number of datapoints (rows) you have in the matrix.  Here, we
are simply labeling the first datapoint for each matrix in the list.
"""

# Code source: Andrew Heusser
# License: MIT

import hypertools as hyp
import scipy.io as sio
import numpy as np

data=sio.loadmat('sample_data/weights.mat')
w=[i for i in data['weights'][0][0:3]]

labels=[]
for idx,i in enumerate(w):
    tmp=[]
    for iidx,ii in enumerate(i):
        if iidx==0:
            tmp.append('Subject ' + str(idx))
        else:
            tmp.append(None)
    labels.append(tmp)

hyp.plot(w,'o',labels=labels)
