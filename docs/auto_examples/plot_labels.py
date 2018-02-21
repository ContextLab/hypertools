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

# import
import hypertools as hyp
import numpy as np

# load example data
data, _ = hyp.load('weights_sample')

# simulate labels
labels=[]
for idx,i in enumerate(data):
    tmp=[]
    for iidx,ii in enumerate(i):
        if iidx==0:
            tmp.append('Subject ' + str(idx))
        else:
            tmp.append(None)
    labels.append(tmp)

# plot
hyp.plot(data, '.', labels=labels)
