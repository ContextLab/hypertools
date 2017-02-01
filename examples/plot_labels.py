# -*- coding: utf-8 -*-
"""
=============================
Choosing the thumbnail figure
=============================

An example to demonstrate how to choose which figure is displayed as the
thumbnail if the example generates more than one figure. This is done by
specifying the keyword-value pair ``sphinx_gallery_thumbnail_number = 2`` as a
comment somewhere below the docstring in the example file.
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
