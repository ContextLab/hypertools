# -*- coding: utf-8 -*-
"""
=============================
Generating a legend
=============================

An example of how to use the `legend` kwarg to generate a legend.
"""

# Code source: Andrew Heusser
# License: MIT

import hypertools as hyp
import scipy.io as sio
import numpy as np

data = hyp.tools.load('weights')
w=[i for i in data[0:2]]

hyp.plot(w,'o', legend=['Group A', 'Group B'])
