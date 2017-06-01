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

data = hyp.tools.load('weights_sample')

hyp.plot(data, '.', legend=['Group A', 'Group B', 'Group C'])
