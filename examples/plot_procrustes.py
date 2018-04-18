# -*- coding: utf-8 -*-
"""
=============================
Aligning two matrices with the procrustes function
=============================

In this example, we load in some synthetic data, rotate it, and then use the
procustes function to get the datasets back in alignment.  The procrustes
function uses linear transformations to project a source matrix into the
space of a target matrix.
"""

# Code source: Andrew Heusser
# License: MIT

# import
import hypertools as hyp
import numpy as np
import scipy

# load example data
geo = hyp.load('spiral')
geo.plot(title='Before Alignment')

# use procrusted to align the data
source, target = geo.get_data()
aligned = [hyp.tools.procrustes(source, target), target]

# after alignment
hyp.plot(aligned, ['-','--'], title='After alignment')
