# -*- coding: utf-8 -*-
"""
=============================
Aligning two matrices with the procrustes function
=============================

In this example, we load in some synthetic data, rotate it, and then use the
procrustes function to get the datasets back in alignment.  The procrustes
function uses linear transformations to project a source matrix into the
space of a target matrix.
"""

# Code source: Andrew Heusser
# License: MIT

# import
import hypertools as hyp
from hypertools.align.procrustes import procrustes

# load example data
data = hyp.load('spiral')
hyp.plot(data, title='Before Alignment')

# use procrustes to align the data
source, target = data
aligned = [procrustes(source, target), target]

# after alignment
hyp.plot(aligned, ['-','--'], title='After alignment')
