# -*- coding: utf-8 -*-
"""
=============================
Hyperalign a list of arrays and create an animated plot
=============================

The sample data is a list of 2D arrays, where each array is fMRI brain activity
from one subject.  The rows are timepoints and the columns are neural
'features'.  First, the matrices are hyperaligned using hyp.tools.align.
"""

# Code source: Andrew Heusser
# License: MIT

import hypertools as hyp

data = hyp.tools.load('spiral')
hyp.plot(data, animate='serial', group=[str(i) for i in range(10) for j in range(100)], duration=2)
