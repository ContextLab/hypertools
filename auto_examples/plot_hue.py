# -*- coding: utf-8 -*-
"""
=========================
Grouping data by category
=========================

The ``hue`` kwarg colors points by a category or a variable. It accepts
one label per row (or one list per dataset), either as strings or as
numbers. String labels are treated as categories: the rows are regrouped
by label and each group gets its own color from the palette, with a legend
naming the groups. Numeric values are binned instead (100 bins by default)
and colored along the palette as a gradient. Both figures below use the
``weights_sample`` brain-activity data: three subjects, each 300 timepoints
of 100 features.
"""

# Code source: Andrew Heusser
# License: MIT

import numpy as np

import hypertools as hyp

# load example data: three subjects, each (300 timepoints, 100 features)
data = hyp.load('weights_sample')

# %%
# Categorical hue: label each timepoint by which third of the recording it
# falls in. The string labels become three colored groups plus a legend.
labels = ['beginning'] * 100 + ['middle'] * 100 + ['end'] * 100
hue = [labels for _ in data]
hyp.plot(data, fmt='.', hue=hue, legend=True)

# %%
# Numeric hue: the timepoint index of every row. Numbers are binned and
# mapped onto the palette as a gradient, so color now encodes time.
time = [np.arange(len(subject)) for subject in data]
hyp.plot(data, fmt='.', hue=time)
