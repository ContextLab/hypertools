# -*- coding: utf-8 -*-
"""
=============================
Labeling your datapoints
=============================

This is an example of how to use the `labels=` kwarg. Passed one entry per
DATASET (rather than one per row), each dataset is annotated once, at the
row named by `label_anchor=` -- here `label_anchor='first'` (the default)
labels the first datapoint of each matrix in the list.
"""

# Code source: Andrew Heusser
# License: MIT

# import
import hypertools as hyp

# load example data
data = hyp.load('weights_sample')

# one label per dataset -- 'Subject 0', 'Subject 1', ... -- anchored to the
# first row of each dataset
labels = [f'Subject {idx}' for idx in range(len(data))]

# plot
hyp.plot(data, fmt='.', labels=labels, label_anchor='first')
