# -*- coding: utf-8 -*-
"""
=============================
Defining a custom corpus for plotting text
=============================

By default, the text samples will be transformed into a vector of word counts
and then modeled using Latent Dirichlet Allocation (# of topics = 100) using a
model fit to a large sample of wikipedia pages.  However, you can optionally
pass your own text to fit the semantic model. To do this define corpus as a
list of documents (strings). A topic model will be fit on the fly and the text
will be plotted.
"""

# Code source: Andrew Heusser
# License: MIT

# load hypertools
import hypertools as hyp

# load the data
text_samples = ['i like cats alot', 'cats r pretty cool', 'cats are better than dogs',
        'dogs rule the haus', 'dogs are my jam', 'dogs are a mans best friend',
        'i haz a cheezeburger?']

# plot it
hyp.plot(text_samples, 'o', corpus=text_samples)
