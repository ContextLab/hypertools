# -*- coding: utf-8 -*-
"""
=============================
Plotting text
=============================

To plot text, simply pass the text data to the plot function.  By default, the
text samples will be transformed into a vector of word counts and then modeled
using Latent Dirichlet Allocation (# of topics = 100) using a model fit to a
large sample of wikipedia pages.  If you specify semantic=None, the word
count vectors will be plotted. To convert the text t0 a matrix (or list of
matrices), we also expose the format_data function.
"""

# Code source: Andrew Heusser
# License: MIT

# load hypertools
import hypertools as hyp

# load the data
data = [['i like cats alot', 'cats r pretty cool', 'cats are better than dogs'],
        ['dogs rule the haus', 'dogs are my jam', 'dogs are a mans best friend'],
        'i haz a cheezeburger?']

# plot it
hyp.plot(data, 'o')

# convert text to matrix without plotting
mtx = hyp.format_data(data, vectorizer='TfidfVectorizer', semantic='NMF')
