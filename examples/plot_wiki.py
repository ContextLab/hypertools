# -*- coding: utf-8 -*-
"""
=============================
Plotting text
=============================

To plot text, simply pass the text data to the plot function.  By default, the
text samples will be transformed into a vector of word counts and then modeled
using Latent Dirichlet Allocation (# of topics = 20).  If you specify
text_model=None, the word count vectors will be plotted. To convert the text to
a matrix (or list of matrices), we also expose the text2mat function.
"""

# Code source: Andrew Heusser
# License: MIT

# load hypertools
import hypertools as hyp

# text data
data = [['i like cats alot', 'cats r pretty cool', 'cats are better than dogs'],
        ['dogs rule the haus', 'dogs are my jam', 'dogs are a mans best friend'],
        'hampsters are a mans best friend']

# plot it
hyp.plot(data, 'o', text='wiki')

# # convert text to matrix without plotting
# mtx = hyp.tools.text2mat(data, vectorizer='tfidf', text_model='NMF')
