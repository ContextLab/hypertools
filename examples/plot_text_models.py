# -*- coding: utf-8 -*-
"""
=============================
Plotting text using different models
=============================

Hypertools includes 3 prefit topic models with different corpuses: wiki, nips
and sotus. The projection of new text into these spaces will be different
because the topics discovered in each model is unique.
"""

# Code source: Andrew Heusser
# License: MIT

# load hypertools
import hypertools as hyp

# load the data
data = [['i like cats alot', 'cats r pretty cool', 'cats are better than dogs'],
        ['dogs rule the haus', 'dogs are my jam', 'dogs are a mans best friend'],
        'i haz a cheeseburger?']

# plot it with the default text model (wiki)
hyp.plot(data, 'o', title='wiki model')

# plot it with a model fit to nips papers
hyp.plot(data, 'o', semantic='nips', title='nips model')

# plot it with a model fit to State of the Union Addresses
hyp.plot(data, 'o', semantic='sotus', title='sotus model')
