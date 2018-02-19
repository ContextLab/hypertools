# -*- coding: utf-8 -*-
"""
=============================
Plotting State of the Union Addresses from 1989-2017
=============================

To plot text, simply pass the text data to the plot function.  Here, we will
plot each SOTU address by first fitting the text data to a topic model, and then
reducing the dimensionality of the topic vectors to visualize them. By default,
hypertools transforms the text data using a model fit to a set of selected
wikipedia pages.

"""

# Code source: Andrew Heusser
# License: MIT

# load hypertools
import hypertools as hyp

# load the data
data, hue, labels = hyp.load('sotus')

# plot it
geo = hyp.plot(data, 'o', hue=labels, labels=labels, title='Transformed using wiki model')

geo.plot(corpus=data, title='Transformed using sotus data')
