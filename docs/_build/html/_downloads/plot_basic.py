# -*- coding: utf-8 -*-
"""
=============================
A basic example
=============================

Here is a basic example where we load in some data (a list of arrays - samples
by features), take the first two arrays in the list and plot them as points
with the 'o'.  Hypertools can handle all format strings supported by matplotlib.
"""

# Code source: Andrew Heusser
# License: MIT

import hypertools as hyp

data = hyp.tools.load('weights')
w=[i for i in data[0:2]]

hyp.plot(w,'o')
