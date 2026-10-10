# -*- coding: utf-8 -*-
"""
=======================================
Aligning two matrices with Procrustes
=======================================

The ``spiral`` dataset holds two copies of the same 3-D spiral, one of
them rotated. Procrustes alignment finds the linear transformation
(rotation, reflection, and scaling) that projects a source matrix onto a
target matrix, so the two spirals land on top of each other. The first
figure shows the two spirals as loaded; the second aligns them with
``model='Procrustes'`` through `hyp.align`, and the third does the same
thing inside a single `hyp.plot` call via its ``align`` kwarg.
"""

# Code source: Andrew Heusser
# License: MIT

import hypertools as hyp

# load example data: two copies of a spiral, one rotated
data = hyp.load('spiral')
hyp.plot(data, title='Before alignment')

# %%
# `hyp.align` with the Procrustes model projects every dataset onto the
# first one (the alignment target; pass ``index=`` to choose another).
aligned = hyp.align(data, model='Procrustes')
hyp.plot(aligned, ['-', '--'], title='After alignment (hyp.align)')

# %%
# The same alignment can run inside `hyp.plot` with the ``align`` kwarg; the
# dictionary form passes the model's keyword arguments (here, aligning onto
# the second spiral instead of the first).
hyp.plot(data, ['-', '--'],
         align={'model': 'Procrustes', 'kwargs': {'index': 1}},
         title='After alignment (align= in hyp.plot)')
