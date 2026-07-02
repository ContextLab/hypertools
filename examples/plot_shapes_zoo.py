# -*- coding: utf-8 -*-
"""
=============================
A zoo of 3D shapes
=============================

HyperTools ships with several classic 3D "shapes zoo" point clouds
(they download once and are then cached in `~/hypertools_data`).  This
example loads four of them and displays each in its own panel of a 2x2
figure by passing pre-created 3D axes to `hyp.plot` via the `ax`
keyword.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import matplotlib.pyplot as plt

import hypertools as hyp

shapes = ['teapot', 'cube', 'dragon', 'bunny']

fig, axes = plt.subplots(2, 2, figsize=(8, 8),
                         subplot_kw={'projection': '3d'})

for shape, ax in zip(shapes, axes.ravel()):
    data = hyp.load(shape)
    hyp.plot(data, 'o', ax=ax, title=shape.capitalize())

plt.tight_layout()
plt.show()
