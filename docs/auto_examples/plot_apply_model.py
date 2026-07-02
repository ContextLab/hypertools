# -*- coding: utf-8 -*-
"""
=============================
Applying models with apply_model
=============================

`hyp.apply_model` is hypertools 2.0's unified model-application core:
datasets are stacked, the model is fit ONCE across all of them, and the
result is unstacked back to the input's structure -- which is what makes
embeddings and cluster assignments comparable across datasets. Models can
be specified by name, as a dict with parameters, as a scikit-learn style
instance, or as a list (pipeline). `return_model=True` hands back the
fitted model for reuse on held-out data.
"""

# Code source: Contextual Dynamics Lab
# License: MIT

import numpy as np
import hypertools as hyp

rng = np.random.default_rng(42)
data1 = np.cumsum(rng.standard_normal((100, 10)), axis=0)
data2 = np.cumsum(rng.standard_normal((100, 10)), axis=0)

# one PCA fit across both datasets -> shared, comparable embedding
embedded = hyp.apply_model([data1, data2], 'PCA', ndims=3)
hyp.plot(embedded, 'o')

# pipelines: PCA to denoise, then TSNE to embed
embedded = hyp.apply_model([data1, data2],
                           [{'model': 'PCA', 'params': {'n_components': 5}},
                            {'model': 'TSNE', 'params': {'n_components': 2}}])
hyp.plot(embedded, 'o')

# reuse a fitted model on held-out data
train, fitted = hyp.apply_model(data1, 'PCA', ndims=3, return_model=True)
held_out = fitted.transform(data2)
print(held_out.shape)
