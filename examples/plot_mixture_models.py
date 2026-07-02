# -*- coding: utf-8 -*-
"""
=============================
Soft clustering with mixture models
=============================

In addition to hard clustering (KMeans, HDBSCAN, ...), hypertools 2.0
supports mixture models: GaussianMixture, BayesianGaussianMixture,
LatentDirichletAllocation, and NMF. `hyp.cluster` returns an
(n_samples, n_components) matrix of membership proportions instead of
discrete labels, and `hyp.plot` colors each observation by blending the
component colors according to its mixture weights -- observations between
clusters render with intermediate colors.
"""

# Code source: Contextual Dynamics Lab
# License: MIT

import numpy as np
import hypertools as hyp

# three OVERLAPPING gaussian blobs (1.5 sd apart) -- points in the overlap
# regions have genuinely mixed memberships, so their colors visibly blend
rng = np.random.default_rng(42)
data = np.vstack([rng.standard_normal((100, 5)) + 1.5 * i for i in range(3)])

# soft cluster assignments: rows sum to 1
proportions = hyp.cluster(data, cluster='GaussianMixture', n_clusters=3)
print(proportions.shape, proportions[:2].round(3))

# colors blend according to membership weights
hyp.plot(data, 'o', cluster='GaussianMixture', n_clusters=3)

# equivalently, pass any (n_samples, k) matrix as hue
hyp.plot(data, 'o', hue=proportions)
