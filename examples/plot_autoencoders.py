"""
=============================
Autoencoder reducers
=============================

`hyp.reduce` supports six torch-backed autoencoder reducers:
`Autoencoder` (shallow), `SparseAutoencoder`, `DeepAutoencoder`,
`ConvolutionalAutoencoder`, `SequenceAutoencoder`, and
`VariationalAutoencoder`. They are used exactly like any other `reduce=`
model -- by name, with parameters passed via the dict spec -- and use
the optional ``torch`` extra, which hypertools installs on demand the first
time one is fit (pre-install it with ``pip install "hypertools[torch]"``).
This example fits a shallow `Autoencoder` and a `VariationalAutoencoder` on
the same data and compares them against PCA: three 2-D embeddings of a
noisy spiral manifold embedded in 10-D, with each point colored by its
position along the spiral, so a reducer that unfolds the manifold shows a
smooth color gradient. Passing `reduce=` a LIST gives one panel per reducer
(`panels=True`), so all three embeddings are computed and drawn by a single
`hyp.plot` call instead of three separate `hyp.reduce` calls plus a
hand-built grid.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import numpy as np
import hypertools as hyp

rng = np.random.default_rng(0)
# a nonlinear 2D manifold (a Swiss-roll-like spiral) embedded in 10D, plus
# noise -- autoencoders can unfold nonlinear structure that PCA (linear)
# cannot
t = np.linspace(0, 3 * np.pi, 300)
manifold = np.column_stack([t * np.cos(t), t * np.sin(t)])
projection = rng.standard_normal((2, 10))
data = manifold @ projection + 0.05 * rng.standard_normal((300, 10))

# a small, fast training budget -- plenty for a gallery example on 10D data
ae_kwargs = {'epochs': 30, 'batch_size': 32, 'random_state': 0}

# the VAE needs a longer budget and a small KL weight: with the default
# kl_weight=1.0 the KL term dominates on a tiny dataset and the latent
# space collapses to a point (a classic VAE failure mode)
vae_kwargs = {'epochs': 100, 'batch_size': 32, 'random_state': 0,
              'kl_weight': 0.001}

# reduce=[...] draws one panel per reducer, each fitting its own full
# pipeline on `data`; hue=t colors each point (in every panel) by its
# position along the spiral. panels=(1, 3) matches the original one-row
# layout (True/'auto' would pick a squarer 2x2 grid instead).
fig = hyp.plot(
    data, '.', hue=t, palette='viridis', ndims=2, panels=(1, 3),
    reduce=['PCA', {'model': 'Autoencoder', 'kwargs': ae_kwargs},
            {'model': 'VariationalAutoencoder', 'kwargs': vae_kwargs}],
    title=['PCA (linear)', 'Autoencoder', 'VariationalAutoencoder'])
