:orphan:

.. _api_ref:

.. currentmodule:: hypertools

API reference
=============

This page lists every public entry point exported from ``hypertools``
(see ``hypertools/__init__.py``), organized roughly in the
:doc:`canonical pipeline order <pipeline_order>` (GH #153): load/impute,
manip, normalize, reduce, align, cluster, predict, plot/analyze, plus the
model-application core, I/O helpers, and the text/reducer model families
used by ``manip=``/``reduce=``.

Load
------------------

.. autosummary::
  :toctree:

  load

Save
------------------

.. autosummary::
  :toctree:

  save

Impute
------------------

.. autosummary::
  :toctree:

  impute

Manip
------------------

.. autosummary::
  :toctree:

  manip

Normalize
------------------

.. autosummary::
  :toctree:

  normalize

Reduce
------------------

.. autosummary::
  :toctree:

  reduce

Autoencoder reducers (GH #162) -- the optional ``torch`` extra, installed
on demand (see :doc:`optional_dependencies`); pass by name (e.g.
``reduce='Autoencoder'``) or by class to `hypertools.reduce`:

.. autosummary::
  :toctree:

  reduce.autoencoders.Autoencoder
  reduce.autoencoders.SparseAutoencoder
  reduce.autoencoders.DeepAutoencoder
  reduce.autoencoders.ConvolutionalAutoencoder
  reduce.autoencoders.SequenceAutoencoder
  reduce.autoencoders.VariationalAutoencoder

Align
------------------

.. autosummary::
  :toctree:

  align

.. autosummary::
  :toctree:

  align.procrustes

Cluster
------------------

.. autosummary::
  :toctree:

  cluster

Predict
------------------

.. autosummary::
  :toctree:

  predict

A bare DataFrame carrying a MultiIndex on one axis is split into groups and
forecast one group at a time: a **column** MultiIndex groups by every level
above the innermost one (the innermost level is the feature axis, so every
group keeps all of the frame's observations), while a **row** MultiIndex
also groups by the outer levels but treats the innermost one as the time
axis, which survives as each group's index. The result is a list of
forecasts, one per group -- see :doc:`hierarchy`.

Plot
------------------

.. autosummary::
  :toctree:

  plot
  HyperAnimation
  FrameContext

Hierarchical (MultiIndex) frames are expanded into one trace per group plus
a derived mean at every level above the leaves. The two axes are read
differently -- a **row** MultiIndex draws one trace per unique full index
tuple, whereas a **column** MultiIndex makes its innermost level the feature
axis and groups by everything above it -- and ``predict=`` then forecasts
every final trace, leaves and means alike. See :doc:`hierarchy` for the
comparison table, the feature-correspondence rule and the return shapes.

Colors
------------------

.. autofunction:: hypertools.plot.colors.image_palette

.. autofunction:: hypertools.plot.colors.get_palette_colors

.. autofunction:: hypertools.plot.colors.continuous_colormap

Set interactive backend
------------------------

.. autosummary::
  :toctree:

  set_interactive_backend

Analyze
------------------

.. autosummary::
  :toctree:

  analyze

Apply model
------------------

.. autosummary::
  :toctree:

  apply_model
  supported_models

Pipeline
------------------

`hypertools.Pipeline` chains fitted pipeline stages for reuse (GH #227
#161). Standalone dispatchers (`hypertools.reduce`, `hypertools.manip`,
...) called with ``return_model=True`` return a `hypertools.Pipeline` when
more than one stage ran (and the single fitted wrapper when only one stage
ran); `hypertools.plot`'s ``return_model=True`` bundle always carries a
`hypertools.Pipeline` under its ``'pipeline'`` key, even for a single
stage. A `hypertools.Pipeline` can be applied to new data via
``.transform()`` and passed back in via ``pipeline=`` to
`hypertools.plot`/`hypertools.analyze`.

.. autosummary::
  :toctree:

  Pipeline

Describe
------------------

.. autosummary::
  :toctree:

  describe

Text vectorization
------------------

.. autosummary::
  :toctree:

  tools.text2mat

Gensim text models (GH #198) -- the optional ``gensim`` extra, installed
on demand (see :doc:`optional_dependencies`); pass by name (e.g.
``vectorizer='Word2Vec'``) to `hypertools.tools.text2mat`:

.. autosummary::
  :toctree:

  tools.gensim_models.Word2VecVectorizer
  tools.gensim_models.Doc2VecVectorizer
  tools.gensim_models.FastTextVectorizer
  tools.gensim_models.LdaVectorizer
  tools.gensim_models.LsiVectorizer
  tools.gensim_models.HdpVectorizer

I/O
------------------

.. autosummary::
  :toctree:

  io.lsl_stream
  io.LSLStream

Exceptions
------------------

HyperTools' I/O, backend, and remote-load/trust errors derive from
`hypertools.HypertoolsError`. Input-validation errors (invalid parameters or
data shapes) raise standard `ValueError`/`TypeError` with actionable messages.

.. autosummary::
  :toctree:

  HypertoolsError
  HypertoolsBackendError
  HypertoolsIOError

Tools
------------------
.. autosummary::
  :toctree:

  tools.format_data

.. autosummary::
  :toctree:

  tools.missing_inds

.. autosummary::
  :toctree:

  tools.df2mat
