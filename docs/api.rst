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

Autoencoder reducers (GH #162) -- optional ``torch`` extra
(``pip install "hypertools[torch]"``); pass by name (e.g.
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

Plot
------------------

.. autosummary::
  :toctree:

  plot
  HyperAnimation

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

`hypertools.Pipeline` is the fitted-model object returned by
``return_model=True`` when more than one pipeline stage runs (GH #227
#161); it can be applied to new data via ``.transform()`` and passed back in
via ``pipeline=`` to `hypertools.plot`/`hypertools.analyze`.

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

Gensim text models (GH #198) -- optional ``gensim`` extra
(``pip install "hypertools[gensim]"``); pass by name (e.g.
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

Exceptions
------------------

All hypertools-raised errors derive from `hypertools.HypertoolsError`:

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
