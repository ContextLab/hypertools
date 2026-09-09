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
  align.score.alignment_score

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

.. _observation-times:

Observation times and future steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Forecasting uses each dataset's own observation times. Datetime, timedelta,
period and unique numeric indexes are sorted together with their values before
fitting. Periods are represented by their start timestamps. Duplicate time
stamps are rejected; repeated numeric row IDs retain their positional meaning
(for example, stacked runs). Arrays and categorical row labels use observation
order. Shuffling timed rows therefore does not change the fitted forecast.

One future step is the **median positive gap** between sorted timestamps.
Override it with ``hyp.predict(data, step='1h')`` for datetime/duration indexes,
or ``step=0.5`` for numeric coordinates. Each dataset gets its own inferred
interval. A fitted model keeps its training interval when applied to new data,
so a learned one-hour transition never silently becomes a three-hour transition.

GaussianProcess fits the actual times, expressed as elapsed multiples of the
model's step. Kalman, ARIMA, AutoRegressor, Laplace and Chronos assume regular
steps: irregular data are **linearly interpolated** column by column onto a
regular grid ending at the latest observation, with a warning. The grid stays
inside the observed time span; training values are never extrapolated. Existing
missing values are not imputed by this operation. Interpolation can smooth
short-lived changes; choose the interval for your data, or use GaussianProcess
to retain the original observation times without interpolation. Models retain
their existing univariate/multivariate behavior.

``hyp.plot(..., predict=...)`` follows the same policy. In ``ndims=1`` mode,
the time index supplies predictor coordinates rather than becoming another
signal column to forecast. The columns of each dataset are forecast together,
then split into lines for drawing. For a step override in a plot, use
``predict={'model': 'Kalman', 'kwargs': {'step': '1h'}}``. Animated forecasts
use only the observations revealed so far and wait until the model has enough
history, including enough interpolated grid points. If preprocessing changes
the row count and discards the corresponding timestamps, pass the analyzed
data with its updated index explicitly instead of guessing its times.

Backtesting at observation times
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``hyp.predict(..., holdout=...)`` sorts timed observations before holding out
the last rows. The model and its interval are fitted on the remaining training
rows only. GaussianProcess evaluates predictions at the actual held-out times.
Regular-grid models forecast far enough to cover those times, then select
matching grid points or **linearly interpolate predictions** between them.
For a held-out time before the first full forecast step, interpolation starts
at the last observed training value. Missing endpoints remain missing;
held-out values are never used in fitting or interpolation.

Returned forecasts, the naive baseline and ``'truth'`` share the held-out
index. ``horizon`` in the scores table counts held-out observations, which can
differ from the number of regular forecast steps. Arrays, categorical row
labels and repeated numeric row IDs are scored by observation position.
Choosing trading-day positions instead of calendar dates is therefore an
explicit modeling choice; the stock-forecasting tutorial demonstrates both.

For example, these observations are scored at times 12 and 20, rather than
being compared with the model's next two regular steps at times 9 and 10:

.. doctest::

   >>> import pandas as pd
   >>> import hypertools as hyp
   >>> from sklearn.gaussian_process.kernels import DotProduct
   >>> timed = pd.DataFrame({'value': [0., 1., 2., 4., 7., 8., 12., 20.]},
   ...                      index=[0., 1., 2., 4., 7., 8., 12., 20.])
   >>> scores, evaluated = hyp.predict(
   ...     timed, model='GaussianProcess', holdout=2, return_forecasts=True,
   ...     kernel=DotProduct(sigma_0=1, sigma_0_bounds='fixed'))
   >>> evaluated['GaussianProcess'].index.tolist()
   [12.0, 20.0]
   >>> evaluated['GaussianProcess'].index.equals(evaluated['truth'].index)
   True

Plot
------------------

.. autosummary::
  :toctree:

  plot
  subplots
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

Colors extracted from an image are put in a deterministic order before they
become a plot palette -- by value, dark to bright, unless ``palette_sort=``
(or ``?sort=`` in an ``'image:<path>'`` spec) asks for another key -- while
``image_palette`` itself, and the lead color of a dataset an image stands
for, keep the most-salient-first order.

.. autofunction:: hypertools.plot.colors.sort_colors

A t x k data matrix is a palette too. ``matrix_palette`` reduces it to three
dimensions with ``hypertools.reduce`` (``palette_reduce=`` in ``plot``,
default ``'PCA'``, with ``palette_manip=``/``palette_normalize=``/
``palette_align=`` passed through), scales each reduced column to [0, 1] as
an RGB channel, sorts the rows (default ``'columns'``: along the first
component) and returns a colormap that a plot resamples by interpolation to
as many colors as it needs.

.. autofunction:: hypertools.plot.colors.matrix_palette

.. autoclass:: hypertools.plot.colors.MatrixColormap

Set interactive backend
------------------------

.. autosummary::
  :toctree:

  set_interactive_backend

Set autoinstall
------------------------

.. autosummary::
  :toctree:

  set_autoinstall

The optional features (the plotly backend, text embeddings, ``Laplace`` and
``Chronos`` forecasting, the torch autoencoders, gensim models, Kaggle and
Hugging Face loading, LSL streaming, 3-D density iso-surfaces, ``.xlsx``
files) are ``pip`` extras that install themselves on demand: the first call
that needs one installs that extra's requirements, prints a one-line
``hypertools:`` notice and carries on. ``set_autoinstall(False)`` turns
this off (for the session, or for one block as a context manager); a
missing extra then raises ``ImportError`` naming the manual ``pip install
"hypertools[<extra>]"`` command. See :doc:`optional_dependencies` for the
extras, the Chrome step behind static plotly export, and how to
pre-install everything.

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
  io.synthetic_outlet

Exceptions
------------------

HyperTools' I/O and backend errors derive from `hypertools.HypertoolsError`.
`HypertoolsOfflineError` (``hyp.load(..., offline=True)`` with no cached copy
to read) is a `HypertoolsIOError`, importable from ``hypertools`` and
``hypertools.io``. `HypertoolsTrustError` is raised by `load` when a remote
payload would have to be unpickled (a pickle, or an object-array .npy/.npz)
and ``trust=True`` was not passed; it subclasses `ValueError`, not
`HypertoolsError`, and is importable from ``hypertools`` (it is defined in
``hypertools.io.sources``). Input-validation
errors (invalid parameters or data shapes) raise standard
`ValueError`/`TypeError` with actionable messages.

.. autosummary::
  :toctree:

  HypertoolsError
  HypertoolsBackendError
  HypertoolsIOError
  HypertoolsOfflineError
  HypertoolsTrustError

Tools
------------------
.. autosummary::
  :toctree:

  text_windows
  damage
  stack

.. autosummary::
  :toctree:

  tools.format_data

.. autosummary::
  :toctree:

  tools.missing_inds

.. autosummary::
  :toctree:

  tools.df2mat
