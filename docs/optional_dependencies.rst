.. _optional_dependencies:

Optional dependencies
=====================

``pip install hypertools`` installs everything the core functionality needs:
plotting with matplotlib, dimensionality reduction, alignment, clustering,
normalization, and ``Kalman``/``ARIMA`` forecasting and imputation. The
heavier model families are declared as ``pip`` extras of ``hypertools`` in
``pyproject.toml``. You can install them ahead of time, or let hypertools
install them the first time a call needs one.

The extras
----------

.. list-table::
   :header-rows: 1
   :widths: 14 24 62

   * - Extra
     - Installs
     - What it adds
   * - ``interactive``
     - plotly, kaleido
     - the interactive backend: ``hyp.plot(..., backend='plotly')`` and
       ``hyp.set_interactive_backend('plotly')``; kaleido exports plotly
       figures and animation frames as static images
   * - ``text``
     - pydata-wrangler[hf] (torch, transformers, sentence-transformers,
       tokenizers, datasets)
     - Hugging Face text embeddings, e.g.
       ``hyp.plot(texts, vectorizer='all-MiniLM-L6-v2')``, and
       ``hyp.load`` of Hugging Face datasets
   * - ``predict``
     - skaters
     - the ``Laplace`` ensemble forecaster for ``hyp.predict``
       (``Kalman``, ``GaussianProcess``, ``AutoRegressor`` and ``ARIMA``
       work with the base install)
   * - ``predict-hf``
     - chronos-forecasting
     - the Hugging Face ``Chronos`` forecaster for ``hyp.predict``
   * - ``torch``
     - torch
     - the six autoencoder reducers (``reduce='Autoencoder'``,
       ``DeepAutoencoder``, ``SparseAutoencoder``,
       ``ConvolutionalAutoencoder``, ``SequenceAutoencoder``,
       ``VariationalAutoencoder``)
   * - ``gensim``
     - gensim
     - ``Word2Vec``/``Doc2Vec``/``FastText`` vectorizers and
       ``LdaModel``/``LsiModel``/``HdpModel`` semantic models for
       ``hyp.tools.text2mat`` and text plotting
   * - ``kaggle``
     - kagglehub
     - ``hyp.load('kaggle/<owner>/<dataset>')``
   * - ``lsl``
     - pylsl
     - ``hyp.io.lsl_stream()``, Lab Streaming Layer input for streaming
       plots
   * - ``density3d``
     - scikit-image
     - smooth 3-D ``density=True`` iso-surfaces on the matplotlib backend
       (without it, 3-D density falls back to a translucent scatter cloud;
       the plotly backend renders a volume either way)
   * - ``io``
     - openpyxl
     - ``.xlsx`` support for ``hyp.load``

Extras combine: ``pip install "hypertools[interactive,torch]"``. The ``dev``
extra holds the test and development dependencies and is not installed on
demand.

Installation on demand
----------------------

You do not have to install an extra before using the feature it provides.
The first call that needs a missing optional package installs that extra's
requirements into the running interpreter (``python -m pip install ...``,
using the requirement strings hypertools declares in ``pyproject.toml``),
prints a one-line notice, and carries on::

    hypertools: installing plotly>=6.1.1, kaleido>=1.0 (needed for the plotly backend) ...

Only the extra's own requirements are installed. hypertools itself is never
reinstalled, so an editable or branch install stays exactly as it is.

If the install fails (no network, no permission to write to the
environment), the call raises ``ImportError`` naming the manual command,
e.g. ``pip install "hypertools[interactive]"``.

Turning it off
~~~~~~~~~~~~~~

Set the environment variable ``HYPERTOOLS_AUTO_INSTALL=0`` (``false``,
``no`` and ``off`` also work); it is read at each call. A missing extra
then raises ``ImportError`` with the manual ``pip install
"hypertools[<extra>]"`` command, and nothing is installed.
This is the setting to use in locked-down environments, in CI images built
ahead of time, and anywhere pip should not run inside a Python process.

Chrome for static plotly export
-------------------------------

Interactive plotly output (HTML, notebooks) needs no browser. Exporting a
plotly figure to PNG/PDF, or saving a plotly animation to a GIF or video,
goes through kaleido, which renders each frame in a headless Chrome. The
first such export checks that kaleido can render and, when it cannot,
provisions what is missing:

- on Debian/Ubuntu images that lack them (a fresh Colab or Kaggle kernel),
  the four system libraries Chrome needs to start -- ``libatk1.0-0``,
  ``libatk-bridge2.0-0``, ``libatspi2.0-0`` and ``libxcomposite1`` -- via
  ``apt-get``. This step runs only when the process is root or has
  password-less ``sudo``; otherwise it is skipped.
- a Chrome build for kaleido (about 150 MB), via ``plotly.io.get_chrome()``.

Both steps print a one-line ``hypertools:`` notice, and both are
skipped when ``HYPERTOOLS_AUTO_INSTALL=0``. When no working Chrome could be
provided, the export raises ``HypertoolsIOError`` with the commands to run
yourself: ``import plotly.io as pio; pio.get_chrome()`` and, on
Debian/Ubuntu, ``apt-get install -y libatk1.0-0 libatk-bridge2.0-0
libatspi2.0-0 libxcomposite1``. Installing Chrome or Chromium through your
system's package manager also works, and so does saving the figure with the
matplotlib backend instead (``backend='matplotlib'``).

Pre-installing everything
-------------------------

To fetch every optional package up front (for an offline machine, a Docker
image, or a CI runner), list the extras you want in one install::

    pip install "hypertools[interactive,text,predict,predict-hf,torch,gensim,kaggle,lsl,density3d,io]"

Chrome is not a pip package; to pre-provision it for static plotly export,
run ``python -c "import plotly.io as pio; pio.get_chrome()"`` after the
install (and the ``apt-get`` line above on a bare Debian/Ubuntu image).
