![Hypertools logo](https://raw.githubusercontent.com/ContextLab/hypertools/v1.1.0/images/hypercube.png)

[![Tests](https://img.shields.io/github/actions/workflow/status/ContextLab/hypertools/test.yml?label=tests)](https://github.com/ContextLab/hypertools/actions/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/hypertools/badge/?version=latest)](https://hypertools.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://img.shields.io/pypi/v/hypertools.svg)](https://pypi.org/project/hypertools/)

"_To deal with hyper-planes in a 14 dimensional space, visualize a 3D space and say 'fourteen' very loudly.  Everyone does it._" - Geoff Hinton


![Hypertools example](https://raw.githubusercontent.com/ContextLab/hypertools/v1.1.0/images/story_trajectories.gif)

## Overview

HyperTools is designed to facilitate
[dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction)-based
visual explorations of high-dimensional data.  The basic pipeline is
to feed in a high-dimensional dataset (or a series of high-dimensional
datasets) and, in a single function call, reduce the dimensionality of
the dataset(s) and create a plot.  The package is built atop many
familiar friends, including [matplotlib](https://matplotlib.org/),
[scikit-learn](http://scikit-learn.org/) and
[seaborn](https://seaborn.pydata.org/).  Our package was featured in 2017 on
Kaggle's now-retired "No Free Hunch" blog
([archived copy](http://web.archive.org/web/20191202152212/http://blog.kaggle.com:80/2017/04/10/exploring-the-structure-of-high-dimensional-data-with-hypertools-in-kaggle-kernels/)).
For a general overview, you may find [this talk](https://www.youtube.com/watch?v=hb_ER9RGtOM) useful (given as part of the [MIND Summer School](https://summer-mind.github.io) at Dartmouth).

## What's new in 1.0

HyperTools 1.0 modernizes the toolbox while keeping the familiar API:

+ **Interactive plotting (optional):** `hyp.plot(..., backend='plotly')`
  renders interactive figures. With the default `backend='auto'`, HyperTools
  automatically uses plotly on Google Colab and Kaggle (where interactive
  figures work best and plotly is preinstalled) and matplotlib everywhere
  else — existing workflows are unchanged. The two backends produce visually
  matched output: identical colors, line/marker styles and sizes, format
  strings, and the signature cube/square framing.
+ **Multicolored lines:** passing continuous values (or a per-observation
  matrix) as `hue` together with a line format colors each trajectory
  continuously along its length, on both backends.
+ **`hyp.apply_model`:** a unified stack → fit-once → unstack core for
  applying any scikit-learn style model (or pipeline of models) across one
  or more datasets, with `return_model=True` for reuse on held-out data.
+ **Mixture-model ("soft") clustering:** `cluster` and `plot` support
  `GaussianMixture`, `BayesianGaussianMixture`, `LatentDirichletAllocation`,
  and `NMF`. `hyp.cluster` returns per-observation membership proportions,
  and `hyp.plot` colors each observation by blending component colors
  according to its mixture weights.
+ **Richer coloring:** the `hue` argument now accepts categorical labels,
  continuous values, or entire matrices (e.g. mixture proportions or model
  weights), which are mapped to colors via the new `mat2colors` helper
  (`from hypertools.plot.colors import mat2colors`).
+ **Nested-list input:** `hyp.plot([[a, b], [c]])` colors datasets by their
  outermost grouping and renders more deeply nested datasets with thinner,
  fainter lines.
+ **Hull surfaces (optional):** `hyp.plot(..., surface=True)` overlays a
  smooth, lit surface over each dataset's convex hull — a filled outline in
  2D, or a shaded, Taubin-smoothed "blob" in 3D. A dict of scalar options
  (e.g. `surface={'alpha': 0.6}`) customizes all surfaces at once; a list
  of bools/dicts (e.g. `surface=[{'alpha': 0.2}, {'alpha': 0.8}]`) controls
  alpha/color/lighting/smoothing per dataset.
+ **Morph animation:** `hyp.plot(datasets, animate='morph')` treats each
  dataset as a point cloud and morphs smoothly between them (Hungarian-
  matched, smoothstep-eased), holding on each one along the way; `rotations`
  accepts a per-segment list for independent camera control over each hold
  and transition, and `surface=True` composes with it to morph a lit hull
  instead of raw points. See `examples/animate_morph_zoo.py` and
  `examples/animate_surface_morph.py`.
+ **More animation styles, and 2-D animation:** in addition to `True`/
  `'parallel'`, `animate='spin'` keeps all the data drawn and spins the
  camera, `'serial'` reveals each dataset one at a time in list order, and
  `'window'` (`hyp.plot(traj, animate='window', focused=2)`) slides a fixed-
  length, fully-opaque window along each trajectory. Every style except
  `'spin'` (which is inherently a 3-D camera move) now also works for
  `ndims=2` data, using a fixed (non-rotating) viewport.
+ **`hyp.Pipeline`:** a scikit-learn-style `Pipeline` chains
  `manip`/`normalize`/`reduce`/`align`/`cluster` stages, fit once and reused
  on new data without refitting -- `hyp.plot(A, ..., return_model=True)`
  returns a bundle whose `'pipeline'` entry can be passed back in as
  `hyp.plot(B, pipeline=bundle['pipeline'])` to apply the exact same fitted
  transformation to a structurally-identical dataset `B`.
+ **`hyp.manip` and manip chaining:** `hyp.manip(data, model='ZScore')`
  applies a manipulation (`Normalize`, `ZScore`, `Smooth`, `Resample`) to
  each dataset. `Smooth` and `Resample` run independently per dataset
  (kernels never cross dataset boundaries); `ZScore` and `Normalize` also
  transform each dataset separately but fit one shared set of statistics
  across all datasets in a list (like `normalize='across'`). A `list` of
  specs chains several manipulations as a `Pipeline`, and
  `hyp.plot(data, manip=[...])` runs the chain at the canonical `manip`
  stage -- first, before `normalize`/`reduce`/`align`/`cluster` -- e.g.
  `hyp.plot(data, manip=[{'model': 'Smooth', 'kwargs': {'kernel_width': 5}},
  'ZScore'], reduce='PCA')`.
+ **Autoencoder reducers (optional):** `hyp.reduce`/`hyp.plot(..., reduce=
  'Autoencoder')` supports six torch-backed autoencoder reducers
  (`Autoencoder`, `DeepAutoencoder`, `SparseAutoencoder`,
  `ConvolutionalAutoencoder`, `SequenceAutoencoder`,
  `VariationalAutoencoder`), from the `[torch]` extra (installed on demand
  the first time one is fit).
+ **gensim text vectorizers/semantic models (optional):** `hyp.plot(texts,
  vectorizer='Word2Vec', semantic=None, reduce='PCA')` (and
  `hyp.tools.text2mat`) add `Word2Vec`/`Doc2Vec`/`FastText` vectorizers and
  `LdaModel`/`LsiModel`/`HdpModel` semantic models, from the `[gensim]`
  extra (installed on demand) -- with an embedding vectorizer (gensim, or a
  Hugging Face model id like `'all-MiniLM-L6-v2'`) the default LDA semantic
  stage is skipped automatically (gensim warns; pass `semantic=None` to
  silence it) and `corpus` is unused. Note that non-default vectorizers
  train on the default `corpus='wiki'` corpus the first time, which can
  take a couple of minutes even for tiny inputs; pass `corpus=` a list of
  your own documents to train on those instead.
+ **LSL streaming (optional):** `hyp.io.lsl_stream(type='EEG')` resolves a
  live Lab Streaming Layer stream and wraps it for `hyp.plot(...,
  stream_init=200, stream_chunk=20)`, from the `[lsl]` extra (installed
  on demand).
+ **`hyp.predict`/`hyp.impute`:** `hyp.predict(data, model='Kalman', t=10)`
  forecasts `t` new rows continuing each dataset. `Kalman`, `GaussianProcess`,
  `AutoRegressor`, and `ARIMA` work with the base install (pykalman/statsmodels
  are core deps); `Laplace` comes from the `[predict]` extra (skaters) and `Chronos`
  from `[predict-hf]`, each installed on demand. `hyp.impute(data,
  model='PPCA')` fills missing (NaN) values in place; the `Kalman` imputer also
  works with the base install. Both take `return_model=True` for reuse.
+ **Kaggle loader:** `hyp.load('kaggle/uciml/iris')` downloads a public
  Kaggle dataset anonymously via `kagglehub`, from the `[kaggle]` extra
  (installed on demand).
+ **Density shading (optional):** `hyp.plot(..., density=True)` overlays a
  subtle KDE "glow" behind the data (a 2D heatmap or 3D volumetric cloud)
  showing where each dataset's points concentrate; off by default.
+ **Colorbars:** `hyp.plot(..., colorbar=True)` draws a colorbar matching
  whatever color mapping is already in use — a continuous gradient for a
  numeric `hue`, or a segmented, labeled bar for discrete groups/clusters.
+ **MultiIndex DataFrames:** a DataFrame with a row `MultiIndex` is expanded
  automatically into one leaf trace per index combination plus a thicker,
  more opaque mean trace per level of grouping, colored by the top-level
  index value.
+ **Per-dataset animation trails:** `chemtrails`/`precog`/`bullettime` each
  accept a list of bools (one per dataset) so different datasets in the same
  animation can show different trail styles, in addition to a single bool
  applied to all of them.
+ **More `hyp.load` sources:** in addition to the built-in example datasets
  and local files, `hyp.load` now resolves Hugging Face dataset ids, Google
  Sheets/Drive links, Dropbox links, and arbitrary URLs, plus more local file
  formats (`.npy`/`.npz`, `.csv`/`.tsv`/`.txt`, `.json`, `.parquet`, `.mat`,
  `.xlsx`/`.xls`).
+ **Faster and cleaner:** `import hypertools` is ~3.5x faster (heavy
  dependencies load lazily); plotting no longer mutates global matplotlib
  settings; the unreliable result cache was removed; HDBSCAN now comes from
  scikit-learn (no extra dependency); packaging follows current standards
  (pyproject.toml, Python 3.10–3.13).
+ **Retired legacy arguments:** the long-deprecated `group` (use `hue`),
  `model`/`model_params` (use `reduce`), and `align(method=...)`/
  `align=True` (use `align='hyper'`) arguments were removed and now raise
  errors instead of being silently accepted. `cluster`'s `ndims=` is no longer a
  standalone reduction step: it is only forwarded to the `reduce=` stage,
  and a warning fires if it is passed without `reduce=`. **Legacy data:**
  geo files saved by hypertools **≥0.8** (pickle-format) still load — the
  internal unpickle-only shim reads them and returns their raw data, with
  retired arguments translated or skipped with a warning on replay. Older
  **pre-0.8 `deepdish`/HDF5-format** geos cannot be read directly under
  HyperTools' required NumPy 2 (the `deepdish` reader is unmaintained and
  imports only under `numpy<2`); `hyp.load` detects them and raises a
  message explaining the one-time out-of-process conversion:

  ```bash
  # in a throwaway environment, convert an old .geo to a modern format once
  python -m venv /tmp/dd && /tmp/dd/bin/pip install "numpy<2" deepdish
  /tmp/dd/bin/python -c "import deepdish as dd, numpy as np; \
      d = dd.io.load('old.geo'); np.savez('old_converted.npz', \
      **{'data': np.asarray(d['data'], dtype=object)})"
  # then load old_converted.npz with hypertools as usual
  ```

## Try it!

Check the [repo](https://github.com/ContextLab/hypertools-paper-notebooks) of
Jupyter notebooks from the HyperTools [paper](https://arxiv.org/abs/1701.08290)
(note: those notebooks predate the 1.0 API described below). For up-to-date,
runnable examples covering every 1.0 feature, see the
[example gallery](http://hypertools.readthedocs.io/en/latest/auto_examples/index.html)
in the docs.

## Installation

To install the latest stable version run:

`pip install hypertools`

To install the latest unstable version directly from GitHub, run:

`pip install -U git+https://github.com/ContextLab/hypertools.git`

Or alternatively, clone the repository to your local machine:

`git clone https://github.com/ContextLab/hypertools.git`

Then, navigate to the folder and type:

`pip install -e .`

(These instructions assume that you have [pip](https://pip.pypa.io/en/stable/installing/) installed on your system)

### Optional extras install themselves on demand

The optional features listed under *What's new* (the plotly backend, HF text
embeddings, the `Laplace` and `Chronos` forecasters, autoencoder reducers,
gensim vectorizers, Kaggle loading, LSL streaming, 3-D density iso-surfaces,
`.xlsx` loading) are declared as extras in `pyproject.toml`:
`pip install "hypertools[interactive]"`, `hypertools[text]`,
`hypertools[predict]`, `hypertools[predict-hf]`, `hypertools[torch]`,
`hypertools[gensim]`, `hypertools[kaggle]`, `hypertools[lsl]`,
`hypertools[density3d]`, `hypertools[io]`. You do not have to install them
ahead of time: the first call that needs one installs that extra's
requirements into the running interpreter (printing a one-line notice) and
carries on. hypertools itself is never reinstalled, so a development or
branch install stays as it is. Static image export with the plotly backend
also provisions what kaleido needs on first use (a Chrome build and, on
Debian/Ubuntu images such as Colab and Kaggle, the system libraries it
lacks). Set `HYPERTOOLS_AUTO_INSTALL=0` to turn this off; a missing extra
then raises `ImportError` with the manual `pip install` command.

## Requirements

+ python>=3.10
+ scikit-learn>=1.4.0
+ pandas>=2.2.0
+ seaborn>=0.13.0
+ matplotlib>=3.8.0
+ scipy>=1.13.0
+ numpy>=2.0.0
+ umap-learn>=0.5.5, numba>=0.59
+ pydata-wrangler>=0.5.1 (data-wrangling core)
+ pykalman>=0.11, statsmodels>=0.14 (Kalman/ARIMA forecasting and imputation)
+ requests, dill, ipympl
+ ffmpeg (for saving animations)

All Python dependencies are declared in `pyproject.toml` and installed
automatically by pip. The base install covers all core functionality
(plotting, dimensionality reduction, alignment, clustering, normalization,
and `Kalman`/`ARIMA` forecasting + imputation) and therefore pulls in the
full scientific stack (NumPy, SciPy, pandas, scikit-learn, matplotlib,
seaborn, UMAP/Numba, statsmodels, pykalman, ipympl, pydata-wrangler); it is
not a minimal footprint. Heavier optional model families are separated into
extras that add features on request (mix and match, e.g.
`pip install "hypertools[interactive,torch]"`):

+ `interactive` -- plotly + kaleido, for `hyp.plot(..., backend='plotly')`.
  kaleido renders static images (PNG/PDF, and the frames of saved plotly
  animations) through a headless Chrome, which hypertools provisions on
  first use (see *Optional extras install themselves on demand* above); if
  that is not possible, the error says what to run. Interactive/HTML plotly
  output needs no browser.
+ `text` -- transformer/sentence-transformers text embeddings (via
  datawrangler's `hf` extra)
+ `predict` -- the skaters `Laplace` ensemble forecaster for `hyp.predict`
  (`Kalman`, `GaussianProcess`, `AutoRegressor`, and `ARIMA` already work
  with the base install)
+ `predict-hf` -- the Hugging Face `Chronos` forecaster for `hyp.predict`
+ `io` -- `.xlsx` support for `hyp.load`
+ `density3d` -- smooth 3-D `density=True` iso-surfaces (scikit-image)
+ `torch` -- the six autoencoder reducers (`reduce='Autoencoder'` and
  variants)
+ `kaggle` -- `hyp.load('kaggle/<owner>/<dataset>')`
+ `lsl` -- `hyp.io.lsl_stream(...)` (Lab Streaming Layer input)
+ `gensim` -- `Word2Vec`/`Doc2Vec`/`FastText` vectorizers and
  `LdaModel`/`LsiModel`/`HdpModel` semantic models
+ `dev` -- test/development dependencies (`pip install -e ".[dev]"`)

## Documentation

Check out our [readthedocs](http://hypertools.readthedocs.io/en/latest/) page for further documentation, complete API details, and additional examples.

## Citing

We wrote a short JMLR paper about HyperTools, which you can read [here](http://jmlr.org/papers/v18/17-434.html), or you can check out a (longer) preprint [here](https://arxiv.org/abs/1701.08290). We also have a repository with example notebooks from the paper [here](https://github.com/ContextLab/hypertools-paper-notebooks).

Please cite as:

`Heusser AC, Ziman K, Owen LLW, Manning JR (2018) HyperTools: A Python toolbox for gaining geometric insights into high-dimensional data.  Journal of Machine Learning Research, 18(152): 1--6.`

Here is a bibtex formatted reference:

```bibtex
@ARTICLE{heusser2018hypertools,
    author  = {Andrew C. Heusser and Kirsten Ziman and Lucy L. W. Owen and Jeremy R. Manning},    
    title   = {HyperTools: a Python Toolbox for Gaining Geometric Insights into High-Dimensional Data},    
    journal = {Journal of Machine Learning Research},
    year    = {2018},
    volume  = {18},	
    number  = {152},	
    pages   = {1-6},	
    url     = {http://jmlr.org/papers/v18/17-434.html}	
}
```

## Contributing

If you'd like to contribute, please first read our [Code of Conduct](https://www.mozilla.org/en-US/about/governance/policies/participation/).

For specific information on how to contribute to the project, please see our [Contributing](https://github.com/ContextLab/hypertools/blob/master/CONTRIBUTING.md) page.

## Testing

CI runs on every push via [GitHub Actions](https://github.com/ContextLab/hypertools/actions/workflows/test.yml)
(badge at the top of this page).

To test HyperTools locally, install pytest (`pip install -e ".[dev]"`) and run `pytest` in the HyperTools folder.

## Examples

See [here](http://hypertools.readthedocs.io/en/latest/auto_examples/index.html) for more examples.

## Plot

```python
import numpy as np
import hypertools as hyp

# two random-walk "datasets" (rows = observations, columns = features)
walk = lambda seed: np.cumsum(np.random.default_rng(seed).standard_normal((300, 10)), axis=0)
list_of_arrays = [walk(1), walk(2)]
list_of_labels = ['A'] * 300 + ['B'] * 300  # one label per observation

hyp.plot(list_of_arrays, animate=True, hue=list_of_labels)
```

![Plot example](https://raw.githubusercontent.com/ContextLab/hypertools/v1.1.0/images/plot.gif)

## Align

```python
import numpy as np
import hypertools as hyp

# rotated, noisy views of one shared trajectory
rng = np.random.default_rng(0)
base = np.cumsum(rng.standard_normal((300, 3)), axis=0)
list_of_arrays = [base @ np.linalg.qr(rng.standard_normal((3, 3)))[0]
                  + 0.05 * rng.standard_normal(base.shape) for _ in range(3)]

hyp.plot(list_of_arrays, align='hyper')
```

### BEFORE

![Align before example](https://raw.githubusercontent.com/ContextLab/hypertools/v1.1.0/images/align_before.gif)

### AFTER

![Align after example](https://raw.githubusercontent.com/ContextLab/hypertools/v1.1.0/images/align_after.gif)


## Cluster

Soft ("mixture-model") clustering, new in 1.0 -- each point's color blends
its component memberships:

```python
import numpy as np
import hypertools as hyp

# three overlapping point clouds
rng = np.random.default_rng(0)
array = np.vstack([rng.standard_normal((100, 3)) + offset
                   for offset in ([0, 0, 0], [4, 0, 0], [0, 4, 0])])

hyp.plot(array, 'o', cluster='GaussianMixture', n_clusters=3)
```

![Cluster Example](https://raw.githubusercontent.com/ContextLab/hypertools/v1.1.0/images/cluster_example.png)


## Surfaces

New in 1.0: overlay a smooth, lit surface over each dataset's convex hull:

```python
import numpy as np
import hypertools as hyp

rng = np.random.default_rng(0)
blob_a = rng.standard_normal((100, 3))
blob_b = rng.standard_normal((100, 3)) + [4, 0, 0]

hyp.plot([blob_a, blob_b], '.', surface=True)
```

![Surface Example](https://raw.githubusercontent.com/ContextLab/hypertools/v1.1.0/images/surface_example.png)


## Describe

```python
import numpy as np
import hypertools as hyp

rng = np.random.default_rng(0)
list_of_arrays = [np.cumsum(rng.standard_normal((200, 20)), axis=0)
                  for _ in range(3)]

hyp.describe(list_of_arrays, reduce='PCA', max_dims=14)
```
![Describe Example](https://raw.githubusercontent.com/ContextLab/hypertools/v1.1.0/images/describe_example.png)
