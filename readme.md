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

## What's new

**1.1** makes hierarchical (`MultiIndex`) DataFrames a first-class input,
grows the animation API, adds model comparisons for forecasting and
imputation, loads text and market data in one line, and installs optional
extras on demand.

**1.0** was a ground-up rewrite that keeps the familiar API. It added an
optional interactive (plotly) backend, `hyp.Pipeline`, `hyp.manip`,
`hyp.predict` and `hyp.impute`, morph and 2-D animations, and more text
embedding models, and it runs on Python 3.10–3.13. A few long-deprecated
arguments were removed.

The [changelog](https://github.com/ContextLab/hypertools/blob/master/CHANGELOG.md)
has the complete list for both releases, including what changed for code and
saved data from earlier versions.

## A quick tour

Each clip below is one of the animated examples from the
[tutorials](https://hypertools.readthedocs.io/en/latest/tutorials.html); follow
the link under a clip for the full code and the full-length animation.

### Many datasets in one space

Six stock-market sectors, each reduced to 3-D on its own and then hyperaligned
into a shared space, with a heavier path for the market as a whole.

![Market sectors example](https://raw.githubusercontent.com/ContextLab/hypertools/v1.1.0/images/tour_market_sectors.gif)

[Tutorial: market sectors](https://hypertools.readthedocs.io/en/latest/tutorials/market_sectors.html)

### Text goes in, geometry comes out

Give `hyp.plot` a list of strings and it embeds them, projects them into 2-D
or 3-D, and plots the result. Here a paragraph about each of five paintings
becomes its own cloud, drawn in a color pulled from the canvas itself.

![Painting embeddings example](https://raw.githubusercontent.com/ContextLab/hypertools/v1.1.0/images/tour_painting_embeddings.gif)

[Tutorial: painting embeddings](https://hypertools.readthedocs.io/en/latest/tutorials/painting_embeddings.html)

### Reveal a dataset over time

`order='serial'` reveals a dataset one piece at a time. This is the Mad
Tea-Party from *Alice's Adventures in Wonderland*: each segment is one turn of
the conversation, colored by speaker and placed by what is being said.

![Conversation example](https://raw.githubusercontent.com/ContextLab/hypertools/v1.1.0/images/tour_conversation_shape.gif)

[Tutorial: the shape of a conversation](https://hypertools.readthedocs.io/en/latest/tutorials/conversation_shape.html)

### Many features, one path

Monthly temperatures in 20 cities around the globe, from 1875 to 2013, drawn as
a single path and colored by the average temperature that month.

![Weather example](https://raw.githubusercontent.com/ContextLab/hypertools/v1.1.0/images/tour_weather_decades.gif)

[Tutorial: weather across the decades](https://hypertools.readthedocs.io/en/latest/tutorials/weather_decades.html)

### Morph between shapes

`animate='morph'` interpolates between point clouds or meshes. These are some
of the built-in shapes that `hyp.load` provides.

```python
import hypertools as hyp
names = ['bunny', 'cube', 'sphere', 'teapot', 'vase']
shapes = [hyp.load(n) for n in names]
hyp.plot(shapes, '.', color='k', animate='morph', title=names)
```

![Morph example](https://raw.githubusercontent.com/ContextLab/hypertools/v1.1.0/images/tour_morph_shapes_zoo.gif)

[Tutorial: morphing the shapes zoo](https://hypertools.readthedocs.io/en/latest/tutorials/morph_shapes_zoo.html)

## Try it!

Check the [repo](https://github.com/ContextLab/hypertools-paper-notebooks) of
Jupyter notebooks from the HyperTools [paper](https://arxiv.org/abs/1701.08290)
(note: those notebooks predate the 1.x API described below). For up-to-date,
runnable examples covering every 1.1 feature, see the
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
lacks). `hyp.set_autoinstall(False)` turns this off (for the session, or
for one block as a context manager); a missing extra then raises
`ImportError` with the manual `pip install` command.

## Requirements

+ python>=3.10
+ scikit-learn>=1.5.2
+ pandas>=2.2.3
+ seaborn>=0.13.0
+ pillow>=10.4.0
+ matplotlib>=3.9.2
+ scipy>=1.14.1
+ numpy>=2.1.0
+ umap-learn>=0.5.5, numba>=0.61.0
+ pydata-wrangler>=0.5.1 (data-wrangling core)
+ pykalman>=0.11, statsmodels>=0.14.3 (Kalman/ARIMA forecasting; Kalman imputation)
+ requests>=2.31.0, dill>=0.3.8, ipympl>=0.9.3
+ ffmpeg (for saving animations)

All Python dependencies are declared in `pyproject.toml` and installed
automatically by pip. The base install covers all core functionality
(plotting, dimensionality reduction, alignment, clustering, normalization,
`Kalman`/`ARIMA` forecasting, and missing-data imputation) and therefore pulls in the
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
