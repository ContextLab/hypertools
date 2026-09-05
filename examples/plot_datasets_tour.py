"""
==================================
A tour of hyp.load's data sources
==================================

`hypertools.load` resolves a plain string dataset name against five
sources, in order: HyperTools' own built-in example datasets (used
throughout the rest of this gallery), then four third-party sources:
`scikit-learn <https://scikit-learn.org>`_'s bundled datasets, `seaborn
<https://seaborn.pydata.org>`_'s example datasets, `FiveThirtyEight
<https://data.fivethirtyeight.com>`_'s published datasets (explicit
``'fivethirtyeight/<slug>'`` prefix), and `Kaggle <https://www.kaggle.com>`_
datasets (explicit ``'kaggle/<owner>/<dataset>'`` prefix, downloaded
anonymously via ``kagglehub`` -- no Kaggle account or API key required).
This example tours the four third-party sources, loading one small dataset
from each and plotting it in a 2x2 grid: one panel per source.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import hypertools as hyp

# Three of these four sources (seaborn, FiveThirtyEight, Kaggle) fetch over
# the network at run time. So that a transient outage doesn't abort the whole
# documentation-gallery build, each load is attempted independently and a
# source that can't be reached is simply skipped (with a printed note); the
# grid is drawn from whichever sources succeeded. In normal interactive use
# you would just call ``hyp.load(name)`` directly, as the first line of each
# block below shows.
sources = [
    # (title, loader, optional numeric-only cleanup for plotting)
    ('sklearn: iris', lambda: hyp.load('iris'), None),
    ('seaborn: penguins', lambda: hyp.load('penguins'),
     lambda df: df.select_dtypes('number').dropna()),
    ('fivethirtyeight: bechdel', lambda: hyp.load('fivethirtyeight/bechdel'),
     lambda df: df.select_dtypes('number').dropna()),
    ('kaggle: uciml/iris', lambda: hyp.load('kaggle/uciml/iris'), None),
]

loaded = []
for title, loader, clean in sources:
    try:
        data = loader()
    except Exception as e:  # network/rate-limit/optional-dep hiccup
        print(f"{title}: skipped ({type(e).__name__}: {e})")
        continue
    print(f"{title}: {getattr(data, 'shape', 'loaded')}")
    loaded.append((title, data if clean is None else clean(data)))

# plot each successfully loaded source side by side, colored/reduced
# automatically by hyp.plot. The rows of these tabular datasets are
# unordered samples, so each is drawn as points ('.') rather than as a
# connected line.
#
# These four sources are unrelated domains with different column counts and
# meanings (iris measurements, penguin measurements, Bechdel-test numeric
# columns, ...), so there is no single shared pipeline fit that would make
# sense across them -- hyp.plot's panels= draws one dataset per panel through
# ONE shared fit, which requires a common column space. We keep the
# explicit-axes form instead, via hyp.subplots (a thin wrapper over
# plt.subplots that pre-sets the 3-D projection and hands back a flat axes
# array), and fit each source's own pipeline independently as before.
n = max(len(loaded), 1)
ncols = 2 if n > 1 else 1
nrows = (n + ncols - 1) // ncols
fig, axes = hyp.subplots(nrows, ncols, size=[5 * ncols, 5 * nrows])
for ax in axes:
    ax.set_axis_off()  # hide any unused panels
for (title, data), ax in zip(loaded, axes):
    ax.set_axis_on()
    hyp.plot(data, '.', ax=ax, title=title, show=False)
fig.tight_layout()
