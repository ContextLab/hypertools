"""
==================================
A tour of hyp.load's data sources
==================================

`hypertools.load` resolves a plain string dataset name against several
built-in and third-party sources, in order (GH #116, #273): built-in
example datasets, `scikit-learn <https://scikit-learn.org>`_'s bundled
datasets, `seaborn <https://seaborn.pydata.org>`_'s example datasets,
`FiveThirtyEight <https://data.fivethirtyeight.com>`_'s published datasets
(explicit ``'fivethirtyeight/<slug>'`` prefix), and `Kaggle
<https://www.kaggle.com>`_ datasets (explicit ``'kaggle/<owner>/<dataset>'``
prefix, downloaded anonymously via ``kagglehub`` -- no Kaggle account or API
key required). This example tours all four, loading one small dataset from
each and plotting it.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import matplotlib.pyplot as plt
import hypertools as hyp

# 1. scikit-learn's bundled 'iris' dataset -- returned as a DataFrame with
# the target appended as a 'target' column
iris = hyp.load('iris')
print(f"iris (sklearn): {iris.shape}, columns={list(iris.columns)}")

# 2. seaborn's 'penguins' dataset -- fetched from the seaborn-data repo and
# returned unchanged
penguins = hyp.load('penguins')
print(f"penguins (seaborn): {penguins.shape}, "
      f"columns={list(penguins.columns)}")

# 3. a FiveThirtyEight dataset -- explicit 'fivethirtyeight/<slug>' prefix
bechdel = hyp.load('fivethirtyeight/bechdel')
print(f"bechdel (fivethirtyeight): {bechdel.shape}")

# 4. a Kaggle dataset -- explicit 'kaggle/<owner>/<dataset>' prefix,
# downloaded anonymously via kagglehub
kaggle_iris = hyp.load('kaggle/uciml/iris')
print(f"iris (kaggle): {kaggle_iris.shape}")

# plot each source side by side, colored/reduced automatically by hyp.plot
fig, axes = plt.subplots(2, 2, subplot_kw={'projection': '3d'},
                          figsize=(10, 10))
hyp.plot(iris, ax=axes[0, 0], title='sklearn: iris')
hyp.plot(penguins.select_dtypes('number').dropna(), ax=axes[0, 1],
         title='seaborn: penguins')
hyp.plot(bechdel.select_dtypes('number').dropna(), ax=axes[1, 0],
         title='fivethirtyeight: bechdel')
hyp.plot(kaggle_iris, ax=axes[1, 1], title='kaggle: uciml/iris')
plt.tight_layout()
plt.show()
