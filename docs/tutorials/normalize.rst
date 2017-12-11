
Normalization
=============

The ``normalize`` is a helper function to z-score your data. This is
useful if your features (columns) are scaled differently within or
across datasets. By default, hypertools normalizes *across* the columns
of all datasets passed, but also affords the option to normalize columns
*within* individual lists. Alternatively, you can also normalize each
row. The function returns an array or list of arrays where the columns
or rows are z-scored (output type same as input type).

Import packages
---------------

.. code:: ipython3

    import hypertools as hyp
    import numpy as np
    from scipy.linalg import toeplitz
    
    %matplotlib inline

Generate synthetic data
-----------------------

.. code:: ipython3

    cluster1 = np.random.multivariate_normal(np.zeros(10), 10 - toeplitz(np.arange(10)), size=100)
    cluster2 = np.random.multivariate_normal(np.zeros(10)+10, 10 - toeplitz(np.arange(10)), size=100)
    data = [cluster1, cluster2]

Visualize the data
------------------

.. code:: ipython3

    geo = hyp.plot(data, '.')



.. image:: normalize_files/normalize_7_0.png


Normalizing (Specified Cols or Rows)
------------------------------------

Or, to specify a different normalization, pass one of the following
arguments as a string, as shown in the examples below.

-  'across' - columns z-scored across passed lists (default)
-  'within' - columns z-scored within passed lists
-  'row' - rows z-scored

Normalizing 'across'
~~~~~~~~~~~~~~~~~~~~

When you normalize 'across', all of the data is stacked/combined, and
the normalization is done on the columns of the full dataset. Then the
data is split back into separate elements.

.. code:: ipython3

    norm = hyp.normalize(data, normalize = 'across')
    geo = hyp.plot(norm, '.')



.. image:: normalize_files/normalize_12_0.png


Normalizing 'within'
~~~~~~~~~~~~~~~~~~~~

When you normalize 'within', normalization is done on the columns of
each element of the data, separately.

.. code:: ipython3

    norm = hyp.normalize(data, normalize = 'within')
    geo = hyp.plot(norm, '.')



.. image:: normalize_files/normalize_15_0.png


Normalizing by 'row'
~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    norm = hyp.normalize(data, normalize = 'row')
    geo = hyp.plot(norm, '.')



.. image:: normalize_files/normalize_17_0.png

