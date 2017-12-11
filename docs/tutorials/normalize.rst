
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
<<<<<<< HEAD
=======
    from scipy.linalg import toeplitz
>>>>>>> c46e7a9822fb6d89bac77708762d883add1dae6e
    
    %matplotlib inline

Generate synthetic data
-----------------------

<<<<<<< HEAD
First, we generate two sets of synthetic data. We pull points randomly
from a multivariate normal distribution for each set, so the sets will
exhibit unique statistical properties.

.. code:: ipython3

    x1 = np.random.randn(10,10)
    x2 = np.random.randn(10,10)
    
    c1 = np.dot(x1, x1.T)
    c2 = np.dot(x2, x2.T)
    
    m1 = np.zeros([1,10])
    m2 = 10 + m1
    
    data1 = np.random.multivariate_normal(m1[0], c1, 100)
    data2 = np.random.multivariate_normal(m2[0], c2, 100)
    
    data = [data1, data2]
=======
.. code:: ipython3

    cluster1 = np.random.multivariate_normal(np.zeros(10), 10 - toeplitz(np.arange(10)), size=100)
    cluster2 = np.random.multivariate_normal(np.zeros(10)+10, 10 - toeplitz(np.arange(10)), size=100)
    data = [cluster1, cluster2]
>>>>>>> c46e7a9822fb6d89bac77708762d883add1dae6e

Visualize the data
------------------

.. code:: ipython3

    geo = hyp.plot(data, '.')



<<<<<<< HEAD
.. image:: normalize_files/normalize_8_0.png
=======
.. image:: normalize_files/normalize_7_0.png
>>>>>>> c46e7a9822fb6d89bac77708762d883add1dae6e


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



<<<<<<< HEAD
.. image:: normalize_files/normalize_13_0.png
=======
.. image:: normalize_files/normalize_12_0.png
>>>>>>> c46e7a9822fb6d89bac77708762d883add1dae6e


Normalizing 'within'
~~~~~~~~~~~~~~~~~~~~

When you normalize 'within', normalization is done on the columns of
each element of the data, separately.

.. code:: ipython3

    norm = hyp.normalize(data, normalize = 'within')
    geo = hyp.plot(norm, '.')



<<<<<<< HEAD
.. image:: normalize_files/normalize_16_0.png
=======
.. image:: normalize_files/normalize_15_0.png
>>>>>>> c46e7a9822fb6d89bac77708762d883add1dae6e


Normalizing by 'row'
~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    norm = hyp.normalize(data, normalize = 'row')
    geo = hyp.plot(norm, '.')



<<<<<<< HEAD
.. image:: normalize_files/normalize_18_0.png
=======
.. image:: normalize_files/normalize_17_0.png
>>>>>>> c46e7a9822fb6d89bac77708762d883add1dae6e

