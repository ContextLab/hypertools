
Dimensionality reduction
========================

The ``reduce`` function reduces the dimensionality of an array or list
of arrays. The default is to use Principal Component Analysis to reduce
to three dimensions, but a variety of models are supported and users may
specify a desired number of dimensions other than three.

Supported models include: PCA, IncrementalPCA, SparsePCA,
MiniBatchSparsePCA, KernelPCA, FastICA, FactorAnalysis, TruncatedSVD,
DictionaryLearning, MiniBatchDictionaryLearning, TSNE, Isomap,
SpectralEmbedding, LocallyLinearEmbedding, and MDS.

Import Hypertools
-----------------

.. code:: ipython3

    import hypertools as hyp
    
    %matplotlib inline

Load your data
--------------

First, we'll load one of the sample datasets. This dataset is a list of
2 ``numpy`` arrays, each containing average brain activity (fMRI) from
18 subjects listening to the same story, fit using Hierarchical
Topographic Factor Analysis (HTFA) with 100 nodes. The rows are
timepoints and the columns are fMRI components.

See the `full
dataset <http://dataspace.princeton.edu/jspui/handle/88435/dsp015d86p269k>`__
or the `HTFA
article <https://www.biorxiv.org/content/early/2017/02/07/106690>`__ for
more info on the data and HTFA, respectively.

.. code:: ipython3

    weights = hyp.load('weights_avg')

Reduce one array
----------------

Let's look at one array from the dataset above.

.. code:: ipython3

    print('Array shape: (%d, %d)' % weights[0].shape)


.. parsed-literal::

    Array shape: (300, 100)


To reduce this array, simply pass the array to ``hyp.reduce``, as below.
We can see that the data has been reduced from 100 features to 3
features (the default when desired number of features is not specified).

.. code:: ipython3

    reduced_array = hyp.reduce(weights[0])
    print('Reduced array shape: (%d, %d)' % reduced_array.shape)


.. parsed-literal::

    Reduced array shape: (300, 3)


Reduce list of arrays
---------------------

A list or numpy array of multiple arrays can also be reduced into a
common space. That is, the data can be combined, reduced as a whole,
then split back into individual elements and outputted via hyp.reduce.

Here we show this with two arrays in the weights dataset. First, let's
examine the arrays in the weights dataset (below).

Now, let's reduce both arrays at once (by passing in the whole of the
weights data) and re-examine the data.

.. code:: ipython3

    reduced_arrays = hyp.reduce(weights)
    print('Shape of first reduced array: ', reduced_arrays[0].shape)
    print('Shape of second reduced array: ', reduced_arrays[1].shape)


.. parsed-literal::

    Shape of first reduced array:  (300, 3)
    Shape of second reduced array:  (300, 3)


We can see that each array has been reduced from 100 features to 3
features (the default when desired number of features is not specified),
with the number of datapoints unchanged.

Reduce list of arrays (TSNE)
----------------------------

You can also opt to use different reduction methods. In the example
below, we reduce multiple arrays at once, using TSNE. The data is
reduced to three dimensions(the default when desired number of features
not specified).

.. code:: ipython3

    reduced_TSNE = hyp.reduce(weights, reduce='TSNE')
    print('Shape of first reduced array: ',reduced_TSNE[0].shape)
    print('Shape of second reduced array: ',reduced_TSNE[1].shape)


.. parsed-literal::

    Shape of first reduced array:  (300, 3)
    Shape of second reduced array:  (300, 3)


Reduce to specified number of dimensions
----------------------------------------

You may prefer to reduce to a specific number of features, rather than
defaulting the three dimensions. To achieve this, simply pass the number
of desired features (as an int) to the ndims argument, as below.

.. code:: ipython3

    reduced_4 = hyp.reduce(weights, ndims = 4)
    print('Shape of first reduced array: ', reduced_4[0].shape)
    print('Shape of second reduced array: ', reduced_4[1].shape)


.. parsed-literal::

    Shape of first reduced array:  (300, 4)
    Shape of second reduced array:  (300, 4)


Reduce list of arrays with specific parameters
----------------------------------------------

For finer control of parameters, a dictionary of model parameters may be
passed to the reduce argument, in addition to the desired reduction
method. See `scikit-learn <http://scikit-learn.org/stable/index.html>`__
model docs for details on parameters supported for each model.

Supported models include: PCA, IncrementalPCA, SparsePCA,
MiniBatchSparsePCA, KernelPCA, FastICA, FactorAnalysis, TruncatedSVD,
DictionaryLearning, MiniBatchDictionaryLearning, TSNE, Isomap,
SpectralEmbedding, LocallyLinearEmbedding, and MDS.

The example below will reduce to the default of three features, since
the desired number of features is not specified.

.. code:: ipython3

    reduced_params = hyp.reduce(weights, reduce={'model' : 'PCA', 'params' : {'whiten' : True}})
