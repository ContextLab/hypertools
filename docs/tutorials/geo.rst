
DataGeometry objects (``geo``)
==============================

The DataGeometry object is the hypertools data object class. A ``geo``
contains the data, figure handles, and transform functions used to
create a plot. Note that this class should not be called directly, but
is used by the hyp.plot function to create a plot object.

In this tutorial we will explore the features of a geo and how it may be
used.

Import Hypertools
-----------------

.. code:: ipython3

    import hypertools as hyp
    import seaborn as sb
    
    %matplotlib inline

Load your data
--------------

In this case, we have used one of the sample datasets built into the
package.

.. code:: ipython3

    geo = hyp.load('mushrooms')

Plot it
-------

We can plot a ``geo`` by calling the plot method

.. code:: ipython3

    geo.plot()



.. image:: geo_files/geo_9_0.png




.. parsed-literal::

    <hypertools.datageometry.DataGeometry at 0x10fe62b70>



Exploring the geo
-----------------

Hypertools version
~~~~~~~~~~~~~~~~~~

The version field contains the version number of hypertools that the geo
was generated with.

.. code:: ipython3

    geo.version




.. parsed-literal::

    '0.4.2'



Data and plot
~~~~~~~~~~~~~

The data field contains the raw data.

.. code:: ipython3

    geo.data.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>cap-shape</th>
          <th>cap-surface</th>
          <th>cap-color</th>
          <th>bruises</th>
          <th>odor</th>
          <th>gill-attachment</th>
          <th>gill-spacing</th>
          <th>gill-size</th>
          <th>gill-color</th>
          <th>stalk-shape</th>
          <th>...</th>
          <th>stalk-surface-below-ring</th>
          <th>stalk-color-above-ring</th>
          <th>stalk-color-below-ring</th>
          <th>veil-type</th>
          <th>veil-color</th>
          <th>ring-number</th>
          <th>ring-type</th>
          <th>spore-print-color</th>
          <th>population</th>
          <th>habitat</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>x</td>
          <td>s</td>
          <td>n</td>
          <td>t</td>
          <td>p</td>
          <td>f</td>
          <td>c</td>
          <td>n</td>
          <td>k</td>
          <td>e</td>
          <td>...</td>
          <td>s</td>
          <td>w</td>
          <td>w</td>
          <td>p</td>
          <td>w</td>
          <td>o</td>
          <td>p</td>
          <td>k</td>
          <td>s</td>
          <td>u</td>
        </tr>
        <tr>
          <th>1</th>
          <td>x</td>
          <td>s</td>
          <td>y</td>
          <td>t</td>
          <td>a</td>
          <td>f</td>
          <td>c</td>
          <td>b</td>
          <td>k</td>
          <td>e</td>
          <td>...</td>
          <td>s</td>
          <td>w</td>
          <td>w</td>
          <td>p</td>
          <td>w</td>
          <td>o</td>
          <td>p</td>
          <td>n</td>
          <td>n</td>
          <td>g</td>
        </tr>
        <tr>
          <th>2</th>
          <td>b</td>
          <td>s</td>
          <td>w</td>
          <td>t</td>
          <td>l</td>
          <td>f</td>
          <td>c</td>
          <td>b</td>
          <td>n</td>
          <td>e</td>
          <td>...</td>
          <td>s</td>
          <td>w</td>
          <td>w</td>
          <td>p</td>
          <td>w</td>
          <td>o</td>
          <td>p</td>
          <td>n</td>
          <td>n</td>
          <td>m</td>
        </tr>
        <tr>
          <th>3</th>
          <td>x</td>
          <td>y</td>
          <td>w</td>
          <td>t</td>
          <td>p</td>
          <td>f</td>
          <td>c</td>
          <td>n</td>
          <td>n</td>
          <td>e</td>
          <td>...</td>
          <td>s</td>
          <td>w</td>
          <td>w</td>
          <td>p</td>
          <td>w</td>
          <td>o</td>
          <td>p</td>
          <td>k</td>
          <td>s</td>
          <td>u</td>
        </tr>
        <tr>
          <th>4</th>
          <td>x</td>
          <td>s</td>
          <td>g</td>
          <td>f</td>
          <td>n</td>
          <td>f</td>
          <td>w</td>
          <td>b</td>
          <td>k</td>
          <td>t</td>
          <td>...</td>
          <td>s</td>
          <td>w</td>
          <td>w</td>
          <td>p</td>
          <td>w</td>
          <td>o</td>
          <td>e</td>
          <td>n</td>
          <td>a</td>
          <td>g</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 22 columns</p>
    </div>



Transformed data
~~~~~~~~~~~~~~~~

The ``xform_data`` field contains the data that have been transformed
according to the user-specified normalize, reduce, and align options (in
this case, the data was automatically reduced to 3 dimensions for
plotting when we called ``hyp.plot``).

.. code:: ipython3

    geo.xform_data




.. parsed-literal::

    [array([[ 1.30051954,  0.29550253,  1.128519  ],
            [ 1.36797007, -0.2888117 ,  1.8208044 ],
            [ 1.59215354, -0.26831319,  1.66597598],
            ...,
            [ 0.40885079,  0.12732114,  0.41722945],
            [-0.89380986,  1.72804909, -0.56610268],
            [ 0.38867308,  0.10699673,  0.58651432]])]



Normalize, reduce and align metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``reduce``, ``align`` and ``normalize`` fields contain information
about the model and parameters used in each of the analyses.

The reduce and align fields contain dictionaries with the model
information and the normalize field contains a string.

.. code:: ipython3

    print(geo.normalize)
    print(geo.reduce)
    print(geo.align)


.. parsed-literal::

    None
    {'model': 'IncrementalPCA', 'params': {'n_components': 3}}
    {'model': None, 'params': {}}


Plotting axes and animations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``ax`` and ``line_ani`` fields hold the plot axes and the animation
setting (in this case None) for plotting, respectively.

To read more about the plot axes and line animation objects and their
utility, see the matlplotlib documentation for
`axes <http://matplotlib.org/api/axes_api.html>`__ and
`animations <http://matplotlib.org/api/animation_api.html>`__,
respectively.

.. code:: ipython3

    geo.ax




.. parsed-literal::

    <matplotlib.axes._subplots.Axes3DSubplot at 0x11aa08048>



.. code:: ipython3

    geo.line_ani

Plotting with geos
------------------

You can also generate a new plot (a new geo) from data stored in the geo
using geo.plot.

This plotting feature accepts all of the keyword arguments supported by
hypertools.plot.

First, let's plot without making any changes.

.. code:: ipython3

    geo.plot()



.. image:: geo_files/geo_29_0.png




.. parsed-literal::

    <hypertools.datageometry.DataGeometry at 0x11ae42400>



Now, let's change the plot using some keyword arguments.

In the example below, the data are re-transformed using all of the same
options as in the original plot, but with the number of dimensions
specified by the reduce model set to 2.

.. code:: ipython3

    geo.plot(ndims = 2)



.. image:: geo_files/geo_31_0.png




.. parsed-literal::

    <hypertools.datageometry.DataGeometry at 0x11afa4668>



Tranforming data using geos
---------------------------

An additional feature afforded by geos is the ability to later analyze
other datasets using the same transformations performed on the original
data in the geo. That is, whatever normalization, alignment, and
reduction parameters were used on the original data in the geo can be
quickly and easily applied to any new dataset using a single line of
code!

This allows for easy comparison of multiple datasets. Here, we load a
built in dataset (the weights dataset) and apply the transform from the
geo data to the first element of weights.

.. code:: ipython3

    weights = hyp.load('weights_avg').get_data()
    transformed = geo.transform(weights)

We can use heatmaps to visualized an element of the new data before and
after it has been transformed by the same means as the geo data.

.. code:: ipython3

    ax = sb.heatmap(weights[0])



.. image:: geo_files/geo_36_0.png


.. code:: ipython3

    ax = sb.heatmap(transformed[0])



.. image:: geo_files/geo_37_0.png


Saving geos
-----------

You can also easily save a geo using geo.save. The geo will save as a
'geo' file, which is a dictionary containing the elements of a data
geometry object saved in the hd5 format using deepdish.

To specify the compression type, pass a string to the compression
argument, as below. See the deepdish documentation for the full list of
compression options:
http://deepdish.readthedocs.io/en/latest/api\_io.html#deepdish.io.save

.. code:: ipython3

    # geo.save('MyGeoFile')
    # geo.save('MyGeoFile', compression = 'blosc')
