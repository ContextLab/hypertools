
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_examples/plot_clusters3.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_examples_plot_clusters3.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_clusters3.py:


=============================
Discovering clusters using HDBSCAN
=============================

To make use of HDBSCAN as the clustering algorithm used to discover clusters,
you must specify it as the cluster argument. If you wish to specify HDBSCAN
parameters you will need the dictionary form which includes both the model
and the params. Since HDBSCAN does not require the number of clusters,
n_clusters does not need to be set.

.. GENERATED FROM PYTHON SOURCE LINES 13-28



.. image-sg:: /auto_examples/images/sphx_glr_plot_clusters3_001.png
   :alt: plot clusters3
   :srcset: /auto_examples/images/sphx_glr_plot_clusters3_001.png
   :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    /Users/jmanning/hypertools/hypertools/plot/plot.py:369: UserWarning: cluster overrides hue, ignoring hue.
      warnings.warn("cluster overrides hue, ignoring hue.")
    /Users/jmanning/hypertools/hypertools/plot/plot.py:577: UserWarning: Could not convert all list arguments to numpy arrays.  If list is longer than 256 items, it will automatically be pickled, which could cause Python 2/3 compatibility issues for the DataGeometry object.
      warnings.warn(

    <hypertools.datageometry.DataGeometry object at 0x128f52c30>





|

.. code-block:: Python


    # Code source: Andrew Heusser and Leland McInnes
    # License: MIT

    # import
    import hypertools as hyp
    import pandas as pd

    # load example data
    geo = hyp.load('mushrooms')

    # plot
    geo.plot(cluster={'model':'HDBSCAN',
                                 'params': {'min_samples':5,
                                            'min_cluster_size':30}})


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 3.566 seconds)

**Estimated memory usage:**  378 MB


.. _sphx_glr_download_auto_examples_plot_clusters3.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_clusters3.ipynb <plot_clusters3.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_clusters3.py <plot_clusters3.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_clusters3.zip <plot_clusters3.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
