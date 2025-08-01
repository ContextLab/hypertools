
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_examples/plot_digits.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_examples_plot_digits.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_digits.py:


=============================
Visualizing the digits dataset
=============================

This example loads in some data from the scikit-learn digits dataset and plots
it.

.. GENERATED FROM PYTHON SOURCE LINES 10-25



.. image-sg:: /auto_examples/images/sphx_glr_plot_digits_001.png
   :alt: plot digits
   :srcset: /auto_examples/images/sphx_glr_plot_digits_001.png
   :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    /Users/jmanning/hypertools/hypertools/plot/plot.py:577: UserWarning: Could not convert all list arguments to numpy arrays.  If list is longer than 256 items, it will automatically be pickled, which could cause Python 2/3 compatibility issues for the DataGeometry object.
      warnings.warn(

    <hypertools.datageometry.DataGeometry object at 0x13220fb90>





|

.. code-block:: Python


    # Code source: Andrew Heusser
    # License: MIT

    # import
    from sklearn import datasets
    import hypertools as hyp

    # load example data
    digits = datasets.load_digits(n_class=6)
    data = digits.data
    hue = digits.target

    # plot
    hyp.plot(data, '.', hue=hue)


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 0.520 seconds)


.. _sphx_glr_download_auto_examples_plot_digits.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_digits.ipynb <plot_digits.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_digits.py <plot_digits.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_digits.zip <plot_digits.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
