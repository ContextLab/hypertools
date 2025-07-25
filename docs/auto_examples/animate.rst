
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_examples/animate.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_examples_animate.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_animate.py:


=============================
Animated plots
=============================

Timeseries plots can be animated by simply passing `animate=True` to the geo (
or when calling hyp.plot).

.. GENERATED FROM PYTHON SOURCE LINES 10-22

.. code-block:: Python


    # Code source: Andrew Heusser
    # License: MIT

    # import
    import hypertools as hyp

    # load example data
    geo = hyp.load('weights_avg')

    # plot
    geo.plot(animate=True, legend=['first', 'second'])

**Estimated memory usage:**  0 MB


.. _sphx_glr_download_auto_examples_animate.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: animate.ipynb <animate.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: animate.py <animate.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: animate.zip <animate.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
