:orphan:

.. _tutorials:

How to use `HyperTools`
=======================

Plot
----------------

.. toctree::
   :maxdepth: 2

   tutorials/plot.ipynb

Analyze
----------------

.. toctree::
  :maxdepth: 2

  tutorials/analyze.ipynb

Normalize
----------------

.. toctree::
  :maxdepth: 2

  tutorials/normalize.ipynb

Reduce
----------------

.. toctree::
  :maxdepth: 2

  tutorials/reduce.ipynb

Align
----------------

.. toctree::
  :maxdepth: 2

  tutorials/align.ipynb

Cluster
----------------

.. toctree::
  :maxdepth: 2

  tutorials/cluster.ipynb

Plotting text
----------------

.. toctree::
  :maxdepth: 2

  tutorials/text.ipynb

Visualizing Hugging Face embeddings
-----------------------------------

.. toctree::
  :maxdepth: 2

  tutorials/hugging_face_embeddings.ipynb

Modern scikit-learn models and dynamics
---------------------------------------

.. toctree::
  :maxdepth: 2

  tutorials/modern_sklearn_dynamics.ipynb

Mapping Wikipedia with modern text embeddings
---------------------------------------------

.. toctree::
  :maxdepth: 2

  tutorials/wikipedia_embeddings.ipynb

Visualizing the shape of a conversation
---------------------------------------

.. toctree::
  :maxdepth: 2

  tutorials/conversation_trajectories.ipynb

Plotting streaming data
-----------------------

.. toctree::
  :maxdepth: 2

  tutorials/streaming_data.ipynb

Streaming from a Lab Streaming Layer (LSL) device
---------------------------------------------------

.. toctree::
  :maxdepth: 2

  tutorials/lsl_streaming.ipynb

Forecasting stock prices with hyp.predict
-----------------------------------------

.. toctree::
  :maxdepth: 2

  tutorials/stock_forecasting.ipynb

Imputing and forecasting a real projectile arc with hyp.impute and hyp.predict
------------------------------------------------------------------------------

.. toctree::
  :maxdepth: 2

  tutorials/projectile_kalman.ipynb

Story trajectories: brain activity while listening to a story
-------------------------------------------------------------

An animated cloud of hyperaligned trajectories showing how all 36 subjects'
whole-brain activity traces out a *shared* path through a low-dimensional space
while they listen to the same spoken story (fMRI data from Simony et al.,
2016). Each subject is preprocessed with a per-subject ``manip`` (Smooth →
Resample → ZScore), reduced to a low-dimensional ``reduce='IncrementalPCA'``
space, and **hyperaligned in that low-dimensional space** (``n_iter=10``); the
spinning animation then reveals the shared 3-D shape. Aligning in the reduced
space (not the full 100-hub space) with a linear reduction is what pulls the
trajectories together -- mean inter-subject correlation roughly doubles after
alignment. See the full gallery example,
:doc:`auto_examples/plot_story_trajectories`.

.. image:: _static/thumbnails/sphx_glr_plot_story_trajectories_thumb.gif
   :width: 400
   :alt: Animated hyperaligned brain-activity trajectories through a spoken story
