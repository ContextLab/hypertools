:orphan:

.. _examples-index:

Gallery of Examples
===================


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Here is a basic example where we load in some data (a list of arrays - samples by features), take the first two arrays in the list and plot them as points with the &#x27;o&#x27;.  Hypertools can handle all format strings supported by matplotlib.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_basic_thumb.png
    :alt:

  :doc:`/auto_examples/plot_basic`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A basic example</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A 2D plot can be created by setting ndims=2.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_2D_thumb.png
    :alt:

  :doc:`/auto_examples/plot_2D`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A 2D Plot</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The n_clusters kwarg can be used to discover clusters in your dataset.  It relies on scikit-learn&#x27;s implementation of k-mean clustering to find clusters, and then labels the points accordingly. You must set the number of clusters yourself.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_clusters_thumb.png
    :alt:

  :doc:`/auto_examples/plot_clusters`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Discovering clusters</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Hypertools supports single-index Pandas Dataframes as input. In this example, we plot the mushrooms dataset from the kaggle database.  This is a dataset of text features describing different attributes of a mushroom. Dataframes that contain columns with text are converted into binary feature vectors representing the presence or absences of the feature (see Pandas.Dataframe.get_dummies for more).">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_dataframe_thumb.png
    :alt:

  :doc:`/auto_examples/plot_dataframe`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Plotting a Pandas Dataframe</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An example of how to use the legend kwarg to generate a legend.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_legend_thumb.png
    :alt:

  :doc:`/auto_examples/plot_legend`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Generating a legend</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The downside to using dimensionality reduction to visualize your data is that some variance will likely be removed. To help get a sense for the integrity of your low dimensional visualizations, we built the describe function, which computes the covariance (samples by samples) of both the raw and reduced datasets, and plots their correlation.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_describe_thumb.png
    :alt:

  :doc:`/auto_examples/plot_describe`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Using describe to evaluate the integrity of your visualization</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example loads in some data from the scikit-learn digits dataset and plots it.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_digits_thumb.png
    :alt:

  :doc:`/auto_examples/plot_digits`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Visualizing the digits dataset</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example loads in some data from the scikit-learn digits dataset and plots it using t-SNE.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_TSNE_thumb.png
    :alt:

  :doc:`/auto_examples/plot_TSNE`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Visualizing the digits dataset using t-SNE</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Explore mode is an experimental feature that allows you to (not surprisingly) explore the points in your dataset.  When you hover over the points, a label will pop up that will help you identify the datapoint.  You can customize the labels by passing a list of labels to the label(s) kwarg. Alternatively, if you don&#x27;t pass a list of labels, the labels will be the index of the datapoint, along with the PCA coordinate.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_explore_thumb.png
    :alt:

  :doc:`/auto_examples/explore`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Explore mode!</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example loads in some data from the scikit-learn digits dataset and plots it using UMAP.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_UMAP_thumb.png
    :alt:

  :doc:`/auto_examples/plot_UMAP`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Visualizing the digits dataset using UMAP</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="To make use of HDBSCAN as the clustering algorithm used to discover clusters, you must specify it as the cluster argument. If you wish to specify HDBSCAN parameters you will need the dictionary form which includes both the model and the params. Since HDBSCAN does not require the number of clusters, n_clusters does not need to be set.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_clusters3_thumb.png
    :alt:

  :doc:`/auto_examples/plot_clusters3`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Discovering clusters using HDBSCAN</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we plot the trajectory of multivariate brain activity for two groups of subjects that have been hyperaligned (Haxby et al, 2011).  First, we use the align tool to project all subjects in the list to a common space. Then we average the data into two groups, and plot.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_align_thumb.png
    :alt:

  :doc:`/auto_examples/plot_align`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Aligning matrices to a common space</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The past trajectory of an animated plot can be visualized with the chemtrails argument.  This displays a low opacity version of the trace behind the current points being plotted.  This can be used in conjunction with the precog argument to plot a low-opacity trace of the entire timeseries.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_chemtrails_thumb.png
    :alt:

  :doc:`/auto_examples/chemtrails`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Chemtrails</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This is a trajectory of brain data plotted in 3D with multidimensional scaling.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_animate_MDS_thumb.png
    :alt:

  :doc:`/auto_examples/animate_MDS`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Animated trajectory plotted with multidimensional scaling</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In addition to plotting dynamic timeseries data, the spin feature can be used to visualize static data in an animated rotating plot.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_animate_spin_thumb.png
    :alt:

  :doc:`/auto_examples/animate_spin`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Create a rotating static plot</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Timeseries plots can be animated by simply passing animate=True to the geo ( or when calling hyp.plot).">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_animate_thumb.png
    :alt:

  :doc:`/auto_examples/animate`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Animated plots</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The future trajectory of an animated plot can be visualized with the precog argument.  This displays a low opacity version of the trace ahead of the current points being plotted.  This can be used in conjunction with the chemtrails argument to plot a low-opacity trace of the entire timeseries.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_precog_thumb.png
    :alt:

  :doc:`/auto_examples/precog`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Precognition</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="By default, the text samples will be transformed into a vector of word counts and then modeled using Latent Dirichlet Allocation (# of topics = 100) using a model fit to a large sample of wikipedia pages.  However, you can optionally pass your own text to fit the semantic model. To do this define corpus as a list of documents (strings). A topic model will be fit on the fly and the text will be plotted.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_corpus_thumb.png
    :alt:

  :doc:`/auto_examples/plot_corpus`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Defining a custom corpus for plotting text</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="To save a plot, simply use the save_path kwarg, and specify where you want the image to be saved, including the file extension (e.g. pdf)">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_save_image_thumb.png
    :alt:

  :doc:`/auto_examples/save_image`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Saving a plot</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to use the analyze function to process data prior to plotting. The data is a list of numpy arrays representing multi-voxel activity patterns (columns) over time (rows).  First, analyze function normalizes the columns of each matrix (within each matrix). Then the data is reduced using PCA (10 dims) and finally it is aligned with hyperalignment. We can then plot the data with hyp.plot, which further reduces it so that it can be visualized.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_analyze_thumb.png
    :alt:

  :doc:`/auto_examples/analyze`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Analyze data and then plot</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we load in some synthetic data, rotate it, and then use the procustes function to get the datasets back in alignment.  The procrustes function uses linear transformations to project a source matrix into the space of a target matrix.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_procrustes_thumb.png
    :alt:

  :doc:`/auto_examples/plot_procrustes`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Aligning two matrices with the procrustes function</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="When plotting, its useful to have a way to color points by some category or variable.  Hypertools does this using the hue kwarg, which takes a list of string category labels or numerical values.  If text labels are passed, the data is restructured according to those labels and plotted in different colors according to your color palette.  If numerical values are passed, the values are binned (default resolution: 100) and plotted according to your color palette.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_hue_thumb.png
    :alt:

  :doc:`/auto_examples/plot_hue`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Grouping data by category</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="hypertools 2.0 accepts arbitrarily nested lists of datasets. Every dataset under the same outermost group shares that group&#x27;s color, and each additional nesting level renders with thinner, fainter lines -- a summary-to-detail visual hierarchy. For example, [[a, b], [c]] colors a and b alike (group 1) and c differently (group 2).">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_nested_lists_thumb.png
    :alt:

  :doc:`/auto_examples/plot_nested_lists`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Nested lists and multilevel styling</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="HyperTools ships with several classic 3D &quot;shapes zoo&quot; point clouds (they download once and are then cached in /hypertools_data).  This example loads four of them and displays each in its own panel of a 2x2 figure by passing pre-created 3D axes to hyp.plot via the ax keyword.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_shapes_zoo_thumb.png
    :alt:

  :doc:`/auto_examples/plot_shapes_zoo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A zoo of 3D shapes</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This is an example of how to use the label(s) kwarg, which must be a list the length of the number of datapoints (rows) you have in the matrix.  Here, we are simply labeling the first datapoint for each matrix in the list.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_labels_thumb.png
    :alt:

  :doc:`/auto_examples/plot_labels`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Labeling your datapoints</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="To plot text, simply pass the text data to the plot function.  By default, the text samples will be transformed into a vector of word counts and then modeled using Latent Dirichlet Allocation (# of topics = 100) using a model fit to a large sample of wikipedia pages.  If you specify semantic=None, the word count vectors will be plotted. To convert the text t0 a matrix (or list of matrices), we also expose the format_data function.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_text_thumb.png
    :alt:

  :doc:`/auto_examples/plot_text`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Plotting text</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="HyperTools 2.0 can render any plot with plotly instead of matplotlib by passing backend=&#x27;plotly&#x27; -- handy for rotating and zooming 3D plots interactively. With the default backend=&#x27;auto&#x27;, hypertools automatically uses plotly on Google Colab and Kaggle notebooks (where plotly is preinstalled and interactivity works best) and matplotlib everywhere else, so existing workflows are unchanged. Both backends produce the same styling: colors, line/marker sizes, format strings, and the signature cube frame.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_interactive_backend_thumb.png
    :alt:

  :doc:`/auto_examples/plot_interactive_backend`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Interactive plotting with the plotly backend</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Here is an example where we generate some synthetic data, and then use the cluster function to get cluster labels, which we can then pass to the hue kwarg to color our points by cluster.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_clusters2_thumb.png
    :alt:

  :doc:`/auto_examples/plot_clusters2`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Using the cluster function to label clusters</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Animations work on the plotly backend too: animate=True reveals trajectories through a sliding time window and animate=&#x27;spin&#x27; rotates the camera, each with interactive play/pause controls in notebooks. Animations on either backend export to gif, animated png, or mp4 -- the file extension picks the format.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_animate_plotly_thumb.png
    :alt:

  :doc:`/auto_examples/animate_plotly`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Animated interactive plots (plotly backend)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Passing continuous values (or a matrix with one row per observation) as hue together with a line format string colors each trajectory continuously along its length -- for example, coloring a trajectory by time, by a behavioral variable, or by mixture proportions. Works on both the matplotlib and plotly backends.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_multicolored_lines_thumb.png
    :alt:

  :doc:`/auto_examples/plot_multicolored_lines`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Multicolored lines</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="When the plot function is called, it returns a DataGeometry object, or geo. A geo contains all the pieces needed to regenerate the plot. You can use the geo plot method to evaluate the same plot with new arguments, like changing the color of the points, or trying a different normalization method.  To save the plot, simply call geo.save(fname), where fname is a file name/path.  Then, this file can be reloaded using hyp.load to be plotted again at another time.  Finally, the transform method can be used to transform new data using the same transformations that were applied to the geo.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_geo_thumb.png
    :alt:

  :doc:`/auto_examples/plot_geo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A DataGeometry object or "geo"</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="To save an animation, simply add the save_path kwarg and specify the path where you want to save the movie, including the extension.  NOTE: this depends on having ffmpeg installed on your computer.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_save_movie_thumb.png
    :alt:

  :doc:`/auto_examples/save_movie`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Saving an animation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The &quot;Datasaurus Dozen&quot; (Matejka &amp; Fitzmaurice, 2017) is a set of 13 datasets that share nearly identical summary statistics (means, standard deviations, and correlations) but look wildly different when plotted.  hyp.load(&#x27;datasaurus&#x27;) returns the datasets as a list of pandas DataFrames; here we plot a few of them side by side as 2D scatter plots to show why it always pays to visualize your data.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_datasaurus_thumb.png
    :alt:

  :doc:`/auto_examples/plot_datasaurus`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The Datasaurus Dozen</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="When you pass a matrix with with missing data, hypertools will attempt to fill in the values using probabalistic principal components analysis (PPCA). Here is an example where we generate some synthetic data, remove some of the values, and then use PPCA to interpolate those missing values. Then, we plot both the original and data with missing values together to see how it performed.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_PPCA_thumb.png
    :alt:

  :doc:`/auto_examples/plot_PPCA`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Interpolating missing data with probabalistic PCA</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Often times its useful to normalize (z-score) you features before plotting, so that they are on the same scale.  Otherwise, some features will be weighted more heavily than others when doing PCA, and that may or may not be what you want. The normalize kwarg can be passed to the plot function.  If normalize is set to &#x27;across&#x27;, the zscore will be computed for the column across all of the lists passed.  Conversely, if normalize is set to &#x27;within&#x27;, the z-score will be computed separately for each column in each list.  Finally, if normalize is set to &#x27;row&#x27;, each row of the matrix will be zscored.  Alternatively, you can use the normalize function found in tools (see the third example).">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_normalize_thumb.png
    :alt:

  :doc:`/auto_examples/plot_normalize`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Normalizing your features</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In addition to hard clustering (KMeans, HDBSCAN, ...), hypertools 2.0 supports mixture models: GaussianMixture, BayesianGaussianMixture, LatentDirichletAllocation, and NMF. hyp.cluster returns an (n_samples, n_components) matrix of membership proportions instead of discrete labels, and hyp.plot colors each observation by blending the component colors according to its mixture weights -- observations between clusters render with intermediate colors.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_mixture_models_thumb.png
    :alt:

  :doc:`/auto_examples/plot_mixture_models`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Soft clustering with mixture models</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="hyp.apply_model is hypertools 2.0&#x27;s unified model-application core: datasets are stacked, the model is fit ONCE across all of them, and the result is unstacked back to the input&#x27;s structure -- which is what makes embeddings and cluster assignments comparable across datasets. Models can be specified by name, as a dict with parameters, as a scikit-learn style instance, or as a list (pipeline). return_model=True hands back the fitted model for reuse on held-out data.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_apply_model_thumb.png
    :alt:

  :doc:`/auto_examples/plot_apply_model`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Applying models with apply_model</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="If you have data with missing values, Hypertools will try to interpolate them using PPCA.  To visualize how well its doing, you can use the missing_inds function and then highlight the values that were interpolated.  Here, we generated some synthetic data, removed some values, and then plotted the original data, data with missing values and highlighted the missing datapoints with stars.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_missing_data_thumb.png
    :alt:

  :doc:`/auto_examples/plot_missing_data`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Using the missing_inds function to label interpolated values</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to plot text data using hypertools. We create sample State of the Union address excerpts covering different political themes and visualize them in a reduced dimensional space. By default, hypertools  transforms the text data using a topic model to capture semantic relationships  between different speech segments.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_sotus_thumb.png
    :alt:

  :doc:`/auto_examples/plot_sotus`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Plotting State of the Union Addresses with Text Analysis</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/plot_basic
   /auto_examples/plot_2D
   /auto_examples/plot_clusters
   /auto_examples/plot_dataframe
   /auto_examples/plot_legend
   /auto_examples/plot_describe
   /auto_examples/plot_digits
   /auto_examples/plot_TSNE
   /auto_examples/explore
   /auto_examples/plot_UMAP
   /auto_examples/plot_clusters3
   /auto_examples/plot_align
   /auto_examples/chemtrails
   /auto_examples/animate_MDS
   /auto_examples/animate_spin
   /auto_examples/animate
   /auto_examples/precog
   /auto_examples/plot_corpus
   /auto_examples/save_image
   /auto_examples/analyze
   /auto_examples/plot_procrustes
   /auto_examples/plot_hue
   /auto_examples/plot_nested_lists
   /auto_examples/plot_shapes_zoo
   /auto_examples/plot_labels
   /auto_examples/plot_text
   /auto_examples/plot_interactive_backend
   /auto_examples/plot_clusters2
   /auto_examples/animate_plotly
   /auto_examples/plot_multicolored_lines
   /auto_examples/plot_geo
   /auto_examples/save_movie
   /auto_examples/plot_datasaurus
   /auto_examples/plot_PPCA
   /auto_examples/plot_normalize
   /auto_examples/plot_mixture_models
   /auto_examples/plot_apply_model
   /auto_examples/plot_missing_data
   /auto_examples/plot_sotus


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_examples_python.zip </auto_examples/auto_examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_examples_jupyter.zip </auto_examples/auto_examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
