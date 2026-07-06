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

    <div class="sphx-glr-thumbcontainer" tooltip="The downside to using dimensionality reduction to visualize your data is that some variance will likely be removed. To help get a sense for the integrity of your low dimensional visualizations, we built the describe function, which computes the covariance (samples by samples) of both the raw and reduced datasets, and plots their correlation.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_describe_thumb.png
    :alt:

  :doc:`/auto_examples/plot_describe`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Using describe to evaluate the integrity of your visualization</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Timeseries plots can be animated by simply passing animate=True when calling hyp.plot.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we plot the trajectory of multivariate brain activity for two groups of subjects that have been hyperaligned (Haxby et al, 2011).  First, we use the align tool to project all subjects in the list to a common space. Then we average the data into two groups, and plot.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_align_thumb.png
    :alt:

  :doc:`/auto_examples/plot_align`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Aligning matrices to a common space</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to use the analyze function to process data prior to plotting. The data is a list of numpy arrays representing multi-voxel activity patterns (columns) over time (rows).  First, analyze function normalizes the columns of each matrix (within each matrix). Then the data is reduced using PCA (10 dims) and finally it is aligned with hyperalignment. We can then plot the data with hyp.plot, which further reduces it so that it can be visualized.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_analyze_thumb.png
    :alt:

  :doc:`/auto_examples/analyze`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Analyze data and then plot</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="When plotting, its useful to have a way to color points by some category or variable.  Hypertools does this using the hue kwarg, which takes a list of string category labels or numerical values.  If text labels are passed, the data is restructured according to those labels and plotted in different colors according to your color palette.  If numerical values are passed, the values are binned (default resolution: 100) and plotted according to your color palette.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_hue_thumb.png
    :alt:

  :doc:`/auto_examples/plot_hue`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Grouping data by category</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="hypertools 1.0 accepts arbitrarily nested lists of datasets. Every dataset under the same outermost group shares that group&#x27;s color, and each additional nesting level renders with thinner, fainter lines -- a summary-to-detail visual hierarchy. For example, [[a, b], [c]] colors a and b alike (group 1) and c differently (group 2).">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_nested_lists_thumb.png
    :alt:

  :doc:`/auto_examples/plot_nested_lists`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Nested lists and multilevel styling</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The colorbar kwarg draws a colorbar reflecting whatever color mapping is already in use (GH #100). For a continuous hue, the colorbar is a continuous gradient spanning the actual value range. For discrete groups (categorical hue, cluster/`n_clusters`, or a plain list of datasets), the colorbar is segmented into one block per group, labeled the same way the legend would be.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_colorbar_thumb.png
    :alt:

  :doc:`/auto_examples/plot_colorbar`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Colorbars</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="This is an example of how to use the label(s) kwarg, which must be a list the length of the number of datapoints (rows) you have in the matrix.  Here, we are simply labeling the first datapoint for each matrix in the list.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_labels_thumb.png
    :alt:

  :doc:`/auto_examples/plot_labels`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Labeling your datapoints</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="To plot text, simply pass the text data to the plot function.  By default, the text samples will be transformed into a vector of word counts and then modeled using Latent Dirichlet Allocation (# of topics = 100) using a model fit to a large sample of wikipedia pages.  If you specify semantic=None, the word count vectors will be plotted. To convert the text t0 a matrix (or list of matrices), we also expose the format_data function.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_text_thumb.png
    :alt:

  :doc:`/auto_examples/plot_text`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Plotting text</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="HyperTools 1.0 can render any plot with plotly instead of matplotlib by passing backend=&#x27;plotly&#x27; -- handy for rotating and zooming 3D plots interactively. With the default backend=&#x27;auto&#x27;, hypertools automatically uses plotly on Google Colab and Kaggle notebooks (where plotly is preinstalled and interactivity works best) and matplotlib everywhere else, so existing workflows are unchanged. Both backends produce the same styling: colors, line/marker sizes, format strings, and the signature cube frame.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="The density kwarg overlays a subtle KDE (kernel density estimate) &quot;glow&quot; behind the data: a 2D alpha-ramped heatmap, or a 3D volumetric cloud, showing where each dataset&#x27;s points are concentrated (GH #108, #191). Density shading is OFF by default (`density=None`) -- pass density=True for the defaults, or a dict to override alpha/`levels`/`grid`/`per_group`.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_density_thumb.png
    :alt:

  :doc:`/auto_examples/plot_density`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Density shading</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The surface kwarg overlays a smooth, lit surface over each dataset&#x27;s convex hull: a filled outline for 2D data, or a shaded, Taubin-smoothed 3D &quot;blob&quot; for 3D data (GH #109). Pass surface=True for sensible defaults, or a dict to customize the alpha, color, lighting, and amount of smoothing.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_surface_thumb.png
    :alt:

  :doc:`/auto_examples/plot_surface`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Surfaces around point clouds</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="chemtrails, precog, and bullettime each accept a per-dataset list of bools instead of a single bool, so different datasets in the same animation can show different trail styles (GH #127): a low-opacity trace of the past (chemtrails), of the future (precog), or of the entire timeseries at once (bullettime -- equivalent to chemtrails AND precog together).">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_animate_trails_mix_thumb.gif
    :alt:

  :doc:`/auto_examples/animate_trails_mix`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Mixing trail styles per dataset</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="In addition to hard clustering (KMeans, HDBSCAN, ...), hypertools 1.0 supports mixture models: GaussianMixture, BayesianGaussianMixture, LatentDirichletAllocation, and NMF. hyp.cluster returns an (n_samples, n_components) matrix of membership proportions instead of discrete labels, and hyp.plot colors each observation by blending the component colors according to its mixture weights -- observations between clusters render with intermediate colors.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_mixture_models_thumb.png
    :alt:

  :doc:`/auto_examples/plot_mixture_models`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Soft clustering with mixture models</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The &quot;Datasaurus Dozen&quot; (Matejka &amp; Fitzmaurice, 2017) is a set of 13 datasets that share nearly identical summary statistics (means, standard deviations, and correlations) but look wildly different when plotted.  hyp.load(&#x27;datasaurus&#x27;) returns the datasets as a list of pandas DataFrames; here we plot all thirteen side by side as 2D scatter plots of small black dots (the . point marker) to show why it always pays to visualize your data.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_datasaurus_thumb.png
    :alt:

  :doc:`/auto_examples/plot_datasaurus`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The Datasaurus Dozen</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="HyperTools ships with a &quot;shapes zoo&quot; of classic 3D point clouds (they download once and are then cached in /hypertools_data).  This example loads every shape in the zoo and displays each in its own panel, plotted as small black dots (the , pixel marker) by passing pre-created 3D axes to hyp.plot via the ax keyword.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_shapes_zoo_thumb.png
    :alt:

  :doc:`/auto_examples/plot_shapes_zoo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A zoo of 3D shapes</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="hyp.plot returns a plain matplotlib (or plotly) Figure -- there is no special container object to learn. Anything you can do with a Figure (``fig.savefig(...)``, grabbing fig.axes[0] to tweak the plot, embedding it in a larger layout, etc.) just works.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_geo_thumb.png
    :alt:

  :doc:`/auto_examples/plot_geo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Working with plot outputs (figures & fitted models)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="hyp.apply_model is hypertools 1.0&#x27;s unified model-application core: datasets are stacked, the model is fit ONCE across all of them, and the result is unstacked back to the input&#x27;s structure -- which is what makes embeddings and cluster assignments comparable across datasets. Models can be specified by name, as a dict with parameters, as a scikit-learn style instance, or as a list (pipeline). return_model=True hands back the fitted model for reuse on held-out data.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="The predict kwarg overlays a forecast on top of your plotted data: a dashed, same-color tail extending t steps past the end of each dataset. Under the hood this calls hypertools.predict, which supports several forecasting models -- &#x27;Kalman&#x27; (a linear-Gaussian state-space filter, used here), &#x27;GaussianProcess&#x27;, &#x27;AutoRegressor&#x27; (any sklearn regressor run recursively), &#x27;ARIMA&#x27;, &#x27;Laplace&#x27;, and &#x27;Chronos&#x27; (a HuggingFace time-series foundation model) -- selected via model= when calling hypertools.predict directly. Calling hyp.predict(data, model=..., t=..., return_model=True) also returns the fitted forecaster alongside the forecast, so the same fitted model can be reused (without re-estimating) on new data.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_predict_thumb.png
    :alt:

  :doc:`/auto_examples/plot_predict`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Forecasting timeseries with predict</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A DataFrame with a row MultiIndex (2 or more levels) is automatically expanded by hyp.plot into one &quot;leaf&quot; trace per unique index combination, plus one thicker, more opaque &quot;mean&quot; trace per level of grouping above the leaves (GH #95). Color is assigned by the top-level index value; leaves are thin and faint, and each successive level of averaging gets a thicker line and higher alpha, up to a fully opaque top-level mean that also carries the legend label.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_multiindex_thumb.png
    :alt:

  :doc:`/auto_examples/plot_multiindex`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MultiIndex DataFrames</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="HyperTools&#x27; &quot;shapes zoo&quot; (bunny, cube, dragon, sphere, teapot, vase, biplane -- see the A zoo of 3D shapes example) can be morphed smoothly from one point cloud to the next with the animate=&#x27;morph&#x27; hyp.plot style (PR #272, maintainer request 2026-07-06 -- see the animate/ rotations/`morph_samples` entries of the hyp.plot docstring for the full spec). Under the hood, an equal-sized sample of points is drawn from each shape, consecutive shapes are matched point-for-point with the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) so that each point travels the shortest total distance to its partner in the next shape, and the coordinates are eased between shapes frame by frame while the camera spins around the scene -- exactly the hand-rolled recipe this example used to implement itself before animate=&#x27;morph&#x27; existed, now built into the library behind a single hyp.plot call. rotations also accepts a per-segment list for finer camera control: below, holds spin a slow, easy-to-watch full rotation while each transition only spins a brisk quarter-turn, so the camera visibly &quot;steps&quot; forward every time one shape morphs into the next.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_shape_morph_thumb.gif
    :alt:

  :doc:`/auto_examples/plot_shape_morph`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Morphing through the shapes zoo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Hypertools fills missing (NaN) values via hypertools.impute before reducing/plotting. This compares two imputers on the weights_avg dataset after randomly knocking out 10% of its entries -- plus three CONSECUTIVE rows where every feature is missing. That fully-missing-row case is the motivating example for GH #169: PPCA reconstructs a row from its own observed features, so a row with NO observed features at all cannot be recovered, so PPCA warns and leaves those rows NaN (they are dropped below purely so the PPCA panel has something plottable). The Kalman imputer instead smooths across time, so it can fill a fully-missing row from the neighboring (observed) timepoints, at the cost of assuming the data are a reasonably smooth timeseries -- its panel keeps every row.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_impute_thumb.png
    :alt:

  :doc:`/auto_examples/plot_impute`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Imputing missing data: PPCA vs Kalman smoothing</div>
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


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Building on the Morphing through the shapes zoo example, HyperTools can also render a smooth, lit convex-hull SURFACE around a moving point cloud (the surface= hyp.plot kwarg -- see hypertools.plot.meshutil and hypertools.plot.surface, GH #109) instead of just the raw points. Combining surface= with animate=&#x27;morph&#x27; (PR #272, maintainer request 2026-07-06) recomputes the moving cloud&#x27;s smoothed hull mesh from scratch on every frame, shaded with a two-light Blinn-Phong model and backface-culled for the current camera angle -- so the &quot;blob&quot; skin flows continuously as the underlying points rearrange themselves, all from one hyp.plot call. Since a convex hull cannot reproduce concave features, holds on concave shapes like the bunny necessarily render as a smooth, rounded blob; that loss of concavity is an expected trade-off of the hull-surface approach, not a bug. Hulls hug the data tightly BY CONSTRUCTION (each smoothing round pulls stray vertices back onto the original hull surface, see hypertools.plot.meshutil.smooth_hull_3d) rather than via any fixed overshoot allowance, so the axes box never needs a hand-computed fudge factor to contain the surface. A final, bounded, grow-only rescale then guarantees at least 99% containment of the actual points; for ordinary, reasonably-sampled clouds this rescale rarely does more than nudge the mesh a few percent, and only grows large for very sparse clouds (rule of thumb: fewer than ~10 points), where a coarse, few-vertex hull loses proportionally more to smoothing and needs more correction to recover that same 99% guarantee.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_animate_surface_morph_thumb.png
    :alt:

  :doc:`/auto_examples/animate_surface_morph`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Morphing hull surfaces through shapes</div>
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
   /auto_examples/plot_describe
   /auto_examples/plot_legend
   /auto_examples/plot_digits
   /auto_examples/plot_TSNE
   /auto_examples/explore
   /auto_examples/plot_UMAP
   /auto_examples/chemtrails
   /auto_examples/animate_MDS
   /auto_examples/animate_spin
   /auto_examples/animate
   /auto_examples/precog
   /auto_examples/plot_align
   /auto_examples/plot_clusters3
   /auto_examples/analyze
   /auto_examples/plot_corpus
   /auto_examples/save_image
   /auto_examples/plot_hue
   /auto_examples/plot_nested_lists
   /auto_examples/plot_colorbar
   /auto_examples/plot_procrustes
   /auto_examples/plot_labels
   /auto_examples/save_movie
   /auto_examples/plot_text
   /auto_examples/plot_interactive_backend
   /auto_examples/plot_clusters2
   /auto_examples/animate_plotly
   /auto_examples/plot_multicolored_lines
   /auto_examples/plot_density
   /auto_examples/plot_surface
   /auto_examples/plot_PPCA
   /auto_examples/animate_trails_mix
   /auto_examples/plot_normalize
   /auto_examples/plot_mixture_models
   /auto_examples/plot_datasaurus
   /auto_examples/plot_shapes_zoo
   /auto_examples/plot_geo
   /auto_examples/plot_apply_model
   /auto_examples/plot_missing_data
   /auto_examples/plot_predict
   /auto_examples/plot_multiindex
   /auto_examples/plot_shape_morph
   /auto_examples/plot_impute
   /auto_examples/plot_sotus
   /auto_examples/animate_surface_morph


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_examples_python.zip </auto_examples/auto_examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_examples_jupyter.zip </auto_examples/auto_examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
