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

    <div class="sphx-glr-thumbcontainer" tooltip="Here is a basic example where we load in some data (a list of arrays - samples by features) and plot all three arrays as points using the &#x27;.&#x27; format string. Hypertools can handle all format strings supported by matplotlib.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="Hypertools supports single-index Pandas Dataframes as input. In this example, we plot the mushrooms dataset from the kaggle database.  This is a dataset of text features describing different attributes of a mushroom. Dataframes that contain columns with text are converted into binary feature vectors representing the presence or absences of the feature (see the top-level pandas.get_dummies function for more). Because the rows of this dataset have no meaningful order, we plot them as points (the &#x27;.&#x27; format string) rather than as a connected line.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_dataframe_thumb.png
    :alt:

  :doc:`/auto_examples/plot_dataframe`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Plotting a Pandas Dataframe</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="A DataFrame with a row MultiIndex (2 or more levels) is automatically expanded by hyp.plot into one &quot;leaf&quot; trace per unique index combination, plus one thicker, more opaque &quot;mean&quot; trace per level of grouping above the leaves. Color is assigned by the top-level index value; leaves are thin and faint, and each successive level of averaging gets a thicker line and higher alpha, up to a fully opaque top-level mean that also carries the legend label.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_multiindex_thumb.png
    :alt:

  :doc:`/auto_examples/plot_multiindex`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MultiIndex DataFrames</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="To save a plot, simply use the save_path kwarg, and specify where you want the image to be saved, including the file extension (e.g. pdf)">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_save_image_thumb.png
    :alt:

  :doc:`/auto_examples/save_image`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Saving a plot</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="HyperTools 1.0 can render any plot with plotly instead of matplotlib by passing backend=&#x27;plotly&#x27; -- handy for rotating and zooming 3D plots interactively. With the default backend=&#x27;auto&#x27;, hypertools automatically uses plotly on Google Colab and Kaggle notebooks (where plotly is preinstalled and interactivity works best) and matplotlib everywhere else, so existing workflows are unchanged. Both backends produce the same styling: colors, line/marker sizes, format strings, and the signature cube frame.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_interactive_backend_thumb.png
    :alt:

  :doc:`/auto_examples/plot_interactive_backend`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Interactive plotting with the plotly backend</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The hue kwarg colors points by a category or a variable. It accepts one label per row (or one list per dataset), either as strings or as numbers. String labels are treated as categories: the rows are regrouped by label and each group gets its own color from the palette, with a legend naming the groups. Numeric values are binned instead (100 bins by default) and colored along the palette as a gradient. Both figures below use the weights_sample brain-activity data: three subjects, each 300 timepoints of 100 features.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_hue_thumb.png
    :alt:

  :doc:`/auto_examples/plot_hue`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Grouping data by category</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="This is an example of how to use the labels= kwarg. Passed one entry per DATASET (rather than one per row), each dataset is annotated once, at the row named by label_anchor= -- here label_anchor=&#x27;first&#x27; (the default) labels the first datapoint of each matrix in the list.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_labels_thumb.png
    :alt:

  :doc:`/auto_examples/plot_labels`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Labeling your datapoints</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The colorbar kwarg draws a colorbar reflecting whatever color mapping is already in use. For a continuous hue, the colorbar is a continuous gradient spanning the actual value range. For discrete groups (categorical hue, cluster/`n_clusters`, or a plain list of datasets), the colorbar is segmented into one block per group, labeled the same way the legend would be.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_colorbar_thumb.png
    :alt:

  :doc:`/auto_examples/plot_colorbar`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Colorbars</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="The density kwarg overlays a subtle KDE (kernel density estimate) &quot;glow&quot; behind the data: a 2D alpha-ramped heatmap, or a 3D volumetric cloud, showing where each dataset&#x27;s points are concentrated. Density shading is OFF by default (`density=None`) -- pass density=True for the defaults, or a dict to override alpha/`levels`/`grid`/`per_group`.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_density_thumb.png
    :alt:

  :doc:`/auto_examples/plot_density`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Density shading</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The surface kwarg overlays a smooth, lit surface over each dataset&#x27;s convex hull: a filled outline for 2D data, or a shaded, Taubin-smoothed 3D &quot;blob&quot; for 3D data. Pass surface=True for sensible defaults, or a dict to customize the alpha, color, lighting, and amount of smoothing.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_surface_thumb.png
    :alt:

  :doc:`/auto_examples/plot_surface`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Surfaces around point clouds</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scikit-learn digits dataset (``hyp.load(&#x27;digits&#x27;)``) holds 1797 8x8 grayscale images of handwritten digits, one per row of 64 pixel columns, plus a target column naming the digit. Restricting to the digits 0-5, the three figures below plot the same 64-dimensional pixel data with three different reducers, colored by digit: the default (PCA) projection in 3-D, then t-SNE and UMAP in 2-D. The nonlinear reducers pull each digit into a much tighter, better-separated cluster than the linear PCA view.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_digits_thumb.png
    :alt:

  :doc:`/auto_examples/plot_digits`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Visualizing the digits dataset</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="hyp.reduce supports six torch-backed autoencoder reducers: Autoencoder (shallow), SparseAutoencoder, DeepAutoencoder, ConvolutionalAutoencoder, SequenceAutoencoder, and VariationalAutoencoder. They are used exactly like any other reduce= model -- by name, with parameters passed via the dict spec -- and use the optional torch extra, which hypertools installs on demand the first time one is fit. This example fits a shallow Autoencoder and a VariationalAutoencoder on the same data and compares them against PCA: three 2-D embeddings of a noisy spiral manifold embedded in 10-D, with each point colored by its position along the spiral, so a reducer that unfolds the manifold shows a smooth color gradient. Passing reduce= a LIST gives one panel per reducer (`panels=True`), so all three embeddings are computed and drawn by a single hyp.plot call instead of three separate hyp.reduce calls plus a hand-built grid.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_autoencoders_thumb.png
    :alt:

  :doc:`/auto_examples/plot_autoencoders`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Autoencoder reducers</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The downside to using dimensionality reduction to visualize your data is that some variance will likely be removed. To help get a sense for the integrity of your low dimensional visualizations, we built the describe function. For each candidate number of dimensions, it reduces the data and correlates the pairwise Euclidean distances between observations in the reduced data with the pairwise distances in the raw (full-dimensional) data, then plots that correlation as a function of the number of dimensions.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_describe_thumb.png
    :alt:

  :doc:`/auto_examples/plot_describe`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Using describe to evaluate the integrity of your visualization</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The &quot;Datasaurus Dozen&quot; (Matejka &amp; Fitzmaurice, 2017) is a set of 13 datasets that share nearly identical summary statistics (means, standard deviations, and correlations) but look wildly different when plotted.  hyp.load(&#x27;datasaurus&#x27;) returns the datasets as a list of pandas DataFrames; here we plot all thirteen side by side, one panel per dataset (`panels=True`), as 2D scatter plots of small black dots (the . point marker) to show why it always pays to visualize your data.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_datasaurus_thumb.png
    :alt:

  :doc:`/auto_examples/plot_datasaurus`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The Datasaurus Dozen</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="hypertools.load resolves a plain string dataset name against five sources, in order: HyperTools&#x27; own built-in example datasets (used throughout the rest of this gallery), then four third-party sources: scikit-learn&#x27;s bundled datasets, seaborn&#x27;s example datasets, FiveThirtyEight&#x27;s published datasets (explicit &#x27;fivethirtyeight/&lt;slug&gt;&#x27; prefix), and Kaggle datasets (explicit &#x27;kaggle/&lt;owner&gt;/&lt;dataset&gt;&#x27; prefix, downloaded anonymously via kagglehub -- no Kaggle account or API key required). This example tours the four third-party sources, loading one small dataset from each and plotting it in a 2x2 grid: one panel per source.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_datasets_tour_thumb.png
    :alt:

  :doc:`/auto_examples/plot_datasets_tour`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A tour of hyp.load's data sources</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="HyperTools ships with a &quot;shapes zoo&quot; of classic 3D point clouds (they download once and are then cached in /hypertools_data).  This example loads every shape in the zoo and displays each in its own panel, plotted as small black dots (the , pixel marker), via one hyp.plot call with panels=.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_shapes_zoo_thumb.png
    :alt:

  :doc:`/auto_examples/plot_shapes_zoo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A zoo of 3D shapes</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we plot the trajectory of multivariate brain activity for two groups of subjects that have been hyperaligned (Haxby et al, 2011).  First, we use the align tool to project all subjects in the list to a common space. Then we average the data into two groups, and plot.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_align_thumb.png
    :alt:

  :doc:`/auto_examples/plot_align`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Aligning matrices to a common space</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The spiral dataset holds two copies of the same 3-D spiral, one of them rotated. Procrustes alignment finds the linear transformation (rotation, reflection, and scaling) that projects a source matrix onto a target matrix, so the two spirals land on top of each other. The first figure shows the two spirals as loaded; the second aligns them with model=&#x27;Procrustes&#x27; through hyp.align, and the third does the same thing inside a single hyp.plot call via its align kwarg.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_procrustes_thumb.png
    :alt:

  :doc:`/auto_examples/plot_procrustes`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Aligning two matrices with Procrustes</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="hyp.apply_model is hypertools 1.0&#x27;s unified model-application core: datasets are stacked, the model is fit ONCE across all of them, and the result is unstacked back to the input&#x27;s structure -- which is what makes embeddings and cluster assignments comparable across datasets. Models can be specified by name, as a dict with parameters, as a scikit-learn style instance, or as a list (pipeline). return_model=True hands back the fitted model for reuse on held-out data.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_apply_model_thumb.png
    :alt:

  :doc:`/auto_examples/plot_apply_model`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Applying models with apply_model</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Every stage dispatcher (`hyp.manip`, hyp.normalize, hyp.reduce, hyp.align, hyp.cluster, hyp.analyze, hyp.impute, hyp.predict) and hyp.plot accept return_model=True to get back the fitted model alongside the transformed result: a single fitted wrapper when only one stage ran, or a fitted hyp.Pipeline when multiple stages ran together (e.g. via the cross-module normalize=/``reduce=``/ align=/``cluster=`` kwargs). The fitted model/`Pipeline` can then be applied to held-out data via .transform() -- WITHOUT refitting -- so a train/test split, or streaming new data through an established projection, only ever fits once.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_pipelines_return_model_thumb.png
    :alt:

  :doc:`/auto_examples/plot_pipelines_return_model`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Fit once, reuse: pipelines and return_model</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Three ways to color a plot by cluster. The first figure passes n_clusters straight to hyp.plot, which runs k-means on the mushrooms dataset and colors each of the 10 discovered clusters. The second calls hyp.cluster directly on two synthetic blobs to get the labels, then hands them to hue, which is useful when the labels are needed for something else too. The third uses the dictionary form of cluster to run HDBSCAN, which chooses the number of clusters itself and can mark points as noise (label -1, colored as their own group). The mushrooms rows are unordered samples, so they are drawn as points (&#x27;.&#x27;) rather than a connected line.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_clusters_thumb.png
    :alt:

  :doc:`/auto_examples/plot_clusters`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Discovering clusters</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="When a dataset contains missing (NaN) entries, hyp.plot fills them in before reducing and plotting, using probabilistic principal components analysis (PPCA) by default. Here a random walk through 10 dimensions is generated, some of its entries are removed, and the original and imputed versions are plotted together. The first figure lets the imputation happen implicitly inside hyp.plot; the second uses hyp.tools.missing_inds to find the rows that contained missing values and marks them with stars, so you can see exactly which points were interpolated.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_missing_data_thumb.png
    :alt:

  :doc:`/auto_examples/plot_missing_data`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Plotting data with missing values</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Hypertools fills missing (NaN) values via hypertools.impute before reducing/plotting. This compares two imputers on the weights_avg dataset after randomly knocking out 10% of its entries -- plus three CONSECUTIVE rows where every feature is missing. That fully-missing-row case is the case that separates the two imputers: PPCA reconstructs a row from its own observed features, so a row with NO observed features at all cannot be recovered, so PPCA warns and leaves those rows NaN (they are dropped below purely so the PPCA panel has something plottable). The Kalman imputer instead smooths across time, so it can fill a fully-missing row from the neighboring (observed) timepoints, at the cost of assuming the data are a reasonably smooth timeseries -- its panel keeps every row.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_impute_thumb.png
    :alt:

  :doc:`/auto_examples/plot_impute`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Imputing missing data: PPCA vs Kalman smoothing</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The predict kwarg overlays a forecast on top of your plotted data: a dashed, same-color tail extending t steps past the end of each dataset. Under the hood this calls hypertools.predict, which supports several forecasting models -- &#x27;Kalman&#x27; (a linear-Gaussian state-space filter), &#x27;GaussianProcess&#x27; (used here), &#x27;AutoRegressor&#x27; (any sklearn regressor run recursively), &#x27;ARIMA&#x27;, &#x27;Laplace&#x27;, and &#x27;Chronos&#x27; (a HuggingFace time-series foundation model) -- selected via model= when calling hypertools.predict directly. Calling hyp.predict(data, model=..., t=..., return_model=True) also returns the fitted forecaster alongside the forecast, so the same fitted model can be reused (without re-estimating) on new data.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_predict_thumb.png
    :alt:

  :doc:`/auto_examples/plot_predict`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Forecasting timeseries with predict</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="To plot text, simply pass the text data to the plot function.  By default, the text samples will be transformed into a vector of word counts and then modeled using Latent Dirichlet Allocation (# of topics = 50) using a model fit to a large sample of wikipedia pages.  If you specify semantic=None, the word count vectors will be plotted. To convert the text to a matrix (or list of matrices), we also expose the format_data function. Note: the wikipedia topic model works best on sentence- or paragraph-length documents with common dictionary words; very short or slang-heavy snippets can land on nearly identical topic vectors.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_text_thumb.png
    :alt:

  :doc:`/auto_examples/plot_text`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Plotting text</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="By default, the text samples will be transformed into a vector of word counts and then modeled using Latent Dirichlet Allocation (# of topics = 50) using a model fit to a large sample of wikipedia pages.  However, you can optionally pass your own text to fit the semantic model. To do this define corpus as a list of documents (strings). A topic model will be fit on the fly and the text will be plotted.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_corpus_thumb.png
    :alt:

  :doc:`/auto_examples/plot_corpus`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Defining a custom corpus for plotting text</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="hyp.load(&#x27;sotus&#x27;) returns the full text of the 29 State of the Union addresses delivered between 1989 and 2017, grouped by president rather than sorted by date, so this example first puts them in date order. Passing the raw speech texts straight to hyp.plot runs hypertools&#x27; default text pipeline: each address is converted to a vector of word counts, modeled with a 50-topic Latent Dirichlet Allocation model fit to a large sample of wikipedia pages, and reduced to 3 dimensions. Because the addresses are plotted in chronological order, the connected line traces a &quot;text trajectory&quot; through semantic space: addresses that emphasize similar themes land near one another, and the trajectory shows how the topics presidents discuss have drifted over three decades.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_sotus_thumb.png
    :alt:

  :doc:`/auto_examples/plot_sotus`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Plotting State of the Union Addresses</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="hyp.plot accepts text directly. Its vectorizer=/``semantic=`` string specs resolve in three tiers: scikit-learn&#x27;s built-ins, then gensim&#x27;s models -- &#x27;Word2Vec&#x27;, &#x27;Doc2Vec&#x27;, &#x27;FastText&#x27; (vectorizer tier) and &#x27;LdaModel&#x27;, &#x27;LsiModel&#x27;, &#x27;HdpModel&#x27; (semantic tier) -- then HuggingFace sentence-transformers. gensim is an optional extra that hypertools installs on demand the first time a gensim model is requested. The two panels embed the same small three-topic corpus in two ways -- gensim&#x27;s Word2Vec (averaged word vectors, no semantic-stage model) on the left, and CountVectorizer counts fed to gensim&#x27;s LDA on the right -- and color each document by its topic. For documents this short, LDA&#x27;s topic proportions are nearly one-hot, so documents that LDA assigns to the same topic land on (almost) the same point in the right-hand panel.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_gensim_text_thumb.png
    :alt:

  :doc:`/auto_examples/plot_gensim_text`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gensim text models</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Timeseries can be animated by passing animate=True to hyp.plot: each trajectory is drawn progressively while the camera rotates around the scene. The data here are two group-average brain-activity trajectories (``hyp.load(&#x27;weights_avg&#x27;)``). The first animation uses the default (PCA) reduction; the second reduces the same data with multidimensional scaling instead, which changes the shape of the path the trajectories trace out.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_animate_thumb.gif
    :alt:

  :doc:`/auto_examples/animate`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Animated plots</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In addition to plotting dynamic timeseries data, the spin feature can be used to visualize static data in an animated rotating plot.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_animate_spin_thumb.gif
    :alt:

  :doc:`/auto_examples/animate_spin`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Create a rotating static plot</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Animated plots can show more than the current position along each trajectory. chemtrails=True leaves a low-opacity trace of the path already traveled behind the moving points; precog=True draws a low-opacity trace of the path still to come ahead of them. Combining both (or passing bullettime=True) shows the entire timeseries at low opacity with the current segment highlighted. The data are two group-average brain-activity trajectories (``hyp.load(&#x27;weights_avg&#x27;)``). See Mixing trail styles per dataset for choosing a different style for each dataset in the same animation.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_animate_trails_thumb.gif
    :alt:

  :doc:`/auto_examples/animate_trails`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Trails: chemtrails and precognition</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="chemtrails, precog, and bullettime each accept a per-dataset list of bools instead of a single bool, so different datasets in the same animation can show different trail styles: a low-opacity trace of the past (chemtrails), of the future (precog), or of the entire timeseries at once (bullettime -- equivalent to chemtrails AND precog together).">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_animate_trails_mix_thumb.gif
    :alt:

  :doc:`/auto_examples/animate_trails_mix`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Mixing trail styles per dataset</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="To save an animation, add the save_path kwarg with the path (and file extension) you want. Saving to .mp4 (or .mov/``.avi``) uses matplotlib&#x27;s ffmpeg writer, so ffmpeg must be installed and on your PATH for those formats; .gif and animated .png exports are written with Pillow and need no external tools. The data are the 36 hyperaligned subjects of the weights dataset, averaged into two groups of 18, so the movie shows two group-average trajectories through the same story.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_save_movie_thumb.gif
    :alt:

  :doc:`/auto_examples/save_movie`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Saving an animation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Animations work on the plotly backend too: animate=True reveals trajectories through a sliding time window and animate=&#x27;spin&#x27; rotates the camera, each with interactive play/pause controls in notebooks. Animations export via the save_path kwarg -- the file extension picks the format. On the plotly backend, interactive html exports are fast (no rasterization, no extra dependencies). Rasterized exports (`.gif`, animated png, mp4) are also supported, but they render every frame through kaleido/Chromium at roughly a few seconds per frame -- keep duration and frame_rate small for those, or use the matplotlib backend for long rasterized animations.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_animate_plotly_thumb.png
    :alt:

  :doc:`/auto_examples/animate_plotly`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Animated interactive plots (plotly backend)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An animated cloud of hyperaligned trajectories showing how 36 subjects&#x27; whole-brain activity traces out a shared path through a low-dimensional space while they listen to the same spoken story.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_story_trajectories_thumb.gif
    :alt:

  :doc:`/auto_examples/plot_story_trajectories`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Story trajectories: brain activity while listening to a story</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="One hyp.plot(..., animate=&#x27;morph&#x27;) call smoothly morphs a cloud of black dots from one shape to the next, holding on each shape before flowing into the following one. HyperTools ships a &quot;shapes zoo&quot; of seven classic 3-D point clouds (``bunny``, cube, dragon, sphere, teapot, vase and biplane), each downloaded once and then cached in ~/hypertools_data -- so this example is fully offline and deterministic after the first run. On a cold cache with no network it says so and morphs five parametric stand-ins instead, so it always renders.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_animate_morph_zoo_thumb.gif
    :alt:

  :doc:`/auto_examples/animate_morph_zoo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Morphing through the shapes zoo, with titles</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Building on the Morphing through the shapes zoo, with titles example, this one wraps the moving point cloud in a smooth, lit convex-hull SURFACE: the surface= kwarg of hyp.plot combined with animate=&#x27;morph&#x27;. The hull mesh is recomputed from the traveling cloud on every frame, shaded with a two-light Blinn-Phong model and backface-culled for the current camera angle, so a blue-teal &quot;skin&quot; flows continuously from the bunny to the cube, the sphere, the teapot and the vase as the points underneath rearrange themselves -- all from one hyp.plot call. alpha=0.25 keeps the points visible as a faint black texture under the surface.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_animate_surface_morph_thumb.gif
    :alt:

  :doc:`/auto_examples/animate_surface_morph`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Morphing hull surfaces through shapes</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A conversation as geometry, in one hyp.plot call on raw dialogue. Each turn (a contiguous run of speech by one speaker) is cut into sliding word windows, and the turns are handed to hyp.plot as a list of lists of strings: the nesting is the grouping, so every turn becomes its own disjoint trajectory through one shared 3-D space. The call embeds every window with a sentence-transformer (``vectorizer=&#x27;all-MiniLM-L6-v2&#x27;``), reduces them together with UMAP, colors each path by speaker through a categorical hue= with a native legend, and order=&#x27;serial&#x27; reveals the turns one at a time with chemtrails=True leaving the spoken path behind each head. title= carries one string per turn -- just the words being spoken, wrapped onto two lines when a turn is long so nothing runs off the figure -- and the library&#x27;s own reveal schedule advances it; the title never names the speaker, because the color does: title_color= takes one color per turn and the library retints the title with the current speaker&#x27;s color every frame, while title_kwargs={&#x27;size&#x27;: ...} sets its size (and the library now reserves the animated 3-D title&#x27;s own vertical margin, sized to the tallest wrapped title, so the figure never has to grow by hand); the legend maps colors to names.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_animate_conversation_thumb.gif
    :alt:

  :doc:`/auto_examples/animate_conversation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The shape of a conversation: one path per turn, colored by its speaker</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Text becomes geometry, tinted by the art itself. A full paragraph describing each of five famous paintings is cut into overlapping word windows and handed to hyp.plot as text -- a list of five lists of strings. One call embeds every window with a sentence-transformer (``vectorizer=&#x27;all-MiniLM-L6-v2&#x27;``), reduces all of them together into one shared 3-D space with UMAP, keeps the five clouds separate (the nesting of the input is the grouping), spins the camera, and annotates each cloud with its painting&#x27;s name.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_animate_painting_embeddings_thumb.gif
    :alt:

  :doc:`/auto_examples/animate_painting_embeddings`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Five paintings, described in words, drawn in their own colors</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Twenty-seven large-cap stocks in six sectors, every month since 2000, drawn as seven paths through one shared 3-D space. Each sector is handed to the library as its own matrix -- months down the rows, that sector&#x27;s stocks across the columns (four or five of them; the counts differ on purpose) -- and each cell is the stock&#x27;s cumulative log return since the first month (a growth curve). Three library calls turn that into the figure:">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_animate_market_sectors_thumb.gif
    :alt:

  :doc:`/auto_examples/animate_market_sectors`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A quarter century of the market: six sectors, one space</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The figure from the HyperTools paper, in one library call, with two companion panels that read off the same clock. Monthly mean temperatures for twenty cities spread across both hemispheres (Bangkok to Montreal, Sydney to Moscow) are treated not as twenty separate series but as twenty features of one measurement: each month is a single 20-dimensional observation of &quot;what the world&#x27;s weather was doing&quot;, and hyp.plot reduces that stream to a 3-D path.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_animate_weather_decades_thumb.gif
    :alt:

  :doc:`/auto_examples/animate_weather_decades`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A century of weather: twenty cities as twenty features, one hot path</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An animated forecast: hyp.plot(..., animate=True, predict=&#x27;Kalman&#x27;) refits the forecaster on the history revealed so far and draws it from the endpoint of the current frame, so the prediction grows and bends with the animation instead of standing still. forecast_trail=True keeps the earlier forecasts on screen as a fading fan, so you can watch the prediction change as history accumulates.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_animate_forecast_thumb.gif
    :alt:

  :doc:`/auto_examples/animate_forecast`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Forecasting three regions' weather while it is drawn</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/plot_basic
   /auto_examples/plot_2D
   /auto_examples/plot_dataframe
   /auto_examples/plot_nested_lists
   /auto_examples/plot_multiindex
   /auto_examples/plot_geo
   /auto_examples/save_image
   /auto_examples/explore
   /auto_examples/plot_interactive_backend
   /auto_examples/plot_hue
   /auto_examples/plot_legend
   /auto_examples/plot_labels
   /auto_examples/plot_colorbar
   /auto_examples/plot_multicolored_lines
   /auto_examples/plot_density
   /auto_examples/plot_surface
   /auto_examples/plot_digits
   /auto_examples/plot_autoencoders
   /auto_examples/plot_describe
   /auto_examples/plot_datasaurus
   /auto_examples/plot_datasets_tour
   /auto_examples/plot_shapes_zoo
   /auto_examples/plot_normalize
   /auto_examples/plot_align
   /auto_examples/plot_procrustes
   /auto_examples/analyze
   /auto_examples/plot_apply_model
   /auto_examples/plot_pipelines_return_model
   /auto_examples/plot_clusters
   /auto_examples/plot_mixture_models
   /auto_examples/plot_missing_data
   /auto_examples/plot_impute
   /auto_examples/plot_predict
   /auto_examples/plot_text
   /auto_examples/plot_corpus
   /auto_examples/plot_sotus
   /auto_examples/plot_gensim_text
   /auto_examples/animate
   /auto_examples/animate_spin
   /auto_examples/animate_trails
   /auto_examples/animate_trails_mix
   /auto_examples/save_movie
   /auto_examples/animate_plotly
   /auto_examples/plot_story_trajectories
   /auto_examples/animate_morph_zoo
   /auto_examples/animate_surface_morph
   /auto_examples/animate_conversation
   /auto_examples/animate_painting_embeddings
   /auto_examples/animate_market_sectors
   /auto_examples/animate_weather_decades
   /auto_examples/animate_forecast


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_examples_python.zip </auto_examples/auto_examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_examples_jupyter.zip </auto_examples/auto_examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
