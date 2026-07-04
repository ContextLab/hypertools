#!/usr/bin/env python
import copy
import warnings
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from .._shared.helpers import *
from .._shared.params import default_params
from ..tools.analyze import analyze
from ..cluster.cluster import cluster as clusterer, mixture_models
from .colors import mat2colors, colors2groups
from ..reduce.reduce import reduce as reducer
from ..tools.format_data import format_data
from .matplotlib_backend import _draw
from .backend import manage_backend
from .plotly_backend import resolve_backend
from .animate import _save_animation, _SVGFrameCollector, _save_animated_svg


@manage_backend
def plot(
    x,
    fmt="-",
    marker=None,
    markers=None,
    markersize=None,
    linewidth=None,
    linestyle=None,
    linestyles=None,
    color=None,
    colors=None,
    palette="hls",
    hue=None,
    labels=None,
    legend=None,
    title=None,
    size=None,
    elev=10,
    azim=-60,
    ndims=3,
    reduce="IncrementalPCA",
    cluster=None,
    align=None,
    normalize=None,
    n_clusters=None,
    save_path=None,
    animate=False,
    duration=30,
    tail_duration=2,
    rotations=1,
    zoom=1,
    chemtrails=False,
    precog=False,
    bullettime=False,
    frame_rate=30,
    interactive=False,
    explore=False,
    backend="auto",
    mpl_backend="auto",
    show=True,
    transform=None,
    vectorizer="CountVectorizer",
    semantic="LatentDirichletAllocation",
    corpus="wiki",
    ax=None,
    frame_kwargs=None,
    stream_init=10000,
    stream_chunk=100,
    stream_max=None,
    stream_window=None,
    return_model=False,
):
    """
    Plots dimensionality reduced data and parses plot arguments

    Parameters
    ----------
    x : Numpy array, DataFrame, String, Geo or mixed list
        Data for the plot. The form should be samples (rows) by features (cols).

    fmt : str or list of strings
        A list of format strings.  All matplotlib format strings are supported.

    linestyle(s) : str or list of str
        A list of line styles

    marker(s) : str or list of str
        A list of marker types

    markersize : int or float
        Size of the markers in points (default: matplotlib's 6.0). Applies
        to both backends.

    linewidth : int or float
        Width of plotted lines in points (default: matplotlib's 1.5 for
        static plots, 1 for animations). Applies to both backends.

    color(s) : str or list of str
        A list of marker types

    palette : str
        A matplotlib or seaborn color palette

    hue : list or numpy array
        Values used to color the plot. Accepts categorical labels (one per
        observation; grouped and colored by category), continuous numeric
        values (mapped through the palette; combined with a line format
        this produces multicolored lines whose color varies continuously
        along each trajectory), or a 2D matrix with one row per observation
        (e.g. mixture proportions or model weights; colors are blended per
        observation). To label a subset of points categorically, use None
        entries (i.e. ['a', None, 'b', 'a']).

    labels : list
        A list of labels for each point. Must be dimensionality of data (x).
        If no label is wanted for a particular point, input None.

    legend : list or bool
        If set to True, legend is implicitly computed from data. Passing a
        list will add string labels to the legend (one for each list item).

    title : str
        A title for the plot

    size : list
        A list of [width, height] in inches to resize the figure

    normalize : str or False
        If set to 'across', the columns of the input data will be z-scored
        across lists (default). If set to 'within', the columns will be
        z-scored within each list that is passed. If set to 'row', each row of
        the input data will be z-scored. If set to False, the input data will
        be returned (default is False).

    reduce : str or dict
        Decomposition/manifold learning model to use.  Models supported: PCA,
        IncrementalPCA, SparsePCA, MiniBatchSparsePCA, KernelPCA, FastICA,
        FactorAnalysis, TruncatedSVD, DictionaryLearning, MiniBatchDictionaryLearning,
        TSNE, Isomap, SpectralEmbedding, LocallyLinearEmbedding, and MDS. Can be
        passed as a string, but for finer control of the model parameters, pass
        as a dictionary, e.g. reduce={'model' : 'PCA', 'params' : {'whiten' : True}}.
        See scikit-learn specific model docs for details on parameters supported
        for each model.

    ndims : int
        An `int` representing the number of dims to reduce the data x
        to. If ndims > 3, will plot in 3 dimensions but return the higher
        dimensional data. Default is None, which will plot data in 3
        dimensions and return the data with the same number of dimensions
        possibly normalized and/or aligned according to normalize/align
        kwargs.

    align : str or dict or False/None
        If str, either 'hyper' or 'SRM'.  If 'hyper', alignment algorithm will be
        hyperalignment. If 'SRM', alignment algorithm will be shared response
        model.  You can also pass a dictionary for finer control, where the 'model'
        key is a string that specifies the model and the params key is a dictionary
        of parameter values (default : 'hyper').

    cluster : str or dict or False/None
        If cluster is passed, HyperTools will perform clustering using the
        specified clustering clustering model. Supportted algorithms are:
        KMeans, MiniBatchKMeans, AgglomerativeClustering, Birch,
        FeatureAgglomeration, SpectralClustering and HDBSCAN (default: None).
        Can be passed as a string, but for finer control of the model
        parameters, pass as a dictionary, e.g.
        reduce={'model' : 'KMeans', 'params' : {'max_iter' : 100}}. See
        scikit-learn specific model docs for details on parameters supported for
        each model. If no parameters are specified in the string a default set
        of parameters will be used.

    n_clusters : int
        If n_clusters is passed, HyperTools will perform k-means clustering
        with the k parameter set to n_clusters. The resulting clusters will
        be plotted in different colors according to the color palette.

    save_path : str
        Path to save the image/movie. Must include the file extension in the
        save path (i.e. save_path='/path/to/file/image.png'). NOTE: If saving
        an animation, FFMPEG must be installed (this is a matplotlib req).
        FFMPEG can be easily installed on a mac via homebrew brew install
        ffmpeg or linux via apt-get apt-get install ffmpeg. If you don't
        have homebrew (mac only), you can install it like this:
        /usr/bin/ruby -e "$(curl -fsSL
        https://raw.githubusercontent.com/Homebrew/install/master/install)".

    animate : bool, 'parallel', 'spin' or 'serial'
        If True or 'parallel', plots the data as an animated trajectory, with
        each dataset plotted simultaneously. If 'spin', all the data is plotted
        at once but the camera spins around the plot. If 'serial', datasets
        appear ONE AT A TIME in list order: each grows point-by-point into
        place while all previous datasets stay fully drawn, and datasets are
        never connected to each other -- useful for e.g. conversation turns
        accumulating in a shared embedding space (default: False).

    backend : str
        Rendering backend: 'matplotlib' (the classic renderer),
        'plotly' (interactive; requires plotly -- install with
        `pip install hypertools[interactive]`), or 'auto' (default), which
        uses plotly on Google Colab / Kaggle notebooks where interactivity
        matters most and matplotlib everywhere else. With the plotly backend,
        the return value is a plotly Figure (any animation frames are
        embedded directly in it, so no separate animation object is
        returned).

    duration (animation only) : float
        Length of the animation in seconds (default: 30 seconds)

    tail_duration (animation only) : float
        Sets the length of the tail of the data (default: 2 seconds)

    rotations (animation only) : float
        Number of rotations around the box over the course of the
        animation (default: 1 -- with the default 30-second duration,
        one revolution every 30 seconds). Identical pacing on both
        backends.

    zoom (animation only) : float
        How far to zoom into the plot, positive numbers will zoom in (default: 0)

    chemtrails (animation only) : bool
        A low-opacity trail is left behind the trajectory (default: False).

    precog (animation only) : bool
        A low-opacity trail is plotted ahead of the trajectory (default: False).

    bullettime (animation only) : bool
        A low-opacity trail is plotted ahead and behind the trajectory
        (default: False).

    frame_rate (animation only) : int or float
        Frame rate for animation in frames per second (default: 30).
        Both backends generate exactly frame_rate * duration frames, so
        matplotlib and plotly animations play at identical speed,
        duration, and framerate.

    interactive : bool
        If True, display the plot using an interactive matplotlib
        backend. Useful for inspecting and manipulating static plots. If
        animate=True, an interactive backend is required and this
        argument has no effect (default: False).

    explore : bool
        Displays user defined labels will appear on hover. If no labels are
        passed, the point index and coordinate will be plotted. To use,
        set explore=True. Note: Explore mode is currently only supported
        for 3D static plots, and is an experimental feature (i.e it may not yet
        work properly).

    mpl_backend : str
        The matplotlib backend used to create interactive and animated
        plots.  May be 'auto' (default), 'disable', or a backend key
        accepted by matplotlib. If 'auto', hypertools will use a backend
        determined automatically based on your environment
        (`hypertools.plot.backend.HYPERTOOLS_BACKEND`). If 'disable',
        experimental backend-switching is disabled and the current global
        matplotlib backend (`matplotlib.get_backend()`) is used.
        Otherwise, try to use the backend specified. NOTES: *This
        feature is experimental*. For a list of interactive matplotlib
        backends, see `matplotlib.rcsetup.interactive_bk`. For a list of
        backends available in IPython, run `%matplotlib --list`. Set the
        `$HYPERTOOLS_BACKEND` environment variable or use
        `hypertools.set_interactive_backend()` to override the backend
        used by 'auto' in non-IPython environments. If `animate=False`
        and `interactive=False`, this argument has no effect. Within the
        `hypertools.set_interactive_backend(backend)` context manager,
        the value of `backend` is prioritized over this argument.

    show : bool
        If set to False, the figure will not be displayed, but the figure,
        axis and data objects will still be returned (default: True).

    transform : list of numpy arrays or None
        The transformed data, bypasses transformations if this is set
        (default : None).

    vectorizer : str, dict, class or class instance
        The vectorizer to use. Built-in options are 'CountVectorizer' or
        'TfidfVectorizer'. To change default parameters, set to a dictionary
        e.g. {'model' : 'CountVectorizer', 'params' : {'max_features' : 10}}. See
        http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text
        for details. You can also specify your own vectorizer model as a class,
        or class instance.  With either option, the class must have a
        fit_transform method (see here: http://scikit-learn.org/stable/data_transforms.html).
        If a class, pass any parameters as a dictionary to vectorizer_params. If
        a class instance, no parameters can be passed.

    semantic : str, dict, class or class instance
        Text model to use to transform text data. Built-in options are
        'LatentDirichletAllocation' or 'NMF' (default: LDA). To change default
        parameters, set to a dictionary e.g. {'model' : 'NMF', 'params' :
        {'n_components' : 10}}. See
        http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
        for details on the two model options. You can also specify your own
        text model as a class, or class instance.  With either option, the class
        must have a fit_transform method (see here:
        http://scikit-learn.org/stable/data_transforms.html).
        If a class, pass any parameters as a dictionary to text_params. If
        a class instance, no parameters can be passed.

    corpus : list (or list of lists) of text samples or 'wiki', 'nips', 'sotus'.
        Text to use to fit the semantic model (optional). If set to 'wiki', 'nips'
        or 'sotus' and the default semantic and vectorizer models are used, a
        pretrained model will be loaded which can save a lot of time.

    ax : matplotlib.Axes
        Axis handle to plot the figure

    frame_kwargs : dict
        Keyword arguments for styling the frame drawn around the plot.
        For 3D plots, the frame is a cube and `frame_kwargs` are
        forwarded to `mpl_toolkits.mplot3d.axes3d.Axes3D.plot_wireframe`.
        For 2D plots, the frame is a square and `frame_kwargs` are
        forwarded to `matplotlib.patches.Rectangle`.

    stream_init : int
        Streaming data only (iterators/generators and Hugging Face
        ``datasets.IterableDataset`` are detected automatically): number of
        initial samples used to estimate the normalization and reduction
        parameters (default: 10000). Those fitted models are then *applied*
        to all future samples, which are added to the plot dynamically.

    stream_chunk : int
        Streaming data only: number of new samples fetched from the stream
        per update (default: 100). Each fetched chunk is projected through
        the fitted models and rendered as one animation frame / live
        redraw, so this sets both the download batch size and the temporal
        resolution of the resulting animation.

    stream_max : int or None
        Streaming data only: stop streaming after this many samples.
        Default None streams continually until the stream is exhausted or
        the user interrupts (Ctrl-C); infinite streams render incoming
        data indefinitely, and any animation being saved via `save_path`
        is finalized whenever streaming stops (including on interrupt).

    stream_window : int or None
        Streaming data only: if set, only the most recent `stream_window`
        samples are displayed (comet style) while older samples scroll off;
        all consumed samples are still retained on the returned geometry.
        Default None displays the full accumulated trajectory.

    return_model : bool
        If True, return a dict bundle
        ``{'fig': ..., 'xform_data': ..., 'animation': ..., 'models': ...}``
        instead of the bare figure, where ``xform_data`` is the
        normalized/reduced/aligned data, ``animation`` is the
        ``matplotlib.animation.Animation`` handle (``None`` unless
        ``animate=True`` with the matplotlib backend), and ``models`` holds
        the reduce/align/cluster specs. Default False.

    Returns
    ----------
    fig : matplotlib.figure.Figure or plotly Figure
        The rendered figure. For animated matplotlib plots a
        ``(fig, animation)`` tuple is returned instead, so the caller can
        retain a reference to the ``matplotlib.animation.FuncAnimation``
        (required to keep the animation alive). When ``return_model=True``,
        a dict
        ``{'fig': ..., 'xform_data': ..., 'animation': ..., 'models': ...}``
        is returned (``animation`` included so the handle isn't dropped for
        animated plots).

    """

    # streaming inputs (issue #101): iterators/generators and Hugging Face
    # IterableDatasets are detected from the structure of the input -- no
    # flag needed. Models are fitted on the first `stream_init` samples and
    # every subsequent sample is projected through the fitted models and
    # added to the plot dynamically (fetched in chunks of `stream_chunk`),
    # continuing until the stream ends, `stream_max` samples have been
    # consumed, or the user interrupts.
    from ..io.streaming import is_stream, plot_stream
    if is_stream(x):
        return plot_stream(
            x, fmt, stream_init=stream_init, stream_chunk=stream_chunk,
            stream_max=stream_max, stream_window=stream_window,
            ndims=ndims, reduce=reduce,
            normalize=normalize, align=align, cluster=cluster,
            n_clusters=n_clusters, save_path=save_path, show=show,
            frame_rate=frame_rate, markersize=markersize,
            linewidth=linewidth, color=color, palette=palette, title=title,
            size=size, elev=elev, azim=azim, ax=ax)

    if ax is not None:
        if ndims > 2:
            if ax.name != "3d":
                raise ValueError(
                    "If passing ax and the plot is 3D, ax must " "also be 3d"
                )

    text_args = {"vectorizer": vectorizer, "semantic": semantic, "corpus": corpus}

    # nested lists (e.g. [[a, b], [c]]) are flattened into a flat list of
    # datasets while recording each leaf's outermost-group index and nesting
    # depth; these drive multilevel styling below (color by outer group,
    # thinner/fainter lines per deeper level)
    nested_groups = nested_depths = None
    if isinstance(x, list) and any(isinstance(el, list) for el in x) \
            and not all(isinstance(el, str) for el in x):
        x, nested_groups, nested_depths = _flatten_nested(x)

    # analyze the data
    if transform is None:
        raw = format_data(x, **text_args)
        xform = analyze(
            raw,
            ndims=ndims,
            normalize=normalize,
            reduce=reduce,
            align=align,
            internal=True,
        )
    else:
        xform = transform

    # Return data that has been normalized and possibly reduced and/or aligned
    xform_data = copy.copy(xform)

    # catch all matplotlib kwargs here to pass on
    mpl_kwargs = {}

    # handle color (to be passed onto matplotlib)
    if color is not None:
        mpl_kwargs["color"] = color
        if colors is not None:
            mpl_kwargs["color"] = colors
            warnings.warn(
                "Both color and colors defined: color will be ignored \
                          in favor of colors."
            )

    # handle linestyle (to be passed onto matplotlib)
    if linestyle is not None:
        mpl_kwargs["linestyle"] = linestyle
        if linestyles is not None:
            mpl_kwargs["linestyle"] = linestyles
            warnings.warn(
                "Both linestyle and linestyles defined: linestyle  \
                          will be ignored in favor of linestyles."
            )

    # handle marker (to be passed onto matplotlib)
    if marker is not None:
        mpl_kwargs["marker"] = marker
        if markers is not None:
            mpl_kwargs["marker"] = markers
            warnings.warn(
                "Both marker and markers defined: marker will be \
                          ignored in favor of markers."
            )

    # handle marker size (to be passed onto matplotlib/plotly)
    if markersize is not None:
        mpl_kwargs["markersize"] = markersize

    # handle line width (to be passed onto matplotlib/plotly)
    if linewidth is not None:
        mpl_kwargs["linewidth"] = linewidth

    # reduce data to 3 dims for plotting, if ndims is None, return this.
    # xform was already formatted (and possibly reduced to ndims) by analyze()
    # above, so skip re-running format_data/PPCA here; reduce() returns the
    # data unchanged when it is already at the target dimensionality.
    if ndims and ndims < 3:
        xform = reducer(xform, ndims=ndims, reduce=reduce, internal=True,
                        format_data=False)
    else:
        xform = reducer(xform, ndims=3, reduce=reduce, internal=True,
                        format_data=False)

    # per-point colors for multicolored lines (set by the hue branch below;
    # computed after interpolation). Dataset lengths are captured now so hue
    # values can be re-interpolated to match the interpolated trajectories.
    multicolor_hue = None
    pre_interp_lengths = [len(xi) for xi in xform]

    # find cluster and reshape if n_clusters
    if cluster is not None:
        if hue is not None:
            warnings.warn("cluster overrides hue, ignoring hue.")
        if isinstance(cluster, (str, bytes)):
            model = cluster
            params = default_params(model) or {}
        elif isinstance(cluster, dict):
            model = cluster["model"]
            model_key = model if isinstance(model, str) \
                else getattr(model, "__name__", str(model))
            params = default_params(model_key,
                                    cluster.get("params", {})) or {}
            if "n_clusters" in cluster and n_clusters is None:
                # top-level convenience:
                # cluster={'model': ..., 'n_clusters': k}
                n_clusters = cluster["n_clusters"]
        else:
            raise ValueError(
                "Invalid cluster model specified; should be" " string or dictionary!"
            )

        if n_clusters is not None:
            if _mixture_name(model) == "HDBSCAN":
                warnings.warn(
                    "n_clusters is not a valid parameter for "
                    "HDBSCAN clustering and will be ignored."
                )
            elif _mixture_name(model) in mixture_models:
                params["n_components"] = n_clusters
            else:
                params["n_clusters"] = n_clusters

        cluster_labels = clusterer(xform, cluster={"model": model, "params": params})

        if _mixture_name(model) in mixture_models:
            # soft assignments: color each observation by the proportion-
            # weighted blend of its components' colors
            if legend is True:
                warnings.warn(
                    "legend is not supported for mixture-model clustering "
                    "(observations have blended colors, not discrete "
                    "groups); ignoring legend."
                )
                legend = None
            if not animate:
                # exact per-point colors (rendered via collections/scatter)
                multicolor_hue = np.asarray(cluster_labels,
                                            dtype=np.float64)
                hue = None
            else:
                # animations render one trace per group: quantize the
                # blended colors into (near-)identical-color groups
                blended = mat2colors(cluster_labels, palette=palette)
                group_ids, group_colors = colors2groups(blended)
                xform, labels = reshape_data(xform, group_ids, labels)
                mpl_kwargs["color"] = [
                    group_colors[gid]
                    for gid in sorted(set(group_ids), key=group_ids.index)
                ]
                hue = group_ids
        else:
            xform, labels = reshape_data(xform, cluster_labels, labels)
            hue = cluster_labels

    elif n_clusters is not None:
        # If cluster was None default to KMeans
        cluster_labels = clusterer(xform, cluster="KMeans", n_clusters=n_clusters)
        xform, labels = reshape_data(xform, cluster_labels, labels)
        if hue is not None:
            warnings.warn("n_clusters overrides hue, ignoring hue.")

    # group data if there is a grouping var
    elif hue is not None:
        if color is not None:
            warnings.warn("Using group, color keyword will be ignored.")

        # classify the hue argument: per-observation numeric matrix
        # (mixture proportions, model weights, ...), continuous 1D values,
        # or discrete grouping labels
        n_obs = sum(len(xi) for xi in xform)
        try:
            hue_array = np.asarray(hue)
        except Exception:
            hue_array = None
        hue_is_matrix = (hue_array is not None and hue_array.ndim == 2
                         and np.issubdtype(hue_array.dtype, np.number)
                         and hue_array.shape[0] == n_obs)
        hue_is_continuous = (hue_array is not None and hue_array.ndim == 1
                             and np.issubdtype(hue_array.dtype, np.number)
                             and hue_array.shape[0] == n_obs)

        if (hue_is_matrix or hue_is_continuous) and not animate:
            # EXACT PER-POINT COLORS: color varies continuously across
            # observations. Datasets stay intact (no group reshape, which
            # would fragment lines and quantize marker colors); per-point
            # colors are computed after interpolation, below, and rendered
            # via collections (lines) or scatter (markers).
            multicolor_hue = np.asarray(hue_array, dtype=np.float64)
            if legend is True:
                warnings.warn("legend is not supported for continuous or "
                              "matrix-valued hue; ignoring legend.")
                legend = None
            hue = None

        elif hue_is_matrix:
            # markers (or animated) path: blend colors per observation,
            # then group observations with (near-)identical colors into
            # traces
            blended = mat2colors(hue_array, palette=palette)
            group_ids, group_colors = colors2groups(blended)
            mpl_kwargs["color"] = [
                group_colors[gid]
                for gid in sorted(set(group_ids), key=group_ids.index)
            ]
            if legend is True:
                warnings.warn("legend is not supported for matrix-valued "
                              "hue; ignoring legend.")
                legend = None
            hue = group_ids

        else:
            # if list of lists, unpack
            if any(isinstance(el, list) for el in hue):
                hue = list(itertools.chain(*hue))

            # if all of the elements are numbers, map them to colors
            if not isinstance(hue[0], tuple):
                if all(isinstance(el, (int, float, np.integer, np.floating))
                       and not isinstance(el, bool) for el in hue):
                    hue = vals2bins(hue)
                elif all(isinstance(el, str) for el in hue):
                    hue = group_by_category(hue)

        # reshape the data according to group
        if hue is not None:
            if n_clusters is None:
                xform, labels = reshape_data(xform, hue, labels)
            # interpolate lines if they are grouped
            if is_line(fmt):
                xform = patch_lines(xform)

    # multilevel styling for nested-list input: every leaf under the same
    # outermost group shares that group's color, and each additional nesting
    # level renders thinner and fainter (summary -> detail)
    elif nested_groups is not None and color is None and colors is None:
        import seaborn as sns
        n_outer = len(set(nested_groups))
        base_colors = sns.color_palette(palette, n_outer)
        mpl_kwargs["color"] = [base_colors[g] for g in nested_groups]
        min_depth = min(nested_depths)
        if any(d != min_depth for d in nested_depths):
            mpl_kwargs["linewidth"] = [
                max(0.5, 2.0 * (0.7 ** (d - min_depth))) for d in nested_depths
            ]
            mpl_kwargs["alpha"] = [
                max(0.3, 0.9 ** (d - min_depth)) for d in nested_depths
            ]

    # handle legend
    if legend is not None:
        if legend is False:
            legend = None
        elif legend is True and hue is not None:
            legend = [item for item in sorted(set(hue), key=list(hue).index)]
        elif legend is True and hue is None:
            legend = [i + 1 for i in range(len(xform))]

        mpl_kwargs["label"] = legend

    # interpolate if its a line plot
    pre_interp_point_counts = [xi.shape[0] for xi in xform]
    if fmt is None or isinstance(fmt, str):
        if is_line(fmt):
            if xform[0].shape[0] > 1:
                xform = interp_array_list(
                    xform, interp_val=frame_rate * duration / (xform[0].shape[0] - 1)
                )
    elif type(fmt) is list:
        for idx, xi in enumerate(xform):
            if is_line(fmt[idx]):
                if xi.shape[0] > 1:
                    # interp_array (singular): xi is one dataset. The
                    # historical interp_array_list call here treated the
                    # 2D array as a LIST of rows, silently replacing the
                    # dataset with a list of per-row interpolations (latent
                    # for years because a bug made is_line() always False)
                    xform[idx] = interp_array(
                        xi, interp_val=frame_rate * duration / (xi.shape[0] - 1)
                    )

    # interpolation adds points, so per-point labels must be re-mapped onto
    # the interpolated trajectories (each label lands at its original
    # point's new index; in-between points get None)
    post_interp_point_counts = [xi.shape[0] for xi in xform]
    if labels is not None and post_interp_point_counts != pre_interp_point_counts:
        labels = _expand_labels(labels, pre_interp_point_counts,
                                post_interp_point_counts)

    # compute per-point colors for multicolored lines now that trajectories
    # have been interpolated (hue values are re-interpolated to match)
    line_colors = None
    if multicolor_hue is not None:
        line_colors = _multicolor_line_colors(
            multicolor_hue, pre_interp_lengths, xform, palette)

    # handle explore flag
    if explore:
        assert (
            xform[0].shape[1] == 3
        ), "Explore mode is currently only supported for 3D plots."
        mpl_kwargs["picker"] = True

    # center
    xform = center(xform)

    # scale
    xform = scale(xform)

    # handle palette with seaborn
    import seaborn as sns
    if isinstance(palette, np.bytes_):
        palette = palette.decode("utf-8")

    # turn kwargs into a list
    kwargs_list = parse_kwargs(xform, mpl_kwargs)

    # handle format strings
    if fmt is not None:
        if type(fmt) is not list:
            draw_fmt = [fmt for i in xform]
        else:
            draw_fmt = fmt
    else:
        draw_fmt = ["-"] * len(x)

    # convert all nans to zeros
    for i, xi in enumerate(xform):
        xform[i] = np.nan_to_num(xi)

    # interactive (plotly) backend: render with plotly and skip the
    # matplotlib pipeline entirely. backend='auto' resolves to plotly only
    # on Colab/Kaggle (see hypertools.plot.plotly_backend for the policy).
    if resolve_backend(backend) == "plotly":
        from .plotly_backend import plotly_draw

        if "color" not in mpl_kwargs:
            import seaborn as sns_local
            mpl_kwargs = dict(mpl_kwargs)
            mpl_kwargs["color"] = sns_local.color_palette(
                palette, len(xform))
            kwargs_list = parse_kwargs(xform, mpl_kwargs)
        fig = plotly_draw(
            xform,
            fmt=draw_fmt,
            kwargs_list=kwargs_list,
            labels=labels,
            legend=legend,
            title=title,
            animate=animate,
            size=size,
            show=show,
            save_path=save_path,
            frame_rate=frame_rate,
            duration=duration,
            rotations=rotations,
            elev=elev,
            azim=azim,
            point_colors=line_colors,
            tail_duration=tail_duration,
            chemtrails=chemtrails,
            precog=precog,
            bullettime=bullettime,
            zoom=zoom,
        )
        ax = None
        data = xform
        line_ani = None
    else:
        # Apply the hypertools palette/style only for the duration of this
        # plot call. Previously sns.set_palette/sns.set_style mutated global
        # matplotlib rcParams as a side effect of plotting (GH issue #259);
        # rc_context restores the user's settings on exit. The figure's axes
        # and artists are created inside the context, so they keep the
        # hypertools styling.
        with plt.rc_context():
            sns.set_palette(palette=palette, n_colors=len(xform))
            sns.set_style(style="whitegrid")

            # draw the plot
            fig, ax, data, line_ani = _draw(
                xform,
                fmt=draw_fmt,
                kwargs_list=kwargs_list,
                labels=labels,
                legend=legend,
                title=title,
                animate=animate,
                duration=duration,
                tail_duration=tail_duration,
                rotations=rotations,
                zoom=zoom,
                chemtrails=chemtrails,
                precog=precog,
                bullettime=bullettime,
                frame_rate=frame_rate,
                elev=elev,
                azim=azim,
                explore=explore,
                show=show,
                size=size,
                ax=ax,
                frame_kwargs=frame_kwargs,
            )

            # exact per-point colors: swap the single-color artists for
            # per-segment-colored line collections or per-point-colored
            # scatter (the cube/square frame and axes from _draw are kept)
            if line_colors is not None:
                if is_line(fmt):
                    _apply_multicolor_lines(ax, xform, line_colors,
                                            kwargs_list)
                else:
                    _apply_multicolor_markers(ax, xform, line_colors,
                                              kwargs_list)

            # tighten layout (static plots only: animated axes are given
            # the full canvas so rotating zoomed cubes don't clip, and
            # tight_layout would shrink them back into subplot margins)
            if not animate:
                plt.tight_layout()
                # tight_layout reserves room for an outside (right-side)
                # legend on 2D axes but NOT on 3D axes, so a wide legend on a
                # 3D plot overflows and clips off the figure's right edge.
                # Pull the axes leftward until the legend fits within the
                # canvas, keeping it fully visible to the right of the plot.
                if legend is not None and ax is not None:
                    _fit_right_legend(fig, ax)

            # save
            if save_path is not None:
                if animate:
                    _save_animation(line_ani, save_path, frame_rate)

                else:
                    plt.savefig(save_path)

    # Return shape (Jeremy decision #2):
    #   - static (matplotlib or plotly): return the Figure alone
    #   - animated matplotlib: return (fig, line_ani) so the caller can keep
    #     a reference to the FuncAnimation (needed to keep it alive); ax is
    #     recoverable as fig.axes[0], so it needs no separate return slot
    #   - animated plotly: frames are embedded in the Figure, so return fig
    #   - return_model=True: return a dict bundle exposing the analyzed
    #     xform_data plus the reduce/align/cluster model specs
    if return_model:
        # gather reduce params (spec, not a fitted instance)
        if isinstance(reduce, dict):
            reduce_dict = reduce
        else:
            reduce_dict = {"model": reduce, "params": {"n_components": ndims}}
        # gather align params
        if isinstance(align, dict):
            align_dict = align
        else:
            align_dict = {"model": align, "params": {}}
        return {
            "fig": fig,
            "xform_data": xform_data,
            "animation": line_ani,
            "models": {
                "reduce": reduce_dict,
                "align": align_dict,
                "cluster": cluster,
            },
        }

    # only animated matplotlib plots set line_ani; plotly and static plots
    # leave it None
    if line_ani is not None:
        return fig, line_ani

    return fig


def _fit_right_legend(fig, ax, pad=0.02, max_iter=5):
    """Shrink the axes so a right-side (outside) legend stays within the
    figure.

    hypertools draws its legend to the RIGHT of the plot via
    ``ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))``.
    matplotlib's ``tight_layout`` reserves horizontal room for such a legend
    on 2D axes, but not on 3D (Axes3D) axes -- so a legend wider than the
    default right margin overflows and clips off the figure's right edge.
    Measure the rendered legend and pull the subplot's right edge leftward
    (via ``subplots_adjust``) until the legend's right edge sits inside the
    canvas, so it renders fully to the right of the plot on both 2D and 3D.
    """
    legend = ax.get_legend()
    if legend is None:
        return
    for _ in range(max_iter):
        fig.canvas.draw()
        try:
            lb = legend.get_window_extent().transformed(
                fig.transFigure.inverted())
        except Exception:
            return
        overflow = lb.x1 - (1.0 - pad)
        if overflow <= 1e-3:
            return
        pos = ax.get_position()
        new_right = pos.x1 - overflow - 0.005
        # never collapse the plot to nothing; stop if we hit the floor
        if new_right <= 0.15 or new_right >= pos.x1:
            return
        fig.subplots_adjust(right=new_right)


def _flatten_nested(x, _depth=1):
    """Flatten arbitrarily nested lists of datasets (arrays/DataFrames) into
    a flat list, recording each leaf's outermost-group index and nesting
    depth. Lists containing strings (text data) are returned un-flattened,
    since nested string lists denote text corpora, not grouped datasets."""
    if _contains_string(x):
        return x, None, None
    leaves, groups, depths = [], [], []
    for outer_idx, el in enumerate(x):
        for leaf, depth in _iter_leaves(el, _depth):
            leaves.append(leaf)
            groups.append(outer_idx)
            depths.append(depth)
    return leaves, groups, depths


def _iter_leaves(el, depth):
    if isinstance(el, list):
        for sub in el:
            yield from _iter_leaves(sub, depth + 1)
    else:
        yield el, depth


def _contains_string(el):
    if isinstance(el, str):
        return True
    if isinstance(el, list):
        return any(_contains_string(sub) for sub in el)
    return False


def _multicolor_line_colors(hue_src, orig_lengths, xform, palette):
    """Per-point RGB colors for multicolored lines.

    hue_src holds one value (or one row) per ORIGINAL observation; the
    trajectories in xform have since been interpolated to a higher temporal
    resolution, so each dataset's hue values are linearly re-interpolated to
    its new length before color mapping. Colors are mapped over the
    CONCATENATED hue values so the scale is shared across datasets.

    Returns a list of (n_i, 3) arrays, one per dataset in xform.
    """
    hue_src = np.asarray(hue_src, dtype=np.float64)
    if hue_src.ndim == 1:
        hue_src = hue_src[:, None]

    splits = np.cumsum(orig_lengths)[:-1]
    pieces = np.vsplit(hue_src, splits)

    interped = []
    for piece, xi in zip(pieces, xform):
        n_new = xi.shape[0]
        if n_new == piece.shape[0]:
            interped.append(piece)
            continue
        old_t = np.linspace(0.0, 1.0, piece.shape[0])
        new_t = np.linspace(0.0, 1.0, n_new)
        interped.append(np.column_stack(
            [np.interp(new_t, old_t, piece[:, c])
             for c in range(piece.shape[1])]))

    stacked = np.vstack(interped)
    colors = mat2colors(
        stacked.ravel() if stacked.shape[1] == 1 else stacked,
        palette=palette)

    out, start = [], 0
    for xi in xform:
        out.append(np.asarray(colors[start:start + xi.shape[0]]))
        start += xi.shape[0]
    return out


def _apply_multicolor_lines(ax, xform, line_colors, kwargs_list):
    """Replace single-color line artists with per-segment-colored
    collections (matplotlib backend)."""
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    for line in list(ax.lines):
        line.remove()

    is_3d = xform[0].shape[1] >= 3
    for i, (xi, ci) in enumerate(zip(xform, line_colors)):
        tkwargs = kwargs_list[i] if i < len(kwargs_list) else {}
        lw = tkwargs.get('linewidth') or plt.rcParams['lines.linewidth']
        if xi.shape[1] == 1:
            pts = np.column_stack([np.arange(xi.shape[0]), xi[:, 0]])
        else:
            pts = xi[:, :3] if is_3d else xi[:, :2]
        segments = np.stack([pts[:-1], pts[1:]], axis=1)
        seg_colors = (ci[:-1] + ci[1:]) / 2.0
        if is_3d:
            coll = Line3DCollection(segments, colors=seg_colors,
                                    linewidths=lw)
            ax.add_collection3d(coll)
        else:
            coll = LineCollection(segments, colors=seg_colors,
                                  linewidths=lw)
            ax.add_collection(coll)


def _expand_labels(labels, old_lengths, new_lengths):
    """Re-map per-point labels onto interpolated trajectories.

    Each original point's label is placed at that point's index in the
    interpolated (longer) trajectory; the interpolated in-between points get
    None (no annotation). Accepts flat label lists or lists nested per
    dataset; returns a flat list matching sum(new_lengths).
    """
    if any(isinstance(el, list) for el in labels):
        flat = list(itertools.chain(*labels))
    else:
        flat = list(labels)

    out = []
    start = 0
    for old_n, new_n in zip(old_lengths, new_lengths):
        piece = flat[start:start + old_n]
        start += old_n
        expanded = [None] * new_n
        for i, lab in enumerate(piece):
            if old_n == 1:
                j = 0
            else:
                j = min(new_n - 1, int(round(i * (new_n - 1) / (old_n - 1))))
            expanded[j] = lab
        out.extend(expanded)
    return out


def _apply_multicolor_markers(ax, xform, point_colors, kwargs_list):
    """Replace single-color marker artists with per-point-colored scatter
    (matplotlib backend). Gives exact per-observation colors -- e.g. mixture
    proportions render as true blends instead of quantized groups."""
    for line in list(ax.lines):
        line.remove()

    is_3d = xform[0].shape[1] >= 3
    for i, (xi, ci) in enumerate(zip(xform, point_colors)):
        tkwargs = kwargs_list[i] if i < len(kwargs_list) else {}
        ms = float(tkwargs.get('markersize')
                   or plt.rcParams['lines.markersize'])
        s = ms ** 2  # scatter sizes are areas in points^2
        if xi.shape[1] == 1:
            ax.scatter(np.arange(xi.shape[0]), xi[:, 0], c=ci, s=s)
        elif is_3d:
            ax.scatter(xi[:, 0], xi[:, 1], xi[:, 2], c=ci, s=s,
                       depthshade=False)
        else:
            ax.scatter(xi[:, 0], xi[:, 1], c=ci, s=s)

def _mixture_name(model):
    """Registry name for a cluster-model spec (string or class)."""
    return model if isinstance(model, str) \
        else getattr(model, "__name__", str(model))

