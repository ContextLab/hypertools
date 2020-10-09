#!/usr/bin/env python
from __future__ import division
import warnings
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from .._shared.helpers import *
from .._shared.params import default_params
from ..tools.analyze import analyze
from ..tools.cluster import cluster as clusterer
from ..tools.reduce import reduce as reducer
from ..tools.format_data import format_data
from .draw import _draw
from .backend import manage_backend
from ..datageometry import DataGeometry


@manage_backend
def plot(x, fmt='-', marker=None, markers=None, linestyle=None, linestyles=None,
         color=None, colors=None, palette='hls', group=None, hue=None,
         labels=None, legend=None, title=None, size=None, elev=10, azim=-60,
         ndims=3, model=None, model_params=None, reduce='IncrementalPCA',
         cluster=None, align=None, normalize=None, n_clusters=None,
         save_path=None, animate=False, duration=30, tail_duration=2,
         rotations=2, zoom=1, chemtrails=False, precog=False, bullettime=False,
         frame_rate=50, interactive=False, explore=False,
         mpl_backend='auto', show=True, transform=None,
         vectorizer='CountVectorizer', semantic='LatentDirichletAllocation',
         corpus='wiki', ax=None):
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

    color(s) : str or list of str
        A list of marker types

    palette : str
        A matplotlib or seaborn color palette

    group : str/int/float or list
        A list of group labels. Length must match the number of rows in your
        dataset. If the data type is numerical, the values will be mapped to
        rgb values in the specified palette. If the data type is strings,
        the points will be labeled categorically. To label a subset of points,
        use None (i.e. ['a', None, 'b','a']).

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

    animate : bool, 'parallel' or 'spin'
        If True or 'parallel', plots the data as an animated trajectory, with
        each dataset plotted simultaneously. If 'spin', all the data is plotted
        at once but the camera spins around the plot (default: False).

    duration (animation only) : float
        Length of the animation in seconds (default: 30 seconds)

    tail_duration (animation only) : float
        Sets the length of the tail of the data (default: 2 seconds)

    rotations (animation only) : float
        Number of rotations around the box (default: 2)

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
        Frame rate for animation (default: 50)

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
        Set the matplotlib backend used to create interactive and
        animated plots.  May be 'auto' (default), 'disable', or a
        backend key accepted by matplotlib. If 'auto', hypertools will
        automatically determine the interactive backend to use
        (`hypertools.plot.backend.HYPERTOOLS_BACKEND`). If 'disable',
        disable experimental behavior and use the current global
        matplotlib backend (`matplotlib.get_backend()`). NOTES: This
        feature is experimental. For a list of interactive matplotlib
        backends, see `matplotlib.rcsetup.interactive_bk`. For a list of
        backends available in IPython, run `%matplotlib --list`. Set the
        `$HYPERTOOLS_BACKEND` environment variable or use
        `hypertools.set_interactive_backend()` to override the backend
        used by 'auto' in non-IPython environments. If animate=False
        and interactive=False, this argument has no effect.

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

    Returns
    ----------
    geo : hypertools.DataGeometry
        A new data geometry object

    """

    # warnings for deprecated API args
    if (model is not None) or (model_params is not None):
        warnings.warn('Model and model_params arguments will be deprecated. Please use \
                      reduce keyword argument. See docs for details: http://hypertools.readthedocs.io/en/latest/hypertools.plot.html#hypertools.plot')
        reduce = {}
        reduce['model'] = model
        reduce['params'] = model_params

    if group is not None:
        warnings.warn('Group will be deprecated. Please use '
                      'hue keyword argument. See docs for details: ' 'http://hypertools.readthedocs.io/en/latest/hypertools.plot.html#hypertools.plot')
        hue = group

    if ax is not None:
        if ndims>2:
            if ax.name!='3d':
                raise ValueError('If passing ax and the plot is 3D, ax must '
                                 'also be 3d')

    text_args = {
        'vectorizer' : vectorizer,
        'semantic' : semantic,
        'corpus' : corpus
    }

    # analyze the data
    if transform is None:
        raw = format_data(x, **text_args)
        xform = analyze(raw, ndims=ndims, normalize=normalize, reduce=reduce,
                    align=align, internal=True)
    else:
        xform = transform

    # Return data that has been normalized and possibly reduced and/or aligned
    xform_data = copy.copy(xform)

    # catch all matplotlib kwargs here to pass on
    mpl_kwargs = {}

    # handle color (to be passed onto matplotlib)
    if color is not None:
        mpl_kwargs['color'] = color
        if colors is not None:
            mpl_kwargs['color'] = colors
            warnings.warn('Both color and colors defined: color will be ignored \
                          in favor of colors.')

    # handle linestyle (to be passed onto matplotlib)
    if linestyle is not None:
        mpl_kwargs['linestyle'] = linestyle
        if linestyles is not None:
            mpl_kwargs['linestyle'] = linestyles
            warnings.warn('Both linestyle and linestyles defined: linestyle  \
                          will be ignored in favor of linestyles.')

    # handle marker (to be passed onto matplotlib)
    if marker is not None:
        mpl_kwargs['marker'] = marker
        if markers is not None:
            mpl_kwargs['marker'] = markers
            warnings.warn('Both marker and markers defined: marker will be \
                          ignored in favor of markers.')

    # reduce data to 3 dims for plotting, if ndims is None, return this
    if (ndims and ndims < 3):
        xform = reducer(xform, ndims=ndims, reduce=reduce, internal=True)
    else:
        xform = reducer(xform, ndims=3, reduce=reduce, internal=True)

    # find cluster and reshape if n_clusters
    if cluster is not None:
        if hue is not None:
            warnings.warn('cluster overrides hue, ignoring hue.')
        if isinstance(cluster, (six.string_types, six.binary_type)):
            model = cluster
            params = default_params(model)
        elif isinstance(cluster, dict):
            model = cluster['model']
            params = default_params(model, cluster['params'])
        else:
            raise ValueError('Invalid cluster model specified; should be'
                             ' string or dictionary!')

        if n_clusters is not None:
            if cluster in ('HDBSCAN',):
                warnings.warn('n_clusters is not a valid parameter for '
                              'HDBSCAN clustering and will be ignored.')
            else:
                params['n_clusters'] = n_clusters

        cluster_labels = clusterer(xform, cluster={'model': model,
                                               'params': params})
        xform, labels = reshape_data(xform, cluster_labels, labels)
        hue = cluster_labels

    elif n_clusters is not None:
        # If cluster was None default to KMeans
        cluster_labels = clusterer(xform, cluster='KMeans', n_clusters=n_clusters)
        xform, labels = reshape_data(xform, cluster_labels, labels)
        if hue is not None:
            warnings.warn('n_clusters overrides hue, ignoring hue.')

    # group data if there is a grouping var
    elif hue is not None:
        if color is not None:
            warnings.warn("Using group, color keyword will be ignored.")

        # if list of lists, unpack
        if any(isinstance(el, list) for el in hue):
            hue = list(itertools.chain(*hue))

        # if all of the elements are numbers, map them to colors
        if all(isinstance(el, int) or isinstance(el, float) for el in hue):
            hue = vals2bins(hue)
        elif all(isinstance(el, str) for el in hue):
            hue = group_by_category(hue)

        # reshape the data according to group
        if n_clusters is None:
            xform, labels = reshape_data(xform, hue, labels)
        # interpolate lines if they are grouped
        if is_line(fmt):
            xform = patch_lines(xform)

    # handle legend
    if legend is not None:
        if legend is False:
            legend = None
        elif legend is True and hue is not None:
            legend = [item for item in sorted(set(hue), key=list(hue).index)]
        elif legend is True and hue is None:
            legend = [i + 1 for i in range(len(xform))]

        mpl_kwargs['label'] = legend

    # interpolate if its a line plot
    if fmt is None or isinstance(fmt, six.string_types):
        if is_line(fmt):
            if xform[0].shape[0] > 1:
                xform = interp_array_list(xform, interp_val=frame_rate*duration/(xform[0].shape[0] - 1))
    elif type(fmt) is list:
        for idx, xi in enumerate(xform):
            if is_line(fmt[idx]):
                if xi.shape[0] > 1:
                    xform[idx] = interp_array_list(xi, interp_val=frame_rate*duration/(xi.shape[0] - 1))

    # handle explore flag
    if explore:
        assert xform[0].shape[1] is 3, "Explore mode is currently only supported for 3D plots."
        mpl_kwargs['picker']=True

    # center
    xform = center(xform)

    # scale
    xform = scale(xform)

    # handle palette with seaborn
    if isinstance(palette, np.bytes_):
        palette = palette.decode("utf-8")
    sns.set_palette(palette=palette, n_colors=len(xform))
    sns.set_style(style='whitegrid')

    # turn kwargs into a list
    kwargs_list = parse_kwargs(xform, mpl_kwargs)

    # handle format strings
    if fmt is not None:
        if type(fmt) is not list:
            draw_fmt = [fmt for i in xform]
        else:
            draw_fmt = fmt
    else:
        draw_fmt = ['-']*len(x)

    # convert all nans to zeros
    for i, xi in enumerate(xform):
        xform[i] = np.nan_to_num(xi)

    # draw the plot
    fig, ax, data, line_ani = _draw(xform, fmt=draw_fmt,
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
                            ax=ax)

    # tighten layout
    plt.tight_layout()

    # save
    if save_path is not None:
        if animate:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=frame_rate, bitrate=1800)
            line_ani.save(save_path, writer=writer)

        else:
            plt.savefig(save_path)

    # # show the plot
    # if show:
    #     plt.show()
    # else:
    #     # safely closes the plot so it doesn't pop up in another call to this function
    #     plt.close('all')

    # gather reduce params
    if isinstance(reduce, dict):
        reduce_dict = reduce
    else:
        reduce_dict = {
            'model' : reduce,
            'params' : {
                'n_components' : ndims
            },
        }

    # gather align params
    if isinstance(align, dict):
        align_dict = align
    else:
        align_dict = {
            'model' : align,
            'params' : {}
        }

    # gather all other kwargs
    kwargs = {
        'fmt' : fmt,
        'marker': marker,
        'markers' : markers,
        'linestyle' : linestyle,
        'linestyles' : linestyles,
        'color' : color,
        'colors' : colors,
        'palette' : palette,
        'hue' : hue,
        'ndims' : ndims,
        'labels' : labels,
        'legend' : legend,
        'title' : title,
        'animate' : animate,
        'duration' : duration,
        'tail_duration' : tail_duration,
        'rotations' : rotations,
        'zoom' : zoom,
        'chemtrails' : chemtrails,
        'precog' : precog,
        'bullettime' : bullettime,
        'frame_rate' : frame_rate,
        'elev' : elev,
        'azim' : azim,
        'explore' : explore,
        'n_clusters' : n_clusters,
        'size' : size
    }
    # turn lists into np arrays so that they don't turn into pickles when saved
    for kwarg in kwargs:
        if isinstance(kwargs[kwarg], list):
            try:
                kwargs[kwarg]=np.array(kwargs[kwarg])
            except:
                warnings.warn('Could not convert all list arguments to numpy '
                              'arrays.  If list is longer than 256 items, it '
                              'will automatically be pickled, which could '
                              'cause Python 2/3 compatibility issues for the '
                              'DataGeometry object.')

    return DataGeometry(fig=fig, ax=ax, data=x, xform_data=xform_data,
                        line_ani=line_ani, reduce=reduce_dict, align=align_dict,
                        normalize=normalize, semantic=semantic,
                        vectorizer=vectorizer, corpus=corpus, kwargs=kwargs)
