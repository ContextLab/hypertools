import copy
import deepdish as dd
import numpy as np
from .tools.normalize import normalize as normalizer
from .tools.reduce import reduce as reducer
from .tools.align import align as aligner
from .tools.format_data import format_data
from ._shared.helpers import convert_text, get_dtype
from .config import __version__


class DataGeometry(object):
    """
    Hypertools data object class

    A DataGeometry object contains the data, figure handles and transform
    functions used to create a plot.  Note: this class should not be called
    directly, but is used by the `hyp.plot` function to create a plot object.

    Parameters
    ----------

    fig : matplotlib.Figure
        The matplotlib figure handle for the plot

    ax : matplotlib.Axes
        The matplotlib axes handle for the plot

    line_ani : matplotlib.animation.FuncAnimation
        The matplotlib animation handle (if the plot is an animation)

    data : list
        A list of numpy arrays representing the raw data

    xform_data : list
        A list of numpy arrays representing the transformed data

    reduce : dict
        A dictionary containing the reduction model and parameters

    align : dict
        A dictionary containing align model and parameters

    normalize : str
        A string representing the kind of normalization

    kwargs : dict
        A dictionary containing all kwargs passed to the plot function

    version : str
        The version of the software used to create the class instance

    """

    def __init__(self, fig=None, ax=None, line_ani=None, data=None, xform_data=None,
                 reduce=None, align=None, normalize=None, semantic=None,
                 vectorizer=None, corpus=None, kwargs=None, version=__version__,
                 dtype=None):

        # matplotlib figure handle
        self.fig = fig

        # matplotlib axis handle
        self.ax = ax

        # matplotlib line_ani handle (if its an animation)
        self.line_ani = line_ani

        # convert to numpy array if text
        if isinstance(data, list):
            data = list(map(convert_text, data))
        self.data = data
        self.dtype = get_dtype(data)

        # the transformed data
        self.xform_data = xform_data

        # dictionary of model and model_params
        self.reduce = reduce

        # 'hyper', 'SRM' or None
        self.align = align

        # 'within', 'across', 'row' or False
        self.normalize = normalize

        # text params
        self.semantic = semantic
        self.vectorizer = vectorizer

        self.corpus = corpus

        # dictionary of kwargs
        self.kwargs = kwargs

        # hypertools version
        self.version = version

    def get_data(self):
        """Return a copy of the data"""
        return copy.copy(self.data)

    def get_formatted_data(self):
        """Return a formatted copy of the data"""
        return format_data(self.data)

    # a function to transform new data
    def transform(self, data=None):
        """
        Return transformed data, or transform new data using the same model
        parameters

        Parameters
        ----------
        data : numpy array, pandas dataframe or list of arrays/dfs
            The data to transform.  If no data is passed, the xform_data from
            the DataGeometry object will be returned.

        Returns
        ----------
        xformed_data : list of numpy arrays
            The transformed data

        """
        # if no new data passed,
        if data is None:
            return self.xform_data
        else:
            formatted = format_data(
                data,
                semantic=self.semantic,
                vectorizer=self.vectorizer,
                corpus=self.corpus,
                ppca=True)
            norm = normalizer(formatted, normalize=self.normalize)
            reduction = reducer(
                norm,
                reduce=self.reduce,
                ndims=self.reduce['params']['n_components'])
            return aligner(reduction, align=self.align)

    # a function to plot the data
    def plot(self, data=None, **kwargs):
        """
        Plot the data

        Parameters
        ----------
        data : numpy array, pandas dataframe or list of arrays/dfs
            The data to plot.  If no data is passed, the xform_data from
            the DataGeometry object will be returned.

        kwargs : keyword arguments
            Any keyword arguments supported by `hypertools.plot` are also supported
            by this method

        Returns
        ----------
        geo : hypertools.DataGeometry
            A new data geometry object

        """

        # import plot here to avoid circular imports
        from .plot.plot import plot as plotter

        if data is None:
            d = copy.copy(self.data)
            transform = copy.copy(self.xform_data)
            if any([k in kwargs for k in ['reduce', 'align', 'normalize',
                                          'semantic', 'vectorizer', 'corpus']]):
                d = copy.copy(self.data)
                transform = None
        else:
            d = data
            transform = None

        # get kwargs and update with new kwargs
        new_kwargs = copy.copy(self.kwargs)
        update_kwargs = dict(transform=transform, reduce=self.reduce,
                       align=self.align, normalize=self.normalize,
                       semantic=self.semantic, vectorizer=self.vectorizer,
                       corpus=self.corpus)
        new_kwargs.update(update_kwargs)
        for key in kwargs:
            new_kwargs.update({key : kwargs[key]})
        return plotter(d, **new_kwargs)

    def save(self, fname, compression='blosc'):
        """
        Save method for the data geometry object

        The data will be saved as a 'geo' file, which is a dictionary containing
        the elements of a data geometry object saved in the hd5 format using
        `deepdish`.

        Parameters
        ----------

        fname : str
            A name for the file.  If the file extension (.geo) is not specified,
            it will be appended.

        compression : str
            The kind of compression to use.  See the deepdish documentation for
            options: http://deepdish.readthedocs.io/en/latest/api_io.html#deepdish.io.save

        """
        if compression is not None:
            warnings.warn("Hypertools has switched from deepdish to pickle "
                          "for saving DataGeomtry objects. 'compression' "
                          "argument has no effect and will be removed in a "
                          "future version",
                          FutureWarning)

        if self.dtype == 'df':
            data = self.data.to_dict('list')
        else:
            data = self.data

        # put geo vars into a dict
        geo = {
            'data' : data,
            'xform_data' : np.array(self.xform_data),
            'reduce' : self.reduce,
            'align' : self.align,
            'normalize' : self.normalize,
            'semantic' : self.semantic,
            'corpus' : np.array(self.corpus) if isinstance(self.corpus, list) else self.corpus,
            'kwargs' : self.kwargs,
            'version' : self.version,
            'dtype' : self.dtype
        }

        # if extension wasn't included, add it
        if fname[-4:]!='.geo':
            fname+='.geo'

        # save
        dd.io.save(fname, geo, compression=compression)
