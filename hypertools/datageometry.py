import copy
import deepdish as dd
import numpy as np
from .tools.normalize import normalize as normalizer
from .tools.reduce import reduce as reducer
from .tools.align import align as aligner
from .tools.format_data import format_data
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
                 reduce=None, align=None, normalize=None, text=None, kwargs=None,
                 version=__version__):

        # matplotlib figure handle
        self.fig = fig

        # matplotlib axis handle
        self.ax = ax

        # matplotlib line_ani handle (if its an animation)
        self.line_ani = line_ani

        # the raw data
        self.data = data

        # the transformed data
        self.xform_data = xform_data

        # dictionary of model and model_params
        self.reduce = reduce

        # 'hyper', 'SRM' or None
        self.align = align

        # 'within', 'across', 'row' or False
        self.normalize = normalize

        # text params
        self.text = text

        # dictionary of kwargs
        self.kwargs = kwargs

        # hypertools version
        self.version = version

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
            reduce_model = {'model' : self.reduce['model'],
                            'params' : self.reduce['params']}
            align_model = {'model' : self.align['model'],
                           'params' : self.align['params']}
            text_model = {'model' : self.text['model'],
                          'params' : self.text['params']}
            return format_data(
                aligner(
                reducer(
                normalizer(data,
                normalize=self.normalize),
                reduce=reduce_model,
                ndims=self.reduce['params']['n_components']),
                align=align_model),
                text=text_model, ppca=True)

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
            data = self.xform_data
            transform = False
            if any([k in kwargs for k in ['reduce', 'align', 'normalize', 'text']]):
                data = self.data
                transform = True
        else:
            transform = True

        # get kwargs and update with new kwargs
        new_kwargs = copy.copy(self.kwargs)
        for key in kwargs:
            new_kwargs.update({key : kwargs[key]})

        return plotter(data, transform=transform, **new_kwargs)

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


        # put geo vars into a dict
        geo = {
            'data' : self.data,
            'xform_data' : self.xform_data,
            'reduce' : self.reduce,
            'align' : self.align,
            'normalize' : self.normalize,
            'text' : self.text,
            'kwargs' : self.kwargs,
            'version' : self.version
        }

        # if extension wasn't included, add it
        if fname[-4:]!='.geo':
            fname+='.geo'

        # save
        dd.io.save(fname, geo, compression=compression)
