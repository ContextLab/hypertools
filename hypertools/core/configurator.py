# noinspection PyPackageRequirements
import datawrangler as dw
import os
from configparser import ConfigParser

from pkg_resources import get_distribution
from .shared import RobustDict


__version__ = get_distribution('hypertools')


def get_default_options(fname=None):
    """
    Parse a config.ini file

    Parameters
    ----------
    :param fname: absolute-path filename for the config.ini file (default: hypertools/hypertools/core/config.ini)
    Returns
    -------
    :return: A dictionary whose keys are function names and whose values are dictionaries of default arguments and
    keyword arguments
    """
    if fname is None:
        fname = os.path.join(os.path.dirname(__file__), 'config.ini')
        
    return RobustDict(dw.core.update_dict(dw.core.get_default_options(),
                      dw.core.get_default_options(fname)),
                      __default_value__={})
