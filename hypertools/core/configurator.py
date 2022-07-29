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
        
    print(f'loading config from: {fname}')
    return RobustDict(dw.core.get_default_options(fname))

    # # add in defaults from datawrangler
    # updated = dw.core.update_dict(config, dw.core.get_default_options())
    
    # # overwrite datawrangler defaults with hypertools config
    # return RobustDict(dw.core.update_dict(updated, config), __default_value__={})
