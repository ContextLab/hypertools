# noinspection PyPackageRequirements
import datawrangler as dw

__version__ = get_distribution('hypertools')


def get_default_options(fname='config.ini'):
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
    return dw.core.update_dict(dw.core.get_default_options(), dw.core.get_default_options(fname))
