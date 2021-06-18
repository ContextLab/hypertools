from configparser import ConfigParser
from pkg_resources import get_distribution

__version__ = get_distribution('hypertools')

def get_default_options(fname='config.ini'):
    config = ConfigParser()
    config.read(fname)
    return config