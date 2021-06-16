from configparser import ConfigParser

def get_default_options(fname='config.ini'):
    config = ConfigParser()
    config.read(fname)
    return config