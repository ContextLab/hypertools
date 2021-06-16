import pandas as pd


def is_array(x):
    return (not ('str' in str(type(x)))) and (type(x).__module__ == 'numpy')


def wrangle_array(data, **kwargs):
    return pd.DataFrame(data, **kwargs)