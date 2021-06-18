import pandas as pd


def is_array(x):
    return (not ('str' in str(type(x)))) and (type(x).__module__ == 'numpy')


def wrangle_array(data, return_model=False, **kwargs):
    if return_model:
        return pd.DataFrame(data, **kwargs), {'model': pd.DataFrame, 'args': [], 'kwargs': kwargs}
    return pd.DataFrame(data, **kwargs)
