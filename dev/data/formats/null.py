def is_null(data):
    return data is None


def wrangle_null(data, return_model=False, **kwargs):
    if return_model:
        return data, {'model': None, 'args': [], 'kwargs': kwargs}
    return data


def is_empty(x):
    return (x is None) or (len(x) == 0)