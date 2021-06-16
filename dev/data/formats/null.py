def is_null(data):
    return data is None


def wrangle_null(data, **kwargs):
    return data


def is_empty(x):
    return (x is None) or (len(x) == 0)