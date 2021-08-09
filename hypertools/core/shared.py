# noinspection PyPackageRequirements
import datawrangler as dw


@dw.decorate.list_generalizer
def unpack_model(m, valid=None, parent_class=None):
    if valid is None:
        valid = []

    if (type(m) is str) and m in [v.__name__ for v in valid]:
        return [v for v in valid if v.__name__ == m][0]
    elif parent_class is not None:
        try:
            if issubclass(m, parent_class):
                return m
        except TypeError:
            pass

    if type(m) is dict and all([k in m.keys() for k in ['model', 'args', 'kwargs']]):
        return dw.core.update_dict(m, {'model': unpack_model(m['model'], valid=valid, parent_class=parent_class)})
    elif type(m) is str:
        return m
    else:
        raise ValueError(f'unknown model: {m}')


class RobustDict(dict):
    """
    Dictionary subclass with more forgiving indexing:
      indexing a `RobustDict` with a key that doesn't exist returns
      None (or another specified default value) instead of throwing an error.
    """
    def __init__(self, *args, **kwargs):
        self.default_value = kwargs.pop('__default_value__', None)
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except NameError:
            return self.default_value

    def __missing__(self, key):
        return self.default_value
