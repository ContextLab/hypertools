# noinspection PyPackageRequirements
import datawrangler as dw


@dw.decorate.list_generalizer
def unpack_model(m, valid=None, parent_class=None):
    if valid is None:
        valid = []
    
    if (type(m) is str) and m in [v.__name__ for v in valid]:
        return [v for v in valid if v.__name__ == m][0]
    elif parent_class is not None and issubclass(m, parent_class):
        return m
    elif type(m) is dict and has_all_attributes(m, ['model', 'args', 'kwargs']):
        return dw.core.update_dict(m, {'model': unpack(m['model'], valid=valid, parent_class=parent_class)})
    else:
        raise ValueError(f'unknown model: {m}')
