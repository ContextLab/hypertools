from hashlib import blake2b as hasher
import os
import io
import requests
import dill

from ..core.configurator import get_default_options
from ..data.hyperdata import HyperData

defaults = get_default_options()


def get_local_fname(x, digest_size=10):
    if os.path.exists(x):
        return x

    h = hasher(digest_size=digest_size)
    h.update(x.encode('ascii'))
    return os.path.join(eval(defaults['data']['datadir']), h.hexdigest() + '.hyp')


def load(x, base_url='https://docs.google.com/uc?export=download', dtype='pickle', **kwargs):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def load_stream(url, params=None):
        if params is None:
            params = {}
        session = requests.Session()
        response = session.get(url, params=params, stream=True)
        token = get_confirm_token(response)
        if token:
            params['confirm'] = token
            response = session.get(url, params=params, stream=True)

        return response.content

    # noinspection PyShadowingNames
    def helper(fname, **helper_kwargs):
        if dtype == 'pickle':
            with open(fname, 'rb') as f:
                return dill.load(f, **helper_kwargs)
        elif dtype == 'numpy':
            if 'allow_pickle' not in helper_kwargs.keys():
                helper_kwargs['allow_pickle'] = True
            data = np.load(fname, **helper_kwargs)
            if type(data) is dict:
                if len(data.keys()) == 1:
                    return data[list(data.keys())[0]]
            return data
        else:
            raise ValueError(f'Unknown datatype: {dtype}')

    assert type(x) is str, IOError('cannot interpret non-string filename')
    fname = get_local_fname(x)
    if os.path.exists(fname):
        return helper(fname, **kwargs)
    else:
        # noinspection PyBroadException
        try:     # is x a Google ID?
            data = load_stream(base_url, params={'id': x})
        except:  # is x another URL?
            if x.startswith('http'):
                data = load_stream(x)
            else:
                raise IOError('cannot find data at source: {x}')
        save(x, data, dtype=dtype)
        return load(x, dtype=dtype, **kwargs)


def save(x, obj, dtype=None, **kwargs):
    assert type(x) is str, IOError('cannot interpret non-string filename')
    fname = get_local_fname(x)

    if type(obj) is bytes:
        with open(fname, 'wb') as f:
            f.write(obj)
    elif type(obj) is str:
        with open(fname, 'w') as f:
            f.write(obj)
    elif dtype == 'pickle':
        with open(fname, 'wb') as f:
            dill.dump(obj, f, **kwargs)
    elif dtype == 'numpy':
        np.savez(fname, obj, **kwargs)
    else:
        raise ValueError(f'cannot save object (specified dtype: {dtype}; observed type: {type(obj)})')

