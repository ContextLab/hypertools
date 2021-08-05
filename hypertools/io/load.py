# noinspection PyPackageRequirements
import datawrangler as dw


def load(x, dtype=None, **kwargs):
    datasets = {'mushrooms': 'https://www.dropbox.com/s/xrw48u2qylo4d1v/mushrooms.pkl?dl=1',
                'spiral': 'https://www.dropbox.com/s/ig3xddadtcdmhqd/spiral.pkl?dl=1',
                'weights_avg': 'https://www.dropbox.com/s/t7tk6t7vhxp12mf/weights_avg.pkl?dl=1',
                'weights_sample': 'https://www.dropbox.com/s/73yrbie2uc27h86/weights_sample.pkl?dl=1',
                'weights': 'https://www.dropbox.com/s/1d0axn0m6u642lb/weights.pkl?dl=1'}

    if (type(x) is str) and x in datasets.keys():
        return dw.io.load(datasets[x], dtype='pickle', **kwargs)
    else:
        return dw.io.load(x, dtype=dtype, **kwargs)
