# noinspection PyPackageRequirements
import datawrangler as dw


def load(x, dtype=None, **kwargs):
    datasets = {'mushrooms': 'https://www.dropbox.com/s/xrw48u2qylo4d1v/mushrooms.pkl?dl=1',
                'spiral': 'https://www.dropbox.com/s/ig3xddadtcdmhqd/spiral.pkl?dl=1',
                'weights_avg': 'https://www.dropbox.com/s/t7tk6t7vhxp12mf/weights_avg.pkl?dl=1',
                'weights_sample': 'https://www.dropbox.com/s/73yrbie2uc27h86/weights_sample.pkl?dl=1',
                'weights': 'https://www.dropbox.com/s/1d0axn0m6u642lb/weights.pkl?dl=1',
                'datasaurus': 'https://www.dropbox.com/s/6wxjyw8p052a5t9/datasaurus.pkl?dl=1',
                'biplane': 'https://www.dropbox.com/s/4b9y9ouvjpjbj6x/biplane.pkl?dl=1',
                'bunny': 'https://www.dropbox.com/s/7d9vo9idqk1hn31/bunny.pkl?dl=1',
                'cube': 'https://www.dropbox.com/s/tkrwe2m4maxl83j/cube.pkl?dl=1',
                'dragon': 'https://www.dropbox.com/s/6w84icbvzh5oilr/dragon.pkl?dl=1',
                'mask': 'https://www.dropbox.com/s/i2yrxsevyncbwb3/egyption_mask.pkl?dl=1',
                'sphere': 'https://www.dropbox.com/s/wp8suye6oh4ze3u/sphere.pkl?dl=1',
                'vase': 'https://www.dropbox.com/s/prquc7ov18zguuu/vase.pkl?dl=1',
                'teapot': 'https://www.dropbox.com/s/f3jj18h3ge2gns6/teapot.pkl?dl=1'}

    if (type(x) is str) and x in datasets.keys():
        return dw.io.load(datasets[x], dtype='pickle', **kwargs)
    else:
        return dw.io.load(x, dtype=dtype, **kwargs)
