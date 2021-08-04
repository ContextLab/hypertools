# noinspection PyPackageRequirements
import datawrangler as dw


def load(x, dtype=None, **kwargs):
    datasets = {'mushrooms': 'https://www.dropbox.com/s/v9kidmydqgyjjhc/mushrooms.npz?dl=1',
                'spiral': 'https://www.dropbox.com/s/98fcbbyubwtn1nk/spiral.npz?dl=1',
                'weights_avg': 'https://www.dropbox.com/s/1cblitouvg2x1zt/weights_avg.npz?dl=1',
                'weights_sample': 'https://www.dropbox.com/s/sjs6wp9k9l0bd1d/weights_sample.npz?dl=1',
                'weights': 'https://www.dropbox.com/s/xjpmxc4nqzcvj5u/weights.npz?dl=1'}

    if (type(x) is str) and x in datasets.keys():
        return dw.io.load(datasets[x], dtype='numpy', **kwargs)['data']
    else:
        return dw.io.load(x, dtype=dtype, **kwargs)
