# noinspection PyPackageRequirements
import datawrangler as dw


def get(x, ind, axis=0):
    if dw.util.array_like(x) and len(x) > 0:
        if not dw.zoo.is_array(x):
            x = np.array(x)
        return np.take(x, ind % x.shape[axis], axis=axis)
    else:
        return x
