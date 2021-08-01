# noinspection PyPackageRequirements
import datawrangler as dw


def get(x, ind):
    if dw.util.array_like(x) and len(x) > 0:
        return x[ind % len(x)]
    else:
        return x
