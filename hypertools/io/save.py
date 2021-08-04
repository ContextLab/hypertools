# noinspection PyPackageRequirements
import datawrangler as dw


def save(*args, **kwargs):
    return dw.io.save(*args, **kwargs)
