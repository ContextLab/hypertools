# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np

from .srm import SRM, DetSRM, RSRM
from .procrustes import Procrustes
from .hyperalign import Hyperalign
from .null import NullAlign


# TODO: update class definitions of all alignment models to automatically funnel/unstack, trim, and pad data

def pad(x, c, max_rows=None):
    if not max_rows:
        max_rows = x.shape[0]

    y = np.zeros([max_rows, c])
    y[:, :x.shape[1]] = x[:max_rows, :]
    return y


def trim_and_pad(data):
    r = np.min([x.shape[0] for x in data])
    c = np.max([x.shape[1] for x in data])
    x = [pad(d, c, max_rows=r) for d in data]
    return x


@dw.decorate.apply_unstacked
def align(data, algorithm='hyper', **kwargs):
    """
    ARGUMENTS:
    :param data: data to reduce (numpy array or compatible, or a pandas
          dataframe or compatible).  Formatted as a 2d matrix whose
          rows are observations and whose columns are feature
          dimensions.  Can also input a list of Note: DataFrame indices are ignored; all DataFrames
          are aligned to the first r rows, where r is the number of rows
          in the shortest member of data.

    :param algorithm: one of: 'hyper', 'srm', 'procrustes'  Can also
          pass a function directly.

    all additional keyword arguments are passed datawrangler.apply_unstacked and then any remaining keyword arguments
          are passed to the alignment function

    RETURNS:
    :return: pandas dataframe (or list of dataframes) with number-of-observations rows and
    c columns, where c is the widest dataset in the list.
    """

    if type(data) == list:
        # noinspection PyCallingNonCallable
        return algorithm(data, **kwargs)
    else:
        return data


def pad_and_align(data, template, c, x, return_model=False):
    aligned = [np.zeros([d.shape[0], c]) for d in data]
    model = []
    for i in range(0, len(x)):
        proj = pro_fit_xform(x[i], template, return_proj=True)
        model.append(proj)
        padded_data = np.copy(aligned[i])
        padded_data[:, :data[i].shape[1]] = data[i]
        aligned[i] = transform(padded_data, proj)

    if return_model:
        return model, aligned
    else:
        return aligned


# debug this...
def hyper(data, n_iter=10, return_model=False):
    """
    data: a list of numpy arrays
    """
    assert n_iter >= 0, 'Number of iterations must be non-negative'

    # STEP 0: STANDARDIZE SIZE AND SHAPE
    #  - find smallest number of rows and max number of columns
    #  - remove extra rows and zero-pad to equalize number of columns
    r, c, x = trim_and_pad(data)

    if n_iter == 0:
        if return_model:
            return [np.eye(c) for _ in data], x
        else:
            return x

    # STEP 1: TEMPLATE
    template = np.copy(x[0])
    for i in range(1, len(x)):
        template += pro_fit_xform(x[i], template / (i + 1))
    template /= len(x)

    # STEP 2: NEW COMMON TEMPLATE
    #  - align each subj to template
    template2 = np.zeros_like(template)
    for i in range(0, len(x)):
        template2 += pro_fit_xform(x[i], template)
    template2 /= len(x)

    # STEP 3: SECOND ROUND OF ALIGNMENTS
    #  - align each subj to template2
    transformed = pad_and_align(data, template2, c, x)

    # TODO: if n_iter == 1, return model + transformed; otherwise compute m0, x0 = hyper(transformed, n_iter=n_iter-1,
    #  return_model=return_model) and then return model * m0, x0
    # if return_model, re-align original data to transformed data
    if return_model:
        model = pro_fit_xform(data, t, return_proj=True)
        return model, transformed
    else:
        return transformed


def srm(data, return_model=False):
    data = [i.T for i in data]
    srm_model = SRM(features=np.min([i.shape[0] for i in data]))
    srm_model.fit(data)
    transformed = [i.T for i in srm_model.transform(data)]

    if return_model:
        return srm_model, transformed
    else:
        return transformed


# TODO: debug this...
def procrustes(data, template=None, return_model=False):
    data2 = [d.copy() for d in data]
    if template is None:
        template = data2[0]
    data2.append(template.copy())

    _, c, x = trim_and_pad(data2)

    return pad_and_align(data2[:-1], template, c, x[:-1], return_model=return_model)
