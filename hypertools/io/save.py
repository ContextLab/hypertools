"""Data/object serialization for HyperTools 1.0.

`save(obj, fname)` writes a hypertools result (array / DataFrame / list /
fitted model) to disk and backs the standalone `hyp.save`. The output
format follows the file extension (see `save` for the table); pickle is
the default for `.pkl`/unknown/missing extensions. No geo special-casing
(DataGeometry is removed in 1.0).

Figure/animation export (png/pdf/svg/gif/mp4) is handled by the plot
backends -- use `hyp.plot(..., save_path=...)` or the returned
figure/animation's own save methods. A static matplotlib Figure passed
here is pickled like any other object; animation results cannot be
pickled (they hold live callbacks) and raise a pointer to `save_path`.

Security note: pickle output is meant for round-tripping a user's own
in-memory hypertools objects (arrays, frames, fitted models) on trusted
local disk, the same trust model as `numpy.save` or `pandas.to_pickle`.
Do not unpickle files from untrusted sources.

Writes are atomic (QC 2026-07, F20-save-001): data is serialized to a
temporary file in the target directory and moved into place only on
success, so a failed save never destroys an existing file at that path
and never leaves a partial file behind.
"""
import os
import pickle
import tempfile
from os.path import expanduser, expandvars
from pathlib import Path

import numpy as np
import pandas as pd

from .._shared.exceptions import HypertoolsIOError

# extensions written as real (non-pickle) formats; everything else --
# including .pkl/.pickle/.p/.geo, unknown extensions, and extensionless
# paths -- is pickled (hyp.load recognizes pickles by their magic bytes
# whatever the extension)
_FORMAT_EXTENSIONS = ('.csv', '.tsv', '.txt', '.npy', '.npz', '.json',
                      '.parquet', '.mat', '.xlsx')


def save(obj, fname, protocol=None):
    """Save `obj` to `fname`, choosing the output format from the file
    extension.

    Extension-aware export (QC 2026-07, F20-save-002/-010): files named
    with a recognized data extension are written in that REAL format, so
    external tools (pandas/numpy/R/Excel/MATLAB) can read them; every
    format also round-trips through :func:`hypertools.load`.

    ========================= ==========================================
    extension                 written as
    ========================= ==========================================
    .csv / .tsv / .txt        delimited text (via ``DataFrame.to_csv``;
                              comma, tab, and comma respectively).
                              non-DataFrame input is converted with
                              ``pd.DataFrame(np.asarray(obj))``; a
                              non-default index is included
    .npy                      ``numpy.save`` binary array
    .npz                      ``numpy.savez`` archive: a list/tuple
                              saves one array per element, a dict one
                              array per key, anything else a single
                              array
    .json                     ``DataFrame.to_json``
    .parquet                  ``DataFrame.to_parquet`` (column names
                              are stringified, as parquet requires)
    .mat                      MATLAB file via ``scipy.io.savemat``
                              (arrays land in the variable ``'data'``;
                              dicts map keys to variables)
    .xlsx                     Excel workbook via ``DataFrame.to_excel``
    anything else             a Python pickle (including .pkl/.pickle/
                              .p/.geo, unknown extensions, and paths
                              with no extension)
    ========================= ==========================================

    Writes are atomic: the data is serialized to a temporary file next to
    the target and moved into place only on success, so a failed save
    never destroys an existing file at that path.

    Parameters
    ----------
    obj : object
        What to save: a numpy array, pandas DataFrame, list of arrays,
        fitted model/Pipeline, dict of results, matplotlib Figure, or (for
        pickle output) any other picklable Python object. Data formats
        (.csv, .npy, ...) require array-/table-like input.
    fname : str or path-like
        Target path. ``~`` and ``$ENV_VARS`` are expanded, matching
        :func:`hypertools.load`. The extension selects the format (table
        above).
    protocol : int, optional
        Pickle protocol version (pickle output only; default: pickle's
        default protocol). Passing it with a data-format extension raises
        ``ValueError``.

    Returns
    -------
    None

    Raises
    ------
    hypertools._shared.exceptions.HypertoolsIOError
        If the object cannot be serialized in the requested format (e.g.
        an unpicklable object holding an open file handle, or a nested
        dict aimed at .csv), or if the target path is invalid (missing
        parent directory, existing directory, no write permission).
    TypeError
        If ``fname`` is not a string or path-like object.

    Examples
    --------
    >>> import numpy as np, pandas as pd, hypertools as hyp
    >>> hyp.save(np.random.rand(10, 3), 'data.pkl')     # pickle
    >>> hyp.save(pd.DataFrame({'a': [1, 2]}), 'data.csv')  # real CSV
    >>> hyp.load('data.csv')['a'].tolist()
    [1, 2]

    .. warning::
        Pickled files can execute arbitrary code when loaded -- only
        share/load pickles with sources you trust. For figures and
        animations, prefer ``hyp.plot(..., save_path='fig.png')``.
    """
    if not isinstance(fname, (str, os.PathLike)):
        raise TypeError(
            'hypertools.save: fname must be a string or path-like object, '
            f'got {type(fname).__name__}. Did you swap the argument '
            'order? The signature is save(obj, fname).')
    path = Path(expanduser(expandvars(os.fspath(fname))))
    if path.is_dir():
        raise HypertoolsIOError(
            f'cannot save to {path}: it is an existing directory, not a '
            f"file. Pass a file path instead (e.g. "
            f"{path / 'results.pkl'}).")
    if not path.parent.is_dir():
        raise HypertoolsIOError(
            f'cannot save to {path}: the parent directory {path.parent} '
            f'does not exist. Create it first (e.g. '
            f'os.makedirs({str(path.parent)!r})).')

    ext = path.suffix.lower()
    if protocol is not None and ext in _FORMAT_EXTENSIONS:
        raise ValueError(
            f'protocol= applies only to pickle output, but {path.name!r} '
            f'selects the {ext} format; drop protocol= or use a pickle '
            'extension (e.g. .pkl)')

    # atomic write: serialize to a temp file in the target directory, then
    # move it into place -- a failed save never truncates an existing file
    # and never leaves a partial file behind (QC 2026-07, F20-save-001)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent),
                                    prefix=f'.{path.name}.', suffix='.tmp')
    os.close(fd)
    try:
        _write_payload(obj, tmp_name, ext, protocol, target=path)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _write_payload(obj, tmp_name, ext, protocol, target):
    """Serialize ``obj`` to the temp file ``tmp_name`` in the format
    selected by ``ext`` (``target`` names the eventual path, for error
    messages only)."""
    try:
        if ext in ('.csv', '.txt'):
            _as_frame(obj, ext, target).to_csv(
                tmp_name, index=_include_index(obj))
        elif ext == '.tsv':
            _as_frame(obj, ext, target).to_csv(
                tmp_name, sep='\t', index=_include_index(obj))
        elif ext == '.npy':
            with open(tmp_name, 'wb') as f:
                np.save(f, np.asarray(obj))
        elif ext == '.npz':
            if isinstance(obj, (list, tuple)):
                arrays = {f'arr_{i}': np.asarray(a)
                          for i, a in enumerate(obj)}
            elif isinstance(obj, dict):
                arrays = {str(k): np.asarray(v) for k, v in obj.items()}
            else:
                arrays = {'arr_0': np.asarray(obj)}
            with open(tmp_name, 'wb') as f:
                np.savez(f, **arrays)
        elif ext == '.json':
            _as_frame(obj, ext, target).to_json(tmp_name)
        elif ext == '.parquet':
            frame = _as_frame(obj, ext, target)
            if not all(isinstance(c, str) for c in frame.columns):
                frame = frame.rename(columns=str)
            frame.to_parquet(tmp_name)
        elif ext == '.mat':
            from scipy.io import savemat
            if isinstance(obj, dict):
                payload = {str(k): np.asarray(v) for k, v in obj.items()}
            else:
                payload = {'data': np.asarray(obj)}
            with open(tmp_name, 'wb') as f:
                savemat(f, payload)
        elif ext == '.xlsx':
            # openpyxl validates the target filename's extension, so write
            # to a buffer and dump the bytes into the (extensionless) temp
            # file ourselves
            import io as _io
            buffer = _io.BytesIO()
            _as_frame(obj, ext, target).to_excel(
                buffer, index=_include_index(obj), engine='openpyxl')
            with open(tmp_name, 'wb') as f:
                f.write(buffer.getvalue())
        else:
            _write_pickle(obj, tmp_name, protocol, target)
    except (HypertoolsIOError, MemoryError, KeyboardInterrupt):
        raise
    except Exception as e:
        raise HypertoolsIOError(
            f'could not save {type(obj).__name__} object to {target} as '
            f'{ext or "pickle"} ({type(e).__name__}: {e}). Use a .pkl '
            'extension to pickle arbitrary Python objects, or convert the '
            'data to an array/DataFrame first.') from e


def _as_frame(obj, ext, target):
    """DataFrame view of ``obj`` for the tabular writers, with a clear
    error when the object has no faithful 2-d tabular form."""
    if isinstance(obj, pd.DataFrame):
        return obj
    try:
        arr = np.asarray(obj)
    except Exception:
        arr = np.asarray(obj, dtype=object)
    if arr.dtype == object or arr.ndim > 2:
        raise HypertoolsIOError(
            f'cannot save {type(obj).__name__} object to {target}: the '
            f'{ext} format holds a single 2-d table, and this object has '
            'no faithful tabular form. Save it as a pickle instead (e.g. '
            f"'{Path(target).stem}.pkl'), or as .npz for a list/dict of "
            'arrays.')
    return pd.DataFrame(arr)


def _include_index(obj):
    """Write the index for DataFrames whose index carries information
    (anything but a fresh 0..n-1 RangeIndex)."""
    if not isinstance(obj, pd.DataFrame):
        return False
    index = obj.index
    return not (isinstance(index, pd.RangeIndex) and index.start == 0
                and index.step == 1)


def _write_pickle(obj, tmp_name, protocol, target):
    """Pickle ``obj`` (in memory first, so nothing is written on failure)
    with actionable errors for the common unpicklable cases."""
    if protocol is None:
        protocol = pickle.DEFAULT_PROTOCOL
    try:
        payload = pickle.dumps(obj, protocol=protocol)
    except (MemoryError, KeyboardInterrupt):
        raise
    except Exception as e:
        message = (f'could not pickle {type(obj).__name__} object for '
                   f'saving to {target} ({type(e).__name__}: {e}).')
        if type(obj).__name__ == 'HyperAnimation':
            message += (
                ' Animations hold live matplotlib callbacks that cannot '
                'be pickled -- export the animation while plotting '
                "instead, e.g. hyp.plot(data, animate=True, "
                "save_path='anim.gif'), or call its .save('anim.gif') "
                'method.')
        else:
            message += (
                ' Objects holding open file handles, lambdas, or locally '
                'defined functions cannot be pickled; remove those '
                'members or save the underlying data instead.')
        raise HypertoolsIOError(message) from e
    with open(tmp_name, 'wb') as f:
        f.write(payload)
