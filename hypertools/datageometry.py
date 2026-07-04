"""
INTERNAL -- not part of the public API.

Retained solely to unpickle the hosted example-dataset geo files so
:func:`hypertools.load` can extract their raw data; hypertools 2.0 users
never receive a DataGeometry (``plot()`` returns a Figure, ``load()``
returns raw data).

This module MUST stay at the import path ``hypertools.datageometry`` so the
hosted example-dataset pickles (created by hypertools < 2.0) resolve their
stored class reference ``hypertools.datageometry.DataGeometry`` when
unpickled. Unpickling bypasses ``__init__`` and restores the instance
``__dict__`` directly, so the minimal class below is sufficient to recover
``self.data`` via :meth:`get_data`.
"""

import copy


class DataGeometry(object):
    """INTERNAL unpickle-only shell for legacy hypertools geo pickles.

    Not part of the public API. Exists only so the hosted example-dataset
    pickles can be unpickled and their raw data extracted via
    :meth:`get_data`. hypertools 2.0 never constructs or returns one of
    these to users.
    """

    def __init__(self, data=None, **kwargs):
        # Kept minimal so hypertools.io.load._load_legacy (deepdish-format
        # geos) can still reconstruct an object. Unpickling the hosted
        # pickles does NOT call this -- it restores __dict__ directly.
        self.data = data
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_data(self):
        """Return a copy of the data."""
        return copy.copy(self.data)
