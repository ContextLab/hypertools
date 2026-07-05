"""Data/object serialization for HyperTools 1.0.

`save(obj, fname)` pickles a hypertools result (array / DataFrame / list /
fitted model) to disk and backs the standalone `hyp.save`. No geo special-casing
(DataGeometry is removed in 1.0).

NOTE: figure/animation export (png/pdf/svg/html/gif/mp4) is NOT handled here —
it depends on the plot backends and lands in Plan 6 (plot). A figure passed here
is pickled like any other object; the format-aware exporter comes later.

Security note: this module uses `pickle` intentionally. `save`/`load` are meant
for round-tripping a user's own in-memory hypertools objects (arrays, frames,
fitted models) on trusted local disk, the same trust model as `numpy.save` or
`pandas.to_pickle`. Do not unpickle files from untrusted sources.
"""
import pickle


def save(obj, fname, **kwargs):
    """Pickle `obj` to `fname`."""
    with open(fname, "wb") as f:
        pickle.dump(obj, f)
