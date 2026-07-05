"""Hyperalignment (Haxby et al., 2011) as an :class:`Aligner` child.

This ports dev-1.0's *rescaled* hyperalignment (``hypertools/tools/align.py``:
``_hyperalign_pass`` + the ``n_iter`` rescale loop) rather than the fork's
version, which omits the per-pass rescale and lets procrustes' optimal scaling
factor (< 1 whenever alignment is imperfect) geometrically collapse the data
toward zero across passes.

Each pass runs the classic procedure -- build a sequential template, refine it,
align every dataset to the refined template -- rescaling the template to the
datasets' mean Frobenius norm at each step, and rescaling the whole aligned set
back to the original norm at the end of the pass. The aligned datasets feed the
next pass, so convergence toward the common space compounds.

The fitter reuses the :func:`hypertools.align.procrustes.align` SVD primitive and
accumulates, per dataset, the *composed* projection across all passes (including
the per-pass rescale scalars). The stored ``proj`` therefore satisfies
``original @ proj == converged_aligned_data`` exactly, so ``transform`` reproduces
dev-1.0's ``align()`` output (which returns aligned data directly) while still
exposing a genuine per-dataset projector.
"""
import numpy as np
import pandas as pd

from .common import Aligner
from .procrustes import align as _proc_align


def _rescale(t, mean_norm):
    """Scale array `t` to have Frobenius norm `mean_norm` (no-op if `t` is ~0)."""
    norm = np.linalg.norm(t)
    return t * (mean_norm / norm) if norm > 0 else t


def _project(source, target):
    """Procrustes projection matrix mapping `source` onto `target`
    (scaling + reflection), reusing the align() primitive so behavior matches
    the Procrustes child."""
    return np.asarray(_proc_align(np.asarray(source, dtype=float),
                                  np.asarray(target, dtype=float)))


def _one_pass(m):
    """One full classic hyperalignment pass with per-step rescale to the
    datasets' mean Frobenius norm.

    Returns ``(aligned, proj, template)`` where ``aligned[x] == m[x] @ proj[x]``
    and ``template`` is the refined common template for this pass.
    """
    m = [np.asarray(x, dtype=float) for x in m]
    mean_norm = np.mean([np.linalg.norm(x) for x in m])

    # STEP 1: initial (sequential) template
    template = np.copy(m[0])
    for x in range(1, len(m)):
        template = template + m[x] @ _project(m[x], template / (x + 1))
    template = _rescale(template / len(m), mean_norm)

    # STEP 2: refined template
    template2 = np.zeros_like(template)
    for x in range(len(m)):
        template2 = template2 + m[x] @ _project(m[x], template)
    template2 = _rescale(template2 / len(m), mean_norm)

    # STEP 3: align every dataset to the refined template
    proj = [_project(m[x], template2) for x in range(len(m))]
    aligned = [m[x] @ proj[x] for x in range(len(m))]
    return aligned, proj, template2


def fitter(data, n_iter=10, **kwargs):
    """Run dev-1.0's rescaled hyperalignment passes and return the composed
    per-dataset projections under key ``'proj'``."""
    assert type(data) is list, 'data must be a list'
    n = len(data)
    dims = [np.asarray(d).shape[1] for d in data]
    if n == 0:
        return {'proj': []}
    if n == 1 or int(n_iter) == 0:
        return {'proj': [np.eye(c) for c in dims]}

    aligned = [np.asarray(d, dtype=float) for d in data]
    orig_norm = np.mean([np.linalg.norm(x) for x in aligned])
    proj = [np.eye(c) for c in dims]  # cumulative original -> aligned projectors

    for _ in range(max(1, int(n_iter))):
        aligned, pass_proj, _template = _one_pass(aligned)
        cur_norm = np.mean([np.linalg.norm(a) for a in aligned])
        s = (orig_norm / cur_norm) if cur_norm > 0 else 1.0
        aligned = [a * s for a in aligned]
        # fold the uniform rescale scalar into each pass projection so that
        # original @ proj continues to equal the (rescaled) aligned data
        proj = [proj[i] @ (pass_proj[i] * s) for i in range(n)]

    return {'proj': proj}


def transformer(data, **kwargs):
    """Apply the stored per-dataset projections to the (trimmed+padded) data."""
    proj = kwargs['proj']
    return [pd.DataFrame(np.asarray(d, dtype=float) @ np.asarray(p),
                         index=d.index)
            for d, p in zip(data, proj)]


class HyperAlign(Aligner):
    """Hyperalignment (Haxby et al., 2011) with dev-1.0's per-pass rescaling.

    :param n_iter: number of hyperalignment passes; the common template is
        re-estimated from the aligned data and all datasets re-aligned to it,
        repeatedly. More iterations give a more stable common space
        (default: 10). ``n_iter=0`` yields identity projections.
    """
    def __init__(self, n_iter=10, **kwargs):
        assert n_iter >= 0, 'n_iter must be non-negative'
        super().__init__(required=['proj'], fitter=fitter, transformer=transformer,
                         data=None, n_iter=n_iter, **kwargs)
