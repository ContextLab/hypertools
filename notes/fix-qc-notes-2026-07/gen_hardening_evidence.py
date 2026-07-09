#!/usr/bin/env python
"""Generate release-hardening evidence (QC 2026-07) for PR #280.

Produces "after-fix works" figures for the VISUAL fixes plus a numeric log for
the non-visual ones. Real data, no mocks, headless (Agg). Screenshots land in
notes/fix-qc-notes-2026-07/evidence/hardening/.
"""
import os
os.environ['MPLBACKEND'] = 'Agg'
import io as _io
import warnings
from contextlib import redirect_stderr

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hypertools as hyp

OUT = os.path.join(os.path.dirname(__file__), 'evidence', 'hardening')
os.makedirs(OUT, exist_ok=True)
log = []


def note(s):
    log.append(s)
    print(s)


rng = np.random.default_rng(0)

# --- B2: 1-D array now plots (was IndexError) -------------------------
fig = hyp.plot(np.cumsum(rng.standard_normal(80)), show=False)
fig.suptitle("B2: hyp.plot(1-D array) — was IndexError, now renders")
fig.savefig(os.path.join(OUT, 'b2_1d_array.png'), dpi=110)
plt.close('all')
note("B2 1-D array: OK -> b2_1d_array.png")

# --- B2: flat list of numbers now plots (was crash) -------------------
fig = hyp.plot(list(np.sin(np.linspace(0, 6 * np.pi, 120))), show=False)
fig.suptitle("B2: hyp.plot([floats...]) — flat numeric list, now renders")
fig.savefig(os.path.join(OUT, 'b2_flat_list.png'), dpi=110)
plt.close('all')
note("B2 flat list: OK -> b2_flat_list.png")

# --- B2: hue length is validated (clear ValueError) -------------------
try:
    hyp.plot(rng.standard_normal((40, 5)), hue=list(range(10)), show=False)
    note("B2 hue-length: NO ERROR (UNEXPECTED)")
except ValueError as e:
    note(f"B2 hue-length ValueError: {str(e)[:120]}")
plt.close('all')

# --- B5: align on mismatched COLUMN counts (was default crash) --------
data = [rng.standard_normal((60, 10)),
        rng.standard_normal((60, 8)),
        rng.standard_normal((60, 12))]
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    aligned = hyp.align(data, align='hyper')
shapes = [np.asarray(a).shape for a in aligned]
note(f"B5 align mismatched cols: OK, aligned shapes={shapes}, "
     f"finite={all(np.isfinite(np.asarray(a)).all() for a in aligned)}")
fig = hyp.plot(aligned, '.', show=False)
fig.suptitle("B5: hyp.align([60x10, 60x8, 60x12]) — mismatched cols, was crash")
fig.savefig(os.path.join(OUT, 'b5_align_mismatched_cols.png'), dpi=110)
plt.close('all')
note("B5 align plot -> b5_align_mismatched_cols.png")

# --- K1: backend selection actually switches the renderer -------------
mpl_fig = hyp.plot(rng.standard_normal((50, 4)), backend='matplotlib', show=False)
note(f"K1 backend='matplotlib' -> {type(mpl_fig).__module__}.{type(mpl_fig).__name__}")
mpl_fig.suptitle("K1: backend='matplotlib' -> matplotlib Figure")
mpl_fig.savefig(os.path.join(OUT, 'k1_matplotlib.png'), dpi=110)
plt.close('all')
try:
    pl_fig = hyp.plot(rng.standard_normal((50, 4)), backend='plotly', show=False)
    note(f"K1 backend='plotly'     -> {type(pl_fig).__module__}.{type(pl_fig).__name__}")
except Exception as e:
    note(f"K1 backend='plotly' EXCEPTION: {type(e).__name__}: {e}")
# preference + context-manager restore
import importlib
_bk = importlib.import_module('hypertools.plot.backend')
before = getattr(_bk, 'PREFERRED_RENDER_BACKEND', None)
with hyp.set_interactive_backend('plotly'):
    inside = getattr(_bk, 'PREFERRED_RENDER_BACKEND', None)
after = getattr(_bk, 'PREFERRED_RENDER_BACKEND', None)
note(f"K1 pref: before={before!r} inside={inside!r} after={after!r} "
     f"(restored={before == after})")

# --- B1: canonical dict spec honors ndims (was silent no-reduce) ------
X = rng.standard_normal((150, 7))
out_dict = hyp.reduce(X, reduce={'model': 'PCA', 'kwargs': {'whiten': True}}, ndims=2)
from sklearn.decomposition import PCA
ref = PCA(n_components=2, whiten=True).fit_transform(X)
same = np.allclose(np.abs(np.asarray(out_dict)), np.abs(ref), atol=1e-6)
note(f"B1 reduce dict+ndims: shape={np.asarray(out_dict).shape} "
     f"(expected (150, 2)); matches PCA(n_components=2,whiten) up to sign={same}")
# return-type stability
r_single = hyp.reduce(X, ndims=3)
note(f"B1 reduce single-array return type: {type(r_single).__name__} "
     f"shape={getattr(r_single, 'shape', None)}")

# --- B3: PPCA impute preserves observed values, shape ------------------
Y = rng.standard_normal((40, 6))
Yn = Y.copy()
mask = rng.random(Y.shape) < 0.15
Yn[mask] = np.nan
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    filled = np.asarray(hyp.impute(Yn, model='PPCA'))
obs_ok = np.allclose(filled[~mask], Y[~mask], atol=1e-9) if filled.shape == Y.shape else False
note(f"B3 PPCA impute: in shape={Yn.shape} out shape={filled.shape} "
     f"observed_preserved(1e-9)={obs_ok} "
     f"filled_finite={np.isfinite(filled).all()}")

# --- B6: HyperAnimation.save('.gif') writes a real gif ----------------
anim = hyp.plot(rng.standard_normal((90, 3)), animate=True, show=False)
gif_path = os.path.join(OUT, 'b6_anim.gif')
anim.save(gif_path)
with open(gif_path, 'rb') as fh:
    magic = fh.read(6)
note(f"B6 HyperAnimation.save('.gif'): {os.path.getsize(gif_path)} bytes, "
     f"magic={magic!r} (GIF89a expected)")
plt.close('all')

with open(os.path.join(OUT, 'RESULTS.txt'), 'w') as fh:
    fh.write("\n".join(log) + "\n")
print("\n=== wrote evidence to", OUT, "===")
