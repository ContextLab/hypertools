# PR #280 — Release-hardening evidence (QC 2026-07)

Base: `dev-1.0-refactor`. Branch: `fix/qc-notes-2026-07`. **Not to be merged; Jeremy reviews.**

This document is the source for the per-batch PR evidence comments. Each batch:
reproduction (before) → fix commit → verification (numeric / screenshot) → independent red-team verdict.

Method: 8 parallel read-only bug-hunt agents swept the whole package (not just the prior audit),
each reproducing REAL bugs with numeric evidence; ~30 confirmed defects were organized into batches
B1–B8 + K1. Every fix was reproduced before, verified after (numeric output and, for plotting,
rendered screenshots), and then adversarially red-teamed by an INDEPENDENT subagent.

---

## K1 — backend routing (`edf6049e`)

**Defect:** `set_interactive_backend('plotly')` then `plot(..., animate=True)` raised
`HypertoolsBackendError` — the call tried to switch the *matplotlib* GUI backend to a name
(`'plotly'`) matplotlib does not know. Deeper: selecting 'plotly'/'matplotlib' never actually
changed the render backend.

**Fix:** `backend.py` gains a module-global `PREFERRED_RENDER_BACKEND`.
`set_interactive_backend('plotly'|'matplotlib')` now sets that render preference instead of
attempting an mpl backend switch (`is_render_backend`); real GUI backends (TkAgg/Qt5Agg) still
switch mpl as before. `plotly_backend.resolve_backend('auto')` consults the preference.
`plot_wrapper` skips the mpl backend switch when the resolved render backend is plotly.
Context-manager form restores the previous preference on exit.

- Repro (before): `hyp.set_interactive_backend('plotly'); hyp.plot(np.random.rand(60,4), animate=True)` → error.
- Verify (after): returns `plotly.graph_objs.Figure`; matplotlib pref returns mpl `Figure`;
  context-manager restores prior preference; real GUI backend names still switch mpl.
- Red-team (K1): **SOLID** (344 passed, 0 failed). Verified: correct figure type both ways;
  context-manager restore incl. prior=None, nested blocks, and exception-in-block (try/finally);
  no state leakage across a 4-plot sequence; animate+plotly no crash. One minor off-spec gap —
  a *capitalized* name (`'Plotly'`) was treated as an mpl backend — **fixed in `a81c7a91`**.

---

## B1 — reduce dispatch (`66675cb0`)

**Defects:**
- I1 (silent-wrong): canonical `reduce={'model':'PCA','kwargs':{'whiten':True}}` + `ndims=2`
  DROPPED the reduction (returned full-dim data) — the documented dict form pre-built the estimator
  and never injected `n_components=ndims`.
- R2/X1: `reduce()` returned a LIST for a single array when `ndims` was None or ≥ n_features,
  but a bare ndarray otherwise — inconsistent return type.
- R3: `describe(max_dims > n_features)` crashed. R6: `reduce(ndims=0)` silently returned `(n,0)`.
  X6: string `ndims` gave a cryptic `TypeError`.

**Fix:** ndims validated at entry (positive int; reject bool/non-int/<1). Canonical dict branch
injects `n_components=ndims` when the model class accepts it (`inspect.signature`). A user-supplied
`n_components` in kwargs still wins. Single-input reduce unwraps to a bare ndarray on both
early-return paths (`internal=True` still returns a list). describe guards empty component ranges.

- Repro (before): `hyp.reduce(iris, reduce={'model':'PCA','kwargs':{'whiten':True}}, ndims=2).shape` → (150,4/5).
- Verify (after): shape (150,2); numerically matches `PCA(n_components=2, whiten=True)`; return type stable.
- Red-team (B1): **SOLID** (part of a 708-passed run). n_components injection correct for 14
  reducers (PCA/TSNE/MDS/Isomap/FastICA/GaussianMixture/KernelPCA/TruncatedSVD/NMF/DictLearning/
  SpectralEmbedding/FactorAnalysis/LDA/UMAP); user-supplied `n_components` wins; PCA output matches
  sklearn (allclose); ndims validation rejects str/0/-1/bool/2.0 and accepts np.int64; return-type
  unwrap consistent (single→ndarray, internal→list, multi→list).

---

## B2 — input coercion + error quality (`7d71975b`)

**Defects:**
- P2: `hyp.plot(np.random.rand(50))` (1-D array) crashed `IndexError` at `helpers.py` `data[0][0]`.
- P3/X5: `hyp.plot([1.,2.,3.])` (flat numeric list) crashed though advertised; Series/tuple too.
- X2: list of numeric lists crashed (`.ndim` on a Python list).
- P1: hue length never validated → silent truncation or `IndexError`.
- X4: unknown align model → `AttributeError`, not a clear ValueError.
- P4/P5/X3: inf / all-NaN / empty gave ugly internal crashes.

**Fix:** `get_type` classifies ndarrays by `dtype.kind` (no `data[0][0]`); 1-D/empty no longer crash.
`format_data` coerces Series→ndarray, tuple→list, flat-numeric-list→one dataset, list-of-numeric-lists→arrays.
`plot` validates hue length against observation count. `align` raises a clear ValueError listing valid
aligner names. Per-dataset non-finite loop raises clear errors for inf, all-NaN, and empty (0-row).

- Repro (before): the five inputs above each crash cryptically.
- Verify (after): all coerce/validate correctly; string arrays still take the text path; clear errors.
- Red-team (B2): **DEFECTS FOUND → all fixed in `111775aa`.** Confirmed solid: get_type for
  1-D/empty/bool/2-D-str/bytes/obj arrays; `[1,2,3]`/`[[1,2,3]]`/`[np.array(...)]`→(3,1); tuple/Series;
  inf & all-NaN clear errors; hue accepts list/continuous/matrix/categorical/Series/single-obs.
  Three real gaps: (1) `align={'model':'Nope'}` (dict form) still hit the cryptic AttributeError;
  (2) 0-d array `np.array(5)` → `IndexError: tuple index out of range`; (3) scalar `hue='red'` →
  `len()` counted characters. Also surfaced a pre-existing PPCA EM infinite-loop hang on
  small/degenerate NaN data — capped in the same commit.

---

## B3 — manip / normalize / impute (`ca13d050`)

**Defects:**
- M3 (WRONG, most significant): default PPCA impute returned the latent PCA scores
  (`self.data @ self.C`) — it ROTATED the observed values and could DROP columns, violating
  "observed values preserved, shape preserved".
- M1 (crash): imputing an all-missing column → `IndexError` (sklearn drops all-NaN cols).
- M2 (wrong-number): ZScore/Normalize on a constant column → whole column NaN, no warning.
- M4 (UX): `manip()` lacked cross-module stage kwargs. M5: Smooth kernel_width edge cases raw scipy errors.

**Fix:** PPCA transformer reconstructs from the un-standardized model data and SPLICES — observed
(non-NaN) values preserved exactly, only NaNs filled, input shape preserved. sklearn imputers use
`keep_empty_features=True` + writable arrays. ZScore/Normalize guard constant columns
(`std==0 → 1`). `manip()` accepts normalize/reduce/ndims/align/cluster stage kwargs. Smooth raises
clear errors for kernel_width > n_rows and (savgol) kernel_width ≤ order.

- Repro (before): PPCA impute corrupts observed values / changes shape on rank-deficient data.
- Verify (after): observed preserved to 1e-9; filled values finite and sane; shape preserved.
- Red-team (B3): **DEFECTS FOUND → all fixed in `4fc2e1b0`.** Confirmed solid: observed preserved to
  1e-9 (full-rank, rank-deficient, AND reuse path); filled values sane (rank-deficient RMSE 0.31
  exploits structure, not garbage); all-missing column (sklearn→0, no read-only crash); constant-col
  ZScore/Normalize→0; manip stage kwargs; Smooth kernel_width errors. One **significant regression**:
  the splice left min_obs-dropped columns full of NaN → `hyp.reduce`/`hyp.plot` crashed with "Input X
  contains NaN" on sparse-column data. Also single-column PPCA → cryptic LinAlgError, and a
  reuse-path `assert ..., ValueError(...)` (raises AssertionError, stripped under -O). All fixed.

---

## B4 — pipeline / analyze / apply_model (`30725456`)

**Defects:**
- C1 (high): `analyze(cluster=...)` returned cluster LABELS, not the transformed data
  (contradicts the docstring). Same for cross-module `reduce/normalize/align(cluster=)`.
- C2: `normalize=False` + another stage + `return_model` → `NotFittedError` on `.transform`.
- C3: hard clusterers (DBSCAN/Agglomerative/OPTICS/MeanShift) crashed (no fit_predict fallback);
  KMeans-in-Pipeline silently returned the distance matrix, not labels.
- C4: `apply_model(model='KMeans', ndims=3)` crashed (n_components forced on KMeans).

**Fix:** analyze returns transformed data and recovers labels by re-applying the fitted non-cluster
steps and reading `model.named_steps['cluster']`. build_pipeline skips `normalize=False`. Hard
clusterers handled via `_step_fit_transform`/`_step_transform` (fit_predict/predict fallback).
class-path `n_components` injected only when the model's `__init__` accepts it (`inspect.signature`).

- Repro (before): analyze(cluster=) loses the data; apply_model(KMeans,ndims) crashes.
- Verify (after): transformed data numerically matches reduce-alone; labels aligned; hard clusterers run.
- Red-team (B4): **primary claims SOLID; convenience defects fixed in `9d5ce9d6`** (300 passed).
  Verified: analyze returns transformed data == reduce-alone (maxdiff 9.2e-16, single & multi-dataset);
  labels row-aligned; no-cluster path byte-identical to legacy; normalize=False skips the stage and
  transform-reuse gives no NotFittedError; n_components guard (KMeans-as-reduce no crash, PCA honors
  ndims); apply_model reuse round-trips. Defect: the documented label-recovery
  (`named_steps['cluster'].transform(data)`) raised NotImplementedError for the 3 hard clusterers with
  no out-of-sample predict (DBSCAN/Agglomerative/Spectral) — now returns stored fit labels on
  same-shape recovery; docstring clarified.

---

## B5 — align (`00607d26`)

**Defects:**
- R1 (high): `hyp.align` crashed by DEFAULT on datasets with different column counts — the canonical
  hyperalignment use case — via the format_data vstack NaN-check.
- R7: align silently trimmed the longer dataset on mismatched ROW counts, no warning.

**Fix:** the NaN check is per-dataset (no cross-dataset vstack), so mismatched column counts pass
through to the aligners. `trim_and_pad` warns when it trims rows.

- Repro (before): `hyp.align([A(50x10), B(50x8)])` crashes.
- Verify (after): hyperalign/SRM/procrustes run on mismatched columns; sane shapes, finite values; row-trim warns.
- Red-team (B5): _[VERDICT PENDING agent ae175fc6ac7b261a1]_

---

## B6 — animation (`176fbd7a`)

**Defects:**
- A1: `HyperAnimation.save('x.svg'/'x.png'/'x.apng')` crashed (delegated to mpl `Animation.save`,
  bypassing the extension dispatcher) though `save_path='x.svg'` worked — asymmetric.
- A2: `duration=0` / `frame_rate=0` → `ZeroDivisionError`. A3: `duration<0` cryptic.
- K2: the margin test was a FALSE POSITIVE — its `_cube_corner_pixels` helper ignored
  `set_box_aspect(1.125)` view scaling, projecting phantom off-canvas corners. The actual render was
  never clipped (118px margin over 30 frames). The TEST was fixed, not the source.

**Fix:** `HyperAnimation.save()` routes through `_save_animation` (extension dispatch) unless an
explicit `writer=` is given; fps derived from the animation interval. duration/frame_rate ≤ 0 raise
clear errors. The margin test measures rendered-ink extent in the mask's physical-pixel space.

- Repro (before): save('x.svg') crashes; duration=0 → ZeroDivisionError; margin test false-fails.
- Verify (after): .gif/.mp4 written with correct magic bytes; clear errors; margin test measures real ink.
- Red-team (B6): _[VERDICT PENDING agent ae175fc6ac7b261a1]_

---

## B7/B8 — text / streaming / predict / edge (`be0dcb5a`)

**Defects:**
- I2: streaming plot ignored canonical reduce `kwargs` (read only legacy `params`).
- I6: `load()` 'weights' docstring wrong (says 2 arrays, returns 36).
- X7: `predict(t=-3)` silently returned an empty (0,n) forecast; float t cryptic.
- I3/I4/I5: text2mat on flat string list / ragged list / empty list.

**Fix:** streaming reduce spec reads canonical `kwargs` (falls back to `params`). predict validates
the horizon: reject bool / int<1 / float with clear messages; datetime targets still pass. text2mat
narrowly suppresses only the sklearn `InconsistentVersionWarning` around the known-safe pretrained
model load. load docstring corrected.

- Repro (before): `predict(df, t=-3)` → empty; streaming kwargs ignored.
- Verify (after): clear horizon errors; datetime/positive-int work; streaming honors kwargs.
- Red-team (B5/B6/B7/B8): **primary claims SOLID; one minor gap fixed in `a81c7a91`** (581 passed).
  Verified: align no false-trim on equal-length data, mismatched row+col counts yield sane finite
  output, unknown-model ValueError lists all 6 real names; animation GIF (magic GIF89a) via Pillow +
  MP4 (ftyp) via ffmpeg, explicit `writer=` wins, `.xyz` fails loudly, fps derived = 30 = 1000/interval;
  streaming honors canonical `kwargs`; text2mat suppression scoped (no leak). Gap: predict's horizon
  check missed `np.True_` (np.bool_) and 0-d int arrays → misleading downstream message; and
  `resolve_t` used `assert` (stripped under -O). Both fixed.

---

## Red-team follow-up fixes (defects the independent audit surfaced)

Five independent red-team subagents (one per batch area) were each told to *prove the fixes wrong*
with real data. They confirmed the primary claims and surfaced a handful of real gaps — all now fixed
with regression tests:

| Commit | Area | What the red-team found |
|-|-|-|
| `4fc2e1b0` | B3 impute | **Regression**: the PPCA splice left min_obs-dropped columns NaN → `reduce`/`plot` crashed on sparse-column data. Densify dropped columns (observed-mean fill), keep observed exact, keep fully-missing rows NaN. Plus single-column PPCA → clear error (was LinAlgError); reuse-path `assert`→`raise`. |
| `9d5ce9d6` | B4 cluster | Documented label recovery raised NotImplementedError for hard clusterers (DBSCAN/Agglomerative/Spectral). Return stored fit labels on same-shape recovery; new-data still raises; docstring clarified. |
| `111775aa` | B2 input | `align={'model':'Nope'}` dict form still cryptic; 0-d array `IndexError`; scalar `hue='red'` char-count message; **PPCA EM infinite-loop hang** on degenerate NaN data (added `max_iter` cap + non-finite-diff break). |
| `a81c7a91` | B8/K1 | predict horizon missed `np.bool_`/0-d arrays; `resolve_t` used `assert` (stripped under -O); capitalized backend name (`'Plotly'`) treated as mpl backend. |

Each fix was reproduced before, verified after (numeric + the screenshots in `evidence/hardening/`),
and covered by new regression tests (24 added across the four commits).

## Full test suite

`pytest tests/` → **1454 passed, 4 skipped, 7 deselected, 0 failed (286s)** (includes the 24 new
red-team regression tests).

The deselected tests are the plotly image-export tests (`test_animation_export.py::test_plotly_*`
and `test_round3.py::test_*_svg_plotly`); they deadlock in the `kaleido` export subprocess in this
headless environment — a pre-existing infrastructure issue, not a regression from these changes.
They pass when run individually with a live kaleido/chromium.
