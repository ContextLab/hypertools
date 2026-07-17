# Changelog

## 1.0.0 (in development)

HyperTools 1.0 is a ground-up modernization of the toolbox. The familiar
one-call API (`plot`, `analyze`, `reduce`, `align`, `cluster`, `normalize`,
`describe`, `load`) is preserved, but the internals, packaging, and feature
set are new.

### Architecture

- **Package restructure:** the old monolithic `tools` module is split into
  focused subpackages -- `reduce/`, `cluster/`, `align/`, `manip/`, `io/`,
  `predict/`, `impute/`, `plot/`, and `core/` -- all built on a shared
  stack -> fit-once -> unstack model-application core (`hyp.apply_model`,
  backed by [pydata-wrangler](https://github.com/ContextLab/data-wrangler)).
- **Canonical pipeline order:** every dispatcher composes stages in the
  same order (manip -> normalize -> reduce -> align -> cluster), and
  `return_model=True` returns fitted models/`hyp.Pipeline` objects that can
  be replayed on new data via `pipeline=`.
- **Return types:** `hyp.plot` returns a matplotlib `Figure` (a
  `HyperAnimation` when `animate=` is used, or a plotly `Figure` with
  `backend='plotly'`); `hyp.load` returns raw data. The 0.x `DataGeometry`
  ("geo") container is retired to an internal unpickle-only shell so
  **pickle-format geo files saved by hypertools ≥0.8 still load** (returning
  their raw data). Older **pre-0.8 `deepdish`/HDF5-format** geo files cannot
  be read under the required NumPy 2 and must be converted once in a
  throwaway `numpy<2` environment (`hyp.load` detects them and explains how;
  see the README "Legacy data" note).
- **Packaging:** `pyproject.toml`-based packaging, Python 3.10-3.13, a
  small fast-importing base install (heavy dependencies load lazily;
  `import hypertools` is roughly 3.5x faster than 0.8.x), and optional
  extras (`interactive`, `text`, `predict`, `predict-hf`, `io`,
  `density3d`, `torch`, `kaggle`, `lsl`, `gensim`, `dev`).

### New features

- Interactive plotly backend (`backend='plotly'`; `backend='auto'` selects
  plotly automatically on Google Colab/Kaggle), visually matched to the
  matplotlib backend.
- New animation styles (`'spin'`, `'serial'`, `'window'`, `'morph'`), 2-D
  animation support, and per-dataset `chemtrails`/`precog`/`bullettime`
  trails.
- Hull surfaces (`surface=`), density shading (`density=`), colorbars,
  multicolored lines via continuous/matrix `hue`, nested-list input, and
  automatic MultiIndex DataFrame expansion.
- Mixture-model ("soft") clustering (`GaussianMixture`,
  `BayesianGaussianMixture`, `LatentDirichletAllocation`, `NMF`).
- `hyp.manip` (Normalize/ZScore/Smooth/Resample) with chaining,
  `hyp.predict` timeseries forecasting (Kalman, ARIMA, GP, autoregression,
  Laplace, Chronos), and `hyp.impute` missing-data imputation (PPCA,
  Kalman, sklearn imputers).
- Optional torch-backed autoencoder reducers, gensim text
  vectorizers/semantic models, Lab Streaming Layer input
  (`hyp.io.lsl_stream`), and new `hyp.load` sources (Kaggle, Hugging Face,
  Google Sheets/Drive, Dropbox, URLs, and more local formats).
- Reproducibility via a top-level `random_state=` on
  `reduce`/`cluster`/`analyze`/`plot`.

### Removed / changed behavior

- Retired legacy arguments now raise errors instead of being silently
  accepted: `group=` (use `hue=`), `model=`/`model_params=` (use
  `reduce=`), and `align=True`/`align(method=...)` (use `align='hyper'`,
  `align='SRM'`, etc.). `cluster`'s `ndims=` is only a passthrough to
  `reduce=` and warns if passed without it.
- Plotting no longer mutates global matplotlib settings; the unreliable
  result cache was removed; HDBSCAN comes from scikit-learn instead of the
  external `hdbscan` package.

### Release audit (2026-07)

Before release, the codebase, documentation, examples, and tutorials were
red-teamed in a 46-unit audit that filed 708 findings (691 confirmed by an
independent verifier). The confirmed code findings were fixed in waves
(350+ fixes merged as of this entry), including these criticals:

- `hyp.load('sotus')` returns the full 29-speech State of the Union corpus
  again (the hosted corpus had been loading incompletely).
- `hyp.align` preserves each dataset's row order (aligned outputs are no
  longer returned with scrambled rows).
- `hyp.manip` smoothing runs per dataset: `Smooth` kernels no longer bleed
  across dataset boundaries when given a list.
- The Kalman forecaster (`hyp.predict(..., model='Kalman')`) actually
  learns its dynamics model instead of filtering with default parameters.
- CSV/TSV parsing bugs in `hyp.load` were fixed, and `hyp.save` writes
  atomically and format-aware.
- `import hypertools` no longer crashes under unusual
  backend-related environment variable configurations.
- Plotting nested lists of datasets (`hyp.plot([[a, b], [c]])`) works
  correctly again.

Docs, README, examples, and tutorials were then re-verified by executing
them against the fixed code.

## 0.8.x and earlier

See [RELEASE_NOTES_0.8.1.md](RELEASE_NOTES_0.8.1.md) and the
[GitHub releases page](https://github.com/ContextLab/hypertools/releases).
