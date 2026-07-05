# Issues to Close on dev-1.0-refactor -> master Merge

This note catalogs all 67 currently-open GitHub issues on `ContextLab/hypertools`, triaged against the `dev-1.0-refactor` branch on 2026-07-04 (6-batch triage pass, `/tmp/savemovie/triage_batch1.md` through `triage_batch6.md`). Each issue was checked against live code/repros on `dev-1.0-refactor` and classified as ADDRESSED (the ask is satisfied), OBSOLETE (the ask targets API surface removed by design, e.g. public `DataGeometry`), STILL_BUG (the reported bug still reproduces), or NOT_ADDRESSED (the feature/design ask was never implemented). This file exists to guide a single triage sweep of GitHub issues once dev-1.0-refactor merges into master: section 2 and 3 below are safe to close at that point; section 4 should stay open.

## 1. Close when dev-1.0 merges to master

Every ADDRESSED or OBSOLETE issue from the triage. Safe to close on merge — the ask is either satisfied by the 1.0 rewrite or moot because the API surface it targeted no longer exists.

| # | title | ADDRESSED/OBSOLETE | justification + code evidence |
|-|-|-|-|
| 265 | Plotting animations in Jupyter incompatible with NumPy 2+ | ADDRESSED | `np.string_` check removed; `reduce/reduce.py:78,98` uses `isinstance(reduce, str)`; numpy>=2.0.0 pinned; `animate=True` runs clean on numpy 2.3.5 |
| 264 | Multiple hypertools figures in a for-loop don't update | ADDRESSED | `memoize` decorator removed entirely (zero grep hits repo-wide); loop repro shows distinct std per iteration |
| 257 | Sphinx documentation not building | ADDRESSED | RTD builds succeed; `docs/_build/html` fresh; `.readthedocs.yaml` valid; extensive 2.0 doc-rebuild commit history |
| 251 | Tests for backend management | ADDRESSED | `tests/test_backend.py` + `tests/plot/test_backend_renames.py` unit-test version-parsing/backend init without a live notebook |
| 244 | Better tests (check values, not just types) | ADDRESSED | `tests/manip/test_normalize_zscore.py` asserts `np.isclose`/`np.allclose` numeric outcomes; suite grew to 44 test files |
| 236 | Extract cluster/latent data from plot() result | ADDRESSED | `plot/plot.py:832-852` returns `{'fig','xform_data','animation','models'}` when `return_model=True` |
| 235 | animate=True doesn't work in Google Colab | ADDRESSED | `plotly_backend.py` `detect_environment()`/`resolve_backend()` route Colab/Kaggle to plotly; simulated-Colab repro produced a 900-frame animated Plotly figure |
| 224 | Documentation: reproduce plot() from constituent parts | ADDRESSED | `plot/plot.py` `return_model=True` docs/impl + `docs/auto_examples/plot_apply_model.py` gallery example document the reduce/align/cluster pipeline directly |
| 220 | matplotlib 3.0.0 axis-compression bug | ADDRESSED | upstream mpl bug fixed years ago; `pyproject.toml` requires `matplotlib>=3.8.0`; no compression observed |
| 217 | Return projection matrix for procrustes | ADDRESSED | `align/procrustes.py` `Procrustes(Aligner).fitter()` stores fitted `proj`; confirmed live `p.proj` populated after `fit_transform` |
| 212 | Dependencies and installation via pip | ADDRESSED | `pyproject.toml` splits base deps vs `[interactive]`/`[text]` extras; `plotly_backend.py:96` gives a `pip install hypertools[interactive]` hint |
| 211 | customize linewidth? | ADDRESSED | `plot/plot.py` `linewidth` kwarg (global + per-line list); confirmed live: `linewidth=[1,5]` -> actual widths `[1.0, 5.0]` |
| 208 | ImportError: numpy.core.multiarray failed to import | ADDRESSED | `pyproject.toml` requires `numpy>=2.0.0`/`scikit-learn>=1.4.0` (no hard `hdbscan` dep); clean import confirmed on numpy 2.3.5 |
| 207 | animation legend looks wrong (duplicated entries) | ADDRESSED | `plot/matplotlib_backend.py:562` sets `_trail_line.set_label('_nolegend_')`; confirmed live legend shows exactly `['1','2']` |
| 206 | handling extra keyword arguments (per-dataset lists) | ADDRESSED | `_shared/helpers.py:107-119` `parse_kwargs()` broadcasts/distributes list-valued kwargs across datasets; wired into both backends |
| 183 | Major version release (roadmap/meta-issue) | ADDRESSED | `align/common.py` `Aligner(BaseEstimator)` fit/transform/fit_transform + `apply_model`/`plot(return_model=True)` implement the fit-once-reuse and sklearn-API asks |
| 158 | memoize doesn't consider default args before caching | ADDRESSED | `memoize` removed entirely (it was the root cause of #264 per `tests/test_regressions.py`); the caching layer this issue asked to improve no longer exists |
| 143 | Gaussian Mixture Models as a cluster method | ADDRESSED | `cluster/cluster.py` `mixture_models` incl. `GaussianMixture`/`BayesianGaussianMixture`, `n_clusters` mapped to `n_components`; verified `cluster='GaussianMixture'` returns shape `(50,3)` |
| 142 | legend isn't updated when color kwarg is specified | ADDRESSED | live repro: legend swatches match `color=['red','green','blue']` exactly, no discrepancy |
| 141 | animation indexing error | ADDRESSED | `group` kwarg renamed to `hue` in 2.0; forced frame renders succeed with/without `hue`, no `IndexError` |
| 120 | save animations as GIFs | ADDRESSED | `plot/animate.py:31-32` uses `animation.PillowWriter` for `ext=='gif'`; real 255KB GIF produced in repro |
| 110 | plot.ly support (or other graphics backend) | ADDRESSED | `plot/plot.py` `backend="auto"`; `plotly_backend.py` `detect_environment`/`resolve_backend`; confirmed `backend='plotly'` returns a `plotly.graph_objs.Figure` |
| 104 | option to animate trajectories in serial or parallel | ADDRESSED | `matplotlib_backend.py` `update_lines_serial` + `animate in [True,'parallel','spin','serial']`; all three modes construct `FuncAnimation` successfully |
| 101 | support streaming data | ADDRESSED | `io/streaming.py` `is_stream()`/`plot_stream()` auto-detect iterators/`IterableDataset` (no flag, per issue's preferred design); repro streams a generator into `fig.stream_info` |
| 95 | support multiindex Pandas DataFrames | ADDRESSED | `datawrangler` unstack used throughout reduce/align/manip; `reduce`/`normalize`/`plot` all run cleanly on a genuine 2-level MultiIndex DataFrame |
| 94 | resample trajectory / hyp.tools.resample | ADDRESSED | `manip/resample.py` `Resample(Manipulator)` uses scipy pchip; `hyp.manip(data, model='Resample', n_samples=50)` upsamples 20->50 rows |
| 227 | DataGeometry-as-pipeline (reusable fitted vectorizer/models) | OBSOLETE | public `DataGeometry` removed by design (`datageometry.py:6-11`); the requested vehicle no longer exists |
| 204 | 1000 stars | OBSOLETE | celebratory GitHub-stars milestone note, not a bug or feature request |
| 193 | Make geo objects iterable/indexable | OBSOLETE | public `DataGeometry` removed in 1.0; no user-facing geo object left to make iterable |
| 164 | DataGeometry enhancements + arg-parsing helpers | OBSOLETE | predicated on a method-rich `DataGeometry` (`geo.plot/.reduce/.normalize/.align`); class removed by design, superseded by the `return_model=True` dict pattern |
| 161 | extend DataGeometry with transform/inverse | OBSOLETE | `DataGeometry` removed; `apply_model(return_model=True)` returns fitted sklearn objects with `.transform`/`.inverse_transform` directly |
| 159 | readthedocs API changes (describe_pca, tools.procrustes, DataGeometry wording) | OBSOLETE | targets pre-1.0 API surface (`describe_pca`, dotted `hypertools.tools.procrustes` path) that no longer exists |

## 2. Bugs fixed on this branch (dev-1.0-refactor)

Confirmed STILL_BUG at triage time; fixed during this pass. Safe to close on merge, referencing the fix below.

| # | title | fix |
|-|-|-|
| 259 | Importing hypertools mutates matplotlib rcParams | Removed the module-level `matplotlib.rcParams["pdf.fonttype"] = 42` from `plot/matplotlib_backend.py` (it ran at import time). The editable-PDF default (`pdf.fonttype`/`ps.fonttype` = 42) is now set inside `plot/backend.py`'s `manage_backend` scope, immediately after its rcParams snapshot — so it applies to hypertools' own saves but is restored when `plot()` returns, never leaking into the user's global config |
| 223 | get_proj AttributeError on 2D labeled plots (rotation/save) | Guarded `update_position()`'s `ax.get_proj()` call behind `hasattr(ax, 'get_proj')` and fixed the tuple-unpack mismatch between `annotate_plot()` (3-tuple for 2D / 4-tuple for 3D) and `update_position()` in `plot/matplotlib_backend.py` |
| 146 & 190 | n_clusters force-injected into DBSCAN/MeanShift/OPTICS (not in registry) | Replaced the hardcoded `model_name != "HDBSCAN"` exemption in `cluster/cluster.py` with a signature-based check (`inspect.signature(model)` for an `n_clusters` param) before defaulting `n_clusters` in, and registered `MeanShift`, `DBSCAN`, `OPTICS`, and `AffinityPropagation` so their string names resolve |
| 148 | show=False leaves figure registered in pyplot | Added `plt.close(fig)` in `plot/plot.py` when `show=False` and the user did not supply their own `ax` (and the figure is a matplotlib Figure), after the save — deregisters it from `Gcf` so it won't reappear via a later `plt.show()`/notebook `flush_figures`. The returned Figure/animation stay valid and savable |
| 214 | wiki-model docstring vs wiki_model key | Fixed `io/load.py` docstring (lines 102, 104) to read `wiki_model` (underscore), matching the actual `EXAMPLE_DATA` dict key |
| (non-issue regression) | reduce.py custom class/instance path raised UnboundLocalError | Initialized `model_params` in the `else` branch of `reduce/reduce.py` (previously undefined for non-str/dict `reduce=` args) — unblocks the "pass a custom sklearn-style class/instance" escape hatch that #162 (autoencoders) would need |

## 3. Leave open (not addressed by 2.0)

NOT_ADDRESSED issues, still-open STILL_BUG issues, and ADDRESSED-but-partial issues where the specific ask (not just the general area) remains unimplemented.

### Specific enhancement/design gaps

| # | title | why still open |
|-|-|-|
| 205 | support for multibyte (CJK) character labels | Still reproduces: matplotlib's default font ships no CJK glyphs (`Glyph ... missing from font(s)` warnings/tofu boxes); no `fontproperties`/font passthrough exists in `plot.py`/`matplotlib_backend.py` |
| 177 | Google Drive + npy/npz/pkl/csv/xls/xlsx loading | Partial: Drive-by-id and npy/npz/pkl/csv/tsv/txt/json/parquet/mat all work in `io/sources.py`, but `.xls`/`.xlsx` has no branch in `_parse_payload()` and neither `openpyxl` nor `xlrd` is a dependency |
| 174 | LDA/NMF as a `reduce()` algorithm | Cluster-side support is done (`cluster/cluster.py` `mixture_models`), but `reduce/reduce.py`'s `models` dict still has no LDA/NMF entry, so "reduce to n-dims via LDA/NMF" isn't wired in |
| 132 | reorder df columns by name across datasets | Still reproduces: `tools/format_data.py`/`df2mat.py` align DataFrames positionally, not by column name (silently misaligns same-named-but-reordered columns); the `align`/`manip` paths via `datawrangler` are unaffected, but `normalize`/`reduce`/`cluster` are not |
| 209 | use isinstance instead of type(x) is T | `type(x) is T` / `type(x) is not T` patterns remain across ~13 call sites: `cluster/cluster.py`, `plot/plot.py`, `plot/matplotlib_backend.py`, `reduce/describe.py`, `manip/resample.py`, `align/procrustes.py`, `align/common.py`, `align/hyperalign.py`, `align/srm.py`, `_shared/helpers.py` |
| 199 | tests for procrustes with non-default params | No test anywhere exercises `Procrustes`/`procrustes()` with `reflection=`, `scaling=`, `oblique=`, or `reduction=` — only default-parameter calls exist |
| 153 | align models in the `apply_model` pipeline | `apply_model()`'s registry covers reduce + cluster + mixture models but not align models (hyperalign/SRM/procrustes), so a reduce->align->reduce chain still fails with `ValueError: unknown model 'hyper'` |
| 154 | consolidate animate/style/labels into dict kwargs | `plot()`'s signature still uses ~15 flat top-level kwargs (`animate`, `chemtrails`, `precog`, `bullettime`, `duration`, `rotations`, ...) rather than the proposed `animate={}`/`style={}`/`labels={}` dicts; only `group`->`hue` rename landed |
| 138 | cluster() align/normalize/model kwargs | `cluster()` itself still has no `align=`/`normalize=` kwargs (only `model_params` via dict spec); the full normalize->reduce->align->cluster pipeline exists only at `plot()`'s orchestration level |

### Pure feature requests (unimplemented, non-whimsical)

| # | title | why still open |
|-|-|-|
| 225 | animate uncertainty (HOP-style bootstrap animation) | No bootstrap/uncertainty/HOP code anywhere in the codebase |
| 198 | more text models via gensim | `text2mat.py` still only registers CountVectorizer/TfidfVectorizer + LDA/NMF; no gensim import anywhere |
| 187 | text download function (twitter/wikipedia scraper) | No `textlookup`/twitter function; `io/sources.py` only fetches known data sources by name/URL |
| 186 | 3D stream graphs (d3-style) | No streamgraph code; `io/streaming.py`'s "streaming" means live data consumption, a different concept |
| 185 | data formatting via pliers | No `pliers` import anywhere; `format_data.py` still only handles text/DataFrame/array via its own helpers |
| 169 | Kalman filter for missing/future data | No kalman code or `hyp.predict` function; missing data still handled only via PPCA (can't fill fully-missing rows or forecast) |
| 162 | autoencoders as a reduce algorithm | `reduce.py`'s model registry is sklearn-only; no torch/keras/autoencoder integration (the custom-class escape hatch is now at least functional post-fix, see section 2) |
| 127 | per-matrix chemtrails/precog/bullettime | These remain single scalar bools in `plot.py`/`matplotlib_backend.py`, not per-dataset lists; a list value is silently truthy-broadcast to all datasets |
| 123 | 2D animations | `matplotlib_backend.py` has a hard `assert x[0].shape[1] == 3` requiring 3D data for animation |
| 116 | kaggle/538 dataset loading functions | No `source='kaggle'`/`'538'` parameter anywhere in `io/load.py` or `io/sources.py` |
| 109 | triangular mesh support | No trimesh/mesh code anywhere in the package |
| 108 | kernel density plots in 2D/3D | No kde/`KernelDensity` code anywhere; still an open design question (3D density rendering approach) |
| 103 | adjustable label opacity | `annotate_plot()`'s label bbox alpha is hardcoded (0.5); no user-facing opacity kwarg |
| 100 | colorbar option | No `colorbar` kwarg in `plot()`; continuous-hue coloring exists but isn't wired to a `ScalarMappable`/`fig.colorbar()` |

### Whimsical / pie-in-the-sky

| # | title | why still open |
|-|-|-|
| 163 | add a soundtrack to animations | No audio/soundtrack code anywhere; a fun but genuine unimplemented ask |
| 191 | replace matplotlib with ipyvolume | Never adopted; 2.0 added a Plotly backend instead, addressing the same interactivity desire via a different, actively-maintained library |
| 113 | grand challenge: streaming brain decoding (OpenBCI) | Explicitly gated on 4 sub-pieces; only streaming (#101) is done — decoding, interactive labels, and live device reading are not |
| 112 | on-the-fly decoding (nearest-event highlighting) | Sub-piece of #113's grand challenge; no decoding mode exists |
| 111 | on-the-fly interactive labels (spacebar prompt) | Sub-piece of #113's grand challenge; no keypress/labeling UI exists |
| 130 | LSL data stream support (OpenBCI/Muse) | No `pylsl`/LSL support; generic streaming (`io/streaming.py`) could wrap it manually, but no dedicated adapter ships |

(Note: #204 "1000 stars" is a similarly non-actionable curiosity but is classified OBSOLETE, not open — see section 1.)
