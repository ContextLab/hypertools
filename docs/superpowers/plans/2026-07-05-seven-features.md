# Seven Long-Standing Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address GH issues #95 (MultiIndex DataFrames), #100 (colorbars), #109→hull surfaces, #108/#191 (subtle density shading), #127 (per-dataset animation trail styles), #142 (legend/color bug), #177 (Drive/file-format loading), with screenshot+numeric evidence posted to PR #272.

**Architecture:** All features integrate into the existing 1.0 design language: `hyp.plot` kwargs flow through `plot/plot.py` and dispatch to BOTH rendering backends (`plot/matplotlib_backend.py`, `plot/plotly_backend.py`); pure-geometry helpers live in small internal modules with unit tests; `hyp.load` extensions land in `io/sources.py`/`io/load.py`. No new heavyweight deps (no trimesh, no mayavi/ipyvolume — plotly IS the answer to #191); scikit-image optional for 3D mpl density.

**Tech Stack:** numpy/scipy (ConvexHull, gaussian_kde, splines), matplotlib Poly3DCollection, plotly Mesh3d/Volume/Contour, pandas MultiIndex, requests (Drive interstitial), pandas read_excel + openpyxl.

## Global Constraints

- Branch `dev-1.0-refactor`; PR #272 into `dev-1.0`. NEVER touch master; NEVER merge the PR.
- Python: ALWAYS `/Users/jmanning/hypertools/.venv/bin/python`, `MPLBACKEND=Agg`.
- Every graphical feature must work in BOTH backends (`backend='matplotlib'` and `backend='plotly'`) with similar visual quality, static AND animated where applicable.
- No mocks in tests — real renders (export PNG, assert on pixels/artists/traces), real network for load tests (large-file live test behind `@pytest.mark.bigdata`, deselected in CI).
- New plot kwargs follow existing patterns: bool-or-dict options (like `predict` accepts str/dict), validated early with clear ValueErrors.
- Local pytest MUST deselect kaleido-deadlock tests: `--deselect tests/test_animation_export.py::test_plotly_gif_export --deselect tests/test_animation_export.py::test_plotly_spin_gif_export --deselect tests/test_animation_export.py::test_plotly_mp4_export --deselect tests/test_animation_export.py::test_plotly_spin_gif_preserves_realtime_duration_export --deselect tests/test_round3.py::test_static_svg_plotly --deselect tests/test_round3.py::test_animated_svg_plotly` (they pass in CI). NOTE: plotly PNG export via kaleido from a plain script worked fine in this session's research — the deadlock risk is specific to those pytest cases; scripts may export PNGs.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Docs: update `docs/api.rst`-equivalent + examples gallery when API changes; `PATH=".venv/bin:$PATH" make -C docs html`.
- Evidence PNGs for the PR go in `docs/images/v1.0-seven-features/` (committed, referenced via raw.githubusercontent URLs).
- Research prototypes (COPY algorithms from here, they are verified):
  `/private/tmp/claude-501/-Users-jmanning-hypertools/7e6531b3-066a-4ce2-b1f6-7c07c5e87b15/scratchpad/surface_research/` (meshutil.py, mpl_proto2.py shade(), plotly_proto.py, proto_2d.py)
  `/private/tmp/claude-501/-Users-jmanning-hypertools/7e6531b3-066a-4ce2-b1f6-7c07c5e87b15/scratchpad/density_research/` (common.py, mpl_2d.py, mpl_3d.py, plotly_tuned.py)

---

### Task 1: Fix `colors=` no-op (resolves #142 follow-up) 

**Files:** Modify `hypertools/plot/plot.py:465-472`; Test `tests/test_plot_colors142.py` (new).

Issue #142's original symptom (legend swatches not tracking explicit colors) is ALREADY FIXED in the refactor (verified with renders in both backends). But verification exposed: `colors=` is only read inside `if color is not None:` — `colors=['red','green','blue']` alone is silently ignored (falls back to hls). 

**Steps:**
- [ ] Failing tests: (a) mpl static — `hyp.plot([d1,d2,d3], colors=['red','green','blue'], legend=['a','b','c'], show=False)`, assert `fig.axes[0].get_lines()[i].get_color()` == pure red/green/blue AND legend handle colors match; (b) same with `color=`; (c) plotly backend — trace line colors match; (d) `color` and `colors` both given → `colors` wins or warns (match existing conflict-warning style at plot.py:616); (e) animated mpl — line colors honored.
- [ ] Fix: hoist `colors` handling out of the `if color is not None:` block (treat `colors` as alias; keep precedence consistent with docstring plot.py:100).
- [ ] Run new tests + `tests/test_plot*.py`; commit `fix(plot): colors= kwarg was a no-op unless color= also passed (GH #142)`.
- [ ] Evidence: render before/after PNGs to `docs/images/v1.0-seven-features/issue142_colors_{before,after}.png` (before = stash fix or reproduce from git show). Add #142 to `notes/issues-to-close-on-merge.md` fixed table (original symptom fixed + alias bug fixed).

### Task 2: `hyp.load` completion (#177)

**Files:** Modify `hypertools/io/sources.py` (+`io/load.py` obsolete cookie handler), `pyproject.toml`; Test `tests/test_load_sources.py` (extend).

Already working: Drive URL/id shapes, Dropbox, HF, npy/npz/csv/tsv/txt/json/parquet/mat/pickle, content sniffing. Gaps to close:
- [ ] **Large-file interstitial**: in `_fetch_bytes` (sources.py:194-217), when a Drive request returns the virus-scan HTML page, parse `<form action="https://drive.usercontent.google.com/download">` + hidden inputs (id/export/confirm/uuid) and re-GET with those params (verified live against public 498MB file `1l_5RK28JRL19wpT22B-DY9We3TVXnnQQ` → Content-Length 498,881,336). Remove the obsolete `download_warning` cookie handler (load.py:326-332). Tests: parse the REAL captured interstitial HTML (save fixture from a live request at implementation time, commit it under tests/data/) + live end-to-end `@pytest.mark.bigdata` test with the 498MB file (register marker in pyproject; deselect via `-m "not bigdata"` default addopts so CI skips).
- [ ] **.xlsx/.xls**: add to `_parse_payload` via `pd.read_excel` (engine auto). Add `openpyxl` to project deps (dev + an `io` extra); `.xls` needs `xlrd` — support with friendly ImportError if missing (same pattern as Kalman's optional-dep error). Test: write a real .xlsx with pandas, load it back through `hyp.load`.
- [ ] **Google Sheets**: rewrite `docs.google.com/spreadsheets/d/<id>` URLs to `/export?format=csv` in `_extract_drive_id`/source resolution so Sheets load as CSV. Live test with a public sheet if one is findable in ≤10 min; otherwise unit-test the URL rewrite and note it.
- [ ] **Remote pickle policy**: for remote (non-builtin-example) sources, unpickling emits a `UserWarning` ("arbitrary code execution — pass trust=True to silence"); `trust=True` kwarg threads from `load()`. Remote `.npy/.npz` use `allow_pickle=False` unless `trust=True`. Builtin EXAMPLE_DATA names exempt. Tests for warning presence/absence + npy behavior (real files served via `python -m http.server` in a fixture — real IO, no mocks).
- [ ] Full `tests/test_load_sources.py` + suite; commit; evidence = pytest output + interstitial live-download log (bytes count) captured to the PR notes.

### Task 3: Colorbars (#100)

**Files:** Modify `hypertools/plot/plot.py` (kwarg + validation + mpl draw), `hypertools/plot/plotly_backend.py`, `hypertools/plot/colors.py` (expose value↔color mapping); Test `tests/test_colorbar.py` (new).

**API:** `hyp.plot(..., colorbar=True)` or `colorbar={'label': str, 'ticks': [...], 'location':...}`.
- Continuous `hue` (the `mat2colors` continuous path + `_multicolor_line_colors`): colorbar maps the hue value range through the SAME palette used for the lines (build `ScalarMappable` from the palette as an mpl `ListedColormap` + `Normalize(vmin,vmax)` over hue values). This answers Jeremy's issue comment: values come from the hue variable.
- Discrete groups (cluster / list-of-datasets / categorical hue): discrete colorbar (`BoundaryNorm`, one segment per group) with tick labels = legend labels / group names (default `1..n` if unnamed) — the `groupvals` story from the issue comment.
- `colorbar=True` with no color mapping at all (single dataset, no hue/cluster): ValueError with a clear message.
- plotly: continuous → set `marker.colorbar`/use a hidden colorbar trace with matching colorscale (`sns.color_palette` → plotly colorscale conversion helper in colors.py); discrete → colorbar with `tickvals` at segment centers, `ticktext` = labels.
- Interaction with legend: both may coexist; reuse `_fit_right_legend` so nothing is cut off (colorbar takes the outer-right slot; verify no overlap by rendering).

**Steps:**
- [ ] Failing tests: continuous mpl (colorbar exists, its cmap colors sampled at min/max equal first/last line-segment colors — numeric assert), discrete mpl (n segments, tick labels match legend), plotly continuous (trace/layout has colorbar with matching colorscale endpoints), plotly discrete, error case, colorbar+legend coexistence (render, assert both artists + no overlap via bbox check), animated mpl with colorbar (present, static across frames).
- [ ] Implement; render evidence PNGs (`colorbar_{continuous,discrete}_{mpl,plotly}.png`) into evidence dir; VIEW them (Read tool) before declaring done.
- [ ] Suite + commit `feat(plot): colorbar support for continuous hue and discrete groups (GH #100)`.

### Task 4: Mesh utilities (#109 groundwork)

**Files:** Create `hypertools/plot/meshutil.py`; Test `tests/test_meshutil.py` (new).

Pure-geometry module, verified algorithms — COPY from research prototype `surface_research/meshutil.py` and `proto_2d.py`:
- `smooth_hull_3d(points, rounds=3, taubin_iters=8, lam=0.5, mu=-0.53, pre_inflate=1.15) -> (verts, faces)`: ConvexHull → orient simplices outward via `hull.equations` → pre-inflate hull verts about centroid → INTERLEAVED [midpoint 1→4 subdivision + Taubin smoothing] per round (interleaving avoids pinched creases).
- `smooth_hull_2d(points, samples_per_edge=20) -> (n,2) closed outline`: ConvexHull vertices → centripetal Catmull-Rom (α=0.5) closed spline.
- `face_normals(verts, faces)`, `blinn_phong_colors(verts, faces, base_rgb, view, lightdir=None, ambient=0.45, diffuse=0.55, fill=0.25, specular=0.30, shininess=48) -> rgba per face` (from mpl_proto2.py `shade()`).
- `backface_cull(verts, faces, view_vector) -> face mask` (the mpl z-sorting fix).

**Steps:**
- [ ] Failing tests: 3D — output faces all outward-oriented (signed volume > 0; normals·(centroid-facecenter) < 0), ≥96% of input points inside (use Delaunay-based containment on the CONVEX pre-smoothed hull for exactness + distance-tolerance check on smoothed mesh), vertex/face counts scale 4^rounds, runtime <50ms for 200 pts/rounds=3 (assert loosely <500ms), degenerate inputs (coplanar 3D points, <4 points) raise clear ValueError; 2D — closed curve, 100% hull-vertex containment, C1 continuity (finite-diff tangent continuity at knots), collinear-points error.
- [ ] Implement + suite + commit `feat(plot): mesh utilities for smooth convex-hull surfaces (GH #109)`.

### Task 5: Surface rendering (#109) — both backends + animation

**Files:** Modify `hypertools/plot/plot.py` (new `surface` kwarg), `hypertools/plot/matplotlib_backend.py`, `hypertools/plot/plotly_backend.py`; Test `tests/test_surface.py` (new).

**API (dict-controlled surface properties, per Jeremy's spec):** `hyp.plot(data, surface=True)` or `surface={'alpha':0.6, 'color':None (inherit dataset color), 'lighting':{'ambient':0.45,'diffuse':0.55,'specular':0.30,'shininess':48,'fill':0.25}, 'smoothing':3 (rounds), 'pre_inflate':1.15, 'keep_points':True}` — list-of-dicts/bools for per-dataset control (broadcast a single value). 2D data → smooth filled hull shape; 3D → lit smooth surface. Validated early: unknown keys ValueError.
- mpl 3D: `Poly3DCollection(verts[faces[cull_mask]], facecolors=blinn_phong_colors(...), edgecolors='none', linewidths=0, shade=False)`; key light = camera +0.7·up −0.5·right; translucent → antialiaseds on + culling (culling is REQUIRED — interior-face cracks otherwise). Recompute cull on `view_init` changes during animation (each frame: remove & re-add collection).
- mpl 2D: `ax.fill(outline, alpha, color, zorder below lines)`.
- plotly 3D: `go.Mesh3d(i/j/k, flatshading=False, lighting=dict(ambient=0.45,diffuse=0.6,specular=0.25,roughness=0.35,fresnel=0.15), lightposition=dict(x=2.5,y=-1.5,z=3.0))`; translucent: keep full mesh, document mild WebGL artifacts, prefer opacity≥0.95 default (default alpha 0.6 mpl / 0.95 plotly? NO — keep one default 0.6 both; note artifacts in docstring).
- plotly 2D: `go.Scatter(fill='toself', line smooth outline)`.
- Animation (both backends): recompute hull mesh per frame from the visible window (`animate='parallel'` window / current morph state); mpl swap collection per frame; plotly full x/y/z/i/j/k per `go.Frame`. Trail styles apply to lines/points, surface follows the full current window.

**Steps:**
- [ ] Failing tests: 3D mpl static (collection exists, faces>0, per-face colors vary → shading active, all verts within data bbox*1.3), 2D mpl (fill patch present, path closed), per-dataset dict list honored (different alphas measurable), plotly Mesh3d trace fields (i/j/k lengths equal, lighting dict round-trips), plotly 2D toself trace, animated mpl (FuncAnimation runs 5 frames, collection swapped each frame, vertex counts may differ), animated plotly (frames carry Mesh3d), invalid key ValueError, 1D data ValueError.
- [ ] Implement; render evidence PNGs static 2D+3D both backends + 3 animation frames; VIEW all.
- [ ] Suite + commit `feat(plot): smooth convex-hull surfaces with lighting, dict API (GH #109)`.

### Task 6: Surface morph gallery demo (#109 showcase)

**Files:** Create `examples/animate_surface_morph.py`; Modify docs gallery config if needed.

- [ ] New demo modeled on `examples/plot_shape_morph.py` (Hungarian-matched morphs through the shapes zoo) but rendering the SMOOTH HULL SURFACE (surface=True path, mpl backend for the gallery mp4) morphing between shapes, points hidden or tiny, camera spinning. Reuse its normalize/sample/match/segment code; per-frame recompute mesh from interpolated cloud. Verify: extract ≥4 frames as PNGs, VIEW them (smooth surfaces, no cracks/black faces, fills frame, no clipping).
- [ ] Gallery build for just this example; commit `docs(examples): surface-morph demo showcasing hull surfaces (GH #109)`.

### Task 7: Density shading (#108/#191)

**Files:** Create `hypertools/plot/density.py`; Modify `plot.py` (`density` kwarg), both backends, `pyproject.toml` (optional `density` extra: scikit-image); Test `tests/test_density.py` (new).

**API:** `density=False` default; `density=True` or `{'alpha':0.2 ('subtle' default), 'levels':3 (3D iso count), 'grid':50, 'per_group':True}`. Colors inherit group/dataset colors.
- 2D mpl: per-group `gaussian_kde` on 200×200 grid (15% pad) → `imshow` with `LinearSegmentedColormap` alpha ramp (r,g,b,0)→(r,g,b,max_alpha=0.2), `interpolation='bilinear'`, zorder below data. (contourf REJECTED: hard boundary.)
- 3D mpl: `skimage.measure.marching_cubes` (optional dep) iso-surfaces at 10/35/65% max density, `Poly3DCollection(..., alpha=0.03/0.05/0.07, shade=False)`; without scikit-image → warn + scatter-fog fallback (`kde.resample(4000)`, s=6, alpha=0.03, depthshade=False).
- plotly 3D: `go.Volume(isomin=0.1, isomax=1.0, surface_count=10, opacity=0.2, opacityscale=[[0,0],[0.3,0.3],[1,1]], colorscale solid group color, showscale=False, hoverinfo='skip')` — transparency ONLY via opacity/opacityscale (rgba colorscale invisible in export).
- plotly 2D: `go.Contour(contours=dict(coloring='heatmap', showlines=False), colorscale=[[0,'rgba(r,g,b,0)'],[1,'rgba(r,g,b,0.30)']], line_width=0, showscale=False)`.
- Animation: density computed ONCE from full data as a static background (KDE eval 536ms @50³ ≫ 33ms frame budget — measured); frames animate lines/points above it.

**Steps:**
- [ ] Failing tests: 2D mpl (image artist present, alpha ramp max ≤ requested, kde peak location ≈ data mean within tolerance — numeric), 3D mpl with skimage (3 collections at expected alphas) AND without (fog fallback + warning; simulate absence via monkeypatch of import? NO MOCKS — run a subprocess with `PYTHONPATH` sans skimage or use importlib to hide only if truly uninstalled; skimage will be an optional extra so base venv state decides which real branch tests run — write both tests with skip conditions on actual availability), plotly Volume trace params, plotly Contour params, per-group colors distinct, `density=True` on animated plot (background present in figure, not in per-frame artists), default off (no density artists), alpha honored numerically.
- [ ] Implement; evidence PNGs 2D+3D × both backends; VIEW (subtlety check: data clearly dominant).
- [ ] Suite + commit `feat(plot): subtle KDE density shading, off by default (GH #108, #191)`.

### Task 8: Per-dataset trail styles (#127)

**Files:** Modify `hypertools/plot/plot.py` (accept lists for `chemtrails/precog/bullettime`), `hypertools/plot/matplotlib_backend.py` (thread per-dataset flags through `_draw`/`animate_plot3D`/updaters; unify closure+fargs), `hypertools/plot/plotly_backend.py` (per-dataset trail traces + per-frame branches); Test `tests/test_animation_styles.py` (extend/new).

- Broadcast scalar→list AFTER cluster/hue reshaping (dataset count changes at plot.py:544-706); validate len==n_datasets else ValueError naming actual counts.
- mpl: index flags per dataset in `update_lines_parallel` loop (matplotlib_backend.py:423-442); trail artist creation per-dataset (:541-575).
- plotly: per-dataset trail trace mask (`:249-250` all-or-nothing today), per-frame branch `:685-696`.
- Conflicting flags for same dataset (precog+chemtrails) already mean "both/full" — keep semantics per existing `:427-435` logic, now per dataset. `animate` MODE (parallel/spin/serial) stays global — document why in docstring (one camera/one frame loop).
- [ ] Failing tests: mixed styles mpl (dataset 0 chemtrails only: trail artist data ends at current index; dataset 1 precog: future segment present — assert on artist xdata at a fixed frame), plotly frames (trail trace lengths differ per dataset per frame), scalar broadcast equivalence (list [True,True] == True output), bad length ValueError, spin/serial modes unaffected.
- [ ] Implement both backends; render 3-frame evidence PNG grid (one dataset chemtrails, one precog, one bullettime) for BOTH backends; VIEW.
- [ ] Suite + commit `feat(animate): per-dataset chemtrails/precog/bullettime (GH #127)`.

### Task 9: MultiIndex DataFrames (#95)

**Files:** Modify `hypertools/tools/format_data.py` or new `hypertools/plot/multiindex.py` (expansion helper), `hypertools/plot/plot.py`; Test `tests/test_multiindex.py` (new).

**Design (Jeremy's spec from issue comments):** When an input DataFrame has a row MultiIndex (nlevels ≥ 2), `hyp.plot` expands it BEFORE backend dispatch into: leaf datasets (one per unique full index combo, linewidth=1) + group-average datasets per level (level-k mean trajectories get linewidth=k+1), with `alpha = 1/(level+1) + 0.2` — leaves are level-0 ⇒ alpha≈1.0? NO: per spec alpha=1/(level+1)+0.2 with level counted so HIGHER levels are MORE opaque: leaves alpha=1/(nlevels)+0.2 … top-level mean alpha=1/(1+1)+0.2=0.7 — RESOLVE precisely: spec says "less transparent as you went up levels"; implement `alpha(level_from_top) = min(1.0, 1/(depth_below_top+1) + 0.2)` so top-level averages are most opaque; document the formula in the docstring and test it numerically. Colors: ALL leaves+averages sharing the same top-level index share one color (palette indexed by unique top-level values, in order of appearance); linestyle args of length = n unique top-level indices are cycled per top-level group. The averaging assumes rows align by position within each leaf group (mean over group at each timepoint index); groups with unequal lengths → mean over the overlapping prefix + warn.
- Expansion output: list of arrays + per-dataset style overrides (linewidth/alpha/color-index/label) that both backends already understand (reuses per-dataset kwargs machinery from Task 8 + existing color pipeline). Legend: one entry per top-level index (leaf/mean traces of that group share color; only the top-level mean carries the legend label, others `_nolegend_`).
- Works static+animated, both backends, since expansion happens upstream of drawing. `hue`/`cluster` + MultiIndex → warn that MultiIndex grouping takes precedence (or error if truly conflicting).
- `hyp.analyze`/manip pipeline: expansion happens after analyze (normalize/reduce/align applied to leaf datasets independently? NO — apply pipeline to leaf datasets, then compute means on TRANSFORMED trajectories so averages live in the reduced space).
- [ ] Failing tests: 2-level MultiIndex (4 subjects × 2 conditions): expect 4 leaf + 2 condition-mean + is there a grand mean? spec: "level 1 averages, level 2 averages" — with 2 levels produce leaf(lw=1)+level-1-means(lw=2)+top-level... define: means per non-leaf level grouping — for levels (cond, subj): leaves are (cond,subj); level-1 groups = unique cond values → cond means lw=2. Grand mean NOT included (no level above top). Numeric asserts: line count, each mean trace equals np.mean of its members (exact), linewidths [1,1,1,1,2,2], alphas per formula, colors: all traces of cond A share color != cond B, legend labels = unique cond values only, 3-level case (2×2×3) counts, linestyle list len==n_top cycles correctly, unequal group lengths → prefix mean + warning, plotly parity (trace widths/opacities), animated run smoke, single-level df unchanged behavior (regression: existing tests untouched).
- [ ] Implement; evidence PNG static 3D mpl + plotly of a synthetic 2-level dataset; VIEW (visual grouping obvious: thick opaque means over thin translucent members).
- [ ] Suite + commit `feat(plot): MultiIndex DataFrames — leaf traces + per-level averages w/ level styling (GH #95)`.

### Task 10: Visual review wave + fixes

- [ ] Dispatch FRESH review subagents (not the implementers) per graphical feature: each gets the evidence PNGs + instructions to hunt rendering bugs (cut-off elements, jaggies, z-fighting, overlap, duplicate legend entries, washed-out or overpowering density, hull cracks). Grid of configs: {2D,3D} × {mpl,plotly} × {static,animated} × {surface,density,colorbar,multiindex,trails}.
- [ ] Fix everything found (fix subagents), re-render, re-review until clean.

### Task 11: Docs, suite, CI, PR evidence

- [ ] Docstrings complete for all new kwargs; README feature bullets; examples: extend `plot_hue.py`-style docs or add `plot_surface.py`, `plot_density.py`, `plot_colorbar.py`, `plot_multiindex.py` gallery examples (small/fast) + Task 6's animated demo; docs build green.
- [ ] Full suite (with local deselects) green; ruff/lint if configured; push; CI 12/12 green.
- [ ] PR #272: one comprehensive comment (sections per issue: what changed, numeric evidence, before/after or feature PNGs via committed raw URLs) + body update (issues-to-close list gains #95 #100 #108 #109 #127 #142 #177 #191 as applicable — #191 closes as "plotly backend supersedes ipyvolume proposal"). Update `notes/issues-to-close-on-merge.md`.
