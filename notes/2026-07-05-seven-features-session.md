# Session notes: 7 long-standing feature requests (2026-07-05)

## Goal (Stop-hook /goal active)
Address ALL of these and update PR #272 with EVIDENCE (screenshots + numerical results):

1. **#95 MultiIndex DataFrames** — level-based line styling. Jeremy's spec from issue comments:
   - linewidth=1 on level-1 lines; group averages get thicker lines (level+1 width)
   - alpha = 1/(level+1) + 0.2 (less transparent higher up)
   - colors determined by HIGHEST-level index (all data + averages sharing top-level index share color/style)
   - linestyle args have length = number of unique highest-level indices
2. **#100 Colorbars** — `hyp.plot(data, colorbar=True)`; need value-mapping story (groupvals / hue values)
3. **#109 → redirect** — NOT sliding-window meshes. Point cloud → convex hull → smooth surface (3D, lighting/shading, dict API for surface props) or smooth shape (2D). Must animate; add NEW shapes-morphing surface demo.
4. **#108 + #191 volumetric/density shading** — very SUBTLE, OFF by default; 2D + 3D.
5. **#127 per-dataset animation styles** — precog on one, chemtrails on another, etc.
6. **#142** — legend not updated when color kwarg specified (2017 report; verify on refactor, fix or close).
7. **#177 Google Drive + formats in hyp.load** — drive file id, .npy/.npz/.pkl/.csv/.xls/.xlsx, others.

Constraints: must use the 1.0 API design language (module template: common.py base + per-model files + funnel dispatcher); matplotlib AND plotly backends, either selectable, similar quality. Deep tests, screenshot-driven dev, NEW subagents to review renders. Never merge PR #272; never touch master.

## Status
- 5 research agents dispatched (codebase survey, hull surfaces, density, #142 repro + #127 arch, gdrive/load).
- Plan to be written at docs/superpowers/plans/2026-07-05-seven-features.md, then SDD execution.
- Branch: dev-1.0-refactor @ 18a6d91f (suite 455 passed, CI 12/12 green).

## Environment gotchas (carry over)
- ALWAYS /Users/jmanning/hypertools/.venv/bin/python, MPLBACKEND=Agg
- Local kaleido/Chromium deadlocks: deselect 6 plotly-export tests locally (test_animation_export.py::test_plotly_{gif,spin_gif,mp4,spin_gif_preserves_realtime_duration}_export, test_round3.py::test_{static,animated}_svg_plotly); pass in CI. Never pkill kaleido.
- gh GraphQL bug → use REST for PR edits/comments (gh api -X PATCH repos/.../pulls/272 -f body=...; POST .../issues/272/comments)
- Docs: PATH=".venv/bin:$PATH" make -C docs html
- Commit trailer: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
- Model names in dispatchers are case-sensitive.
- Google Drive MCP connector NOT authorized in this session — irrelevant (we implement drive download in hyp.load ourselves).

## Task 7 (#108 + #191 density shading) — DONE (2026-07-05)
- New `hypertools/plot/density.py` (mirrors `surface.py`'s pattern): `normalize_density_arg`/`broadcast_density` (bool/dict only, no per-dataset list, no `color` key -- always inherits dataset color), `fit_kde`/`kde_grid_2d`/`kde_grid_3d` (gaussian_kde; <3 points or singular covariance -> warn+skip), `alpha_colormap` (mpl alpha-ramp), `iso_surfaces_3d` (marching_cubes wrapper, `HAS_SKIMAGE` flag).
- `plot.py`: `density=None` kwarg, validated + 1D-guarded same spot as `surface=`, `density_list`/`density_colors` broadcast+resolved identically to `surface_list`/`surface_colors`, passed to both backends. Full dict API documented in the `plot()` docstring.
- mpl backend: 2D `imshow` alpha-ramp (zorder=-1, below data); 3D iso-surfaces via skimage (10/35/65% levels, alpha 0.03/0.05/0.07 * alpha/0.2) else scatter-fog fallback (`kde.resample(4000)`, s=6, alpha 0.03*ratio, `depthshade=False`) + `UserWarning` suggesting `pip install hypertools[density3d]` or `backend='plotly'`. Animated: density drawn ONCE from the FULL per-dataset data inside `animate_plot3D`, before `FuncAnimation` is created; never touched by `update_lines_*`.
- plotly backend: 2D `go.Contour` (`coloring='heatmap'`, alpha-ramped colorscale to `1.5*alpha`), 3D `go.Volume`. **Tuning note**: the brief's literal formula (`isomin=0.1, opacity=min(1.5*alpha,0.5), opacityscale=[[0,0],[0.3,0.3],[1,1]]`) rendered essentially INVISIBLE in real 2-dataset scenes -- verified visually with kaleido PNG exports. Root cause: hypertools fits ALL datasets into one shared `[-1,1]` scene cube, so any one dataset's own KDE grid occupies only a small fraction of it; WebGL volume ray-marching renders low opacity as near-zero at that scale. Retuned to `isomin=0.05, surface_count=15, opacity=min(3.0*alpha,0.6), opacityscale=[[0,0],[0.3,0.4],[1,0.8]]` -- confirmed visible-but-subtle. Density traces for BOTH dims are seeded/appended OUTSIDE `trace_indices` in `_add_animation` (added a `data_trace_start` param threading the front-seeded-2D-trace offset through) so they're never touched by frame updates, unlike `surface=` which is static-2D-only.
- scikit-image installed into `.venv` (0.26.0); pinned `>=0.22.0` as new `density3d` extra in `pyproject.toml` + added to `dev` extras + `docs/doc_requirements.txt`.
- `tests/test_density.py`: 30 cases incl. a REAL meta-path-finder subprocess test (importlib `find_spec` raising `ImportError` for `skimage`, NOT a mock) for the fog fallback. All pass; full suite `578 passed, 6 deselected`.
- Evidence PNGs: `docs/images/v1.0-seven-features/density_{2d_mpl,3d_mpl,3d_plotly,2d_plotly,anim_mpl}.png`, all visually verified (data dominant, glow subtle).
- **Real bug found (matplotlib, not hypertools-specific, not fixed -- documented only):** `FuncAnimation` connects a `'draw_event'` listener that calls `_start()`/resets to frame 0 on the FIRST real `fig.canvas.draw()` (including one triggered by `fig.savefig()`). If you manually drive frames via `ani._func(k, *ani._args)` (the pattern `test_surface.py`/`test_density.py` use to test animation frame content WITHOUT ever drawing) and THEN call `fig.canvas.draw()`/`savefig()` to screenshot a "mid-animation" frame for evidence, that first draw silently resets everything back to frame 0 -- looks like missing artists/lines. Workaround: call `fig.canvas.draw()` ONCE right after creating the animation (before manually stepping frames) to consume/exhaust that listener harmlessly, then step + draw/save normally. Cost me significant time misdiagnosing this as a density-specific data-reuse/caching bug (it reproduces with or without density=, and regardless of RNG seed reuse -- pure red herrings). Worth a `docs/gallery` or contributor-notes mention if anyone else hits "my animated evidence screenshot has no data" again.
