# Session 2026-07-03 — HyperTools 2.0 class-based refactor: Plans 1–2 complete

Branch `dev-2.0-refactor` (off dev-2.0). PR target dev-2.0. Never push master.
Goal: complete the refactor + open PR, 100% tests passing, 100% of spec tasks done.

## Status: Plans 1 & 2 COMPLETE and green (276 passed)

- **Spec:** `docs/superpowers/specs/2026-07-03-hypertools-2.0-class-refactor-design.md` (approved).
- **Plan docs:** `docs/superpowers/plans/2026-07-03-hypertools-2.0-plan-{1,2}-*.md`.
- **SDD ledger (recovery map):** `.superpowers/sdd/progress.md` — read FIRST on resume.
- Commits: Plan 1 = 9612e88/a419634/aa0efb39/4c60d765; Plan 2 = 268d0a83/6c364cf0/1ccf6325/b89ad8db.

### Environment (critical)
- Use `/Users/jmanning/hypertools/.venv/bin/python` for ALL work (3.12.10). Bare `python` = broken anaconda 3.9.
- pandas pinned `>=2.2,<3` (dw 0.4.0 breaks on pandas 3.0 → **data-wrangler#30**; lift pin when fixed, add pandas 3.0 to CI as pre-PR gate).
- Full suite ~13 min; run only at plan close. Baseline 242 → now 276 (with core tests).
- SUBAGENTS return terse stubs — controller MUST verify git state + run tests directly.

## Plan 3 (external + manip) — DESIGN NOTES from reading the fork sources

`external/`: straightforward — move `_externals/ppca.py` → `external/ppca.py`, `_externals/srm.py` → `external/brainiak.py` (keep license headers), shim `_externals`.

`manip/` needs REAL design decisions (fork code is dw/DataFrame-based and partly broken — validate, don't trust):

1. **arrays vs DataFrames.** The fork's `Manipulator` children (`Normalize`/`ZScore`/`Smooth`/`Resample`) all operate on pandas DataFrames via `@dw.decorate.apply_stacked` / `dw.unstack`/`dw.stack`. dev-2.0's `hyp.normalize` is array-based with `'across'`/`'within'`/`'row'`/`'column'`/`'zscore'` modes. Per the Plan 2 architecture decision, manip should migrate to DataFrame-flow behind the dw funnel — but the classic `hyp.normalize(x, normalize='across')` API must keep working (classic + alias). Decide: does the new `Normalize` subsume dev-2.0's modes, or does `hyp.normalize` stay a separate compat function mapping onto manip? Recommended: keep `hyp.normalize` as a thin compat wrapper preserving the mode API; add `hyp.manip` + the Manipulator classes as the new surface.
2. **Smooth = savgol, NOT gaussian.** Fork `Smooth` uses `scipy.signal.savgol_filter(kernel_width, order)`. The classic weights-trajectory gif needs **gaussian smoothing (var=300)** (see memory + scripts/generate_weights_trajectory.py). So fork `Smooth` does NOT reproduce historical behavior. Plan 3 must either add a gaussian mode to `Smooth` or provide a gaussian smoother; validate against the weights recipe.
3. **`Resample` needs `core.get`.** Fork resample imports `from ..core import get` — a helper returning `v[i]` if v is a list else `v` (elementwise kwarg indexing). Add `get` to `core/shared.py` (or `core/util.py`) as a Plan 3 prerequisite task.
4. **Fork bugs to fix/validate:** `Smooth.transformer` `maintain_bounds` uses `smoothed[c].loc[mask]=...` (pandas chained-assignment risk); unused `scipy.interpolate` import; `Normalize.fitter` builds `pd.Series(index=data.columns)` without dtype (may warn on pandas 2.3). Write REAL tests (single array, list, DataFrame, polars) and fix as needed.
5. **Manipulator base** (`jeremy/master:hypertools/manip/common.py`) + dispatcher (`manip.py`) already read — port with `core.unpack_model`/`apply_defaults`/`apply_model`. `search=['sklearn.preprocessing']` in dispatcher.

Fork files to port/validate: `jeremy/master:hypertools/manip/{common,manip,normalize,smooth,resample,zscore}.py`.
dev-2.0 to preserve: `hypertools/tools/normalize.py` (the `'across'`/`'within'`/`'zscore'` mode semantics + `hyp.normalize` API).

## Remaining plans
- Plan 3: external + manip (design notes above).
- Plan 4: reduce + align + cluster (the big arrays→DataFrames migration behind the funnel; soft-mixture clustering).
- Plan 5: io (load/sources/streaming/save).
- Plan 6: plot + colors (dual backend, robust coloring, multilevel-index styling).
- Plan 7: top-level API + aliases + DELETE DataGeometry + retire tools/ shims.
- Plan 8: docs/gallery/notebooks migration + Playwright visual verify + PR evidence.
- Then: whole-branch review (opus) + open PR into dev-2.0, lift pandas pin when dw#30 lands.
