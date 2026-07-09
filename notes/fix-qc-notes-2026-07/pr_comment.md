## Release-hardening pass — full-package bug hunt, fixes, and independent red-team

Following up on the QC-notes fixes, this pass widens scope to the whole package (not just the
original audit items) to get 1.0 release-ready. Method for every issue: **reproduce → verify
(numeric output + screenshots) → fix → adversarial red-team by an INDEPENDENT subagent → re-verify**.

### What was done
1. **Bug hunt** — 8 parallel read-only agents swept the whole package; ~30 confirmed, reproduced
   defects (silent-wrong > crash > cryptic > UX), organized into batches B1–B8 + the K1 backend issue.
2. **Fixes** — one commit per batch (`edf6049e`, `66675cb0`, `7d71975b`, `ca13d050`, `30725456`,
   `00607d26`, `176fbd7a`, `be0dcb5a`), each with regression tests.
3. **Independent red-team** — 5 subagents, one per batch area, each told to *prove the fixes wrong*
   with real data. They confirmed the primary claims and surfaced a handful of real gaps.
4. **Red-team fixes** — 4 follow-up commits (`4fc2e1b0`, `9d5ce9d6`, `111775aa`, `a81c7a91`) with
   24 more regression tests.

### Highlights (what was broken → now fixed)
- **PPCA imputation returned latent PCA scores** — it rotated observed values and dropped columns,
  violating "fills missing values in place". Now splices (observed preserved to 1e-9, shape kept).
  The red-team then caught that the splice left `min_obs`-dropped columns NaN → crashed `reduce`/
  `plot`; fixed by densifying dropped columns (`4fc2e1b0`).
- **`align` crashed by default on datasets with different column counts** — the canonical
  hyperalignment case. Fixed; also warns on row-trim.
- **`analyze(cluster=...)` returned labels, not the transformed data** (contradicted its docstring).
  Now returns data; labels recoverable for every clusterer including hard ones.
- **Canonical `reduce={'model':...,'kwargs':{...}}` + `ndims` silently dropped the reduction.** Fixed.
- **1-D arrays, flat numeric lists, tuples, Series, 0-d arrays** all crashed on input. Now coerced.
- **`set_interactive_backend('plotly')` + animate raised `HypertoolsBackendError`** and never actually
  switched the renderer. Fixed with a render-backend preference (now case-insensitive too).
- **PPCA EM could loop forever** on small/degenerate NaN data (>25s hangs) → capped with a clear
  non-convergence warning.
- Plus: constant-column ZScore/Normalize NaN, all-missing-column impute crash, hard-clusterer
  pipelines, animation `.save()` format dispatch, `predict` horizon validation, streaming kwargs, and
  several cryptic-error → clear-error improvements.

### Evidence (screenshots — "after-fix, now works")
![1-D array now plots](https://raw.githubusercontent.com/ContextLab/hypertools/fix/qc-notes-2026-07/notes/fix-qc-notes-2026-07/evidence/hardening/b2_1d_array.png)
![align mismatched columns](https://raw.githubusercontent.com/ContextLab/hypertools/fix/qc-notes-2026-07/notes/fix-qc-notes-2026-07/evidence/hardening/b5_align_mismatched_cols.png)

Numeric verification (`notes/fix-qc-notes-2026-07/evidence/hardening/RESULTS.txt`):
```
B1 reduce dict+ndims: shape=(150, 2) matches PCA(n_components=2,whiten) up to sign=True
B3 PPCA impute: in (40,6) out (40,6) observed_preserved(1e-9)=True filled_finite=True
K1 backend='matplotlib' -> matplotlib.figure.Figure; backend='plotly' -> plotly Figure; pref restored=True
B5 align mismatched cols: aligned shapes=[(60,12),(60,12),(60,12)] finite=True
B6 HyperAnimation.save('.gif'): 25082292 bytes, magic=GIF89a
```

### Red-team verdicts (per area)
- **B1 reduce**: SOLID across 14 reducers.
- **B2 input**: 3 gaps found (align dict-form unknown model, 0-d array, scalar hue) + a pre-existing
  PPCA hang — all fixed (`111775aa`).
- **B3 manip/impute**: observed-preservation solid; 1 significant regression (dropped-column NaN) +
  single-column + assert — fixed (`4fc2e1b0`).
- **B4 pipeline/analyze**: data-recovery, normalize-skip, n_components, apply_model all solid; hard-
  clusterer label recovery fixed (`9d5ce9d6`).
- **B5/B6/B7/B8 align/animation/edge**: solid; predict `np.bool_`/0-d horizon gap + assert fixed
  (`a81c7a91`).
- **K1 backend**: SOLID; case-sensitivity gap fixed (`a81c7a91`).

### Tests
Full suite: **1454 passed, 4 skipped, 7 deselected, 0 failed** (includes the 24 new red-team
regression tests). The only excluded tests are the plotly image-export tests
(`test_animation_export.py::test_plotly_*`, `test_round3.py::test_*_svg_plotly`), which deadlock in the
`kaleido` export subprocess in this headless CI env — a pre-existing infra issue, not a regression;
they pass individually with a live kaleido/chromium.

Full per-batch reproductions, numeric tables, and the fix rationale are in
`notes/fix-qc-notes-2026-07/PR_EVIDENCE.md` on this branch.

**This branch is for review only — please do not merge; base is `dev-1.0-refactor`, `master` untouched.**
