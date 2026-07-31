# Session 2026-07-30/31 — Plan 1 review round 12 (v4.2 → v4.3) + transient-network hardening

Branch `dev-1.0`. Nothing pushed. Working tree limited to the files listed below.

## What the maintainer asked for

1. **High** — the artist-persistence prose contradicted the lifetime table two paragraphs above it.
2. **Medium** — `FrameContext.artists` declared `List[Any]` but plotly spin supplied a `tuple`.
3. **Low** — `images/README.md`'s "Deleting also saves nothing" was too absolute.
4. Re-run the release gates; restart for OMC 4.15.7.

All three findings were **confirmed**, two of them by measurement rather than by reading.

## Measurements taken (the evidence, not the argument)

| claim | how it was checked | result |
|-|-|-|
| plotly non-spin frames are per-frame | drove all 5 styles, compared `fig.frames[0].data[0] is fig.frames[1].data[0]` | **isolated** for `True`/`serial`/`window`/`morph` (data in every frame); `spin` carries **no frame data at all** (0/4) |
| `.set_color()` is matplotlib-only | tried `.line.color` on plotly traces | settable on `go.Scatter` + `go.Scatter3d`; `go.Mesh3d` has **no** `.line` |
| container types diverge | grepped all record sites | 7 matplotlib updaters pass lists, 4 plotly branches pass `frame_traces`/tuple — **11 recorders, 1 construction site** (`FrameHooks.dispatch`) |
| frozen dataclass + `__post_init__` | built and ran the exact dataclass | coercion works; `FrozenInstanceError` on rebind; `AttributeError` on `.append` |
| tuple change breaks a test | ran the comparison | `(17,4,0) == [17,4,0]` is **False** → existing test fixed |
| custom numpydoc sections under `-W` | minimal sphinx project, repo's exact extensions | **build succeeded, 0 warnings** |

## Plan 1 → v4.3

- **H1** persistence rule kept portable, re-founded on the correct per-backend reason; both
  failure modes now stated as opposites. Guide gains a labelled `# PLOTLY ONLY` `.line.color`
  counterpart beside the `# MATPLOTLIB ONLY` `.set_color()` one.
- **H2** `artists`/`datasets` → `Tuple[Any, ...]`, `revealed_counts` → `Optional[Tuple[int, ...]]`,
  normalized in `FrameContext.__post_init__` (one construction site covers all eleven recorders).
  Imports change to `from dataclasses import dataclass` / `from typing import Any, Optional, Tuple`.
- **H3** *(found by this audit)* Task 8 called `_serial_current_index`; Task 7 defines
  `serial_current_index`. Task 8 would have failed with `ImportError`. Also stopped Task 8
  recomputing `_counts`, a duplicate of `lengths` already bound at `plotly_backend.py:2823`.
- **H4** *(found by this audit)* `assert ctx.revealed_counts == drawn` would have gone red under H2.

New tests: `test_plotly_non_spin_frames_are_isolated_per_frame` (×4),
`test_frame_context_containers_are_canonical_tuples` (×6),
`test_animation_guide_gives_both_failure_modes_not_just_persistence`.

**Arithmetic:** Task 7 31→41, Task 9 18→19, total 127→**138**, final **2,689 passed / 13 skipped**,
checkpoints **2657 / 2670 / 2689**. Task 9's `def test_` column also corrected 8→9 (it has always
held 9 defs; the *collected* figure of 18 was right, so no total was ever wrong).

## The suite failure — diagnosed, not dismissed

The full run came back **1 failed**: `tests/test_load_sources.py::test_load_google_sheet_live`.
Google Sheets read-timed-out (60s) and Google Drive answered **500**. It **passed on re-run in 2.5s**.

Root cause was not the outage — it was that this test was the **only live-fetch test in its own
file without the transient guard its siblings already had** (`_skip_on_transient_network`, added
2026-07 for an HF ReadTimeout, and itself unit-tested).

Fixed:
- guard applied to all **six** live-fetch tests in that file (Sheets, Drive spiral, Dropbox,
  generic-URL pair, `plot()` auto-load, and the bigdata 476MB interstitial download);
- `_TRANSIENT_NETWORK` gained `'500 server error'` / `'internal server error'` — the predicate's own
  docstring already claimed "5xx" but only listed 502/503/504. Matched as **phrases**, never a bare
  `' 500'`, which would false-positive on a real assertion like `"shape 500 != 499"`;
- the classifier's unit test now pins the verbatim outage message, a bare-500 message, **and** two
  assertion strings that must *not* classify as transient.

**Verified both directions end-to-end** with a throwaway probe (deleted, never committed — it
monkeypatches, so it must not ship): the real outage exception → **Skipped**; a wrong-shaped
DataFrame → **AssertionError**. Guarding is not weakening.

## The second gap, also fixed

`tests/test_load_sklearn_seaborn.py::test_load_penguins_is_seaborn` fetches seaborn-data from GitHub
with **no** transient guard (that file's own docstring: it "hit[s] the real seaborn-data GitHub repo
over the network"). It did not fail today, but it is the same latent flake.

A fourth private copy of the predicate would have been wrong — the suite already runs **three**
deliberate strategies:

| strategy | where | when to use it |
|-|-|-|
| retry inside the library | `_github_get_with_retry` (`test_load_538_kaggle.py`) | we own the request |
| release-gated skip | `HYPERTOOLS_REQUIRE_*` (`test_dataset_compat.py`) | a silent skip would hide a release regression |
| transient predicate | was `test_load_sources.py`, now shared | a one-off fetch through someone else's endpoint |

`tests/__init__.py` exists, so `tests` is a real package and a shared module imports cleanly (a
`conftest.py` would **not** have been importable by name here). Extracted the predicate to
**`tests/_netskip.py`**, documenting all three strategies so the next person picks rather than
copies. `test_load_sources.py` imports it under its original private names, so that file's own unit
tests still exercise the real implementation unchanged.

## The release machinery caught my own mistake

The re-run failed on
`tests/test_packaging_artifacts.py::test_sdist_contains_only_tracked_files_plus_allowlist`:

    AssertionError: 1 untracked file(s) leaked into the sdist: ['tests/_netskip.py']

Correct and useful — I created `tests/_netskip.py` but never `git add`ed it, so it would have
shipped in the sdist while being invisible to git. `git add` fixed it. Worth recording as evidence
that the 1.0 release gates earn their keep on ordinary work, not just at release time.

## Operational trap, hit twice in one session

`cmd | tail -N` reports **tail's** exit status, not the command's. This produced a
"completed (exit code 0)" notification for a pytest run that had **1 failed**, and earlier in the
session the same shape reported success for a `make html` that died with `Error 2`. Long-running
gates should be run as `cmd > log 2>&1; rc=$?; tail log; exit $rc` (or use `PIPESTATUS`), never
piped straight into `tail`.

## Open / next
- **44 pre-existing pyflakes findings across `tests/`** (unused imports, unused locals). Surfaced
  because I linted my own changed files. **The project runs no linter in CI** — verified: nothing in
  `.github/workflows/*.yml` or `pyproject.toml` references flake8/ruff/pylint/pyflakes — so these are
  not a failing gate, just latent hygiene. My three changed files are pyflakes-clean, which is the
  right bar for this round. Two were fixed in passing because they sat in a file I was already
  editing: an unused `contextlib` import (left behind by the extraction) and
  `test_plot_mixed_dtype_dataframe`'s unused `geo` binding — stale pre-1.0 `DataGeometry` naming,
  now `fig` **with an added assertion on the return type**, so the test checks more than it did.
  Adopting a linter suite-wide is a separate call.
- **Restart** for OMC 4.15.7 — the loaded 4.2.15 hook emitted ~20 false "Command failed" /
  "Edit operation failed" notices this session, every one on a call that succeeded.
- Plans 3 and 4 still have **never had a review round**. Plan 3 is v2, predating `FrameHooks`,
  the both-axis precondition and the `trace_data` contract. It adds forecast rendering to the same
  four plotly branches where the `:2975` undercount and the spin-artists defect were found.

## Files touched

    docs/superpowers/plans/2026-07-26-hypertools-1.1-animation-core.md   (v4.2 -> v4.3)
    docs/superpowers/plans/2026-07-27-hypertools-1.1-forecast-animation.md (container-type cite)
    docs/superpowers/plans/README-hypertools-1.1.md                     (lifetime + containers)
    images/README.md                                                    (deletion economics)
    tests/test_load_sources.py                                          (transient guards)
    notes/session_2026-07-30_plan1-v4.3-review-round12.md                (this file)
