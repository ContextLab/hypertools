# Session 2026-07-31 — review round 13: test-integrity fix + Plan 1 v4.4

Branch `dev-1.0`. Nothing pushed. Previous round committed as `524e53cc`.

## What the maintainer asked for

| # | severity | finding |
|-|-|-|
| 1 | **High** | `tests/_netskip.py` substring-matched the whole exception string, including a bare `'timeout'`, so genuine resolver regressions classified as transient |
| 2 | **Medium** | the newly guarded live tests had no strict release execution — they could skip forever behind a green gate |
| 3 | **Medium** | the round-12 session note wrongly said Plan 3 had never been reviewed |
| 4 | **Low** | the portable callback rule was still slightly too absolute |

**All four confirmed. Three of the four reproduced by execution before any edit.**

## Finding 1 — reproduced exactly, then fixed structurally

The reviewer's three claimed false verdicts all reproduced against the committed helper:

| input | old verdict |
|-|-|
| aggregate: `KeyError: parser regression` + `ReadTimeout: timed out` | **transient** (wrong) |
| `ValueError: timeout must be positive` | **transient** (wrong) |
| `AssertionError: timeout metadata missing` | **transient** (wrong) |

The third case is the dangerous one and it is *structural*, not a marker-list gap:
`load_source` aggregates **every** attempted resolver into one `HypertoolsIOError`
(`sources.py:664-670`), and — verified — raises it **outside** any `except` block, so there is no
`__cause__`/`__context__` to walk and the per-resolver detail exists only as text. A real failure in
the resolver the test wanted therefore shares a message with an unrelated fallback's timeout, and
any-substring-wins excused it.

**The fix is a change of evidence, not a longer marker list.** Classification is now:

1. **Exception type** — the raised exception and its whole `__cause__`/`__context__` chain matched
   against `TRANSIENT_TYPES` by class name across each MRO. Prose cannot reach this path.
2. **HTTP status** — `HTTPError` is transient only for **5xx**.
3. **Aggregate text, last** — each attempt line classified independently, and **any defect-shaped
   line vetoes the whole message**.

Supporting rules, each of which exists because a specific case demanded it:

- A `Name:`-shaped token only counts as evidence if it *looks* like an exception class, so
  `Google Sheets: ...` does not read `Sheets` as a defect.
- `Error`/`Exception` as bare words are excluded (they appear inside `500 Server Error`).
- Lines with no type name (`local file: not found at ...`) are expected chain noise and ignored —
  without this a genuine outage would stop skipping.
- `DEFECT_TYPES` (AssertionError, KeyError, ValueError, …) veto the **chain walk** too, so a defect
  raised inside an `except Timeout:` block cannot inherit that timeout's verdict.
- The exception path prefixes each chain entry with its own type name before text analysis, so
  `ValueError('request timed out')` reads as `ValueError: request timed out` and is a defect.

**Two behaviour changes worth knowing:**

- **A 4xx now FAILS instead of skipping.** A 404 on a dataset URL means the URL moved — a real
  regression. The old "5xx" marker list could not express this because it only ever saw text.
- `'timeout'` as a bare word is now inert. Only `'timed out'`-class multi-word phrases, exception
  types, or 5xx statuses carry a verdict.

**Verified: 21 cases pass**, covering all three reviewer cases, all 9 assertions the previous unit
test already made, live `requests` exception objects, a real requests-wrapping-urllib3 chain, and
500/503/404/403 status discrimination.

## Finding 2 — strict mode + a gate that actually runs it

`HYPERTOOLS_REQUIRE_LIVE_SOURCES=1` makes `skip_on_transient_network` re-raise instead of skipping.
Read from the environment **on every call**, not captured at import, so a real test can exercise the
branch with `monkeypatch.setenv` (real env var, real context manager — nothing stubbed).

New CI job **`live-source-gate`** runs `test_load_sources.py` + `test_load_sklearn_seaborn.py` with
that flag. Confirmed the gap the reviewer described: `dataset-gate` runs only
`test_dataset_compat.py` + `test_dataset_integrity.py`, so Sheets/Drive/Dropbox/URL/HF/seaborn could
have been dead for weeks behind a green matrix.

The job installs `.[dev]`, not the base install `dataset-gate` uses: the HF tests
`importorskip('datasets')`, which ships in `[dev]`, and letting those skip inside a strict gate would
reopen the same hole. The 476MB Drive interstitial test is `bigdata`-marked and stays deselected.

**The new gate caught a defect in its own first run**, which is the best evidence it works.
`test_skip_on_transient_network_skips_dns_but_reraises_real` asserts a transient error produces
`Skipped` — exactly what strict mode disables — so it failed under
`HYPERTOOLS_REQUIRE_LIVE_SOURCES=1`. The test was reading ambient configuration instead of pinning
its own. Fixed by having it `monkeypatch.delenv` the flag: a **unit test of the machinery must be
hermetic**, while the *live-fetch* tests around it are exactly the ones the gate is meant to
un-skip. Both modes now pass 41/41. (The newer `test_require_live_sources_...` already controlled
the variable in both directions, which is why it passed from the start.)

## Finding 3 — my own error, and a guard so it does not recur

The round-12 note claimed *"Plans 3 and 4 still have never had a review round"* while calling Plan 3
**v2** in the same sentence — which is what a v2 *is*. `notes/audit/review_plan3_forecast_animation.md`
exists (23KB, 8 defects / 4 fatal) and the plan-set README records it.

Root cause: I read the absence of a row in one table as the absence of a review. The README's review
table lists only reviews that produced a **standalone audit file**; Plan 1's rounds 6-13 were
conversational and live in `notes/session_*.md`. Added that caveat to the README so the next reader
does not repeat it. The accurate claims:

- **Plan 3: reviewed once, never RE-reviewed** against the contracts settled afterwards.
- **Plan 4: never reviewed** — verified, there is no `review_plan4_*.md`.

## Finding 4 — an assignment rule, not a ban on decisions

v4.3 said *"never write a mutation that fires on one frame only"*, which forbids highlighting a
single frame — legitimate, and portable. Restated: **assign the complete desired value on every
invocation, including the default.** It bans a per-frame *assignment*, not a per-frame *decision*;
the condition belongs in the **value**, not around the call. Worked example added to both
`FrameContext.artists` and the guide. Plan → **v4.4**; wording only, so **no test count, task or
interface moved** and v4.3's arithmetic stands (138 additions, 2689 passed / 13 skipped,
checkpoints 2657 / 2670 / 2689).

The existing guide test gained two assertions (`'assign the complete value'`,
`'highlighting exactly one frame'`) so the absolute form cannot return — still **one** test.
All six of its assertions verified against the guide text the plan specifies.

## The baseline moved, so every Plan 1 checkpoint had to move with it

Round 13's own fix added **3 tests**, taking the verified baseline **2551 → 2554**. Plan 1's drift
detection works by comparing each task's full-suite run against a stated running total, so leaving
those totals alone would have fired a **false drift at every checkpoint** — and a false drift
everywhere is indistinguishable from a real one anywhere. All 12 derived counts shifted by +3:

    2563, 2568, 2579, 2589, 2609, 2619, 2660, 2673, 2692

Re-derived from the per-task additions rather than hand-edited (9+5+11+10+20+10+41+13+19 = 138, and
2554 + 138 = 2692), with each substitution refusing to apply unless it matched exactly once. The
historical revision-note rows for v3/v4/v4.1/v4.2/v4.3 were deliberately left alone — they record
what was true at those versions.

## Process note

Edited a test-fixture string **while the full suite was running**, so the in-flight run was
**stopped and restarted** rather than reported. A gate result that predates the tree it claims to
describe is worse than no result. (The edit itself: a synthetic home-directory-shaped path in a
fixture string tripped the personal-path scan; changed to a `/nonexistent/...` prefix.)

## Plan 3 re-review + Plan 4 first review (steps 4 and 5)

Both dispatched and complete. Written up as audit files so the review table stops being the only
record: `notes/audit/review_plan3_v2_recheck.md` and
`notes/audit/review_plan4_examples_and_tutorials.md`.

**Three fatal findings between them. Neither plan is implementation-ready.**

| plan | fatal | headline |
|-|-|-|
| 3 | 1 | `_update_forecasts(frame, …)` is registered on `hooks.callbacks`, but `FrameHooks.dispatch` calls `callback(ctx)` with a **FrameContext** → `TypeError` on the first frame |
| 4 | 2 | (a) 4 of the 5 rewrites **miss their own contracted native-ratio floors** — measured by running the plan's own script on the plan's own code; (b) `recency_fade` indexes `ctx.revealed_counts[i]` while iterating `ctx.artists`, which is ~2N under `chemtrails=True` → `IndexError` |

Plan 3's tuple-`FrameContext` contract, `trace_data`/`xform_data` handling and `min_history`
separation all came back **clean** — the v4.3 contracts did land. Plan 4's baseline table was wrong
in all five notebook rows, and its "all five notebooks ship zero executed outputs" claim is false
(measured 2/6, 4/7, 1/6, 2/6, 2/7).

## Environment defect found in passing — and it is a live hazard

`.venv/lib/python3.12/site-packages/hypertools` holds a **stale, non-editable 1.0.0**. Verified:

| cwd | resolves to | version | has `antialias=` |
|-|-|-|-|
| repo root | `hypertools/__init__.py` (the tree) | 1.0.1 | yes |
| anywhere else | `site-packages/hypertools` | **1.0.0** | **no** |

So behaviour depends on the working directory. `pytest` runs from the root and shadows it, which is
why every gate this session was honest — but any directly-run script from another directory silently
tests a released package instead of the tree. Fixed with an editable reinstall, **after** the suite
finished (changing the environment mid-run would have invalidated it), and all gates re-run after.

## Open / next
- **Restart** for OMC 4.15.7 still pending — the loaded 4.2.15 hook emitted false "Command failed" /
  "Edit operation failed" notices throughout this session, every one on a call that succeeded.
- 44 pre-existing pyflakes findings across `tests/` (no linter in CI) — unchanged, still Jeremy's call.

## Files touched

    .github/workflows/test.yml                                          (live-source-gate job)
    docs/superpowers/plans/2026-07-26-hypertools-1.1-animation-core.md   (v4.3 -> v4.4)
    docs/superpowers/plans/README-hypertools-1.1.md                     (rule + review-table caveat)
    notes/session_2026-07-30_plan1-v4.3-review-round12.md                (Plan 3 correction)
    tests/_netskip.py                                                   (structural classifier)
    tests/test_load_sources.py                                          (3 new/extended tests)
    notes/audit/review_plan3_v2_recheck.md                              (NEW - re-review)
    notes/audit/review_plan4_examples_and_tutorials.md                  (NEW - first review)
    notes/session_2026-07-31_plan1-v4.4-review-round13.md                (this file)
