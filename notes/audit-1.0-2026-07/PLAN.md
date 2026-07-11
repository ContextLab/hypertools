# HyperTools 1.0 Release Audit — Master Plan

**Started:** 2026-07-11 (Fri) · **Branch:** `audit/release-1.0-2026-07` (off `dev-1.0-refactor` @ e0f4e33e) · **Target PR:** #272 (dev-1.0-refactor → dev-1.0)

## Mission (Jeremy's bar, verbatim)

> "this update represents the 1.0 release of hypertools, our lab's flagship software project. it needs to be as perfect as we can make it: all functions work correctly · super smooth and reliable performance · all documentation up to date, including tutorials · any examples provided in the documentation (README.md, sphinx documentation, API strings, tutorials, etc.) MUST be verified as running · the code is well organized and easy to read · the API is consistent across all functions · all desired functionality is present. verifying that everything works isn't just about making sure the code runs. it's about correct screenshots, correct numbers, and consistency across the entire toolbox. … red-teaming each function/feature using actual screenshots, code runs, and brainstorming edge cases in addition to common use cases. verify everything works as expected using INDEPENDENT subagents (i.e., no self reviews allowed!). CRITICAL: NOTHING is out of scope: you MUST fix ANY issues that are surfaced in this massive audit."

**Goal:** complete the audit → implement ALL fixes → post comprehensive report to PR #272 → update the PR → all CI tests green.

## Ground rules (binding)

1. **NEVER touch `master`.** Base of all work: `dev-1.0-refactor`.
2. **No self-review**: whoever writes code never judges it. Auditors ≠ verifiers ≠ fix reviewers (fresh, independent dispatches each time).
3. **No mocks, ever.** Real runs, real datasets, real servers, real screenshots.
4. Env: `/Users/jmanning/hypertools/.venv/bin/python`, `MPLBACKEND=Agg`. Never kaleido locally (deadlocks on this Mac) — plotly checks go through `write_html` + Playwright screenshots.
5. Every substantive claim needs direct evidence: exact commands, verbatim outputs, PNG/GIF screenshots.
6. Fix every issue **when noticed** — nothing punted, nothing "pre-existing".
7. Commit early and often on the audit branch; ledger updated continuously.
8. Docs updated whenever code/examples change; pip changes → requirements updated.

## Phases

| # | Phase | Method | Gate |
|-|-|-|-|
| 0 | Setup: branch, scaffolding, cache pre-warm | inline | tree clean, dirs exist |
| 1 | Function red-team — 24 units | Workflow: independent auditors | findings JSON per unit |
| 2 | Docs red-team — 14 units (README, sphinx, 54 gallery examples, 16 tutorials, docstrings, links, drift) | same Workflow | findings JSON per unit |
| 3 | Cross-cutting — 8 units (API consistency, errors, perf, warnings, packaging, code org ×2, issue-tracker) | same Workflow | findings JSON per unit |
| 4 | Triage: dedup → adversarial verification of EVERY finding by fresh agents (visual findings get vision verifiers) | Workflow | CONFIRMED/REFUTED + severity |
| 5 | Fixes: every confirmed issue fixed (test-first where feasible), one commit per fix/batch | inline + implementer agents | tests pass per fix |
| 6 | Re-audit: every touched unit re-red-teamed by NEW independent agents; full pytest; docs rebuild; whole-diff independent review | Workflow + inline | zero regressions |
| 7 | Merge → `dev-1.0-refactor`, push, 12/12 CI green (scaffolding removed pre-merge) | inline | CI `success` |
| 8 | Report: methodology + coverage matrix + every finding w/ evidence + fixes w/ commits → PR #272 comment; update PR description | inline | posted |
| 9 | Notes + memory wrap-up | inline | — |

## Audit units

**Functions (24):** F01 plot-static-core · F02 plot-hue · F03 plot-pipeline-integration · F04 plot-animate-window · F05 plot-animate-special (spin/chemtrails/precog/bullettime) · F06 plot-backends (plotly parity) · F07 plot-density-surface · F08 plot-inputs (DFs/MultiIndex/text/NaN/weird shapes) · F09 plot-save-return · F10 plot-remaining-kwargs sweep · F11 reduce+describe · F12 align · F13 cluster · F14 manip+normalize · F15 analyze · F16 predict · F17 impute · F18 load-hosted (+legacy geo unpickle) · F19 load-external (538/kaggle/URL/local) · F20 save round-trips · F21 apply_model+Pipeline · F22 io-streaming+LSL (real stream) · F23 core/config/exceptions · F24 colors/interactive/fonts helpers

**Docs (14):** D01 README (every block run verbatim) · D02 sphinx build+warnings+thumbnails+autodoc coverage · D03–D06 gallery examples (4 batches × ~14, ALL 54 run for real) · D07–D10 tutorials (4 batches × 4, ALL 16 executed end-to-end) · D11 docstring examples: plot pkg · D12 docstring examples: everything else · D13 link validation (every URL manually fetched) · D14 docs-vs-code drift (signatures, version strings, extras)

**Cross-cutting (8):** X1 API consistency + full export census · X2 error-message quality (deliberate misuse everywhere) · X3 performance/reliability (timings, memory growth, import time) · X4 warning hygiene (catalog all runtime warnings) · X5 packaging (wheel/sdist/extras/fresh-venv) · X6 code org: plot pkg · X7 code org: rest · X8 all 28 open ContextLab issues cross-checked vs 1.0 ("all desired functionality present")

## Red-team method (every auditor)

1. Read target source + docstrings; enumerate every documented parameter/behavior.
2. **Brainstorm ≥15 edge cases** (recorded, even if untested).
3. Run for real: ≥5 common workflows · every feasible documented param · edge cases · every docstring example VERBATIM · deliberate misuse (judge error messages).
4. Visual outputs → PNG evidence in `notes/audit-1.0-2026-07/evidence/<unit>/`; expected-vs-observed for each.
5. Every numeric claim in touched docs recomputed.
6. Full findings → `notes/audit-1.0-2026-07/findings/<unit>.json`; auditors NEVER modify tracked files.

**Finding schema:** `{id: "F02-003", severity: critical|major|minor|doc|style|enhancement, title, description, repro (complete runnable code), expected, actual, evidence[], docs_impact[]}` — critical = wrong results/crash on reasonable use; major = broken documented feature or bad visual; enhancement = missing desired functionality (in scope!).

## Verification & fix protocol (no self-review)

- Phase 4 verifiers get ONLY `{repro, expected, actual, evidence}` — never the auditor's reasoning. They re-run and rule CONFIRMED/REFUTED. Visual findings: vision agents examine the PNGs.
- Phase 5: fixes by controller/implementers; each fix ships with a real regression test.
- Phase 6: fixed units re-audited by agents that did not write the fix; whole-branch diff reviewed by independent reviewer agents (quality + security + simplicity).
- Full local suite + docs build re-run after ALL fixes (re-run everything if anything changed).

## Evidence & report protocol

- Evidence committed on the audit branch; PR links pinned to **commit SHAs** (raw.githubusercontent.com/…/<SHA>/…) so they survive later cleanup — last round's branch-path links 404'd after scaffolding removal; not this time.
- Audit branch pushed to origin and kept alive after merge.
- Report: coverage matrix (unit × tested/pass/fail), every finding + verdict + fix commit, before/after screenshots for visual fixes, CI matrix result.

## Resume protocol (if context lost)

Read `LEDGER.md` (phase status + finding tally + commits) → `findings/*.json` → workflow journal (`<transcriptDir>/journal.jsonl`). Trust ledger + `git log` over memory. Workflow runs are resumable via `resumeFromRunId`.
