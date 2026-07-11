# Audit Ledger — HyperTools 1.0 (2026-07-11)

Working truth for the release audit. Update after every phase transition, wave completion, and fix commit.

## Phase status

| Phase | Status | Notes |
|-|-|-|
| 0 setup | in progress | branch `audit/release-1.0-2026-07` @ e0f4e33e; tree cleaned (gallery-regen noise reverted) |
| 1-3 red-team waves (46 units) | not started | |
| 4 verification | not started | |
| 5 fixes | not started | |
| 6 re-audit | not started | |
| 7 merge + CI | not started | |
| 8 PR report | not started | |
| 9 wrap-up | not started | |

## Key facts

- PR #272 = dev-1.0-refactor → dev-1.0 (sole open PR; report target)
- CI: push to [master, dev, dev-1.0, dev-1.0-refactor] → 12 jobs (3 OS × py3.10-3.13), full pytest
- Local suite baseline: 1490 passed / 4 skipped / 8 deselected / 304 warnings (2026-07-10, dev @ e0f4e33e)
- hypertools version: 1.0.0.dev0 · exports: HyperAnimation, Pipeline, align, analyze, apply_model, cluster, config, core, datageometry, describe, external, impute, io, load, manip, normalize, plot, predict, reduce, save, set_interactive_backend, tools
- 54 gallery examples · 16 tutorial notebooks · 6 README code blocks
- Optional deps ALL installed locally: chronos 2.3.1, kagglehub 1.0.2, kaleido 1.3.0 (BANNED locally — deadlocks), playwright 1.61.0, plotly 6.8.0, pylsl 1.18.2, umap-learn 0.5.11

## Findings tally

(updated after Phase 4 verification)

| Severity | Filed | Confirmed | Fixed |
|-|-|-|-|
| critical | - | - | - |
| major | - | - | - |
| minor | - | - | - |
| doc | - | - | - |
| style | - | - | - |
| enhancement | - | - | - |

## Fix commits

(one line per fix: `<sha> <finding-ids> <summary>`)

## Workflow runs

(runId + purpose, for resume)

- `wf_bc33c3c0-640` — Phase 1-3 red-team, 46 auditors (F18 first → 44 parallel → X3 alone). Script: `~/.claude/projects/-Users-jmanning-hypertools/7e6531b3-066a-4ce2-b1f6-7c07c5e87b15/workflows/scripts/audit-1p0-redteam-wf_bc33c3c0-640.js`

## Seed observations (controller pre-warm, 2026-07-11 00:47)

Cross-check that auditors independently find these (auditor-quality canary):

1. `hyp.load('sotus')` (and likely other hosted geo pickles) emits 3× sklearn `InconsistentVersionWarning` — estimators pickled under sklearn 1.0.2, unpickled under 1.8.0 (CountVectorizer, LatentDirichletAllocation, Pipeline). Users see scary warnings + "invalid results" risk on a flagship dataset load. → F18/X4.
2. `hyp.load('sotus')` returns a `Pipeline` (len 2), not raw data — 1.0 contract says load() returns raw data. Verify intended vs bug. → F18.
3. Baseline: weights=list(36), spiral=list(2), mushrooms=DataFrame(8124).
