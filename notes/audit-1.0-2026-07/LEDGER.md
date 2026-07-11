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
