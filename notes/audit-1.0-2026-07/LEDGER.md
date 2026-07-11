# Audit Ledger — HyperTools 1.0 (2026-07-11)

Working truth for the release audit. Update after every phase transition, wave completion, and fix commit.

## Phase status

| Phase | Status | Notes |
|-|-|-|
| 0 setup | done | branch `audit/release-1.0-2026-07` @ e0f4e33e; tree cleaned |
| 1-3 red-team waves (46 units) | wave 1 done: 16/46 units, 224 findings | 14 completed + 2 orphan-valid (F15, F16 — re-running); 30 units hit monthly spend cap, 1 (F13) server error → RESUMED as wave 2 |
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
- 54 gallery examples · 16 tutorial notebooks · 6 README code blocks · exports: 22 public names
- Optional deps ALL installed locally: chronos 2.3.1, kagglehub 1.0.2, kaleido 1.3.0 (BANNED locally — deadlocks; CI covers those paths), playwright 1.61.0, plotly 6.8.0, pylsl 1.18.2, umap-learn 0.5.11
- Evidence (43MB) is NOT committed (notes/audit-1.0-2026-07/.gitignore); findings JSONs ARE. Curated evidence → orphan branch at report time, SHA-pinned links.

## Environment fixes (not code findings)

- 2026-07-11 01:59: `.venv` held a STALE NON-EDITABLE hypertools snapshot shadowing the working tree for any non-repo-root cwd (flagged independently by 10+ wave-1 auditors as [infra]; they worked around via PYTHONPATH — their results are valid). Fixed: `pip install -e . --no-deps --force-reinstall`; verified `import hypertools` from /tmp now resolves to the repo tree. All [infra] stale-venv findings close as environment-resolved.

## Wave-1 findings tally (16 units, 224 filed; pre-verification)

| Severity | Filed |
|-|-|
| critical | 4 |
| major | 38 |
| doc | 33 |
| minor | 109 |
| style | 29 |
| enhancement | 11 |

Criticals filed: F18-001 (load('sotus') returns broken sklearn Pipeline — Drive id duplicated with nips_model; canary CONFIRMED), F06-001 ($HYPERTOOLS_BACKEND env var crashes import), F08-001 (plain list-of-lists matrix crashes plot with nonsensical color= error; == F01-004), F12-001 (trim_and_pad silently scrambles row order for non-RangeIndex DataFrames).

Unit status: F01 18f · F02 14f · F03 15f · F04 12f · F05 15f · F06 12f · F07 8f · F08 17f · F09 15f · F10 17f · F11 18f · F12 10f · F18 9f · F20 10f · orphan-valid: F15 16f, F16 18f. Full detail: findings/*.json.

Auditor-quality canaries: both pre-warm seeds (sotus Pipeline bug, sklearn version warnings) independently found by F18 ✓. F03 auditor also proved pipeline order normalize→reduce→align via exact coordinate equality and caught repo CLAUDE.md's Data Flow section listing the wrong order.

## Fix commits

(one line per fix: `<sha> <finding-ids> <summary>`)

## Workflow runs

- `wf_bc33c3c0-640` — Phase 1-3 red-team wave 1: 14/46 completed (+2 orphan-valid), 30 spend-capped, 1 server error. 3.83M subagent tokens, 69 min.
- wave 2 = resume of the same run (see below for run id once launched)

## Seed observations (controller pre-warm, 2026-07-11 00:47)

1. ✓ CONFIRMED as F18-load-hosted-001 (critical): `hyp.load('sotus')` returns a broken sklearn Pipeline — sha-identical to nips_model (Drive id duplicated).
2. ✓ CONFIRMED as F18-load-hosted-002 (major): hosted *_model pickles are sklearn-1.0.2; version warnings + repr crashes under sklearn 1.8.
3. Baseline: weights=list(36), spiral=list(2), mushrooms=DataFrame(8124).
