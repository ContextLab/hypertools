# Audit Ledger — HyperTools 1.0 (2026-07-11)

Working truth for the release audit. Update after every phase transition, wave completion, and fix commit.

## Phase status

| Phase | Status | Notes |
|-|-|-|
| 0 setup | done | branch `audit/release-1.0-2026-07` @ e0f4e33e; tree cleaned |
| 1-3 red-team waves (46 units) | **DONE 46/46 — 708 findings** (16 crit, 98 major, 142 doc, 306 minor, 118 style, 28 enh) | 4 waves (3 spend-cap resumes); ~13.1M subagent tokens, ~4650 tool uses total |
| 4 verification | RUNNING — run `wf_592422d1-611` | 46 blind adversarial verifiers (effort=high), verdicts → verdicts/*.json; first launch failed (args passed as JSON-string → script now hardcodes units) |
| 5 fixes | RUNNING | 5A `wf_76a828b2-710`: 8 parallel implementers, disjoint ownership (io / manip+normalize / align / predict+impute / plot-backend.py / core+_shared+packaging / cluster+reduce / tools-analyze). 5B `wf_87ffe0b7-a93`: SEQUENTIAL pipeline over plot pkg minus backend.py (B1 static/inputs/kwargs → B2 hue/colors → B3 animation/save → B4 density/surface). Agents do NOT git; controller reviews diffs, runs full suite, commits per batch. Then 5C docs/examples/tutorials (needs fixed code), 5D leftovers (X1/X4/X6/X7/X8 minors+style, D13 links, D14 drift). |
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

## Wave-2 update (2026-07-11 ~02:50)

Cumulative: 29 units workflow-completed + 3 orphan-valid JSONs (D01-readme, D05-gallery-data-text, D10-tutorials-embeddings-lsl) = **32/46 units on disk, 466 findings filed** (10 critical, ~72 major).
New this wave: F13, F14, F15, F16, F17, F21, F22, F23, F24, D02, D03, D04, D06, D07, D08.
Criticals added: F14-001 (Smooth on a list smooths ACROSS dataset boundaries — silent subject mixing), F16-001 (default Kalman forecaster never learns dynamics — flat forecasts), F16-002 (1-D input to predict → 1×n row), D07-001 (plot.ipynb tutorial crashes fresh execution: hue 8120 vs 8124), D01-crit + D05-crit (in orphan JSONs — details there).
Wave 3 = second resume for remaining 17: D01, D05, D09, D10, D11, D12, D13, D14, F19 (server-error retry), X1-X8.

## Fix commits

(one line per fix: `<sha> <finding-ids> <summary>`)

## Workflow runs

- `wf_bc33c3c0-640` — Phase 1-3 red-team wave 1: 14/46 completed (+2 orphan-valid), 30 spend-capped, 1 server error. 3.83M subagent tokens, 69 min.
- wave 2 = resume of the same run (see below for run id once launched)

## Seed observations (controller pre-warm, 2026-07-11 00:47)

1. ✓ CONFIRMED as F18-load-hosted-001 (critical): `hyp.load('sotus')` returns a broken sklearn Pipeline — sha-identical to nips_model (Drive id duplicated).
2. ✓ CONFIRMED as F18-load-hosted-002 (major): hosted *_model pickles are sklearn-1.0.2; version warnings + repr crashes under sklearn 1.8.
3. Baseline: weights=list(36), spiral=list(2), mushrooms=DataFrame(8124).
