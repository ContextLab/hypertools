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

- `885325c7` A2-manip (22 findings): per-dataset Smooth (F14-001/D01-001 critical), validation, doctest Examples. 290+6 tests green pre-commit.
- `ff7153e6` A3-align (10): trim_and_pad row-order preservation (F12-001 critical), align=False no-op, kwarg validation.
- `3517435f` A5-backend (7): HYPERTOOLS_BACKEND import crash (F06-001 critical), switch-state safety, eager validation.
- `04985a4a` A8-tools (12): empty-list guard before LDA path (X2-005), analyze False-skip, df2mat pandas-3 (X4-001).

- `462fe6ee` A1-io (40): sotus speeches restored (critical ×3), CSV sep fix (critical), unknown-ext guard (critical), atomic format-aware save(), model-pickle repair-on-load.
- `c3f6f6da` A4-predict-impute (37): Kalman em_vars (2 criticals: flat forecasts + zero-fill impute), 1-D series (critical), degenerate guards.
- `4c492f81` A6-core-packaging (26): config.ini in wheel, np.seterr side effect gone, venv droppings excluded.
- `2179688b` A7-cluster-reduce (21): False-skip, TSNE/describe, honest errors.

5A verification: 616 passed / 2 skipped (expected guards) pre-commit. Integration checks DONE 2026-07-12: corpus='sotus' → (2,50) topic vectors ✓; hyperalign n_itr → TypeError with did-you-mean ✓; impute([])/predict([]) → clear ValueErrors ✓.

Partial edits from spend-capped agents (A1/A4/A6/A7/B1) were REVERTED before these commits; those agents re-ran fresh on this base.

## Wave-5A COMPLETE (8/8, 178 findings fixed)

New: A1-io 40 (sotus speeches restored via dw corpus — verified 29 docs; CSV sep fix; format-aware atomic save(); repair-on-load for stale model pickles), A4-predict-impute 37 (Kalman em_vars: impute sweep r=0.997/0.996/0.997/0.996 vs pre-fix 0.995/0.777/0/0; predict sine r=0.984; PPCA default r 0.125→0.977), A6-core-packaging 26 (config.ini IN wheel — real build verified; no venv droppings; np.seterr side effect gone), A7-cluster-reduce 21 fixed + 18 plot.py-side escalations.

## Wave-5B COMPLETE (4/4, 127 findings fixed; full-suite gate running)

B1 62 fixed (data-faithful static lines incl. X3-002; ro- fmt colors; list-of-lists ONE dataset; kwarg did-you-mean validation; plot() Examples doctests; cyclic-palette 5/6-trim for continuous hue). B2 24 (NaN-hue neutral color, Series index, singleton category, palette lists/cliff, colorbar names). B3 34 (chemtrails future-leak fixed; hue+animate animates on mpl; per-dataset frame grid to longest; figure-leak fixed; apng clobber fixed; pathlib; ffmpeg errors wrapped). B4 7 (plotly surface shows enclosed points — Playwright-verified; degenerate-density warnings; kwarg validation).

**MAINTAINER SIGN-OFF FLAG for PR report:** continuous-hue cyclic palettes (hls/husl default) now sample 5/6 of the hue circle so endpoints are distinguishable (was: both ends red, dist 0.03→0.6). CHANGES DEFAULT LOOK of continuous-hue plots. Implemented + tested + documented; Jeremy should confirm he likes it.

## New controller/5C items from 5B escalations

- CLAUDE.md Data Flow order: swap Alignment/Reduction (canonical: manip→normalize→reduce→align→cluster) — controller, trivial.
- _shared/helpers.py:118/:133 vals2colors linspace(min, max+1) → (min, max) (F24-005) — controller + test.
- Verify set_interactive_backend('bogus') raises after A5 (F24-015 claim overlap).
- __init__.py: exceptions re-export, supported_models export, shadowing-imports doc note (accumulate F23-005, F21-005, F24-002, F07-007, F11-014, F16-017).
- 5C: examples/plot_hue.py int() numpy fix (F02-012), examples/save_movie.py data[:18] (F09-011), docs/api.rst HyperAnimation entry (F04-008), examples/plot_describe.py covariance→distance prose (F11-009), examples/plot_clusters3.py params→kwargs (F13-018), plot_apply_model params→kwargs (F21-014), plot_pipelines_return_model trim note (F21-015), tutorials cluster.ipynb/analyze.ipynb cell fixes (F13-017, F15-007/008), plot.ipynb hue 8120 fix (D07-001).

## Post-5B plot.py escalation batch (B5) — dispatch after B4 lands

From A7: F13-001/002/003/004/005/007/009/010/016/020/021/022 (plot.py cluster integration: FeatureAgglomeration guard, n_clusters exemption grammar, random_state threading, bundle k mismatch, small-int-hue categorical palette, cluster=False, spec-kwargs precedence + dict KeyError, LDA/NMF caveat, class/instance specs, k-default docs, legend numeric sort). From A1: F22-004 (stream kwarg whitelist warn), F22-010 (plot.py:1003 stale geometry ref). From A4: F17-006 remainder (format_data.py:262 + plot.py stale PPCA comments).

## Controller items (mine, after 5B — no agent owns these files)

- hypertools/__init__.py: re-export HypertoolsError/BackendError/IOError (F23-005), export supported_models (F21-005), module docstring note on function-shadows-subpackage imports (F16-017, F11-014, F06-009, F01-014-refuted-nuance).
- hypertools/config.py: importlib.metadata version + drop py<3.8 fallback (F23-009).
- hypertools/io/sources.py:254,388: exception cross-ref path canonicalization (F23-006) [A1's file — trivial, post-wave].
- pyproject.toml: numba>=0.59 floor INSIDE the umap extra (X5-003 proposal adjusted — not a core dep).

## Wave 5B/B5/controller COMMITTED

- `82dc8cd0` wave 5B (127 findings, plot package; full suite 2049 green + docstring-gate fix).
- `5ddbbf3b` controller batch (exports, vals2colors coverage, config version, numba floor, CLAUDE.md order, api.rst HyperAnimation/supported_models/Exceptions).
- `e8e8b9ae` B5 escalations (16 items: cluster-spec unification via _resolve_cluster_spec, int-hue categorical palette, stream kwarg warnings, stale PPCA text, X6 leftovers). Full suite 2088 passed.
- Note: B5 caught + fixed stale test_helpers expectations from MY controller commit — cross-checking worked as designed.

## Wave 5C COMPLETE + COMMITTED (100 fixed, 68 verified already-fixed)

- `dc063dc6` C5 docstring residuals (13 fixed, 29 already-fixed verified).
- `cc0ccf3f` C1-C4: README all-blocks-verbatim + CHANGELOG.md created; 32 examples fixed / 40 executed with judged evidence (plot_sotus RESTORED to real 29-address demo); ALL 15 tutorials re-executed fresh in place, 0 error cells (D07-001 critical fixed; real pylsl outlet; live yfinance + cached fallback CSVs).
- `78bd7212` cleanup: 20 stale tracked docs/modules/generated + orphaned spin.gif removed.

## Wave 5D RUNNING (`wf_e2894081-f6c`, 2 agents)

D1 code residue: NEW streaming-ndims regression (our own fix tripped a new warning in tutorial outputs — fix + re-execute 3 notebooks), streaming save/.mp4 + peek + plotly-backend + lsl validation, reduce/describe warnings, predict/impute polish, plot leftovers verification, 32 docstring underlines, 11 http links. D2 docs infra: CONTRIBUTING modernization, doc_requirements, Makefile, post_build, gallery CSS, favicon, analyze.ipynb link; release-time pin swap list → needs_controller.

## Phase 6a results (2026-07-17)

- Docs gate CLOSED: full forced-regeneration build succeeded; 22 title-overline warnings fixed in example sources; second build ZERO warnings. Gallery committed `c7adf48b`.
- Full suite: **2113 passed, 1 failed** — `test_default_options_load_path_independently`. ROOT CAUSE: site-packages again holds a REAL hypertools snapshot (old configurator, no config.ini) — the editable install was clobbered DURING Phase 6b, almost certainly by the R8 packaging re-auditor pip-installing a wheel into the shared venv (my re-audit prompt omitted the no-pip-install ban the fix-wave prompts had). Repo-cwd tests unaffected (import repo tree); only the cwd-independent subprocess test sees the snapshot. FIX AFTER 6b LANDS: pip uninstall hypertools + `pip install -e . --no-deps`; re-run the test; verify /tmp import; confirm from R8's report.

## Reconciliation state (post-5D, cluster inheritance applied)

516 fixed/already-fixed · 54 skipped (cross-referrals) · 18 escalated (handled) · 12 env-resolved · 5 dropped (refuted) · 5 deferred (structural, justified) · **98 UNRESOLVED — all X-unit re-findings needing code-state verification** (incl. 2 majors: X3-002 static-line ~ fixed as F01-001 but title dodged the cluster regex; X8-001 mixed multi-row+1-row pchip crash — verify). Plan: after 6b + venv restore, one verification agent checks all 98 against current code → true residue gets a final fix round.

## Phase 6b results (11/11 agents, 2026-07-17)

**366 fixes VERIFIED by fresh adversarial re-runs** (io 31/31, manip+align 28/28, predict/impute 35/35, cluster/reduce/analyze 54/55, backend/core 41+, plot areas high-90s%). Environment recurrence: an agent pip-installed pre-audit code (git@e0f4e33e) into the shared venv at 04:25 — re-auditors caught it, forced repo sys.path, results valid; venv restored by controller (editable, /tmp import verified, packaging tests 6/6). PREVENTION: 6c prompts ban pip install outright.

Real residue → wave 6c (RUNNING, `wf_5a36559a-342`, 4 agents):
- G1: unfixed F08 input family (Categorical, MaskedArray silent-unmask, datetime64, type-naming, nested arrays, vectorizer-typo 401, bool list, DataFrame axis labels) + plot docstring cites + plotly px comment + dup dims warning.
- G2: dispatcher consistency majors (align duplicate-index silent misalignment; minimal dict {'model':'PCA'} crash; cluster args-key discard) + params/kwargs warning parity, describe vstack, manip False-skip + assert fixes, None/empty-input unification, Pipeline dup steps, AutoRegressor lags bounds, impute empty-vs-NaN, datetime horizon, tuple input, Smooth NaN.
- G3: security/quality on our own fixes (0600 mkstemp mode demotion, PermissionError wrapping, gzip-bomb cap, extensionless-pickle sniff) + stream HEAD-phase salvage, ffmpeg precheck, LSL resolve, lsl.py:63 ref.
- G4: __init__ docstring FALSE import claim (controller's own miss — caught by re-audit), CLAUDE.md interactive.py shim note, readme absolute image URLs (PyPI), sdist tests graft, py3.14 classifier, SPDX license form.

Also cleaned: stray 102MB hypertools-dev venv + ancient dist/ artifacts deleted + gitignored (`9d2acf7a`).

AFTER 6c: full suite → reconciliation-verification agent for the 98 X-items (against settled code) → Phase 7.

## Wave 6c COMMITTED + X-reconciliation (2026-07-17/18)

- `ac854046` wave 6c (42 items; full suite **2258 passed / 0 failed**).
- X-reconciliation (2 verifiers over the 98 open X-ids): **53 fixed** (by earlier waves under other ids), **5 by-design** (documented), **40 still-present minors** → wave 6d.
- Wave 6d RUNNING (`wf_c5dcbd70-c61`): H1 validation/warnings polish (SRM features, describe max_dims, apply_model ndims parity, plot kwarg types, n_clusters=0, align 3-D rejection, hyper-alias deprecation, cluster int types, predict seam row, stacklevel sweep, UMAP/HyperAnimation/arima warning hygiene, PPCA rank-deficiency error, dw Pandas4Warning targeted filter) + H2 code-org/licensing (procrustes dedupe + index param, brainiak Apache-2.0 header + pca-magic license text, shared-helper dedup, parse_args removal, context.py removal, __all__, Clusterer/Reducer exports, dev/+RELEASE_NOTES cleanup, CLAUDE.md _externals fix, describe helper tests).
- Deferred-with-justification (documented, not code-fixed): X1-005 data-arg naming unification (API rename too invasive pre-release), X1-010 public seeds for align/predict/impute (erroring beats silent no-op; enhancement for 1.1), X8-005 stale GH issue comments (handled at Phase-8 report time), X8-007 fig.number cosmetics (deliberate unregistration prevents leaks), X7-022 configurator published-record intent (docstring states it).
- PR evidence curator running (staging ~20-24 before/after images + manifest to scratchpad).

## Phase 6 plan (after 5D commits)

1. FULL suite gate. 2. `make clean && make html` FULL docs rebuild (regenerates all 54 gallery examples against fixed code — catches example breakage, refreshes auto_examples + thumbnails; commit regenerated artifacts; resolves D02-002/003/004). 3. Independent re-audit: ~8 fresh adversarial agents over the fixed areas (try to BREAK the fixes + hunt regressions). 4. Whole-branch independent review (quality/security/simplicity) over dev-1.0-refactor..HEAD. 5. Final reconciliation (5C/5D merged + dedup-cluster inheritance). Then Phase 7 merge + CI.

## Controller integration queue (verify/do after waves land)

1. Verify F15-005: hyperalign unknown-kwarg (n_itr) now raises by name (A3 claims fixed).
2. Verify X2-005 remainder: impute([]) and predict([]) raise no-data errors (A4 contract).
3. Verify corpus='sotus' end-to-end through text2mat after A1 lands registry fix.
4. F06-010: plotly_backend.py:838 title-size comment (2-line fix, do at integration — B-pipeline owns file now).
5. F06-009 + F01-014: document plot()-shadows-subpackage quirk (recommend `from hypertools.plot import backend`) — 5C docs wave; NOTE F01-014 verdict: `import hypertools.plot.plot` works; only attribute access fails.
6. F15-007/008: docs/tutorials/analyze.ipynb cell-6 comment + cell-21 params→kwargs — 5C docs wave.
7. A2 note: docs/pipeline_order.rst could add one-line shared-stats caveat for ZScore/Normalize on lists — 5C.
8. F06-006: save_path .html plotly-only docstring — covered by B3 brief; verify.

## Workflow runs

- `wf_bc33c3c0-640` — Phase 1-3 red-team wave 1: 14/46 completed (+2 orphan-valid), 30 spend-capped, 1 server error. 3.83M subagent tokens, 69 min.
- wave 2 = resume of the same run (see below for run id once launched)

## Seed observations (controller pre-warm, 2026-07-11 00:47)

1. ✓ CONFIRMED as F18-load-hosted-001 (critical): `hyp.load('sotus')` returns a broken sklearn Pipeline — sha-identical to nips_model (Drive id duplicated).
2. ✓ CONFIRMED as F18-load-hosted-002 (major): hosted *_model pickles are sklearn-1.0.2; version warnings + repr crashes under sklearn 1.8.
3. Baseline: weights=list(36), spiral=list(2), mushrooms=DataFrame(8124).
