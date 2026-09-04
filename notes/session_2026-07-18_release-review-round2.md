# Release review round 2 (2026-07-18) — Jeremy's 2nd review: 2 blockers + 6 findings

Base branch: `dev-1.0-refactor` (NEVER touch master). venv: `.venv/bin/python`, `MPLBACKEND=Agg`.
Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## Status

HEAD: `e3963ec7` (Phase A `6fd02d19` + CI-green fixup `e3963ec7`). Blocker #1 (sotus) still PENDING upload.

### CI-green fixup e3963ec7 (after first CI run 29648935938 found 3 issues):
- **mushrooms compat hash** failed on py3.10 (pandas2 `object` vs pandas3 `str` dtype, values identical). FIXED: normalized canonical hash (object/str/category→'text' on str values). PROVEN identical under pandas 2.3.3 + 3.0.3 reading cached parquet. Baseline regenerated via committed `scripts/gen_rehosted_compat_baseline.py` (imports test's hash fn).
- **docs-clean** failed at nbsphinx notebooks = missing `pandoc`; added to docs-clean apt + .readthedocs.yaml. The GALLERY (incl. both Plotly examples) built CLEANLY → blocker #2 Chrome fix CONFIRMED by CI.
- **windows-3.13 test_plotly_gif_export** TimeoutError = pre-existing kaleido→Chrome flake; pre-fetch Chrome in test matrix (best-effort, continue-on-error).

### DONE (Phase A — code-only, verified locally, ready to commit+push):
- **#2 (blocker) RTD Plotly/Chrome**: `.readthedocs.yaml` post_install `plotly_get_chrome -y`;
  new CI job `docs-clean` (git archive → assert docs/auto_examples absent → Chrome → `sphinx -W -E -a`);
  `docs/conf.py` fallback comment corrected. PROVEN locally: plotly_get_chrome + kaleido rendered a real PNG.
- **#3 datasaurus indexes**: root cause = convert script did `np.asarray(df)` dropping index+cols.
  Original frames = 13 shuffled contiguous 142-row blocks; index = global row range.
  FIX: `_DATASAURUS_INDEX_STARTS` constant in load.py (142,1278,1136,0,994,284,852,1562,1420,710,426,1704,568);
  `_parse_rehosted` npz_df_xy rebuilds `pd.DataFrame(a, columns=['x','y'], index=arange(s,s+len(a)))`.
  PROVEN: frame-by-frame `.equals(pre-1.0 original)` == True. NO datasaurus re-upload needed (hosted npz unchanged).
  Committed baseline `tests/data/rehosted_compat_baseline.json` (16 datasets) + `tests/test_dataset_compat.py`.
- **#4 trust docstrings**: sources.py `load_source` + `_parse_payload` now say REFUSES/raises HypertoolsTrustError.
- **#5 changelog**: dropped "small ... base install" wording, mirrors README ("not a minimal footprint").
- **#6 atomic downloads**: `_download_example_data` writes temp (tempfile.mkstemp in DATA_DIR) → integrity → `os.replace`;
  soft `filelock` per-dataset lock (`_dataset_download_lock`, filelock is optional so soft-import). `_download_example_data_once(dest,name)` sig changed. 3 new real thread/atomicity tests.
- **#7 merge strategy**: CONTRIBUTING.md "Merging (maintainers)" = squash-merge (Jeremy forbade history rewrite). .git=1.4G, tree 115M/680 files.
- **#8 sdist smoke**: test.yml wheel-smoke job now installs+smoke-tests BOTH wheel AND sdist in separate venvs. PROVEN: fresh-venv sdist install + smoke = exit 0.

### PENDING (Phase B — blocker #1 sotus, needs Jeremy Dropbox upload):
sotus currently loads via datawrangler (allow_pickle=True, no HT checksum) = supply-chain bypass.
- **FILE BUILT + VERIFIED**: `~/Desktop/sotus.json.gz` (also scratchpad/datasets-out/...), 359240 bytes,
  sha256 = `20068fb4fe21a171c6c40a788c122ede30648e10ea44c3ea955301cfabc4c7b3`, 29 speeches, round-trip identical to hyp.load('sotus').
- **AWAITING**: Jeremy uploads to same Dropbox folder → gives `?dl=1` link.
- **THEN apply Phase B loader diff (below), push, CI green → blocker #1 closed.**

## Phase B sotus loader diff (apply AFTER upload, with real URL):
1. `EXAMPLE_DATA['sotus']` = the new Dropbox `?dl=1` URL (replaces `'datawrangler-zoo:sotus'`).
2. Delete the `# 'sotus' loads via datawrangler...` comment above EXAMPLE_DATA.
3. `_REHOSTED['sotus'] = 'jsongz_strlist'` (+ add to the header comment).
4. `_parse_rehosted`: add `if fmt == 'jsongz_strlist': gzip+json.load → return [str(d) for d in docs]`.
5. `_EXAMPLE_DATA_SHA256['sotus'] = '20068fb4fe21a171c6c40a788c122ede30648e10ea44c3ea955301cfabc4c7b3'`.
6. Remove `_load_sotus_corpus()` and the `if dataset == 'sotus': return _load_sotus_corpus()` branch in `_load_example_data`.
7. Tighten `tests/test_dataset_integrity.py::test_all_downloadable_builtins_are_pinned`:
   drop the datawrangler exclusion; require EVERY EXAMPLE_DATA name pinned + assert NO `datawrangler-zoo:` entries remain.
8. Verify `tests/test_io_audit_load.py::test_load_sotus_returns_the_29_speeches` still green (loads from json.gz; no chatter).
9. compat baseline already has sotus hash `e6b8a49f...` (datawrangler-derived == json.gz result), stays green.

## Verification commands
- `MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_dataset_compat.py tests/test_dataset_integrity.py -q`
- Full suite gate: `MPLBACKEND=Agg .venv/bin/python -m pytest -q` (~8 min, was 2357 passed).
- scratchpad scripts: gen_compat_baseline.py, build_sotus.py (throwaway; delete or move to scripts/ when done).
