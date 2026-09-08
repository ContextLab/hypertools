# v1.1 release audit — paused September 7, 2026

> **Reading guide.** Everything from here down to the heading
> "## Updates by the Claude session" is the ORIGINAL Codex audit checkpoint,
> unchanged. The section under that heading was appended afterwards by the
> Claude Code session that implemented fixes; it states, per original finding,
> what changed and how it was verified. A resumed audit should treat the
> original sections as its own record and the update section as claims to
> re-verify.

## Request, constraints, and status

Jeremy requested a detailed release-readiness audit of open PR #286 AND all
changes since v1.0.0, including changes already merged into master,
documentation, descriptions, tutorials, packaging, and release operations.
**Do not implement fixes or publish anything: deliver findings and ALL next
steps.** Another Claude Code session is running tests and editing concurrently.

Jeremy paused this audit because credits were running low and explicitly
authorized writing this notes file. This is a checkpoint, **not the final
report**. No library, test, documentation, git-index, or remote changes were
made by this audit. Scratch copies, logs, reproductions, builds, and executed
notebooks were written outside the checkout. This notes file is the only
intentional repository edit by this audit.

**Provisional recommendation: do not release yet.** Two demonstrated control
failures (offline loading and automatic-install opt-out), panel composition
defects, and failing documentation examples remain. Existing green CI is
valuable but does not cover all these paths or the staged changes.

No subagents were used. Continue read-only apart from updating this report.

## Exact source states

- Repository: `/Users/jmanning/hypertools`, origin ContextLab/hypertools.
- PR: https://github.com/ContextLab/hypertools/pull/286
  “Fix 1.1 release audit findings and complete documentation follow-ups”.
- PR base/master: `96ac8b7f43c132f6f455ad1be3ffc98e84adead5`.
- Pushed PR head: `687e98c90a297951863f07ae1bd6b40d53c4efdb`.
- Local HEAD at audit start: `f4f7ab8d2d596582298879e97661a7084f4cda97`
  on `fix/1.1-release-review`; extra local commit is session notes.
- v1.0.0: `647ce929fb0fcc39dfbe17d73282ee54dbe4aaf6`.
- v1.1.0 tag/draft still points to `96ac8b7f...`, excluding the PR fixes.
- 105 commits in `v1.0.0..HEAD`; 32 in `origin/master..HEAD`.
  Full diff: 469 files, 114,634 additions, 7,488 deletions (includes notebooks,
  planning/history and tests; not all runtime code).
- Substantial staged changes existed before this audit, notably new public
  `hyp.set_autoinstall`, its docs/tests, and notebook installation-cell cleanup.
  These are NOT covered by the green pushed-PR CI.

An initial tracked-file working-tree snapshot was captured at
**2026-09-08 03:12:53 UTC** (September 7 local time):

`/tmp/hypertools-audit-20260907/snapshot`

It includes the staged working source, excludes generated build/gallery trees,
and contains 943 files. Alongside it are `manifest.json` (SHA-256 per file),
`status`, `refs`, `cached.patch`, and `unstaged.patch`. Use these to distinguish
the reviewed state from concurrent edits. NOTE: executing Sphinx/notebooks and
building artifacts subsequently generated files and videos in the TEMPORARY
snapshot; the initial manifest remains the original source fingerprint.

At pause, concurrent changes since that capture included
`.claude/CLAUDE.md`, the two autosummary stubs
`docs/hypertools.FrameContext.rst` and `docs/hypertools.io.LSLStream.rst`,
`notes/session_2026-09-05_release-1.1-review.md`, and
`tests/test_lsl_streaming.py`. HEAD was still `f4f7ab8d` at the last check.
Do not revert, stage, or otherwise disturb the other session's changes.

## Confirmed findings to carry into the final report

Line numbers below refer to the captured working source and may shift.

### 1. High: `offline=True` still downloads hosted built-in datasets

- `hypertools/io/load.py:487` promises “never open a connection”.
- `_resolve` at approximately lines 651–653 calls `_load_example_data(dataset)`
  without passing the offline setting.
- `_load_example_data`, approximately lines 814–841, downloads on cache miss
  and deletes/redownloads a corrupt cached file, independently of `offline`.
- **Real reproduction:** point the module's `DATA_DIR` at a fresh temporary
  directory, register a passive `sys.addaudithook` recording `socket.connect`,
  and call `hyp.load('spiral', offline=True)`. It returned two datasets,
  created the cache file, and attempted two actual HTTPS socket connections.
  No network functions were mocked; the user's real cache was untouched.
- Evidence: `/tmp/hypertools-audit-20260907/offline-builtin.log`.
- Fix recommendation: propagate offline policy into the hosted-dataset cache
  path; serve hash-valid cached data, but raise `HypertoolsOfflineError` for
  absent/corrupt entries without downloading. Cover cache miss AND corruption,
  including hosted `*_model` datasets. Preserve integrity verification.
- Current offline tests cover URLs and iris/helix/local files, missing this path.
  This feature already exists on master; it is not only a staged-change issue.

### 2. High, staged API: `set_autoinstall(False)` is lost in animation export

- New API stores its effective setting in process-global `_AUTO_INSTALL` in
  `hypertools/_shared/lazy_import.py` (around lines 65–155).
- `hypertools/plot/plotly_backend.py:2279` routes animated file exports to
  `_export_animation_file`, then `_render_frames_via_subprocess`.
- The `subprocess.Popen` around line 3226 does not propagate the effective
  Python setting. The worker calls `ensure_kaleido_chrome` at
  `hypertools/plot/_kaleido_export_worker.py:57–58` in a fresh interpreter.
- **Real reproduction:** separate temporary interpreter using existing real
  dependencies, with kaleido genuinely absent. Parent calls
  `hyp.set_autoinstall(False)` then saves a tiny Plotly animation to GIF.
  Parent reports False; child reports True; child invokes
  `python -m pip install -q 'plotly>=6.1.1' 'kaleido>=1.0'` anyway.
- Pip was deliberately prevented from accessing indexes (`PIP_NO_INDEX=1`,
  `PIP_CONFIG_FILE=/dev/null`); no dependency was installed or changed.
  Returned exception was a worker RuntimeError containing the pip failure,
  instead of respecting the no-install policy.
- Evidence: `probe_export_policy.py`, `export-policy.log`, `exportenv/` under
  `/tmp/hypertools-audit-20260907/`. Final rerun imported snapshot code in BOTH
  parent and child (run from `/tmp`, not the checkout).
- Fix recommendation: explicitly pass the effective setting to subprocesses,
  preserving Python-over-environment precedence in both directions, and test
  public animated export with a genuinely missing dependency. The same lost
  policy can affect Chrome provisioning. Do not claim that apt/Chrome actually
  ran in this reproduction: the demonstrated action was pip invocation.
- `tests/test_lazy_import.py:131` also has a stale exemption comment claiming
  parent provisioning always precedes worker launch; animation export does not.

### 3. Medium: shared panels discard their fitted pipelines

- `_plot_panels` (`hypertools/plot/plot.py:3176–3220`) obtains a real fitted
  bundle from its shared probe, but passes panel slices through `transform=`
  with `pipeline=None`, and returns those bundles without retaining the probe
  pipeline (return near lines 3306–3316; Plotly equivalent near 3410).
- Reproduction, both backends:

  ```python
  data = [np.random.default_rng(i).normal(size=(20, 5)) for i in range(2)]
  b = hyp.plot(data, panels=True, reduce='PCA', return_model=True,
               show=False, backend=backend)
  b['panel_models'][0]['pipeline']  # None
  ```

- Calling `.transform` on that value fails; fitted shared PCA cannot be
  recovered from the returned panel bundles. A regular non-panel bundle
  returns its fitted pipeline. Current panel tests check coordinates and keys,
  not replay of the returned model.
- Fix recommendation: preserve and expose the actual shared pipeline (define
  its multi-dataset replay semantics clearly), and test held-out replay without
  refitting on both backends. This is a new-panel-feature gap, not proof that
  ordinary `plot(return_model=True)` replay is broken.
- Initial reproduction output was in the conversation, not a dedicated log;
  code and result above are sufficient to reproduce. Other probes are saved.

### 4. Medium: panels do not partition valid palette/forecast arguments

Both backends reproduced these. `_PANEL_PER_DATASET_KWARGS` at
`hypertools/plot/plot.py:2761`, and slicing loops around 3159/3214, are the
starting points. Test shared AND independent panel fits when fixing.

- `hyp.plot([a,b], palette=['viridis','magma'], panels=True, show=False)`
  raises “2 per-dataset palettes but 1 dataset(s)”; the same call without
  panels succeeds. Nested explicit per-dataset color lists fail similarly.
- `predict='Kalman', t=3, forecast_fmt=['--', ':'], panels=True` raises
  “got 2 for 1 forecast(s)” instead of assigning one style to each panel.
- `predict='Kalman', t=3, forecast_palette=['red','blue'], panels=True`
  draws BOTH panel forecasts red. Actual artist/trace colors were inspected.
- A two-model/two-dataset `forecast_hue=['a','b','c','d']` (documented
  model-major form) works without panels but fails inside each panel, which
  still receives all four values for its two forecasts.
- A forecaster fitted on both datasets works in the ordinary plot path but
  fails under panels: each panel receives the entire two-dataset model and
  `predict_new` rejects “1 new dataset(s) ... 2 fitted model(s)”. Review whether
  the existing `Forecaster.for_dataset(i)` should be used here.
- Evidence: `probes.py` and `probes.log` under the scratch directory;
  per-dataset palette WITHOUT hue was verified in an additional inline probe.
- Fix recommendation: resolve per-dataset versus per-model ownership before
  splitting panels; preserve palette assignment, slice all per-forecast forms
  correctly, and select fitted-model dataset state appropriately.
- **Avoid a false positive:** per-dataset palettes combined with CONTINUOUS
  hue are explicitly unsupported in the docstring. The earlier exploratory
  probe of that combination failed as designed and is not a finding.
- Likewise `resample=10` with original-length 20-row hue was exploratory;
  do not report it without checking the documented post-resampling contract.

### 5. Medium: executable load documentation has two failures

- Sphinx doctest execution: **315 examples, 2 failures**, both on
  `hypertools.load`. Its parameter documentation includes
  `hypertools.load(arr)` and `hypertools.load([arr, 'spiral'])` after importing
  numpy but before importing `hypertools` (`hypertools/io/load.py:345–349`).
  Both raise `NameError: name 'hypertools' is not defined` in that context.
- Evidence: `doctest-network.log` and `doctest-network/output.txt`.
- Fix recommendation: make the example self-contained, run Sphinx doctests
  as a release gate (ideally CI), and audit other literal examples separately.
- Existing HTML CI does NOT run the doctest builder. The hierarchy pytest
  file checks structure/prose, despite its stale opening docstring claiming a
  `test_every_doctest_in_the_guide_runs` test exists. The hierarchy doctests
  themselves passed in this audit.
- Command used `-D plot_gallery=0` to avoid re-executing the gallery during
  doctests. This produced an extra Sphinx config-type warning (string vs bool)
  caused by this audit invocation, **not a package defect**. On resume rerun
  with a programmatic boolean config override or appropriate typed setting to
  remove that warning. The two NameErrors are independent real failures.
- First sandbox run could not access datasets; do not count its network
  failures. The network-enabled rerun is the authoritative result above.

### 6. Release/documentation operations still incomplete

- PyPI JSON was checked: latest **1.0.0**, no **1.1.0** release entry.
- GitHub v1.1.0 is a draft, tag at old master `96ac8b7f`, not the PR fixes.
- PR release-gate is SKIPPED by design; it only runs on master/tag pushes.
- README's current optional-dependencies URL returns **404**:
  https://hypertools.readthedocs.io/en/latest/optional_dependencies.html
  The page exists locally; final RTD deployment must be verified.
- Checked 64 external URLs from README, top-level docs and tutorial markdown:
  58 successful; that one 404; four Wikimedia URLs returned 403; Princeton
  DataSpace provenance link returned 401. Treat 403/401 as access checks needing
  human/browser verification, not established dead links.
- Evidence: `doc-urls.json`, `url-results.json`, `urls.log`.
- #284 and #285 remain open. The former still has an unchecked final
  CI/tag/draft-release item; the latter has all body checkboxes checked. Both
  are intended to close on merge of #286. Saved bodies/comments in
  `issue284.json` and `issue285.json`.
- PR body still summarizes an earlier review/test count and omits much of the
  subsequent plotting work. Refresh its description/evidence for the final
  implementation, and do not let automatic issue closure imply release
  publication has already happened.

## Other observations — not established blockers

- `alignment_score([constant, constant], metric='dispersion')` returns NaN
  with a RuntimeWarning; `metric='isc'` correctly rejects all-constant input.
  Inputs with NaNs yield NaN scores; 1-D arrays produce low-level shape errors.
  These are robustness/documentation candidates, not demonstrated corruption
  for the documented finite 2-D nondegenerate inputs. Evidence: `probes.log`.
- Existing scoring intentionally warns about unscored cells and can rank a
  model with incomplete coverage. Do not relabel documented behavior a bug
  without a concrete contradictory use case.
- `docs/doc_requirements.txt` still repeats older core lower bounds
  (sklearn 1.4.0, pandas 2.2.0, matplotlib 3.8.0) while pyproject/README have
  corrected floors. Since CI installs the package first, this did not show a
  broken resolution; recommend removing redundant drift or synchronizing it.
- Some staged tests stub `_pip_install` despite the project's no-mocks policy;
  prefer real isolated dependency-absence checks for the final regression tests.
- Existing important fixes in the PR include ownership/leakage protections in
  backtesting, Delay feature-label collisions, cache-write concurrency, font
  precedence, optional NumPy-2-compatible floors, forecast ownership/styling,
  hierarchy replay, and panel rendering. Reviewed their code/tests and prior
  findings; do not present already-fixed issues as remaining defects.

## Validation completed in this audit

1. GitHub PR #286: **16 successful jobs** at pushed `687e98c9`:
   12 OS/Python matrix jobs (Ubuntu/Windows/macOS × Python 3.10–3.13),
   wheel-smoke, docs-clean, dataset-gate and live-source-gate. Release-gate
   skipped. No formal review decision was set; merge state CLEAN at inspection.
   Run: https://github.com/ContextLab/hypertools/actions/runs/34176254864
   This is hosted evidence, not a fresh full-suite run by this audit.
2. Sphinx doctests: 313/315 pass; two documented failures above.
3. **24 of 25 tutorial notebooks freshly executed successfully**, using the
   snapshot code and real data/models in the existing dependency environment:
   normalize, cluster, reduce, align, analyze, pipelines, hierarchy,
   projectile_kalman, manip, plot, text, stock_forecasting,
   modern_sklearn_dynamics, animate_forecast, io, streaming_data,
   hugging_face_embeddings, wikipedia_embeddings, conversation_trajectories,
   conversation_shape, painting_embeddings, morph_shapes_zoo, market_sectors,
   and lsl_streaming. `market_sectors` passed in 279 seconds.
   **weather_decades was in progress when paused; do not count as passed.**
   See `notebooks/results.json`, `notebooks/lsl-result.json`, executed notebooks,
   `notebooks-network.log`, and `lsl-notebook.log`.
   - Package install cells were skipped so the reviewed package/environment
     would not be replaced. `HYPERTOOLS_AUTO_INSTALL=0` was set.
   - Existing dataset/model caches and installed optional dependencies were
     used. This does NOT prove fresh-machine installs or Colab first-use flows.
   - Each notebook had a fresh kernel. LSL passed with normal/default local
     discovery, not a special loopback config. Its synthetic outlet was closed.
   - No stored error outputs existed in any of the initial 25 notebooks.
   - Compared text outputs: expected stochastic clustering changes, path
     differences, file-byte counts, and existing GP convergence warnings; no
     confirmed tutorial claim contradicted by those comparisons so far.
     These scratch executions did not apply the repository helper's path scrub.
4. Full clean-source Sphinx HTML/gallery build was running, approximately
   **90% through gallery generation** at pause (`animate_surface_morph.py`).
   **Not a completed/passing build.** `html.log`, partial `html/`.
   Prior pushed PR docs-clean was green, but staged API docs require this check.
5. Wheel and sdist built from snapshot with `python -m build --no-isolation`.
   `twine check` passed for both. All **110 package files** in EACH match the
   snapshot byte-for-byte. Bundled fonts and MIT/Apache/OFL materials present;
   no notes/agent/CLAUDE files leaked into these artifacts.
   - Wheel SHA-256:
     `5aec0b6e4441545cf3efaf9418668c1fc217289198545b92e7768a34bacc933d`
   - sdist SHA-256:
     `1baa800ad25f297d42f9dde56b6949558e3f2bdc9d38c9713cb5ffbd7b337ca6`
   - `dist/`, `build.log`, `wheel-install.log` in scratch directory.
   - Installed built wheel into an isolated target; repository
     `scripts/wheel_smoke_test.py` **passed**, importing the installed artifact
     at `installed-wheel/site-packages/hypertools/__init__.py`.
     Dependencies were reused from the existing venv; this is NOT a fresh
     dependency-resolution environment. The sdist was inspected but not
     freshly installed in this audit.
6. Actual feature-combination probes on both plotting backends, and the real
   offline/install-policy reproductions described above.

Environment: Python 3.12.14; numpy 2.3.5; pandas 3.0.3; scikit-learn 1.8.0;
matplotlib 3.10.8; plotly 6.8.0; Sphinx 9.1.0; nbclient 0.11.0.

## Concurrent Claude test run

The other session's notes now report **5315 passed, 1 failed**:
`tests/test_lsl_streaming.py::test_lsl_stream_resolves_by_type`.
It attributed the failure to unrelated EEG outlets on this host/network and
changed the test to use a unique stream type; a full rerun was pending.
This is the OTHER SESSION'S recorded result, not independently inspected raw
pytest output. Obtain the final log and commit before citing a final suite pass.

Its note mentions an audit notebook kernel among other possible outlets.
Our LSL notebook used a distinct NAME, but default type EEG can still collide
with generic by-type discovery. On resume, use a private LSL SessionID/config
if repeating LSL work alongside tests; do not repeat the LSL notebook needlessly.
Do not stop other sessions' kernels or outlets. The audit did not modify the
test or global LSL settings.

## Pause mechanics and scratch inventory

At Jeremy's pause request, SIGINT was sent ONLY to the matching audit processes:

- `/tmp/hypertools-audit-20260907/run_notebooks.py`
- Sphinx HTML command whose output directory is
  `/tmp/hypertools-audit-20260907/html`

Both targeted `pkill` commands succeeded. Notebook client should clean up its
kernel on interrupt; confirm those jobs exited on resume before restarting.
Other sessions were left untouched. Useful tool session IDs, if still available:
notebooks `97634`; HTML `8218`. All other audit checks had completed.

Post-interrupt log confirmation: HTML ends `Interrupted!`. The notebook runner
recorded `weather_decades FAIL` with `zmq.error.ZMQError: Socket operation on
non-socket` as its kernel was interrupted. **This is a pause-induced incomplete
execution, NOT a confirmed tutorial failure.** Rerun that notebook on resume.

Scratch root: `/tmp/hypertools-audit-20260907/` (macOS may expose this as
`/private/tmp/...`). Temporary files may not survive a reboot; this notes file
preserves the essential findings and reproductions if that happens.

## Resume plan and final release next steps

1. Re-read current user instructions and this checkpoint. Re-query HEAD, status,
   PR checks, tag/master and draft state; diff current files against the initial
   manifest. Revalidate findings affected by Claude edits. Do not implement fixes
   unless Jeremy changes the read-only instruction.
2. Finish ONLY `weather_decades.ipynb`; do not rerun the 24 passing notebooks
   absent relevant code changes. Restart the HTML build in the scratch snapshot
   (gallery cache can reuse completed examples); record completed count/errors.
3. Once HTML is complete, run `docs/post_build.py` against the temporary build
   with `READTHEDOCS_OUTPUT` pointing there; validate internal file/anchor links,
   generated notebook install cells, and rendered pages with
   `scripts/verify_docs_playwright.py` using `HYPERTOOLS_DOCS_HTML` and
   `HYPERTOOLS_DOCS_SCREENSHOTS`. Visually inspect representative static,
   hierarchy, forecasting, panel, animation, and tutorial pages. This has NOT
   yet been done in this audit. Do not claim every figure was visually reviewed.
4. Rerun doctests with a correctly typed gallery-disable setting to cleanly
   separate the two actual example failures from the invocation warning.
5. Confirm focused panel findings with minimal logged examples, including
   shared/independent modes, and inspect any additional scientific or backward
   compatibility concerns that remain after reviewing the full v1.0 diff.
   The review is broad but not an exhaustive line-by-line verification of all
   114k added lines. Avoid unsupported “everything else is correct” claims.
6. Obtain Claude's final full-suite results/skips and source identity. Require
   a final full run and all hosted checks after ALL intended fixes are committed
   and pushed. Track bigdata/HF forecaster exclusions explicitly. Do not duplicate
   a running full suite without a reason.
7. Deliver a detailed prioritized report: recommendation, exact reviewed state,
   reproducible remaining defects, docs/validation findings, coverage limits,
   and complete release checklist. Include source lines and evidence links.

For the eventual release (recommendations only; no authorization to execute):

- Fix the confirmed defects with real behavioral regression checks; propagate
  both plotting backends and test composition rather than just scalar forms.
- Freeze the intended release source, reconcile staged changes, refresh PR body,
  CHANGELOG date/notes and accurate feature descriptions, then obtain green
  final PR checks and merge. Resolve #284/#285's release bookkeeping explicitly.
- Follow `RELEASE_CHECKLIST.md`: generate/publish the gallery notebooks and
  manifest from the EXACT final release commit. The release gate requires
  `manifest.source_commit == HEAD`; any later commit requires regeneration and
  republication. Coordinate this with master checks so an old manifest is not
  mistaken for a code failure. Do not use this uncommitted snapshot's artifacts.
- Build fresh wheel/sdist from that clean final commit; twine-check, inspect
  contents/licenses, record digests, install EACH in a fresh environment and
  run end-user smoke tests. Exercise optional feature installs/opt-out/export,
  supported minimum dependencies, and known skipped integration paths.
- Require all master gates; move the still-unpublished v1.1.0 draft tag to that
  exact commit only after checking publication state again; require tag CI.
- Upload the exact verified artifacts to PyPI, replace draft GitHub assets and
  notes, then publish. Verify version/install/import/README after publication.
- Build final tag docs on RTD, update stable/default appropriately, confirm
  latest/stable navigation, gallery Colab badges, tutorial media and the
  currently-404 optional-dependencies link. Test a fresh Colab first-use flow;
  local cached notebook execution is not a substitute.
- Review conda-forge version/dependency-floor update and fresh conda install;
  announce only after release/docs/install checks pass. Archive evidence and
  organize remaining nonblocking enhancements; branch cleanup is optional.

**Do not declare the audit complete or the release ready when resuming solely
because prior CI was green.** Finish the explicitly outstanding validation and
reconcile the confirmed findings with whatever has changed meanwhile.


## Updates by the Claude session (2026-09-07 late / 2026-09-08) — NOT part of the original audit

Written by the Claude Code session working on PR #286 after Jeremy handed over
this checkpoint. Each entry names the original finding, the fix, the tests,
and the commit once it exists. Line numbers refer to the working tree at the
time of writing. "UPDATE" entries may be re-verified by the next Codex round;
the original findings above are left untouched.

### UPDATE — Finding 1 (offline built-ins): FIXED
- `hypertools/io/load.py`: `_resolve` passes `offline=` into `_load_example_data`;
  a cache miss under `offline=True` raises `HypertoolsOfflineError` (from
  `hypertools.io.sources`, where the class lives) BEFORE the data directory is
  created; a cached file failing its SHA-256 pin raises the same error and is
  left in place (online still deletes and re-downloads). The `*_model` pipelines
  share the path. The `offline` parameter docstring states this.
- Tests (`tests/test_load_offline.py`, real observables: a passive
  `sys.addaudithook` on `socket.connect` installed before importing hypertools,
  DATA_DIR redirected to a temp dir, plus the existing blackhole proxy):
  `test_offline_refuses_an_uncached_builtin_without_any_connection`
  (spiral, weights, wiki_model), `test_offline_refuses_a_corrupt_cached_builtin_and_keeps_the_file`,
  `test_offline_serves_a_verified_cached_builtin_without_any_connection`.
  The miss test fails against the pre-fix `load.py` (checked).
- `HypertoolsOfflineError` docstring (`hypertools/io/sources.py`) now covers
  hosted built-ins.

### UPDATE — Finding 2 (`set_autoinstall(False)` lost in animation export): FIXED
- `hypertools/_shared/lazy_import.py`: new `subprocess_env(env=None)` returns a
  copy of the environment with `HYPERTOOLS_AUTO_INSTALL` set to '1'/'0' from
  the parent's EFFECTIVE `auto_install_enabled()` (Python-over-environment in
  both directions). `hypertools/plot/plotly_backend.py::_render_frames_via_subprocess`
  launches the worker with `env=subprocess_env()`. The only other subprocess
  sites (`taskkill`, `ffmpeg`) run no hypertools code.
- Tests: `tests/test_animation_export.py` builds, per the audit's own method, a
  `venv --without-pip` whose site-packages symlinks every entry of the dev
  environment except `kaleido*` (preflight: `find_spec('kaleido') is None`),
  runs this checkout with `PIP_NO_INDEX=1`, `PIP_CONFIG_FILE=os.devnull`:
  parent `set_autoinstall(False)` -> the export raises the ImportError naming
  `pip install "hypertools[interactive]"` and `set_autoinstall(True)`, no
  install notice, no pip text, kaleido still absent; parent Python True over
  env 0 -> the worker reaches (blocked) pip. Both tests fail with the `env=`
  line removed. `tests/test_lazy_import.py`: the stale worker exemption comment
  is reworded; the `_pip_install` spy test is replaced by a real-observable
  test (capsys: no install notice; policy error text; module still absent);
  new `test_subprocess_env_carries_the_effective_setting_to_a_child_interpreter`.
- docs/optional_dependencies.rst says the setting reaches the export subprocess.

### UPDATE — Finding 5 (load doctest failures; doctest gate): FIXED (docstring); gate: local pipeline
- `hypertools.load` docstring example now imports hypertools before using it.
  `sphinx -b doctest -D plot_gallery=False`: 316 tests, 0 failures
  (`hypertools.load` 15/15); the remaining warning is the str-vs-bool `-D`
  invocation warning the audit noted, not a package defect.
- `tests/test_docs_hierarchy_guide.py` docstring corrected (no
  `test_every_doctest_in_the_guide_runs` exists; the guide's doctest blocks run
  under `make doctest`).
- CI doctest job: not added in this pass (see the open items below).

### UPDATE — Finding 6 (release/documentation operations): PARTLY
- `docs/doc_requirements.txt` core floors synchronized with pyproject
  (scikit-learn>=1.4.2, pandas>=2.2.2, matplotlib>=3.9.0), with a comment that
  pyproject is the declaration.
- README optional-dependencies URL: the page exists on this branch
  (`docs/optional_dependencies.rst`); RTD "latest" builds master, so the 404
  persists until #286 merges. Re-check after the merge and the RTD build.
- PR body refresh, #284/#285 bookkeeping, tag/draft/PyPI steps: to be done at
  merge/release time per RELEASE_CHECKLIST.md (unchanged position: nothing is
  published until Jeremy merges).

### UPDATE — Other observations: `alignment_score` degenerate input: FIXED
- `hypertools/align/score.py`: inputs must be 2-D numeric arrays of finite
  values (clear `ValueError` naming the dataset and its shape / NaN count);
  `metric='dispersion'` raises `ValueError` when every dataset is constant
  (was NaN with a RuntimeWarning), matching `'isc'`. Docstring `Raises` updated.
  Tests: `tests/test_align_score.py` (+6).
- Scoring with unscored cells: left as documented (the audit did not call it a bug).

### UPDATE — Finding 3 (panel bundles drop the pipeline): FIXED
- `hypertools/plot/plot.py` (`_plot_panels` / `_plot_panels_plotly`): the shared
  probe's fitted pipeline is stored in every `panel_models[i]['pipeline']` and
  as top-level `bundle['pipeline']` (None for independent / per-reducer grids,
  where each panel bundle carries its own fit). Replay semantics in the
  `panels` docstring: a shared-fit panel's pipeline is THE shared pipeline, so
  `.transform(held_out)` projects into the same space without refitting.
- Tests (`tests/test_plot_panels_audit.py`): held-out replay equals the
  single-axes bundle's transform and differs from a fresh fit, both backends,
  shared and independent modes.

### UPDATE — Finding 4 (panels do not partition arguments): FIXED
- `hypertools/plot/plot.py`: one narrowing rule (`_panel_narrow_kwargs`) both
  fit-mode loops call: per-dataset lists (`palette`, `legend`, `alpha`,
  `marker`, `linestyle`, colour lists ...) are sliced to the panel's dataset;
  per-forecast lists (`forecast_fmt`, `forecast_hue` model-major,
  `forecast_palette` pre-resolved grid-wide so labels keep the single-axes
  colours) are picked with the same model-collection splitter `plot()` uses;
  a forecaster fitted on every dataset is bound with `Forecaster.for_dataset(i)`
  (recursing into collections; clear ValueError on a count mismatch). The
  colour-tuple guard now applies to `color`/`colors` only (an `alpha=[.3,.6,.9]`
  list was read as one RGB triple).
- Tests: `tests/test_plot_panels_audit.py` (33 in total with finding 3):
  palette names + nested colour lists, alpha x3, legend list, forecast_fmt,
  forecast_palette (list and 'husl'), model-major forecast_hue (2 models x 2
  datasets; colour + linestyle vs the single-axes figure), fitted forecaster
  (equals the ordinary path, differs from a refit; mismatch raises), and a
  docstring-vs-roster test scanning `inspect.getdoc(hyp.plot)` for every
  per-dataset / per-forecast / model-major parameter. Existing panel suites
  (123) and the round-3/4 suites (34) still pass.
- Found while fixing: the ORDINARY path rejected the documented list of
  `{category: color}` dicts (one per dataset naming its own categories) once
  `hue=` regrouped two datasets into four runs ("lists 2 per-dataset palettes
  but 4 dataset(s)"). `_seaborn_palette_arg` now treats an all-dict list like
  a single dict (ambient cycle = default palette; the dicts resolve by name).
  Test: `tests/test_plot_palette_forms.py::test_list_of_category_dicts_with_distinct_categories_per_dataset` (both backends).

### Open items after this pass
- A Sphinx doctest step in CI (docs-clean job) — not added yet; the local
  release pipeline runs the HTML build with `-W`; consider adding `-b doctest`.
- Final full-suite run, commit, push and hosted CI after ALL fixes (pending).

## Codex round 6 (resumed audit)

Reviewed 2026-09-08: HEAD / pushed PR #286 head
`3f4b087d02d4a5572ade50837ac040fa8a531c8c`, base/master
`96ac8b7f43c132f6f455ad1be3ffc98e84adead5`. Read this checkpoint in full,
CHANGELOG's release-review section, and `git log --oneline master..HEAD`.
Working tree was clean at start and immediately before this append. No
tracked file other than this append was intentionally changed, and no new
repository files were left. No other process's kernel or LSL outlet was
stopped. No subagents used. Evidence: `/tmp/hypertools-round6/` and
`/tmp/hypertools-round6-pytest.log`.

This run's sandbox disallows local socket binding, external DNS/network
from shell commands, and successful Chrome startup. Approval is unavailable.
These restrictions prevent completing the notebook/browser portions; they
are NOT evidence that those features fail on an unrestricted machine.

### Original findings 1–6: independently checked

Commands below use the repo's `.venv/bin/python`, with `MPLBACKEND=Agg`,
`MPLCONFIGDIR=/tmp/hypertools-round6/mpl`, `HYPERTOOLS_AUTO_INSTALL=0`, and
`show=False` for direct plot probes unless a policy test explicitly enables
installation in its throwaway interpreter.

1. **FIXED (original severity major): offline hosted datasets.**
   `hypertools/io/load.py:662,826,842`. Command:
   `.venv/bin/python /tmp/hypertools-round6/offline.py`.
   A passive socket.connect audit hook is installed BEFORE importing
   hypertools; DATA_DIR is redirected to a temporary pathlib.Path.
   Missing AND corrupt `spiral`, `weights`, and `wiki_model` each raise
   `HypertoolsOfflineError`; all show `connects=0`; corrupt bytes remain
   unchanged. A real pre-existing, verified spiral cache copy loads as
   `[(1000, 3), (1000, 3)]`, also with zero connects. Log: `offline.log`.
   No remaining fix recommended for this reproduction.
   The first exploratory probe incorrectly assigned DATA_DIR a str; those
   AttributeErrors in `probe.log` are harness errors, superseded by
   `offline.py`/`offline.log`.

2. **FIXED for the original lost-worker-policy bug (major); new related
   defects below.** `hypertools/plot/plotly_backend.py:3239` passes
   `env=subprocess_env()`. Command:
   `.venv/bin/python /tmp/hypertools-round6/policy.py`.
   Reuses the REAL isolated missing-kaleido interpreter and public export
   driver from `tests/test_animation_export.py`, with PIP_NO_INDEX=1 and
   PIP_CONFIG_FILE=/dev/null. `off` prints `parent False`, then a policy
   failure naming the interactive extra and set_autoinstall(True), without
   entering the install-failure branch. `on` prints `parent True` despite
   env=0, then `installing it automatically failed (CalledProcessError)`.
   No package installed. Log: `policy.log`.
   Additional real child-interpreter probe (`precedence.log`) explicitly
   prints `env 1 Python False parent False child False` and
   `env 0 Python True parent True child True`.
   Parent error TYPE is RuntimeError in both export cases, not the
   documented ImportError (R6-3). Overlapping contexts break policy (R6-1).

3. **FIXED (major/medium original): fitted panel pipelines.**
   `hypertools/plot/plot.py:3371` onward.
   `.venv/bin/python /tmp/hypertools-round6/probe.py` reports a Pipeline
   and held-out transform shape `(4, 3)` for both backends and both fit
   modes. All 33 `tests/test_plot_panels_audit.py` tests passed in the
   combined run below, including held-out replay matching ordinary PCA,
   replay of drawn coordinates, shared object identity, independent fits,
   and the negative control that fresh held-out fitting differs.
   No remaining fix recommended for the original PCA reproduction.

4. **PARTLY FIXED (major/medium original): argument partitioning.**
   `hypertools/plot/plot.py:2890–3009`.
   `.venv/bin/python /tmp/hypertools-round6/probe.py` exercised palette
   names, nested colour lists, forecast_fmt lists, red/blue forecast
   palettes, two-model model-major forecast_hue, and a forecaster fitted
   on two datasets. These original cases succeed on BOTH backends under
   shared AND independent fits. Forecast colours are red then blue,
   format styles are -- then :, and model-major colours stay attached to
   their intended forecasts. The 33 behavioral tests also check fitted
   forecasts against ordinary replay and distinguish them from refitting.
   New repeated-colour counterexample fails (R6-2 below), so CHANGELOG's
   broad claim that every per-forecast form is partitioned is premature.
   Explicit plain legend colours are per FINAL legend entry, not per
   input dataset; the four-entry/three-entry mismatch in exploratory
   `probe.log` is NOT reported as a defect. Plotly deliberately rejects
   plain legend colour lists and supports (label, colour) pairs instead.

5. **FIXED for the original two NameErrors (minor/medium original);
   whole doctest validation is NOT green here.**
   `hypertools/io/load.py:346` now imports hypertools. Public reproduction:
   `import numpy as np, hypertools; arr=np.zeros((10,3));
   print(hypertools.load(arr) is arr);
   print([type(d).__name__ for d in hypertools.load([arr,'spiral'])])`
   prints `True` and `['ndarray', 'list']`.
   `.venv/bin/python -m sphinx -b doctest /tmp/hypertools-round6/source/docs
   /tmp/hypertools-round6/doctest` with
   `HYPERTOOLS_DOCS_PLOT_GALLERY=0` ran **316 tests, 7 failures, zero
   setup/cleanup failures**. No gallery-config type warning: the bool
   override works. Three failures are inaccessible penguins, 538/bechdel,
   and Kaggle data; four in hierarchy are a failed Chrome render and
   three downstream undefined plotly_bundle checks. The original two
   examples passed. Logs: `doctest.log`, `doctest/output.txt`.
   CI now DOES run `sphinx -b doctest -W` in docs-clean
   (`.github/workflows/test.yml:292`); the Claude update's pending-CI item
   and tests/AGENTS.md's claim that no CI job runs doctests are stale.

6. **PARTLY FIXED / release operations still pending (major as a release
   prerequisite, not an instruction to publish during review).**
   `RELEASE_CHECKLIST.md:18` and PR #286 description.
   `git rev-parse 'v1.1.0^{}'` still returns `96ac8b7f...`, excluding
   the PR fixes. GitHub connector GETs confirm PR open at reviewed HEAD,
   issues #284/#285 open, and #284's final CI/tag/draft checkbox unchecked.
   PR body still reports 4885 tests and the earlier review, omitting this
   API and subsequent rounds. HEAD CI run 34187936967 was in progress
   at last inspection (dataset/live-source/wheel succeeded; release-gate
   skipped; matrix/docs pending). URL:
   https://github.com/ContextLab/hypertools/actions/runs/34187936967
   Local optional_dependencies.html and both API pages build successfully.
   Current external RTD/PyPI publication could not be reliably rechecked:
   shell gh/DNS denied; web RTD fetch failed; web PyPI result was stale;
   connector release-by-tag returned 404 (not proof that a private draft
   disappeared). Do not repeat the previous live 404/PyPI observations as
   fresh verified facts. Refresh PR evidence, finish final gates, re-cut
   release artifacts/tag after merge, and verify deployed RTD/PyPI.

### Remaining findings

**R6-1 — Major: overlapping OFF contexts can turn installation ON.**
`hypertools/_shared/lazy_import.py:181,187–189` restores a stale global
snapshot unconditionally. From initial True: thread A enters False;
thread B enters False; A exits; B is still INSIDE its False block but sees
True. After B exits the initial True is incorrectly left False.

This is a process-global setting, not a thread-local one. The docs offer
one-block use and recommend it wherever pip must not run, but do not state
that overlapping thread contexts are unsupported. There is no lock/token
bookkeeping; atomic assignment alone cannot solve the lifetime ordering.

Real public-export reproduction, no mocks:
`MPLBACKEND=Agg MPLCONFIGDIR=/tmp/hypertools-round6/mpl PIP_NO_INDEX=1
PIP_CONFIG_FILE=/dev/null /tmp/hypertools-round6/policyenv/noenv/bin/python
/tmp/hypertools-round6/thread_export.py`.
The interpreter was created from the repo venv by policy.py and genuinely
lacks kaleido. threading.Events impose the ordering above. Observed:
`inside set_autoinstall(False): True`, followed by a worker install attempt
and `installing it automatically failed (CalledProcessError)`; finally
`after both contexts: False`. No package installed. `thread-export.log`.
Suggested fix: define the concurrency contract and maintain scoped
policies with lifetime-aware tokens under synchronization (or explicit
context-local overrides carried into export workers); do not restore an
exited scope over another active scope. Document global/thread semantics
and add this deterministic real concurrent regression. Merely locking the
individual assignments does not fix this ordering.

**R6-2 — Major: panels deduplicate colours instead of forecast labels.**
`hypertools/plot/plot.py:2932–2937`, especially `_colour not in _ordered`.
Different categories may intentionally share one colour. Narrowing drops
that duplicate, yielding too few palette entries for the panel categories.

Command: `.venv/bin/python /tmp/hypertools-round6/duplicate_palette.py`.
The essential public call is:

```python
x = [np.random.default_rng(i).normal(size=(20, 3)).cumsum(0)
     for i in range(2)]
hyp.plot(x, panels=True, panel_fit=fit, predict=['Kalman', 'ARIMA'], t=3,
         forecast_hue=['a', 'a', 'b', 'b'],
         forecast_palette=['red', 'red'], backend=backend, show=False)
```

For BOTH backends and BOTH fit modes, panels=False succeeds and
panels=True raises `ValueError: palette= supplies 1 color(s) but 2 are
required (one per category/component)`. `duplicate-palette.log`.
Suggested fix: track first-seen nonmissing LABELS; append their colours
once per label, retaining repeated colours across distinct labels. Add a
behavioral comparison against ordinary plotting with repeated colours.

**R6-3 — Minor: public animated export contradicts the documented error type.**
`hypertools/plot/plotly_backend.py:3277` converts worker errors into
RuntimeError; `docs/optional_dependencies.rst:94`, `docs/api.rst:168`,
and `hypertools/_shared/lazy_import.py:145` promise ImportError for a
missing extra with installation disabled. `policy.py`/`policy.log` prints
`RAISED RuntimeError` for the off case; ImportError exists only in the
worker traceback string. A caller's `except ImportError` does not catch it.
Suggested fix: pass a structured worker error kind back and raise the
promised public ImportError (and preserve appropriate export error types),
or explicitly document the worker-export exception if intentional.

Tests added in the audit-fix commit are generally meaningful real
behavioral checks, especially held-out replay with a negative control and
missing-dependency subprocess export. Two weaknesses matter here:
`tests/test_animation_export.py:581,603` only require `RAISED` plus message
substrings, so R6-3 passes them; assert the public exception class too.
`tests/test_plot_panels_audit.py:468` checks the parameter roster, not the
correctness of partitioning. It is a useful structural guard but cannot
prove the universal CHANGELOG claim, and current distinct-colour examples
miss R6-2. The new policy tests cover nested sequential contexts, not
concurrent lifetime ordering. Old fake-kaleido/seeded-mkdtemp export tests
pre-exist these audit-fix commits; do not attribute those to this patch.

### Validation completion and limitations

Combined command (all requested suites):

```sh
MPLBACKEND=Agg HYPERTOOLS_AUTO_INSTALL=0 .venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/test_load_offline.py tests/test_lazy_import.py \
  tests/test_animation_export.py tests/test_plot_panels_audit.py \
  tests/test_align_score.py tests/test_plot_review_round3.py \
  tests/test_plot_review_round4.py tests/test_figure_review_gaps.py \
  tests/test_plot_forecast_legend_style.py tests/test_plot_panels_geometry.py
```

**162 passed, 11 failed, 7 setup errors, 19 warnings, 138.51 seconds.**
Seven setup errors are the offline test proxy's forbidden localhost bind;
10 failures are Chrome startup/export. The remaining failure is CAUSED BY
MY HYPERTOOLS_AUTO_INSTALL=0 invocation: the tomli real-install test expects
installation on. Correctly rerun separately with HYPERTOOLS_AUTO_INSTALL=1:
**1 skipped** for inaccessible package network (`install-test.log`). It is
not a library failure. Original offline behavior was independently
verified without the proxy (above). All 33 panel audit tests passed.

The five named round-3/4/figure/forecast/panel-geometry modules contain
73 tests: **70 passed; 3 pixel-parity tests could not render Chrome**.
Thus their non-rendered assertions still hold, but visual Plotly parity
is NOT independently confirmed this round.

Weather command:
`.venv/bin/python scripts/execute_tutorial.py
/tmp/hypertools-round6/source/docs/tutorials/weather_decades.ipynb
--out-dir /tmp/hypertools-round6/notebooks`.
Source is a `git archive HEAD` copy because the notebook also saves
weather_decades.mp4 beside itself even with --out-dir. The helper's
NotebookClient fails before cell execution at socket.bind with
`PermissionError: [Errno 1] Operation not permitted` (`weather.log`).
**Weather remains NOT freshly executed/passed.** No tracked notebook or
video was touched. Do not count the previous pause-induced run either.

Docs command:
`HYPERTOOLS_DOCS_PLOT_GALLERY=0
READTHEDOCS_GIT_IDENTIFIER=fix/1.1-release-review .venv/bin/python -m sphinx
-b html -W --keep-going /tmp/hypertools-round6/source/docs
/tmp/hypertools-round6/html` (plus MPLBACKEND=Agg, writable MPLCONFIGDIR,
autoinstall off). **Build succeeded**, no Sphinx warnings. Source copied
from HEAD; the earlier scratch gallery cache was reused. **Zero gallery
examples freshly executed**: this does NOT finish the earlier full-gallery
execution requirement. `docs/post_build.py` on the TEMPORARY source with
READTHEDOCS_OUTPUT=/tmp/hypertools-round6/html succeeded and processed
51 example pages/thumbnails. `check_docs.py` parses all **171 HTML pages**:
**0 missing internal file/anchor links**. API signatures for
hypertools.set_autoinstall and hypertools.align.score.alignment_score are
present in their generated HTML; optional_dependencies.html exists with
Turning it off. Logs: `html.log`, `post-build.log`, `links.json`.
`scripts/verify_docs_playwright.py` with external output/screenshots paths
fails at its HTTP server's local socket bind (`browser-docs.log`). Browser
rendering of the docs is not verified here.

Visual/public-API probes: `render.py`, `render.log`, `row.py`, `row.log`,
`renders/` (PNG for matplotlib; JSON/HTML for plotly). I personally opened
and inspected SIX fresh matplotlib PNGs: hue-regrouped animated model
collection + forecast_trail + truth; cluster-regrouped equivalent;
column-MultiIndex animation; repeated-full-tuple row-MultiIndex animation;
panels with forecasts/truth/legends/colorbars/two-line titles; and
hyp.subplots with legends and colorbars arriving in separate calls and
three-line titles. No additional layout defect observed in those renders.
Both backends build the corresponding figures/bundles/frames. Every
Plotly `fig.write_image` attempt fails with BrowserFailedError (installed
Chrome closes immediately); **no new Plotly pixels were inspected**.
The tests also cover legend_colors pairs on both backends and ordinary
matplotlib legend-colour lists plus legend_kwargs positions.

Exploratory combinations correctly rejected and NOT findings:
truth= with hierarchical inputs (documented unsupported), unique full
row-MultiIndex tuples with predict= (one-row traces; documented error),
and animated panels (documented static-only ownership). Repeated full
row tuples work; column hierarchy works. The initial render harness used
the wrong bundle key `anim`; fixed to `animation` before rendering, not a
library defect. No claim that every valid combination was exhaustively
examined or every forecast numerically compared on held-out data.

### Next steps / verdict

1. Fix R6-1/R6-2 and settle R6-3's exception contract; add real regressions
   for repeated colours, concurrent policy scopes, and public error types.
2. Re-run the affected suites and the full suite at final HEAD. Finish
   weather, a fresh gallery build, doctests with working network/Chrome,
   and Plotly PNG/browser visual inspection in an environment permitting
   those operations. This audit does not provide a green release gate.
3. Refresh PR #286 evidence and wait for final hosted checks; reconcile
   #284/#285 release bookkeeping. After merge follow RELEASE_CHECKLIST.md
   against the exact final commit: notebook manifest publication,
   wheel/sdist build/install checks, tag/draft/artifacts and tag CI,
   publication and deployed RTD/latest/stable/PyPI verification. No release
   operation was performed or authorized by this review.

VERDICT: FINDINGS

## Updates by the Claude session after Codex round 6 (2026-09-08) — NOT part of the Codex text above

### UPDATE — Round 6 finding 1 (overlapping `set_autoinstall` contexts): FIXED
- `hypertools/_shared/lazy_import.py`: the single saved-and-restored global is
  replaced by a lock-guarded scope stack (`_AUTO_INSTALL_SCOPES`): every
  `set_autoinstall` object is pushed when created; a `with` block removes ITS
  OWN entry on exit wherever it sits; `auto_install_enabled()` returns the
  newest entry still in force (else the environment). Contract documented in
  the class docstring and docs/optional_dependencies.rst: process-global,
  shared by every thread, newest call in force decides, a direct call stays
  until the next call. The reviewer's `/tmp/hypertools-round6/thread_export.py`
  now prints `inside set_autoinstall(False): False`, the export raises
  `ImportError`, and `after both contexts: True`.
- Tests (`tests/test_lazy_import.py`): a deterministic overlapping-context
  test (older block exits first; direct call underneath a block) and the
  two-thread shape with events.

### UPDATE — Round 6 finding 3 (export raises RuntimeError, docs promise ImportError): FIXED
- `hypertools/plot/_kaleido_export_worker.py` writes `{type, message}` to
  `.worker-error.json` in the frames directory before exiting non-zero;
  `plotly_backend._worker_error` re-raises an `ImportError` (a missing extra
  with installation off, or a failed install) or `HypertoolsIOError` (no
  usable Chrome) as that type without retrying; any other worker failure is
  still the `RuntimeError` with the stderr tail. The reviewer's
  `/tmp/hypertools-round6/policy.py` prints `RAISED ImportError` for both
  cases. `tests/test_animation_export.py` asserts `RAISED ImportError`.

### UPDATE — Round 6 finding 2 (panels discard repeated forecast colours): FIXED
- `hypertools/plot/plot.py` (`_panel_slice_forecast_kwargs`): the per-panel
  `forecast_palette` is built with one slot per distinct LABEL (first
  appearance order) instead of per distinct colour, so labels that share a
  colour keep their entries. The reviewer's
  `/tmp/hypertools-round6/duplicate_palette.py`: all 8 combinations OK (were
  4 ValueErrors).
- Tests (`tests/test_plot_panels_audit.py`, real artist/trace colours against
  the single-axes figure): the reviewer's exact case; per-dataset hue with 1
  and 2 models; `forecast_cluster=`; a cycling palette NAME with more labels
  than colours. The roster test the reviewer called weak now has a behavioural
  companion: for each of 12 per-dataset roster entries a two-dataset call with
  distinct values asserts each panel shows only its own value (equal to the
  single-axes dataset's value, different from the other panel); `truth=` list
  and nested hue/labels likewise; a test ties the roster to the case list.
  Animation-only entries (chemtrails/precog/bullettime) and `density` (no list
  form) are the documented exclusions.
- Found while fixing (fixed next, see below): plotly dropped the colour letter
  of a data `fmt=` string on the ordinary path (`'r-'` drew the palette
  colour); plotly `panels=True` on 2-column data without `ndims=` raised
  "Trace type 'scatter' is not compatible with subplot type 'scene'".

### UPDATE — incidental plotly defects found in round 6 follow-up: FIXED
- `hypertools/plot/plot.py`: the plotly palette-injection branch gives each
  dataset whose `fmt` names a colour letter that colour (matplotlib
  precedence: the letter beats `palette=`, loses to explicit `color=`/`hue=`,
  consumes no cycle slot); static, animated, fmt lists, `panels=` and
  `hyp.subplots` cells. Panel cells are lowered to 2-D when every dataset has
  fewer than 3 columns (`_panel_cell_ndims`), on both backends; matplotlib
  drew 2-column panels inside cubes and crashed on 1-column data. Known
  limit: mixed-width independent panels share one cell type (the maximum).
- Tests: `tests/test_plot_review_round6.py` (18); the fmt xfail in
  `tests/test_plot_panels_audit.py` is removed so the assertion is live.

## Codex round 7 (from its run log; the run hit the usage limit before writing its report or this section)

Extracted verbatim by the Claude session from the Codex stdout log (scratchpad codex/run9b.log, lines 7080-7300) on 2026-09-08 08:00. Codex re-verified originals 1-6 and round-6 findings 1-3 (all FIXED, 6 pending release ops) and reported:

**R7-1 — MAJOR: the new panel dimensionality inference runs before feature
expansion, breaking valid Delay -> PCA plots.**
`hypertools/plot/plot.py:3380–3384`, `_panel_cell_ndims` at `3294–3306`, and
matplotlib's `ndims` replacement at `3538–3543`.

Two raw columns do not imply two columns after the analysis pipeline. The
new helper lowers the requested cell type using RAW data; its early return
for requested <= 2 means the later shared-fit check cannot restore 3-D.
Independent matplotlib panels also change the requested PCA fit to 2-D.

Public reproduction (both backends, both panel_fit modes):

```python
x = [np.random.default_rng(i).normal(size=(20, 2)) for i in range(2)]
hyp.plot(x, panels=True, panel_fit=fit,
         manip={'model': 'Delay', 'kwargs': {'dims': 3}},
         reduce='PCA', ndims=3, backend=backend,
         return_model=True, show=False)
```

Command: `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/hypertools-round7/mpl
HYPERTOOLS_AUTO_INSTALL=0 .venv/bin/python /tmp/hypertools-round7/edges.py`
(`edges.log`; focused form is `delay_minimal.py`). Ordinary plotting returns
3-D traces and two `(18, 3)` arrays on both backends. With panels:

- matplotlib/shared: `ValueError: the data to plot has 3 dimensions, but
  static plots support at most 2; reduce=None disables dimensionality reduction`.
- matplotlib/independent: **silently returns `(18, 2)`** per panel on
  rectilinear axes despite `ndims=3`.
- plotly/shared and plotly/independent: `ValueError: Trace type 'scatter3d'
  is not compatible with subplot type 'xy' at grid position (1, 1)`.

Regression attribution: from `/tmp`, run the repo `.venv/bin/python` on
`delay_minimal.py` with `PYTHONPATH=/tmp/hypertools-round6/source` (the saved
3f4b087d source). `delay-before.log` prints OK and `(18, 3)` for ALL FOUR
combinations. Thus this is introduced by c700c85f, not a preexisting limitation.
I opened the new ordinary 3-D and erroneous 2-D matplotlib PNGs; the latter
really renders two flat panels. Plotly trace construction fails before export.
Suggested fix: determine each cell's projection from its actual analyzed data,
retaining the originally requested analysis dimensionality and fitted pipeline;
do not infer final width from raw data when manip/pipeline can change it.
Cover feature-expanding manip/pipeline input, shared/independent fits and
reducer comparisons without adding a second fit.

**R7-2 — MINOR: a fmt-pinned first call consumes a palette slot on composed
Plotly figures, contrary to matplotlib and the new fmt contract.**
`hypertools/plot/plot.py:10577` records `offset + len(xform)` even though the
new branch at `10487–10495` excludes lettered fmt entries from consuming the
palette. The matplotlib caller-axes counter at `10726` has the same issue
for an initially empty `hyp.subplots` cell.

Command: same environment, `.venv/bin/python
/tmp/hypertools-round7/fmt_minimal.py` (`fmt-minimal.log`). In each backend,
plot a `(20,3)` array with `fmt='r-'`, `palette=['navy','gold','green']`, then
plot a second array into that figure's axes/figure with the same palette and
no fmt colour. Observed:

```
matplotlib ['r', '(0.0, 0.0, 0.5019607843137255)']
plotly ['rgba(255,0,0,1.0)', 'rgba(255,215,0,1.0)']
```

The second line is navy on matplotlib, gold on Plotly. `edges.py` also proves
both backends' empty subplots-cell path uses gold, whereas a single call with
`fmt=['r-', '-']` uses navy. Matplotlib comparison PNGs were opened and
inspected; Plotly JSON records the actual trace colors (Chrome cannot render
here). Suggested fix: track consumed cycle slots, consistently for a normal
figure and a cell, rather than total drawn datasets; include explicit colours
and hue in the accounting rules. Test a pinned-colour call followed by an
uncoloured call, alongside the equivalent single call, on both backends.

**R7-3 — MINOR: direct set_autoinstall calls leak every superseded handle.**
`hypertools/_shared/lazy_import.py:194–195` appends a strong reference for
EVERY call; only `__exit__` removes one. Direct calls never exit, so an older
direct setting, documented as superseded, stays alive for the interpreter's
lifetime. This is a retention bug, not a recurrence of the overlapping-OFF
policy failure.

Command: `.venv/bin/python /tmp/hypertools-round7/policy_extra.py`
(`policy-extra.log`; MPLBACKEND=Agg). 100,000 public direct calls alternating
False/True, deletion of local handles, and `gc.collect()` leave the first
handle alive through a weakref and retain **8,004,888 bytes** by tracemalloc.
No private policy state is mutated by the probe. Suggested fix: bound storage
to active scopes and the necessary current direct baseline, dropping
superseded direct records; preserve the verified out-of-order scope semantics.
Add a real weakref/collection regression.

**Nits / test and documentation drift.**

- `tests/test_animation_export.py:603` still asserts only `'RAISED'` in the
  Python-ON/env-OFF branch, although the OFF branch at 581 now asserts
  `RAISED ImportError`. `nl -ba` inspection confirms this; the public driver
  DOES return ImportError today. Tighten the ON branch too; the update's
  broad statement that the export tests assert the type overstates coverage.
- `tests/test_lazy_import.py:349,353,356,362–363` ignores Event.wait timeout
  results and does not assert that joined threads terminated. Source
  inspection shows a timed-out wait can let the test run without its intended
  overlap. Assert successful waits and completed threads. This does not undo
  my independent event-ordered public export reproduction, which passed.
- `tests/AGENTS.md:23` still says no CI job runs doctests. The actual
  `.github/workflows/test.yml` docs-clean step and fetched hosted logs show
  that it does. Update the repository guidance; do not remove the CI check.
- The generated viewcode `[docs]` backlinks for `synthetic_outlet` and
  `HypertoolsTrustError` use alias anchors absent from their destination
  pages. Relevant canonical declarations: `docs/api.rst:246,268` and
  `docs/hypertools.io.lsl.synthetic_outlet.rst:6`,
  `docs/hypertools.io.sources.HypertoolsTrustError.rst:6`.
  The actual missing targets are `#hypertools.io.synthetic_outlet` and
  `#hypertools.HypertoolsTrustError`. The pages exist; their canonical API
  anchors use the defining-module names. Suggested fix: make viewcode links
  use those canonical anchors or add the alias targets. See `links.json`.
  My link parser resolves `/tmp` before indexing pages so symlink differences
  cannot silently skip anchor checks; the previous round's parser did not.

## Updates by the Claude session after Codex round 7 (2026-09-08) — NOT part of the Codex text above

### UPDATE — R7-3 (superseded direct `set_autoinstall` handles retained): FIXED
- `hypertools/_shared/lazy_import.py`: a new setting replaces a superseded one
  that no block holds open (a direct call's, or the `_Baseline` an exited block
  left), keeping only its value; a block that replaced one puts that value
  back on exit. Semantics verified unchanged (older-block-exits-first, direct
  call under a block). The reviewer's `/tmp/hypertools-round7/policy_extra.py`:
  first handle alive False, retained 1208 bytes after 100k direct calls (was
  8,004,888). Test: `test_superseded_direct_calls_are_not_retained` (weakref +
  bounded stack + block restore).

### UPDATE — nits: FIXED
- `tests/test_animation_export.py` ON-branch asserts `RAISED ImportError` too.
- `tests/test_lazy_import.py` two-thread test asserts every Event hand-off
  succeeded and both threads finished.
- `tests/AGENTS.md` (gitignored local guidance) says the docs-clean CI job runs
  the doctest builder.
- viewcode backlinks: sphinx's viewcode records ONE "referenced-as" module per
  source file (`refname`), so objects first documented via `hypertools.io` /
  `hypertools` got alias backlinks for every other object in that file.
  `io.synthetic_outlet` and `HypertoolsTrustError` are now documented under
  those public names (stubs renamed; `HypertoolsTrustError` is re-exported
  from `hypertools` beside `HypertoolsOfflineError`, in `__all__`), so both
  backlinks resolve.

### UPDATE — R7-1 (panel cell dimensionality inferred from raw width): FIXED
- `hypertools/plot/plot.py`: every `panels=` mode fits through one
  `_panel_probe()` (a `plot(..., return_model=True)` call whose figure is
  discarded) and draws each panel from the ANALYZED rows via `transform=`;
  `_panel_cell_ndims` applies to the probe output only, so Delay-expanded
  2-column data gets 3-D cells on both backends in shared / independent /
  reducer-list modes, with the requested `ndims` and each panel's fitted
  pipeline kept. A `pipeline=` whose reduce keeps more than 3 columns draws
  through the single call's projection; a 2-wide panel in a 3-D grid is
  zero-padded for plotly's scene cell; probe warnings are re-emitted once. A
  counting PCA confirms 1 fit (shared) / 1 per panel (independent). The
  reviewer's `/tmp/hypertools-round7/delay_minimal.py` and `edges.py` cases
  pass on both backends and both modes.

### UPDATE — R7-2 (fmt-pinned datasets consume a palette slot): FIXED
- `_palette_slots_consumed()` counts only cycle-coloured datasets (explicit
  `color=`, `hue=` and fmt colour letters consume none), computed before the
  plotly branch injects palette colours; `datasets_drawn` (plotly) and
  `ax._hyp_palette_offset` (matplotlib, now recorded on hypertools' own axes
  too) use it, so a plain figure, a cell and plotly agree with the single call.
- Tests: `tests/test_plot_review_round7.py` (32).

## Codex round 8

Red-team review in progress. Results are appended incrementally.

### Verified fixes and initial checks

- Reviewed HEAD `1cad1e63` (runtime fix `14e0965e`). Read the audit and UPDATE claims, release-review CHANGELOG and `git log --oneline master..HEAD`; inspected `git diff 3f4b087d..HEAD --stat`. Pre-existing staged changes: `notes/session_2026-09-05_release-1.1-review.md`, `tests/_netskip.py`, `tests/test_load_sources.py`; left untouched. Scratch/evidence: `/tmp/hypertools-round8/`. No output-file path was supplied with this request, so this authorized notes append is the persistent report.
- **R7-1 fixed for the reported Delay reproduction:** `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/hypertools-round8/mpl HYPERTOOLS_AUTO_INSTALL=0 .venv/bin/python /tmp/hypertools-round7/delay_minimal.py` prints OK and two `(18, 3)` arrays for both backends and both fit modes (`delay.log`). The requested round-7 and round-6 test modules pass, including actual 3-D artists/traces, independent numerical equivalence, reducer comparisons, 1-/2-column defaults, pipeline expansion and fit counters.
- **R7-2 fixed for the reported fmt reproduction:** same environment, `... /tmp/hypertools-round7/fmt_minimal.py` prints red then navy on both backends (`fmt.log`). Round-7 tests also pass for empty subplot cells, mixed lettered/unlettered calls, explicit colour/hue and ordinary cycle advancement.
- Requested combined pytest command, with `PIP_NO_INDEX=1 PIP_CONFIG_FILE=/dev/null`, completed: **65 passed, 2 failed, 15 warnings in 27.61 s** (`pytest.log`). Failures are the intentional real pip-install test blocked by that environment and the installed-Chrome render test; details/limitations will be recorded after inspecting tracebacks. This is not a green whole-suite claim.
- Source check confirms the export ON branch now asserts `RAISED ImportError` (`tests/test_animation_export.py:603`) and the overlapping-thread test asserts all Event hand-offs and completed joins (`tests/test_lazy_import.py:364–369`); the latter passed in the combined run. Local `tests/AGENTS.md` now correctly names the CI doctest builder.

- **R7-3 fixed for superseded direct handles:** reused `/tmp/hypertools-round7/policy_extra.py` under `.venv/bin/python` (`policy-extra.log`): 100,000 calls, first handle alive **False**, last alive True, retained **1208 bytes**. Python-over-environment inheritance prints the correct value in both directions. The existing overlapping-context regression and weakref test pass; a NEW entry-boundary race is recorded below.
- **Animated Plotly export on/off verified:** `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/hypertools-round8/mpl PIP_NO_INDEX=1 PIP_CONFIG_FILE=/dev/null .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_animation_export.py -k honours_set_autoinstall`: **2 passed, 31 deselected, 16.24 s** (`export.log`). Real missing-kaleido interpreters; OFF refuses without pip; ON reaches blocked pip; both return public ImportError and leave kaleido absent.

### R8-1 — MAJOR: policy compaction races with context entry and restores installation ON over a direct OFF baseline

- Location: `hypertools/_shared/lazy_import.py:214–223`. A context object's constructor registers it before `__enter__` marks it active. Another thread constructing a context in that interval treats the first object as a superseded direct call, drops its identity and keeps only its enabled value. The first block's exit then cannot remove its setting or restore its prior baseline. The second block restores that expired setting instead.
- Verified: `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/hypertools-round8/mpl .venv/bin/python /tmp/hypertools-round8/policy_threads.py` (`policy-threads.log`). Start with public `hyp.set_autoinstall(False)`; A constructs a True context, B enters a False context before A enters, A enters/exits, then B exits. Thread Events assert every hand-off and joins assert termination. Output: `INITIAL False {'a_inside_after_b_enter': False, 'b_inside_after_a_exit': False} FINAL True EXPECTED FINAL False`. No private state is changed, no mocking, no pip. A small context factory delays returning the actual public handle to expose the valid scheduling boundary between construction and entry. Fresh nested objects in one thread restore False normally.
- Suggested fix: synchronize registration/activation and retain enough lifetime information for a constructed handle subsequently entered as a block, without strongly retaining all superseded direct handles. Merely locking `_entered = True` does not recover an already-discarded record. Add a deterministic construction/entry interleaving test starting from a direct OFF baseline.

### R8-2 — MINOR: mixed one-column/three-column independent panels still crash

- Location: `hypertools/plot/plot.py:3579–3593`. One global cell projection is selected and only TWO-column arrays are padded for a 3-D cell; a one-column series is passed through unchanged.
- Verified: `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/hypertools-round8/mpl HYPERTOOLS_AUTO_INSTALL=0 .venv/bin/python /tmp/hypertools-round8/edges.py` (`edges.log`), case `mixed13`: two `(24, 1)` / `(24, 3)` arrays, `panels=2, panel_fit='independent', reduce=None, show=False`. Matplotlib raises `TypeError: Axes3D.plot() missing 1 required positional argument: 'ys'`; Plotly raises `ValueError: Trace type 'scatter' is not compatible with subplot type 'scene'`. The one-/two-column independent grid succeeds on both. Shared fitting correctly rejects unequal input widths and is not the finding.
- Suggested fix: use each panel's analyzed dimensionality for its cell, or deliberately convert a one-column series to index/value/zero coordinates in a shared 3-D grid. Preserve the series' index semantics. Extend the mixed-width regression beyond `[2, 3]` to `[1, 3]`; document a uniform-projection limitation if retained. Attribution against the pre-fix source is pending; this is a remaining edge-case gap, not yet claimed newly introduced.

### R8-3 — MINOR: an intervening explicit-colour/categorical-hue Plotly call resets previously consumed palette slots

- Location: `hypertools/plot/plot.py:10572–10574,10701`. `_plotly_palette_offset` starts at zero and reads the existing figure/cell count only inside `if "color" not in mpl_kwargs`. Explicit `color=` and categorical `hue=` skip that read, then write `0 + 0` as the total consumed count.
- Verified: `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/hypertools-round8/mpl HYPERTOOLS_AUTO_INSTALL=0 .venv/bin/python /tmp/hypertools-round8/composition.py` (`composition.log`). With palette `['navy','gold','green']`, draw an ordinary array, append another with `color='black'` (or categorical hue with two runs), then append an ordinary array. The last colour is **navy `(0,0,128)` on Plotly**, **gold `(255,215,0)` on matplotlib**, for both a plain figure and `hyp.subplots(1,1)` cell. An intervening `fmt='r-'` or continuous hue correctly keeps gold. The first ordinary call consumed one slot; the pinned call should preserve it.
- Suggested fix: read the existing consumed count independently of colour injection, then add only the new slots taken. Extend `tests/test_plot_review_round7.py:342` beyond a pinned FIRST call (offset zero) to ordinary → pinned → ordinary, for explicit colour and categorical hue, both figure and cell. The current test cannot detect resetting a nonzero offset.

### Attribution and documentation checks

- Archived ONLY `hypertools/` from `c700c85f` under `/tmp/hypertools-round8/before`, then ran the same probes from `/tmp` with `PYTHONPATH=/tmp/hypertools-round8/before` and the repo's `.venv/bin/python`. `policy-threads-before.log` ends `FINAL False EXPECTED FINAL False`: **R8-1 is introduced by 14e0965e**. `edges-before.log` already has both mixed `[1,3]` errors: **R8-2 pre-exists the last fix**, an uncovered remaining gap (do not attribute it to 14e0965e).
- Existing `docs/_build/html` was checked, not rebuilt. `.venv/bin/python /tmp/hypertools-round8/docs_links.py` (`docs-links.log`) finds BOTH old source-view backlinks still broken, but both NEW public-name stub anchors present. `stat` dates `_modules/hypertools/io/lsl.html` September 6 versus the new stub September 8: this is a mixed/stale build, not proof the committed stub rename fails in a clean build. Source uses the correct public module and directive; `HypertoolsTrustError` is re-exported. Require clean-output backlink validation; do not claim existing backlinks resolve. No full gallery or notebook was rerun.
- **NIT — CHANGELOG.md:828** says in bold that `set_autoinstall` “keeps one record per superseded direct call”, the behavior the fix removes; following prose correctly says records are replaced. Verified by reading the release-review section and the real 100k-call weakref probe above. Suggested wording: “does not retain superseded direct calls.”
- **NIT — hypertools/plot/plot.py:5949–5953** still describes each cell as the single-axes projection and says 1-/2-column data always draws 2-D regardless of ndims. It should explicitly refer to ANALYZED width and document the present uniform-grid behavior for mixed widths; the verified Delay expansion and mixed `[2,3]` regression test contradict a raw-width reading. `panel_fit` prose at 6035 promises equivalence to an individual call, which the `[1,3]` crash violates.
- Export documentation in `docs/optional_dependencies.rst` accurately describes tested ON/OFF worker inheritance and ImportError. Its context lifetime promise is violated by R8-1, rather than being an undocumented unsupported-thread scenario.

### Additional successful probes / limits

- `valid_forecasts.py` (same Agg/autoinstall-off environment) exercises 18 combinations: both backends × shared/independent/reducer grids (`panels=2`) × no grouping/categorical hue/KMeans regrouping, with `predict='Kalman', t=3, truth=` in the documented plotted space. All construct successfully and real role-tagged artists/traces include every expected forecast and truth overlay (`valid-forecasts.log`). Matplotlib splits each truth into a line and markers, hence twice the Plotly truth-trace count. This checks ownership counts/construction, not every forecast number or pixel.
- `edges.py` also confirms plain reducer-list grids with integer `panels=2`, wide PCA plus hue/cluster, and mixed `[1,2]` independent panels construct on both backends. The initial `truth5` and later four-column-truth exploratory errors are NOT findings: ordinary calls reject them too; `truth` must be in the three-dimensional PLOTTED space here, as the error says. The corrected valid probe above is authoritative.
- Existing Chrome render fails with `HypertoolsIOError` reporting that installed Chrome closes immediately. The real-install test fails because this audit deliberately sets `PIP_NO_INDEX=1`; neither proves a library regression. No successful new Plotly image export or browser pixel inspection is claimed. Successful export-policy tests exercise missing dependencies, not successful Chrome rendering.
- Corrections to report line references: the reversed CHANGELOG bold sentence is at **CHANGELOG.md:823**, not 828; the pinned-first-call test relevant to R8-3 is **tests/test_plot_review_round7.py:351–360**, not 342.

### R8-4 — MAJOR: independent/reducer panels recluster the probe data after dropping the caller's random_state

- Location: `hypertools/plot/plot.py:3554–3557,3595–3597` (also the reducer-probe path at 3536). The probe runs `cluster=`, but each draw retains `cluster=` and replaces `random_state` with None. Its plotted clustering is a new unseeded fit instead of the clustering specified by the caller and fitted in the probe/bundled pipeline. Previously an independent/reducer panel used the original seeded call directly.
- Verified: `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/hypertools-round8/mpl HYPERTOOLS_AUTO_INSTALL=0 LOKY_MAX_CPU_COUNT=2 .venv/bin/python /tmp/hypertools-round8/cluster_artists.py` (`cluster-artists.log`). Two `(24,5)` arrays from RNG seeds 1 and 2; `reduce='PCA', ndims=3, cluster='KMeans', n_clusters=3, random_state=88, antialias=False, fmt='o'`. An individual call draws clusters of **[4,9,11] points**. The first `panels=2, panel_fit='independent'` cell draws **[8,8,8] on matplotlib**, **[5,6,13] on Plotly** in the recorded run. These are actual artist/trace point counts, not just relabelled clusters; unseeded counts may vary per run. The analyzed PCA coordinates are identical (`cluster-groups.log`), so the extra clustering changes membership.
- Attribution: same script from `/tmp` with `PYTHONPATH=/tmp/hypertools-round8/before` (archived `c700c85f`) prints **[4,9,11] for both the individual call and panel** on BOTH backends (`cluster-artists-before.log`). Introduced by `14e0965e`.
- Independent real `CountKMeans(KMeans)` probe (`clusters.py`, overrides fit only to increment a counter and then calls sklearn's real fit) reports **6 fits** for two independent panels or two reducer panels, versus **4 before**; ordinary calls remain 2 and shared panels remain 4 (`clusters.log`, `clusters-before.log`). The earlier repeated cluster fits already existed; the additional fits in independent/reducer grids are new. The new `test_no_panel_is_fitted_twice` counts only PCA and cannot catch this.
- Suggested fix: preserve the fitted clustering and row-to-category ownership from each probe for drawing, or omit clustering from the probe and execute it exactly once during the final draw with the caller's resolved seed/spec, placing THAT fitted state in the bundle. Add real per-panel cluster membership/seed comparisons to individual calls, both backends and reducer grids, plus cluster fit-count coverage. Avoid merely relaxing the seed contract.

### Test review and completion

- `tests/test_plot_review_round7.py` uses real figures, real artist/trace colours and real PCA; its assertions are generally behavioral, not tautological. The important gaps are concrete: `[2,3]` is the only mixed-width test (misses R8-2), the pinned-colour/hue test starts at offset zero and hue is continuous only (misses R8-3), and the no-second-fit test at 241–250 covers only PCA and only matplotlib (misses R8-4). The wide-pipeline test checks numerical equality only for shared mode; broaden independent numerical coverage when modifying this area.
- Latest `tests/test_lazy_import.py` assertions correctly verify weakref collection and successful event hand-offs. The stack-length assertion in the retention test is tied to implementation, but the weakref assertion independently tests the user-visible lifetime defect; it is not a tautological test. The concurrency test signals only AFTER entry, excluding the construction/entry boundary in R8-1. Add that boundary case rather than discarding the existing overlap test.
- A third independent concurrency probe, `.venv/bin/python /tmp/hypertools-round8/three_threads.py` with Agg and writable MPLCONFIGDIR, passes: three fresh contexts enter sequentially, then exit B/A/C, with all Event waits and joins asserted; each post-exit check remains False and restores the direct False baseline (`three-threads.log`). Ordinary nested fresh contexts also pass (`policy-threads.log`). The remaining failure is specifically the construction/entry race.
- Final HEAD remains `1cad1e63caeaf408e12590e41ab108b9e801beec`. `git status --short` shows only this notes append plus the three staged files present at review start. No tracked source/test/docs file edited, staged, committed or deleted; no new repository file left by this review; no other process or outlet stopped. All new scripts/logs are under `/tmp/hypertools-round8/`. Original findings 1–6 and round-6 findings 1–3 were not rerun as standalone audits, per this round's instructions.

### Next steps / verdict

1. Fix R8-1 and R8-4 before merge/release; cover the exact entry interleaving and seeded per-panel cluster membership with real regressions. Fix R8-3's retained offset and R8-2's one-column mixed-grid handling on both backends.
2. Correct the two documentation nits; extend the identified test gaps. Re-run the affected suites and final hosted checks after those changes. Treat the local blocked pip/Chrome checks as validation limits, not passing gates.
3. Check BOTH public-name viewcode backlinks in a clean docs output. Reuse the existing full docs/doctest/notebook pipeline evidence as directed; this review did not rebuild the gallery or rerun notebooks. Follow the existing release checklist after fixes and final approvals; no merge/publication performed.

**Round 8 review complete: two major findings, two minor findings, and documentation/test-coverage nits.**

VERDICT: FINDINGS

## Updates by the Claude session after Codex round 8 (2026-09-08) — NOT part of the Codex text above

### UPDATE — R8-1 (construct-then-enter race dropped a live context): FIXED
- `hypertools/_shared/lazy_import.py`: the scope state is now the LIVE handles
  (weak references, construction order) plus a BASELINE (the newest direct
  call's value, with its sequence number). A handle that dies without ever
  entering a block was a direct call: its weakref callback folds its value
  into the baseline and removes its record at once; a handle that is alive
  but not yet entered is never touched; a block removes only its own record
  on exit and its record is marked finished so its later death is ignored.
  `auto_install_enabled()` takes whichever is newer by call order: the top
  live record or the baseline. Verified with the reviewer's
  `/tmp/hypertools-round8/policy_threads.py` (FINAL False as expected; inside
  A's later-entered block the value is B's False because the newest CALL
  decides, per the documented contract), round 7's `policy_extra.py`
  (retained 1080 bytes after 100k direct calls) and round 6's
  `thread_export.py` (after both contexts: True).
- Tests (`tests/test_lazy_import.py`, 18): the exact construct/enter
  interleaving across threads with every hand-off asserted, plus the earlier
  overlapping-block, direct-under-block and retention tests (stack empty
  after discarded direct calls).
- CHANGELOG nit: the reversed bold sentence now reads "no longer retains
  superseded direct calls".

### UPDATE — R8-4 (panels recluster without the seed): FIXED
- `hypertools/plot/plot.py`: the panel probe returns its fitted cluster labels
  (`_PanelClusterLabels`); every panel replays them (whole for independent /
  reducer grids, sliced per dataset for shared) with NO clusterer fit in the
  draw; `return_model=True` bundles carry `models['cluster_labels']`. Fit
  counts: ordinary 2, shared 2, independent 4, reducers 4 (was 6/6). Seeded
  memberships equal the individual call's on both backends in every mode.

### UPDATE — R8-3 (pinned call resets the plotly palette offset): FIXED
- The offset read is hoisted out of the colour-injection branch, so a
  `color=` / categorical-hue call keeps the prior count (figure and cell).

### UPDATE — R8-2 (mixed one-/three-column independent panels crash): FIXED
- `_panel_rows_in_3d` maps a 1-column series to (row index, value, 0) in a
  3-D grid (date index by position), both backends and fit modes.

### UPDATE — docstring nit: FIXED
- `panels=` prose describes ANALYZED width and the unequal-width rule;
  `panel_fit='independent'` prose covers seeded clustering.
- Tests: `tests/test_plot_review_round8.py` (52; 34 fail against the
  pre-fix plot.py).

## Codex round 9

Red-team review of HEAD `650808f0` (branch `fix/1.1-release-review`, targeting master). Read the original audit and all UPDATE sections, rounds 6–8, the complete release-review CHANGELOG section, `git log --oneline master..HEAD`, and `git diff c700c85f..HEAD --stat`. UPDATE entries are claims, not independent evidence. Initial status contained only this round's notes heading. No separate `-o` path was supplied; this authorized notes append is the persistent output. Scratch scripts/logs: `/tmp/hypertools-round9/`. No full gallery build or notebook execution; original findings 1–6 and round-6 findings 1–3 are not repeated as standalone audits.

### Incremental verification

- **R8-1 construction/entry reproduction fixed:** `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/hypertools-round9/mpl HYPERTOOLS_AUTO_INSTALL=0 .venv/bin/python /tmp/hypertools-round8/policy_threads.py` prints both in-block observations False, `FINAL False EXPECTED FINAL False`, and fresh nested contexts restore False. Evidence: `/tmp/hypertools-round9/policy_threads.log`. This reruns the actual public thread interleaving, not a source-only inference.
- **R8-1 retention and existing overlap checks pass:** reused `/tmp/hypertools-round7/policy_extra.py`: 100,000 direct calls leave BOTH weakrefs dead and retain 1080 bytes (bookkeeping overhead, no retained handles). Python/environment child inheritance is correct in both directions. `/tmp/hypertools-round8/three_threads.py` keeps False after each B/A/C exit and restores False; all Event waits/joins asserted. Logs: `policy_extra.log`, `three_threads.log` under round9.
- **R8-4 seeded drawing and fit counts fixed for the reported cases:** reused `cluster_artists.py`: individual and independent-panel cluster sizes both `[4, 9, 11]` on BOTH backends. `clusters.py`: ordinary/shared/independent/reducer grids use 2/2/4/4 real KMeans fits on each backend, so drawing adds no fits beyond the probes. The ordinary call still fits twice (pre-existing, not newly introduced). Round-8 tests pass for actual point-set equality, bundle `models['cluster_labels']`, shared label slices, reducer grids, dict/n_clusters-only/mixture spellings.
- **R8-3 offset fix verified:** reused `composition.py`; all 16 combinations (both backends, figure/cell, fmt/explicit color/categorical or continuous hue inserted after an ordinary call) finish GOLD `(255,215,0)`, preserving the nonzero offset. Evidence: `composition.log`.
- **R8-2 reported mixed-width crash fixed:** reused `edges.py`; `[1,3]` and `[1,2]` independent panels construct on BOTH backends. Round-8 tests inspect coordinates/projection, including dated one-column data. Shared unequal raw widths still give the documented error. `edges.py`'s invalid five-column truth calls also fail on the ordinary path and are NOT findings.
- Focused combined pytest: `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/hypertools-round9/mpl PIP_NO_INDEX=1 PIP_CONFIG_FILE=/dev/null PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_plot_review_round8.py tests/test_plot_review_round7.py tests/test_plot_review_round6.py tests/test_lazy_import.py`: **118 passed, 2 failed, 16 warnings in 29.76 s** (`pytest.log`). Both failures are environment limits: real tomli installation deliberately blocked by PIP_NO_INDEX, and installed Chrome closes immediately during rendering. All 52 round-8 plot tests, all 50 round-6/7 plot tests and the other 16 lazy-import tests pass. No successful Chrome rendering is claimed.
- **Documentation nits corrected in source:** CHANGELOG.md:823 now says “no longer retains superseded direct calls”; plot.py:6069–6081 describes ANALYZED width and uniform mixed-width projection, including index/value/floor handling; panel_fit prose includes seeded clustering. These match the verified basic cases above. No docs build rerun.
- **Animated Plotly export ON/OFF still passes:** `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/hypertools-round9/mpl PIP_NO_INDEX=1 PIP_CONFIG_FILE=/dev/null PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_animation_export.py -k honours_set_autoinstall`: **2 passed, 31 deselected in 15.95 s** (`export.log`). Real missing-kaleido children; OFF refuses without pip, ON reaches blocked pip, both expose ImportError and leave kaleido absent.

### R9-1 — MINOR: a live Requests certificate error still skips as transient

- Location: `tests/_netskip.py:186–189` (`_exception_is_transient`, before the new phrase classifier at 224–228); coverage gap at `tests/test_load_sources.py:610–634`.
- What is wrong: the new dropped-TLS versus certificate distinction works for the wrapped aggregate STRING used by its regression test, but not for the preferred live-exception input. `requests.exceptions.SSLError` inherits `ConnectionError`, so the structural MRO test returns True before checking the certificate. Ordinary CI can skip an actual certificate failure, contrary to CHANGELOG.md:842–845's unqualified claim. Strict live-source mode still prevents skipping. This structural behavior pre-exists the latest patch; it is a remaining gap in the newly claimed distinction, not a new regression.
- Verified: `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/hypertools-round9/mpl HYPERTOOLS_AUTO_INSTALL=0 PIP_NO_INDEX=1 PIP_CONFIG_FILE=/dev/null PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/jmanning/hypertools .venv/bin/python /tmp/hypertools-round9/extra.py` (`extra.log`). Real Requests/stdlib SSL exception classes, no monkeypatching: drop prints `live True text True`, `SKIPPED`; certificate prints `live True text False`, **`SKIPPED`** through the actual `skip_on_transient_network` context manager. These are real exception instances, not a live HTTPS fetch. The first invocation lacked repository PYTHONPATH for importing tests and was a harness error; the corrected run is the evidence.
- Suggested fix: treat SSL/certificate types before generic ConnectionError/URLError/MaxRetryError ancestry, inspecting the causal chain; add behavioral skip/re-raise tests for live wrapped EOF and certificate errors, alongside the aggregate-string test. Do not weaken the strict gate.

### R9-2 — MAJOR: shared clustered panels lose the global cluster-to-colour mapping

- Location: `hypertools/plot/plot.py:3675–3691,3717–3719` passes each shared label slice for replay but no global category/colour mapping; the drawing call resolves colours again from the categories present in that slice.
- What is wrong: shared fitting now correctly preserves membership IDs, but the same colour can represent DIFFERENT globally fitted clusters in adjacent panels. With two well-separated datasets, the joint call uses red and blue for clusters 0 and 1; shared panels report labels `[1]` and `[0]` respectively, yet BOTH draw red. This silently undermines comparison of shared clusters. Current shared regression tests compare bundle labels and group sizes, not colour-to-label identity when a panel lacks a category.
- Verified: `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/hypertools-round9/mpl HYPERTOOLS_AUTO_INSTALL=0 PIP_NO_INDEX=1 PIP_CONFIG_FILE=/dev/null PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /tmp/hypertools-round9/membership_colors.py` (`membership_colors.log`). Inputs: 24x3 Gaussian clouds (seeds 1/2, scale .01), centered at -10/+10; `reduce=None, cluster='KMeans', n_clusters=2, random_state=88, palette=['red','blue'], fmt='o', antialias=False, return_model=True, show=False`; compare ordinary call to `panels=2`. Actual matplotlib line colours are `[[red],[red]]`, actual Plotly scene/scene2 marker colours both `rgba(255,0,0,1.0)`, versus joint red/blue. Bundle panel IDs are `[[1],[0]]` on both. No mocks or pixel inference.
- Suggested fix: retain the shared probe's complete cluster category order and label-to-colour/legend mapping, then slice observations without compacting that mapping per cell. Cover absent categories and verify each label's actual artist/trace colour against the joint call on both backends, including forecasts inheriting that colour. Attribution against c700c85f is checked separately below; do not infer previous shared clustering was correct.

### R9-3 — MAJOR: padding a mixed-width panel changes its forecast model and values

- Location: `hypertools/plot/plot.py:3711–3724` pads analyzed one-column rows into `(index, value, 0)` and passes those artificial THREE features through `transform=` while leaving `predict=`/`truth=` for the subsequent plot call. Padding intended for display therefore changes forecasting input.
- Verified: `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/hypertools-round9/mpl HYPERTOOLS_AUTO_INSTALL=0 PIP_NO_INDEX=1 PIP_CONFIG_FILE=/dev/null PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /tmp/hypertools-round9/mixed_forecast.py` (`mixed_forecast.log`). Two cumulative Gaussian datasets, seeds 1/2 and widths 1/3, 24 rows; compare the first dataset alone with `panels=2, panel_fit='independent'`; both use `reduce=None, predict='Kalman', t=3, show=False, return_model=True`. BOTH backends return ordinary forecast values `[2.4040928045, 2.2453572543, 1.9666166291]` but panel value-column forecasts `[3.2026539966, 3.5246851840, 3.4823615582]` (maximum difference **1.5157449291**). Ordinary bundle forecasts have shape `(3,1)`, panel forecasts `(3,3)`; this is numerical model output, not a pixel/rescaling difference.
- Related observable: a real Kalman fitted on the first one-column dataset works in the ordinary call but its first panel raises `ValueError: the fitted forecaster expects 1 feature(s) ... new dataset has 3`. `extra.py` also shows one-column `truth=` accepted by the ordinary call but rejected in the mixed grid as “1 column(s) ... trace ... 3” despite `reduce=None`. The mixed `[1,3]` grid previously crashed outright (R8-2); this is a newly reachable semantic defect in that fix, not a claim that pre-fix forecasting worked.
- Suggested fix: keep forecasting and returned analyzed/model data in each panel's original analyzed feature space; convert observation, forecast and truth coordinates into the common display projection only when rendering. Preserve numeric/date future-index semantics for the synthetic display x coordinate. Add a real individual-versus-panel forecast numeric comparison and a fitted-model/truth case on BOTH backends. Extend tests beyond observed-row correlations and construction.

### R9-4 — MINOR: marker-only categorical hue consumes palette slots in composed figures

- Location: `hypertools/plot/plot.py:257–280` (`_palette_slots_consumed`) and its call at 10698. It infers “hue consumes no slot” only from explicit resolved colours/line colours. Categorical `hue=` with marker-only `fmt='o'` groups through the ambient cycle without either marker, and gets counted as ordinary datasets.
- Verified: `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/hypertools-round9/mpl HYPERTOOLS_AUTO_INSTALL=0 PIP_NO_INDEX=1 PIP_CONFIG_FILE=/dev/null PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /tmp/hypertools-round9/composed_markers.py` (`composed_markers.log`). Draw an ordinary 24x3 array, append another with `hue=['a']*12+['b']*12, fmt='o'`, then append ordinary data, all with `palette=['navy','gold','green','purple']`. The last trace is PURPLE `(128,0,128)` on both backends, both plain figure and `hyp.subplots(1,1)` cell, versus GOLD for the otherwise identical `fmt='-'` or `fmt='r-o'` call. Colouring by the same categorical hue should not advance the ordinary cycle only when lines are hidden. This contradicts the new helper's documented rule and CHANGELOG's unqualified hue-composition claim. The R8-3 reset itself remains fixed; this is an uncovered marker-only branch.
- Suggested fix: carry explicit colour ownership (including whether hue controls it) into cycle accounting instead of deducing ownership from renderer kwargs. Add ordinary -> marker-only categorical hue -> ordinary tests with actual artist/trace colours on both figure and cell paths, and verify hue colours ignore fmt colour letters as documented.

### Attribution, test review, and remaining verification

- **R9-3 fitted-model confirmation with the documented dataset count:** `mixed_forecast_fitted_multi.py` (same environment/command prefix as `mixed_forecast.py`) fits one real Kalman forecaster on BOTH unequal-width datasets via `hyp.predict(x, model='Kalman', t=3, return_model=True)`. The ordinary first-dataset plot with `fitted.for_dataset(0)` succeeds, while the grid with the full two-dataset fitted forecaster raises the same 1-versus-3-feature error on BOTH backends (`mixed_forecast_fitted_multi.log`). Thus the failure also holds with the proper multi-dataset fitted model; it is not merely the first probe's one-dataset model being unsuitable for the second panel.
- **Before-state attribution:** reused round8's archived `c700c85f` package, running from `/tmp` with `PYTHONPATH=/tmp/hypertools-round8/before` and `/Users/jmanning/hypertools/.venv/bin/python`. `membership_colors_compat.py` (same probe, tolerating the old absent cluster_labels key) shows the OLD shared grid reclusters each panel into BOTH red and blue (`membership_colors_before.log`). The new replay removes those extra fits but introduces the all-red appearance for distinct global labels: R9-2 is a remaining mapping gap in the new replay, not a claim the old shared memberships were right. `composed_markers.py` against that archive also advances incorrectly (`composed_markers_before.log`); R9-4 is an incompletely fixed composition contract, not newly introduced marker-only behavior. No archive/source copy was placed in the repository.
- **R9-1 source-line correction:** the decisive MRO early return is **tests/_netskip.py:190–193**; new SSL text handling is **228–232**; CHANGELOG's claim is **840–843**. Earlier approximate line references in this round should be read as these exact references.
- **TLS aggregate case does work:** focused `... .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_load_sources.py -k transient` passes **5 tests, 30 deselected in 1.67 s** (`netskip-tests.log`), including the exact Dropbox dropped-TLS/certificate string pair. That is useful coverage, but it does not exercise the live Requests exception path in R9-1.
- **Additional requested public probes pass:** reused round8 `valid_forecasts.py`, same Agg/autoinstall-off environment. All **18** combinations (both backends × shared/independent/reducer-list `panels=2` × none/categorical hue/KMeans regrouping) construct and contain the expected forecast/truth role counts (`valid_forecasts.log`). This checks ownership/counts for equal-width 3-column data, not all numerical forecasts. `edges.py` also verifies wide `ndims=4` PCA+hue/cluster and integer reducer grids construct on both backends; `extra.py` verifies shared/independent GaussianMixture replay constructs. A harmless loky core-count warning reflects this sandbox's sysctl result, not a library failure.
- **Round-6 thread/export probe still preserves scope policy:** `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/hypertools-round9/mpl PIP_NO_INDEX=1 PIP_CONFIG_FILE=/dev/null PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /tmp/hypertools-round6/thread_export.py` reports `inside set_autoinstall(False): False`, `after both contexts: True` (`thread_export.log`). In this interpreter kaleido is installed, so actual export fails with HypertoolsIOError for unusable Chrome; it does not install. The separate missing-kaleido ON/OFF tests above cover the ImportError branches. Fresh nested scopes, three-thread out-of-order exits and the delayed-entry race all pass; no new defect in those policy sequences found.
- **Test review:** `tests/test_plot_review_round8.py:134–177` compares real clustered point sets against separate individual calls; that is behavioral and not tautological. The fit counter at **213–230** is meaningful for “no extra panel fits,” but deliberately measures relative to an ordinary call, which still fits twice; it cannot establish exactly one total fit. Shared coverage at **180–194** checks actual labels and group sizes but neither absent-category colours nor exact row identity per shared cluster (R9-2). Mixed-width tests at **365–425** check observed-row correlation, shape and flat z, but never forecast numbers or fitted models/truth (R9-3). Palette tests at **292–339** use categorical hue with the default line fmt only (R9-4). Docstring substring assertions at **428–432** are structural guards, not evidence that those promises hold. None of this warrants removing the existing tests: add the concrete missing behaviors.
- **Latest lazy-import tests:** retention at `tests/test_lazy_import.py:374–396` has an independent weakref assertion and final policy checks in addition to implementation-specific stack length. New delayed-entry concurrency test at **399–441** checks every Event hand-off, thread termination, in-block values and final baseline; it exercises the reported race without mocking the policy. These are useful tests, not tautologies. Their restoration fixture edits bookkeeping only for isolation. No claim of reusable/reentrant SAME-handle context support is made by this audit; requested fresh nested/multi-thread contexts were exercised.
- **Documentation review:** read relevant plot docstrings, CHANGELOG and `docs/optional_dependencies.rst`; source nits from round 8 are corrected. Remaining drift is tied to the concrete findings: unqualified certificate rejection, hue cycle preservation, shared colour identity and independent-panel behavior do not cover the failing paths above. The Claude session reports full_verify15 results of 5547 pytest passes, Sphinx zero warnings, doctest 316/0 and smoke 344 in `notes/session_2026-09-05_release-1.1-review.md:230`. These remain that session's recorded evidence, not this audit's fresh executions or independently inspected full logs. No gallery, doctests or notebooks were rerun here.

### Next steps / verdict

1. Fix R9-2's shared cluster colour/legend mapping and R9-3's forecasting through padded display coordinates before merge/release, with the concrete public numeric/artist regressions on BOTH backends.
2. Fix R9-1's live SSL classification and R9-4's marker-only hue accounting; extend the focused tests and align the documentation claims. Preserve all existing useful regressions.
3. Re-run affected tests and the established final CI/docs pipeline on the resulting exact head. Successful Chrome/export rendering and full-install validation require the existing unrestricted pipeline; this sandbox's failures do not supply those gates. No merge or publication performed.

Final reviewed HEAD: `650808f0b5880005c19497fb7f7a6d71362869ae`. Final tracked-file status contains only the authorized append to this notes file. All scratch scripts and evidence live under `/tmp`; no other tracked file was modified/staged/committed/deleted, no new repository file left, and no other process's kernel or LSL outlet stopped. Focused pytest totals: **125 passed, 2 environment-limited failures** across the combined review suites, export-policy tests and transient-classifier selection. The exact R8-1..R8-4 reproductions are fixed; this round found **two major and two minor remaining defects**.

VERDICT: FINDINGS

## Updates by the Claude session after Codex round 9 (2026-09-08) — NOT part of the Codex text above

### UPDATE — R9 minor (live certificate failures skipped): FIXED
- `tests/_netskip.py`: `_certificate_failure()` vetoes the live verdict when an
  `SSLCertVerificationError` (or the word 'certificate') sits anywhere in the
  chain or in an exception's `args` (how requests carries the ssl error;
  its `SSLError` inherits from `ConnectionError`, which alone read as
  transient). The reviewer's `/tmp/hypertools-round9/extra.py`: drop
  SKIPPED, certificate RAISED. Test:
  `test_a_live_certificate_failure_is_never_skipped_while_a_tls_drop_is`.

### UPDATE — R9-1 (shared cluster colours per panel): FIXED
- `_PanelClusterLabels` carries the probe's full sorted category set; every
  panel colours line runs and marker groups by the GLOBAL mapping (legend
  names unchanged). Reviewer probe: blue/red on both backends.

### UPDATE — R9-2 (display padding fed the forecast): FIXED
- `_PanelLift` / `_panel_lift_for`: a narrow panel's rows stay in their own
  analyzed space for forecasting, `truth=` and the bundle (equal to the
  individual call, (3, 1) forecasts), and only the DRAWN rows and overlays
  are lifted into the 3-D cell. Fitted two-dataset Kalman and 1-column
  truth work.

### UPDATE — R9-3 (marker-only hue advances the palette): FIXED
- `_palette_slots_consumed(..., category_colored=)` is fed from hue/cluster,
  so a categorical hue consumes no slot whatever the fmt; a marker-only
  categorical hue now resolves explicit category colours (a `'ro'` + hue
  call drew all red on both backends before).
- Tests: `tests/test_plot_review_round9.py` (57).
- Agent observation, queued: `ndims=1` series mode forecasts on the drawn
  (index, value) pairs rather than the values alone (differs from
  `hyp.predict(series)`); to fix with the next wave.

## Codex round 10

Red-team review started 2026-09-08. Findings and verification evidence are appended incrementally below.

### Initial verification

- Reviewed HEAD `25b3ba6c` on `fix/1.1-release-review`. Read the original audit, every UPDATE, rounds 6–9, the release-review CHANGELOG, datatype survey, and `git log --oneline master..HEAD`; `git diff 650808f0..HEAD --stat` reports 44 files. Only this notes file was modified at start. No -o path was supplied in the conversation; this authorized notes append is the persistent report, with scratch evidence under `/tmp/hypertools-round10/`.
- Reused round9 `extra.py`, `membership_colors.py`, and `composed_markers.py`: live TLS drop SKIPPED, certificate RAISED SSLError; shared cluster panels draw blue/red corresponding to global labels 1/0 on BOTH backends; every figure/cell marker-hue composition ends gold on BOTH backends. Command: `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/hypertools-round10/mpl HYPERTOOLS_AUTO_INSTALL=0 PIP_NO_INDEX=1 PIP_CONFIG_FILE=/dev/null PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/jmanning/hypertools .venv/bin/python` with `runpy.run_path` over those scripts. Evidence: `round9.log`. Extra's narrow fitted/truth paths construct with (3,1) forecasts. The old mixed_forecast scripts index column 1 of the formerly padded output and now hit IndexError in their PRINT statement; this is a stale reviewer harness, not a library error. Independent numerical verification follows.

### R10-1 — MINOR: MatrixColormap breaks the advertised matplotlib Colormap contract

- Location: `hypertools/plot/colors.py:450–468` (constructor and `__call__`), docstring at 445–447 explicitly promises inherited `resampled` and `reversed` behavior.
- What is wrong: the constructor passes a LIST of RGB tuples as `LinearSegmentedColormap`'s segment-data argument, which requires channel-indexed segment data. The custom float sampler hides that until inherited operations initialize/read it. Integer sampling, `.resampled()`, and `.reversed()` all fail; the custom sampler also rejects an array of alpha values supported by Colormap.
- Verified with real `matrix_palette(np.random.default_rng(0).normal(size=(8,5)))`, `.venv/bin/python` under Agg/autoinstall-off (`/tmp/hypertools-round10/colormap.log`). Float sampling succeeds; `c(0)`, `c(np.arange(3))`, and `c.resampled(8)(np.linspace(0,1,8))` raise `TypeError: list indices must be integers or slices, not str`; `c.reversed()` raises `AttributeError: 'list' object has no attribute 'items'`; `c([0.,1.], alpha=[.2,.8])` raises TypeError. The persistent script `palettes.py` repeats these public calls.
- Suggested fix: initialize valid channel segment data (e.g. through `from_list`), or implement a Colormap subclass with correct integer, reversed, resampled, alpha/masked/bad/under/over semantics. Add real round-trip reversal and integer/float/resampled sampling tests; current tests only use the custom float arm.

- Focused pytest completed: `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/hypertools-round10/mpl PIP_NO_INDEX=1 PIP_CONFIG_FILE=/dev/null PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_plot_review_round9.py tests/test_palette_matrix_and_sort.py tests/test_polars_inputs.py tests/test_polars_inputs_wave1.py tests/test_datatype_gate.py tests/test_plot_review_round7.py tests/test_plot_review_round6.py tests/test_lazy_import.py`: **265 passed, 2 failed, 16 warnings in 43.15s** (`pytest.log`). Failures are the real tomli-install test with pip deliberately blocked, and unusable installed Chrome; these are environment limitations, not new library findings. All new palette/polars/static-gate/round9 tests pass.

## Updates by the Claude session after Codex round 10 (2026-09-08) — NOT part of the Codex text above

### UPDATE — R10-1 (MatrixColormap broke the Colormap contract): FIXED
- `hypertools/plot/colors.py::MatrixColormap.__init__` now builds the parent's
  channel segment data exactly as `LinearSegmentedColormap.from_list` does
  (plus an alpha segment), so the inherited lookup table, integer sampling,
  `resampled()`, `reversed()`, bad/under/over colours and masked/NaN input all
  work; the exact float sampler accepts an alpha ARRAY and defers masked,
  non-finite and integer input to the parent. Fewer than two anchors raises.
- Test: `test_matrix_colormap_supports_the_inherited_colormap_operations`
  (int index, `np.arange`, `resampled(8)`, `reversed()` at the ends and within
  the table's resolution, alpha array, bytes, masked, NaN, `isinstance(..., Colormap)`).
- Codex round 10 hit the usage limit right after this finding (retry 3:46 PM);
  round 11 (scratchpad codex/prompt13.txt, relaunch13.sh at 15:48) continues
  from here.
