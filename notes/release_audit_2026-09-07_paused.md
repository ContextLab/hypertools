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
