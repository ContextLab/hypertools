# Fix-design intel (controller investigations while waves run)

Investigation-only — no code changes until Phase 4 verification completes.

## F18-001 (critical): load('sotus') returns broken sklearn Pipeline

**Root cause chain (all verified by real runs 2026-07-11):**
1. `hypertools/io/load.py` EXAMPLE_DATA maps BOTH `'sotus'` and `'nips_model'` to Drive id `1J0MBhpRwdT2WChfWJ4HXYq6jU4XpyJPm` (cached files sha-identical). Same duplication exists on `master` — ancient data-hosting bug, not a 1.0 regression; 0.x example even has a workaround comment.
2. Corpus identification via LDA vocabulary (controller run): `1J0MB…` topics = *neurons, cortex, estimator, lemma, polynomial* → it IS the **nips_model** (correct entry). The `'sotus'` mapping is the paste error. `sotus_model` (`16_n9…`) topics = *surplus, hillary, bosnia, saddam, inspectors* → genuinely SOTU-trained (correct).
3. All six historical `'sotus'` ids recovered from git history (`19qG1mm…`, `1D2dsr…`, `1EgCUf…`, `1JTUzf…`, `1UBjWR…`, `1wrSzof…`) are **dead on Drive** (HypertoolsIOError on real probes) → original speeches file unrecoverable.
4. **Recovery path found:** `datawrangler.zoo.text.get_corpus('sotus')` (datawrangler is a core dependency) returns ndarray of **29 real SOTU addresses 1989–2018** — verified by content ("Mr. Speaker, Mr. President, and distinguished Members of the House and Senate…"). Exactly matches the load() docstring claim.

**Blast radius:** `text2mat.py:246-269` — `corpus='sotus'` loads the broken pipeline as corpus *text* (`np.array(load('sotus'))`), so the documented `corpus='sotus'` feature in plot()/format_data()/text2mat() is broken by the same root cause.

**Proposed fix (pending verification phase):** repoint `'sotus'` in the hosted-data registry to delegate to `dw.zoo.text.get_corpus('sotus')` (decide return type: list of str for consistency with text handling); regression test = real load + content assertion (starts with "Mr. Speaker", len==29, all strings); fixes corpus='sotus' path too; update load() docstring dataset table (also covers F18-003/004 doc fixes).
Note: dw's get_corpus prints "loading corpus: sotus...done!" to stdout — decide whether to suppress around the call.

## F18-002 (major): hosted *_model files are sklearn-1.0.2 pickles

repr()/get_params()/clone() crash under sklearn 1.8 (`transform_input` attr missing); .transform() still works. Re-pickling needs the training corpora + re-hosting. Options (decide at fix time):
- (a) repair-on-load: after unpickle, backfill missing sklearn-1.8 attrs (surgical, no re-hosting, keeps model outputs byte-stable);
- (b) retrain LDA models from dw corpora on first use + local cache (kills stale-pickle class entirely; changes topic values → check docs/tests for baked numbers);
- (c) re-host modern pickles (needs hosting decision — GitHub release asset or Jeremy's Drive).
Leaning (a) for 1.0 stability + file (b) as tracked enhancement, but let the verification/fix phase decide.

## Remaining criticals — root causes + fix sketches (from auditor line-level diagnosis)

### F06-001: $HYPERTOOLS_BACKEND env var crashes `import hypertools`
`plot/backend.py:550-556` `_init_backend`: reorder branch calls `backends.index(HYPERTOOLS_BACKEND)` using the module GLOBAL (still None) instead of the local `env_backend` → ValueError; `finally:` at ~691 references never-assigned `working_backend` → UnboundLocalError masks everything; import dies. Two latent bugs in the same branch: case-mismatch (`in` check lowercases, `index` doesn't) and bad splice `(backends[:idx], *backends[idx+1:])` nests a tuple as element 0 (should be `(*backends[:idx], ...)`). Fix: use `env_backend` with case-normalized lookup, fix splice, make `finally` guard `working_backend`, add real regression test spawning a subprocess with the env var set.

### F06-002: failed backend switch permanently corrupts state
`backend.py:1052-1065` `__enter__` sets `IN_SET_CONTEXT=True` and assigns `HYPERTOOLS_BACKEND` BEFORE `switch_backend()`; on raise, `__exit__` never runs → flag stuck, bad backend kept, BACKEND_WARNING cleared — reachable from plain `hyp.plot(animate=True)` after a bad `set_interactive_backend`, and from the class docstring's own TkAgg example on Tk-less machines. Fix: mutate globals only after a successful switch (or try/except restore); plus eager validation in set_interactive_backend (covers F06-003).

### F08-001 (== F01-004): plain list-of-lists numeric matrix crashes plot()
`plot/plot.py:1574-1576` `_flatten_nested` recurses into inner numeric lists and yields SCALARS as leaves → N*M singleton "datasets" → internally-generated color list of len N*M vs 1 merged dataset → nonsense "color= was given as..." error. `format_data` alone handles the same input; docstring advertises it. Fix: leaf-detect a pure numeric matrix (inner elements all scalars) before recursing so `[[1,2],[3,4]]` is ONE dataset; preserve the legit nested-GROUPS feature (examples/plot_nested_lists.py must stay identical — regression-test both).

### F12-001: trim_and_pad silently scrambles row order (non-RangeIndex)
`align/common.py` ~L41: `rows = list(set(index_values))` + `.loc[rows]` → hash-order rows for DatetimeIndex/string/shuffled-int indices; silent because align returns bare arrays. Consistent across datasets so dispersion metrics look fine — pure silent data corruption for timeseries users. Fix: preserve first-dataset index order: `common = set.intersection(*[set(d.index) for d in data]); rows = [r for r in data[0].index if r in common]` (order-preserving, deterministic); regression test with DatetimeIndex + identifiable rows (col0==i) asserting exact order preservation.

## Newer criticals (waves 2-3) — root causes + fix sketches

### F14-001 == D01-001: Smooth leaks data across dataset boundaries
`manip/smooth.py:95` — `_transform_stacked` decorated `@dw.decorate.apply_stacked`: a LIST is vertically stacked BEFORE the savgol/gaussian/boxcar filter, so subject i's edges contaminate subject i+1 (~kernel_width/2 samples per side, every boundary). Contradicts manip.py docstring ("per-dataset manipulation"), docs/pipeline_order.rst, and README lines 80-86. Resample does it correctly (`resample.py:110` dw.unstack + per-dataset). Fix: per-dataset application for Smooth (mirror resample's pattern). Regression: A=zeros(30,3), B=ones(30,3) → sm[0][-1]==[0,0,0] and sm[1][0]==[1,1,1] exactly. Audit the other manipulators for the same decorator misuse.

### F16-001: predict Kalman never learns dynamics (flat forecasts)
`predict/kalman.py` — `KalmanFilter(...).em(x, n_iter)` without `em_vars`; pykalman's default fits ONLY covariances+initial state, never transition/observation matrices → A stays identity → forecast = last state repeated. Verified: sine fit leaves transition==[[1.]], all forecast rows identical. Fix: pass em_vars including 'transition_matrices' (+observation_matrices); regression: sine forecast correlates with held-out truth (r>0.5) and transition != I.

### D05-001: impute Kalman zero-fills ALL missing values on wide data (D>=50)
`impute/kalman.py` — same em() default + n_dim_state=d full-dim state + 5 EM iters: smoothed means collapse to the zero prior for wide data. Sweep: D=5 r=0.948, D=20 r=0.645, D>=50 exactly 0.0 fills (incl. plot_impute.py's own (100,100) data — the example's "from neighboring timepoints" claim is false today). Fix: em_vars fix first, re-sweep; if wide-D still degenerate, low-rank state (n_dim_state=min(d,k)) or documented guard + warning on degenerate fills (std==0 vs observed). Fix plot_impute example claims to match verified behavior.

### F16-002: hyp.predict(1-D series) destroyed by row-vector wrangling
(200,) → (1,200) DataFrame via the dw funnel: default Kalman crashes deep in pykalman; ARIMA/Laplace/Chronos return (30,200) silent nonsense (input echoed). Fix: predict's input funnel treats 1-D as univariate series (n,1) — document; regression: predict(sine, t=30).shape==(30,1) + correlation check; list [1.0, 2.0, ...] scalar-list case too.

### F19-001: single-column .csv/.txt silently corrupted by delimiter sniffing
`io/sources.py:744-745` (and :774 extensionless): sep=None + engine='python' → csv.Sniffer picks an in-word character for single-column files → mangled multi-column DataFrame, no warning; empty .csv raises raw _csv.Error (F19-005). Fix: .csv defaults sep=','; sniff only as fallback (and validate sniff result); single-column .txt guard; friendly empty-file error. Regression: single-col csv/txt round-trips exactly; multi-delim files still sniff right.

### D07-001: plot.ipynb tutorial crashes on fresh Run-All
Cells 26/29 build hue as 5×int(8124/5)=8120 labels vs 8124 rows → the (correct) QC hue-length validation now raises; everything after cell 26 dead; committed outputs are stale (still show figures). Fix: build hue with np.array_split sizing (handles remainder), re-execute the notebook fresh end-to-end, commit updated outputs. Sweep other tutorials for the same 'int(n/k)*k' pattern.

## Phase-4 design refinement (verification wave)

Per-UNIT verifier agents (~46) rather than per-finding (~450): each verifier gets its unit's findings as {id, title, repro, expected, actual, evidence paths} ONLY (no auditor reasoning), re-runs every repro against the audit branch, examines evidence PNGs directly (agents read images), returns per-finding verdicts {CONFIRMED|REFUTED|CANNOT-REPRO, severity_adjust, notes}. Controller pre-pass before dispatch: dedup obvious cross-unit clusters ([infra] stale-venv ×10 → env-resolved, skip; plot-shadows-subpackage ×4 → one finding; docstring-no-examples ×3 → one; fmt-list-length ×2; etc.).

## Environment notes

- macOS has no `timeout` command — use Bash tool timeout param or python-side timeouts (bitten twice across sessions).
- `hyp.load(<bare-drive-id>)` downloads from Drive by design (sources.py `_DRIVE_ID_RE`) — used for the id probes above; also confirms F18-007's "typo-shaped ids hit the network" observation.
