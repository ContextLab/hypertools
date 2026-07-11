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

## Environment notes

- macOS has no `timeout` command — use Bash tool timeout param or python-side timeouts (bitten twice across sessions).
- `hyp.load(<bare-drive-id>)` downloads from Drive by design (sources.py `_DRIVE_ID_RE`) — used for the id probes above; also confirms F18-007's "typo-shaped ids hit the network" observation.
