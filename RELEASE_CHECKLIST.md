# HyperTools 1.0 release checklist

The docs, notebooks, and README deliberately ship in **dev form** on the
`dev-1.0` branch and must flip to **release form** at publish. Several of those
flips cannot be done earlier (PyPI does not yet have 1.0; the `v1.0.0` tag does
not exist yet), so they are done here, on `master`, in order — and the
**`release-gate` CI job** (runs only on `master` / tags) hard-fails until every
flip is done, so nothing can be forgotten.

Run everything from a **clean `master` checkout on the `master` branch** (not a
detached tag checkout — the notebook migrator detects the branch via
`git rev-parse --abbrev-ref HEAD`, which returns `HEAD` when detached).

## 0. Pre-flight (on `dev-1.0`)

- [ ] `dev-1.0` CI fully green (push + PR workflows).
- [ ] Full suite green locally: `pytest` (2470+ passed, 0 failed).
- [ ] Decide the release date and the version (`1.0.0`).

## 1. Merge to `master`

- [ ] Merge `dev-1.0` → `master` (PR #281). Do NOT delete `dev-1.0` yet (the
      pre-release notebooks still reference it until step 2 runs).

## 2. Flip everything to release form (on `master`)

- [ ] **Notebooks → PyPI spec + clean note (automated).**
      `python scripts/add_colab_install_cell.py`
      Rewrites all 15 tutorial install cells `... @ git+…@dev-1.0` →
      `hypertools[<extras>]` (extras preserved) and strips the
      `(<branch> preview)` / "On release this becomes …" note. It also
      re-generates the gallery (`docs/auto_examples`) install cells; `docs/conf.py`
      produces the identical line automatically on a `master`/tag build.
- [ ] **README images: commit SHA → `v1.0.0` tag (8 URLs).**
      `sed -i '' 's#/ContextLab/hypertools/fc2429cb550de10611031ae34d3c723267daeea4/#/ContextLab/hypertools/v1.0.0/#g' readme.md`
      (drop the `''` after `-i` on GNU sed). Verify: `grep -c 'fc2429cb' readme.md` → 0.
- [ ] **CHANGELOG date.** Edit `CHANGELOG.md`: `## 1.0.0 (unreleased)` →
      `## 1.0.0 (YYYY-MM-DD)` with the real release date.
- [ ] (Optional prose) `docs/tutorials/stock_forecasting.ipynb` has a
      free-text "hypertools 1.0 preview" comment the migrator does not touch —
      reword if desired (not gate-enforced).
- [ ] **Verify the gate locally BEFORE committing:**
      `HYPERTOOLS_REQUIRE_RELEASE=1 pytest -v tests/test_notebook_install_gate.py tests/test_release_readiness_gate.py`
      → all green (no branch installs, no preview note, images on the tag,
      CHANGELOG dated).
- [ ] Commit all of the above on `master` in one release commit.

## 3. Build + verify artifacts

- [ ] `python -m build` → `dist/hypertools-1.0.0.tar.gz` + `…-py3-none-any.whl`.
- [ ] `twine check dist/*` → PASSED.
- [ ] sdist/wheel contain the bundled font + licenses: `tar tzf dist/*.tar.gz | grep -E 'NotoSans|OFL|CHANGELOG'`.
- [ ] Fresh-venv smoke: install the wheel in a throwaway venv, `import hypertools`, `hypertools.__version__ == '1.0.0'`.

## 4. Publish to PyPI, THEN tag

Order matters: the migrated notebooks install `hypertools[…]` from PyPI, so PyPI
must have 1.0 first; the README images resolve at the `v1.0.0` tag, so the tag
must point at the release commit from step 2.

- [ ] `twine upload dist/*` (PyPI now serves 1.0.0).
- [ ] `git tag -a v1.0.0 -m "HyperTools 1.0.0"` at the step-2 release commit.
- [ ] `git push origin v1.0.0` (and `git push origin master`).

## 5. Post-release verification

- [ ] **`release-gate` CI green** on the `master` push AND on the `v1.0.0` tag
      (plus `docs-clean`, `dataset-gate`, `wheel-smoke`).
- [ ] `pip install hypertools` in a clean env → installs `1.0.0`; run the
      README quick-start snippet.
- [ ] **Read the Docs**: trigger/confirm a build of the `v1.0.0` tag (and
      point the "stable"/default version at it). The released docs' Colab
      install cells must show `%pip install "hypertools[interactive]"`
      (no `git+`) — `docs/conf.py` emits this automatically on a tag build.
- [ ] PyPI project page renders the README with all 8 images resolving (they
      now point at the `v1.0.0` tag).
- [ ] After the release is confirmed good, delete the `dev-1.0` /
      `dev-1.0-refactor` branches if desired (the released artifacts no longer
      reference them; the `release-gate` guarantees this).

## What the `release-gate` enforces (so you can't forget)

`tests/test_notebook_install_gate.py` + `tests/test_release_readiness_gate.py`,
run with `HYPERTOOLS_REQUIRE_RELEASE=1` by the `release-gate` CI job on
`master`/tags, fail if ANY of these survive:

| Check | Release form required |
|-|-|
| tutorial install cells | `hypertools[…]` PyPI spec (no `git+`/`@<branch>`) |
| tutorial install-cell note | no `(… preview)` / "On release this becomes …" |
| README image URLs | `…/ContextLab/hypertools/v1.0.0/images/…` (a version tag, not a commit SHA) |
| README branch refs | no `dev-1.0-refactor` / `hypertools.git@dev…` |
| CHANGELOG heading | `## 1.0.0 (YYYY-MM-DD)`, not `(unreleased)` |

Always-on (every branch): no notebook installs the defunct `dev-1.0-refactor`;
all tutorial branch-installs share one branch; every README image is a single
consistent ref and exists in the tree.
