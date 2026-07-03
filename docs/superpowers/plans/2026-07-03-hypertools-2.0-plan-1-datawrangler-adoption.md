# HyperTools 2.0 — Plan 1: datawrangler Adoption & Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `datawrangler` (`pydata-wrangler`) as a hypertools dependency and prove — with a real probe test suite — that dw 0.4.0 exposes every API symbol and behavior the class-based refactor will build on, before any module is written against it.

**Architecture:** This is spec step 0 (verification-first). We declare the dependency, install it, and land a `tests/core/test_dw_probe.py` suite that asserts dw's decorate/stack/core/zoo/zoo.text surface exists and round-trips real data (numpy, pandas, polars, text). Any gap is filed as a `ContextLab/data-wrangler` issue (a parallel Claude Code instance fixes dw at the source) and marked `xfail` with the issue link, so the gate is honest rather than silently green. Nothing in `hypertools/` changes yet — the existing suite stays green.

**Tech Stack:** Python ≥3.10, pytest, `pydata-wrangler>=0.4.0` (+ its `hf` extra for transformer embeddings), pandas (primary), polars (secondary, via dw), scikit-learn.

## Global Constraints

Copied verbatim from the design spec (`docs/superpowers/specs/2026-07-03-hypertools-2.0-class-refactor-design.md`); every task inherits these:

- **Branch:** work on `dev-2.0-refactor`; PR target is `dev-2.0`. **Never push `master`.**
- **Python floor:** `requires-python = ">=3.10"`; CI matrix is `['3.10','3.11','3.12','3.13']` × {ubuntu, windows, macos}. dw declares support for 3.9–3.12 — reconcile 3.13 in this plan.
- **pandas-first:** pandas is the default/primary DataFrame; polars is supported through dw but never the default.
- **datawrangler coordination:** when a dw bug or missing/changed API is found, file a GitHub issue on `ContextLab/data-wrangler` via `gh issue create -R ContextLab/data-wrangler` with a minimal repro and the hypertools call site. Prefer filing over an internal workaround. Track filed issues in `notes/`.
- **Testing:** real calls only — no mocks, no stubs. If real functionality doesn't work, the test fails/raises.
- **Commits:** commit after each task with a descriptive message ending in the `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer. Do not push unless Jeremy asks.

## File Structure

- `pyproject.toml` — add `pydata-wrangler` to `dependencies`, a new `text` extra (`pydata-wrangler[hf]`), and `polars` to `dev`. **Modify only.**
- `tests/core/__init__.py` — new empty package marker so `tests/core/` is collectable. **Create.**
- `tests/core/test_dw_probe.py` — the dw API-surface + behavior probe suite. **Create.** Single responsibility: assert dw provides what the refactor needs. Never imports `hypertools`.
- `.github/workflows/test.yml` — add `dev-2.0-refactor` to trigger branches; reconcile the 3.13 row. **Modify only.**
- `notes/datawrangler_coordination.md` — running log of filed dw issues + the pinned-version decision. **Create.**

---

### Task 1: Declare the datawrangler dependency

**Files:**
- Modify: `pyproject.toml:28-39` (dependencies), `:41-44` (optional-dependencies), `:45-57` (dev extra)

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `import datawrangler as dw` works in the dev environment; a `hypertools[text]` extra that installs `pydata-wrangler[hf]`.

- [ ] **Step 1: Add the base dependency**

In `pyproject.toml`, add to the `dependencies` list (after the `"dill>=0.3.8",` line, keeping alphabetical-ish grouping is not required — match existing order):

```toml
dependencies = [
    "scikit-learn>=1.4.0",
    "pandas>=2.2.0",
    "seaborn>=0.13.0",
    "matplotlib>=3.8.0",
    "scipy>=1.13.0",
    "numpy>=2.0.0",
    "umap-learn>=0.5.5",
    "requests>=2.31.0",
    "dill>=0.3.8",
    "ipympl>=0.9.3",
    # HyperTools 2.0: data-wrangling core (funnel/stack/unstack/format detection,
    # model dispatch, text+HF embeddings). https://github.com/ContextLab/data-wrangler
    "pydata-wrangler>=0.4.0",
]
```

- [ ] **Step 2: Add the `text` extra and polars dev dep**

Replace the `[project.optional-dependencies]` block so it reads:

```toml
[project.optional-dependencies]
# Interactive plotly backend (auto-enabled on Colab/Kaggle, where plotly ships
# preinstalled; local users opt in with `pip install hypertools[interactive]`).
interactive = ["plotly>=5.20.0", "kaleido>=0.2.1"]
# Transformer / sentence-transformers text embeddings, provided by datawrangler's
# hf extra (torch/transformers/sentence-transformers/tokenizers/datasets). Opt-in;
# never pulled by the base install.
text = ["pydata-wrangler[hf]>=0.4.0"]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "nbformat>=5.9.0",
    "nbclient>=0.9.0",
    "ipykernel>=6.29.0",
    "plotly>=5.20.0",
    "kaleido>=0.2.1",
    # real Hugging Face streaming tests (tests/test_streaming.py)
    "datasets>=2.20.0",
    # kill hung tests (kaleido subprocesses have wedged for 6h on Windows CI)
    "pytest-timeout>=2.3.0",
    # explicit polars-DataFrame input tests (polars also arrives transitively via dw)
    "polars>=0.20.0",
]
```

- [ ] **Step 3: Install and confirm the import resolves**

Run:
```bash
cd /Users/jmanning/hypertools
pip install -e ".[dev]"
python -c "import datawrangler as dw; print('dw', dw.__version__)"
```
Expected: installs without resolver errors; prints `dw 0.4.0` (or newer). If pip reports a dependency conflict (e.g. numpy/pandas pin clash between hypertools and dw), STOP and file a `ContextLab/data-wrangler` issue with the exact resolver output, then record it in `notes/datawrangler_coordination.md` (Task 4 creates that file — create it now if needed).

- [ ] **Step 4: Confirm the existing suite is still green (no regression from the new dep)**

Run:
```bash
MPLBACKEND=Agg pytest -q -x
```
Expected: same pass count as before Plan 1 (baseline ~239 passing). The new dep must not perturb existing tests.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "$(printf 'build: add datawrangler (pydata-wrangler) dependency + text extra\n\nHyperTools 2.0 refactor step 0: adopt datawrangler for the wrangling core.\nAdds pydata-wrangler>=0.4.0 to base deps, a hypertools[text] extra mapping\nto pydata-wrangler[hf] (transformer embeddings, opt-in), and polars to dev\nfor explicit polars-DataFrame tests.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

### Task 2: Probe the datawrangler API surface (symbol existence)

**Files:**
- Create: `tests/core/__init__.py` (empty)
- Create: `tests/core/test_dw_probe.py`

**Interfaces:**
- Consumes: `import datawrangler as dw` (from Task 1).
- Produces: `tests/core/test_dw_probe.py::test_dw_symbols_exist` — the canonical list of dw symbols the refactor depends on. Later plans reference these exact names: `dw.decorate.funnel`, `dw.decorate.apply_stacked`, `dw.decorate.list_generalizer`, `dw.stack`, `dw.unstack`, `dw.core.update_dict`, `dw.core.apply_defaults`, `dw.core.get_default_options`, `dw.zoo.is_dataframe`, `dw.zoo.is_array`, `dw.zoo.is_multiindex_dataframe`, `dw.zoo.text.wrangle_text`, `dw.zoo.text.apply_text_model`, `dw.zoo.text.is_hugging_face_model`, `dw.zoo.text.get_text_model`, `dw.zoo.text.get_corpus`.

- [ ] **Step 1: Create the test package marker**

Create `tests/core/__init__.py` as an empty file:

```python
```

- [ ] **Step 2: Write the symbol-existence probe (failing-first)**

Create `tests/core/test_dw_probe.py`:

```python
"""datawrangler API-surface probe.

Verification-first gate for HyperTools 2.0 (spec step 0). Asserts that the
installed datawrangler exposes every symbol and behavior the class-based
refactor builds on. A failure here means dw drifted from what the fork used:
file a ContextLab/data-wrangler issue, then mark the specific check xfail with
the issue link (see notes/datawrangler_coordination.md). Real calls only.
"""
import importlib

import numpy as np
import pandas as pd
import pytest

import datawrangler as dw


# (module path, attribute) pairs the refactor depends on.
REQUIRED_SYMBOLS = [
    ("datawrangler.decorate", "funnel"),
    ("datawrangler.decorate", "apply_stacked"),
    ("datawrangler.decorate", "list_generalizer"),
    ("datawrangler", "stack"),
    ("datawrangler", "unstack"),
    ("datawrangler.core", "update_dict"),
    ("datawrangler.core", "apply_defaults"),
    ("datawrangler.core", "get_default_options"),
    ("datawrangler.zoo", "is_dataframe"),
    ("datawrangler.zoo", "is_array"),
    ("datawrangler.zoo", "is_multiindex_dataframe"),
    ("datawrangler.zoo.text", "wrangle_text"),
    ("datawrangler.zoo.text", "apply_text_model"),
    ("datawrangler.zoo.text", "is_hugging_face_model"),
    ("datawrangler.zoo.text", "get_text_model"),
    ("datawrangler.zoo.text", "get_corpus"),
]


@pytest.mark.parametrize("module_path,attr", REQUIRED_SYMBOLS)
def test_dw_symbols_exist(module_path, attr):
    module = importlib.import_module(module_path)
    assert hasattr(module, attr), (
        f"datawrangler {dw.__version__} is missing {module_path}.{attr}; "
        f"file a ContextLab/data-wrangler issue and xfail this param with the link"
    )
```

- [ ] **Step 3: Run the probe**

Run:
```bash
MPLBACKEND=Agg pytest tests/core/test_dw_probe.py::test_dw_symbols_exist -v
```
Expected: all parametrized cases PASS. **If any FAIL:** file the gap with
```bash
gh issue create -R ContextLab/data-wrangler \
  --title "0.4.x missing <module>.<attr> used by hypertools 2.0" \
  --body "hypertools 2.0 refactor needs \`<module>.<attr>\`. Present in the fork's dw usage; absent in 0.4.0. Minimal repro: \`import datawrangler as dw; dw...\`. Please restore/rename."
```
then wrap the failing param with `pytest.param(..., marks=pytest.mark.xfail(reason="dw#<N>: <attr> missing", strict=True))` and record the issue number in `notes/datawrangler_coordination.md`.

- [ ] **Step 4: Commit**

```bash
git add tests/core/__init__.py tests/core/test_dw_probe.py
git commit -m "$(printf 'test: probe datawrangler API surface (symbol existence)\n\nCanonical list of dw symbols the 2.0 refactor depends on. Gate is honest:\nmissing symbols become filed data-wrangler issues + xfail-with-link, not\nsilent green.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

### Task 3: Probe datawrangler behavior (real round-trips)

**Files:**
- Modify: `tests/core/test_dw_probe.py` (append behavioral tests)

**Interfaces:**
- Consumes: the symbols asserted in Task 2.
- Produces: proof that `dw.stack`/`dw.unstack` round-trip a list of DataFrames, that `@dw.decorate.funnel` generalizes a function over numpy / pandas / polars / list inputs, and that dw's sklearn text path embeds documents — the exact behaviors Plans 2–6 rely on.

- [ ] **Step 1: Write the stack/unstack round-trip test**

Append to `tests/core/test_dw_probe.py`:

```python
def test_stack_unstack_roundtrip():
    a = pd.DataFrame(np.arange(6).reshape(3, 2), columns=["x", "y"])
    b = pd.DataFrame(np.arange(6, 14).reshape(4, 2), columns=["x", "y"])
    stacked = dw.stack([a, b])
    assert dw.zoo.is_multiindex_dataframe(stacked), "stack should yield a MultiIndex frame"
    restored = dw.unstack(stacked)
    assert isinstance(restored, list) and len(restored) == 2
    pd.testing.assert_frame_equal(
        restored[0].reset_index(drop=True), a.reset_index(drop=True), check_dtype=False
    )
    pd.testing.assert_frame_equal(
        restored[1].reset_index(drop=True), b.reset_index(drop=True), check_dtype=False
    )
```

- [ ] **Step 2: Write the funnel generalization test (numpy / pandas / list)**

Append:

```python
def test_funnel_generalizes_over_input_types():
    @dw.decorate.funnel
    def n_columns(data, **kwargs):
        # funnel hands the body a DataFrame (or list of DataFrames)
        if isinstance(data, list):
            return [d.shape[1] for d in data]
        return data.shape[1]

    arr = np.arange(6).reshape(3, 2)
    df = pd.DataFrame(arr, columns=["x", "y"])
    assert n_columns(arr) == 2
    assert n_columns(df) == 2
    assert n_columns([arr, arr]) == [2, 2]
```

- [ ] **Step 3: Write the polars-input test**

Append:

```python
def test_funnel_accepts_polars():
    pl = pytest.importorskip("polars")

    @dw.decorate.funnel
    def n_rows(data, **kwargs):
        return data.shape[0]

    pdf = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    assert n_rows(pdf) == 3
```

- [ ] **Step 4: Write the sklearn text-embedding test (no hf extra needed)**

Append:

```python
def test_dw_text_sklearn_embedding():
    docs = ["the cat sat", "the dog ran", "cats and dogs"]
    # CountVectorizer is a pure-sklearn text model; no torch/hf required.
    embedded = dw.wrangle(docs, text_kwargs={"model": "CountVectorizer"})
    frame = embedded[0] if isinstance(embedded, list) else embedded
    assert dw.zoo.is_dataframe(frame)
    assert frame.shape[0] == 3, "one row per document"
    assert frame.shape[1] >= 1, "at least one vocabulary column"
```

- [ ] **Step 5: Run all behavioral probes**

Run:
```bash
MPLBACKEND=Agg pytest tests/core/test_dw_probe.py -v
```
Expected: all PASS. **If a behavioral test fails** because dw's call signature differs (e.g. `dw.wrangle` text kwarg name, or `funnel` body receives a different type): confirm the real dw API with `python -c "import datawrangler as dw; help(dw.wrangle)"`, adjust the test to dw's actual 0.4.0 contract IF the difference is cosmetic (kwarg name), OR file a data-wrangler issue + `xfail` if it's a genuine capability gap. Record either outcome in `notes/datawrangler_coordination.md`.

- [ ] **Step 6: Commit**

```bash
git add tests/core/test_dw_probe.py
git commit -m "$(printf 'test: probe datawrangler behavior (stack/unstack, funnel, polars, text)\n\nProves the real round-trips Plans 2-6 depend on: MultiIndex stack/unstack,\nfunnel generalization over numpy/pandas/list/polars, and sklearn text\nembedding via dw.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

### Task 4: Reconcile Python 3.13, CI triggers, and the dw-issue workflow

**Files:**
- Modify: `.github/workflows/test.yml:5` (push branches), `:7` (PR branches), `:14-16` (matrix)
- Create: `notes/datawrangler_coordination.md`

**Interfaces:**
- Consumes: the installed dw from Task 1.
- Produces: a CI config that runs on `dev-2.0-refactor`, a documented decision on the 3.13 row, and a living log of dw issues.

- [ ] **Step 1: Determine whether dw imports on Python 3.13**

Run:
```bash
python --version
python -c "import sys, datawrangler as dw; print(sys.version.split()[0], 'dw', dw.__version__, 'OK')"
```
Record the local Python version and result. If local Python is not 3.13, additionally check availability:
```bash
python3.13 -c "import datawrangler" 2>&1 | head -3 || echo "3.13 interpreter or dw-on-3.13 unavailable locally"
```

- [ ] **Step 2: Decide and apply the CI matrix**

Edit `.github/workflows/test.yml`:

- Line 5 — add the refactor branch to push triggers:
  ```yaml
      branches: [ master, dev, dev-2.0, dev-2.0-refactor ]
  ```
- Line 7 — add it to PR triggers:
  ```yaml
      branches: [ master, dev, dev-2.0 ]
  ```
  (PRs target `dev-2.0`; leave as-is plus `dev-2.0` already implied — ensure `dev-2.0` is present.)
- Lines 14-16 — the matrix decision:
  - **If Step 1 showed dw imports cleanly on 3.13:** leave `python-version: ['3.10', '3.11', '3.12', '3.13']` unchanged.
  - **If dw fails on 3.13:** file a data-wrangler issue (below), then temporarily scope 3.13 out of the dw-dependent path by leaving the matrix but adding `continue-on-error: ${{ matrix.python-version == '3.13' }}` to the `test` job's `runs-on` step block is not valid at job level — instead cap the matrix to `['3.10', '3.11', '3.12']` with an inline comment `# 3.13 re-enabled when data-wrangler#<N> lands` and record the issue number.

- [ ] **Step 3: File the 3.13 issue if needed**

Only if Step 1 showed a 3.13 failure:
```bash
gh issue create -R ContextLab/data-wrangler \
  --title "Support Python 3.13" \
  --body "hypertools 2.0 CI targets py3.10-3.13. datawrangler 0.4.0 classifiers stop at 3.12 and import fails on 3.13 with: <paste traceback>. Please add 3.13 support so hypertools can keep its full matrix."
```

- [ ] **Step 4: Create the coordination log**

Create `notes/datawrangler_coordination.md`:

```markdown
# datawrangler coordination log

The 2.0 refactor adopts datawrangler (`pydata-wrangler`) for the wrangling core.
A parallel Claude Code instance maintains `/Users/jmanning/data-wrangler`
(repo: `ContextLab/data-wrangler`, default branch `main`). When we hit a dw bug
or missing/changed API, we file an issue there rather than working around it.

## Pinned version
- Adopted: `pydata-wrangler>=0.4.0` (latest on PyPI: 0.4.0, 2025-06-14).
- Local dev version: <fill from Task 1 Step 3>.

## Python 3.13 status
- Local probe result (Task 4 Step 1): <fill in>.
- CI matrix decision: <full 3.10-3.13 | capped at 3.12 pending data-wrangler#N>.

## Filed issues
| # | Title | Blocking? | Status |
|-|-|-|-|
| (none yet) | | | |

## API notes / deltas from the fork's dw usage
- <record any 0.4.0 signature differences discovered by the probe here>
```

- [ ] **Step 5: Confirm CI file is valid YAML and commit**

Run:
```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/test.yml')); print('workflow YAML OK')"
```
Expected: `workflow YAML OK`.

```bash
git add .github/workflows/test.yml notes/datawrangler_coordination.md
git commit -m "$(printf 'ci: run on dev-2.0-refactor; reconcile py3.13 with datawrangler; log dw coordination\n\nAdds the refactor branch to CI triggers, records the py3.13/dw decision, and\nstarts the data-wrangler issue-coordination log.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

## Self-Review

**1. Spec coverage (this plan = spec step 0 only):**
- "Install `pydata-wrangler[hf]`, pin" → Task 1. ✓
- "dw-API probe test" → Tasks 2 (symbols) + 3 (behavior). ✓
- "reconcile CI py-matrix (dw 3.9–3.12 vs our 3.10–3.13)" → Task 4. ✓
- "file data-wrangler issues for gaps" workflow → Tasks 2/3/4 remediation steps + `notes/datawrangler_coordination.md`. ✓
- "polars supported through the funnel, with real tests" (first proof) → Task 3 Step 3. ✓
- Text/HF routing verified (sklearn path) → Task 3 Step 4; the transformer-embed (hf extra) real test is deferred to the text/plot plan, noted in spec §12. ✓
- Downstream steps 1–12 are explicitly out of scope for Plan 1 (separate plans).

**2. Placeholder scan:** The `<fill in>` markers in `notes/datawrangler_coordination.md` are intentional runtime-recorded values, not plan placeholders; every code/command step contains complete content. No "TBD/TODO/handle edge cases" in executable steps.

**3. Type consistency:** The symbol names in Task 2's `REQUIRED_SYMBOLS` match those used in Task 3's behavioral tests (`dw.stack`, `dw.unstack`, `dw.decorate.funnel`, `dw.zoo.is_multiindex_dataframe`, `dw.zoo.is_dataframe`) and are the same names Plans 2–6 will consume. `dw.wrangle` (Task 3 Step 4) is dw's top-level wrangle entry point; if the probe reveals its text-kwarg name differs in 0.4.0, Task 3 Step 5 handles the reconciliation.

---

## Execution Handoff

After this plan runs green, Plan 2 (core layer — eval-free `apply_model`, `configurator`+`config.ini`, `unpack_model`, `util`, `exceptions`) is written next, informed by the probe's confirmation of dw's real 0.4.0 surface.
