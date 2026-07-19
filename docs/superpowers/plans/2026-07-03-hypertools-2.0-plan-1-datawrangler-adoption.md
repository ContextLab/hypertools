# HyperTools 2.0 — Plan 1: datawrangler Adoption & Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `datawrangler` (`pydata-wrangler`) as a hypertools dependency and prove — with a real probe test suite — that dw 0.4.0 exposes every API symbol and behavior the class-based refactor will build on.

**Architecture:** Spec step 0 (verification-first). Declare the dependency, pin pandas to the range where dw works, and land `tests/core/test_dw_probe.py` asserting dw's decorate/stack/core/zoo/zoo.text surface exists and round-trips real data (numpy, pandas, polars, text). Nothing in `hypertools/` changes yet — the existing suite stays green.

**Tech Stack:** Python 3.12 (`.venv`), pytest, `pydata-wrangler>=0.4.0`, pandas (`>=2.2,<3` for now — see constraints), polars (via dw), scikit-learn.

## Global Constraints

Copied from the design spec + verified during controller recon; every task inherits these:

- **Interpreter:** run EVERYTHING through `/Users/jmanning/hypertools/.venv/bin/python` and
  `/Users/jmanning/hypertools/.venv/bin/pip`. **Never** use bare `python`/`pip`/`pytest` — on
  this machine those resolve to anaconda Python 3.9, which is <3.10 and has a broken NumPy ABI.
  The `.venv` is Python 3.12.10 with an editable hypertools install.
- **pandas is pinned `>=2.2,<3`** (currently resolves to 2.3.3). datawrangler 0.4.0's type
  detection breaks under pandas 3.0 (filed **data-wrangler#30**). This ceiling is TEMPORARY;
  lift it when dw#30 lands. Do not "fix" it by removing the ceiling.
- **Branch:** `dev-2.0-refactor`; PR target `dev-2.0`. **Never push `master`.**
- **datawrangler coordination:** when a dw bug or missing API is found, file a
  `ContextLab/data-wrangler` issue via `gh issue create -R ContextLab/data-wrangler` with a
  minimal repro + hypertools call site; log it in `notes/datawrangler_coordination.md`. Prefer
  filing over an internal workaround. (data-wrangler#30 is already filed.)
- **Testing:** real calls only — no mocks. If real functionality doesn't work, the test fails.
- **Commits:** commit after each task; message ends with the trailer
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. Do not push unless asked.

## Known-good facts from controller recon (do not re-derive)

- dw 0.4.0 is already installed in `.venv` (base, no hf extra). All 17 required symbols exist:
  `dw.decorate.{funnel,apply_stacked,list_generalizer}`, `dw.{stack,unstack,wrangle}`,
  `dw.core.{update_dict,apply_defaults,get_default_options}`,
  `dw.zoo.{is_dataframe,is_array,is_multiindex_dataframe}`,
  `dw.zoo.text.{wrangle_text,apply_text_model,is_hugging_face_model,get_text_model,get_corpus}`.
- Under pandas 2.3.3: `dw.stack`/`dw.unstack` round-trip; `dw.zoo.is_dataframe`/
  `is_multiindex_dataframe` return True; `dw.wrangle(docs, model='CountVectorizer')` embeds text
  to a DataFrame. The probe suite's happy path therefore PASSES.

## File Structure

- `pyproject.toml` — add `pydata-wrangler` to `dependencies`; add the `<3` ceiling to the
  existing pandas pin; add a `text` extra (`pydata-wrangler[hf]`). **Modify only.**
- `tests/core/__init__.py` — empty package marker. **Create.**
- `tests/core/test_dw_probe.py` — the dw API-surface + behavior probe suite. **Create.** Never imports `hypertools`.
- `.github/workflows/test.yml` — add `dev-2.0-refactor` to triggers; note the 3.13/dw status. **Modify only.**
- `notes/datawrangler_coordination.md` — running log of dw issues + the pinned-version decision. **Create.**

---

### Task 1: Declare the datawrangler dependency and pandas ceiling

**Files:**
- Modify: `pyproject.toml` (dependencies list; optional-dependencies)

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `import datawrangler as dw` resolves in `.venv`; a `hypertools[text]` extra mapping to `pydata-wrangler[hf]`; pandas constrained to `>=2.2,<3`.

- [ ] **Step 1: Edit the `dependencies` list**

In `pyproject.toml`, change the pandas line and append datawrangler so the block reads:

```toml
dependencies = [
    "scikit-learn>=1.4.0",
    # pandas pinned below 3.0 TEMPORARILY: datawrangler 0.4.0 type detection breaks on
    # pandas 3.0 (data-wrangler#30). Lift the <3 ceiling once that lands.
    "pandas>=2.2.0,<3",
    "seaborn>=0.13.0",
    "matplotlib>=3.8.0",
    "scipy>=1.13.0",
    "numpy>=2.0.0",
    "umap-learn>=0.5.5",
    "requests>=2.31.0",
    "dill>=0.3.8",
    "ipympl>=0.9.3",
    # HyperTools 2.0: data-wrangling core (funnel/stack/unstack/format detection, model
    # dispatch, text+HF embeddings). https://github.com/ContextLab/data-wrangler
    "pydata-wrangler>=0.4.0",
]
```

- [ ] **Step 2: Add the `text` extra**

In `[project.optional-dependencies]`, add the `text` extra between `interactive` and `dev`:

```toml
# Transformer / sentence-transformers text embeddings via datawrangler's hf extra
# (torch/transformers/sentence-transformers/tokenizers/datasets). Opt-in; never in base install.
text = ["pydata-wrangler[hf]>=0.4.0"]
```

- [ ] **Step 3: Reinstall editable and confirm dw resolves in `.venv`**

Run:
```bash
cd /Users/jmanning/hypertools
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -c "import datawrangler as dw, pandas as pd; print('dw', dw.__version__, '| pandas', pd.__version__)"
```
Expected: installs cleanly; prints `dw 0.4.0 | pandas 2.3.3` (pandas must be < 3). If pip resolves pandas ≥ 3, the `<3` ceiling was not applied — fix Step 1.

- [ ] **Step 4: Smoke-check for regression**

This task only edits `pyproject.toml` text; the packages it references (`pydata-wrangler`,
pandas `<3`) are already installed identically in `.venv`, and the controller established a
fresh green baseline of **242 passed** on this exact environment state (recorded in
`.superpowers/sdd/progress.md`). A full 13-minute rerun would only re-confirm 242, so run a
fast smoke check instead:

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_reduce.py tests/test_normalize.py tests/test_format_data.py -q -p no:cacheprovider
```
Expected: all pass, no new import/collection errors. If anything fails here, the dependency/pin
change regressed something — stop and report. (The full suite runs at the end of the plan.)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "$(printf 'build: add datawrangler dep; pin pandas <3 for dw 0.4.0 (data-wrangler#30)\n\nHyperTools 2.0 step 0: adopt datawrangler for the wrangling core. Adds\npydata-wrangler>=0.4.0, a hypertools[text] extra -> pydata-wrangler[hf]\n(opt-in transformer embeddings), and a temporary pandas<3 ceiling because\ndw 0.4.0 type detection breaks on pandas 3.0 (data-wrangler#30).\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

### Task 2: Probe the datawrangler API surface (symbol existence)

**Files:**
- Create: `tests/core/__init__.py` (empty)
- Create: `tests/core/test_dw_probe.py`

**Interfaces:**
- Consumes: `import datawrangler as dw`.
- Produces: `tests/core/test_dw_probe.py::test_dw_symbols_exist` — canonical list of dw symbols later plans consume.

- [ ] **Step 1: Create the test package marker**

Create `tests/core/__init__.py` as an empty file.

- [ ] **Step 2: Write the symbol-existence probe**

Create `tests/core/test_dw_probe.py`:

```python
"""datawrangler API-surface probe.

Verification-first gate for HyperTools 2.0 (spec step 0). Asserts the installed
datawrangler exposes every symbol and behavior the class-based refactor builds
on. A failure means dw drifted from what we verified at 0.4.0: file a
ContextLab/data-wrangler issue, then mark the specific check xfail with the
issue link (see notes/datawrangler_coordination.md). Real calls only.
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
    ("datawrangler", "wrangle"),
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

- [ ] **Step 3: Run the probe (expected PASS — all symbols confirmed present at 0.4.0)**

Run:
```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/core/test_dw_probe.py::test_dw_symbols_exist -v
```
Expected: all 17 parametrized cases PASS. **If any FAIL** (dw version drift): file the gap with `gh issue create -R ContextLab/data-wrangler`, wrap the failing param with `pytest.param(..., marks=pytest.mark.xfail(reason="dw#<N>: <attr> missing", strict=True))`, and record it in `notes/datawrangler_coordination.md`.

- [ ] **Step 4: Commit**

```bash
git add tests/core/__init__.py tests/core/test_dw_probe.py
git commit -m "$(printf 'test: probe datawrangler API surface (symbol existence)\n\nCanonical list of dw symbols the 2.0 refactor depends on. Missing symbols\nbecome filed data-wrangler issues + xfail-with-link, not silent green.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

### Task 3: Probe datawrangler behavior (real round-trips)

**Files:**
- Modify: `tests/core/test_dw_probe.py` (append behavioral tests)

**Interfaces:**
- Consumes: the symbols asserted in Task 2.
- Produces: proof that `dw.stack`/`dw.unstack` round-trip a list of DataFrames, that `@dw.decorate.funnel` generalizes over numpy/pandas/polars/list, and that dw embeds text — the behaviors Plans 2–6 rely on.

- [ ] **Step 1: Append the stack/unstack round-trip test**

```python
def test_stack_unstack_roundtrip():
    a = pd.DataFrame(np.arange(6).reshape(3, 2), columns=["x", "y"])
    b = pd.DataFrame(np.arange(6, 14).reshape(4, 2), columns=["x", "y"])
    stacked = dw.stack([a, b])
    assert dw.zoo.is_multiindex_dataframe(stacked), "stack should yield a MultiIndex frame"
    restored = dw.unstack(stacked)
    assert isinstance(restored, list) and len(restored) == 2
    assert restored[0].shape == (3, 2) and restored[1].shape == (4, 2)
    np.testing.assert_array_equal(restored[0].to_numpy(), a.to_numpy())
    np.testing.assert_array_equal(restored[1].to_numpy(), b.to_numpy())
```

- [ ] **Step 2: Append the funnel generalization test (numpy / pandas / list)**

```python
def test_funnel_generalizes_over_input_types():
    @dw.decorate.funnel
    def n_columns(data, **kwargs):
        if isinstance(data, list):
            return [d.shape[1] for d in data]
        return data.shape[1]

    arr = np.arange(6).reshape(3, 2)
    df = pd.DataFrame(arr, columns=["x", "y"])
    assert n_columns(arr) == 2
    assert n_columns(df) == 2
    assert n_columns([arr, arr]) == [2, 2]
```

- [ ] **Step 3: Append the polars-input test**

```python
def test_funnel_accepts_polars():
    pl = pytest.importorskip("polars")

    @dw.decorate.funnel
    def n_rows(data, **kwargs):
        if isinstance(data, list):
            return [d.shape[0] for d in data]
        return data.shape[0]

    pdf = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    assert n_rows(pdf) == 3
```

- [ ] **Step 4: Append the text-embedding test (verified call; sklearn path, no torch)**

```python
def test_dw_text_sklearn_embedding():
    docs = ["the cat sat", "the dog ran", "cats and dogs"]
    # dw.wrangle routes text through its text zoo; 'CountVectorizer' is the pure-sklearn
    # path (no torch/hf required). Returns one row per document.
    embedded = dw.wrangle(docs, model="CountVectorizer")
    frame = embedded[0] if isinstance(embedded, list) else embedded
    assert dw.zoo.is_dataframe(frame)
    assert frame.shape[0] == 3, "one row per document"
    assert frame.shape[1] >= 1, "at least one feature column"
```

- [ ] **Step 5: Run all behavioral probes**

Run:
```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/core/test_dw_probe.py -v
```
Expected: all PASS (verified during recon under pandas 2.3.3). **If a behavioral test fails** because dw's signature differs from what's recorded here: confirm dw's real contract with `.venv/bin/python -c "import datawrangler as dw, inspect; print(inspect.signature(dw.wrangle))"`, adjust the test to dw's actual API if the difference is cosmetic (kwarg name), OR file a data-wrangler issue + `xfail` if it's a genuine gap. Record the outcome in `notes/datawrangler_coordination.md`.

- [ ] **Step 6: Commit**

```bash
git add tests/core/test_dw_probe.py
git commit -m "$(printf 'test: probe datawrangler behavior (stack/unstack, funnel, polars, text)\n\nProves the real round-trips Plans 2-6 depend on: MultiIndex stack/unstack,\nfunnel generalization over numpy/pandas/list/polars, and sklearn text\nembedding via dw.wrangle.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

### Task 4: CI triggers, py3.13/dw note, and the dw-issue log

**Files:**
- Modify: `.github/workflows/test.yml` (push/PR trigger branches)
- Create: `notes/datawrangler_coordination.md`

**Interfaces:**
- Consumes: the installed dw.
- Produces: CI that runs on `dev-2.0-refactor`; a coordination log seeded with data-wrangler#30.

- [ ] **Step 1: Add the refactor branch to CI triggers**

Edit `.github/workflows/test.yml`:
- Push branches (line ~5): `branches: [ master, dev, dev-2.0, dev-2.0-refactor ]`
- Leave the matrix `python-version: ['3.10', '3.11', '3.12', '3.13']` unchanged. Add a comment above it:
  ```yaml
  # dw 0.4.0 classifiers stop at py3.12; the 3.13 row exercises dw-on-3.13. If a 3.13
  # job fails on dw install/import, file a data-wrangler issue and temporarily drop 3.13
  # here (comment referencing the issue) rather than working around it.
  ```

- [ ] **Step 2: Create the coordination log (seeded with dw#30)**

Create `notes/datawrangler_coordination.md`:

```markdown
# datawrangler coordination log

The 2.0 refactor adopts datawrangler (`pydata-wrangler`) for the wrangling core. A parallel
Claude Code instance maintains `/Users/jmanning/data-wrangler` (repo `ContextLab/data-wrangler`,
default branch `main`). When we hit a dw bug or missing API, we file an issue there rather than
working around it.

## Environment
- Adopted `pydata-wrangler>=0.4.0` (PyPI latest 0.4.0, 2025-06-14).
- Refactor interpreter: `.venv` (Python 3.12.10). pandas pinned `>=2.2,<3` (see dw#30).

## Filed issues
| # | Title | Blocking? | Status |
|-|-|-|-|
| 30 | pandas 3.0 type detection (`type(x).__module__` strings) breaks is_dataframe / is_multiindex_dataframe → stack/unstack fail | Not blocking (pandas pinned <3) | open |

## Pending: lift the pandas<3 ceiling once dw#30 lands, then add pandas 3.0 to the CI matrix.

## API notes / deltas from the fork's dw usage
- Text embedding entry: `dw.wrangle(docs, model='CountVectorizer')` (also accepts
  `text_kwargs={'model': ...}`); fits on the bundled minipedia corpus by default.
```

- [ ] **Step 3: Confirm the workflow is valid YAML and commit**

Run:
```bash
.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/test.yml')); print('workflow YAML OK')"
```
Expected: `workflow YAML OK`.

```bash
git add .github/workflows/test.yml notes/datawrangler_coordination.md
git commit -m "$(printf 'ci: run on dev-2.0-refactor; seed datawrangler coordination log (dw#30)\n\nAdds the refactor branch to CI triggers, documents the py3.13/dw status, and\nstarts the data-wrangler issue-coordination log.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

## Self-Review

**1. Spec coverage (this plan = spec step 0 only):** install/pin dw → Task 1; dw-API probe →
Tasks 2 (symbols) + 3 (behavior); CI reconcile → Task 4; data-wrangler issue workflow → Task 4
log + remediation steps; first polars proof → Task 3 Step 3; sklearn text path → Task 3 Step 4
(transformer/hf real test deferred to the text plan). Steps 1–12 of the spec are out of scope.

**2. Placeholder scan:** the `<fill …>` in the ledger is a runtime value, not a plan
placeholder; every code/command step is complete. No "TBD/handle edge cases".

**3. Type consistency:** `REQUIRED_SYMBOLS` names match Task 3's usage (`dw.stack`, `dw.unstack`,
`dw.decorate.funnel`, `dw.wrangle`, `dw.zoo.is_dataframe`, `dw.zoo.is_multiindex_dataframe`) and
what Plans 2–6 consume.

## Execution Handoff

After this plan runs green, Plan 2 (core layer) is written next, informed by the confirmed dw
0.4.0 surface and the pandas<3 baseline.
