# HyperTools 2.0 — Plan 5: io/ (load + sources + streaming + save) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Re-home dev-2.0's working `load`/`sources`/`streaming` (DATA side) into a new `hypertools/io/` package with `tools/` re-export shims, and add a data-serialization `io/save.py` (`hyp.save`/`hyp.io.save`), while keeping the full suite green.

**Architecture:** Strangler step 7, additive. dev-2.0 is the trusted template; the moves mirror Plan 4's reduce/cluster re-homing (git mv + relative-import fixups + shim). `io/` sits at the same depth as `tools/`, so `..X` imports are unchanged but `.sibling` imports (e.g. `from .analyze`, `from .reduce`) must be repointed. **Scope note (deviation from spec §7, documented):** the spec's `save.py` figure-exporter (`hyp.save(fig,'out.gif')` → png/pdf/svg/html/gif/mp4) depends on the plot module/backends, which don't exist until Plan 6 — so Plan 5's `save.py` covers only **data/object serialization** (a clean `dw.io.save` passthrough, else pickle); the figure-export formats are deferred to Plan 6. `plot_stream` (the animated stream renderer inside `streaming.py`) is moved wholesale with the file for now; its extraction to `plot/` is also deferred to Plan 6. **GEO REMOVAL (per Jeremy, reaffirmed 2026-07-03):** DataGeometry is being deleted entirely — do NOT build geo-preserving machinery. `save.py` is a plain serializer with NO `.geo` special-casing. `io/load.py` still *returns* a geo in Plan 5 only as strangler-continuity (test_load.py + consumers still expect it); the geo return type + `DataGeometry` deletion + `.geo` handling are all retired in Plan 7. Plan 5 adds no new geo coupling.

**Tech Stack:** Python 3.12 (`.venv`), pytest, datawrangler 0.5.0, requests, pickle, numpy, pandas.

## Global Constraints

- **Interpreter:** ALL commands use `/Users/jmanning/hypertools/.venv/bin/python`. Never bare `python`/`pip`/`pytest`.
- **pandas `>=2.2.0`** (no upper pin — dw 0.5.0). Validated on pandas 3.0.3 / dw 0.5.0 / numpy 2.3.5.
- **Branch:** `dev-2.0-refactor`; never push master.
- **Strangler / green:** old import paths keep working via shims; the full suite (currently **~317 passed** after Plan 4) stays green. Focused tests per task; full suite only at plan close.
- **Source-of-truth rule:** dev-2.0 behavior preserved exactly. Acceptance gates that must stay green: `tests/test_load.py`, `tests/test_load_sources.py`, `tests/test_streaming.py`. Do NOT weaken assertions. NOTE: some load tests hit the network / download real datasets — that is intended (real calls, no mocks); if a network test is flaky, re-run, do not mock.
- **Naming note:** `hyp.load` is a classic callable; `hyp.io` is a NEW subpackage with NO competing classic callable, so `hyp.io` resolves to the subpackage and `hyp.io.load`/`hyp.io.save` work. `hyp.save` is a new classic callable (data serialization for now). (Contrast the reduce/align/cluster collision — those have classic callables owning their names; io does not.)
- **eval-free, no mocks, real calls.** Commits after each task; trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Don't push.

## File Structure

- `hypertools/io/__init__.py`, `io/load.py` (← tools/load.py), `io/sources.py` (← tools/sources.py), `io/streaming.py` (← tools/streaming.py). **Create** (moved).
- `hypertools/io/save.py` — **NEW** (data serialization).
- `hypertools/tools/{load,sources,streaming}.py` — become **shims**. **Modify.**
- `hypertools/__init__.py` — add `save` + `io` surface (Task 2). **Modify.**
- `tests/io/…` — **Create.**

---

### Task 1: io/ package — move load + sources + streaming (shim tools/)

**Files:**
- Create: `hypertools/io/__init__.py`, `hypertools/io/load.py`, `hypertools/io/sources.py`, `hypertools/io/streaming.py`
- Modify: `hypertools/tools/load.py`, `hypertools/tools/sources.py`, `hypertools/tools/streaming.py` (→ shims)
- Test: `tests/io/__init__.py`, `tests/io/test_io_module.py`

**Interfaces:**
- Produces: `hypertools.io.load.load`, `hypertools.io.sources.{is_loadable_string, ...}`, `hypertools.io.streaming.{is_stream, row_to_vector, _fit_stream_models, plot_stream, ...}`. `tools.{load,sources,streaming}` re-export these (every current importer keeps working).

- [ ] **Step 1: Write the failing test** — create `tests/io/__init__.py` (empty) and `tests/io/test_io_module.py`:

```python
def test_io_load_importable():
    from hypertools.io.load import load
    assert callable(load)


def test_io_sources_importable():
    from hypertools.io.sources import is_loadable_string
    assert callable(is_loadable_string)


def test_io_streaming_data_side_importable():
    from hypertools.io.streaming import is_stream, row_to_vector
    assert callable(is_stream) and callable(row_to_vector)


def test_tools_shims_are_same_objects():
    from hypertools.io.load import load as new_load
    from hypertools.tools.load import load as old_load
    from hypertools.io.streaming import is_stream as new_is_stream
    from hypertools.tools.streaming import is_stream as old_is_stream
    assert new_load is old_load and new_is_stream is old_is_stream
```

- [ ] **Step 2: Run — expect failure** (`hypertools.io` missing).

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/io/test_io_module.py -q -p no:cacheprovider`

- [ ] **Step 3: Move the three files with git (preserve history)**

```bash
cd /Users/jmanning/hypertools
git mv hypertools/tools/load.py hypertools/io/load.py
git mv hypertools/tools/sources.py hypertools/io/sources.py
git mv hypertools/tools/streaming.py hypertools/io/streaming.py
```

- [ ] **Step 4: Fix relative imports in the moved files.** `io/` is the same depth as `tools/`, so `..X` (e.g. `..datageometry`, `.._shared.exceptions`) are UNCHANGED, but any `.sibling` import (a former `tools/` sibling) must become `..tools.sibling` OR the sibling's new home. Grep every relative import and fix:

```bash
grep -nE "^\s*from \.[a-zA-Z_]|^\s*from \.\.[a-zA-Z_]|^\s*import \." hypertools/io/load.py hypertools/io/sources.py hypertools/io/streaming.py
```
Known fixes required:
- `io/load.py`: `from .analyze import analyze` → `from ..tools.analyze import analyze`. (Keep `from ..datageometry import DataGeometry` and `from .._shared.exceptions import HypertoolsIOError` as-is.) If `load` imports `from .sources import ...` or `from .streaming import ...`, repoint to `from .sources import ...` (still valid — now a sibling inside `io/`) — verify the sibling actually lives in `io/` now.
- `io/streaming.py`: `from .reduce import _resolve_model` → `from ..reduce.reduce import _resolve_model` (reduce was re-homed in Plan 4). Also any `from .normalize import` → `from ..tools.normalize import`; any plot/draw imports (used by `plot_stream`) → repoint to their `..plot....` location. Fix each until import succeeds.
- `io/sources.py`: repoint any `.sibling` import to `..tools.<sibling>` or the sibling's new `io.`/`reduce.` home.

After each fix, verify the module imports:
```bash
.venv/bin/python -c "import hypertools.io.load, hypertools.io.sources, hypertools.io.streaming; print('io modules import OK')"
```

- [ ] **Step 5: Create `hypertools/io/__init__.py`:**

```python
from .load import load
from . import sources
from . import streaming
```

- [ ] **Step 6: Recreate the three `tools/` files as shims** (re-export EVERYTHING each module currently exposes to other importers — grep first to be exhaustive):

`hypertools/tools/load.py`:
```python
# Moved to hypertools.io.load (HyperTools 2.0). Shim preserves the old path.
from ..io.load import *  # noqa: F401,F403
from ..io.load import load  # noqa: F401
```
`hypertools/tools/sources.py`:
```python
# Moved to hypertools.io.sources (HyperTools 2.0). Shim preserves the old path.
from ..io.sources import *  # noqa: F401,F403
from ..io.sources import is_loadable_string  # noqa: F401
```
`hypertools/tools/streaming.py`:
```python
# Moved to hypertools.io.streaming (HyperTools 2.0). Shim preserves the old path.
# NOTE: plot_stream (the animated renderer) rides along here for now; its move to
# hypertools.plot is deferred to Plan 6.
from ..io.streaming import *  # noqa: F401,F403
from ..io.streaming import is_stream, row_to_vector, _fit_stream_models, plot_stream  # noqa: F401
```
(Before finalizing each shim, run `grep -rn "tools.load\|tools.sources\|tools.streaming\|from .load\|from .sources\|from .streaming" hypertools --include=*.py` to find every symbol other modules import from these paths, and make sure the shim re-exports each. `plot/plot.py` and `datageometry.py` are likely importers.)

- [ ] **Step 7: Run tests + focused regression + import smoke**

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/io/test_io_module.py tests/test_load.py tests/test_load_sources.py tests/test_streaming.py -q -p no:cacheprovider
.venv/bin/python -c "import hypertools as hyp; print('hyp import OK; load callable:', callable(hyp.load))"
```
Expected: all pass (existing load/sources/streaming tests green via shims). Note: `test_load.py` may download datasets — allow network time; do not mock.

- [ ] **Step 8: Commit**

```bash
git add hypertools/io hypertools/tools/load.py hypertools/tools/sources.py hypertools/tools/streaming.py tests/io
git commit -m "$(printf 'refactor(io): re-home load+sources+streaming into hypertools.io; shim tools\n\nplot_stream rides along in io.streaming for now (moves to plot/ in Plan 6).\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 2: io/save.py (data serialization) + wire hyp.save/hyp.io + plan-close

**Files:**
- Create: `hypertools/io/save.py`
- Modify: `hypertools/io/__init__.py`, `hypertools/__init__.py`
- Test: `tests/io/test_save.py`

**Interfaces:**
- Produces: `hypertools.io.save.save(obj, fname, **kwargs)` — serializes data/objects to disk. `hyp.save` (new classic callable) and `hyp.io` (subpackage: `hyp.io.load`, `hyp.io.save`). Figure-export formats (png/pdf/svg/html/gif/mp4) are OUT OF SCOPE (Plan 6).

- [ ] **Step 1: Write the failing test** — create `tests/io/test_save.py`:

```python
import numpy as np
import os


def test_save_roundtrips_array(tmp_path):
    from hypertools.io.save import save
    import pickle
    arr = np.random.RandomState(0).rand(6, 3)
    fname = str(tmp_path / "data.pkl")
    save(arr, fname)
    assert os.path.exists(fname)
    with open(fname, "rb") as f:
        loaded = pickle.load(f)
    assert np.allclose(np.asarray(loaded), arr)


def test_hyp_save_and_io_exposed(tmp_path):
    import hypertools as hyp
    assert callable(hyp.save)
    # hyp.io is the subpackage (no competing classic callable)
    assert hasattr(hyp.io, "load") and hasattr(hyp.io, "save")
    fname = str(tmp_path / "x.pkl")
    hyp.save(np.arange(5), fname)
    assert os.path.exists(fname)
```

- [ ] **Step 2: Run — expect failure.**

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/io/test_save.py -q -p no:cacheprovider`

- [ ] **Step 3: Create `hypertools/io/save.py`** — a plain data-serialization saver. Prefer `dw.io.save` when available (the fork delegated to it), falling back to pickle. NO `.geo` / DataGeometry special-casing (geo is being deleted — Plan 7). Figure export is explicitly deferred to Plan 6.

```python
"""Data/object serialization for HyperTools 2.0.

`save(obj, fname)` writes a hypertools result (array / DataFrame / list /
fitted model) to disk and exposes a standalone `hyp.save`. Prefers
datawrangler's serializer when available, else pickle. No geo special-casing
(DataGeometry is removed in 2.0).

NOTE: figure/animation export (png/pdf/svg/html/gif/mp4) is NOT handled here —
it depends on the plot backends and lands in Plan 6 (plot). A figure passed here
is pickled like any other object; the format-aware exporter comes later.
"""
import pickle


def save(obj, fname, **kwargs):
    """Serialize `obj` to `fname` (datawrangler serializer if available, else pickle)."""
    try:
        import datawrangler as dw
        if hasattr(dw, "io") and hasattr(dw.io, "save"):
            return dw.io.save(obj, fname, **kwargs)
    except Exception:
        pass
    with open(fname, "wb") as f:
        pickle.dump(obj, f)
```
(If `dw.io.save` exists but its signature/behavior differs from `(obj, fname)`, prefer the plain pickle path so the test's pickle round-trip holds; validate empirically and document in the report. Do NOT weaken the round-trip test.)

- [ ] **Step 4: Update `hypertools/io/__init__.py`:**

```python
from .load import load
from .save import save
from . import sources
from . import streaming
```

- [ ] **Step 5: Wire `hyp.save` + `hyp.io` into `hypertools/__init__.py`.** Inspect first (`grep -n "^from\|^import\|^from \.io" hypertools/__init__.py`). Add (near the other classic imports):
```python
from .io.save import save
from . import io
```
Do NOT remove any existing export. Verify `hyp.load` (classic), `hyp.save`, and `hyp.io.load`/`hyp.io.save` all resolve.

- [ ] **Step 6: Run the save tests + public-API smoke:**

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/io/ -q -p no:cacheprovider
.venv/bin/python -c "import hypertools as hyp; print('save', callable(hyp.save), '| io.load', hasattr(hyp.io,'load'), '| io.save', hasattr(hyp.io,'save'))"
```

- [ ] **Step 7: PLAN-CLOSE full-suite regression** (controller may run this; ~13 min):

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest -q -p no:cacheprovider
```
Expected: prior count + new io tests, no regressions, exit 0.

- [ ] **Step 8: Commit**

```bash
git add hypertools/io/save.py hypertools/io/__init__.py hypertools/__init__.py tests/io/test_save.py
git commit -m "$(printf 'feat(io): add data-serialization save + expose hyp.save/hyp.io; close Plan 5\n\nFigure-export formats (png/gif/mp4/...) deferred to Plan 6 (need plot backends).\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Self-Review

**1. Spec coverage (§7 io):** load/sources/streaming re-homed with shims → Task 1; `save` (data-serialization portion) + `hyp.save`/`hyp.io` → Task 2. Figure-export `save.py` formats and the `plot_stream`→`plot/` extraction are explicitly DEFERRED to Plan 6 (documented deviation — both depend on the plot module). Streams stay first-class via the preserved `is_stream`/`row_to_vector`/`_fit_stream_models` DATA side.

**2. Placeholder scan:** concrete code or exact git-mv + grep-driven import-fixup instructions for every step; no TBD.

**3. Type consistency:** shims re-export the exact names other modules import (`load`, `is_loadable_string`, `is_stream`, `row_to_vector`, `_fit_stream_models`, `plot_stream`); `hyp.io` is a genuine subpackage (no classic-callable collision); `save(obj, fname)` signature consistent between `io/save.py` and both tests.

## Execution Handoff

After Plan 5, Plan 6 (plot + colors) builds the dual-backend plot module, extracts `plot_stream` from `io/streaming.py` into `plot/`, completes `save.py` with figure-export formats, and addresses the owed gaussian `Smooth` mode for the weights-trajectory recipe.
