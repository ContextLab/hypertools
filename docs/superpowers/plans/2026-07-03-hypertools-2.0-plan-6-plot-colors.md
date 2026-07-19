# HyperTools 2.0 — Plan 6: plot/ reorg + colors + gaussian Smooth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. **Detailed source context:** `.superpowers/sdd/plan6-plot-research.md` (read for any task touching plot internals).

**Goal:** Deliver the geo-INDEPENDENT plot-module work — add a gaussian smoothing mode to `manip.Smooth`, consolidate coloring into `plot/colors.py`, rename the renderer files to the spec's names (`matplotlib_backend.py`/`plotly_backend.py`), and extract animation helpers into `plot/animate.py` — all while keeping the current `DataGeometry` return value (and the full suite) intact.

**Architecture:** The plot module ALREADY implements the 2.0 functional vision (matplotlib default, plotly optional, `backend='auto'` → plotly only on Colab/Kaggle, robust `mat2colors` coloring, mpl+plotly animations with gif/mp4/svg export). Plan 6 is **reorganization + one real feature gap (gaussian smooth)** — NOT new plotting. **CRITICAL SEQUENCING:** `plot()` constructs and returns a `DataGeometry` in every code path and `tests/test_plot.py` has ~20 geo assertions; the geo→figure return-type flip is therefore entangled with geo deletion and is **deferred to Plan 7** (done atomically with the `test_plot.py` rewrite). Plan 6 must NOT change `plot()`'s return value, must NOT touch `plot_stream` extraction (it reads `geo.ax`), and must NOT delete `DataGeometry`. Renames use shims so internal importers/tests stay green.

**Tech Stack:** Python 3.12 (`.venv`), pytest, datawrangler 0.5.0, matplotlib, plotly, scipy, seaborn, numpy, pandas.

## Global Constraints

- **Interpreter:** ALL commands use `/Users/jmanning/hypertools/.venv/bin/python`. Never bare `python`/`pip`/`pytest`.
- **pandas `>=2.2.0`** (dw 0.5.0). Validated on pandas 3.0.3 / dw 0.5.0 / numpy 2.3.5.
- **Branch:** `dev-2.0-refactor`; never push master.
- **Strangler / green:** old import paths keep working via shims; the full suite (currently **~323 passed** after Plan 5) stays green. Focused tests per task; full suite only at plan close.
- **Geo untouched in Plan 6:** do NOT change `plot()`'s return type, do NOT delete/modify `DataGeometry`, do NOT extract `plot_stream`. All geo removal is Plan 7.
- **rcParams (#259):** never mutate global `rcParams` at import; preserve the existing `plt.rc_context()` / snapshot-restore patterns when moving code.
- **Source-of-truth rule:** dev-2.0 code is trusted; these are moves + one additive feature. Acceptance gates that must stay green: `tests/test_plot.py`, `tests/test_backend.py`, `tests/test_colors.py`, `tests/test_interactive.py`, `tests/test_animation_export.py`, `tests/test_plotly_trails.py`, `tests/test_nested.py`, `tests/manip/`. Do NOT weaken assertions.
- **eval-free, no mocks, real calls.** Commits after each task; trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Don't push.

## File Structure

- `hypertools/manip/smooth.py` — add gaussian mode. **Modify.** (Task 1)
- `hypertools/plot/colors.py` — **NEW** (consolidates `tools/colors.py` + `_shared` `vals2colors`/`vals2bins`). `tools/colors.py` → shim. (Task 2)
- `hypertools/plot/matplotlib_backend.py` (← draw.py), `hypertools/plot/plotly_backend.py` (← interactive.py); `draw.py`/`interactive.py` → shims. **Rename+shim.** (Task 3)
- `hypertools/plot/animate.py` — **NEW** (animation save helpers + animated_svg). (Task 4)
- `tests/plot/…` — **Create** (new module tests; existing top-level plot tests stay as gates).

---

### Task 1: gaussian smoothing mode for manip.Smooth (weights-trajectory recipe)

**Files:**
- Modify: `hypertools/manip/smooth.py`
- Test: `tests/manip/test_smooth_gaussian.py`

**Interfaces:**
- Produces: `Smooth(axis=0, mode='savgol', kernel_width=11, order=3, var=300, maintain_bounds=True)` — `mode='gaussian'` applies `scipy.ndimage.gaussian_filter1d(data, sigma=sqrt(var), axis=...)` per column; `mode='savgol'` (default) unchanged. Reachable via `hyp.manip(data, model='Smooth', mode='gaussian', var=300)`.

**Context (from research §9):** the weights-trajectory gif uses gaussian smoothing with **`var=300`, `sigma=sqrt(var)`**, applied on `axis=0` (time) per dataset, via `scipy.ndimage.gaussian_filter1d` (NOT scipy.signal). `scripts/generate_weights_trajectory.py` hand-rolls this today; Plan 6 folds it into `Smooth`. The current `Smooth` is savgol-only (`kernel_width`/`order`). Preserve savgol as the default `mode` so existing `tests/manip/test_smooth_resample.py` stays green.

- [ ] **Step 1: Write the failing test** — create `tests/manip/test_smooth_gaussian.py`:

```python
import numpy as np
import pandas as pd
from hypertools.manip.smooth import Smooth
from hypertools.manip.manip import manip


def test_gaussian_smooth_reduces_variance():
    rng = np.random.RandomState(0)
    t = np.linspace(0, 4 * np.pi, 200)
    clean = np.sin(t)
    noisy = clean + rng.normal(0, 0.5, size=t.shape)
    df = pd.DataFrame({"x": noisy})
    out = Smooth(mode="gaussian", var=300).fit_transform(df)
    # gaussian-smoothed signal is closer to clean than the noisy input
    assert np.mean((out["x"].to_numpy() - clean) ** 2) < np.mean((noisy - clean) ** 2)


def test_gaussian_matches_scipy_reference():
    from scipy.ndimage import gaussian_filter1d
    rng = np.random.RandomState(1)
    x = rng.rand(120, 3)
    df = pd.DataFrame(x)
    out = np.asarray(Smooth(mode="gaussian", var=300, axis=0).fit_transform(df))
    ref = gaussian_filter1d(x.astype(float), sigma=np.sqrt(300), axis=0)
    assert np.allclose(out, ref, atol=1e-8)


def test_savgol_still_default():
    rng = np.random.RandomState(2)
    df = pd.DataFrame(rng.rand(100, 2))
    # default mode is savgol; must not raise and must change the data
    out = Smooth(kernel_width=11, order=3).fit_transform(df)
    assert np.asarray(out).shape == (100, 2)


def test_gaussian_via_dispatcher():
    df = pd.DataFrame(np.random.RandomState(3).rand(80, 2))
    out = manip(df, model="Smooth", mode="gaussian", var=300)
    assert np.asarray(out).shape == (80, 2)
```

- [ ] **Step 2: Run — expect failure.**

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/manip/test_smooth_gaussian.py -q -p no:cacheprovider`

- [ ] **Step 3: Read `hypertools/manip/smooth.py`.** It has a `@dw.decorate.funnel` `fitter` (computes per-column `max`/`min` for `maintain_bounds`) and a `@dw.decorate.apply_stacked` transformer that per-column calls `scipy.signal.savgol_filter(data[c].values, kernel_width, order)`, with an `axis=1` transpose-recursion branch (added in Plan 3 Task 5b). Add a `mode` kwarg (`'savgol'` default, `'gaussian'`) and a `var` kwarg (default 300) to `Smooth.__init__`, thread them through, and in the transformer branch on `mode`:
  - `mode == 'savgol'`: existing `scipy.signal.savgol_filter(col, kwargs['kernel_width'], kwargs['order'])` path (unchanged).
  - `mode == 'gaussian'`: `scipy.ndimage.gaussian_filter1d(np.asarray(col, dtype=float), sigma=np.sqrt(kwargs['var']))` per column (the per-column loop already fixes the axis, so 1-D `gaussian_filter1d` per column is correct; keep the existing `axis=1` transpose-recursion working for both modes). Keep `maintain_bounds` clamping (`np.clip`) applying to both modes.
  Add `from scipy.ndimage import gaussian_filter1d` at module top (alongside the existing scipy import). Keep the `mode`/`var` params flowing through the `Manipulator` kwargs the same way `kernel_width`/`order` do.

- [ ] **Step 4: Run — expect pass** (fix until green; do not weaken):

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/manip/test_smooth_gaussian.py tests/manip/ -q -p no:cacheprovider
```
(Run all of `tests/manip/` to confirm savgol default + axis=1 behavior didn't regress.)

- [ ] **Step 5: Retire the hand-rolled smoothing in the script** — in `scripts/generate_weights_trajectory.py`, replace the local `def smooth(datasets, var=300): [gaussian_filter1d(...)]` calls with `hyp.manip(d, model='Smooth', mode='gaussian', var=300)` (per dataset, or over the list), so the recipe uses the library. Do NOT run the full gif pipeline (slow/UMAP/ffmpeg); just make the code use the new mode and note it. If the script is not runnable in this environment, update the smoothing lines + add a comment referencing the new `Smooth(mode='gaussian')` and leave a note in the report.

- [ ] **Step 6: Commit**

```bash
git add hypertools/manip/smooth.py tests/manip/test_smooth_gaussian.py scripts/generate_weights_trajectory.py
git commit -m "$(printf 'feat(manip): add gaussian mode to Smooth (var=300, sigma=sqrt(var)) for weights recipe\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 2: consolidate coloring into plot/colors.py (shim tools/colors.py)

**Files:**
- Create: `hypertools/plot/colors.py`
- Modify: `hypertools/tools/colors.py` (→ shim)
- Test: `tests/plot/__init__.py`, `tests/plot/test_colors_module.py`

**Interfaces:**
- Produces: `hypertools.plot.colors.{mat2colors, colors2groups}` (moved from `tools/colors.py`) plus re-exported `vals2colors`, `vals2bins` (from `_shared/helpers.py`, imported — NOT moved out of `_shared`, which many modules still `import *` from). `tools.colors` re-exports `mat2colors`/`colors2groups` (keeps `tests/test_colors.py` + `plot.py`'s `from ..tools.colors import mat2colors, colors2groups` green).

**Context (research §4):** `tools/colors.py` (127 lines) has `mat2colors(m, palette='hls', n_bins=100)`, `colors2groups(colors, res=6)`, and private helpers (`_is_numeric`, `_flatten_if_nested`, `_get_palette`). `_shared/helpers.py` has the legacy `vals2colors(vals, cmap='GnBu', res=100)` / `vals2bins(vals, res=100)`. Do NOT alter `_shared/helpers.py` (it is `import *`-ed widely); just re-export its color fns from `plot/colors.py` for a single import surface.

- [ ] **Step 1: Write the failing test** — create `tests/plot/__init__.py` (empty) and `tests/plot/test_colors_module.py`:

```python
import numpy as np


def test_plot_colors_mat2colors():
    from hypertools.plot.colors import mat2colors
    out = mat2colors(np.array([0.0, 0.5, 1.0]))
    assert np.asarray(out).shape == (3, 3)


def test_plot_colors_reexports_legacy():
    from hypertools.plot.colors import vals2colors, vals2bins, colors2groups
    assert callable(vals2colors) and callable(vals2bins) and callable(colors2groups)


def test_tools_colors_shim_same_objects():
    from hypertools.plot.colors import mat2colors as new_m
    from hypertools.tools.colors import mat2colors as old_m
    assert new_m is old_m
```

- [ ] **Step 2: Run — expect failure.**

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/plot/test_colors_module.py -q -p no:cacheprovider`

- [ ] **Step 3: Move `tools/colors.py` → `plot/colors.py` with git**, then re-point its relative imports for the new depth (plot/ is same depth as tools/, so `..X` unchanged; `.sibling` → `..tools.sibling` if any). Append re-exports of the legacy color helpers so `plot/colors.py` is the single surface:

```bash
git mv hypertools/tools/colors.py hypertools/plot/colors.py
```
At the end of `hypertools/plot/colors.py` add:
```python
# Legacy continuous-color helpers live in _shared.helpers (import *-ed widely);
# re-export them here so plot.colors is the single coloring surface.
from .._shared.helpers import vals2colors, vals2bins  # noqa: F401,E402
```
(If `plot/colors.py` has other `from .X` imports, fix them; verify `.venv/bin/python -c "import hypertools.plot.colors"`.)

- [ ] **Step 4: Recreate `hypertools/tools/colors.py` as a shim:**

```python
# Moved to hypertools.plot.colors (HyperTools 2.0). Shim preserves the old path.
from ..plot.colors import *  # noqa: F401,F403
from ..plot.colors import mat2colors, colors2groups  # noqa: F401
```

- [ ] **Step 5: Update `plot/plot.py`'s import** of colors — change `from ..tools.colors import mat2colors, colors2groups` → `from .colors import mat2colors, colors2groups` (now a sibling in `plot/`). Verify no other in-repo importer breaks: `grep -rn "tools.colors\|from .colors\|from ..plot.colors" hypertools --include=*.py`.

- [ ] **Step 6: Run tests — expect pass:**

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/plot/test_colors_module.py tests/test_colors.py tests/test_nested.py -q -p no:cacheprovider
.venv/bin/python -c "import hypertools; from hypertools.tools.colors import mat2colors; print('OK')"
```

- [ ] **Step 7: Commit**

```bash
git add hypertools/plot/colors.py hypertools/tools/colors.py hypertools/plot/plot.py tests/plot
git commit -m "$(printf 'refactor(plot): consolidate coloring into plot.colors; shim tools.colors\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 3: rename renderers → matplotlib_backend.py / plotly_backend.py (shims)

**Files:**
- Rename: `hypertools/plot/draw.py` → `hypertools/plot/matplotlib_backend.py`; `hypertools/plot/interactive.py` → `hypertools/plot/plotly_backend.py`
- Modify: `hypertools/plot/plot.py` (imports), and recreate `draw.py`/`interactive.py` as shims
- Test: `tests/plot/test_backend_renames.py`

**Interfaces:**
- Produces: `hypertools.plot.matplotlib_backend.{_draw, ...}`, `hypertools.plot.plotly_backend.{detect_environment, resolve_backend, plotly_draw, _parse_fmt, _camera_eye, ...}`. `plot.draw` / `plot.interactive` re-export them (keeps `tests/test_interactive.py`'s `from hypertools.plot.interactive import ...` green).

**Context (research §1):** `plot.py` imports `from .draw import _draw`, `from .interactive import resolve_backend`. `test_interactive.py` imports `detect_environment, resolve_backend, plotly_draw, _parse_fmt, _camera_eye` from `hypertools.plot.interactive`. `draw.py`/`interactive.py` do NOT import `plot.py` (no circularity). `backend.py` (env/GUI switching) is a DIFFERENT file — leave it named `backend.py` (spec keeps it; its rename is not required).

- [ ] **Step 1: Write the failing test** — create `tests/plot/test_backend_renames.py`:

```python
def test_matplotlib_backend_module():
    from hypertools.plot.matplotlib_backend import _draw
    assert callable(_draw)


def test_plotly_backend_module():
    from hypertools.plot.plotly_backend import detect_environment, resolve_backend, plotly_draw
    assert callable(detect_environment) and callable(resolve_backend) and callable(plotly_draw)


def test_old_paths_still_work_via_shim():
    from hypertools.plot.draw import _draw as d
    from hypertools.plot.interactive import plotly_draw as p
    from hypertools.plot.matplotlib_backend import _draw as d2
    from hypertools.plot.plotly_backend import plotly_draw as p2
    assert d is d2 and p is p2
```

- [ ] **Step 2: Run — expect failure.**

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/plot/test_backend_renames.py -q -p no:cacheprovider`

- [ ] **Step 3: git mv the two files:**

```bash
git mv hypertools/plot/draw.py hypertools/plot/matplotlib_backend.py
git mv hypertools/plot/interactive.py hypertools/plot/plotly_backend.py
```
Fix any now-wrong relative imports INSIDE the moved files (they used `from .._shared.helpers import *` — unchanged at plot/ depth; check for `from .backend`/`from .draw`/`from .interactive` sibling refs and repoint to the new names). Verify: `.venv/bin/python -c "import hypertools.plot.matplotlib_backend, hypertools.plot.plotly_backend"`.

- [ ] **Step 4: Update `plot/plot.py` imports** — `from .draw import _draw` → `from .matplotlib_backend import _draw`; `from .interactive import resolve_backend` (and any other `from .interactive import ...`) → `from .plotly_backend import ...`. Grep for every `from .draw`/`from .interactive` across the repo and repoint or rely on the shims below.

- [ ] **Step 5: Recreate `draw.py` and `interactive.py` as shims:**

`hypertools/plot/draw.py`:
```python
# Renamed to hypertools.plot.matplotlib_backend (HyperTools 2.0). Shim preserves the old path.
from .matplotlib_backend import *  # noqa: F401,F403
from .matplotlib_backend import _draw  # noqa: F401
```
`hypertools/plot/interactive.py`:
```python
# Renamed to hypertools.plot.plotly_backend (HyperTools 2.0). Shim preserves the old path.
from .plotly_backend import *  # noqa: F401,F403
from .plotly_backend import (detect_environment, resolve_backend, plotly_draw,  # noqa: F401
                             _parse_fmt, _camera_eye)
```
(Before finalizing the `interactive.py` shim, confirm the exact symbol list `tests/test_interactive.py` imports and re-export each.)

- [ ] **Step 6: Run tests — expect pass:**

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/plot/test_backend_renames.py tests/test_interactive.py tests/test_plot.py tests/test_plotly_trails.py -q -p no:cacheprovider
```

- [ ] **Step 7: Commit**

```bash
git add hypertools/plot tests/plot/test_backend_renames.py
git commit -m "$(printf 'refactor(plot): rename draw->matplotlib_backend, interactive->plotly_backend; shims\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 4: extract animation helpers into plot/animate.py + plan-close

**Files:**
- Create: `hypertools/plot/animate.py`
- Modify: `hypertools/plot/plot.py` (import the moved helpers), `hypertools/_shared/animated_svg.py` (leave in place; re-export from animate)
- Test: `tests/plot/test_animate_module.py`

**Interfaces:**
- Produces: `hypertools.plot.animate.{save_animation, combine_frames_svg, ...}` — the animation SAVE helpers (`_save_animation`, `_SVGFrameCollector`, `_save_animated_svg`) moved out of `plot.py`, plus a re-export of `combine_frames_svg` from `_shared/animated_svg.py`. `plot.py` imports them from `.animate`.

**Context (research §5):** `plot.py` holds `_save_animation(line_ani, save_path, frame_rate)` (dispatches svg/gif/png/mp4), `_SVGFrameCollector(animation.AbstractMovieWriter)`, `_save_animated_svg(...)`; `_shared/animated_svg.py::combine_frames_svg(frame_svgs, duration)` is pure-string (no mpl/plotly deps) and already shared by both backends. Move ONLY the save helpers to `plot/animate.py`; keep the plotly export helpers where they are (in `plotly_backend.py`) to avoid churn. Do NOT change animation BEHAVIOR or `plot()`'s return.

- [ ] **Step 1: Write the failing test** — create `tests/plot/test_animate_module.py`:

```python
def test_animate_module_exposes_save_helpers():
    from hypertools.plot.animate import save_animation
    assert callable(save_animation)


def test_animate_reexports_svg_combiner():
    from hypertools.plot.animate import combine_frames_svg
    assert callable(combine_frames_svg)
```

- [ ] **Step 2: Run — expect failure.**

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/plot/test_animate_module.py -q -p no:cacheprovider`

- [ ] **Step 3: Create `hypertools/plot/animate.py`** — move `_save_animation`, `_SVGFrameCollector`, `_save_animated_svg` out of `plot.py` into it (verbatim bodies, fixing imports: they need `matplotlib.animation`, `PIL`, and `from .._shared.animated_svg import combine_frames_svg`). Expose a public alias `save_animation = _save_animation` and re-export `combine_frames_svg`. Keep the private names too (so `plot.py` can import them).

- [ ] **Step 4: Update `plot/plot.py`** — replace the moved function DEFINITIONS with `from .animate import _save_animation, _SVGFrameCollector, _save_animated_svg` (and use them exactly as before). Verify no behavior change.

- [ ] **Step 5: Run tests + animation export gate:**

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/plot/test_animate_module.py tests/test_animation_export.py -q -p no:cacheprovider
```
(`test_animation_export.py` exercises the moved save helpers via `hyp.plot(..., animate=True, save_path=...)`; gif/apng run unconditionally, mp4 skips without ffmpeg.)

- [ ] **Step 6: PLAN-CLOSE full-suite regression** (controller runs this; ~13 min):

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest -q -p no:cacheprovider
```
Expected: prior count + new plot/manip module tests, no regressions, exit 0.

- [ ] **Step 7: Commit**

```bash
git add hypertools/plot/animate.py hypertools/plot/plot.py tests/plot/test_animate_module.py
git commit -m "$(printf 'refactor(plot): extract animation save helpers into plot.animate; close Plan 6\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Self-Review

**1. Spec coverage (§8 plot + colors):** gaussian `Smooth` (weights recipe acceptance) → Task 1; `plot/colors.py` consolidation → Task 2; `matplotlib_backend.py`/`plotly_backend.py` renames → Task 3; `plot/animate.py` extraction → Task 4. The dual-backend / `backend='auto'` / robust-coloring / animation behaviors already exist and are preserved (research §2–§5). **Deferred to Plan 7 (documented, dependency-driven):** plot's geo→figure return flip + `test_plot.py` rewrite, `plot_stream` extraction (reads `geo.ax`), unified figure-export `save`, `_shared/helpers` geo coupling — all entangled with `DataGeometry` deletion.

**2. Placeholder scan:** concrete code/tests for the gaussian mode; exact git-mv + import-repoint + shim instructions for the moves; no TBD.

**3. Type consistency:** `Smooth(mode=, var=)` threads through the existing `Manipulator` kwargs; shims re-export the exact symbols importers/tests use (`mat2colors`/`colors2groups`; `_draw`; `detect_environment`/`resolve_backend`/`plotly_draw`/`_parse_fmt`/`_camera_eye`; `_save_animation`); `plot.py` import repoints match the new module names.

## Execution Handoff

After Plan 6, **Plan 7** is the big geo removal: delete `DataGeometry`, flip `plot()` to return figures/animations (`return_model=True` threads models) + rewrite `test_plot.py`, extract `plot_stream` into `plot/`, add the unified figure-export `save`, audit/remove `_shared/helpers` geo coupling (`check_geo`/`get_type`/`get_dtype`), retire the `tools/` shims, resolve the classic-callable-vs-subpackage name collision, and decide RSRM (vendor into `external.brainiak` or document as dropped). Then Plan 8 (docs/gallery/notebooks + Playwright) and the whole-branch review + PR.
