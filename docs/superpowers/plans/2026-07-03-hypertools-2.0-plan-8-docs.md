# HyperTools 2.0 — Plan 8: docs/gallery/notebooks migration + Playwright verify Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. **REQUIRED READING:** `.superpowers/sdd/plan8-docs-research.md` — the exact file/line inventory (§2 gallery table, §3 line_ani examples, §4 notebooks, §7 .md5/artifacts, §8 Playwright).

**Goal:** Migrate the Sphinx gallery examples, tutorial notebooks, and any geo-referencing docs to the 2.0 figure-return / no-geo API, verify the built HTML with Playwright, and assemble PR evidence — completing the refactor.

**Architecture:** The library is code-complete (Plans 1–7; 318 tests pass; `plot()`→Figure, `load()`→raw data, no public `DataGeometry`). Docs still call the old geo API. Fix the `examples/*.py` SOURCES (sphinx-gallery regenerates `docs/auto_examples/`), rewrite the two geo-*about* pages, migrate+re-execute the 13 tutorial notebooks, rebuild, and Playwright-verify.

**Tech Stack:** Python 3.12 (`.venv`), Sphinx + sphinx-gallery + nbsphinx + furo, matplotlib, plotly+kaleido, ffmpeg, playwright, jupyter/nbclient.

## Global Constraints

- **Interpreter:** ALL commands `/Users/jmanning/hypertools/.venv/bin/python`. Never bare `python`/`pip`/`pytest`. `MPLBACKEND=Agg` for headless plotting.
- **Branch** `dev-2.0-refactor`; never push master. Commit trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Don't push.
- **New API (what every migration targets):** `fig = hyp.plot(...)`; animated matplotlib `fig, ani = hyp.plot(..., animate=True)`; plotly `fig = hyp.plot(..., backend='plotly')` (frames embedded); `ax = fig.axes[0]`; `data = hyp.load(name)` returns raw arrays/DataFrame (NOT a geo, no `.get_data()`); `data = hyp.load(name, reduce=..., ndims=N)` returns analyzed data directly; transformed data via `hyp.plot(..., return_model=True)['xform_data']` or a direct `hyp.reduce(...)`. There is NO public `DataGeometry`, no `geo.plot()` replay, no `geo.line_ani`.
- **No mocks / real calls:** doc build executes real examples; Playwright drives the real built HTML. Per Jeremy's testing rules, no stubs.
- **Keep the 318-green suite green:** doc migration must not touch library code; if an example reveals a library bug, file it / fix the library separately (not silently in a doc commit).

## File Structure

- `examples/*.py` (21 to migrate + 2 content rewrites) — **Modify.** (Tasks 1–2)
- `docs/tutorials/*.ipynb` (13) — **Modify + re-execute.** (Task 3)
- `docs/auto_examples/*.py.md5` (39) — **Delete before build.** (Task 4)
- `scripts/verify_docs_playwright.py` — **NEW.** (Task 5)
- `docs/images/v2.0-docs/` — **NEW** evidence dir. (Task 5)

---

### Task 1: migrate the 21 gallery example scripts (geo → figure API)

**Files:** the 21 `examples/*.py` in research §2's table EXCEPT `plot_geo.py` (Task 2). Test: the doc-build in Task 4 is the integration gate; per-file, verify each runs standalone.

- [ ] **Step 1:** For each of the 20 non-`plot_geo` files in §2 (plot_dataframe, animate, plot_labels, plot_align, save_image, save_movie, plot_2D, animate_MDS, plot_describe, plot_legend, precog, plot_basic, plot_procrustes, animate_spin, chemtrails, plot_clusters3, explore, plot_clusters, plot_hue, analyze): apply the mechanical migration:
  - `geo = hyp.load(NAME)` → `data = hyp.load(NAME)` (raw). `geo.get_data()` → just `data` (drop `.get_data()`). `hyp.load(NAME).get_data()` → `hyp.load(NAME)`.
  - `geo = hyp.load(...); geo.plot(**kw)` → `hyp.plot(data, **kw)` (top-level call; there is no `.plot()` method).
  - `data = geo.data` → use the loaded `data` directly.
  - **The 6 animation files** (animate, animate_MDS, animate_spin, chemtrails, precog, save_movie): `ani_geo = geo.plot(animate=True, ...); ani = ani_geo.line_ani` → `fig, ani = hyp.plot(data, animate=True, ...)`. The variable `ani` must remain a `matplotlib.animation.Animation` (sphinx-gallery's `matplotlib_animations` auto-detects it by type — keep it assigned to a local var).
  - `save_image.py` / `save_movie.py`: `geo.plot(..., save_path=p)` → `hyp.plot(data, ..., save_path=p)` (save still works via `save_path`).
- [ ] **Step 2:** Verify each migrated example RUNS standalone (real execution, Agg):
  ```bash
  for f in <each migrated file>; do MPLBACKEND=Agg .venv/bin/python "$f" && echo "OK $f" || echo "FAIL $f"; done
  ```
  Fix until each runs clean (animations may be slow; that's fine). Do NOT leave any example calling `.get_data()`/`.line_ani`/`geo.plot`.
- [ ] **Step 3:** Commit `docs(examples): migrate 20 gallery examples to figure-return / raw-load API`.

---

### Task 2: content-rewrite the two geo-*about* pages

**Files:** `examples/plot_geo.py`, `docs/tutorials/geo.ipynb`. Modify `examples/README.txt`/gallery index if it references the geo example.

- [ ] **Step 1:** `examples/plot_geo.py` is titled/about the `DataGeometry` object, which no longer exists publicly. REWRITE it to explain the 2.0 return model instead: `fig = hyp.plot(...)` returns a Figure; `hyp.plot(..., return_model=True)` returns `{'fig','models','xform_data'}`; `hyp.load(...)` returns raw data. Retitle (e.g. "Working with plot outputs (figures & fitted models)"). Keep it a runnable gallery example. (If a cleaner move is to DELETE it and add a short section elsewhere, do that + update the gallery index — decide and note it.)
- [ ] **Step 2:** `docs/tutorials/geo.ipynb` — same: repurpose to a "plot outputs / return_model" tutorial, or retire it and remove its `tutorials.rst` entry. Re-execute (Task 3 covers execution mechanics).
- [ ] **Step 3:** Commit `docs: repurpose geo example+tutorial to the 2.0 return model (DataGeometry removed)`.

---

### Task 3: migrate + re-execute the 13 tutorial notebooks

**Files:** `docs/tutorials/*.ipynb` (align, analyze, cluster, geo[from T2], plot, hugging_face_embeddings, reduce, modern_sklearn_dynamics, conversation_trajectories, normalize, text, streaming_data, wikipedia_embeddings).

- [ ] **Step 1:** For each notebook, edit code cells to the new API (same mapping as Task 1: drop `geo`/`.get_data()`/`geo.plot()`/`.line_ani`/`.xform_data`; use `fig = hyp.plot(...)`, raw `hyp.load(...)`). Grep each first: `grep -l 'geo\|DataGeometry\|line_ani\|get_data\|xform_data' docs/tutorials/*.ipynb`.
- [ ] **Step 2:** Re-execute each notebook so committed outputs are correct (`nbsphinx_execute='never'` ships committed outputs):
  ```bash
  MPLBACKEND=Agg .venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace docs/tutorials/<nb>.ipynb
  ```
  NOTE: `hugging_face_embeddings.ipynb` / `wikipedia_embeddings.ipynb` / `text.ipynb` need the `hypertools[text]` (`pydata-wrangler[hf]`) extra — install it into `.venv` if not present (`.venv/bin/pip install "pydata-wrangler[hf]>=0.5.0"`) before executing those; if a notebook needs GPU/large downloads that are infeasible here, note it and execute the others (do NOT fake outputs).
- [ ] **Step 3:** Re-inject Colab cells: `.venv/bin/python scripts/add_colab_install_cell.py`. Commit `docs(tutorials): migrate + re-execute 13 tutorials to 2.0 API`.

---

### Task 4: rebuild the docs (delete .md5, make html, verify artifacts)

**Files:** delete `docs/auto_examples/*.py.md5` (39); build output under `docs/_build/`.

- [ ] **Step 1:** Confirm build deps: `ffmpeg -version`, `.venv/bin/python -c "import kaleido, sphinx_gallery, nbsphinx, furo"`. Install any missing doc deps: `.venv/bin/pip install -r docs/doc_requirements.txt`.
- [ ] **Step 2:** Force re-execution: `rm -f docs/auto_examples/*.py.md5`.
- [ ] **Step 3:** Build: `cd docs && make clean && make html` (this executes ALL 37 examples + renders animations — slow, 15–40 min; run in background, capture log). The build MUST finish with no sphinx errors and no example execution failures. Fix any example that fails during the real build (they were spot-checked in Task 1 but the gallery build is the real gate).
- [ ] **Step 4:** Verify artifacts (spec §14 — don't trust build success): `ffprobe` every generated mp4 under `docs/_build/html/_downloads/`/`docs/auto_examples/images/` to confirm valid video; confirm gallery thumbnails + the animated GIFs exist. Report counts.
- [ ] **Step 5:** Commit the regenerated `docs/auto_examples/` + deleted `.md5`s: `docs: rebuild gallery under 2.0 API`.

---

### Task 5: Playwright visual verification + PR evidence

**Files:** `scripts/verify_docs_playwright.py` (new), `docs/images/v2.0-docs/` (new evidence dir).

- [ ] **Step 1:** Install Playwright into `.venv`: `.venv/bin/pip install playwright && .venv/bin/python -m playwright install chromium`.
- [ ] **Step 2:** Write `scripts/verify_docs_playwright.py`: serve `docs/_build/html` (`python -m http.server`), drive it with Playwright (chromium) to screenshot the gallery index + a representative set of example pages + tutorial pages, and ASSERT: plots/images render (non-blank), embedded animations (`<video>`/`<img gif>`) are present, and the "Open in Colab" badge + branch-aware install URL (`@dev-2.0-refactor`) are present. Save screenshots to `docs/images/v2.0-docs/`. Real browser drive, no stub.
- [ ] **Step 3:** Run it: `.venv/bin/python scripts/verify_docs_playwright.py` → all assertions pass; screenshots saved. Visually spot-check a few PNGs (Read them) to confirm real rendering.
- [ ] **Step 4:** Assemble PR evidence in a notes/markdown file: test counts (before 129/239 → after 318), the data-wrangler issues filed (dw#30 resolved in 0.5.0), the dep upgrade (dw 0.5/pandas 3), the geo-removal summary, screenshot references, and the doc-build/Playwright results. Commit `docs: Playwright visual verify + PR evidence; Plan 8 complete`.
- [ ] **Step 5:** Final full-suite + import-time check: `MPLBACKEND=Agg .venv/bin/python -m pytest -q` (318 green) and measure `import hypertools` time (roadmap <1s target — report actual).

---

## Self-Review

**1. Spec coverage (§14):** gallery examples migrated (T1–T2), tutorial notebooks migrated+re-executed (T3), `.py.md5` deletion + rebuild + `ffprobe` artifacts (T4), Playwright visual verify + Colab badges + PR evidence (T5). The `geo`-*about* pages get real content rewrites, not just API swaps (T2).

**2. Placeholder scan:** each task lists exact files (via research §2/§4) + commands; the two content rewrites and the HF-extra notebooks are called out as judgment points, not hidden.

**3. Consistency:** the geo→figure mapping is identical across T1/T3 (`fig = hyp.plot`, raw `load`, `fig.axes[0]`, `(fig, ani)`); matches the library's actual Task-4/Task-7 return contract.

## Execution Handoff

After Plan 8: the **whole-branch review** (opus, via superpowers:requesting-code-review over `git merge-base dev-2.0 HEAD..HEAD`) covering all ~50 commits, address findings, then **open the PR into `dev-2.0`** (never master) with the evidence from T5. The name-collision (deferred) and any Minor findings tracked in the ledger get triaged in the review. Lift complete.
