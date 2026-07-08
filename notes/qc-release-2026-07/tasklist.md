These are the items the audit surfaced. Add your own as you work the checklist and notebook, then send the exported notes back and I'll fix them. The **static** commit audit came back clean (no high-risk logic in any of the 185 commits), but **running the verification notebook against the built library surfaced four real code issues** (P0/P1 below) — exactly what this pass was for. Everything else is documentation accuracy.

### P0 · Confirmed code bugs (found by running the verification notebook)

**P0-1 · `hyp.normalize(..., return_model=True).transform(new_data)` crashes.** `hypertools/tools/normalize.py:94` (`Normalizer.transform`) raises `IndexError: tuple index out of range` for **all three** modes (`across`/`within`/`row`) — it indexes `i.shape[1]` on what is a 1-D row. The entire normalize return-model *reuse* path (part of the #227 "return_model on every module" contract) is broken. It slipped through because Task 6's test likely exercised a shape/list form that dodges this line. Fix `Normalizer.transform` to apply the stored fit-time statistics to new data correctly, and add a real 2-D-array reuse test.

**P0-2 · `hyp.manip` rejects the `kwargs`-only dict spec.** `hyp.manip(x, model={'model':'Smooth','kwargs':{...}})` raises `ValueError: unknown model`, but `{'model':'Smooth','args':[],'kwargs':{...}}` (with an explicit `args`) and the list form both work. A canonical dict spec should accept `kwargs` without requiring an `args` key — this is an inconsistency with `reduce`/`cluster` (which check `'args' in x or 'kwargs' in x`). Fix manip's single-spec dict resolution to treat `args` as optional.

### P1 · Rough edges (found by running the notebook)

**P1-1 · `Pipeline.inverse_transform` can't pass through `ZScore`/`Normalize`.** `Pipeline(['ZScore','PCA']).inverse_transform(...)` raises `NotImplementedError` because `ZScore` has no `inverse_transform` — even though it stores the mean/std needed to invert. Consider giving the invertible manipulators (`ZScore`, `Normalize`) an `inverse_transform` so pipeline round-trips work through them.

**P1-2 · `vectorizer='Word2Vec'` breaks with the default `semantic`.** `hyp.plot(docs, vectorizer='Word2Vec')` (no `semantic=`) raises `ValueError: Negative values in data passed to LatentDirichletAllocation` — the default semantic model can't consume the negative embeddings that Word2Vec/FastText produce. Either default `semantic=None` when an embedding vectorizer is selected, or raise a clear, actionable error. (README/notebook examples correctly pass `semantic=None`.)



### A · Documentation accuracy — recommended before release

**A1 · CLAUDE.md is stale (highest leverage — AI tools read it every session).** 7 wrong statements: (1) tests run from repo root / `tests/`, not `hypertools/`; (2) `[dev]` does **not** install all deps (omits `text`, `predict-hf`); (3) DataGeometry is now an internal unpickle-only shell, not the central container; (4) Main-API list omits `manip`/`predict`/`impute`/`save`/`apply_model`/`Pipeline`/`set_interactive_backend`/`io`; (5) "Tools Module" list is wrong — `reduce`/`cluster`/`align`/`load` moved to their own subpackages; new `reduce/ cluster/ align/ manip/ io/ predict/ impute/ core/` unmentioned; (6) `plot/draw.py` is now a shim for `matplotlib_backend.py`; new plot modules unlisted; (7) "Python 3.9+" → floor is 3.10.

**A2 · API docstring inaccuracies (9).** `align` documented as default `'hyper'` in `plot`/`analyze`/`load` but the real default is `None` — the worst one (implies data is hyperaligned by default when it isn't). Also: `plot` `ndims` says "None→3" (is `3`); `zoom` says default 0 (is 1); `normalize` doc self-contradictory; `plot` `cluster` example uses `reduce=`+deprecated `'params'`; `plot` `return_model` Returns bundle omits the `'pipeline'` key; `set_interactive_backend` example calls `geo.plot()` which no longer exists.

**A3 · API docstring incompleteness (6).** `reduce`/`cluster` model-name lists in `plot`/`analyze`/`load`/`describe` omit UMAP, the mixture reducers, and the six autoencoders; `apply_model` documents only the deprecated `{'model','params'}` dict form (not the canonical `args`/`kwargs`).

**A4 · README.** One broken import path: `hypertools.tools.colors.mat2colors` → `hypertools.plot.colors.mat2colors` (line 52). "What's new in 1.0" omits several shipped features worth showcasing: `hyp.Pipeline`, manip chaining, autoencoders (`[torch]`), gensim (`[gensim]`), LSL (`[lsl]`), `predict`/`impute`, and the `window`/2-D animation modes; Requirements names only 2 of 11 extras.

**A5 · Sphinx tutorials (2 build-affecting).** `geo.ipynb` (retired DataGeometry) is still published via `tutorials.rst`; 8 stale 0.x `.rst` tutorials duplicate the new executed `.ipynb`, producing 8 "multiple files found" warnings and risking that Sphinx publishes the retired-API `.rst` instead of the current notebook. Fix: delete the superseded `docs/tutorials/{align,analyze,cluster,geo,normalize,plot,reduce,text}.rst` (+ their `*_files/`) and drop the geo section. Also: `docs/index.rst` title underline too short (1 cosmetic warning).

### B · Code items to confirm (not defects — intentional 1.0 changes worth a second look)

**B1 · SRM alignment semantics changed** (commit `22cab2b49c`). The classic `align='SRM'` path no longer does `n_iter` repeated re-fits (single fit via the new SRM class), dropped two legacy warnings (len-1 list, features>samples), and `align=True` now raises. Confirm SRM alignment quality is still what you expect on real data.

**B2 · pandas 3.0 now permitted** (commit `60b598f294`; `pandas<3` pin lifted, `>=2.2`). A CI acceptance gate pins pandas 3.0 on one job, but the runtime surface is wider than 0.x. Confirm you're comfortable supporting pandas 3.

### C · Optional / housekeeping

**C1 · Add `AGENTS.md`** (absent) — the emerging cross-tool standard; mirror the corrected CLAUDE.md so non-Claude agents get the same guidance.
**C2 · `CONTRIBUTING.md` is stale** — points at a mozsprint milestone and a defunct Gitter channel; no mention of the `pip install -e ".[dev]"` / `pytest` workflow.
**C3 · Cosmetic sphinx warnings** — scipy intersphinx 404 (double slash) and the sphinx-gallery pickle-cache warning; both harmless, optional to silence.
**C4 · `pipeline_order.rst`** calls the story-trajectories walkthrough a "tutorial" but it's a gallery example — reword the cross-reference.
