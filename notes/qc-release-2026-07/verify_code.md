All four items reproduced successfully. I'm in read-only mode and cannot create the `verify_code.md` file (no write/redirect access), so I'm reporting the full findings here.

## Verification findings (branch dev-1.0-refactor)

**P0-1 — PASS.** `hyp.normalize(...return_model=True).transform(bare 2-D array)`:
- across/within/row all printed `(10, 4)`, no exception.
- `across` reuse applies FIT-TIME mean/std: `np.allclose(out, (new - m.mean_)/m.std_)` → `True` (mean_/std_ are the 50-row fit stats).
- Regression tests present & hardened: `tests/test_normalize.py::test_normalizer_reuse_across_bare_array` (asserts shape + fit-time-stat allclose), `_within_and_row_bare_array`, `_list_returns_list`.

**P0-2 — PASS.** `hyp.manip(x, model={...})`: kwargs-only, args-only, model-only, and legacy `params` dicts all returned `(30,5)`. reduce kwargs-only dict → `(30,2)`, cluster kwargs-only → `(30,)` (note: kwargs are `reduce=`/`cluster=`, not `model=`). Test present: `tests/test_manip_chaining.py::test_manip_single_dict_spec_optional_args_kwargs` (parametrized 4 spec shapes) plus `test_manip_kwargs_only_dict_applies_kwargs` (asserts kwargs actually reach constructor).

**P1-1 — PASS.** ZScore & Normalize round-trip `allclose True`; `Pipeline(['ZScore','PCA']).inverse_transform` recovers original `True`; Smooth & Resample raise `NotImplementedError('... is not invertible ...')`. Tests present: `test_manip_chaining.py` (zscore/normalize round-trip, pipeline, `test_lossy_manipulators_not_invertible` match="not invertible"). Old test in `tests/test_pipeline.py` was UPDATED not deleted: `test_inverse_transform_succeeds_through_invertible_zscore` (comment notes "it used to raise") + new `test_inverse_transform_raises_through_lossy_step` for a lossy step.

**P1-2 — PASS.** Default-semantic `hyp.plot(docs, vectorizer='Word2Vec')` returned a `matplotlib.figure.Figure`, no crash, and warned: "...skipping the semantic stage and returning the embeddings directly. Pass semantic=None to silence this warning." `semantic=None` → Figure, no skip-warning. CountVectorizer+LDA → Figure (LDA runs). Tests present in `tests/test_gensim_text.py`: `test_word2vec_default_semantic_skips_topic_model_with_warning` (pytest.warns match="skipping the semantic stage"), `test_word2vec_explicit_semantic_none_no_skip_warning`, `test_plot_lda_semantic_end_to_end`.

---
PASS P0-1 (across reuses fit-time mean/std). PASS P0-2 (reduce/cluster use reduce=/cluster= kwarg, not model=). PASS P1-1 (old pipeline test updated, not deleted). PASS P1-2 (warns + returns Figure). All 4 verified with real calls; hardened regression tests exist for each. No file written — read-only mode blocks creating verify_code.md.

All four bug fixes verified via real calls with the project venv (MPLBACKEND=Agg).

PASS P0-1 — normalize reuse: across/within/row all →(10,4); across reuses FIT-TIME mean/std (allclose True). Hardened tests in test_normalize.py.
PASS P0-2 — manip kwargs/args/model/legacy-params dicts all work; reduce/cluster kwargs-only work (via reduce=/cluster=). Tests in test_manip_chaining.py.
PASS P1-1 — ZScore/Normalize round-trip + Pipeline recover original; Smooth/Resample raise NotImplementedError('not invertible'). Old pipeline test UPDATED not deleted.
PASS P1-2 — Word2Vec default warns "skipping the semantic stage" + returns Figure; semantic=None no warn; Count+LDA runs. Tests in test_gensim_text.py.

Note: read-only mode blocked creating verify_code.md; full findings are in my message above.

All four items verified with real calls against `/Users/jmanning/hypertools/.venv/bin/python` (MPLBACKEND=Agg), branch dev-1.0-refactor.

PASS P0-1 · PASS P0-2 · PASS P1-1 · PASS P1-2 — no failures.

- P0-1: across/within/row all →(10,4); across reuses fit-time mean/std (allclose True). Hardened tests in tests/test_normalize.py.
- P0-2: manip kwargs/args/model/legacy-params dicts work; reduce/cluster kwargs-only work (kwargs are reduce=/cluster=). tests/test_manip_chaining.py.
- P1-1: ZScore/Normalize + Pipeline round-trip recover original; Smooth/Resample raise NotImplementedError('not invertible'); old test_pipeline.py test UPDATED not deleted.
- P1-2: Word2Vec default warns "skipping the semantic stage" + returns Figure; semantic=None no warn; Count+LDA runs. tests/test_gensim_text.py.

Read-only mode blocked creating verify_code.md; full evidence is in my findings message above.

Verification complete. All four bug fixes PASS.

- P0-1 (normalize reuse): PASS
- P0-2 (manip single dict spec): PASS
- P1-1 (ZScore/Normalize invertible): PASS
- P1-2 (Word2Vec semantic skip): PASS

Each was reproduced with real calls and each has a hardened regression test. Full evidence and test locations are in my findings message above. No failures. (Note: read-only mode prevented writing verify_code.md, so findings were reported inline.)

All four fixes verified: PASS P0-1, PASS P0-2, PASS P1-1, PASS P1-2. No failures. Full evidence and regression-test locations are in my findings above.