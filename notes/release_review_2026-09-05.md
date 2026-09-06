# HyperTools 1.1 release review — 2026-09-05

Reviewed draft source: `96ac8b7f43c132f6f455ad1be3ffc98e84adead5` (master and v1.1.0 at review start). Changes are submitted on `fix/1.1-release-review`; master and the release tag are not modified.

## Findings fixed in the PR

| Priority | Finding | Fix and evidence |
| --- | --- | --- |
| High | Forecast backtests reused an instance fitted on dataset 1 for dataset 2. On a sine series followed by a positive quadratic series, `AutoRegressor()` produced negative quadratic forecasts and MAE 2551.68, versus 0.00246 when passing the class. | Deep-copy the model for each dataset; preserve caller state. Regression tests compare actual returned forecasts for class, instance, and dictionary forms. |
| High | Scoring accepted previously fitted forecasters/imputers even when their learned state could contain the held-out values. Imputation scoring also fitted caller-owned instances. | Require unfitted instances for scoring, copy them before fitting, and document the distinction from ordinary fitted-model replay. Real fitted/unfitted model tests cover both forms. |
| Medium | `Delay` silently lost features when distinct pandas column labels had identical string representations (`1` and `'1'`); a 2-column, 2-lag input produced only 2 output columns. | Reject colliding labels with a renaming instruction; tests include mixed-type and duplicate labels. |
| Medium | URL-cache temporary names used only the process ID. Threads caching the same URL collided: 82 of 100 concurrent writes failed with `FileNotFoundError`. | Use a unique temporary file per write, clean it on error, and atomically replace the destination. Test 100 real writes across 12 threads and verify payload/metadata and cleanup. |
| Medium | System-installed Noto Sans took precedence over the bundled Regular face, contradicting deterministic font selection and failing the existing font regression test on this machine. | Register bundled faces ahead of equal-scoring system faces. A fresh interpreter with another real same-family font proves the bundled file wins. |
| Medium | Optional dependency minimums allowed gensim 4.3 and scikit-image 0.22, predating NumPy 2 support despite the library requiring NumPy>=2. | Raise floors to gensim>=4.4.0 and scikit-image>=0.23.2 in extras/dev/docs. Real minimum-version feature tests pass under NumPy 2.3.5. |
| Documentation | Public plotting/predict/impute docstrings described shipped features as 1.2; dependency prose implied ARIMA imputation. | Correct version labels and separate forecasting from imputation support. |
| Documentation | The “convert now” forecast example still hand-wrote URL download/cache logic after the native cache landed. | Use `hyp.load(ARCHIVE, cache=True)`, regenerate and execute the tutorial. Its committed video remains byte-identical. |
| Tooling | The browser verifier expected `docs-notebooks/master`, searched highlighted HTML for contiguous `pip install`, and demanded an autoplay call in deliberately paused Plotly animations. | Validate versioned notebook links, rendered code text, loaded frames/play controls, and execute a real transition in Chromium. Allow evidence/build paths outside the checkout. |

## Source and regression-test map

- Forecast/imputation ownership: `hypertools/predict/backtest.py`, `hypertools/impute/backtest.py`; `tests/test_predict_backtest.py`, `tests/test_impute_backtest.py`.
- Cache atomicity: `hypertools/io/sources.py`; `tests/test_load_url_cache.py`.
- Delay collisions: `hypertools/manip/delay.py`; `tests/test_manip_delay.py`.
- Font precedence: `hypertools/plot/fonts.py`; `tests/test_fonts_bold.py`.
- Dependency compatibility: `pyproject.toml`, `docs/doc_requirements.txt`; real minimum-version runs of `tests/test_gensim_text.py` and `tests/test_density.py`, plus packaging/optional-import checks.
- Documentation: dispatcher docstrings, `docs/optional_dependencies.rst`, `readme.md`, `CHANGELOG.md`, `examples/animate_forecast.py`, and its executed tutorial notebook.
- Browser verification: `scripts/verify_docs_playwright.py`.

## Review coverage and validation

- Reviewed #284 and #285 bodies against code, tests, tutorial/example sources, API documentation, and release evidence. The implemented API choices include `Smooth(center=False)` (instead of the proposed conflicting `align=`), `alignment_score`, and matplotlib-only `companion=`; broader animated panels and launch-example visual rewrites remain the explicitly deferred scope.
- Combined behavioral suite: **4885 passed, 19 skipped, 2 deselected** in 15m45s, with the local LSL configuration below. The subsequent optional-floor changes affect metadata only and were checked separately against the actual minimum packages and rebuilt metadata.
- Focused behavioral/docstring checks: **94 passed**. Native-example/tutorial gates: **232 passed, 8 skipped**.
- An unrestricted earlier full run: **4862 passed, 19 skipped, 18 failed**. One failure was the bundled-font bug fixed above (that run had already imported the old code). The other 17 were LSL discovery on the host network. With isolated loopback discovery, all **63 LSL/audit tests passed, 1 skipped**. No LSL tests or assertions were weakened.
- The LSL run used a temporary `LSLAPICFG` with a private SessionID, `KnownPeers = {127.0.0.1}`, and machine-scoped discovery. These are the upstream-supported [LSL configuration settings](https://labstreaminglayer.readthedocs.io/info/lslapicfg.html); no user/global network settings were changed.
- A clean source-copy Sphinx build with `-W --keep-going` executed **51/51 gallery examples** and passed. Rebuilt again after updating the forecast example/tutorial. Post-build processing injected versioned Colab links and updated all 51 thumbnails.
- **166 HTML pages**, no missing internal file/anchor targets in the final post-processed build; **25 tutorial notebooks**, no saved error outputs. All **51 generated notebooks** passed the release-install checker.
- Real Chromium checks passed on **8 representative pages**: gallery, static plots, matplotlib videos, live Plotly animation, plot tutorial, and alignment tutorial. Screenshots were inspected for layout and nonblank plots.
- The revised forecast tutorial was freshly executed; launch clips and the forecast video are unchanged in the PR. The remaining tutorials were reviewed through stored outputs and gates, not all re-executed during this review.
- Release/packaging/native checks: **256 passed, 6 skipped** in the initial source checkout; the network-blocked manifest check was rerun unrestricted, yielding **10/10 release-readiness checks passed**.
- Optional minimums: gensim **4.4.0** and scikit-image **0.23.2** were installed into a temporary target directory, leaving the main environment dependencies unchanged. Real Word2Vec training and marching-cubes generation succeeded under NumPy **2.3.5**; the corresponding feature suites passed **79 tests, 1 skipped**. Upstream evidence: [gensim 4.4.0 adds NumPy 2 support](https://github.com/piskvorky/gensim/releases/tag/4.4.0), and [scikit-image 0.23 release notes describe NumPy 2 compatibility/builds](https://scikit-image.org/docs/stable/release_notes/release_0.23.html). The older floors could retain incompatible binary builds; latest-version CI did not test this boundary.
- Rebuilt metadata and optional-import tests: **20 passed** after refreshing the editable metadata without changing installed dependencies.
- Ruff and `git diff --check` passed. Existing matrix CI and tag CI on `96ac8b7f` were green when inspected. PR CI validates the new branch separately.
- Draft wheel and sdist package contents were compared byte-for-byte with the original release commit. All **110 package files** in each artifact match that commit; they do not contain this PR's fixes yet.

## Issues and public-release disposition

The PR resolves the remaining implementation/documentation defects discovered while checking #284 and #285. Both issues are linked for closure on merge, rather than closed before fixes reach master.

Before public release, merge the PR and perform the normal release re-cut from the resulting commit: rebuild and republish the gallery/notebook manifest, rebuild wheel/sdist, update the draft assets/tag, and confirm the release/tag gates. The manifest intentionally pins an exact source commit. The already-green draft at `96ac8b7f` cannot stand in for these checks after the fixes merge.

Large-download tests remain excluded by the project's default `not bigdata` marker. A passing test suite is not a guarantee that every third-party service, model, or platform combination is defect-free.

Detailed local logs and browser evidence are under `/tmp/hypertools-review/` (not shipped in the package).
