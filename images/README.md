# `images/` — asset manifest

Every file here is either **currently served** by `README.md` or kept deliberately as a
**compatibility asset**. Nothing in this directory is unused, and nothing should be deleted on the
grounds that it is "unreferenced in the repository" — several files are referenced only from places
git cannot see.

## Why this file exists

`README.md` serves these images over absolute
`https://raw.githubusercontent.com/ContextLab/hypertools/<ref>/images/...` URLs, and third parties
do the same. A file with no in-repo reference may still be load-bearing for a GitHub pull-request
body, a release note, an issue comment, or an external post. Deleting one breaks those links
**silently** — nothing in CI would fail, and we would never see the 404.

Deleting does not reduce existing git history — every blob is retained permanently — so it only
shrinks future working trees, checkouts and source archives, while risking broken branch-based
hot-links. For the files below that trade is not worth taking.

## Currently served by `README.md`

| file | used for |
|-|-|
| `story_trajectories.gif` | the welcome/hero animation (also the Read the Docs landing page, `docs/index.rst`) |
| `plot.gif` | the `hyp.plot` example |
| `align_before.gif`, `align_after.gif` | the alignment before/after pair |
| `cluster_example.png` | the clustering example |
| `describe_example.png` | the `describe` example |
| `surface_example.png` | the surface example |
| `hypercube.png` | the project logo |

## Compatibility assets — keep, do not delete

| file | provenance | why it stays |
|-|-|-|
| `demo_density.png` | added 2026-07-18, `437ee022` | One of four "1.0 feature-demonstration figures for the release PR" — density shading (`density=True`) |
| `demo_multicolored.png` | ⟢ same commit | multicolored lines + colorbar (continuous hue) |
| `demo_plotly.png` | ⟢ same commit | the interactive plotly backend (`backend='plotly'`, via kaleido) |
| `demo_predict.png` | ⟢ same commit | trajectory forecasting (`hyp.predict`, Kalman) |
| `hypercube.pdf` | added 2017-01-24, `cd774ebd` | the original **vector source** for `hypercube.png`, the project logo |

The four `demo_*.png` files were generated specifically for embedding in the `dev-1.0` → `master`
release pull request, so their absence from repository prose is **expected, not evidence that they
are dead** — the references live in the PR body and related release material. They total ~196 KB.
`hypercube.pdf` is 804 KB and is the only vector form of the logo we have; regenerating a raster
from it is possible, the reverse is not.

## `hypertools.gif`

Not served by `README.md` or the docs any more — `story_trajectories.gif` replaced it as the
welcome animation — but **kept**, because it is the documented output target of
`scripts/round17_evidence/readme_media.py` and the stated visual reference in
`scripts/generate_story_trajectories.py` and `scripts/generate_weights_trajectory.py`. Deleting it
would dangle those generators.

## Adding an image

Put it here, reference it from `README.md` with an absolute `raw.githubusercontent.com` URL pinned
to the same ref as the other images (a release-gate test asserts every README image shares one ref
and exists in the tree — `tests/test_release_readiness_gate.py`), and add a row above.
