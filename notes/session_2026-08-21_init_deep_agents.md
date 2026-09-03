# Session: /init-deep AGENTS.md generation (2026-08-21)

Generated hierarchical AGENTS.md: root + 10 subdirs (plot, core, tools, reduce, align, manip, io, predict, impute, tests).

## Decisions
- Skipped: cluster (913 lines, root covers), external (vendored), _externals/_shared (shims), docs/examples/scripts/notes (root covers).
- Explore agents (9 fired) all stalled 30-43 min on this repo's 71k files (docs/_build + __pycache__ + build/lib); 4 queued ones timed out. Cancelled the 5 runners and gathered all facts directly (bash + codegraph + pyproject read). Root AGENTS.md NOTES now warns about this.
- codegraph index exists and works well here (blast-radius caller counts). No Python LSP installed (ruff only).

## Key facts captured (verify against code if re-generating)
- plot.py is 8254 lines; plot/ = 22k of 37k package lines.
- Two align() functions: tools/align.py (classic wrapper) vs align/align.py (registry).
- core/hierarchy.py vs plot/hierarchy.py: grouping logic vs trace construction; predict must never import from plot.
- impute/ppca.py wraps external/ppca.py with splice-preservation contract.
- pyproject.toml is heavily annotated with audit IDs — dep floors are numpy-2-driven, extras policy is "never in base install".
- Tests: 0 mock imports confirmed; importorskip + subprocess import-blocker pattern for extras; bigdata marker; timeout=1200 thread.

## If context lost
All 11 files are written and verified. Remaining: none. Commit if user asks.
