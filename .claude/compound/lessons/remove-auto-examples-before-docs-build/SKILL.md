---
name: remove-auto-examples-before-docs-build
description: Use when building hypertools docs with sphinx -W after deleting or renaming a gallery example.
created: 2026-10-03
origin: notes kept in CLAUDE.md
---
A "clean" docs build (sphinx -W -E -a) does NOT remove sphinx-gallery's generated
docs/auto_examples/ (gitignored). After deleting or renaming an example, `rm -rf
docs/auto_examples` first, or -W fails on "document isn't included in any toctree" for the
stale pages.
