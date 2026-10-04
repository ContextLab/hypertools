---
name: relpath-strings-use-forward-slash
description: Use when a hypertools test compares repo-relative paths as strings (allowlists, rosters, scanner findings).
created: 2026-10-03
origin: notes kept in CLAUDE.md
---
Build such paths with `os.path.relpath(...).replace(os.sep, '/')`. Otherwise the test
passes on macOS and Linux and fails every Windows job on 'docs\tutorials\align.ipynb'
against 'docs/tutorials/align'.
