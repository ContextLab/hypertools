---
name: readme-is-lowercase
description: Use when a hypertools test or script opens the repository readme by path.
created: 2026-10-03
origin: notes kept in CLAUDE.md
---
The repository's readme is lowercase `readme.md`. A test that opens 'README.md' passes on
macOS (case-insensitive) and fails on every Linux CI job. Check the tracked file name with
`git ls-files` before hard-coding a path in a test.
