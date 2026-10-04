---
name: windows-os-replace-retry
description: Use when writing a cache or atomic-write rename with os.replace in hypertools, or reading a PermissionError from Windows CI.
created: 2026-10-03
origin: notes kept in CLAUDE.md
---
os.replace onto a file other threads are reading or renaming raises PermissionError
(WinError 5) on Windows only. Wrap cache and atomic-write renames in a short retry that
accepts an existing identical destination, as _replace_retrying in
hypertools/io/sources.py does. Expect Windows CI to be the only place such a test fails.
