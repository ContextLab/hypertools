<!-- skillnote:begin -->
## Notes (skill-compounder)

- **2026-09-03** Executing a docs/tutorials launch notebook locally runs its Colab '%pip install ... git+...@dev-1.0' cell, which overwrites the venv's editable hypertools with the stale REMOTE branch mid-run (seen 2026-09-03 as '48 dimensions ... static plots support at most 2'). scripts/execute_tutorial.py now tags 'pip install' cells skip-execution in memory; after any other notebook run, check 'pip show hypertools' says Editable, and 'pip install -e .[dev]' if not. <!-- id:n4007036607x464 -->
<!-- skillnote:end -->
