---
name: datatype-gate-no-raw-isinstance
description: Use when writing or delegating hypertools library code that checks whether an input is a DataFrame, Series or array.
created: 2026-10-09
origin: project hypertools, session f69d921d
---
Classify inputs with the shared helpers in hypertools/_shared/helpers.py
(is_frame_dataset, is_series_like, is_array_dataset, as_pandas_dataframe), never with
`isinstance(x, pd.DataFrame)` or `isinstance(x, pd.Series)`. tests/test_datatype_gate.py
scans the package and fails with "datatype check(s) outside the shared coercion layer"
for each raw check, so polars and other frames are treated like pandas ones.
An agent's targeted test run usually does not include that file, and the failure then
shows up only in the full suite: put `tests/test_datatype_gate.py` in the test list of
any prompt that adds input handling. To tell a row-labelled pandas Series from a polars
one, use `is_series_like(x)` plus a non-callable `x.index`.
