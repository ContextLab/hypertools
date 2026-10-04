---
name: yahoo-sec-api-traps
description: Use when fetching price history from the Yahoo v8 chart API or filings data from SEC XBRL.
created: 2026-10-03
origin: notes kept in CLAUDE.md
---
Yahoo v8 chart with `range=max&interval=1d` silently returns 3-MONTH bars (AAPL: 169 rows
since 1984); pass explicit period1/period2 epoch bounds to get daily data. SEC XBRL needs
a User-Agent with a contact (the project's pyproject email). `companyconcept` can be EMPTY
for a filer (ABT, KO) whose `companyfacts` has the concept, so fall back to companyfacts.
