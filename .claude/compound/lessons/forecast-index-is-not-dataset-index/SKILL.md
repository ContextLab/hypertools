---
name: forecast-index-is-not-dataset-index
description: Use when writing hypertools code that indexes schedules, lines or runs by a forecast.
created: 2026-10-03
origin: notes kept in CLAUDE.md
---
Forecast index != dataset index. A predict=[...] collection's forecasts are MODEL-MAJOR
(forecast i = model i//n_datasets, dataset i%n_datasets), and hue=/cluster= regrouping
makes RUN index != dataset index. Any code that touches a forecast must translate through
_model_forecast_owner / forecast_datasets (forecast -> source dataset) and _forecast_owner
/ _seg_ds (dataset -> final run) before indexing. The bug hides when the counts coincide
and shows as an IndexError in animated reveal lookups or wrong plotly hyp_dataset tags.
