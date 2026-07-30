# Temperatures dataset + MultiIndex design (verified 2026-07-26)

Source the maintainer pointed at:
https://github.com/ContextLab/hypertools-paper-notebooks/blob/master/temperatures.ipynb

## The data (fetched, HTTP 200)
- `data/temperatures.csv` -- 1965 rows x 43 cols. `Year`, `Month`, and for each of 20 cities both
  `<City>` (absolute degC) and `<City>_anomaly`. Range 1850-2013; **1645 complete rows after
  dropna, covering 1875-2013**.
- `data/temperature_locs.csv` -- 20 rows: `City`, `Lat`, `Long`. Hemisphere is derivable from
  `Lat` sign with no extra data.
  - **Northern: 16** (Bangkok, Bombay, Cairo, Chicago, Istanbul, London, Los_Angeles, Mexico,
    Montreal, Moscow, New_York, Rome, Seoul, Shanghai, Somalia, Tokyo)
  - **Southern: 4** (Cape_Town, Santiago, Sao_Paulo, Sydney)
- Note the imbalance: 16 vs 4. The current demo uses a deliberately balanced 6/6. Any rewrite has
  to decide whether to keep all 20 (honest, unbalanced) or subset to a balanced panel.

## What the original paper notebook actually did
`hyp.plot(temps, group=years.flatten(), palette='RdBu_r', normalize='across')` where
`temps` is (n_months, n_cities) -- **observations are months, features are cities**, one
trajectory through "city space" colored by year. `group=` is the pre-1.0 spelling of `hue=`.
Everything else in that notebook is seaborn regplots, not hypertools.

## The MultiIndex request is currently BLOCKED by a real limitation
Built the requested structure from the real data -- rows indexed by `(Hemisphere, City)`,
features `[temp, sin(month), cos(month), date]`, 32900 rows -- and plotted it:

| call | line artists | distinct colors |
|-|-|-|
| `hyp.plot(mi)` | 22 | **2** |
| `hyp.plot(mi, hue=<continuous temp>)` | 22 | **2** |

So color collapses to the OUTER index level (hemisphere) and a continuous `hue=` is silently
ignored. This is exactly the limitation `examples/animate_weather_decades.py` cites as its reason
for passing a plain list of loops instead of a MultiIndex. GH #95 ("support multiindex Pandas
DataFrames") is CLOSED, so the feature exists -- what is missing is hue/grouping fidelity inside it.

## Design gap this exposes
The maintainer's "hemisphere outer, city inner" implies **hierarchical grouping**:
- outer level -> the color scale / legend group (per-hemisphere colormap)
- inner level -> the individual drawn trajectory (one line per city)
- and a continuous `hue=` must survive both

hypertools today offers only single-level grouping by the outer index, with categorical color.
Supporting the requested design means teaching MultiIndex expansion to (a) split lines at the
inner level, (b) group color/legend at the outer level, and (c) respect a continuous hue.

## Open question for the maintainer
Where does time live? Options: rows = (hemisphere, city) with time as columns (one point per
city, no trajectory); or repeated (hemisphere, city) rows ordered by time (what was tested here,
giving per-city trajectories). The second is the only one that reproduces the current animation.
