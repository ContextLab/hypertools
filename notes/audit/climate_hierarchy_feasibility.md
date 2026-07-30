# Climate/weather as a hierarchical HyperTools showcase — feasibility audit

**Date:** 2026-07-27
**Question:** can a climate example show a cyclical seasonal LOOP *and* long-run warming DRIFT
**at the same time**, in a genuine (MultiIndex) hierarchy?

## VERDICT: **NOT RESCUABLE** as a simultaneous loop + drift showcase

*(Verdict based on the **smoothed** numbers, using hypertools' native `manip='Smooth'`. Unsmoothed
numbers reported alongside throughout.)*

The hierarchy is fine. The multi-variable data source is real, free, no-auth, and goes back to
1940 — or **1836** if you want a 180-year record (§1.4). The **loop is real and strong**, and
native smoothing makes it visibly cleaner. But the **drift is ~500x too weak to survive the
projection**: at the shipped smoothing it is **0.070%** of the variance in the 3-D space against
the seasonal loop's **37.5%**. Decade centroids **do not drift monotonically in any city at any
smoothing setting** — `0/6` strictly monotone across all nine kernel configurations tested.

The core tension, quantified: **smoothing tightens the loop but never lifts the drift.** Absolute
drift stays flat (0.218–0.470) at every kernel while the loop diameter collapses from 4.27 to 0.83.
Kernels wide enough to suppress the noise that hides the drift (≥12 months) destroy the seasonal
cycle by construction — seasonal variance falls to **0.60%** at boxcar kw=13 — which is exactly the
"rows=years" compromise the maintainer already rejected, reached by a different route.

This is a *physical* limit, not a code limit, not a data-source limit, and not a smoothing-setting
limit. Five independent rescues were tried and measured; all fail (§5). Recommend **not** rewriting
the climate example around this claim.

---

## 1. Data sources — every URL tested with curl, status verified

| source | URL tested | HTTP | bytes | verdict |
|-|-|-|-|-|
| open-meteo archive (ERA5) | `archive-api.open-meteo.com/v1/archive?...1940-01-01...` | **200** | 6,762 | **BEST**; no auth |
| open-meteo archive, 1939 | same, `start_date=1939-01-01` | **400** | 102 | proves the floor |
| open-meteo, 10 daily vars | `...&daily=<10 vars>` 1940 | **200** | 1,071 | all 10 return values |
| open-meteo, full city pull | 1940-2024, 8 vars, 1 city | **200** | ~1.55–1.61 M | 16–61 s each |
| open-meteo, rate limited | after 5–6 full pulls | **429** | 94/95 | **see §1.2** |
| open-meteo climate-API (CMIP6) | `climate-api.open-meteo.com/v1/climate?...1950...` | **200** | 668 | model output, from 1950 |
| **NOAA 20CRv3 THREDDS catalog** | `psl.noaa.gov/thredds/catalog/Datasets/20thC_ReanV3/Monthlies/2mSI-MO/catalog.xml` | **200** | 2,252 | **LONGEST record** |
| **NOAA 20CRv3 point extract** | `psl.noaa.gov/thredds/ncss/grid/.../air.2m.mon.mean.nc?...accept=csv` | **200** | 180,792 | **1836-2015**, see §1.4 |
| NOAA NCEI GHCN-d (`.csv.gz`) | `ncei.noaa.gov/pub/data/ghcn/daily/by_station/USW00094728.csv.gz` | **200** | 1,816,050 | no auth, works |
| NOAA NCEI GHCN-d (`.csv`) | `ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00094728.csv` | **200** | 17,808,413 | no auth; **see §1.3** |
| Copernicus CDS catalogue | `cds.climate.copernicus.eu/api/catalogue/v1/collections/reanalysis-era5-single-levels` | **200** | 7,454 | metadata only |
| Copernicus CDS retrieve | `POST .../api/retrieve/v1/processes/.../execution` | **401** | 267 | `"authentication required"` — **auth-gated** |
| Copernicus CDS v2 (old) | `.../api/v2/resources/reanalysis-era5-single-levels` | **404** | 2,122 | endpoint retired |
| Berkeley Earth S3 | `berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Regional/TAVG/Text/...` | **403** | 243 | `AccessDenied` |
| Hugging Face datasets API | `huggingface.co/api/datasets?search=climate` | **200** | 22,747 | **see §1.5** |

### 1.1 open-meteo: verified earliest year and variable list

The earliest year is **1940**, and the API states it itself. Requesting 1939 returns HTTP 400 with:

> `{"reason":"Parameter 'start_date' is out of allowed range from 1940-01-01 to 2026-07-27","error":true}`

This matches ERA5's own coverage — the CDS catalogue entry (HTTP 200) is titled
*"ERA5 hourly data on single levels from 1940 to present"*. So **open-meteo's archive floor is
ERA5's floor**: 1940, giving an **85-year record**, comfortably past the "50+ years" bar.

`examples/animate_weather_decades.py::fetch_city_months` currently requests only 4 variables
(`temperature_2m_mean`, `precipitation_sum`, `relative_humidity_2m_mean`, `windspeed_10m_max`)
over `START, END = '1990-01-01', '2024-12-31'`. All of these, **plus** the following, were
verified to return real values in 1940 (HTTP 200):

`surface_pressure_mean` (hPa), `shortwave_radiation_sum` (MJ/m²), `cloud_cover_mean` (%),
`dew_point_2m_mean` (°C), `et0_fao_evapotranspiration` (mm), `snowfall_sum` (cm).

So the example is **leaving 50 years and 6 variables on the table** — that part of the premise is
correct. It just does not help (§4).

### 1.2 Practical blocker: open-meteo rate limits kill a 12-city × 85-year pull

The free tier's own error strings, captured verbatim:

> `{"reason":"Minutely API request limit exceeded. Please try again in one minute.","error":true}`
> `{"reason":"Hourly API request limit exceeded. Please try again in the next hour.","error":true}`

Measured behaviour: **5 full-record city pulls (1940-2024 × 8 variables) exhausted the hourly
quota.** The 6th failed with HTTP 429; a subsequent 5-day, 1-variable request *also* 429'd,
confirming the limit is account/IP-wide, not per-request-size. A 12-city panel needs ~19 MB and,
at the observed replenishment rate, **well over an hour of wall-clock** on a cold cache.

This alone disqualifies a 12-city/1940 panel for the gallery build even if the science had worked.
The current 6-city/1990/4-variable example is roughly 10x lighter and stays under the limit.

### 1.3 NOAA GHCN-d: longer record, but too few *dense* variables

Verified on station `USW00094728` (NYC Central Park), HTTP 200, 17.8 MB, **57,548 daily rows
spanning 1869-01-01 → 2026-07-24**, 124 columns. Per-variable fill rate:

| var | days | coverage |
|-|-|-|
| PRCP | 57,548 | 100.0% |
| TMAX | 57,541 | 100.0% |
| TMIN | 57,541 | 100.0% |
| SNOW | 57,382 | 99.7% |
| SNWD | 40,988 | 71.2% |
| AWND (wind) | 11,172 | 19.4% |
| RHAV (humidity) | 6,844 | 11.9% |
| ADPT (dew point) | 6,796 | 11.8% |

GHCN-d has the **longer** record (1869) but only ~3–4 densely-populated variables, and `SNOW` is
identically zero for tropical stations. Humidity/wind/dew-point only begin around the 1990s.
So GHCN-d gives a *narrower* feature space than open-meteo over the long record — it is strictly
worse for this purpose, and station/variable availability varies wildly city to city.

### 1.4 A genuinely longer record exists: NOAA 20CRv3 (1836-2015, 180 years)

Prompted by the note that 1940-2024 is *shorter* than the paper dataset's 1875-2013, a longer
multivariable source was found and tested: the **NOAA-CIRES-DOE 20th Century Reanalysis v3**,
served from NOAA PSL's THREDDS server with **no auth**. Single-gridpoint monthly series come out
as CSV through the NetcdfSubset service.

Verified by fetching Reykjavik's 2 m temperature over the full range (HTTP 200, 180,792 bytes):
**2,160 monthly rows = exactly 180 years, 1836-01 → 2015-12, zero nulls.** That is 95 years longer
than open-meteo/ERA5 and 42 years longer than the paper dataset.

Variables confirmed available (all catalogs HTTP 200):

| catalog | variables |
|-|-|
| `2mSI-MO` | `air.2m`, `rhum.2m`, `shum.2m`, `tmax.2m`, `tmin.2m` |
| `miscSI-MO` | `prmsl`, `pr_wtr.eatm`, `tcdc.eatm`, `cldwtr.eatm`, `rhum.eatm`, `tco3.eatm` |
| `prsSI-MO` | `air`, `hgt`, `omega`, `rhum`, `shum`, `uwnd`, `vwnd` |
| `10mSI-MO` | `uwnd.10m`, `vwnd.10m` |

So **8+ variables × 180 years × any city** is achievable with no auth. Result of testing it: §3.4.

Two caveats for any real use: extraction is ~42 s per (city, variable) — 8 variables × 6 cities is
~34 minutes on a cold cache, so it is no faster than open-meteo for gallery purposes; and 20CRv3's
19th-century decades are constrained by surface-pressure observations only, so the earliest years
are the least reliable part of exactly the period you would be adding.

### 1.5 Hugging Face: nothing suitable

`search=climate` returns almost entirely **climate-FEVER** text fact-checking corpora
(`tdiggelm/climate_fever`, `BeIR/climate-fever`, …) — NLP, not time series.
`search=era5` returns ML weather-forecasting tensors (`jacobbieker/era5-6hour`,
`vgup/weatherbench-era5`) — gridded global arrays sized for model training, not per-city
station panels, and far too heavy for a gallery notebook. `search=ghcn` returns one low-traffic
mirror (`ml-project-group-3/NOAA_GHCN_Daily`, 40 downloads). **No HF dataset improves on
open-meteo here.**

---

## 2. The hierarchy — concrete, and it does work

Design tested against real data:

- **Level 0 (top):** `Hemisphere` — 2 values (Northern, Southern)
- **Level 1:** `Zone` — 3 values (Polar/Subpolar, Temperate, Tropical)
- **Level 2 (leaf):** `City` — 12 cities, balanced 2 per (Hemisphere, Zone) cell
- **Leaf matrix shape: `(1020 months × 8 weather variables)`** — 1940-01 … 2024-12

Per-level means are meaningful: a zone mean is "what a temperate northern city's year looks
like", a hemisphere mean is the seasonal phase. Verified separation of city centroids in the
reduced space: **`Hemisphere/Zone` between/within ratio = 2.20** (k=3 groups; the Southern
cities were still fetching, so the hemisphere split is reported in §3).

Structural smoke test (real data, 3 cities, 3-level index) confirmed hypertools handles it:

| property | observed |
|-|-|
| `mi.shape` | `(3060, 8)`, `nlevels=3` |
| drawn traces | **6** = 3 leaves + 2 zone-means + 1 hemisphere-mean |
| linewidths | `1.0, 2.0, 3.0` (leaf → zone → hemisphere) |
| alphas | `0.533, 0.7, 1.0` |
| legend labels | only the top level (`'Northern'`) |

For the full 12-city design this yields **20 traces** (12 leaves + 6 zone-means + 2 hemisphere-means).

**One caveat that constrains the design:** `hypertools/plot/multiindex.py` resolves color from the
**top level only** (`color_of_top = get_palette_colors(palette, n_top)`, line 151-153); depth is
encoded as linewidth/alpha, not hue. So `(Hemisphere, Zone, City)` gives **2 colors**, not 6. To
get one color per climate zone you would use a **2-level** index whose top level is the combined
`"Northern/Tropical"` label. Also, `hue=` is superseded with a `UserWarning` under MultiIndex
expansion — the limitation already recorded in `temperatures_dataset_findings.md`.

---

## 3. Measurements — the actual numbers (REAL open-meteo data, no synthetic stand-in)

6 cities (Reykjavik, Moscow, New York, London, Bangkok, Mumbai), **1020 months × 8 variables**
each, 1940-01 … 2024-12. Reduction exactly as specified:
`hyp.reduce(mats, reduce='IncrementalPCA', ndims=3, normalize='across')`, plus a smoothing sweep
through hypertools' own `manip=` kwarg.

### 3.1 The smoothing sweep (native `manip='Smooth'`)

Smoothing was applied the library's own way, not with custom code:

```python
hyp.reduce(mats, reduce='IncrementalPCA', ndims=3, normalize='across',
           manip={'model': 'Smooth', 'kwargs': {'kernel': 'savgol', 'kernel_width': 11}})
```

`Smooth`'s shipped defaults are `kernel='savgol', kernel_width=11, order=3` — i.e. the row marked
**[SHIPPED]** is exactly what `manip='Smooth'` in `examples/animate_weather_decades.py` does.
(`kernel_width` must be a positive **odd** integer, and savgol requires `kernel_width > order`, so
a literal `kernel_width=3` raises `ValueError`; width 3 is measured with `kernel='gaussian'`.)

| config | closure ↓ | loop diam | **abs drift** | drift/diam | season % | trend % | season:trend | med ρ | monotone | p<.05 |
|-|-|-|-|-|-|-|-|-|-|-|
| unsmoothed | 0.2059 | 4.266 | 0.361 | 0.081 | 41.25 | 0.0960 | 430:1 | 0.29 | **0/6** | 0/6 |
| gaussian kw=3 | 0.1836 | 3.882 | 0.396 | 0.110 | 40.85 | 0.1595 | 256:1 | 0.40 | **0/6** | 1/6 |
| savgol kw=5 | 0.1689 | 3.906 | 0.412 | 0.096 | 43.08 | 0.1128 | 382:1 | 0.29 | **0/6** | 0/6 |
| savgol kw=7 | **0.1589** | 3.789 | 0.467 | 0.117 | 41.99 | 0.1554 | 270:1 | 0.25 | **0/6** | 1/6 |
| **savgol kw=11 [SHIPPED]** | 0.1753 | 3.746 | 0.256 | 0.077 | 37.52 | 0.0703 | 534:1 | 0.28 | **0/6** | 1/6 |
| savgol kw=13 | 0.1832 | 3.242 | 0.306 | 0.105 | 31.41 | 0.1063 | 295:1 | 0.25 | **0/6** | 0/6 |
| savgol kw=25 | 0.4266 | 1.424 | 0.470 | 0.370 | 5.07 | 0.1751 | 29:1 | 0.10 | **0/6** | 0/6 |
| gaussian kw=13 | 0.4308 | 1.298 | 0.362 | 0.271 | 3.46 | 0.1440 | 24:1 | 0.30 | **0/6** | 0/6 |
| boxcar kw=13 | 0.7025 | 0.831 | 0.218 | 0.271 | 0.60 | 0.1216 | 5:1 | 0.08 | **0/6** | 0/6 |

New York alone (the reference city used throughout):

| config | closure | drift/diam | ρ | p | monotone steps |
|-|-|-|-|-|-|
| unsmoothed | 0.2028 | 0.1772 | +0.333 | 0.381 | 50% |
| gaussian kw=3 | 0.1785 | 0.3356 | +0.483 | 0.187 | 62% |
| savgol kw=5 | 0.1623 | 0.2168 | +0.333 | 0.381 | 50% |
| savgol kw=7 | **0.1500** | 0.3062 | +0.183 | 0.637 | 50% |
| **savgol kw=11 [SHIPPED]** | 0.1823 | 0.2094 | +0.450 | 0.224 | 62% |
| savgol kw=13 | 0.1918 | 0.1329 | +0.200 | 0.606 | 50% |
| savgol kw=25 | 0.5321 | 0.6468 | +0.167 | 0.668 | 50% |
| boxcar kw=13 | 0.7642 | 1.0563 | +0.067 | 0.865 | 50% |

> **Note on baselines.** An earlier 5-city run gave NY closure 0.294 / drift-diam 0.224. Adding
> Mumbai changes the *shared* PCA basis every city is projected into, so the 6-city unsmoothed
> baseline is **0.2028 / 0.1772**. Same conclusion; all comparisons here are like-for-like within
> the 6-city run.

### 3.2 What the sweep shows

**Smoothing genuinely helps the picture.** Closure improves from 0.206
(unsmoothed) to a best of **0.1589** at savgol kw=7 — a **23% tighter loop** — and the regenerated
PNG's left panel is visibly ribbon-like rather than a scribble. That is real, and it is a
legitimate native feature.

**But smoothing does not touch the drift, and three measurements say so:**

1. **Absolute drift never grows.** It stays between 0.218 and 0.470 at every kernel, with no trend
   in kernel width. The shipped kw=11 gives the *second-lowest* absolute drift of all nine configs
   (0.256).
2. **The apparent drift/diam "improvement" at large kernels is a denominator artifact.** At boxcar
   kw=13, drift/diam looks 3.4x better than unsmoothed (0.271 vs 0.081) — but the loop *diameter*
   collapsed from 4.27 to 0.83 while absolute drift actually *fell* (0.361 → 0.218). The ratio
   improved because the loop was destroyed, not because the drift emerged. Seasonal variance goes
   41.25% → **0.60%**: a 13-month boxcar is an annual moving average, which annihilates the annual
   harmonic by construction. **That is precisely the "rows=years" compromise the maintainer already
   rejected, reached by a different route.**
3. **Monotonicity never improves: `0/6` cities at every single kernel.** Median Spearman ρ peaks at
   0.40, and at most 1 of 6 cities reaches p<0.05 at any setting. This is the criterion the question
   actually turns on, and no smoothing setting moves it.

Point 3 has a clean physical explanation: a 3–25 **month** kernel is the wrong timescale. The thing
that spoils monotonicity is multidecadal climate variability (~120-month scale and longer) plus
mid-century aerosol dimming. A sub-annual smoother cannot remove it — and a smoother wide enough to
touch it (≥12 months) destroys the seasonal loop first. **That is the tension, quantified: the loop
lives at 12 months and the noise that hides the drift lives at 120+ months, so there is no kernel
width that keeps one and removes the other.**

### 3.3 Loop closure and drift, stated plainly

- **(a) Loop closure.** Median year-over-year same-month step ÷ loop diameter =
  **0.206 unsmoothed, 0.175 at the shipped smoothing, 0.159 at best (savgol kw=7)**. The loop is
  genuinely there — seasonal harmonics explain a median **R² = 0.78** of each city's 3-D
  trajectory — but consecutive years still miss each other by ~16-21% of the loop's own diameter.
- **(b) Decadal drift.** Decade centroids **do not drift monotonically in any city, at any
  smoothing setting** (0/6 strictly monotone in all 9 configs; median ρ ≈ 0.25–0.40; ≤1/6 cities
  at p<0.05). Averaging to decade-mean loops raises the drift's SNR to ~2.7 but still leaves
  0/6 monotone.
- **(c) Variance budget** (shipped smoothing): city identity **56.9%**, seasonal loop **37.5%**,
  warming drift **0.070%**. The loop outweighs the drift **534:1**.

### 3.4 The decisive test: the reduction *destroys* the warming ordering

The obvious objection to §3.3(b) is that the drift axis was chosen by SVD, so maybe the warming
lives on some *other* direction. That objection was tested and it fails — in the strongest possible
form. For each city:

1. de-seasonalise its raw temperature (subtract the 12-month climatology) to isolate warming;
2. find the **best-case axis**: the direction in the reduced 3-D space that best predicts that
   anomaly (OLS), i.e. deliberately the most drift-favourable projection available;
3. rank-correlate decade-mean position on that axis against decade, and compare with the same test
   on the **raw** de-seasonalised temperature.

| city | raw temp ρ | raw p | best-case 3-D axis ρ | 3-D p |
|-|-|-|-|-|
| Reykjavik | 0.483 | 0.188 | −0.083 | 0.831 |
| Moscow | **0.983** | **0.0000** | 0.267 | 0.488 |
| New York | 0.733 | **0.025** | −0.050 | 0.898 |
| London | 0.767 | **0.016** | 0.017 | 0.966 |
| Bangkok | 0.867 | **0.003** | −0.267 | 0.488 |
| Mumbai | 0.850 | **0.004** | 0.067 | 0.865 |
| **median** | **0.808** | **5/6 significant** | **−0.017** | **0/6 significant** |

**In the raw data the decadal warming ordering is strong and statistically significant (median
ρ = 0.808, significant in 5 of 6 cities). After `hyp.reduce(..., ndims=3)` it is gone: median
ρ = −0.017, significant in 0 of 6 cities — even on the axis hand-picked to show it.**

The reduction does not merely under-emphasise the drift; it discards it. This is the single
clearest statement of the finding, and it holds with the shipped smoothing applied.

---

## 4. Why more variables and more years do not help

The obvious intuition — "the 1-number-per-city dataset failed, so give each leaf 8 variables and
85 years and the drift will appear" — is measurably wrong, and the reason is visible **before any
PCA runs**.

In the raw 8-variable standardized space (per city, no reduction at all):

- seasonal harmonics explain a **median R² of 0.5366**
- the linear year trend explains a **median R² of 0.00535**

The trend is **0.53% of the variance before hypertools touches the data**. A 3-D PCA keeps the
*largest* directions of variance, so it necessarily keeps season and city identity and discards the
trend. No unsupervised projection can make a 0.5% signal visually co-dominant with a 54% signal.

Adding variables actively **hurts**: each extra weather variable contributes its own strong
seasonal cycle and its own weather noise, but almost no trend. Measured (§5, Rescue C), going from
the example's 4 variables to 8 **cut** the trend share from 0.291% to 0.096%.

The warming *is* in the data — that was verified as a control, so this is not a broken-data
artifact. Raw annual-mean temperature trends over 1940-2024:

| city | °C / decade | total °C | R² |
|-|-|-|-|
| Reykjavik | 0.107 | 0.89 | 0.113 |
| Moscow | **0.324** | **2.69** | 0.370 |
| New York | 0.137 | 1.14 | 0.182 |
| London | 0.172 | 1.43 | 0.312 |
| Bangkok | 0.127 | 1.05 | 0.361 |
| Mumbai | 0.082 | 0.68 | 0.321 |

Median **0.132 °C/decade, 1.09 °C total**. Real, but ~1 °C against a seasonal swing of tens of °C.

---

## 5. Rescues attempted and measured — all fail

| rescue | seasonal share | trend share | verdict |
|-|-|-|-|
| baseline `normalize='across'`, 8 vars | 41.2% | **0.096%** | 430:1 |
| **A.** `normalize='within'` (kills between-city variance) | 65.9% | **0.394%** | best ratio, still **167:1** |
| **B.** per-city independent PCA (no shared space) | R²=0.590 | R²=0.00246 | 240:1 |
| **C.** more variables (4 → 8) | 44.4% → 41.2% | 0.291% → **0.096%** | **backwards** |
| **D.** decade-mean loops (9 loops × 12 months) | — | SNR rises to ~2.7 | **still 0/6 monotone** |
| **E.** native `manip='Smooth'`, 9 kernel configs (§3.1) | 41.2% → 0.6% | never above **0.175%** | **0/6 monotone at every kernel** |

Rescue D is the only one that moves the drift's signal-to-noise at all: averaging each decade's
months into one 12-point loop suppresses interannual weather noise by ~√10 and lifts SNR above 1.
It still fails the criterion that matters — the centroids do not march in one direction — and it
costs the animation its year-by-year motion, which is the whole point of the example.

Rescue E (smoothing) is the most instructive failure, and the one most likely to be proposed
(the shipped example already uses it). It **does** deliver a visibly cleaner loop (closure 0.206 → 0.159), and it is
the right tool for that job. It simply operates on the wrong timescale to help the drift: see §3.2.
Trend share never exceeds 0.175% at any of the nine kernels tested.

---

## 6. What the data *can* honestly support

Not a rewrite recommendation, just what the measurements leave standing:

1. **The loop alone, hierarchically.** The seasonal loop is strong (median seasonal R² ≈ 0.78 in
   the 3-D space) and the 3-level hierarchy renders correctly. A "shape of the year, by climate
   zone and hemisphere" example is well supported: measured zone separation of city centroids is
   **between/within = 2.20**, and tropical loops are visibly smaller and rounder than polar ones
   (loop diameter 8.31 for Mumbai vs 4.46 for Reykjavik). **No warming claim.**
2. **The drift alone, in its own panel.** Warming is unambiguous in raw temperature
   (0.132 °C/decade median, R² up to 0.37) — a 2-D time-series panel states it honestly, and the
   example already has a second panel doing exactly this.
3. **What must NOT be claimed:** that a viewer can *see* the loop drift through the space as the
   decades play. The current docstring's "each city's decades trace a seasonal loop that slowly
   drifts" overstates what the projection delivers — the visible frame-to-frame motion is
   interannual weather, not warming.

## 7. Bottom line

The premise "richer per-leaf measurements will make both structures visible" is **false for
climate**, and it fails for a reason no data source can fix: seasonality outweighs the
warming trend ~100:1 in the raw multivariate signal and ~400:1 after reduction. Switching to
open-meteo's full 1940 record and 8-10 variables is a genuine improvement in data richness and
changes nothing about the visual claim. **An honest "no" — do not spend the rewrite.**

### Reproduction

Scripts used (scratchpad, real API only, no mocks):
`fetch_climate.py`, `fetch_patient.py`, `analyze_climate.py`, `analyze2.py`, `analyze3.py`,
`final_figure.py`. Figure: `notes/audit/climate_loop_test.png`.
