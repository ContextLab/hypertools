# Hierarchical (MultiIndex) showcase example -- candidate datasets

Brainstorm + feasibility evidence, 2026-07-27. No code changed.

Companion to `notes/audit/temperatures_dataset_findings.md` (which documents why the
ContextLab-paper city-temperature dataset cannot satisfy the brief).

## Ground rules used for scoring

From `hypertools/plot/multiindex.py` (read, verified):

- Row MultiIndex, `L = index.nlevels >= 2`. `expand_multiindex` makes one leaf DataFrame
  per unique full index tuple; `build_multiindex_styles` adds one mean array per unique
  value-combination of every non-leaf prefix.
- `linewidth = 1 + (L - 1 - level_idx)`; `alpha = min(1, 1/(level_idx+1) + 0.2)`;
  color from the **top level only**; legend label on the **top-level means only**.
- So trace count = `n_leaves + sum over k in [0, L-2] of n_distinct_prefixes(k)`.

Hard constraint (the make-or-break criterion): each leaf must be an
`(n_timesteps x n_features)` matrix with `n_features >= 3`, ideally many more. One scalar
per timestep per leaf = automatic reject.

Second criterion: two timescales visible at once -- something cyclical/recurring AND
something drifting/progressing.

**Evidence standard used below:** for candidates 1-4 I actually downloaded the real data,
built the real `pd.MultiIndex`, called `hyp.plot`, counted the drawn artists, and measured
cycle radius vs. net drift in the reduced 3-D space. Those numbers are quoted verbatim.
For candidates 5-10 I verified URL reachability and file structure only, and say so.

---

## Reachability results (all tested with curl, 2026-07-27)

| # | URL | HTTP | bytes | notes |
|-|-|-|-|-|
| 1 | `https://api.delphi.cmu.edu/epidata/fluview/?regions=hhs1,...,hhs10&epiweeks=201040-202340` | 200 | 6790 records | JSON, `"result": 1, "message": "success"` |
| 2 | `https://raw.githubusercontent.com/Kjyesta30/Beijing-Multi-Site-Air-Quality/master/Data%20Source/Original%20Data/PRSA_Data_Aotizhongxin_20130301-20170228.csv` | 200 | 2,835,916 | branch is `master`, not `main`; all 12 stations 200 |
| 3 | `https://physionet.org/files/gaitpdb/1.0.0/GaCo01_01.txt` | 200 | 1,116,032 | dir listing 200, 306 trial files enumerated |
| 3b | `https://physionet.org/files/gaitpdb/1.0.0/demographics.txt` | 200 | 15,665 | ID/Study/Group/Age/HoehnYahr/UPDRS/Speed |
| 4 | `https://phenocam.nau.edu/data/archive/harvard/ROI/harvard_DB_1000_3day.csv` | 200 | 559,870 | 14 site/ROI files all 200 |
| 4b | `https://phenocam.nau.edu/api/roilists/?format=json&limit=1000` | 200 | 1,024,745 | 1546 ROIs, veg-type codes |
| 5 | `https://api.energy-charts.info/public_power?country=de&start=2015-01-01&end=2015-01-03` | 200 | 42,937 | 21 named series |
| 5b | `https://api.energy-charts.info/public_power?bzn=NO1&...` (also NO3, SE1, SE3, DK1, DK2, IT-North, IT-Sicily) | 200 x8 | 43,199 each | 21 series per zone |
| 6 | `https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00094728.csv` | 200 | 17,808,413 | 57,548 rows x 124 cols, 1869-2026 |
| 7 | `https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NA.list.v04r01.csv` | 206 (range GET) | -- | header confirmed |
| 8 | `https://data.open-power-system-data.org/time_series/latest/time_series_60min_singleindex.csv` | 200 | **130,339,665** | too big; last-modified 2020-10-06 |
| 9 | `http://mocap.cs.cmu.edu/subjects/07/07.asf` and `.../07_01.amc` | 206 x2 | -- | ASF/AMC, not tabular |
| -- | `https://covid.ourworldindata.org/data/owid-covid-data.csv` | **000** | 0 | connection failed, 3 attempts, 45 s timeout |
| -- | `https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv` | 206 | -- | OWID *is* reachable at this host |
| -- | `https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip` | 200 | -- | **`.zip` -- `hyp.load` cannot read it** |
| -- | `https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip` | 200 | -- | same `.zip` problem |
| -- | `https://archive.ics.uci.edu/static/public/298/bach+chorales.zip` | **404** | -- | dead |

`hyp.load()` behaviour on these URLs (actually called, not assumed):

| URL | result |
|-|-|
| Beijing raw-GitHub CSV | **OK** -- `DataFrame (35064, 18)`, cols `No, year, month, day, hour, PM2.5, PM10, SO2, NO2, CO, ...` |
| PhysioNet `GaCo01_01.txt` | **OK** -- `DataFrame (12118, 19)`, but no header row in the file so the first data row becomes the column names; needs `pd.read_csv(url, sep='\t', header=None)` |
| PhenoCam `*_3day.csv` | **FAILS** -- `HypertoolsIOError: could not load ...`; the file starts with ~20 `#` comment lines. Needs `pd.read_csv(url, comment='#')` (1 line) |

`hyp.load` supported payloads (from `hypertools/io/load.py`): npy/npz/csv/tsv/txt/json/
parquet/mat/xlsx + `.gz` variants. **Plain `.zip` is not supported**, which rules out
reading any UCI archive directly.

---

## Candidate 1 -- CDC ILINet influenza surveillance (Delphi Epidata) **[RECOMMENDED #1]**

**Hierarchy.** Level 0 = Census region (Northeast / South / Midwest / West, 4 values);
level 1 = HHS region (`hhs1`..`hhs10`, 10 leaves). Optionally a 3rd level of the 51
states -- but see weakness below.

**Leaf matrix.** One row = one epidemiological week. Columns = `num_ili`, `num_patients`,
`num_providers`, `num_age_0` (0-4y), `num_age_1` (5-24y), `num_age_3` (25-49y),
`num_age_4` (50-64y), `num_age_5` (65+y), `wili`, `ili`.
**Verified leaf = (679, 10)** for epiweeks 201040-202340, and
**0.0 null fraction on every one of the 10 columns**.

**Verified plot.** `hyp.plot(mi, legend=True, normalize='across')` drew
**14 traces** = 10 leaves at `linewidth 1.0` + 4 census-region means at `linewidth 2.0`,
legend `['Northeast', 'South', 'Midwest', 'West']`. Exactly the documented behaviour.

**Two timescales -- YES, and measured.** Reducing each region to 3-D and slicing into
52-week seasons, the per-season loop radius for `hhs1` is:

```
[6910.1, 6777.9, 7330.7, 7733.1, 7499.5, 7255.6, 8769.1, 7594.6, 7328.8,
 23852.1, 11963.4, 8501.6, 6636.0]
```

Nine tidy comparable annual loops (~6.8k-8.8k), then a **3x excursion in the 2019-20
season (23852)**, a still-inflated 2020-21 (11963), then a return to a *different*
resting place. That is a genuine cycle-plus-regime-change, not a story I had to invent.

**Data source.** `https://api.delphi.cmu.edu/epidata/fluview/` -- free, no auth, JSON,
tiny (the full 10-region 13-year pull is 6790 records). Delphi Epidata API, CMU;
ILINet data are US Government public-domain, the API asks for citation. Verified HTTP 200.

**Visual for a ~15 s animation.** Each region emits a loop per flu season, all four
census-region means bold on top of ten faint region leaves. Nine seasons trace nested,
comparable orbits; then 2020 throws one loop three times the size of anything before it
and the whole bundle settles somewhere new. The animation reads as "a heartbeat, then an
arrhythmia."

**Prep effort -- LOW.** One `urllib`/`requests` call, `pd.DataFrame(raw['epidata'])`,
`sort_values('epiweek')`, attach a 2-level MultiIndex from a 10-entry dict. ~12 lines,
after which it is all hypertools.

**HONEST WEAKNESSES.**
1. Of the 10 columns only ~6 are independent: `num_age_0/1/3/4/5` sum to `num_ili`, and
   `wili`/`ili` are `num_ili / num_patients` rescalings. Still comfortably >= 3, but do
   not claim "10 independent features".
2. Raw counts scale with region population, so leaves need `normalize=` (a hypertools
   kwarg) or the big regions dominate the shared PCA.
3. **The 2020 spike is partly an artifact** and the tutorial must say so: ILINet counts
   "influenza-like illness" visits, and in March 2020 COVID patients presented with ILI
   symptoms while routine care collapsed. Selling it as "a huge flu season" would be
   dishonest.
4. State level does *not* work as a 3rd hierarchy level: I checked `regions=ca,ny,tx` and
   **`num_age_0` .. `num_age_5` are all `null`** there, dropping the leaf to ~3
   independent features.
5. JSON, so not a `hyp.load(url)` one-liner.

---

## Candidate 2 -- Beijing multi-site air quality, 2013-2017 **[RECOMMENDED #2]**

**Hierarchy.** Level 0 = zone (urban / suburban / rural, 3 values); level 1 = station
(12 leaves: 8 urban, 3 suburban, Dingling rural).

**Leaf matrix.** One row = one time bin, columns = `PM2.5, PM10, SO2, NO2, CO, O3, TEMP,
PRES, DEWP, WSPM` -- 10 genuinely independent physical measurements.
**Verified leaves: (48, 10) monthly**, `(209, 10)` weekly, or `(35064, 18)` raw hourly
straight out of `hyp.load`.

**Verified plot.** `hyp.plot(mi, legend=True)` drew **15 traces** = 12 leaves
(`linewidth 1.0`) + 3 zone means (`linewidth 2.0`), legend
`['urban', 'suburban', 'rural']`. Total frame `(576, 10)`, 2 NaNs.

**Two timescales -- YES but modest.** Measured on the reduced 3-D coords, per-station
`net_drift / cycle_radius` over the 4 annual cycles:

```
Aotizhongxin  radius=454.8  drift=96.1  ratio=0.21
Changping     radius=409.4  drift=47.3  ratio=0.12
Dingling      radius=309.5  drift=59.0  ratio=0.19
```

Real drift, in the right direction (the 2013-2017 "war on pollution"), but only ~1/5 of
the seasonal swing.

**Data source.** Canonical: UCI ML Repo #501, but it ships as a `.zip` which `hyp.load`
cannot read. Working per-station CSVs (verified 200, 2.6-2.9 MB each, all 12):
`https://raw.githubusercontent.com/Kjyesta30/Beijing-Multi-Site-Air-Quality/master/Data%20Source/Original%20Data/PRSA_Data_<Station>_20130301-20170228.csv`
`hyp.load()` on that URL returns a `(35064, 18)` DataFrame directly (verified).
Underlying data: Zhang et al. 2017, UCI, CC BY 4.0.

**Visual for a ~15 s animation.** A bundle of 12 seasonal loops -- red urban, green
suburban, blue rural -- with bold zone means running through them, sliding steadily across
the space as the four years play. Clean, readable, and a story anyone recognises.

**Prep effort -- LOW.** 12 `hyp.load(url)` calls (or `pd.read_csv`), a
`groupby(Grouper(freq='MS')).mean()`, one zone dict, `pd.MultiIndex.from_arrays`. ~15 lines.

**HONEST WEAKNESSES.**
1. **Verified failure mode:** with raw units the plot collapses to a flat sliver, because
   `PRES` (~1000 hPa) and `CO` (~1000 ug/m3) swamp `SO2` (~10). Fixed by
   `normalize='across'` -- one hypertools kwarg, but the tutorial must not omit it. I have
   the before/after PNGs.
2. Only **4 annual cycles**, and drift/radius ~= 0.2. This is a milder version of exactly
   the temperature failure: the seasonal loop dominates the drift. Honest, not fatal, but
   do not oversell "warming-style drift".
3. **Hosting is fragile.** The only per-file mirror I could verify is a 1-star personal
   GitHub repo with **no license file** and default branch `master`. If this becomes a
   shipped tutorial it should be re-hosted on the hypertools Dropbox alongside the other
   `EXAMPLE_DATA` entries, with the UCI/CC-BY attribution carried over.
4. The zone assignment (urban/suburban/rural) is my own grouping of the 12 sites, defensible
   from their locations but not a field in the data.

---

## Candidate 3 -- PhysioNet gait in Parkinson's disease (`gaitpdb`)

**Hierarchy.** 3 levels: group (control / Parkinson) -> study (`Ga` Yogev, `Ju` Hausdorff,
`Si` Frenkel-Toledo) -> subject. Enumerated from the live directory listing:
**306 trial files, 165 unique subjects**, split `GaCo 38 / GaPt 75 / JuCo 25 / JuPt 104 /
SiCo 29 / SiPt 35`. Trial suffixes `_01` (165), `_02` (52), `_03`..`_07`, and `_10` (27 --
the 10-minute walks). `demographics.txt` (200, 15,665 B) adds Age / Height / HoehnYahr /
UPDRS / TUAG / walking speed.

**Leaf matrix.** One row = one 10 ms sample (100 Hz). Columns = 8 left-foot force sensors,
8 right-foot force sensors, left total, right total = **18 features**. Raw files are
`12119 x 19` (col 0 is time). I tested a 30 s window decimated to 20 Hz -> **leaf (600, 18)**.

**Verified plot.** `hyp.plot(mi, legend=True, normalize='across')` on 8 trials drew
**16 traces** = 8 leaves (`lw 1.0`) + 6 (group, study) means (`lw 2.0`) + 2 group means
(`lw 3.0`), legend `['control', 'Parkinson']`. Textbook 3-level behaviour.

**Two timescales -- NO. This is the one that fails the second criterion, and I am
flagging it hard.** The cycle is beautiful: dominant period 1.10-1.45 s with
autocorrelation at that lag of **0.84-0.92** across all 8 trials. But the drift is not
there -- `net_drift / cycle_radius` came out **0.03, 0.21, 0.19, 0.06, 0.06, 0.06, 0.04,
0.06**. Steady-state walking is quasi-stationary; there is nothing progressing. The
between-group separation is real and large, but it is a *grouping* effect, not the
second timescale the maintainer asked for.

**Data source.** `https://physionet.org/files/gaitpdb/1.0.0/<ID>_<trial>.txt` -- verified
200, tab-separated plain text, ~0.5-1.2 MB per trial. Free, no auth. ODC-BY 1.0. Citable
(Hausdorff et al.; Goldberger et al. 2000 for PhysioNet).

**Visual.** The best-looking of everything I rendered: control orbits and Parkinson orbits
form two cleanly separated sheets of dense cyclic texture. Genuinely striking as a still.

**Prep effort -- VERY LOW.** Filename encodes group and study; one `pd.read_csv(url,
sep='\t', header=None)` per trial.

**Verdict.** Ranks 3rd only because of the missing drift. If the brief were relaxed to
"cycles plus a strong group contrast", this would be #1.

---

## Candidate 4 -- PhenoCam vegetation phenology

**Hierarchy.** 3 levels: vegetation type -> site -> ROI. From the ROI API (1546 ROIs):
`GR 204, EN 180, AG 161, DB 161, EB 73, SH 68, WL 60, UN 34, TN 32, XX 13`. Restricting to
ROIs spanning 2012-2024 leaves ~17, which I turned into 14 leaves over 6 vegetation types.

**Leaf matrix.** One row = one month (from the 3-day product). Columns used:
`gcc_mean, rcc_mean, r_mean, g_mean, b_mean, gcc_90, rcc_90, max_solar_elev`.
**Verified leaf = (144, 8)** for 2013-2024 (one site, `caryinstitute`, was (139, 8)).

**Verified plot.** `hyp.plot(mi, legend=True)` drew **34 traces** = 14 leaves (`lw 1.0`) +
14 (vegtype, site) means (`lw 2.0`) + 6 vegtype means (`lw 3.0`), 6 legend entries. The
deepest hierarchy I got working.

**Two timescales -- CYCLE YES, DRIFT SUSPECT.** 12 crisp annual green-up/senescence loops
per leaf. But the measured drift is wildly inconsistent:

```
DN canadaOBS      ratio=0.80     DB harvard        ratio=0.24
EN canadaOBS      ratio=0.60     DB harvardbarn    ratio=4.50   <-- implausible
DB caryinstitute  ratio=1.79     EN harvardbarn    ratio=3.24   <-- implausible
SH kendall        ratio=2.43     EN howland1       ratio=0.16
```

The ratio-4.5 and ratio-3.2 leaves are almost certainly **camera / exposure / ROI-redraw
artifacts, not ecology** -- the two implausible ones share a site (`harvardbarn`). The real
phenological trend is a *phase* shift (green-up arriving earlier), which shows up as
rotation within the loop, not as translation, and is therefore much harder to see.

**Data source.** `https://phenocam.nau.edu/data/archive/<site>/ROI/<site>_<VEG>_<roi>_3day.csv`
-- verified 200 for all 14, 0.4-0.6 MB each. ROI catalogue at
`https://phenocam.nau.edu/api/roilists/?format=json&limit=1000` (200, 1.0 MB). Free, no
auth. PhenoCam data are CC BY 4.0.

**Visual.** 12 nested annual loops per leaf, bold means per vegetation type. Pretty, and
the color axis has an obvious real-world meaning (it is literally the color of the forest).
But at 14 leaves in a shared PCA space my render was a hairball.

**Prep effort -- LOW-MEDIUM.** The `#` header block breaks `hyp.load` (verified), so
`pd.read_csv(url, comment='#')`. Site/ROI/veg-type must come from filenames or the API.

**HONEST WEAKNESSES.** The 8 features are strongly collinear -- `gcc = g/(r+g+b)`,
`rcc = r/(r+g+b)`, and `gcc_90` tracks `gcc_mean` -- so the effective rank is ~4, not 8.
`max_solar_elev` is pure astronomy, a free perfect sinusoid that will dominate the cycle
if left in; that is arguably cheating and I would drop it. And the drift is contaminated
by instrument artifacts, which is a bad look for a teaching example.

---

## Candidate 5 -- European electricity generation mix by bidding zone (energy-charts)

**Hierarchy.** Level 0 = country, level 1 = bidding zone. Real multi-zone countries:
Norway (NO1-NO5), Sweden (SE1-SE4), Denmark (DK1, DK2), Italy (IT-North, IT-Sicily, ...).
I verified **8 zones return HTTP 200** (`NO1, NO3, SE1, SE3, DK1, DK2, IT-North,
IT-Sicily`), 43,199 bytes each for a 2-day pull.

**Leaf matrix.** One row = one 15-min/hourly step. **21 named series** per zone, verified:
`Nuclear, Hydro Run-of-River, Biomass, Fossil brown coal / lignite, Fossil hard coal,
Fossil oil, Fossil gas, Geothermal, Hydro water reservoir, Hydro pumped storage, Waste,
Wind offshore, Wind onshore, Solar, Load, Residual load, Renewable share of load, ...`

**Two timescales -- YES, and the best "loop inflates" story of the whole set.** Daily
solar/load cycle + weekly cycle + annual cycle, riding on a decade of energy transition in
which the *amplitude of the daily loop grows* as solar capacity is built out. A cycle that
visibly inflates is a stronger visual than a cycle that translates.

**Data source.** `https://api.energy-charts.info/public_power?bzn=<ZONE>&start=&end=`
-- Fraunhofer ISE, free, no auth, JSON, CC BY 4.0. Verified 200.

**NOT VERIFIED END-TO-END.** I did not build the MultiIndex or plot it. The likely
problems: the series list differs per zone (Norway has no lignite), so leaves would need a
common column subset; and covering a decade at 15-min resolution means many paginated
requests. Also note the alternative Open Power System Data file is **130,339,665 bytes**
(verified `content-length`) and last updated 2020-10-06 -- too big and too stale for a
tutorial.

---

## Candidate 6 -- NOAA GHCN-Daily, one CSV per station

**Hierarchy.** Country/climate zone -> station. Any number of leaves, one URL each.

**Leaf matrix -- THIS IS WHERE IT FAILS.** `USW00094728.csv` (verified 200,
17,808,413 bytes) is 57,548 rows x 124 columns covering 1869-2026, which sounds ideal.
But I measured actual coverage since 1970 and only five elements clear 90%:

```
PRCP 1.000  TMAX 1.000  TMIN 1.000  SNOW 1.000  SNWD 0.959
AWND 0.541  WSF2 0.529  WDF2 0.529  WSF5 0.528  WDF5 0.526
```

So a *reliable* leaf is `(n_days, 3)` -- TMAX/TMIN/PRCP -- because SNOW/SNWD are
identically zero at any low-latitude station and wind is only ~53% present. Three features
technically clears the bar, but barely, and it would not survive a station in Bangkok.

**Two timescales.** Seasonal loop plus warming drift -- but this is precisely the failure
mode already documented: the seasonal amplitude (~20 C in NYC) dwarfs the century warming
(~1.5 C), so drift/radius would be ~0.08. **Worse than Beijing on the criterion the
maintainer cares about.**

**Verdict: reject.** Free, public domain, trivially easy -- and still not enough features.
The open-meteo ERA5 archive (used by `examples/animate_weather_decades.py`) is the better
version of this idea, with ~10 daily variables, but it duplicates an existing example and
I hit a **HTTP 429 rate limit** on `archive-api.open-meteo.com` while testing.

---

## Candidate 7 -- Built-in `weights` (fMRI, HTFA)

**Verified shapes** (loaded, not assumed): `weights` = list of **36 arrays, each
(300, 100)**; `weights_sample` = 3 x (300, 100); `weights_avg` = 2 x (100, 100).

**Leaf matrix -- PASSES EASILY.** 300 timepoints x 100 HTFA parameters is the largest
per-leaf feature count of anything here, and it is hypertools' home turf with zero hosting.

**Hierarchy -- THIS IS THE PROBLEM.** The `load()` docstring says only "brain activity
(fMRI) from subjects listening to the same story ... one PER SUBJECT (36 arrays)". There is
**no condition/group label shipped with the data**. `weights_avg` being "one per group"
hints at a 2x18 structure, but I could not confirm it from the payload and I will not
assert it. Building the demo on an invented grouping would be exactly the dishonesty the
brief warns about.

**Two timescales -- NO.** Story listening has no cycle.

**Verdict: reject for this brief**, despite being the most convenient dataset in the repo.

---

## Candidate 8 -- Built-in `sotus` + `sotus_model` (narrative embeddings)

**Verified:** `hyp.load('sotus')` returns **29 strings**, 21k-28k characters each
(1989-2018). `sotus_model` is a CountVectorizer -> LDA pipeline producing 50-dim topic
vectors.

**Hierarchy.** Party -> President -> Speech is a genuine 3-level tree (2 parties,
5 presidents, 29 speeches) and needs no new data.

**Leaf matrix.** Slice each speech into N windows of narrative time; leaf =
`(N, 50)` topic vectors. With N = 50 that is (50, 50) -- passes the constraint.

**Two timescales -- PARTLY, and I would not claim otherwise.** The drift is real and
strong (topics move across three decades). The "cycle" is a *rhetorical arc repeated across
leaves* -- every SOTU opens the same way and closes on "God bless America" -- not a cycle
traversed repeatedly *within* one leaf. That is a different and weaker claim than what the
brief asks for.

**Prep.** ~10 lines, all hypertools. Zero hosting. Extends `examples/plot_sotus.py`.

**Verdict.** The best *convenience* option and a legitimately nice tutorial, but it does
not honestly satisfy "cyclical AND drifting".

---

## Candidate 9 -- IBTrACS tropical cyclone tracks

**Hierarchy.** Basin -> season -> storm. Thousands of leaves available.

**Leaf matrix.** One row = one 6-hourly fix; columns `LAT, LON, WMO_WIND, WMO_PRES,
DIST2LAND, USA_WIND, USA_PRES, USA_R34_NE/SE/SW/NW, ...` -- easily 10+. Header verified by
range GET (HTTP 206) on
`https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NA.list.v04r01.csv`.

**Two timescales -- NO, not as posed.** A storm track is a one-way life-cycle arc
(intensify -> peak -> decay), not a repeated cycle. The wind-pressure relation traces a
hysteresis loop, but only once per storm. The seasonal cycle lives *between* leaves, not
within them, so it will not appear as a loop in any leaf trajectory.

**Verdict: reject on criterion 3.** Also, leaves have wildly different lengths, which
triggers the aggregated unequal-length `UserWarning` in `build_multiindex_styles`.

---

## Candidate 10 -- Rejected outright (recorded so nobody re-tries them)

| Idea | Why rejected |
|-|-|
| **NBA/sports player-game stats** (`fivethirtyeight/nba-raptor` verified 206) | Hierarchy and features are fine (team -> player, ~20 stats per game) but there is **no cycle** -- a season is a one-way arc. Fails criterion 3. |
| **Spotify/audio features per track** | Leaf would be (n_tracks_in_album ~10, ~9 features). `n_timesteps = 10` is too short for a trajectory, and track order is not time. |
| **UCI Bach chorales** (12 pitch-class features -- would have been a lovely circle-of-fifths cycle) | Source **404** (`archive.ics.uci.edu/static/public/298/bach+chorales.zip`). Dead. |
| **PAMAP2 wearables** (52 IMU channels, periodic gait + HR drift -- genuinely great on paper) | Ships only as a **`.zip`** which `hyp.load` cannot read, and the archive is ~1.6 GB. Would need re-hosting. |
| **Sleep EEG (Sleep-EDF)** -- ~90-min NREM/REM cycles *plus* progressive lightening across the night is the single best two-timescale story available anywhere | Data are EDF; requires `mne`/`pyedflib` plus spectral band-power computation. That is precisely the 200-lines-of-wrangling failure we are correcting. |
| **OWID COVID** | `covid.ourworldindata.org` returned **HTTP 000** (connection failure) on 3 attempts. `catalog.ourworldindata.org/.../compact.csv` works (206), but the weekly-reporting cycle is a reporting artifact and the tutorial would be teaching people to admire an artifact. |
| **CMU mocap** (`07.asf`, `07_01.amc` both 206) | ASF/AMC skeleton+motion format, not tabular. Needs a parser. Same wrangling failure. |
| **Movebank telemetry** | Most studies need auth; open ones typically ship lat/lon/time only -- 2-3 features, at the bar at best. |
| **OECD/FRED/BLS economics** | SDMX or API-key-gated; heavy wrangling for a 3-4 feature leaf. |

---

## Ranking

| rank | candidate | leaf shape (verified) | levels | traces (verified) | cycle | drift | prep |
|-|-|-|-|-|-|-|-|
| **1** | **CDC ILINet / Delphi** | (679, 10) | 2 | 14 | strong (annual) | **strong** (3x 2020 excursion, measured) | low |
| **2** | **Beijing air quality** | (48, 10) / (209, 10) | 2 | 15 | strong (annual) | modest (ratio 0.12-0.21, measured) | low |
| 3 | PhysioNet gaitpdb | (600, 18) | 3 | 16 | **very strong** (ac 0.84-0.92) | **absent** (ratio 0.03-0.21) | very low |
| 4 | PhenoCam | (144, 8) | 3 | 34 | strong (annual) | contaminated by camera artifacts | low-med |
| 5 | energy-charts zones | ~(n, 21) not built | 2 | not run | strong (daily+annual) | strong (loop inflates) | medium |
| 6 | `sotus` built-in | (N, 50) | 3 | not run | weak (arc, not cycle) | strong | very low |
| 7 | GHCN-Daily | (n_days, **3**) | 2 | not run | strong | ~0.08, worse than Beijing | very low |
| 8 | `weights` built-in | (300, 100) | **none shipped** | n/a | none | none | none |
| 9 | IBTrACS | (n_fixes, 10+) | 3 | not run | none within leaf | n/a | low |

### Recommendation #1 -- CDC ILINet influenza surveillance

Pick this. It is the only candidate where I measured a two-timescale structure that is
both strong and honest: nine comparable annual loops (radius 6.8k-8.8k) followed by a 3x
excursion (23.8k) and a shifted resting state. The 14-trace hierarchy renders exactly as
`multiindex.py` documents, the leaf is (679, 10) with zero missing values, the API is free
and unauthenticated, and the whole prep is ~12 lines.

**Its weakness, stated plainly:** only ~6 of the 10 columns are independent (the five age
buckets sum to `num_ili`; `wili`/`ili` are ratios of two other columns), and the dramatic
2020 loop is *partly a measurement artifact* -- COVID patients presented with
influenza-like illness while routine care collapsed. The tutorial has to explain that
rather than let the viewer read it as a record flu season. If we are not willing to write
that paragraph, drop to #2.

### Recommendation #2 -- Beijing multi-site air quality

Pick this if we want the more familiar story and the simpler load path (`hyp.load()` reads
the CSV URL directly -- verified). Twelve stations, three zones, 15 traces, a 10-feature
leaf of genuinely independent physical quantities, and a real 2013-2017 cleanup drift.

**Its weakness, stated plainly:** the drift is only ~20% of the seasonal radius over just
four cycles, so it is a *milder* version of the same imbalance that killed the temperature
idea -- honest, but not dramatic. And it **collapses to a flat sliver without
`normalize='across'`** (verified; I have both PNGs), so the tutorial cannot skip that
kwarg. Separately, the only verified per-file mirror is an unlicensed 1-star personal
GitHub repo on branch `master`; before shipping, re-host on the hypertools Dropbox next to
the other `EXAMPLE_DATA` entries and carry the UCI / CC-BY-4.0 attribution.

### The one to NOT pick, despite how good it looks

PhysioNet `gaitpdb` produced the best-looking render of the entire exercise -- two cleanly
separated sheets of dense cyclic texture -- has the richest leaf (600 x 18 independent
force channels), a real 3-level hierarchy over 165 subjects, and a stable licensed host.
It fails on exactly one thing, and it is the thing that was asked for: measured
`drift/radius` of **0.03-0.21** means steady-state walking has no second timescale.
Recommending it would cost a rewrite.

---

## Reproduction

Scratch scripts and the rendered PNGs (`fig_beijing.png`, `fig2_beijing.png`,
`fig_phenocam.png`, `fig2_phenocam.png`, `fig2_gait.png`, `fig_flu.png`) live in this
session's scratchpad, not in the repo. Everything above can be regenerated from the URLs
in the reachability table; nothing here depends on a cached artifact.
