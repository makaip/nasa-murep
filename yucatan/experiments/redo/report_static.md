<!-- Hand-written sections 5-8 appended by report.py. Section 6 numbers come from kmeans_compare.json (run 2026-11-07); update here if they change. -->

## 5. Interpretation notes and issues found

1. **The a_CDOM-Chl correlation is mostly built into the algorithms.** `pickler_utils.calculate_cdom` / `calculate_cdom_slopes`
   compute ln a412 = -3.070 - 1.285 ln Rrs443 + 1.107 ln Rrs547, and ln S275 and ln S300 use the same two bands with
   opposite-sign coefficients (B1 ~ -B2). All three CDOM products are therefore near-monotonic functions of the
   Rrs443/Rrs547 ratio. MODIS Chl-a (OCI/OC3M) is also a blue/green band-ratio algorithm. That explains why:
   (a) a412, S275, S300 and log Chl give the same |Cliff's delta| in every month and method (section 3);
   (b) rho(a412, Chl) is about 0.97 everywhere, inside and outside;
   (c) the log-log scatter has a sharp kink near Chl ~ 0.3 mg/m3 and a412 ~ 0.04-0.05 1/m, where the two
   algorithms' functional forms differ (OC3M's max-band switch, the CI/OCx blend at low Chl).
   The pattern the supervisor saw (strong at low CDOM, tapering higher up) is real in the data (quartile rho 0.67,
   0.78, 0.96, 0.87; upper-half rho inside the zone 0.73-0.84 vs lower half 0.96-0.98). It is best described as a
   property of the retrieval algorithms, not independent biogeochemical evidence. Recommend saying this explicitly
   in the paper, and not presenting the CDOM-Chl correlation, or CDOM slope differences, as independent findings.
2. **Inside vs outside differences are large and consistent where a zone exists.** For May-Sep, IF zone pixels have
   about 4x higher a412 and Chl and lower S275/S300 (|Cliff's d| 0.7-0.9). The -1 sigma zone gives the same direction
   (|d| 0.73-0.89 in May-Sep). In winter (IF, Nov-Feb) differences vanish or reverse. Given note 1, these numbers
   mostly restate that the zone has higher Chl / a different blue/green ratio.
3. **Does correlation differ inside vs outside?** -1 sigma: no meaningful difference (paired month-year Wilcoxon
   p = 0.056, median diff -0.0006). IF: rho is lower inside (median 0.91 vs 0.98 across 94 month-years, p = 2e-14),
   which is consistent with the IF zone sitting mostly above the kink, where rho tapers. Naive Fisher-z p-values are
   meaningless at n ~ 10^6-10^7 autocorrelated pixel-days.
4. **K-means ~97% along SST is not reproduced.** With standardized SST, log10 Chl, SSL the squared centroid separation
   is split roughly evenly: SST 24-37%, Chl 29-40%, SSL 26-47%, global or per month, k=2 or k=3. Global K-means mostly
   splits by season: its "cold" cluster covers 61-73% of pixels in Dec-Apr and 8-27% in May-Sep, and only 17% of its
   zone falls in May-Sep. The 97% figure probably came from a different setup (unstandardized features, or SST-only
   distances). It should be recomputed or dropped.
5. **DBSCAN at old-notebook parameters finds essentially one cluster.** 110/120 month-years have 1 cluster, about 0.5%
   noise, and a separate cold cluster in only 9/120 month-years with no month-to-month consistency (IoU 0.04). This
   documents that density clustering does not resolve the zone at these settings. It is under-, not over-, segmented here.
6. **How often each approach "worked"** (zone present in a month-year, >=1% of pixels): -1 sigma 118/120 (by
   construction it always flags about 13-16% of pixels), IF 70/120 (95/120 with >=1 px; all 10 years in May-Sep,
   2/10 in Jan-Feb), K-means 120/120 (always, by construction), DBSCAN 0/120. Share of zone pixel-months in May-Sep:
   IF 68%, -1 sigma 49%, per-month K-means 33%, global K-means 17%. Only IF is seasonally selective.
7. **-1 sigma vs IF agree poorly on the exact pixels** (IoU mean 0.12; 0.19-0.28 in May-Aug). IF is nested inside -1
   sigma: 86.5% of IF zone pixels are also -1 sigma pixels. -1 sigma is roughly 6x larger and extends offshore
   along the whole shelf edge (see figs/zonemap_2018_07).
8. **Depth mask removes the coldest core.** In July 2018 the coldest monthly-mean SST is in water shallower than 10 m
   along the north coast, which is excluded from every method. The zone is what is cold *beyond* 10 m.
9. **Bugs / inconsistencies in the existing figure pipeline** (as found on 2026-11-07; the pickle-model and scatterplot items were FIXED on 2026-11-08, see section 10):
   - `figs/bayth_preclassified.pkl` (violins, monthly/yearly maps) was produced with `iforest_pipeline4.joblib`, while
     every script's `MODEL_PATH` points to `iforest_pipeline5.joblib`. 95/120 month-year maps differ (IoU p4-p5 ~ 0.75-0.83).
   - `scatterplots/common.py` classifies **per day** at the **95th** percentile with a **-20 m** depth cut, while the
     violins and maps use **monthly** maps, **90th** percentile and **-10 m**. Scatter "inside" pixels therefore
     do not match the mapped zone. This likely explains the earlier "violin months with data where the maps show none"
     and "scatter plots didn't change".
   - `violins/all_years.py` counts shallow (<10 m) pixels and pixels without a monthly class as "non-upwelling".
     The redo excludes them from both groups.
   - No negative a_CDOM exists in v4 (clipped to [0, 1] in preprocessing), so the old violins could not have cut
     negative values from v4 data. If the earlier draft showed negatives, they came from the older v3 pickle.

## 6. K-means preprocessing comparison (old-notebook pipeline replicated on v4)

Source of the old setup: `nasa-murep-local/cdom_sst/experiments/kmeans/ssl_kmeans_no_norm.ipynb` (raw) and `ssl_kmeans.ipynb` (StandardScaler).
Both used the v3 pickle, **daily** pixels from 2010-2019, features [log10 Chl, SST, SSL] (log10 *was* applied to Chl in both), SSL mean-filled and gaussian-blurred (sigma 2 px),
a 1,000,000-pixel random sample (seed 42), k=3, `KMeans(random_state=42, n_init=10)`, and no depth mask. Replicated here on v4 by `experiments/redo/kmeans_compare.py`.
Every valid daily pixel is then predicted; 'cold' = the cluster with the lowest SST centroid. Separation share, PCA and silhouette are computed in the space K-means ran in.

| Variant | k | sep. share Chl / SST / SSL (%) | PCA var. ratio (%) | silhouette | cold cluster % of pixel-days | cold share in May-Sep | month-years cold >=1% (/120) |
|---|---|---|---|---|---|---|---|
| a0 raw, exact old (no depth mask) | 2 | 0.9 / 99.0 / 0.1 | 94.8 / 4.7 / 0.5 | 0.552 | 46.1 | 9.0% | 109 |
| a0 raw, exact old (no depth mask) | 3 | 0.8 / 99.1 / 0.1 | 94.8 / 4.7 / 0.5 | 0.491 | 28.6 | 3.2% | 79 |
| a raw, depth>=10 m | 2 | 0.9 / 99.0 / 0.1 | 96.3 / 3.2 / 0.5 | 0.561 | 45.7 | 8.9% | 107 |
| a raw, depth>=10 m | 3 | 0.9 / 99.0 / 0.1 | 96.3 / 3.2 / 0.5 | 0.511 | 28.7 | 3.3% | 77 |
| b log10Chl+StandardScaler | 2 | 37.0 / 38.9 / 24.1 | 61.5 / 21.0 / 17.4 | 0.372 | 55.7 | 23.4% | 120 |
| b log10Chl+StandardScaler | 3 | 26.3 / 23.4 / 50.3 | 61.5 / 21.0 / 17.4 | 0.355 | 45.1 | 14.7% | 120 |
| c Yeo-Johnson (standardize) | 2 | 37.4 / 36.6 / 26.0 | 62.1 / 21.2 / 16.7 | 0.374 | 56.1 | 24.8% | 120 |
| c Yeo-Johnson (standardize) | 3 | 31.3 / 33.1 / 35.6 | 62.1 / 21.2 / 16.7 | 0.307 | 43.5 | 12.5% | 119 |

Raw feature variance (1M sample, no depth mask): log10 Chl 0.228, SST 3.939 degC^2, SSL 0.0257 m^2, so SST is 94.0% of the total raw variance (old notebook printed stds 0.476 / 1.983 / 0.160 on v3).
Old v3 output of `ssl_kmeans_no_norm.ipynb`: centroids [log10Chl, SST, SSL] = [-0.379, 24.51, 0.276], [-0.801, 29.15, 0.413], [-0.567, 26.88, 0.375], PCA PC1 94.8%. That gives a squared-separation share of 99.1% on SST.

**Verdict:** the ~97% along-SST separation **reproduces only without scaling** (about 99% on v4, 99.1% from the old v3 centroids). It comes from units:
SST variance (~4 degC^2) is about 17x log10 Chl and about 150x SSL, so unscaled Euclidean K-means is effectively 1-D K-means on SST,
and PC1 is just SST. Once features are put on a common scale (StandardScaler or Yeo-Johnson), separation is spread roughly evenly across the three features.
In every variant the cold cluster is mostly a winter/season cluster, not a May-Sep zone. The ~97% number should not be presented as evidence that
SST is what defines the upwelling zone; at most it is a statement about unscaled features.

## 7. Provenance

nasa-murep = this repo (new). nasa-murep-local = `F:\Programming\GitHub\nasa-murep-local` (old, v3-era).

| Item | Where it came from |
|---|---|
| Data | `E:\satdata\Custom\yucatan_sst_chlor_cdomslope_ssl_bathy_2010-2019_v4.pkl` (built by nasa-murep `yucatan/data/preprocessing/`) |
| CDOM formulas (finding: CDOM is a function of Rrs443/Rrs547; no negatives) | nasa-murep `yucatan/data/preprocessing/pickler_utils.py` (`calculate_cdom`, `calculate_cdom_slopes`), `pickler_config.py` (coefficients, MIN/MAX_CDOM = 0/1) |
| Isolation forest classification | nasa-murep `experiments/visualization/bayth_classification.py` (`_classify`, imported unchanged by `redo/iforest.py`); model `experiments/isolation_forest/models/iforest_pipeline5.joblib` (recipe in `isolation_forest/train.py`); surrogate RF settings from `isolation_forest/run.py` |
| pipeline4 vs pipeline5 finding | nasa-murep `yucatan/figs/bayth_preclassified.pkl` (git-ignored) vs models 4/5; check in `redo/iforest.py` |
| Scatterplot per-day / 95th pct / -20 m finding | nasa-murep `experiments/visualization/scatterplots/common.py` (current main) |
| Violin shallow-pixel finding | nasa-murep `experiments/visualization/violins/all_years.py` (current main + uncommitted edit) |
| **-1 sigma** (`redo/sigma.py`) | **New reimplementation, not a port.** nasa-murep has no SST-threshold analysis. The rule (monthly-mean SST < that month's spatial mean - 1 sd, deep pixels) was the rule set for this redo. The old -1 sigma in nasa-murep-local `cdom_sst/experiments/w_hist/histograms_no_fitting.ipynb` / `fit_curves.ipynb` is different: a **week-of-year** threshold (mean - 1 sd of all SST values in that calendar week, all years and pixels pooled, v3, no depth mask), used for histogram shading and a "% below" figure. `fit_curves_peak_finding.ipynb` uses yet another rule (Gaussian-peak C1 - 1.5 degC). |
| **K-means** (`redo/kmeans.py`) | **New reimplementation.** Modeled on nasa-murep-local `cdom_sst/experiments/kmeans/ssl_kmeans.ipynb` (features, k=3, StandardScaler, random_state/n_init) but fitted on monthly-mean maps with a depth mask. It does not apply the SSL blur and is not an exact replica. `redo/kmeans_compare.py` (section 6) is the exact replication of `ssl_kmeans_no_norm.ipynb` / `ssl_kmeans.ipynb`. |
| **DBSCAN** (`redo/dbscan.py`) | New reimplementation; eps/min_samples from nasa-murep-local `cdom_sst/experiments/clustering/ssl_dbscan.ipynb` (0.28/50) and `dbscan_clustering.ipynb` (0.27/40, pooled reference). |
| Stats, comparison, figures | New (`redo/stats.py`, `report.py`, `figures.py`); y-axis rounding reuses nasa-murep `experiments/visualization/plot_style.py`. |

## 8. Violin / map model check (correction to section 5, item 9; state as of 2026-11-07, resolved 2026-11-08, see section 10)

- At runtime, `violins/all_years.py`, `yearly_class/IF_bayth_*.py` and `raw/bayth.py` (main, including the uncommitted edits) all call
  `load_preclassified()` -> `yucatan/figs/bayth_preclassified.pkl`. None of them classify or load a model. The uncommitted edits only change a
  legend label, an axis label and land/coastline zorder.
- `generate_all_figures.py` does **not** regenerate the pickle. Only running `bayth_classification.py` directly does, and that would use pipeline5.
- The pickle (mtime 2026-10-08 17:46) reproduces **exactly** with `iforest_pipeline4.joblib` (0/120 month-years differ). With pipeline5, 95/120 differ.
  The pickle predates `bayth_classification.py` (added in commit 5a3a5d4, 2026-10-16). It was made by the earlier map scripts
  (`raw_all_years/IF_bayth_*.py`, `raw/bayth.py`), whose `MODEL_PATH` was pipeline4.
- **Makai is right that the violins were fixed.** Before 5a3a5d4, `violins/all_years.py` classified **per day** with **pipeline5**, the
  **95th** percentile and **-20 m**, while the maps used monthly pipeline4 / 90th / -10 m. That was the "violin months with data where maps show none" bug.
  5a3a5d4 switched the violins to the shared pickle, so violins and maps now agree with each other.
- My earlier statement needs narrowing: the violins do **not** currently mismatch the maps. What remains is that (i) the shared pickle, and so
  every current violin and map, is pipeline4 while all code says pipeline5 (zone IoU p4 vs p5 about 0.75-0.83; coverage 2.5% vs 2.2%), and
  (ii) `scatterplots/common.py` still uses the old per-day pipeline5 / 95th / -20 m rule, so the scatter plots do not match the violins or maps.
  Fix: pick one model, regenerate the pickle (`python bayth_classification.py`), and move the scatterplots onto the pickle.

## 10. 2026-11-08: standardized on iforest_pipeline5 (single shared classification)

**What was done.** The old pickle was copied to `figs/_backup_pipeline4_2026-11-08/bayth_preclassified_pipeline4.pkl`, together with all
pre-existing figures. Then `python experiments/visualization/bayth_classification.py` (repo `.venv`, scikit-learn 1.7.2) regenerated
`figs/bayth_preclassified.pkl` with pipeline5. `scatterplots/common.py` was rewritten to read that pickle instead of classifying each day.
`generate_all_figures.py` was rerun (exit 0; 88 PNGs regenerated in figs/bayth, histograms, raw, scatterplots, violins). 24 PNGs that no
current script produces (`figs/yearly_class/*_IF_anom|class.png`, `figs/raw/anom.png`, `figs/raw/class.png`, written by scripts deleted in
5a3a5d4 and therefore stale pipeline4 outputs) were moved, not deleted, to `figs/_stale_not_produced_by_current_scripts/`. Check:
`redo/iforest.py` zones vs the new pickle: **0 of 120 month-years differ** (pipeline4: 95 differ). The redo zones were already pipeline5,
so the redo statistics and figures are unchanged.

**Audit of every script that labels pixels as upwelling** (all identical: model `iforest_pipeline5.joblib`; monthly scoring of the
month-year mean of same-day-median SST/SSL anomalies; score >= 90th percentile of that month's scores; standardized SST anomaly < -0.5;
standardized SSL anomaly < 1.0; depth >= 10 m (bathymetry <= -10 m)):

| Script | Source of zone labels | Settings |
|---|---|---|
| experiments/visualization/bayth_classification.py | computes them (writes figs/bayth_preclassified.pkl) | the definition above |
| experiments/visualization/violins/all_years.py | reads the pickle | inherits |
| experiments/visualization/yearly_class/IF_bayth_all_years.py, IF_bayth_across_years.py | read the pickle | inherits |
| experiments/visualization/raw/bayth.py | reads the pickle | inherits |
| experiments/visualization/scatterplots/common.py (+ pooled.py, yearly.py) | reads the pickle (changed 2026-11-08; was per-day, 95th percentile, -20 m) | inherits |
| experiments/visualization/scatterplots/monthly.py | no zone (calendar May-Sep vs Oct-Apr only) | n/a |
| experiments/visualization/raw/main.py, histograms/weekly.py | no zone (raw maps / histograms) | n/a |
| experiments/redo/iforest.py | recomputes via `bayth_classification._classify` (same code, pipeline5); verified identical to the pickle (0/120 differ) | identical |
| experiments/redo/stats.py, figures.py, corr_diff.py, report.py | read redo/iforest.py zones | identical |
| experiments/isolation_forest/run.py | exploratory single-window script (80th percentile, no depth mask); not used for any figure | different, not a figure path |

Remaining population difference (not classification): the main-figure scripts count every non-zone pixel in the box, including pixels
shallower than 10 m and pixels without a monthly class, as "outside". The redo excludes those from both groups. Zone membership is identical.

**-1 sigma rule actually implemented (redo/sigma.py).** For each month-year (y, m), let T_ym(p) be the mean of the daily MUR L4 SST
(`analysed_sst`, v4 pickle) at pixel p over all days of that month. Let D = {p : GEBCO depth >= 10 m and T_ym(p) finite}. Then
mu_ym = mean over D of T_ym(p), sd_ym = population sd (ddof 0) over D of T_ym(p), and p is in the zone if p in D and T_ym(p) < mu_ym - sd_ym.
It is a spatial threshold within each monthly-mean map, applied per month-year. Flagged: **12.87%** of deep pixel-months.

Paper wording vs rules (numbers on v4 from `redo/sigma_variants.py`):
- Results 2, "-1 sigma threshold relative to the weekly mean SST", matches the OLD notebook rule
  (nasa-murep-local `cdom_sst/experiments/w_hist/histograms_no_fitting.ipynb`, cells 2/7/8): one climatological SST map per ISO
  week-of-year (mean of all days in that week across all years); threshold = that map's spatial mean - 1 sd (ddof 1) over all valid pixels,
  no depth mask. The notebook printed 13.20% on v3; on v4 it gives **13.21%** (12.21% if restricted to depth >= 10 m).
- Methods, "the coldest -1 sigma tail of the monthly SST distribution (~13% of pixels)": the ~13% figure comes from the weekly-climatology
  rule above, so "monthly" is inaccurate for the old analysis. It does describe the redo rule (monthly, per month-year: 12.87% with
  depth >= 10 m, 13.48% without; monthly climatology with depth >= 10 m: 12.08%).
- Suggested text if the redo rule is used: "pixels whose monthly-mean SST was more than one standard deviation below that month's spatial
  mean SST (depth >= 10 m; 12.9% of pixel-months)". If the old weekly figures are kept, the Methods should say weekly climatology.

**'v4' vs pipeline4.** 'v4' is the data pickle version (`yucatan_sst_chlor_cdomslope_ssl_bathy_2010-2019_v4.pkl`). It is unrelated
to `iforest_pipeline4.joblib`. Everything here uses the v4 data; IF results use pipeline5.

**a_CDOM(412) and slope retrieval / clipping** (`data/preprocessing/pickler_utils.py`, coefficients `pickler_config.py`):
ln a_CDOM(412) = -3.070 - 1.285 ln Rrs443 + 1.107 ln Rrs547, then a = exp(ln a). So a is always > 0 and can never be negative. Pixels
with Rrs443 <= 0 or Rrs547 <= 0 have no log and are dropped (NaN) before binning. Values outside [MIN_CDOM, MAX_CDOM] = [0, 1] are
set to NaN (dropped), not clamped. S275-295 = exp(-3.258 + 0.336 ln Rrs443 - 0.279 ln Rrs547) and
S300-600 = exp(-3.640 + 0.186 ln Rrs443 - 0.146 ln Rrs547) are also always positive and are not range-checked.
Raw-pixel counts over all 5,364 AQUA MODIS L2 OC files 2010-2019 in the pickler box (`redo/cdom_clip_check.py`, no flag masking,
same as the pickler):

| Quantity | Count | % |
|---|---|---|
| pixels with finite Rrs443 and Rrs547 | 140,071,740 | 100 |
| dropped because Rrs443 <= 0 or Rrs547 <= 0 | 178,825 (Rrs443 <= 0: 163,308; Rrs547 <= 0: 72,400) | 0.128 |
| a_CDOM computed | 139,892,915 | |
| a_CDOM < 0 | 0 | 0 |
| a_CDOM > 1 (dropped by the clip) | 165,937 | 0.119 of computed |
| raw a_CDOM range | 7.3e-6 to 8978 m^-1 | |
| raw S275-295 range / S300-600 range (no clip) | 0.0015-0.331 / 0.0043-0.077 nm^-1 | |

So the clip removes only implausible values (up to ~9000 m^-1, from near-zero Rrs443) on 0.12% of raw pixels, and negative a_CDOM
cannot occur with this log-space MLR. Negative values could only come from a different (linear) algorithm or the older v3 data. Minor
inconsistency: those 0.12% of pixels lose a_CDOM but keep their slopes, because the slopes are not range-checked.
