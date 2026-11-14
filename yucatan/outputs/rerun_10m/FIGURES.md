# Paper figures, 10 m depth cutoff in all cases (rerun 2026-11-13)

All files: `figs/`, PNG at 300 dpi. Generator: `yucatan/experiments/rerun_10m/figures_10m.py` (one script; run with the repo .venv, `figures_10m.py all`, ~40 min). Numbering follows the 11-10 draft's figure list.

Convention: pixels shallower than 10 m (GEBCO, `yucatan/experiments/depth_cutoff.py`) are in neither the inside nor the outside group. Maps draw them light gray ("Shallow (< 10 m)") and land darker gray. Violins keep the legend wording "No data" (star = month with no class). Data: redo .npy caches of the v4 pickle; IF zones = `outputs/redo/cache/zones/iforest.npz` (identical to `figs/bayth_preclassified.pkl`); -1 sigma zones = `zones/sigma.npz` (redo/sigma.py, monthly rule over deep pixels).

| Paper figure | File | What | Function in figures_10m.py | Was made by |
|---|---|---|---|---|
| Fig. 6 | fig06.png | Weekly (week-of-year climatology, 2010-2019) SST histogram, deep pixels; bins <= weekly mean - 1 sd of deep pixels highlighted | fig06 | nasa-murep-local w_hist/histograms_no_fitting.ipynb (whole grid) |
| Fig. 7 | fig07.png | -1 sigma monthly-rule binary maps, Jan-Dec 2018 | fig07 | nasa-murep-local feature_space/sst_upwelling_video.ipynb (no depth mask) |
| Fig. 8 | fig08.png | IF class 2018 | fig08 / if_class_map | visualization/yearly_class/IF_bayth_all_years.py |
| Fig. 9 | fig09.png | IF anomaly score 2018 (copy of S71); cyan contour = class | score_map_figure | (not in this repo; isolation_forest notebooks) |
| Fig. 10 | fig10.png | IF class, 2010-2019 climatology | fig10 | visualization/yearly_class/IF_bayth_across_years.py |
| Fig. 12 | fig12.png | -1 sigma violins pooled 2010-2019 (a412, S275-295, S300-600, log10 Chl) | fig12 | nasa-murep-local feature_space/sst_upwelling_pooled.ipynb |
| Fig. 13 | fig13.png | IF violins 2018: a_CDOM(412) | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. 14 | fig14.png | IF violins 2018: S275-295 | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. 15 | fig15.png | IF violins 2018: S300-600 | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. 16 | fig16.png | IF violins 2018: log10 Chl-a | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. 17 | fig17.png | Chl vs a_CDOM(412), May 2018, inside vs outside IF class | per_year_and_pooled | nasa-murep-local chlor_vs_cdom_2018 notebooks |
| Fig. 18 | fig18.png | Chl vs a_CDOM(412), May-Sep 2010-2019, inside vs outside | per_year_and_pooled | visualization/scatterplots/pooled.py |
| Fig. 19 | fig19.png | Chl vs a_CDOM(412), Oct-Apr 2010-2019, inside vs outside | per_year_and_pooled | visualization/scatterplots/pooled.py |
| Fig. 20 | fig20.png | Chl vs a_CDOM(412), all months 2010-2019, inside vs outside | per_year_and_pooled | visualization/scatterplots/pooled.py |
| Fig. S2 | S02.png | -1 sigma violins by year (one panel per year 2010-2019): a_CDOM(412) | sigma_yearly | nasa-murep-local feature_space/sst_upwelling_all_years.ipynb |
| Fig. S3 | S03.png | same: S275-295 | sigma_yearly | same |
| Fig. S4 | S04.png | same: S300-600 | sigma_yearly | same |
| Fig. S5 | S05.png | same: log10 Chl-a | sigma_yearly | same |
| Fig. S6 | S06.png | 2010: IF class map | per_year_and_pooled | visualization/yearly_class/IF_bayth_all_years.py |
| Fig. S7 | S07.png | 2010: IF anomaly score map (cyan contour = class) | score_map_figure | isolation_forest notebooks |
| Fig. S8 | S08.png | 2010: IF violins a_CDOM(412) | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S9 | S09.png | 2010: IF violins S275-295 | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S10 | S10.png | 2010: IF violins S300-600 | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S11 | S11.png | 2010: IF violins log10 Chl-a | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S12 | S12.png | 2010: Chl vs a_CDOM(412): upwelling months (May-Sep) vs non-upwelling months (Oct-Apr), all deep pixels | per_year_and_pooled | visualization/scatterplots/monthly.py |
| Fig. S13 | S13.png | 2010: Chl vs a_CDOM(412): May-Sep, inside vs outside IF class | per_year_and_pooled | visualization/scatterplots/yearly.py |
| Fig. S14 | S14.png | 2011: IF class map | per_year_and_pooled | visualization/yearly_class/IF_bayth_all_years.py |
| Fig. S15 | S15.png | 2011: IF anomaly score map (cyan contour = class) | score_map_figure | isolation_forest notebooks |
| Fig. S16 | S16.png | 2011: IF violins a_CDOM(412) | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S17 | S17.png | 2011: IF violins S275-295 | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S18 | S18.png | 2011: IF violins S300-600 | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S19 | S19.png | 2011: IF violins log10 Chl-a | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S20 | S20.png | 2011: Chl vs a_CDOM(412): upwelling months (May-Sep) vs non-upwelling months (Oct-Apr), all deep pixels | per_year_and_pooled | visualization/scatterplots/monthly.py |
| Fig. S21 | S21.png | 2011: Chl vs a_CDOM(412): May-Sep, inside vs outside IF class | per_year_and_pooled | visualization/scatterplots/yearly.py |
| Fig. S22 | S22.png | 2012: IF class map | per_year_and_pooled | visualization/yearly_class/IF_bayth_all_years.py |
| Fig. S23 | S23.png | 2012: IF anomaly score map (cyan contour = class) | score_map_figure | isolation_forest notebooks |
| Fig. S24 | S24.png | 2012: IF violins a_CDOM(412) | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S25 | S25.png | 2012: IF violins S275-295 | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S26 | S26.png | 2012: IF violins S300-600 | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S27 | S27.png | 2012: IF violins log10 Chl-a | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S28 | S28.png | 2012: Chl vs a_CDOM(412): upwelling months (May-Sep) vs non-upwelling months (Oct-Apr), all deep pixels | per_year_and_pooled | visualization/scatterplots/monthly.py |
| Fig. S29 | S29.png | 2012: Chl vs a_CDOM(412): May-Sep, inside vs outside IF class | per_year_and_pooled | visualization/scatterplots/yearly.py |
| Fig. S30 | S30.png | 2013: IF class map | per_year_and_pooled | visualization/yearly_class/IF_bayth_all_years.py |
| Fig. S31 | S31.png | 2013: IF anomaly score map (cyan contour = class) | score_map_figure | isolation_forest notebooks |
| Fig. S32 | S32.png | 2013: IF violins a_CDOM(412) | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S33 | S33.png | 2013: IF violins S275-295 | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S34 | S34.png | 2013: IF violins S300-600 | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S35 | S35.png | 2013: IF violins log10 Chl-a | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S36 | S36.png | 2013: Chl vs a_CDOM(412): upwelling months (May-Sep) vs non-upwelling months (Oct-Apr), all deep pixels | per_year_and_pooled | visualization/scatterplots/monthly.py |
| Fig. S37 | S37.png | 2013: Chl vs a_CDOM(412): May-Sep, inside vs outside IF class | per_year_and_pooled | visualization/scatterplots/yearly.py |
| Fig. S38 | S38.png | 2014: IF class map | per_year_and_pooled | visualization/yearly_class/IF_bayth_all_years.py |
| Fig. S39 | S39.png | 2014: IF anomaly score map (cyan contour = class) | score_map_figure | isolation_forest notebooks |
| Fig. S40 | S40.png | 2014: IF violins a_CDOM(412) | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S41 | S41.png | 2014: IF violins S275-295 | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S42 | S42.png | 2014: IF violins S300-600 | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S43 | S43.png | 2014: IF violins log10 Chl-a | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S44 | S44.png | 2014: Chl vs a_CDOM(412): upwelling months (May-Sep) vs non-upwelling months (Oct-Apr), all deep pixels | per_year_and_pooled | visualization/scatterplots/monthly.py |
| Fig. S45 | S45.png | 2014: Chl vs a_CDOM(412): May-Sep, inside vs outside IF class | per_year_and_pooled | visualization/scatterplots/yearly.py |
| Fig. S46 | S46.png | 2015: IF class map | per_year_and_pooled | visualization/yearly_class/IF_bayth_all_years.py |
| Fig. S47 | S47.png | 2015: IF anomaly score map (cyan contour = class) | score_map_figure | isolation_forest notebooks |
| Fig. S48 | S48.png | 2015: IF violins a_CDOM(412) | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S49 | S49.png | 2015: IF violins S275-295 | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S50 | S50.png | 2015: IF violins S300-600 | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S51 | S51.png | 2015: IF violins log10 Chl-a | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S52 | S52.png | 2015: Chl vs a_CDOM(412): upwelling months (May-Sep) vs non-upwelling months (Oct-Apr), all deep pixels | per_year_and_pooled | visualization/scatterplots/monthly.py |
| Fig. S53 | S53.png | 2015: Chl vs a_CDOM(412): May-Sep, inside vs outside IF class | per_year_and_pooled | visualization/scatterplots/yearly.py |
| Fig. S54 | S54.png | 2016: IF class map | per_year_and_pooled | visualization/yearly_class/IF_bayth_all_years.py |
| Fig. S55 | S55.png | 2016: IF anomaly score map (cyan contour = class) | score_map_figure | isolation_forest notebooks |
| Fig. S56 | S56.png | 2016: IF violins a_CDOM(412) | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S57 | S57.png | 2016: IF violins S275-295 | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S58 | S58.png | 2016: IF violins S300-600 | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S59 | S59.png | 2016: IF violins log10 Chl-a | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S60 | S60.png | 2016: Chl vs a_CDOM(412): upwelling months (May-Sep) vs non-upwelling months (Oct-Apr), all deep pixels | per_year_and_pooled | visualization/scatterplots/monthly.py |
| Fig. S61 | S61.png | 2016: Chl vs a_CDOM(412): May-Sep, inside vs outside IF class | per_year_and_pooled | visualization/scatterplots/yearly.py |
| Fig. S62 | S62.png | 2017: IF class map | per_year_and_pooled | visualization/yearly_class/IF_bayth_all_years.py |
| Fig. S63 | S63.png | 2017: IF anomaly score map (cyan contour = class) | score_map_figure | isolation_forest notebooks |
| Fig. S64 | S64.png | 2017: IF violins a_CDOM(412) | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S65 | S65.png | 2017: IF violins S275-295 | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S66 | S66.png | 2017: IF violins S300-600 | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S67 | S67.png | 2017: IF violins log10 Chl-a | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S68 | S68.png | 2017: Chl vs a_CDOM(412): upwelling months (May-Sep) vs non-upwelling months (Oct-Apr), all deep pixels | per_year_and_pooled | visualization/scatterplots/monthly.py |
| Fig. S69 | S69.png | 2017: Chl vs a_CDOM(412): May-Sep, inside vs outside IF class | per_year_and_pooled | visualization/scatterplots/yearly.py |
| Fig. S70 | S70.png | 2018: IF class map | per_year_and_pooled | visualization/yearly_class/IF_bayth_all_years.py |
| Fig. S71 | S71.png | 2018: IF anomaly score map (cyan contour = class) | score_map_figure | isolation_forest notebooks |
| Fig. S72 | S72.png | 2018: IF violins a_CDOM(412) | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S73 | S73.png | 2018: IF violins S275-295 | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S74 | S74.png | 2018: IF violins S300-600 | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S75 | S75.png | 2018: IF violins log10 Chl-a | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S76 | S76.png | 2018: Chl vs a_CDOM(412): upwelling months (May-Sep) vs non-upwelling months (Oct-Apr), all deep pixels | per_year_and_pooled | visualization/scatterplots/monthly.py |
| Fig. S77 | S77.png | 2018: Chl vs a_CDOM(412): May-Sep, inside vs outside IF class | per_year_and_pooled | visualization/scatterplots/yearly.py |
| Fig. S78 | S78.png | 2019: IF class map | per_year_and_pooled | visualization/yearly_class/IF_bayth_all_years.py |
| Fig. S79 | S79.png | 2019: IF anomaly score map (cyan contour = class) | score_map_figure | isolation_forest notebooks |
| Fig. S80 | S80.png | 2019: IF violins a_CDOM(412) | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S81 | S81.png | 2019: IF violins S275-295 | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S82 | S82.png | 2019: IF violins S300-600 | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S83 | S83.png | 2019: IF violins log10 Chl-a | per_year_and_pooled | visualization/violins/all_years.py |
| Fig. S84 | S84.png | 2019: Chl vs a_CDOM(412): upwelling months (May-Sep) vs non-upwelling months (Oct-Apr), all deep pixels | per_year_and_pooled | visualization/scatterplots/monthly.py |
| Fig. S85 | S85.png | 2019: Chl vs a_CDOM(412): May-Sep, inside vs outside IF class | per_year_and_pooled | visualization/scatterplots/yearly.py |

Not regenerated (no inside/outside split, or not made from this repo): Fig. 1 (study-area map; the caption says the IF class is masked at 10 m), Figs 2-4 (raw monthly maps, `visualization/raw/main.py`), Fig. 5 and Fig. 11 (weekly density of all variables, `visualization/histograms/weekly.py`), Fig. S1 (clustering).

Notes:
- -1 sigma (Fig. 7, 12, S2-S5) now uses the monthly rule over deep pixels (threshold = mean - 1 sd of deep valid pixels, `<`). The draft's versions used the same monthly rule on the whole grid (threshold over all valid pixels, `<=`, shallow water classified), so their inside group changes as well as the outside.
- Violin KDEs use at most 200,000 randomly drawn values per violin (Fig. 12: 50,000, as the original notebook) after the 3-sd trim of the old scripts; exact medians are in `paper_stats.md`.
- Scatters: at most 50,000 points per panel, random draw (seed 42), as the old scripts.
- The original repo scripts (`visualization/violins/all_years.py`, `scatterplots/common.py`, `yearly_class/*.py`) were also changed to the deep-only convention but were not rerun; `yucatan/figs/` still holds the old PNGs.
