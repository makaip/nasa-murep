"""Method comparison + assemble outputs/redo/results.json and results.md."""

import json

import numpy as np

import common as c

METHODS = ["sigma", "iforest", "kmeans", "kmeans_month", "dbscan"]
STAT_METHODS = ["sigma", "iforest", "kmeans", "kmeans_month"]
KEYS = [(y, m) for y in c.YEARS for m in c.MONTHS]
SUMMER = {5, 6, 7, 8, 9}
VAR_NAMES = {"a412": "a_CDOM(412) [1/m]", "s275": "S275-295 [1/nm]", "s300": "S300-600 [1/nm]",
             "log10_chl": "log10 Chl-a [mg/m3]"}


def load(name):
    return json.load(open(c.CACHE_DIR / name))


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else None


def coverage(zones):
    by_month, clim = zones
    pct = {k: 100.0 * (g == 1).sum() / max((g >= 0).sum(), 1) for k, g in by_month.items()}
    n_zone = {k: int((g == 1).sum()) for k, g in by_month.items()}
    tot_in = sum(n_zone.values())
    tot_cls = sum(int((g >= 0).sum()) for g in by_month.values())
    summer_in = sum(v for k, v in n_zone.items() if k[1] in SUMMER)
    return {
        "pct_overall_pixel_months": 100.0 * tot_in / tot_cls,
        "pct_by_calendar_month_mean_over_years": {m: float(np.mean([pct[(y, m)] for y in c.YEARS])) for m in c.MONTHS},
        "pct_by_month_year": {f"{y}-{m:02d}": round(pct[(y, m)], 3) for y, m in KEYS},
        "frac_zone_pixels_in_May_Sep": summer_in / tot_in if tot_in else None,
        "n_month_years_with_zone_ge1px": sum(v > 0 for v in n_zone.values()),
        "n_month_years_with_zone_ge1pct": sum(pct[k] >= 1.0 for k in KEYS),
        "n_month_years_with_zone_by_calendar_month": {m: sum(n_zone[(y, m)] > 0 for y in c.YEARS) for m in c.MONTHS},
        "clim_pct_by_month": {m: 100.0 * float((clim[m] == 1).sum()) / max(int((clim[m] >= 0).sum()), 1) for m in c.MONTHS},
        "n_clim_months_with_zone": sum(bool((clim[m] == 1).any()) for m in c.MONTHS),
    }


def pair_iou(a, b):
    za, ca = c.load_zones(a)
    zb, cb = c.load_zones(b)
    vals = {f"{y}-{m:02d}": iou(za[(y, m)] == 1, zb[(y, m)] == 1) for y, m in KEYS}
    v = np.array([x for x in vals.values() if x is not None])
    clim = {m: iou(ca[m] == 1, cb[m] == 1) for m in c.MONTHS}
    bym = {m: (float(np.mean([vals[f"{y}-{m:02d}"] for y in c.YEARS if vals[f"{y}-{m:02d}"] is not None]))
               if any(vals[f"{y}-{m:02d}"] is not None for y in c.YEARS) else None) for m in c.MONTHS}
    # share of b's zone pixels that are inside a's zone (containment)
    inter = sum(int(((za[k] == 1) & (zb[k] == 1)).sum()) for k in KEYS)
    nb = sum(int((zb[k] == 1).sum()) for k in KEYS)
    return {"by_month_year": vals, "mean": float(v.mean()), "median": float(np.median(v)),
            "n_defined": int(len(v)), "by_calendar_month_mean": bym, "climatology": clim,
            f"frac_{b}_zone_inside_{a}_zone": inter / nb if nb else None}


def fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        if x != 0 and (abs(x) < 10 ** -nd or abs(x) >= 1e5):
            return f"{x:.2e}"
        return f"{x:.{nd}f}"
    return str(x)


def pfmt(p):
    if p is None:
        return "-"
    return "<1e-300" if p == 0 else f"{p:.1e}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(fmt(x) if not isinstance(x, str) else x for x in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    res = {"setup": {}, "coverage": {}, "kmeans": None, "iforest": None, "dbscan": None,
           "stats": {}, "comparison": {}}
    stacks, meta = c.load_stacks()
    deep = c.deep_mask(meta)
    res["setup"] = {
        "data": str(c.DATA_PATH), "model": str(c.MODEL_PATH),
        "n_days": int(len(meta["dates"])), "first_day": str(meta["dates"][0].date()),
        "last_day": str(meta["dates"][-1].date()), "grid": list(deep.shape),
        "n_deep_pixels": int(deep.sum()), "depth_threshold_m": 10,
        "valid_fraction_daily": {v: float(np.isfinite(stacks[v]).mean()) for v in c.VARS},
    }
    for m in METHODS + ["iforest_p4"]:
        res["coverage"][m] = coverage(c.load_zones(m))
    res["kmeans"] = load("kmeans_info.json")
    ifo = load("iforest_info.json")
    imp = ifo.pop("surrogate_importances")
    arr = np.array([[d["anom__sst_anom"], d["anom__ssl_anom"]] for d in imp])
    ifo["surrogate_importance_summary"] = {
        "n_month_years": len(arr),
        "sst_anom_mean": float(arr[:, 0].mean()), "sst_anom_sd": float(arr[:, 0].std()),
        "ssl_anom_mean": float(arr[:, 1].mean()), "ssl_anom_sd": float(arr[:, 1].std()),
        "sst_anom_min": float(arr[:, 0].min()), "sst_anom_max": float(arr[:, 0].max()),
        "by_calendar_month_sst_mean": {m: float(np.mean([d["anom__sst_anom"] for d in imp if d["month"] == m]))
                                       for m in c.MONTHS}}
    ifo["surrogate_importances"] = imp
    p4 = load("iforest_p4_info.json")
    ifo["pipeline4_check"] = {"months_mismatched_vs_preclassified_pickle": len(p4["preclassified_pickle_check"]["months_mismatched"]),
                              "pipeline5_months_mismatched_vs_pickle": len(ifo["preclassified_pickle_check"]["months_mismatched"])}
    res["iforest"] = ifo
    db = load("dbscan_info.json")
    pm = db["per_month"]
    ncl = np.array([d.get("n_clusters", 0) for d in pm])
    noise = np.array([d.get("noise_fraction", np.nan) for d in pm])
    ious = np.array([x for x in db["consecutive_month_zone_iou"] if x is not None and np.isfinite(x)])
    db["summary"] = {
        "n_clusters_distribution": {int(k): int((ncl == k).sum()) for k in np.unique(ncl)},
        "n_clusters_mean": float(ncl.mean()), "n_clusters_sd": float(ncl.std()),
        "noise_fraction_mean": float(np.nanmean(noise)), "noise_fraction_median": float(np.nanmedian(noise)),
        "noise_fraction_range": [float(np.nanmin(noise)), float(np.nanmax(noise))],
        "noise_fraction_month_to_month_abs_change_mean": float(np.nanmean(np.abs(np.diff(noise)))),
        "n_clusters_changes_between_consecutive_months": int((np.diff(ncl) != 0).sum()),
        "month_years_with_separate_cold_cluster": int(sum(d.get("zone") is not None for d in pm)),
        "consecutive_zone_iou_mean_where_defined": float(ious.mean()) if len(ious) else None,
        "n_consecutive_pairs_iou_defined": int(len(ious)),
    }
    res["dbscan"] = db
    for m in STAT_METHODS:
        res["stats"][m] = load(f"stats_{m}.json")
    res["comparison"] = {
        "iou_sigma_vs_iforest": pair_iou("sigma", "iforest"),
        "iou_kmeans_month_vs_iforest": pair_iou("kmeans_month", "iforest"),
        "iou_kmeans_vs_iforest": pair_iou("kmeans", "iforest"),
        "iou_iforest_p5_vs_p4": pair_iou("iforest_p4", "iforest"),
    }
    for extra in ("kmeans_compare", "corr_diff"):
        f = c.CACHE_DIR / f"{extra}.json"
        if f.exists():
            res[extra] = json.load(open(f))
    json.dump(res, open(c.OUT_DIR / "results.json", "w"), indent=1, default=float)
    write_md(res)
    print("report done")


def write_md(r):
    L = []
    s = r["setup"]
    cov = r["coverage"]
    L.append("# Yucatan upwelling / CDOM redo analysis - results\n")
    L.append("Generated by `yucatan/experiments/redo/` (run `run_all.py`). All numbers in `results.json`.\n")
    L.append("## 0. Data and common setup\n")
    L.append(f"- Data: `{s['data']}` (v4). Variables: sst, ssl, chlorophyll, cdom_412, cdom_slope_275_295, "
             f"cdom_slope_300_600 (+ `*_n_points` scalars); metadata has lat/lon centres+edges, bbox, bathymetry (GEBCO). "
             f"DataFrame 3653 days x 12 columns, 2010-01-01 .. 2020-01-01; trimmed to 2010-2019 -> "
             f"{s['n_days']} days ({s['first_day']} .. {s['last_day']}), grid {s['grid'][0]}x{s['grid'][1]}.")
    vf = s["valid_fraction_daily"]
    L.append("- Daily valid-pixel fraction (whole grid): " + ", ".join(f"{k} {100*v:.1f}%" for k, v in vf.items()) + ".")
    L.append(f"- Depth mask: GEBCO bathymetry <= -10 m -> {s['n_deep_pixels']:,} of 62,500 pixels. The "
             "Yucatan bbox used by the existing figure scripts contains all of these, so the stats domain is simply all deep pixels.")
    L.append("- One zone map per month-year (from monthly-mean fields) + 12 climatology maps. Daily pixels are labelled by their "
             "own month-year map; shallow / unclassified pixels are excluded from both inside and outside.")
    L.append(f"- IF model: `{r['iforest']['model']}` (n_estimators=200, contamination=auto, random_state=42, "
             f"Yeo-Johnson on [sst_anom, ssl_anom]) - matches train.py, so it was reused (no retrain). Run in the repo `.venv` "
             "(scikit-learn 1.7.2, the version the model was saved with).")
    L.append("- a_CDOM(412): v4 contains **no negative values** (min 7e-6 m^-1). The MLR output is exp(ln a) and preprocessing "
             "keeps only 0 <= a <= 1 (`pickler_config.MIN_CDOM/MAX_CDOM`), so 'keep negatives' is a no-op here.\n")

    L.append("## 1. Zone coverage (% of classified deep pixels inside the zone)\n")
    ms = ["sigma", "iforest", "kmeans", "kmeans_month", "dbscan"]
    hdr = ["Month"] + ms
    rows = [[c.MONTH_NAMES[m - 1]] + [cov[x]["pct_by_calendar_month_mean_over_years"][m] for x in ms] for m in c.MONTHS]
    rows.append(["**All (pixel-months)**"] + [cov[x]["pct_overall_pixel_months"] for x in ms])
    L.append("Mean over 2010-2019 of the per-month-year percentage.\n")
    L.append(table(hdr, rows) + "\n")
    rows = [["Month-years with a zone (>=1 px) / 120"] + [cov[x]["n_month_years_with_zone_ge1px"] for x in ms],
            ["Month-years with a zone (>=1% of px) / 120"] + [cov[x]["n_month_years_with_zone_ge1pct"] for x in ms],
            ["Climatology months with a zone / 12"] + [cov[x]["n_clim_months_with_zone"] for x in ms],
            ["Share of zone pixel-months in May-Sep"] + [cov[x]["frac_zone_pixels_in_May_Sep"] for x in ms]]
    L.append(table(["Metric"] + ms, rows) + "\n")
    L.append("Month-years with a zone, by calendar month (out of 10 years):\n")
    rows = [[c.MONTH_NAMES[m - 1]] + [cov[x]["n_month_years_with_zone_by_calendar_month"][m] for x in ms] for m in c.MONTHS]
    L.append(table(hdr, rows) + "\n")
    L.append("Climatology-map zone coverage (%):\n")
    rows = [[c.MONTH_NAMES[m - 1]] + [cov[x]["clim_pct_by_month"][m] for x in ms] for m in c.MONTHS]
    L.append(table(hdr, rows) + "\n")

    k = r["kmeans"]
    L.append("## 2a. K-means (standardized SST, log10 Chl-a, SSL)\n")
    L.append("Old notebooks (`ssl_kmeans*.ipynb`) used k=3; k=2 also fitted. **Global** = one model fitted on 1M pixels sampled "
             f"from all {k['global']['n_pooled_pixel_months']:,} deep pixel-months, global standardization (as the old notebooks). "
             "**Per month** = standardized and fitted separately per month-year map. Zone = coldest-SST centroid.\n")
    pv = k["global"]["pca_explained_variance_ratio"]
    L.append(f"PCA (global, standardized): PC1 {100*pv[0]:.1f}%, PC2 {100*pv[1]:.1f}%, PC3 {100*pv[2]:.1f}%. "
             f"Loadings PC1 [SST, log10Chl, SSL] = {np.round(k['global']['pca_components'][0], 3).tolist()}. "
             f"Per-month mean: " + ", ".join(f"{100*x:.1f}%" for x in k['per_month']['pca_explained_variance_ratio_mean']) + ".\n")
    rows = []
    for kk in ("2", "3"):
        g = k["global"]["by_k"][kk]
        for lab, key in (("pairwise", "pairwise_sq_sep_share"), ("zone vs others", "zone_vs_others_sq_sep_share"),
                         ("between-cluster SS", "between_ss_share")):
            rows.append([f"global k={kk}", lab] + [100 * g[key][f] for f in ("sst", "log10_chl", "ssl")])
        p = k["per_month"]["by_k"][kk]
        for lab, key in (("pairwise", "pairwise_sq_sep_share"), ("zone vs others", "zone_vs_others_sq_sep_share")):
            rows.append([f"per-month k={kk} (mean [min-max])", lab] +
                        [f"{100*p[key][f]['mean']:.1f} [{100*p[key][f]['min']:.1f}-{100*p[key][f]['max']:.1f}]"
                         for f in ("sst", "log10_chl", "ssl")])
    L.append("Share (%) of squared centroid separation along each feature:\n")
    L.append(table(["Model", "Measure", "SST", "log10 Chl", "SSL"], rows) + "\n")
    g3 = k["global"]["by_k"]["3"]
    L.append("Global k=3 centroids (physical units SST degC, log10 Chl, SSL m) and cluster fractions: " +
             "; ".join(f"[{', '.join(f'{v:.2f}' for v in cc)}] {100*fr:.1f}%" for cc, fr in zip(g3["centers_phys"], g3["cluster_fraction"])) + ".\n")

    ifo = r["iforest"]
    si = ifo["surrogate_importance_summary"]
    L.append("## 2c. Isolation forest\n")
    L.append(f"Surrogate RF (100 trees, depth 5) predicting the IF score from transformed features, fitted per month-year "
             f"(n={si['n_month_years']}): SST anomaly importance {100*si['sst_anom_mean']:.1f}% +/- {100*si['sst_anom_sd']:.1f}% "
             f"(range {100*si['sst_anom_min']:.1f}-{100*si['sst_anom_max']:.1f}%), SSL anomaly {100*si['ssl_anom_mean']:.1f}% "
             f"+/- {100*si['ssl_anom_sd']:.1f}%. By calendar month (SST %): " +
             ", ".join(f"{c.MONTH_NAMES[int(m)-1]} {100*v:.0f}" for m, v in si["by_calendar_month_sst_mean"].items()) + ".\n")
    pc = r["comparison"]["iou_iforest_p5_vs_p4"]
    L.append(f"**Model mismatch found:** `figs/bayth_preclassified.pkl` (used by the existing violin and map figures) reproduces "
             f"exactly with `iforest_pipeline4.joblib` ({ifo['pipeline4_check']['months_mismatched_vs_preclassified_pickle']} "
             f"month-years differ) but not with the configured `iforest_pipeline5.joblib` "
             f"({ifo['pipeline4_check']['pipeline5_months_mismatched_vs_pickle']} of 120 month-years differ). Same recipe, different fit. "
             f"pipeline5 vs pipeline4 zone IoU: mean {pc['mean']:.3f}, median {pc['median']:.3f}; overall zone coverage "
             f"{cov['iforest']['pct_overall_pixel_months']:.2f}% (p5) vs {cov['iforest_p4']['pct_overall_pixel_months']:.2f}% (p4). "
             "Results here use pipeline5.\n")

    d = r["dbscan"]["summary"]
    L.append("## 2d. DBSCAN (appendix)\n")
    L.append(f"eps=0.28, min_samples=50 (from `ssl_dbscan.ipynb`), per month-year, within-month standardized [SST, log10 Chl, SSL]. "
             f"Number of clusters per month-year: {d['n_clusters_distribution']} (mean {d['n_clusters_mean']:.2f} +/- {d['n_clusters_sd']:.2f}); "
             f"changes between consecutive months: {d['n_clusters_changes_between_consecutive_months']}/119. "
             f"Noise fraction: mean {100*d['noise_fraction_mean']:.1f}%, median {100*d['noise_fraction_median']:.1f}%, "
             f"range {100*d['noise_fraction_range'][0]:.1f}-{100*d['noise_fraction_range'][1]:.1f}%, mean month-to-month change "
             f"{100*d['noise_fraction_month_to_month_abs_change_mean']:.1f} pp. Month-years with a separate cold cluster (not the dominant one): "
             f"{d['month_years_with_separate_cold_cluster']}/120; consecutive-month zone IoU (where defined, n={d['n_consecutive_pairs_iou_defined']}): "
             f"{fmt(d['consecutive_zone_iou_mean_where_defined'])}.")
    pr = r["dbscan"]["pooled_reference"]
    L.append(f"Pooled reference (as `dbscan_clustering.ipynb`: 100k pooled pixel-months, eps=0.27, min_samples=40): "
             f"{pr['n_clusters']} clusters, noise {100*pr['noise_fraction']:.1f}%, largest cluster {100*pr['largest_cluster_fraction']:.1f}% of points.\n")

    L.append("## 3. Inside vs outside statistics (daily pixels, labelled by month-year zone map)\n")
    L.append("Cliff's delta > 0: inside larger. p = two-sided Mann-Whitney U (n are pixel-days, autocorrelated; "
             "p-values are tiny for nearly everything, so read the effect size).\n")
    for m in STAT_METHODS:
        st = r["stats"][m]
        L.append(f"### {m}\n")
        rows = []
        for var, nm in VAR_NAMES.items():
            p = st["pooled"][var]
            rows.append([nm, p["n_in"], p["n_out"], p["median_in"], p["median_out"], p["cliffs_delta"], pfmt(p["mwu_p"])])
        L.append("Pooled 2010-2019:\n")
        L.append(table(["Variable", "n in", "n out", "median in", "median out", "Cliff's d", "MWU p"], rows) + "\n")
        if m in ("sigma", "iforest"):
            L.append("By calendar month (median in / median out / Cliff's d; n in):\n")
            rows = []
            for mo in c.MONTHS:
                row = [c.MONTH_NAMES[mo - 1]]
                b = st["by_month"][str(mo)]
                for var in VAR_NAMES:
                    x = b[var]
                    row.append("-" if x["n_in"] < 2 else f"{fmt(x['median_in'],4)} / {fmt(x['median_out'],4)} / {fmt(x['cliffs_delta'],2)}")
                row.append(b["a412"]["n_in"])
                rows.append(row)
            L.append(table(["Month", "a412", "S275", "S300", "log10 Chl", "n in (a412)"], rows) + "\n")
        cr = st["pooled"]["corr_a412_chl"]
        rows = []
        for lab in ("inside", "outside"):
            x = cr[lab]
            if x.get("spearman") is None:
                rows.append([lab, x["n"]] + ["-"] * 7)
                continue
            rows.append([lab, x["n"], x["spearman"], x["pearson_loglog"], x["loglog_slope_log_a_on_log_chl"],
                         x.get("lower_half", {}).get("spearman"), x.get("upper_half", {}).get("spearman"),
                         x.get("lower_half", {}).get("loglog_slope"), x.get("upper_half", {}).get("loglog_slope")])
        L.append("a_CDOM(412) vs Chl-a, pooled (Spearman rho is identical on log-log axes; Pearson given on log10-log10):\n")
        L.append(table(["Zone", "n", "Spearman rho", "Pearson r log-log", "log-log slope",
                        "rho lower half a412", "rho upper half a412", "slope lower", "slope upper"], rows) + "\n")
        fz = cr["fisher_z_in_vs_out"]
        mt = st["monthyear_rho_test"]
        L.append(f"Inside vs outside rho: Fisher z = {fmt(fz['z'] if fz else None, 1)}, p = {pfmt(fz['p'] if fz else None)} (naive n). "
                 f"Paired over month-years (n={mt['n_month_years']}, >=30 px each side): median rho in {fmt(mt['median_rho_in'])}, "
                 f"out {fmt(mt['median_rho_out'])}, median diff {fmt(mt['median_diff'])}, rho_in < rho_out in "
                 f"{fmt(100*mt['frac_in_lt_out'] if mt['frac_in_lt_out'] is not None else None, 0)}% of month-years, Wilcoxon p = {pfmt(mt['wilcoxon_p'])}.\n")
        if m in ("sigma", "iforest"):
            rows = []
            for mo in c.MONTHS:
                b = st["by_month"][str(mo)]["corr_a412_chl"]
                fzm = b["fisher_z_in_vs_out"]
                rows.append([c.MONTH_NAMES[mo - 1], b["inside"]["n"], b["inside"].get("spearman"),
                             b["outside"].get("spearman"), pfmt(fzm["p"]) if fzm else "-"])
            L.append("Spearman rho by calendar month:\n")
            L.append(table(["Month", "n in", "rho in", "rho out", "Fisher-z p"], rows) + "\n")
    st = r["stats"]["sigma"]
    L.append("### Does the a_CDOM - Chl correlation taper at high a_CDOM? (all classified pixels, by a_CDOM quartile)\n")
    rows = [[q["quartile"], f"{q['a412_range'][0]:.4f}-{q['a412_range'][1]:.4f}", q["n"], q["spearman"]]
            for q in st["taper_all_pixels_quartiles"]]
    L.append(table(["a412 quartile", "range (1/m)", "n", "Spearman rho"], rows) + "\n")
    L.append("Note: rho within a restricted range is always lower than over the full range (range restriction), so compare "
             "quartiles with each other, not with the full-range rho.\n")

    L.append("## 4. Method comparison\n")
    io = r["comparison"]["iou_sigma_vs_iforest"]
    rows = [[c.MONTH_NAMES[m - 1], io["by_calendar_month_mean"][m], io["climatology"][m],
             r["comparison"]["iou_kmeans_month_vs_iforest"]["by_calendar_month_mean"][m]] for m in c.MONTHS]
    L.append(f"IoU of zone maps, -1 sigma vs IF: mean {io['mean']:.3f}, median {io['median']:.3f} over {io['n_defined']} month-years "
             f"where either has a zone; share of IF zone pixels that are inside the -1 sigma zone: "
             f"{100*io['frac_iforest_zone_inside_sigma_zone']:.1f}%.\n")
    L.append(table(["Month", "IoU sigma-IF (mean over years)", "IoU sigma-IF (climatology)", "IoU kmeans_month-IF"], rows) + "\n")
    L.append("May-Sep share and 'how often each approach worked': see section 1 tables.\n")
    static = (c.YUCATAN_DIR / "experiments" / "redo" / "report_static.md").read_text(encoding="utf-8")
    static = static.split("-->", 1)[1] if static.startswith("<!--") else static
    (c.OUT_DIR / "results.md").write_text("\n".join(L) + static, encoding="utf-8")


if __name__ == "__main__":
    main()
