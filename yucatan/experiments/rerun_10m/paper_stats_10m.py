"""Paper statistics with the 10 m depth cutoff applied in ALL cases -> outputs/rerun_10m/paper_stats.{md,json}.

Convention ("(b) deep-only", rerun 2026-11-13): pixels with GEBCO depth < 10 m (shared constant
experiments/depth_cutoff.py) are excluded from BOTH the inside and the outside group of every
comparison. inside = class 1 of the month-year zone map (always deep); outside = deep (>= 10 m)
pixels of that map that are not in the class.

Classes (unchanged from redo):
  * IF: pipeline5 monthly zones (score >= 90th pct, SST z < -0.5, SSL z < 1.0, depth >= 10 m),
    outputs/redo/cache/zones/iforest.npz (== figs/bayth_preclassified.pkl).
  * -1 sigma (monthly rule, redo/sigma.py): per month-year monthly-mean MUR SST < mean - 1 sd of the
    deep valid pixels of that map; only deep pixels are classified.
Daily pixels take the zone map of their own month-year. Statistics machinery (histogram medians,
Cliff's delta, month-year block bootstrap, correlations) is redo/paper_stats.py, unchanged.

For "what changed" the same run also computes the previous main-figure convention:
  * prev (a): outside = every pixel with data that is not inside (incl. < 10 m water)
  * prev -1 sigma figures (nasa-murep-local sst_upwelling_*.ipynb): monthly mean SST <= mean - 1 sd of
    ALL valid pixels of the whole grid, all pixels classified (no depth mask), outside = rest.

Run with the repo .venv:  .venv\\Scripts\\python.exe yucatan\\experiments\\rerun_10m\\paper_stats_10m.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "redo"))
import common as c  # noqa: E402
import paper_stats as ps  # noqa: E402
import sigma_variants as sv  # noqa: E402
import depth_cutoff  # noqa: E402  (on sys.path via common)

bc = ps.bc
warnings.filterwarnings("ignore")

OUT = c.YUCATAN_DIR / "outputs" / "rerun_10m"
OUT.mkdir(parents=True, exist_ok=True)
PREV = json.load(open(c.OUT_DIR / "paper_stats.json"))
MN = c.MONTH_NAMES


# ---------------------------------------------------------------------------
# extra zones / sensitivities
# ---------------------------------------------------------------------------
def sigma_nodepth_zones(stacks, meta):
    """Old -1 sigma figure rule: whole grid, threshold over all valid pixels, <= threshold."""
    out = {}
    for k, v in c.monthly_means(stacks, meta, "sst").items():
        ok = np.isfinite(v)
        if ok.sum() < 2:
            out[k] = c.zone_grid(np.zeros_like(ok), ok)
            continue
        thr = np.nanmean(v) - np.nanstd(v)
        out[k] = c.zone_grid(ok & (v <= thr), ok)
    return out


def if_deep_percentile_sensitivity(stacks, meta):
    """IF class if the 90th-pct score threshold and the SST/SSL z-scores were computed over deep pixels
    only (the classification currently uses all valid pixels for those, then applies depth >= 10 m)."""
    preprocess, iforest, features = bc._load_model(c.MODEL_PATH)
    deep = c.deep_mask(meta)
    cached, _ = c.load_zones("iforest")
    anom = {v: c.monthly_mean_anomaly(stacks, meta, v) for v in ("sst", "ssl")}
    import pandas as pd
    rows = {}
    for k in anom["sst"]:
        a_sst, a_ssl = anom["sst"][k], anom["ssl"][k]
        valid = np.isfinite(a_sst) & np.isfinite(a_ssl) & deep
        if valid.sum() < 10:
            rows[k] = (0, int((cached[k] == 1).sum()))
            continue
        X = pd.DataFrame({"sst_anom": a_sst[valid], "ssl_anom": a_ssl[valid]})[features]
        s = -iforest.decision_function(preprocess.transform(X))
        z = lambda a: (a[valid] - a[valid].mean()) / a[valid].std()
        up = (s >= np.percentile(s, bc.PERCENTILE_UPWELLING)) & (z(a_sst) < bc.SST_ANOM_Z_MAX) & (z(a_ssl) < bc.SSL_ANOM_Z_MAX)
        rows[k] = (int(up.sum()), int((cached[k] == 1).sum()))
    n_deep = int(deep.sum())
    by_m = {}
    for m in c.MONTHS:
        alt = [rows[(y, m)][0] for y in c.YEARS]
        cur = [rows[(y, m)][1] for y in c.YEARS]
        by_m[m] = {"years_with_class_current": sum(a > 0 for a in cur), "years_with_class_deep_pct": sum(a > 0 for a in alt),
                   "mean_pct_deep_current": 100 * np.mean(cur) / n_deep, "mean_pct_deep_deep_pct": 100 * np.mean(alt) / n_deep,
                   "area_2018_current": rows[(2018, m)][1], "area_2018_deep_pct": rows[(2018, m)][0]}
    return by_m


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
F = {"a412": "{:.4f}", "s275": "{:.5f}", "s300": "{:.5f}", "log10_chl": "{:.3f}"}
TITLE = {"a412": "a_CDOM(412) (m^-1)", "s275": "S275-295 (nm^-1)", "s300": "S300-600 (nm^-1)",
         "log10_chl": "Chl-a (log10 mg m^-3; medians also shown in mg m^-3)"}


def med(v, key):
    if not v["n"]:
        return "-"
    f = F[key]
    s = f"{f.format(v['median'])} [{f.format(v['q1'])}, {f.format(v['q3'])}]"
    return s + (f" ({10 ** v['median']:.2f})" if key == "log10_chl" else "")


def dif(d, key):
    ci = d.get("median_diff_ci95")
    s = f"{d['median_diff_in_minus_out']:+.4g}"
    if ci:
        s += f" [{ci[0]:+.4g}, {ci[1]:+.4g}]"
    if key == "log10_chl":
        s += f" (x{10 ** d['median_diff_in_minus_out']:.1f})"
    return s


def violin_md(L, T, prev_T, title, old_rule=None):
    """T: new tables (in vs out_b); prev_T: previous (redo json) tables -> prev (a) columns;
    old_rule: tables of the old no-depth -1 sigma rule (in vs out_a) or None."""
    for key, rows in T.items():
        L.append(f"\n**{title}: {TITLE[key]}**\n")
        hdr = "| Month | n in | median in [IQR] | n out | median out [IQR] | diff in-out [95% CI] | Cliff's delta | prev out (a) median | prev diff (a) |"
        if old_rule:
            hdr += " old fig rule: n in | median in | median out | diff |"
        L.append(hdr)
        L.append("|" + "---|" * (hdr.count("|") - 1))
        for r, v in rows.items():
            name = MN[int(r) - 1] if str(r).isdigit() else r
            d = v["diff_out_b"]
            pv = prev_T[key][str(r)]
            cells = [name, f"{v['in']['n']:,}", med(v["in"], key), f"{v['out_b']['n']:,}", med(v["out_b"], key),
                     dif(d, key) if v["in"]["n"] else "no class",
                     "-" if d["cliffs_delta_in_vs_out"] is None else f"{d['cliffs_delta_in_vs_out']:+.3f}",
                     F[key].format(pv["out_a"]["median"]) if pv["out_a"]["n"] else "-",
                     f"{pv['diff_out_a']['median_diff_in_minus_out']:+.4g}" if pv["in"]["n"] else "-"]
            if old_rule:
                o = old_rule[key][r]
                cells += [f"{o['in']['n']:,}", F[key].format(o["in"]["median"]) if o["in"]["n"] else "-",
                          F[key].format(o["out_a"]["median"]),
                          f"{o['diff_out_a']['median_diff_in_minus_out']:+.4g}" if o["in"]["n"] else "-"]
            L.append("| " + " | ".join(cells) + " |")


CORR_SCOPES = {"iforest": ["May-Sep (Fig 19)", "all months (Fig 18)", "Oct-Apr (Fig 20)", "May 2018 (Fig 17)"],
               "sigma": ["all months (Fig 18)", "May-Sep (Fig 19)"]}


# Figure numbers as listed in the rerun request (17 May 2018, 18 May-Sep, 19 Oct-Apr, 20 all seasons).
# redo/paper_stats.py labelled the scopes 18 = all months, 19 = May-Sep, 20 = Oct-Apr; check against the draft.
SCOPE_FIG = {"May 2018": "Fig 17", "May-Sep": "Fig 18", "Oct-Apr": "Fig 19", "all months": "Fig 20"}


def scope_name(sc):
    base = sc.split(" (")[0]
    return f"{base} ({SCOPE_FIG[base]})"


def corr_md(L, C, prev_C, old_rule=None):
    L.append("| Class | Scope | n in | Spearman in | Pearson log10 in | n out | Spearman out | Pearson log10 out "
             "| Spearman diff [95% CI] | Pearson diff [95% CI] | prev (a): Spearman out / diff | prev (a): Pearson out / diff |")
    L.append("|" + "---|" * 12)
    for meth, scopes in CORR_SCOPES.items():
        for sc in scopes:
            r, p = C[meth][sc], prev_C[meth][sc]
            i, o = r["in"], r.get("out_b", {})
            if "spearman" not in i:
                L.append(f"| {meth} | {scope_name(sc)} | {i.get('n', 0):,} | - | - | {o.get('n', 0):,} | | | | | | |")
                continue
            d = r.get("diff_out_b", {})
            fd = lambda k: (f"{d[k]['diff']:+.3f}" + (f" [{d[k]['ci95'][0]:+.3f}, {d[k]['ci95'][1]:+.3f}]" if "ci95" in d[k] else " (no CI: one month-year)")) if k in d else "-"
            pa, pd_ = p.get("out_a", {}), p.get("diff_out_a", {})
            fp = lambda k: f"{pa[k]:.3f} / {pd_[k]['diff']:+.3f}" if k in pa and k in pd_ else "-"
            L.append(f"| {meth} | {scope_name(sc)} | {i['n']:,} | {i['spearman']:.3f} | {i['pearson_log10']:.3f} | {o['n']:,} | {o['spearman']:.3f} "
                     f"| {o['pearson_log10']:.3f} | {fd('spearman')} | {fd('pearson_log10')} | {fp('spearman')} | {fp('pearson_log10')} |")
    if old_rule:
        L.append("")
        L.append("Old -1 sigma figure rule (no depth mask, outside = all not-inside) for reference: " + "; ".join(
            f"{scope_name(sc)}: Spearman in {old_rule[sc]['in']['spearman']:.3f} / out {old_rule[sc]['out_a']['spearman']:.3f}, "
            f"Pearson log10 in {old_rule[sc]['in']['pearson_log10']:.3f} / out {old_rule[sc]['out_a']['pearson_log10']:.3f}"
            for sc in CORR_SCOPES["sigma"]) + ".")


def same(a, b, tol=1e-9):
    if isinstance(a, dict):
        return all(same(a[k], b.get(k) if isinstance(b, dict) else None, tol) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol * max(1, abs(a))
    return a == b


def write_md(o):
    L = ["# Paper summary statistics, 10 m depth cutoff in all cases (rerun 2026-11-13)", "",
         "Generated by `yucatan/experiments/rerun_10m/paper_stats_10m.py` (statistics code: `redo/paper_stats.py`); "
         "full numbers in `paper_stats.json`. Previous numbers: `yucatan/outputs/redo/paper_stats.md`.", "",
         "**Convention:** pixels shallower than 10 m (GEBCO; constant `yucatan/experiments/depth_cutoff.py`) are excluded "
         "from BOTH groups. inside = pixels in the class (always >= 10 m); outside = deep (>= 10 m) pixels not in the class. "
         "This is the old \"(b)\" column. The \"prev (a)\" columns repeat the previous main-figure convention "
         "(outside = every pixel with data not in the class, incl. < 10 m) so the change can be read off directly.", "",
         "IF class: pipeline5 monthly zones (score >= 90th pct, SST z < -0.5, SSL z < 1.0, depth >= 10 m). "
         "-1 sigma class: monthly rule (per month-year monthly-mean MUR SST < spatial mean - 1 sd of the deep valid pixels; "
         "only deep pixels classified; redo/sigma.py). Daily pixels take the zone map of their own month-year. "
         f"n = pixel-days. Medians/IQR from {o['bins']}-bin histograms per month-year. CIs: block bootstrap, B = {o['bootstrap_B']} "
         "(blocks = years for one calendar month pooled 2010-2019; month-years for May-Sep pooled; a single month-year has no CI). "
         "Cliff's delta: inside vs outside.", ""]

    # 1 ------------------------------------------------------------------
    s = o["sigma_share"]
    L += ["## 1. Share of deep (>= 10 m) pixels in the -1 sigma class", "",
          f"- Weekly rule (week-of-year climatology: 52-53 maps of mean daily MUR SST over 2010-2019, threshold = mean - 1 sd of the "
          f"deep valid pixels, used for Fig. 6): **{s['old_weekly_climatology_depth10']:.2f}%** of deep pixel-weeks "
          f"(previously quoted 13.21% = same rule over the whole grid, no depth mask).",
          f"- Monthly rule (per month-year, deep pixels, used for Fig. 7, Fig. 12, S2-S5 and all -1 sigma statistics here): "
          f"**{s['redo_monthly_rule_pct_of_classified_deep_pixel_months']:.2f}%** of {s['n_classified_pixel_months']:,} deep pixel-months "
          f"(ddof = 1 variant: {s['redo_monthly_per_month_year_depth10']:.2f}%).",
          f"- For reference: monthly rule on the whole grid without depth mask (old Fig. 7/12 notebooks) "
          f"{s['redo_monthly_per_month_year_nodepth']:.2f}%; monthly climatology (12 maps), deep {s['monthly_climatology_depth10']:.2f}%.", ""]

    # 2 ------------------------------------------------------------------
    L += ["## 2. -1 sigma: inside vs outside, pooled 2010-2019 (Fig. 12), by calendar month and May-Sep", "",
          "Rows Jan-Dec give the monthly medians (May-Sep rows are the upwelling-season monthly medians); the May-Sep row is "
          "pooled. The S300-600 table's diff column is the S300-600 gap by month. \"old fig rule\" = the rule the draft's "
          "-1 sigma figures used (no depth mask, threshold over the whole grid, outside incl. shallow)."]
    violin_md(L, o["violins_sigma_pooled"], PREV["violins_sigma_pooled"], "-1 sigma, pooled 2010-2019",
              old_rule=o.get("violins_sigma_nodepth_pooled"))

    # 3 ------------------------------------------------------------------
    L += ["", "## 3. IF: inside vs outside", "",
          "### 3a. 2018 by month (Figs 13-16) and 2018 May-Sep", "",
          "A single month-year has no block CI; the 2018 May-Sep row has a CI over its 5 month-years."]
    violin_md(L, o["violins_iforest_2018"], PREV["violins_iforest_2018"], "IF, 2018")
    L += ["", "### 3b. Pooled 2010-2019 by calendar month and May-Sep"]
    violin_md(L, o["violins_iforest_pooled"], PREV["violins_iforest_pooled"], "IF, pooled 2010-2019")

    # 4 ------------------------------------------------------------------
    cl = o["corr_a412_clip"]
    L += ["", "## 4. a_CDOM(412) vs Chl-a correlations, inside vs outside (Figs 17-20)", "",
          f"Pixel-days with a_CDOM >= 10^-3.5 and Chl > 0 (clip drops {cl['n_dropped_all_pixels']:,} pixel-days on the whole grid, "
          f"{cl['n_dropped_deep_pixels']:,} of them deep). Spearman on raw values, Pearson on log10-log10. "
          "CI: paired month-year block bootstrap of r_in - r_out (diff = in - out).", ""]
    corr_md(L, o["corr_a412_chl"], PREV["corr_a412_chl"],
            old_rule=o["corr_a412_chl"].get("sigma_nodepth"))

    # 5 ------------------------------------------------------------------
    z = o["if_zone_counts"]
    L += ["", "## 5. IF class by calendar month (Table 2) and mean classified area", "",
          f"Deep (>= 10 m) pixels: {z['deep_pixels']:,}. Areas are % of deep pixels.", "",
          "| Month | years with class | years | mean area, all 10 yrs (px, % deep) | mean area, years with class (px, % deep) | climatology class (px) |",
          "|---|---|---|---|---|---|"]
    for m, r in z["by_calendar_month"].items():
        L.append(f"| {MN[int(m) - 1]} | {r['n_years_with_class']} | {', '.join(map(str, r['years']))} | "
                 f"{r['mean_px_all_years']:,.0f} ({r['mean_pct_deep_all_years']:.2f}%) | "
                 f"{r['mean_px_years_with_class']:,.0f} ({r['mean_pct_deep_years_with_class']:.2f}%) | {z['climatology'][m]:,} |")
    L += ["", "Months with a class, by year: " + "; ".join(
        f"{y}: {', '.join(MN[m - 1] for m in ms)}" for y, ms in z["years_with_class"].items()), ""]

    # 6 ------------------------------------------------------------------
    L += ["## 6. 2018 IF class area at 5 / 10 / 20 m cutoffs", "",
          "Score threshold and z-scores unchanged; only the depth mask differs.", "",
          "| Month | >= 5 m | >= 10 m | >= 20 m | 10 m = zones used |", "|---|---|---|---|---|"]
    for m, r in o["depth_cutoff_2018"].items():
        L.append(f"| {MN[int(m) - 1]} | {r['5m']:,} | {r['10m']:,} | {r['20m']:,} | {r['10m_matches_cache']} |")
    dp = next(iter(o["depth_cutoff_2018"].values()))["deep_px"]
    L += ["", f"Pixels per cutoff: 5 m {dp['5m']:,}, 10 m {dp['10m']:,}, 20 m {dp['20m']:,}.", ""]

    sens = o["if_deep_percentile_sensitivity"]
    L += ["Sensitivity (not used anywhere else): the IF 90th-percentile score threshold and the SST/SSL z-scores are computed over "
          "all valid pixels (incl. < 10 m) before the 10 m mask is applied. If they were computed over deep pixels only:", "",
          "| Month | years with class (current / deep-only pct) | mean area % deep (current / deep-only pct) | 2018 area px (current / deep-only pct) |",
          "|---|---|---|---|"]
    for m, r in sens.items():
        L.append(f"| {MN[int(m) - 1]} | {r['years_with_class_current']} / {r['years_with_class_deep_pct']} | "
                 f"{r['mean_pct_deep_current']:.2f} / {r['mean_pct_deep_deep_pct']:.2f} | {r['area_2018_current']:,} / {r['area_2018_deep_pct']:,} |")
    L.append("")

    # 7 ------------------------------------------------------------------
    sl = o["slope"]
    L += ["## 7. Bathymetric slope inside the climatological IF class", "",
          f"GEBCO on the 250 x 250 grid (~{sl['grid_spacing_km_approx'][0]:.2f} km E-W x {sl['grid_spacing_km_approx'][1]:.2f} km N-S).", "",
          "| Region | n px | max (deg) | 95th pct (deg) | median (deg) | % < 1 deg |", "|---|---|---|---|---|---|"]
    for k in ("climatology_class_union", "any_month_year_class", "all_deep_pixels"):
        r = sl[k]
        L.append(f"| {k} | {r['n_px']:,} | {r['max_deg']:.3f} | {r['p95_deg']:.3f} | {r['median_deg']:.3f} | {r['pct_below_1deg']:.1f} |")

    # 8 ------------------------------------------------------------------
    L += ["", "## 8. K-means: share of squared centroid separation per feature (%)", "",
          "redo/kmeans_compare.py (v4, depth >= 10 m except variant a0, which is the old no-depth setup kept for reference).", "",
          "| Variant | k | log10 Chl | SST | SSL |", "|---|---|---|---|---|"]
    for v, ks in o["kmeans_sq_sep_share_pct"].items():
        for k, sh in ks.items():
            L.append(f"| {v} | {k} | {sh['log10_chl']} | {sh['sst']} | {sh['ssl']} |")
    L += ["", "Raw (unscaled) feature variance share (%): " + "; ".join(
        f"{v}: Chl {d['log10_chl']}, SST {d['sst']}, SSL {d['ssl']}" for v, d in o["kmeans_raw_variance_share_pct"].items()), ""]

    # 9 ------------------------------------------------------------------
    L += ["## 9. What changed vs yucatan/outputs/redo/paper_stats.md", ""] + o["changes"]
    (OUT / "paper_stats.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def summarize_changes(o):
    ch = []
    unchanged = [k for k in ("if_zone_counts", "slope", "depth_cutoff_2018", "kmeans_sq_sep_share_pct", "kmeans_raw_variance_share_pct")
                 if same(json.loads(json.dumps(o[k], default=float)), PREV[k])]
    ch.append("- Unchanged (identical to the previous run): " + ", ".join(unchanged) +
              ". The IF and -1 sigma classes themselves already required depth >= 10 m, so Table 2, class areas, "
              "slope and depth-cutoff numbers and every **inside** median/IQR/n are unchanged.")
    # largest outside change
    big = []
    for tab in ("violins_sigma_pooled", "violins_iforest_pooled", "violins_iforest_2018"):
        for key, rows in o[tab].items():
            for r, v in rows.items():
                pv = PREV[tab][key][str(r)]
                if not pv["out_a"]["n"]:
                    continue
                d = v["out_b"]["median"] - pv["out_a"]["median"]
                rel = d / abs(pv["out_a"]["median"]) if key != "log10_chl" else 10 ** d - 1
                big.append((abs(rel), tab, key, r, pv["out_a"]["median"], v["out_b"]["median"]))
    big.sort(reverse=True)
    ch.append("- **Outside** medians change because < 10 m water (higher a_CDOM and Chl, lower slopes) leaves the outside group. "
              "Largest relative changes of an outside median:")
    for rel, tab, key, r, a, b in big[:8]:
        name = MN[int(r) - 1] if str(r).isdigit() else r
        ch.append(f"  - {tab} {key} {name}: {F[key].format(a)} -> {F[key].format(b)} ({100 * rel:.1f}%)")
    maysep = [x for x in big if x[3] == "May-Sep"]
    ch.append("  - May-Sep pooled rows: " + "; ".join(f"{t.replace('violins_', '')} {k} {F[k].format(a)} -> {F[k].format(b)}"
                                                        for _, t, k, r, a, b in maysep) + ".")
    # sign flips of the median difference
    flips = []
    for tab in ("violins_sigma_pooled", "violins_iforest_pooled", "violins_iforest_2018"):
        for key, rows in o[tab].items():
            for r, v in rows.items():
                pv = PREV[tab][key][str(r)]
                if v["in"]["n"] and pv["in"]["n"]:
                    a, b = pv["diff_out_a"]["median_diff_in_minus_out"], v["diff_out_b"]["median_diff_in_minus_out"]
                    if np.sign(a) != np.sign(b):
                        flips.append(f"{tab} {key} {r}: {a:+.4g} -> {b:+.4g}")
    ch.append("- Sign changes of an inside-minus-outside median difference: " + ("; ".join(flips) if flips else "none") + ".")
    # CI that now excludes / includes zero
    ci_ch = []
    for tab in ("violins_sigma_pooled", "violins_iforest_pooled"):
        for key, rows in o[tab].items():
            for r, v in rows.items():
                pv = PREV[tab][key][str(r)]
                cb, ca = v["diff_out_b"].get("median_diff_ci95"), pv["diff_out_a"].get("median_diff_ci95")
                if cb and ca and ((cb[0] > 0 or cb[1] < 0) != (ca[0] > 0 or ca[1] < 0)):
                    name = MN[int(r) - 1] if str(r).isdigit() else r
                    ci_ch.append(f"{tab.replace('violins_', '')} {key} {name}: prev (a) [{ca[0]:+.4g}, {ca[1]:+.4g}] -> "
                                 f"deep-only [{cb[0]:+.4g}, {cb[1]:+.4g}]")
    ch.append("- Rows whose 95% CI of the median difference excludes 0 under one convention but not the other: " +(", ".join(ci_ch) if ci_ch else "none") + ".")
    # correlations
    cc = []
    for meth, scopes in CORR_SCOPES.items():
        for sc in scopes:
            n, p = o["corr_a412_chl"][meth][sc], PREV["corr_a412_chl"][meth][sc]
            if "spearman" in n.get("out_b", {}) and "spearman" in p.get("out_a", {}):
                cc.append(f"{meth} {scope_name(sc)}: Spearman out {p['out_a']['spearman']:.3f} -> {n['out_b']['spearman']:.3f}, "
                          f"Pearson log10 out {p['out_a']['pearson_log10']:.3f} -> {n['out_b']['pearson_log10']:.3f}")
    ch.append("- Correlations: inside r unchanged; outside r (prev (a) -> deep-only): " + "; ".join(cc) + ".")
    s, ps_ = o["sigma_share"], PREV["sigma_share"]
    ch.append(f"- -1 sigma share: the number to quote is now the deep-pixel one: weekly rule {s['old_weekly_climatology_depth10']:.2f}% "
              f"(was quoted as {ps_['old_weekly_climatology_nodepth']:.2f}% without depth mask), monthly rule "
              f"{s['redo_monthly_rule_pct_of_classified_deep_pixel_months']:.2f}% (unchanged).")
    ch.append("- -1 sigma figures (Fig. 7, 12, S2-S5) previously used the no-depth monthly rule (whole-grid threshold, `<=`). "
              "They now use the deep monthly rule, so for -1 sigma the **inside** group of the figures changes too; "
              "the \"old fig rule\" columns in Sec. 2 show the old figure numbers.")
    return ch


def main():
    stacks, meta = c.load_stacks()
    print("sigma_variants ...", flush=True)
    sv.main()                                      # refresh cache/sigma_variants.json (weekly/monthly shares)
    extra = {"sigma_nodepth": sigma_nodepth_zones(stacks, meta)}
    out = ps.main(extra_zones=extra, write_markdown=False)
    out["setup"] = __doc__.split("\n\n")[1].replace("\n", " ")
    out["depth_cutoff_m"] = depth_cutoff.MIN_DEPTH_M
    print("IF deep-percentile sensitivity ...", flush=True)
    out["if_deep_percentile_sensitivity"] = if_deep_percentile_sensitivity(stacks, meta)
    out = json.loads(json.dumps(out, default=float))
    out["changes"] = summarize_changes(out)
    json.dump(out, open(OUT / "paper_stats.json", "w"), indent=1)
    write_md(out)
    print("paper_stats_10m done", flush=True)


if __name__ == "__main__":
    if "--md-only" in sys.argv:
        o = json.load(open(OUT / "paper_stats.json"))
        o["changes"] = summarize_changes(o)
        json.dump(o, open(OUT / "paper_stats.json", "w"), indent=1)
        write_md(o)
    else:
        main()
