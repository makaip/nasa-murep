"""Only the statistics the 2026-11-10 draft asks for -> outputs/redo/paper_stats_requested.md.

Reads outputs/redo/paper_stats.json (written by paper_stats.py); no new computation. The full
tables, other scopes and the methods checks stay in paper_stats.md as supporting material.
"""

import json

import common as c

S = json.load(open(c.OUT_DIR / "paper_stats.json"))
MN = c.MONTH_NAMES
VARS = [("a412", "a_CDOM(412) (m^-1)", "{:.3f}"), ("s275", "S275-295 (nm^-1)", "{:.4f}"),
        ("s300", "S300-600 (nm^-1)", "{:.4f}"), ("log10_chl", "Chl-a (mg m^-3)", "{:.2f}")]


def val(key, x, f):
    """Chl is stored as log10; show it in mg m^-3."""
    return f.format(10 ** x if key == "log10_chl" else x)


def violin_table(L, T, title, months):
    for key, lab, f in VARS:
        L += [f"**{title}: {lab}**", "",
              "| Month | n in | median in [IQR] | median out (a / b) | diff in - out (a / b) [95% CI, b] |",
              "|---|---|---|---|---|"]
        for m in months:
            r = T[key][str(m)] if str(m) in T[key] else T[key][m]
            name = MN[m - 1] if isinstance(m, int) else m
            if not r["in"]["n"]:
                L.append(f"| {name} | 0 | no class | {val(key, r['out_a']['median'], f)} / {val(key, r['out_b']['median'], f)} | - |")
                continue
            iq = f"{val(key, r['in']['median'], f)} [{val(key, r['in']['q1'], f)}, {val(key, r['in']['q3'], f)}]"
            if key == "log10_chl":
                da = 10 ** r["in"]["median"] / 10 ** r["out_a"]["median"]
                db = 10 ** r["in"]["median"] / 10 ** r["out_b"]["median"]
                diff = f"x{da:.1f} / x{db:.1f} (ratio)"
            else:
                d = r["diff_out_b"]
                ci = d.get("median_diff_ci95")
                diff = (f"{r['diff_out_a']['median_diff_in_minus_out']:+.4f} / {d['median_diff_in_minus_out']:+.4f}"
                        + (f" [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else ""))
            L.append(f"| {name} | {r['in']['n']:,} | {iq} | {val(key, r['out_a']['median'], f)} / "
                     f"{val(key, r['out_b']['median'], f)} | {diff} |")
        L.append("")


def main():
    sg, iv2018, ivp = S["violins_sigma_pooled"], S["violins_iforest_2018"], S["violins_iforest_pooled"]
    sh = S["sigma_share"]
    L = ["# Statistics requested in the 2026-11-10 draft", "",
         "Source: `yucatan/experiments/redo/paper_stats.py` -> `paper_stats.json` (full tables and checks in "
         "`paper_stats.md`). Setup: v4 data 2010-2019; IF = pipeline5 monthly zones (90th pct, SST z < -0.5, "
         "SSL z < 1.0, depth >= 10 m); -1 sigma = monthly rule (redo/sigma.py). n = pixel-days.", "",
         "Outside: **(a)** all pixels not in the class (incl. water < 10 m; what the main figures use) / "
         "**(b)** deep (>= 10 m) non-upwelling only. 95% CIs are block bootstraps (years for one calendar month, "
         "month-years for May-Sep), B = 2000.", ""]

    L += ["## 1. Share of pixels classified by -1 sigma (Abstract, Sec. 2, 3.2, Figs 6-7)", "",
          f"- Week-of-year climatology rule (used for Fig. 6 and the 2018 maps in Fig. 7): "
          f"**{sh['old_weekly_climatology_nodepth']:.2f}%** of pixels on v4 (13.20% on v3).",
          f"- Monthly rule (redo): **{sh['redo_monthly_rule_pct_of_classified_deep_pixel_months']:.2f}%** of "
          f"{sh['n_classified_pixel_months']:,} deep pixel-months.", ""]

    L += ["## 2. -1 sigma violins, pooled 2010-2019 (Sec. 3.4, Fig. 12)", ""]
    violin_table(L, sg, "-1 sigma", list(range(1, 13)) + ["May-Sep"])
    a, s3 = sg["a412"], sg["s300"]
    gaps = sorted(((abs(s3[str(m) if str(m) in s3 else m]["diff_out_b"]["median_diff_in_minus_out"]), MN[m - 1])
                   for m in range(1, 13)), reverse=True)
    ms = a["May-Sep"]
    L += ["Check of the draft sentences:",
          f"- \"median absorption inside ... about 0.10 to 0.15 per meter\": not for -1 sigma. May-Sep monthly medians "
          f"inside are {min(a[str(m) if str(m) in a else m]['in']['median'] for m in range(5, 10)):.3f}-"
          f"{max(a[str(m) if str(m) in a else m]['in']['median'] for m in range(5, 10)):.3f} m^-1 "
          f"(pooled May-Sep {ms['in']['median']:.3f}); only Dec and Jan reach about 0.10. (0.10-0.15 fits the IF class, Sec. 3 below.)",
          f"- \"non-upwelling ... below about 0.05\": yes for May-Sep ({ms['out_a']['median']:.3f} / {ms['out_b']['median']:.3f} pooled); "
          f"Jan (a) is {a[str(1) if '1' in a else 1]['out_a']['median']:.3f}.",
          f"- \"gap in S300-600 largest in June to August\": largest in {gaps[0][1]} and {gaps[1][1]}, then {gaps[2][1]}, "
          f"{gaps[3][1]}; \"largest in July-August\" is accurate.", ""]

    L += ["## 3. IF violins (Sec. 3.4, Figs 13-16)", "", "2018 by month (February has no class):", ""]
    violin_table(L, iv2018, "IF 2018", list(range(1, 13)) + ["May-Sep"])
    L += ["Pooled 2010-2019, May-Sep:", ""]
    violin_table(L, ivp, "IF 2010-2019", ["May-Sep"])
    p, y = ivp, iv2018
    chl = lambda T, g: 10 ** T["log10_chl"]["May-Sep"][g]["median"]
    L += ["Suggested sentence:", "",
          f"> Inside the IF class, median a_CDOM(412) in May-September was {p['a412']['May-Sep']['in']['median']:.2f} m^-1 "
          f"(IQR {p['a412']['May-Sep']['in']['q1']:.2f}-{p['a412']['May-Sep']['in']['q3']:.2f}; n = {p['a412']['May-Sep']['in']['n']:,} "
          f"pixel-days, 2010-2019) compared with {p['a412']['May-Sep']['out_b']['median']:.3f} m^-1 outside, with lower spectral slopes "
          f"(S275-295 {p['s275']['May-Sep']['in']['median']:.3f} vs {p['s275']['May-Sep']['out_b']['median']:.3f} nm^-1; "
          f"S300-600 {p['s300']['May-Sep']['in']['median']:.3f} vs {p['s300']['May-Sep']['out_b']['median']:.3f} nm^-1) and higher "
          f"chlorophyll ({chl(p, 'in'):.2f} vs {chl(p, 'out_b'):.2f} mg m^-3). In 2018 the May-September medians were "
          f"{y['a412']['May-Sep']['in']['median']:.3f} vs {y['a412']['May-Sep']['out_b']['median']:.3f} m^-1 "
          f"(n = {y['a412']['May-Sep']['in']['n']:,}).", "",
          "(Outside values here use convention (b); with (a) they change by <= 0.001 m^-1 and 0.0005 nm^-1.)", ""]

    L += ["## 4. a_CDOM(412) vs Chl-a correlations (Sec. 3.5, Figs 17-20)", "",
          "Pixel-days with a_CDOM >= 10^-3.5 and Chl > 0. Pearson on log10 values; Spearman on raw values (rank-based). "
          "CI: paired month-year block bootstrap of r_in - r_out.", "",
          "| Method | Scope | n in / out (a) / out (b) | Spearman in / out (a) / out (b) | Spearman diff (a); (b) [95% CI] | Pearson in / out (a) / out (b) | Pearson diff (a); (b) [95% CI] |",
          "|---|---|---|---|---|---|---|"]
    for meth, lab in (("iforest", "IF"), ("sigma", "-1 sigma")):
        for sc, r in S["corr_a412_chl"][meth].items():
            cells = [lab, sc, f"{r['in']['n']:,} / {r['out_a']['n']:,} / {r['out_b']['n']:,}"]
            for k in ("spearman", "pearson_log10"):
                cells.append(f"{r['in'][k]:.3f} / {r['out_a'][k]:.3f} / {r['out_b'][k]:.3f}")
                parts = []
                for o in ("out_a", "out_b"):
                    d = r["diff_" + o][k]
                    parts.append(f"{d['diff']:+.3f}" + (f" [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]" if "ci95" in d else ""))
                cells.append("; ".join(parts))
            L.append("| " + " | ".join(cells) + " |")
    ms_ = S["corr_a412_chl"]["iforest"]["May-Sep (Fig 19)"]
    L += ["", f"Draft check: the values the draft labels Pearson (0.92 / 0.97, diff -0.056, CI -0.073 to -0.040, n = 50 "
          f"month-years) are the IF May-Sep **Spearman** values (here {ms_['in']['spearman']:.3f} / {ms_['out_b']['spearman']:.3f}, "
          f"diff {ms_['diff_out_b']['spearman']['diff']:+.3f} [{ms_['diff_out_b']['spearman']['ci95'][0]:+.3f}, "
          f"{ms_['diff_out_b']['spearman']['ci95'][1]:+.3f}], {ms_['in']['n_month_years_ge30']} month-years). "
          "The draft's 'Spearman 0.946 / 0.980' are the IF all-months Spearman values. The -1 sigma 0.973 / 0.975 are "
          "all-months Spearman with outside (b).", ""]

    z = S["if_zone_counts"]
    L += ["## 5. IF class by calendar month (Table 2, Sec. 3.3)", "",
          f"Percent of the {z['deep_pixels']:,} deep (>= 10 m) pixels.", "",
          "| Month | years with class | years | mean area, all 10 years | climatology class |", "|---|---|---|---|---|"]
    for m in range(1, 13):
        r = z["by_calendar_month"][str(m)]
        cl = z["climatology"][str(m)]
        L.append(f"| {MN[m - 1]} | {r['n_years_with_class']} | {', '.join(map(str, r['years']))} | "
                 f"{r['mean_px_all_years']:,.0f} px ({r['mean_pct_deep_all_years']:.2f}%) | {'none' if cl == 0 else f'{cl:,} px'} |")
    L += ["", "Draft check: March has 8 years (2016 is missing from the draft), October has 9 (no class in 2012), "
          "2012 has no class in January **or October**, 2018 has a class in every month except February; climatology has "
          "no class in Jan, Feb, Mar, Nov, Dec (as in the draft).", ""]

    km = S["kmeans_sq_sep_share_pct"]
    rv = S["kmeans_raw_variance_share_pct"]["a_raw"]
    L += ["## 6. K-means centroid-separation shares (Sec. 2, Table 1, S1)", "",
          "Share of squared centroid separation per feature (Chl / SST / SSL, %):", "",
          f"- Yeo-Johnson, k = 2: {km['c_yeojohnson']['2']['log10_chl']} / {km['c_yeojohnson']['2']['sst']} / {km['c_yeojohnson']['2']['ssl']} (matches draft)",
          f"- Yeo-Johnson, k = 3: {km['c_yeojohnson']['3']['log10_chl']} / {km['c_yeojohnson']['3']['sst']} / {km['c_yeojohnson']['3']['ssl']} (matches draft)",
          f"- Unstandardized: SST takes {km['a_raw']['2']['sst']}% of the separation (k = 2 and 3) and {rv['sst']}% of the raw "
          "variance; neither is the draft's 97%.", ""]

    sl, dc = S["slope"], S["depth_cutoff_2018"]
    cs = sl["climatology_class_union"]
    L += ["## 7. Bathymetric slope and depth cutoffs (Sec. 2)", "",
          f"- Slope within the climatological IF class (GEBCO on the ~{sl['grid_spacing_km_approx'][0]:.1f} x "
          f"{sl['grid_spacing_km_approx'][1]:.1f} km grid): max {cs['max_deg']:.2f} deg, 95th percentile {cs['p95_deg']:.2f} deg, "
          f"{cs['pct_below_1deg']:.0f}% of pixels below 1 deg. \"Stays below about 1 degree\" holds.", "",
          "2018 IF class area (pixels) by depth cutoff:", "",
          "| Month | >= 5 m | >= 10 m | >= 20 m |", "|---|---|---|---|"]
    for m in range(1, 13):
        r = dc[str(m)]
        L.append(f"| {MN[m - 1]} | {r['5m']:,} | {r['10m']:,} | {r['20m']:,} |")
    L.append("")
    (c.OUT_DIR / "paper_stats_requested.md").write_text("\n".join(L), encoding="utf-8")
    print("written")


if __name__ == "__main__":
    main()
