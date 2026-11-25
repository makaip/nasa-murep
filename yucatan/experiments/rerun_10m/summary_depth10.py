"""OLD vs NEW summary for the depth10-everywhere rerun -> outputs/redo/paper_stats_depth10.md.

Columns:
  OLD      outputs/redo/paper_stats.json, main convention of the 11-10 draft: inside vs outside incl. < 10 m water
           (out_a); IF gates over all valid pixels.
  11-13    outputs/rerun_10m/paper_stats.json (branch rerun-10m-cutoff): outside deep-only (out_b), but IF
           90th-pct threshold and z-scores still computed over all valid pixels incl. < 10 m.
  draft    value as quoted in the 11-13 draft (only where the coordinator supplied it).
  NEW      outputs/redo/depth10_stats/paper_stats.json: 10 m cutoff for inside, outside AND the IF gates.
"""

import json
import re
from pathlib import Path

Y = Path(__file__).resolve().parents[2]
OLD = json.load(open(Y / "outputs/redo/paper_stats.json"))
R13 = json.load(open(Y / "outputs/rerun_10m/paper_stats.json"))
NEW = json.load(open(Y / "outputs/redo/depth10_stats/paper_stats.json"))
SETS = [("OLD", OLD, "out_a", "diff_out_a"), ("11-13", R13, "out_b", "diff_out_b"), ("NEW", NEW, "out_b", "diff_out_b")]
MN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
F = {"a412": "{:.4f}", "s275": "{:.5f}", "s300": "{:.5f}", "log10_chl": "{:.3f}"}
VN = {"a412": "a_CDOM(412) m^-1", "s275": "S275-295 nm^-1", "s300": "S300-600 nm^-1", "log10_chl": "Chl-a log10 mg m^-3"}

# values quoted in the 11-13 draft (from the coordinator's note)
DRAFT = {
    ("sigma", "maysep", "a412", "out"): "0.025 [0.019-0.034]",
    ("sigma", "maysep", "a412", "nout"): "16.5e6",
    ("iforest", "maysep", "a412", "out"): "0.027",
    ("iforest", "maysep", "a412", "nout"): "19.4e6",
    ("iforest", "maysep", "a412", "diff"): "+0.079 [+0.072, +0.085]",
    ("sigma", "gap", "Jul"): "-0.0042", ("sigma", "gap", "Aug"): "-0.0042",
    ("iforest", "gap", "Jul"): "-0.0050", ("iforest", "gap", "Aug"): "-0.0050",
    "weekly": "12.2%",
}


def mq(v, key):
    if not v or not v.get("n"):
        return "-"
    f = F[key]
    return f"{f.format(v['median'])} [{f.format(v['q1'])}, {f.format(v['q3'])}]"


def dci(d):
    if not d or d.get("median_diff_in_minus_out") is None:
        return "-"
    s = f"{d['median_diff_in_minus_out']:+.4g}"
    ci = d.get("median_diff_ci95")
    return s + (f" [{ci[0]:+.4g}, {ci[1]:+.4g}]" if ci else "")


def table(L, head, rows):
    L.append("| " + " | ".join(head) + " |")
    L.append("|" + "---|" * len(head))
    for r in rows:
        L.append("| " + " | ".join(r) + " |")
    L.append("")


def violin_block(L, tab, meth, row, draft_key):
    """Rows: stat x convention for one table/row (e.g. May-Sep pooled)."""
    for key in ("a412", "s275", "s300", "log10_chl"):
        cells = {}
        for name, o, ok, dk in SETS:
            v = o[tab][key][str(row)]
            ins = v["in"]["n"]
            cells[name] = [f"{v['in']['n']:,}", mq(v["in"], key), f"{v[ok]['n']:,}", mq(v[ok], key),
                           dci(v[dk]) if ins else "no class",
                           "-" if not ins or v[dk]["cliffs_delta_in_vs_out"] is None else f"{v[dk]['cliffs_delta_in_vs_out']:+.3f}"]
        stats = ["n inside", "median inside [IQR]", "n outside", "median outside [IQR]", "diff in-out [95% CI]", "Cliff's delta"]
        dk_map = {"n outside": "nout", "median outside [IQR]": "out", "diff in-out [95% CI]": "diff"}
        rows = []
        for i, s in enumerate(stats):
            d = DRAFT.get((meth, draft_key, key, dk_map.get(s, "")), "") if draft_key else ""
            rows.append([f"{VN[key]}: {s}", cells["OLD"][i], cells["11-13"][i], d, cells["NEW"][i]])
        table(L, ["Statistic", "OLD", "11-13 rerun", "11-13 draft", "NEW"], rows)


def monthly_medians(L, tab, keys, meth=None, gap=False):
    for key in keys:
        rows = []
        for m in range(1, 13):
            r = [MN[m - 1]]
            for name, o, ok, dk in SETS:
                v = o[tab][key][str(m)]
                if gap:
                    r.append(f"{v[dk]['median_diff_in_minus_out']:+.4g}" if v["in"]["n"] else "no class")
                else:
                    fi = F[key].format(v["in"]["median"]) if v["in"]["n"] else "-"
                    r.append(f"{fi} / {F[key].format(v[ok]['median'])}")
            if gap:
                r.insert(3, DRAFT.get((meth, "gap", MN[m - 1]), ""))
            rows.append(r)
        if gap:
            table(L, ["Month", "OLD gap", "11-13 rerun gap", "11-13 draft", "NEW gap"], rows)
        else:
            L.append(f"*{VN[key]}: median inside / median outside*\n")
            table(L, ["Month", "OLD", "11-13 rerun", "NEW"], rows)


def corr(L):
    scopes = {"iforest": ["May-Sep", "all months", "Oct-Apr", "May 2018"], "sigma": ["all months", "May-Sep"]}
    fig = {"May 2018": "Fig 17", "May-Sep": "Fig 18", "Oct-Apr": "Fig 19", "all months": "Fig 20"}
    rows = []
    for meth, scs in scopes.items():
        for sc in scs:
            for stat, lab in (("spearman", "Spearman"), ("pearson_log10", "Pearson log10")):
                r = [f"{'IF' if meth == 'iforest' else '-1 sigma'} {sc} ({fig[sc]})", lab]
                for name, o, ok, dk in SETS:
                    k = next(k for k in o["corr_a412_chl"][meth] if k.split(" (")[0] == sc)
                    e = o["corr_a412_chl"][meth][k]
                    if stat not in e["in"]:
                        r.append("-")
                        continue
                    d = e[dk][stat]
                    ci = f" [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]" if "ci95" in d else " (no CI)"
                    r.append(f"in {e['in'][stat]:.3f} (n {e['in']['n']:,}) / out {e[ok][stat]:.3f} (n {e[ok]['n']:,}); "
                             f"diff {d['diff']:+.3f}{ci}")
                rows.append(r)
    table(L, ["Class, scope", "r", "OLD", "11-13 rerun", "NEW"], rows)


def zone_table(L):
    rows = []
    for m in range(1, 13):
        r = [MN[m - 1]]
        for o in (OLD, NEW):
            z = o["if_zone_counts"]["by_calendar_month"][str(m)]
            r += [f"{z['n_years_with_class']} ({', '.join(str(y) for y in z['years'])})",
                  f"{z['mean_pct_deep_all_years']:.2f}% / {z['mean_pct_deep_years_with_class']:.2f}%"]
        rows.append(r)
    table(L, ["Month", "OLD years with class", "OLD mean area % deep (all 10 yrs / yrs with class)",
              "NEW years with class", "NEW mean area % deep (all 10 yrs / yrs with class)"], rows)
    clo, cln = OLD["if_zone_counts"]["climatology"], NEW["if_zone_counts"]["climatology"]
    L.append("Climatology (Fig 10) class pixels by month, OLD -> NEW: " +
             ", ".join(f"{MN[int(m) - 1]} {clo[m]:,} -> {cln[m]:,}" for m in clo) + ".\n")


def depth_table(L):
    rows = []
    for m in range(1, 13):
        a, b = OLD["depth_cutoff_2018"][str(m)], NEW["depth_cutoff_2018"][str(m)]
        rows.append([MN[m - 1]] + [f"{a[k]:,} -> {b[k]:,}" for k in ("5m", "10m", "20m")])
    table(L, ["Month (2018)", ">= 5 m (OLD -> NEW)", ">= 10 m (OLD -> NEW)", ">= 20 m (OLD -> NEW)"], rows)
    dp = NEW["depth_cutoff_2018"]["1"]["deep_px"]
    L.append(f"Pixels per cutoff: 5 m {dp['5m']:,}, 10 m {dp['10m']:,}, 20 m {dp['20m']:,}. OLD: score threshold and "
             "z-scores over all valid pixels, then depth mask. NEW: threshold and z-scores over the pixels at or below "
             "the cutoff.\n")


def figure_list(L, changed):
    src = (Y / "outputs/rerun_10m/FIGURES.md").read_text(encoding="utf-8")
    tree = json.load(open(Path(__file__).with_name("flat_to_tree.json"))) if Path(__file__).with_name("flat_to_tree.json").exists() else {}
    rows = []
    for line in src.splitlines():
        m = re.match(r"\| (Fig\. S?\d+) \| (\S+\.png) \| ([^|]+) \|", line)
        if m:
            fig, f, what = m.groups()
            rows.append([fig, f"`yucatan/outputs/rerun_10m/figs/{tree[f]}`" if f in tree else f"`yucatan/outputs/redo/figs_depth10/{f}`", what.strip(), changed.get(f, "?")])
    table(L, ["Paper figure", "File", "What", "Changed vs 11-13 rerun figure"], rows)


def main(changed=None):
    changed = changed or {}
    L = ["# Paper stats with the 10 m cutoff in all cases, incl. the IF gates (branch depth10-everywhere, 2026-11-24)", "",
         "Generated by `yucatan/experiments/rerun_10m/summary_depth10.py`. Full tables: "
         "`yucatan/outputs/redo/depth10_stats/paper_stats.md` / `.json`.", "",
         "**Columns**", "",
         "- **OLD**: `outputs/redo/paper_stats.json` (11-10 draft). Outside group = every pixel with data not in the class, "
         "including water shallower than 10 m. IF gates over all valid pixels.",
         "- **11-13 rerun**: branch `rerun-10m-cutoff` (`outputs/rerun_10m/`). Outside = deep (>= 10 m) pixels only, "
         "but the IF 90th-percentile score threshold and the SST/SSL z-scores were still computed over all valid pixels "
         "(incl. < 10 m) before the 10 m mask.",
         "- **11-13 draft**: value as quoted in the 11-13 draft (only where supplied).",
         "- **NEW**: 10 m cutoff everywhere. Inside and outside are both >= 10 m, and the IF percentile and z-scores "
         "are computed over the >= 10 m valid pixels only (`bayth_classification.GATES_DEEP_ONLY = True`). "
         "-1 sigma: weekly and monthly rules both threshold over deep valid pixels and classify deep pixels only "
         "(already so in the 11-13 rerun, so -1 sigma NEW = 11-13).", "",
         "n = pixel-days. Medians/IQR from 4000-bin histograms; CIs block bootstrap B = 2000 (month-year blocks for May-Sep "
         "pooled, year blocks for one calendar month pooled; a single month-year has no CI). Cliff's delta inside vs outside.", "",
         "**Key points**", "",
         "- The values the 11-13 draft already quotes still hold after rounding, except the IF May-Sep a_CDOM(412) difference: "
         "+0.079 [+0.072, +0.085] becomes **+0.078 [+0.071, +0.084]**. -1 sigma numbers are unchanged from the 11-13 rerun.",
         "- What is new is the IF gates over >= 10 m. The IF class grows slightly: May-Sep inside n is 866,086 -> 921,964 (+6%) and the "
         "climatological class (Fig 10) gains a small November patch (0 -> 97 px). The largest single change is April 2018: "
         "817 -> 666 px, with the inside a_CDOM(412) median going from 0.0710 to 0.0807. Table 2 (which years have a class) does not change.",
         "- No inside-minus-outside difference changes sign. All May-Sep CIs still exclude 0. The correlation differences move by <= 0.003.", ""]

    s_o, s_n = OLD["sigma_share"], NEW["sigma_share"]
    L += ["## 1. Share of pixels in the -1 sigma class", ""]
    table(L, ["Rule", "OLD", "11-13 rerun", "11-13 draft", "NEW"], [
        ["Weekly (week-of-year climatology, Fig 6)", f"{s_o['old_weekly_climatology_nodepth']:.2f}% (whole grid, no depth mask)",
         f"{R13['sigma_share']['old_weekly_climatology_depth10']:.2f}% of deep pixel-weeks", DRAFT["weekly"],
         f"{s_n['old_weekly_climatology_depth10']:.2f}% of deep pixel-weeks"],
        ["Monthly (per month-year, Fig 7/12, all -1 sigma stats)",
         f"{s_o['redo_monthly_rule_pct_of_classified_deep_pixel_months']:.2f}% (deep)",
         f"{R13['sigma_share']['redo_monthly_rule_pct_of_classified_deep_pixel_months']:.2f}%", "",
         f"{s_n['redo_monthly_rule_pct_of_classified_deep_pixel_months']:.2f}% of {s_n['n_classified_pixel_months']:,} deep pixel-months"]])

    L += ["## 2. -1 sigma class, pooled 2010-2019", "", "### May-Sep pooled", ""]
    violin_block(L, "violins_sigma_pooled", "sigma", "May-Sep", "maysep")
    L += ["### Monthly a_CDOM(412) medians", ""]
    monthly_medians(L, "violins_sigma_pooled", ["a412"])
    L += ["### S300-600 gap (median inside - median outside) by month", ""]
    monthly_medians(L, "violins_sigma_pooled", ["s300"], "sigma", gap=True)

    L += ["## 3. IF class, pooled 2010-2019", "", "### May-Sep pooled", ""]
    violin_block(L, "violins_iforest_pooled", "iforest", "May-Sep", "maysep")
    L += ["### Monthly a_CDOM(412) medians", ""]
    monthly_medians(L, "violins_iforest_pooled", ["a412"])
    L += ["### S300-600 gap by month", ""]
    monthly_medians(L, "violins_iforest_pooled", ["s300"], "iforest", gap=True)

    L += ["## 4. IF class, 2018 only", "", "### 2018 May-Sep", ""]
    violin_block(L, "violins_iforest_2018", "iforest", "May-Sep", None)
    L += ["### 2018 monthly medians (Figs 13-16)", ""]
    monthly_medians(L, "violins_iforest_2018", ["a412", "s275", "s300", "log10_chl"])
    L += ["### 2018 S300-600 gap by month", ""]
    monthly_medians(L, "violins_iforest_2018", ["s300"], "iforest_2018", gap=True)

    L += ["## 5. a_CDOM(412) vs Chl-a correlations (Figs 17-20)", "",
          "Spearman on raw values, Pearson on log10-log10; diff = r_in - r_out with paired month-year block-bootstrap 95% CI.", ""]
    corr(L)

    L += ["## 6. Table 2: years with an IF class each month, and mean classified area", "",
          "(11-13 rerun = OLD here: the 11-13 rerun did not change the IF zones.) Area = % of the 41,144 deep pixels.", ""]
    zone_table(L)

    L += ["## 7. 2018 IF class area (pixels) at 5 / 10 / 20 m", ""]
    depth_table(L)

    L += ["## 8. Regenerated figures", "",
          "All redrawn on this branch from the NEW zones into `yucatan/outputs/redo/figs_depth10/` (flat paper names), then copied into `yucatan/outputs/rerun_10m/figs/` in the original `yucatan/figs/` layout and names (`experiments/rerun_10m/to_original_layout.py`; map in `figs/MAP.json`; raw/ and histograms/ redrawn with `YUC_DEPTH10=1`). The paths below are the new ones. "
          "(generator `yucatan/experiments/rerun_10m/figures_10m.py all`, 300 dpi). In every violin and scatter the outside "
          "group is (zone == 0) & depth >= 10 m; maps draw < 10 m water light gray and do not classify it. "
          "Last column: share of pixels that differ from the 11-13 rerun PNG (flat `outputs/rerun_10m/figs/*.png` at commit b3fdb4b, now replaced). For the scatters "
          "(Figs 17-20, S12/S13-type panels), most of the difference comes from the random 50,000-point subsample being redrawn. "
          "S12-type panels do not depend on the class at all and still differ by ~8-17%. The maps and violins change only slightly "
          "(<= 1.1% of pixels), and -1 sigma figures are pixel-identical.", ""]
    figure_list(L, changed)
    L.append("Not regenerated (no inside/outside split): Fig 1, Figs 2-4, Fig 5, Fig 11, Fig S1.")
    (Y / "outputs/redo/paper_stats_depth10.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import sys
    ch = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else None
    main(ch)
