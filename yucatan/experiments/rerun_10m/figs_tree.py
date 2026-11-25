"""Index outputs/rerun_10m/figs (depth10-everywhere figures drawn by the ORIGINAL visualization scripts),
compare it with yucatan/figs, write figs/MAP.json and section 8 of outputs/redo/paper_stats_depth10.md.

Regenerate the figures first (YUC_FIGS_ROOT=<repo>/yucatan/outputs/rerun_10m/figs YUC_DEPTH10=1):
  visualization/yearly_class/IF_bayth_all_years.py      bayth/<year>_IF_class.png, bayth/<year>_IF_anom.png
  visualization/yearly_class/IF_bayth_across_years.py   bayth/climatology_IF_class.png
  visualization/violins/all_years.py                    violins/<var>_<year>.png
  YUC_ZONES=sigma visualization/violins/all_years.py    sigma/<var>_<year>.png
  visualization/scatterplots/{monthly,yearly,pooled}.py scatterplots/...
  visualization/raw/main.py, visualization/histograms/weekly.py   raw/, histograms/
Kept extras from rerun_10m/figures_10m.py (paper-style renders): violins/*_2018_paper.png (Figs 13-16),
scatterplots/.../chlor_vs_cdom_may_2018.png (Fig 17), histograms/sst_weekly_climatology_sigma.png (Fig 6),
sigma/2018_sigma_class.png (Fig 7), sigma/violins_pooled_2010_2019.png (Fig 12).
"""
import json
import re
from pathlib import Path

from PIL import Image

Y = Path(__file__).resolve().parents[2]
NEW, OLD = Y / "outputs/rerun_10m/figs", Y / "figs"
MD = Y / "outputs/redo/paper_stats_depth10.md"
YEARS = range(2010, 2020)
VIOL = ["CDOM_a(412)", "CDOM_S275_295", "CDOM_S300_600", "Log10_Chlorophyll"]
VNAME = {"CDOM_a(412)": "a_CDOM(412)", "CDOM_S275_295": "S275-295", "CDOM_S300_600": "S300-600",
         "Log10_Chlorophyll": "log10 Chl-a"}
V = "visualization/"
SC = "scatterplots/chlor_vs_cdom_upwelling_"

ent = {}   # path -> (paper, what, generator)
for i, y in enumerate(YEARS):
    b = 6 + 8 * i
    ent[f"bayth/{y}_IF_class.png"] = (f"Fig. S{b}" + (" / Fig. 8" if y == 2018 else ""), f"{y}: IF class map",
                                      V + "yearly_class/IF_bayth_all_years.py")
    ent[f"bayth/{y}_IF_anom.png"] = (f"Fig. S{b + 1}" + (" / Fig. 9" if y == 2018 else ""),
                                     f"{y}: IF anomaly score map (no legend / outline)",
                                     V + "yearly_class/IF_bayth_all_years.py")
    for j, v in enumerate(VIOL):
        ent[f"violins/{v}_{y}.png"] = (f"Fig. S{b + 2 + j}", f"{y}: IF violins {VNAME[v]}", V + "violins/all_years.py")
        ent[f"sigma/{v}_{y}.png"] = (f"Fig. S{2 + j} ({y} panel)", f"{y}: -1 sigma violins {VNAME[v]}",
                                     "YUC_ZONES=sigma " + V + "violins/all_years.py")
    ent[f"{SC}months_2010_2019/chlor_vs_cdom_{y}.png"] = (
        f"Fig. S{b + 6}", f"{y}: Chl vs a_CDOM(412), May-Sep vs Oct-Apr, deep pixels", V + "scatterplots/monthly.py")
    ent[f"{SC}zones_yearly/chlor_vs_cdom_upwelling_months_{y}.png"] = (
        f"Fig. S{b + 7}", f"{y}: Chl vs a_CDOM(412), May-Sep, inside vs outside IF class", V + "scatterplots/yearly.py")
ent["bayth/climatology_IF_class.png"] = ("Fig. 10", "IF class, 2010-2019 climatology",
                                         V + "yearly_class/IF_bayth_across_years.py")
for k, (f, w) in {"upwelling_months": ("Fig. 18", "May-Sep"), "nonupwelling_months": ("Fig. 19", "Oct-Apr"),
                  "all_seasons": ("Fig. 20", "all months")}.items():
    ent[f"{SC}zones_pooled/chlor_vs_cdom_all_years_{k}.png"] = (
        f, f"Chl vs a_CDOM(412), {w} 2010-2019, inside vs outside IF class", V + "scatterplots/pooled.py")
F10 = "rerun_10m/figures_10m.py (paper style, extra)"
for j, v in enumerate(VIOL):
    ent[f"violins/{v}_2018_paper.png"] = (f"Fig. {13 + j}", f"IF violins 2018 {VNAME[v]}", F10)
ent[f"{SC}zones_yearly/chlor_vs_cdom_may_2018.png"] = ("Fig. 17", "Chl vs a_CDOM(412), May 2018, inside vs outside", F10)
ent["histograms/sst_weekly_climatology_sigma.png"] = ("Fig. 6", "Weekly SST climatology histogram, -1 sigma bins", F10)
ent["sigma/2018_sigma_class.png"] = ("Fig. 7", "-1 sigma class maps, 2018", F10)
ent["sigma/violins_pooled_2010_2019.png"] = ("Fig. 12", "-1 sigma violins pooled 2010-2019", F10)
for d in ("raw", "histograms"):
    for f in sorted((NEW / d).glob("*.png")):
        ent.setdefault(f"{d}/{f.name}", (None, f"{d}/ figure, < 10 m pixels dropped", V + f"{d}/ (YUC_DEPTH10=1)"))

new = {p.relative_to(NEW).as_posix() for p in NEW.rglob("*.png")}
old = {p.relative_to(OLD).as_posix() for d in ("bayth", "violins", "scatterplots", "raw", "histograms")
       for p in (OLD / d).rglob("*.png")}
missing_from_new = sorted(old - new)
unindexed = sorted(new - set(ent))
absent = sorted(set(ent) - new)
size_diff = []
mp = {}
for k in sorted(ent):
    if k not in new:
        continue
    paper, what, gen = ent[k]
    sz = Image.open(NEW / k).size
    o = {"paper": paper, "what": what, "generator": gen, "size": list(sz), "in_original_tree": k in old}
    if k in old:
        osz = Image.open(OLD / k).size
        o["original_size"] = list(osz)
        if osz != sz:
            size_diff.append((k, osz, sz))
    mp[k] = o
json.dump(mp, open(NEW / "MAP.json", "w"), indent=1)

lines = ["## 8. Regenerated figures", "",
         "`yucatan/outputs/rerun_10m/figs/` mirrors `yucatan/figs/` (same folders, file names, image sizes and "
         "plotting code). Only the data differ: < 10 m pixels are in neither group, and the IF gates (90th-pct score, "
         "SST/SSL z) are computed over deep pixels (zones in `yucatan/figs/bayth_preclassified.pkl`). Class maps leave "
         "< 10 m water blank (white) instead of blue. Drawn by the original `experiments/visualization/` scripts with "
         "`YUC_FIGS_ROOT=yucatan/outputs/rerun_10m/figs YUC_DEPTH10=1`; index: `figs/MAP.json`; "
         "built by `experiments/rerun_10m/figs_tree.py`. Extras with no counterpart in `yucatan/figs/`: "
         "`bayth/*_IF_anom.png` (score maps in the class-map layout), `sigma/` (per-year -1 sigma violins drawn by "
         "the violin script with `YUC_ZONES=sigma`, named like `violins/`, plus Fig 7 and Fig 12) and the paper-style "
         "renders listed with generator `figures_10m.py`.", "",
         f"Files: {len(new)} PNG ({len(new & old)} with an original counterpart, {len(new - old)} extras). "
         f"Original files missing here: {len(missing_from_new)}. Size differences vs original: {len(size_diff)}.", ""]
for k, a, b in size_diff:
    lines.append(f"- size differs: `{k}` original {a[0]}x{a[1]}, new {b[0]}x{b[1]}")
for k in missing_from_new:
    lines.append(f"- missing: `{k}`")
lines += ["", "| Paper figure | File (under `yucatan/outputs/rerun_10m/figs/`) | What | Generator |", "|---|---|---|---|"]


def order(k):
    p = mp[k]["paper"] or "~"
    m = re.match(r"Fig\. (S?)(\d+)", p)
    return (0, m.group(1) == "S", int(m.group(2)), k) if m else (1, False, 0, k)


for k in sorted(mp, key=order):
    o = mp[k]
    lines.append(f"| {o['paper'] or '-'} | `{k}` | {o['what']} | `{o['generator']}` |")
lines += ["", "Not regenerated (no inside/outside split): Fig 1, Figs 2-4, Fig 5, Fig 11, Fig S1.", ""]
txt = MD.read_text(encoding="utf-8")
MD.write_text(txt[:txt.index("## 8. Regenerated figures")] + "\n".join(lines), encoding="utf-8")
print(f"{len(new)} png; missing {missing_from_new}; size diffs {size_diff}; unindexed {unindexed}; absent {absent}")
