"""Copy the depth10-everywhere figures (outputs/redo/figs_depth10, flat paper names) into
outputs/rerun_10m/figs using the original yucatan/figs layout and file names.

raw/ and histograms/ are not made by figures_10m.py; they are redrawn directly into the tree by
visualization/raw/main.py and visualization/histograms/weekly.py with YUC_DEPTH10=1 and
YUC_FIGS_ROOT=outputs/rerun_10m/figs. Figures with no counterpart in the original tree
(IF score maps, -1 sigma figures, Fig. 6, Fig. 17) get new names (EXTRA below).
Writes outputs/rerun_10m/figs/MAP.json: tree path -> paper figure.
"""
import json
import shutil
from pathlib import Path

Y = Path(__file__).resolve().parents[2]
SRC, DST = Y / "outputs/redo/figs_depth10", Y / "outputs/rerun_10m/figs"
YEARS = range(2010, 2020)
VIOL = ["CDOM_a(412)", "CDOM_S275_295", "CDOM_S300_600", "Log10_Chlorophyll"]
SIG = ["cdom_412", "cdom_slope_275_295", "cdom_slope_300_600", "log10_chlorophyll"]

ORIG, EXTRA = {}, {}   # tree path -> (source file, paper figure)
for i, y in enumerate(YEARS):
    b = 6 + 8 * i
    paper = lambda k: f"Fig. S{b + k}" + (" / Fig. 8" if (y == 2018 and k == 0) else "")
    ORIG[f"bayth/{y}_IF_class.png"] = (f"S{b:02d}.png", paper(0))
    EXTRA[f"bayth/{y}_IF_anom.png"] = (f"S{b + 1:02d}.png", f"Fig. S{b + 1}" + (" / Fig. 9" if y == 2018 else ""))
    for j, v in enumerate(VIOL):
        ORIG[f"violins/{v}_{y}.png"] = (f"S{b + 2 + j:02d}.png", f"Fig. S{b + 2 + j}")
    ORIG[f"scatterplots/chlor_vs_cdom_upwelling_months_2010_2019/chlor_vs_cdom_{y}.png"] = (f"S{b + 6:02d}.png", f"Fig. S{b + 6}")
    ORIG[f"scatterplots/chlor_vs_cdom_upwelling_zones_yearly/chlor_vs_cdom_upwelling_months_{y}.png"] = (f"S{b + 7:02d}.png", f"Fig. S{b + 7}")
ORIG["bayth/climatology_IF_class.png"] = ("fig10.png", "Fig. 10")
P = "scatterplots/chlor_vs_cdom_upwelling_zones_pooled/chlor_vs_cdom_all_years_"
ORIG[P + "upwelling_months.png"] = ("fig18.png", "Fig. 18")
ORIG[P + "nonupwelling_months.png"] = ("fig19.png", "Fig. 19")
ORIG[P + "all_seasons.png"] = ("fig20.png", "Fig. 20")
# 2018 IF violins drawn as paper Figs 13-16 (same data/limits as the per-year 2018 violins)
for j, v in enumerate(VIOL):
    EXTRA[f"violins/{v}_2018_paper.png"] = (f"fig{13 + j}.png", f"Fig. {13 + j}")
EXTRA["scatterplots/chlor_vs_cdom_upwelling_zones_yearly/chlor_vs_cdom_may_2018.png"] = ("fig17.png", "Fig. 17")
EXTRA["histograms/sst_weekly_climatology_sigma.png"] = ("fig06.png", "Fig. 6")
EXTRA["sigma/2018_sigma_class.png"] = ("fig07.png", "Fig. 7")
EXTRA["sigma/violins_pooled_2010_2019.png"] = ("fig12.png", "Fig. 12")
for j, v in enumerate(SIG):
    EXTRA[f"sigma/violins_{v}_by_year.png"] = (f"S{2 + j:02d}.png", f"Fig. S{2 + j}")

# remove the old flat files (fig*.png, S*.png) at the top of figs/
for f in list(DST.glob("fig*.png")) + list(DST.glob("S*.png")):
    f.unlink()
used = set()
for dst, (src, _) in {**ORIG, **EXTRA}.items():
    (DST / dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SRC / src, DST / dst)
    used.add(src)
unused = sorted(p.name for p in SRC.glob("*.png") if p.name not in used)
mp = {k: {"paper": v[1], "from": v[0], "original_tree": k in ORIG} for k, v in sorted({**ORIG, **EXTRA}.items())}
for d in ("raw", "histograms"):
    for f in sorted((DST / d).glob("*.png")):
        k = f"{d}/{f.name}"
        mp.setdefault(k, {"paper": None, "from": f"visualization/{d}/ (YUC_DEPTH10=1)", "original_tree": True})
json.dump(mp, open(DST / "MAP.json", "w"), indent=1)
print(len(ORIG), "original-name copies,", len(EXTRA), "extras; unused sources:", unused)
