"""% of pixels flagged by different -1 sigma SST rules on v4 (for correcting the paper text).

Old rule (nasa-murep-local cdom_sst/experiments/w_hist/histograms_no_fitting.ipynb, cells 2/7/8,
v3 pickle, printed "13.20%"): for each ISO week-of-year w, average the daily MUR SST maps of
all days in week w over all years -> one climatological map per week; threshold_w = mean - 1 sd
of that map's valid pixels (whole grid, no depth mask); flag pixels < threshold_w; % = flagged
pixel-weeks / valid pixel-weeks.

Redo rule (redo/sigma.py): per month-year, monthly-mean MUR SST map; threshold = spatial mean
- 1 sd over valid pixels with depth >= 10 m; flag deep pixels below it.
"""

import json
import warnings

import numpy as np

import common as c

warnings.filterwarnings("ignore")


def pct_flag(maps, mask):
    flagged = total = 0
    for m in maps:
        ok = np.isfinite(m) & mask
        if ok.sum() < 2:
            continue
        v = m[ok]
        flagged += int((v < v.mean() - v.std(ddof=1)).sum())
        total += int(ok.sum())
    return 100.0 * flagged / total


def main():
    stacks, meta = c.load_stacks()
    sst = stacks["sst"]
    deep = c.deep_mask(meta)
    allpx = np.ones_like(deep)
    week = meta["dates"].isocalendar().week.to_numpy()
    weekly = [c.nanmean0(np.asarray(sst[week == w], dtype=float)) for w in np.unique(week)]
    monthly = c.monthly_means(stacks, meta, "sst")
    out = {
        "old_weekly_climatology_nodepth": pct_flag(weekly, allpx),
        "old_weekly_climatology_depth10": pct_flag(weekly, deep),
        "redo_monthly_per_month_year_depth10": pct_flag(monthly.values(), deep),
        "redo_monthly_per_month_year_nodepth": pct_flag(monthly.values(), allpx),
        "monthly_climatology_depth10": pct_flag(c.climatology(monthly).values(), deep),
        "note": "sd uses ddof=1 here (pandas default, as in the old notebook); redo/sigma.py uses ddof=0, "
                "which changes the redo % by < 0.01 pp at ~40k pixels per map.",
    }
    json.dump(out, open(c.CACHE_DIR / "sigma_variants.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
