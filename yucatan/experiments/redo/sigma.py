"""-1 sigma SST rule (single rule).

Pixel is inside the zone if its monthly-mean SST is below that month's spatial mean minus
one spatial standard deviation, both computed over valid deep (>= 10 m) pixels of the
same monthly-mean SST map. Applied per month-year and to the 2010-2019 climatology.
"""

import json
import numpy as np

import common as c


def classify(sst_map, deep):
    classified = deep & np.isfinite(sst_map)
    if classified.sum() < 2:
        return c.zone_grid(np.zeros_like(classified), classified)
    v = sst_map[classified]
    thr = v.mean() - v.std()
    return c.zone_grid(classified & (sst_map < thr), classified)


def main():
    stacks, meta = c.load_stacks()
    deep = c.deep_mask(meta)
    sst = c.monthly_means(stacks, meta, "sst")
    by_month = {k: classify(v, deep) for k, v in sst.items()}
    clim = {m: classify(v, deep) for m, v in c.climatology(sst).items()}
    c.save_zones("sigma", by_month, clim)
    json.dump({"rule": "monthly-mean SST < spatial mean - 1 sd (deep valid pixels)"},
              open(c.CACHE_DIR / "sigma_info.json", "w"))
    print("sigma done")


if __name__ == "__main__":
    main()
