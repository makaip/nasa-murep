"""How much does the [0, 1] clip on a_CDOM(412) matter? Counts on raw MODIS L2 pixels.

Re-applies the pickler's exact formulas (data/preprocessing/pickler_utils.py, coefficients in
pickler_config.py) to raw Rrs_443 / Rrs_547 from the AQUA MODIS L2 OC files (E:\\satdata\\cdom),
restricted to the pickler bbox, no flag masking (same as the pickler, which only drops fill
values via xarray decoding). STEP = 1: every file of every year is used.

Counts: pixels with both Rrs finite; with Rrs_443 <= 0 or Rrs_547 <= 0 (log undefined, so the
pickler drops them: no a_CDOM and no slopes); a_CDOM < 0 (impossible: a = exp(.)); a_CDOM > 1
(set to NaN by the clip); slope ranges (slopes are exp(.) and never clipped).
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data" / "preprocessing"))
from pickler_config import (CDOM_412_COEFFS as A, S275_295_COEFFS as S1, S300_600_COEFFS as S2,  # noqa: E402
                            LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)

import common as c  # noqa: E402

STEP = 1


def count(path):
    try:
        g = xr.open_dataset(path, group="geophysical_data")
        n = xr.open_dataset(path, group="navigation_data")
        r1, r2 = g["Rrs_443"].values.astype(float), g["Rrs_547"].values.astype(float)
        lat, lon = n["latitude"].values, n["longitude"].values
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    box = (lon >= LON_MIN) & (lon <= LON_MAX) & (lat >= LAT_MIN) & (lat <= LAT_MAX)
    fin = box & np.isfinite(r1) & np.isfinite(r2)
    pos = fin & (r1 > 0) & (r2 > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        l1, l2 = np.log(r1[pos]), np.log(r2[pos])
    a = np.exp(A["B0"] + A["B1"] * l1 + A["B2"] * l2)
    s1 = np.exp(S1["B0"] + S1["B1"] * l1 + S1["B2"] * l2)
    s2 = np.exp(S2["B0"] + S2["B1"] * l1 + S2["B2"] * l2)
    return {"finite": int(fin.sum()), "rrs443_le0": int((fin & (r1 <= 0)).sum()),
            "rrs547_le0": int((fin & (r2 <= 0)).sum()), "either_le0": int((fin & ~pos).sum()),
            "computed": int(pos.sum()), "a_lt0": int((a < 0).sum()), "a_gt1": int((a > 1).sum()),
            "a_max": float(a.max()) if a.size else None, "a_min": float(a.min()) if a.size else None,
            "s275_min": float(s1.min()) if a.size else None, "s275_max": float(s1.max()) if a.size else None,
            "s300_min": float(s2.min()) if a.size else None, "s300_max": float(s2.max()) if a.size else None}


def main():
    files = []
    for y in c.YEARS:
        fs = sorted(glob.glob(rf"E:\satdata\cdom\Yucatan Peninsula_{y}-01-01_{y}-12-31\*.nc"))
        files += fs[::STEP]
    res = Parallel(n_jobs=16)(delayed(count)(f) for f in files)
    ok = [r for r in res if "error" not in r]
    tot = {k: sum(r[k] for r in ok) for k in ("finite", "rrs443_le0", "rrs547_le0", "either_le0", "computed", "a_lt0", "a_gt1")}
    ext = lambda k, f: f(r[k] for r in ok if r[k] is not None)
    out = {"files_used": len(ok), "files_failed": len(res) - len(ok), "step": STEP, **tot,
           "pct_dropped_rrs_le0": 100 * tot["either_le0"] / tot["finite"],
           "pct_a_gt1_of_computed": 100 * tot["a_gt1"] / tot["computed"],
           "a_min": ext("a_min", min), "a_max": ext("a_max", max),
           "s275_range": [ext("s275_min", min), ext("s275_max", max)],
           "s300_range": [ext("s300_min", min), ext("s300_max", max)]}
    json.dump(out, open(c.CACHE_DIR / "cdom_clip_check.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
