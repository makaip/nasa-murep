"""Isolation forest, final settings, using the existing trained model (iforest_pipeline5).

Monthly scoring: monthly mean of same-day-median anomalies of SST and SSL; score >= 90th
percentile of that month's scores; standardized SST anomaly < -0.5; standardized SSL
anomaly < 1.0; depth >= 10 m. Classification logic is imported unchanged from
experiments/visualization/bayth_classification.py so it is identical to the figures.

Also: surrogate random forest (100 trees, max_depth 5, as in isolation_forest/run.py)
predicting the anomaly score from the two transformed features, per month-year.
"""

import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import common as c

sys.path.insert(0, str(c.YUCATAN_DIR / "experiments" / "visualization"))
warnings.filterwarnings("ignore", category=UserWarning)  # sklearn version-mismatch warning
import bayth_classification as bc  # noqa: E402


def main(model_path=c.MODEL_PATH, name="iforest"):
    stacks, meta = c.load_stacks()
    deep = c.deep_mask(meta)
    bathy = meta["bathymetry"]
    preprocess, iforest, features = bc._load_model(model_path)
    assert bc.PERCENTILE_UPWELLING == 90 and bc.SST_ANOM_Z_MAX == -0.5
    assert bc.SSL_ANOM_Z_MAX == 1.0 and bc.BAYTH_THRESHOLD == -10

    anom = {v: c.monthly_mean_anomaly(stacks, meta, v) for v in ("sst", "ssl")}
    clim_anom = {v: c.climatology(anom[v]) for v in ("sst", "ssl")}

    def run(a_sst, a_ssl):
        res, _ = bc._classify({"sst": a_sst, "ssl": a_ssl}, preprocess, iforest, features, bathy)
        if res is None:
            return c.zone_grid(np.zeros(deep.shape, bool), np.zeros(deep.shape, bool)), None
        classified = deep & np.isfinite(res["classes"])
        return c.zone_grid(res["classes"] == 1, classified), res

    by_month, clim, scores = {}, {}, {}
    importances = []
    for k in anom["sst"]:
        by_month[k], res = run(anom["sst"][k], anom["ssl"][k])
        if res is None:
            continue
        valid = np.isfinite(res["scores"])
        X = pd.DataFrame({f"{n}_anom": anom[n][k][valid] for n in ("sst", "ssl")})[features]
        Xt = preprocess.transform(X)
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        rf.fit(Xt, res["scores"][valid])
        importances.append({"year": k[0], "month": k[1],
                            **{n: float(w) for n, w in zip(Xt.columns, rf.feature_importances_)}})
    for m in c.MONTHS:
        clim[m], _ = run(clim_anom["sst"][m], clim_anom["ssl"][m])

    # consistency check against the precomputed pickle used by the existing figures
    mismatch = []
    try:
        pre = bc.load_preclassified()
        for (y, m), g in by_month.items():
            p = pre["by_year"].get(y, {}).get(m)
            pin = np.zeros(deep.shape, bool) if p is None else (p["classes"] == 1)
            n_diff = int((pin != (g == 1)).sum())
            if n_diff:
                mismatch.append({"year": y, "month": m, "pixels_differ": n_diff})
        pre_ok = True
    except Exception as e:  # noqa: BLE001
        pre_ok = f"could not load: {e}"

    c.save_zones(name, by_month, clim)
    info = {
        "model": str(model_path.name),
        "model_params": {"n_estimators": iforest.n_estimators, "contamination": iforest.contamination,
                         "random_state": iforest.random_state, "max_samples_": int(iforest.max_samples_),
                         "features": list(features),
                         "yeo_johnson_lambdas": preprocess.named_transformers_["anom"].lambdas_.tolist()},
        "surrogate_importances": importances,
        "preclassified_pickle_check": {"loaded": pre_ok, "months_mismatched": mismatch},
    }
    json.dump(info, open(c.CACHE_DIR / f"{name}_info.json", "w"), indent=1)
    print("iforest done; preclassified mismatches:", len(mismatch))


if __name__ == "__main__":
    main()
    # sensitivity: pipeline4 built the pre-2026-11-08 pickle (backup: figs/_backup_pipeline4_2026-11-08/)
    main(c.MODEL_PATH.parent / "iforest_pipeline4.joblib", "iforest_p4")
