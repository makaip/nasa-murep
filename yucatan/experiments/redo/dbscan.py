"""DBSCAN (appendix) on the same features as K-means: [SST, log10 Chl-a, SSL] of monthly
mean maps, deep pixels, standardized within each month-year map.

Parameters reused from the old notebook clustering/ssl_dbscan.ipynb: eps=0.28,
min_samples=50. (clustering/dbscan_clustering.ipynb used eps=0.27, min_samples=40 on a
100k pooled daily sample; that pooled variant is also run for reference.)

Zone = the non-noise cluster with the lowest mean SST, only if that cluster is not the
dominant (largest) cluster; otherwise DBSCAN found no separate cold zone that month.
Instability: number of clusters, noise fraction, and IoU of the zone between consecutive
months.
"""

import json

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import common as c
from kmeans import FEATURES, feature_maps

EPS, MIN_SAMPLES = 0.28, 50


def run_one(F, deep):
    classified = deep & np.all(np.isfinite(F), axis=-1)
    inside = np.zeros(deep.shape, bool)
    if classified.sum() < MIN_SAMPLES:
        return c.zone_grid(inside, classified), None
    X = StandardScaler().fit_transform(F[classified])
    lab = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(X)
    ids = [l for l in np.unique(lab) if l >= 0]
    sizes = {int(l): int((lab == l).sum()) for l in ids}
    stats = {"n_clusters": len(ids), "noise_fraction": float((lab == -1).mean()),
             "n_pixels": int(len(lab)), "cluster_sizes": sizes, "zone": None}
    if len(ids) >= 2:
        sst = F[classified][:, 0]
        cold = min(ids, key=lambda l: sst[lab == l].mean())
        largest = max(ids, key=lambda l: sizes[l])
        if cold != largest:
            inside[classified] = lab == cold
            stats["zone"] = int(cold)
    return c.zone_grid(inside, classified), stats


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else np.nan


def main():
    stacks, meta = c.load_stacks()
    deep = c.deep_mask(meta)
    fmaps = feature_maps(stacks, meta)
    clim_maps = {f: c.climatology(fmaps[f]) for f in FEATURES}
    keys = [(y, m) for y in c.YEARS for m in c.MONTHS]
    Fs = [np.stack([fmaps[f][k] for f in FEATURES], axis=-1) for k in keys]
    res = Parallel(n_jobs=16)(delayed(run_one)(F, deep) for F in Fs)
    by_month = {k: r[0] for k, r in zip(keys, res)}
    stats = [{"year": k[0], "month": k[1], **(r[1] or {})} for k, r in zip(keys, res)]
    cres = Parallel(n_jobs=12)(delayed(run_one)(np.stack([clim_maps[f][m] for f in FEATURES], -1), deep)
                               for m in c.MONTHS)
    clim = {m: r[0] for m, r in zip(c.MONTHS, cres)}
    c.save_zones("dbscan", by_month, clim)

    ious = [iou(by_month[a] == 1, by_month[b] == 1) for a, b in zip(keys[:-1], keys[1:])]

    # pooled reference run as in dbscan_clustering.ipynb (100k sample, eps .27, min 40)
    rng = np.random.default_rng(42)
    pooled = np.concatenate([F[deep & np.all(np.isfinite(F), -1)] for F in Fs])
    samp = StandardScaler().fit_transform(pooled[rng.choice(len(pooled), 100_000, replace=False)])
    lab = DBSCAN(eps=0.27, min_samples=40, n_jobs=-1).fit_predict(samp)
    pooled_info = {"n": 100_000, "eps": 0.27, "min_samples": 40,
                   "n_clusters": int(len(set(lab) - {-1})), "noise_fraction": float((lab == -1).mean()),
                   "largest_cluster_fraction": float(np.bincount(lab[lab >= 0]).max() / len(lab))}

    json.dump({"eps": EPS, "min_samples": MIN_SAMPLES, "per_month": stats,
               "clim": [r[1] for r in cres], "consecutive_month_zone_iou": ious,
               "pooled_reference": pooled_info},
              open(c.CACHE_DIR / "dbscan_info.json", "w"), indent=1)
    print("dbscan done")


if __name__ == "__main__":
    main()
