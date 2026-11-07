"""K-means on standardized [SST, log10 Chl-a, SSL] of monthly-mean maps.

Two variants (both reported; old notebook ssl_kmeans*.ipynb used k=3, StandardScaler,
random_state=42, n_init=10, features [log10 Chl, SST, SSL]):
  * "kmeans"        : one model fitted on pooled deep pixels of all 120 monthly-mean maps
                      (global standardization, as in the old notebooks), applied to every
                      month-year map and to the climatology maps.
  * "kmeans_month"  : features standardized within each month-year map and a separate
                      k-means fitted per map (removes the seasonal SST cycle).
Zone = cluster whose centroid has the lowest SST. k=3 (old notebook) is primary; k=2 is
also fitted for the separation diagnostics.

Separation diagnostic: share of squared centroid separation along each feature,
sum over centroid pairs (c_a,j - c_b,j)^2 / sum over j, in standardized units.
"""

import json
from itertools import combinations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import common as c

FEATURES = ["sst", "log10_chl", "ssl"]
K_PRIMARY = 3


def feature_maps(stacks, meta):
    sst = c.monthly_means(stacks, meta, "sst")
    chl = c.monthly_means(stacks, meta, "chlorophyll")
    ssl = c.monthly_means(stacks, meta, "ssl")
    with np.errstate(divide="ignore", invalid="ignore"):
        lchl = {k: np.where(v > 0, np.log10(v), np.nan) for k, v in chl.items()}
    return {"sst": sst, "log10_chl": lchl, "ssl": ssl}


def stack_features(fmaps, key, deep):
    F = np.stack([fmaps[f][key] for f in FEATURES], axis=-1)
    classified = deep & np.all(np.isfinite(F), axis=-1)
    return F[classified], classified


def separation_share(centers):
    sq = np.zeros(centers.shape[1])
    for a, b in combinations(range(len(centers)), 2):
        sq += (centers[a] - centers[b]) ** 2
    return dict(zip(FEATURES, (sq / sq.sum()).tolist()))


def zone_pair_share(centers):
    z = int(np.argmin(centers[:, 0]))
    others = [i for i in range(len(centers)) if i != z]
    sq = sum((centers[z] - centers[o]) ** 2 for o in others)
    return dict(zip(FEATURES, (sq / sq.sum()).tolist()))


def between_ss_share(centers, counts):
    mu = np.average(centers, axis=0, weights=counts)
    ss = (counts[:, None] * (centers - mu) ** 2).sum(axis=0)
    return dict(zip(FEATURES, (ss / ss.sum()).tolist()))


def main():
    stacks, meta = c.load_stacks()
    deep = c.deep_mask(meta)
    fmaps = feature_maps(stacks, meta)
    clim_maps = {f: c.climatology(fmaps[f]) for f in FEATURES}
    keys = [(y, m) for y in c.YEARS for m in c.MONTHS]
    rng = np.random.default_rng(42)
    info = {"features": FEATURES, "k_primary": K_PRIMARY}

    # ---------------- global variant ----------------
    pooled = np.concatenate([stack_features(fmaps, k, deep)[0] for k in keys])
    fit_X = pooled[rng.choice(len(pooled), size=min(1_000_000, len(pooled)), replace=False)]
    scaler = StandardScaler().fit(fit_X)
    Z = scaler.transform(fit_X)
    pca = PCA().fit(Z)
    info["global"] = {"n_pooled_pixel_months": int(len(pooled)), "n_fit": int(len(fit_X)),
                      "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                      "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                      "pca_components": pca.components_.tolist(), "by_k": {}}
    models = {}
    for k in (2, 3):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Z)
        counts = np.bincount(km.labels_, minlength=k).astype(float)
        models[k] = km
        info["global"]["by_k"][k] = {
            "centers_std": km.cluster_centers_.tolist(),
            "centers_phys": scaler.inverse_transform(km.cluster_centers_).tolist(),
            "cluster_fraction": (counts / counts.sum()).tolist(),
            "pairwise_sq_sep_share": separation_share(km.cluster_centers_),
            "zone_vs_others_sq_sep_share": zone_pair_share(km.cluster_centers_),
            "between_ss_share": between_ss_share(km.cluster_centers_, counts),
        }
    km = models[K_PRIMARY]
    zone_label = int(np.argmin(km.cluster_centers_[:, 0]))

    def apply_global(F_by_feature):
        F = np.stack([F_by_feature[f] for f in FEATURES], axis=-1)
        classified = deep & np.all(np.isfinite(F), axis=-1)
        inside = np.zeros(deep.shape, bool)
        if classified.any():
            inside[classified] = km.predict(scaler.transform(F[classified])) == zone_label
        return c.zone_grid(inside, classified)

    by_month = {k: apply_global({f: fmaps[f][k] for f in FEATURES}) for k in keys}
    clim = {m: apply_global({f: clim_maps[f][m] for f in FEATURES}) for m in c.MONTHS}
    c.save_zones("kmeans", by_month, clim)

    # ---------------- per-month variant ----------------
    per = {2: [], 3: []}
    pcas = []

    def fit_month(F_by_feature):
        F = np.stack([F_by_feature[f] for f in FEATURES], axis=-1)
        classified = deep & np.all(np.isfinite(F), axis=-1)
        inside = np.zeros(deep.shape, bool)
        out = {}
        if classified.sum() < 50:
            return c.zone_grid(inside, classified), out
        Zm = StandardScaler().fit_transform(F[classified])
        out["pca"] = PCA().fit(Zm).explained_variance_ratio_.tolist()
        for k in (2, 3):
            kmm = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Zm)
            out[k] = {"pairwise": separation_share(kmm.cluster_centers_),
                      "zone_vs_others": zone_pair_share(kmm.cluster_centers_)}
            if k == K_PRIMARY:
                inside[classified] = kmm.labels_ == int(np.argmin(kmm.cluster_centers_[:, 0]))
        return c.zone_grid(inside, classified), out

    by_month_m = {}
    for k in keys:
        by_month_m[k], out = fit_month({f: fmaps[f][k] for f in FEATURES})
        if out:
            pcas.append(out["pca"])
            for kk in (2, 3):
                per[kk].append(out[kk])
    clim_m = {m: fit_month({f: clim_maps[f][m] for f in FEATURES})[0] for m in c.MONTHS}
    c.save_zones("kmeans_month", by_month_m, clim_m)

    def summarize(lst, field):
        arr = np.array([[d[field][f] for f in FEATURES] for d in lst])
        return {f: {"mean": float(arr[:, i].mean()), "median": float(np.median(arr[:, i])),
                    "min": float(arr[:, i].min()), "max": float(arr[:, i].max())}
                for i, f in enumerate(FEATURES)}

    info["per_month"] = {"n_months": len(pcas),
                         "pca_explained_variance_ratio_mean": np.mean(pcas, axis=0).tolist(),
                         "by_k": {kk: {"pairwise_sq_sep_share": summarize(per[kk], "pairwise"),
                                       "zone_vs_others_sq_sep_share": summarize(per[kk], "zone_vs_others")}
                                  for kk in (2, 3)}}
    json.dump(info, open(c.CACHE_DIR / "kmeans_info.json", "w"), indent=1)
    print("kmeans done")


if __name__ == "__main__":
    main()
