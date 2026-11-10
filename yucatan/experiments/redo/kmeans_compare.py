"""K-means preprocessing comparison, replicating the OLD notebooks' pipeline on v4.

Old notebooks (nasa-murep-local/cdom_sst/experiments/kmeans/, v3 pickle):
  * ssl_kmeans_no_norm.ipynb : features [log10 Chl, SST, SSL], RAW (no scaling), k=3
  * ssl_kmeans.ipynb         : same features, StandardScaler, k=3
  Both: DAILY pixels of the whole 2010-2019 record, no depth mask, SSL daily grid
  NaN-filled with its mean and gaussian-blurred (sigma=2 px), valid = all three finite and
  Chl > 0, 1,000,000-point random sample (np.random.seed(42)), KMeans(random_state=42,
  n_init=10).

Variants run here (k = 2 and 3 each):
  a0  raw, exact old setup (no depth mask)
  a   raw + depth >= 10 m
  b   StandardScaler + depth >= 10 m
  c   PowerTransformer(yeo-johnson, standardize=True) + depth >= 10 m
Per variant: share of squared centroid separation per feature (in the space K-means ran
in), PCA variance ratios (same space), silhouette (20k subsample, same space), and, after
predicting every valid daily pixel: share of cold-cluster pixel-days in May-Sep and number
of month-years where the cold cluster covers >= 1% of that month-year's pixel-days.
Cold cluster = centroid with the lowest SST in physical units.
"""

import json
from itertools import combinations

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import PowerTransformer, StandardScaler

import common as c

FEATURES = ["log10_chl", "sst", "ssl"]   # old notebook column order


def blur_ssl(ssl):
    out = np.empty_like(ssl, dtype=np.float32)
    for i in range(len(ssl)):
        g = np.asarray(ssl[i], dtype=float)
        nan = np.isnan(g)
        if nan.all():
            out[i] = np.nan
            continue
        f = g.copy()
        f[nan] = np.nanmean(g)
        b = gaussian_filter(f, sigma=2.0)
        b[nan] = np.nan
        out[i] = b
    return out


def sep_share(centers):
    sq = np.zeros(centers.shape[1])
    for a, b in combinations(range(len(centers)), 2):
        sq += (centers[a] - centers[b]) ** 2
    return dict(zip(FEATURES, (sq / sq.sum()).tolist()))


def main():
    stacks, meta = c.load_stacks()
    T = len(meta["dates"])
    ssl_b = blur_ssl(stacks["ssl"])
    chl = np.asarray(stacks["chlorophyll"])
    sst = np.asarray(stacks["sst"])
    deep = c.deep_mask(meta)
    d = meta["dates"]
    my = ((d.year - c.YEARS[0]) * 12 + (d.month - 1)).to_numpy()
    month = d.month.to_numpy()

    def table(mask2d):
        ok = np.isfinite(sst) & np.isfinite(ssl_b) & np.isfinite(chl) & (chl > 0) & mask2d[None]
        t_idx = np.nonzero(ok)[0]
        X = np.column_stack([np.log10(chl[ok]), sst[ok], ssl_b[ok]]).astype(float)
        return X, my[t_idx], month[t_idx]

    data = {"nodepth": table(np.ones_like(deep)), "deep": table(deep)}
    variants = {"a0_raw_nodepth": ("nodepth", None), "a_raw": ("deep", None),
                "b_standard": ("deep", "std"), "c_yeojohnson": ("deep", "yj")}
    out = {}
    for name, (dom, prep) in variants.items():
        X, myv, mon = data[dom]
        np.random.seed(42)
        idx = np.random.choice(len(X), size=min(1_000_000, len(X)), replace=False)
        Xs = X[idx]
        tf = None
        if prep == "std":
            tf = StandardScaler().fit(Xs)
        elif prep == "yj":
            tf = PowerTransformer(method="yeo-johnson", standardize=True).fit(Xs)
        Z = tf.transform(Xs) if tf else Xs
        Zall = tf.transform(X) if tf else X
        pca = PCA().fit(Z)
        res = {"n_pixel_days": int(len(X)), "n_fit": int(len(Xs)),
               "raw_variance": dict(zip(FEATURES, Xs.var(axis=0).tolist())),
               "raw_variance_share": dict(zip(FEATURES, (Xs.var(axis=0) / Xs.var(axis=0).sum()).tolist())),
               "fit_space_variance": dict(zip(FEATURES, Z.var(axis=0).tolist())),
               "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
               "pca_pc1_loadings": dict(zip(FEATURES, pca.components_[0].tolist())), "by_k": {}}
        rng = np.random.default_rng(0)
        sil_idx = rng.choice(len(Z), 20_000, replace=False)
        for k in (2, 3):
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Z)
            centers_phys = tf.inverse_transform(km.cluster_centers_) if tf else km.cluster_centers_
            cold = int(np.argmin(centers_phys[:, 1]))
            lab = km.predict(Zall)
            is_cold = lab == cold
            n_my = np.bincount(myv, minlength=120)
            n_cold_my = np.bincount(myv[is_cold], minlength=120)
            frac_my = np.divide(n_cold_my, n_my, out=np.zeros(120), where=n_my > 0)
            summer = np.isin(mon, [5, 6, 7, 8, 9])
            res["by_k"][k] = {
                "centers_fit_space": km.cluster_centers_.tolist(),
                "centers_phys": centers_phys.tolist(),
                "cluster_fraction_fit": (np.bincount(km.labels_, minlength=k) / len(km.labels_)).tolist(),
                "sq_sep_share": sep_share(km.cluster_centers_),
                "silhouette_20k": float(silhouette_score(Z[sil_idx], km.labels_[sil_idx])),
                "cold_cluster": cold,
                "cold_share_of_all_pixel_days": float(is_cold.mean()),
                "cold_share_in_May_Sep": float(summer[is_cold].mean()) if is_cold.any() else None,
                "month_years_cold_ge1pct": int((frac_my >= 0.01).sum()),
                "cold_pct_by_calendar_month": {m: float(100 * is_cold[mon == m].mean()) for m in c.MONTHS},
            }
            print(name, k, res["by_k"][k]["sq_sep_share"], flush=True)
        out[name] = res
    json.dump(out, open(c.CACHE_DIR / "kmeans_compare.json", "w"), indent=1)
    print("kmeans_compare done")


if __name__ == "__main__":
    main()
