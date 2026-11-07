"""Shared inside-vs-outside statistics for any zone method.

Daily pixels (deep >= 10 m, 2010-2019) are labelled by their month-year zone map
(1 inside, 0 outside, -1 not classified -> dropped).

Per calendar month (all years pooled) and pooled over all months:
  * n, median inside / outside, Mann-Whitney U two-sided p, Cliff's delta (inside vs
    outside; > 0 means inside values are larger) for a_CDOM(412), S275-295, S300-600,
    log10 Chl-a
  * Spearman rho of a_CDOM(412) vs Chl-a inside and outside (rank-based, so identical on
    log-log axes), Pearson r on log10-log10, Fisher-z test of rho_in vs rho_out (n is
    pixel-days, which are spatially/temporally autocorrelated, so this p is
    anti-conservative), plus a paired Wilcoxon test over month-years of
    rho_in - rho_out (one value per month-year -> far fewer, more independent samples)
  * "taper" check: rho and log-log OLS slope within the lower vs upper half of a_CDOM
    (split at the median of that group), and by a_CDOM quartile for all pixels.
"""

import json
import sys

import numpy as np
from scipy import stats as ss

import common as c

VARS = {"a412": "cdom_412", "s275": "cdom_slope_275_295", "s300": "cdom_slope_300_600",
        "log10_chl": "chlorophyll"}
_PIX = None


def pixel_table():
    """Flattened deep-pixel observations, built once: {key: values}, plus t, pix, my."""
    global _PIX
    if _PIX is not None:
        return _PIX
    stacks, meta = c.load_stacks()
    deep = c.deep_mask(meta).ravel()
    deep_idx = np.where(deep)[0]
    T = len(meta["dates"])
    d = meta["dates"]
    my = ((d.year - c.YEARS[0]) * 12 + (d.month - 1)).to_numpy()
    out = {}
    for key, var in VARS.items():
        a = np.asarray(stacks[var], dtype=np.float32).reshape(T, -1)[:, deep_idx]
        if key == "log10_chl":
            with np.errstate(divide="ignore", invalid="ignore"):
                a = np.where(a > 0, np.log10(a), np.nan)
        out[key] = a
    _PIX = {"vals": out, "deep_idx": deep_idx, "my": my, "month": d.month.to_numpy(), "T": T}
    return _PIX


def labels_for(method):
    P = pixel_table()
    by_month, _ = c.load_zones(method)
    keys = [(y, m) for y in c.YEARS for m in c.MONTHS]
    Z = np.stack([by_month[k].ravel()[P["deep_idx"]] for k in keys])  # (120, npix)
    return Z[P["my"]]                                                   # (T, npix)


def cliffs_delta_mwu(x, y):
    u, p = ss.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
    return float(2.0 * u / (len(x) * len(y)) - 1.0), float(p)


def compare_groups(x, y):
    r = {"n_in": int(len(x)), "n_out": int(len(y)),
         "median_in": float(np.median(x)) if len(x) else None,
         "median_out": float(np.median(y)) if len(y) else None,
         "mwu_p": None, "cliffs_delta": None}
    if len(x) >= 2 and len(y) >= 2:
        r["cliffs_delta"], r["mwu_p"] = cliffs_delta_mwu(x, y)
    return r


def corr_block(a, chl_log):
    """a: a412 (linear), chl_log: log10 chl. Spearman is rank-invariant to log."""
    n = len(a)
    if n < 10:
        return {"n": n, "spearman": None}
    rho = ss.spearmanr(a, chl_log).statistic
    la = np.log10(a)
    pr = ss.pearsonr(la, chl_log).statistic
    slope = np.polyfit(chl_log, la, 1)[0]
    med = np.median(a)
    lo, hi = a <= med, a > med
    half = {}
    for name, sel in (("lower_half", lo), ("upper_half", hi)):
        if sel.sum() >= 10:
            half[name] = {"n": int(sel.sum()), "a412_range": [float(a[sel].min()), float(a[sel].max())],
                          "spearman": float(ss.spearmanr(a[sel], chl_log[sel]).statistic),
                          "loglog_slope": float(np.polyfit(chl_log[sel], la[sel], 1)[0])}
    return {"n": int(n), "spearman": float(rho), "pearson_loglog": float(pr),
            "loglog_slope_log_a_on_log_chl": float(slope), "a412_median": float(med), **half}


def fisher_z(r1, n1, r2, n2):
    if r1 is None or r2 is None or n1 < 10 or n2 < 10:
        return None
    se = np.sqrt(1.06 / (n1 - 3) + 1.06 / (n2 - 3))   # Fieller et al. for Spearman
    z = (np.arctanh(r1) - np.arctanh(r2)) / se
    return {"z": float(z), "p": float(2 * ss.norm.sf(abs(z)))}


def quartile_taper(a, chl_log):
    qs = np.quantile(a, [0, .25, .5, .75, 1])
    out = []
    for i in range(4):
        sel = (a >= qs[i]) & (a <= qs[i + 1])
        out.append({"quartile": i + 1, "a412_range": [float(qs[i]), float(qs[i + 1])], "n": int(sel.sum()),
                    "spearman": float(ss.spearmanr(a[sel], chl_log[sel]).statistic)})
    return out


def method_stats(method):
    P = pixel_table()
    L = labels_for(method)
    month = np.broadcast_to(P["month"][:, None], L.shape)
    my = np.broadcast_to(P["my"][:, None], L.shape)
    res = {"method": method, "by_month": {}, "pooled": {}}

    groups = {}
    for key, A in P["vals"].items():
        ok = np.isfinite(A) & (L >= 0)
        groups[key] = (A[ok], L[ok] == 1, month[ok])
    # paired a412-chl observations
    A, C = P["vals"]["a412"], P["vals"]["log10_chl"]
    ok = np.isfinite(A) & np.isfinite(C) & (A > 0) & (L >= 0)
    ca, cc, cin, cm, cmy = A[ok], C[ok], L[ok] == 1, month[ok], my[ok]

    for m in list(c.MONTHS) + ["pooled"]:
        row = {}
        for key, (v, inside, mon) in groups.items():
            sel = np.ones(len(v), bool) if m == "pooled" else (mon == m)
            row[key] = compare_groups(v[sel & inside], v[sel & ~inside])
        sel = np.ones(len(ca), bool) if m == "pooled" else (cm == m)
        cin_ = corr_block(ca[sel & cin], cc[sel & cin])
        cout = corr_block(ca[sel & ~cin], cc[sel & ~cin])
        row["corr_a412_chl"] = {"inside": cin_, "outside": cout,
                                "fisher_z_in_vs_out": fisher_z(cin_.get("spearman"), cin_["n"],
                                                               cout.get("spearman"), cout["n"])}
        if m == "pooled":
            res["pooled"] = row
        else:
            res["by_month"][m] = row

    # paired month-year test of rho_in vs rho_out
    diffs = []
    for k in range(len(c.YEARS) * 12):
        s = cmy == k
        si, so = s & cin, s & ~cin
        if si.sum() >= 30 and so.sum() >= 30:
            ri = ss.spearmanr(ca[si], cc[si]).statistic
            ro = ss.spearmanr(ca[so], cc[so]).statistic
            if np.isfinite(ri) and np.isfinite(ro):
                diffs.append({"year": c.YEARS[0] + k // 12, "month": k % 12 + 1, "rho_in": float(ri),
                              "rho_out": float(ro), "n_in": int(si.sum())})
    d = np.array([x["rho_in"] - x["rho_out"] for x in diffs])
    res["monthyear_rho_test"] = {
        "n_month_years": len(diffs),
        "median_rho_in": float(np.median([x["rho_in"] for x in diffs])) if diffs else None,
        "median_rho_out": float(np.median([x["rho_out"] for x in diffs])) if diffs else None,
        "median_diff": float(np.median(d)) if len(d) else None,
        "frac_in_lt_out": float((d < 0).mean()) if len(d) else None,
        "wilcoxon_p": float(ss.wilcoxon(d).pvalue) if len(d) >= 5 else None,
        "rows": diffs,
    }
    res["taper_all_pixels_quartiles"] = quartile_taper(ca, cc)
    res["taper_inside_quartiles"] = quartile_taper(ca[cin], cc[cin]) if cin.sum() > 100 else None
    res["taper_outside_quartiles"] = quartile_taper(ca[~cin], cc[~cin])
    return res


def main(methods):
    for m in methods:
        r = method_stats(m)
        json.dump(r, open(c.CACHE_DIR / f"stats_{m}.json", "w"), indent=1)
        print("stats done", m, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["sigma", "iforest", "kmeans", "kmeans_month"])
