"""Is Spearman rho(a_CDOM(412), Chl-a) different inside vs outside the zone?

Methods: iforest (main) and sigma (comparison). Scopes: all months, May-Sep, and each
calendar month where both zones have data. Daily deep (>= 10 m) pixels with valid a412 > 0
and Chl > 0, labelled by their month-year zone map.

Tests (rho_in - rho_out):
 a. Naive Fisher z: z = (atanh r_in - atanh r_out) / sqrt(1.06/(n_in-3) + 1.06/(n_out-3))
    (1.06 = Fieller et al. 1957 variance factor for Spearman), n = pixel-days.
    Anti-conservative: pixel-days are strongly autocorrelated in space and time.
 b. Fisher z with effective n, two versions:
    b1. Bretherton et al. (1999)-style: for each zone, n_eff = n * prod_d (1 - r_a,d r_c,d)/(1 + r_a,d r_c,d)
        over d in {east-west, north-south, day-to-day}, where r_x,d is the lag-1
        autocorrelation of the rank-transformed variable along d (pairs where both pixels are in
        the same zone and valid; computed on all months, reused for every scope).
    b2. Month-year as the unit: n_eff = number of month-years contributing >= 30 pixel-days to
        that zone.
 c. Block bootstrap (B = 2000, paired: the same resampled blocks are used for inside and
    outside): (c1) blocks = whole month-years; (c2) blocks = month-year x 25x25-pixel tile.
    95% percentile CI of rho_in - rho_out; two-sided p = 2 * min(P(diff* <= 0), P(diff* >= 0)),
    floored at 1/B. For speed rho* is the Pearson correlation of the scope-level ranks
    accumulated from per-block sufficient statistics. This equals the Spearman rho exactly for
    the original sample, and approximates it for resamples (ranks are not recomputed after
    resampling). A block permutation test is not done: zone labels are defined by the data
    (cold, high-Chl pixels), so swapping labels does not give a meaningful null.
 d. Wilcoxon signed-rank test of rho_in - rho_out over month-years with >= 30 pixel-days in
    each zone (n = number of such month-years).
"""

import json
import warnings

import numpy as np
from scipy import stats as ss

import common as c

warnings.filterwarnings("ignore")
B = 2000
TILE = 25
MIN_PX = 30
METHODS = ["iforest", "sigma"]


def fisher(r1, n1, r2, n2):
    if n1 <= 3 or n2 <= 3:
        return None
    se = np.sqrt(1.06 / (n1 - 3) + 1.06 / (n2 - 3))
    z = (np.arctanh(r1) - np.arctanh(r2)) / se
    return {"z": float(z), "p": float(2 * ss.norm.sf(abs(z))), "n_in": float(n1), "n_out": float(n2)}


def lag1(R, axis):
    """Lag-1 autocorrelation of grid R (NaN = not in sample) along axis."""
    a = np.moveaxis(R, axis, 0)
    x, y = a[:-1].ravel(), a[1:].ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 100:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def block_sums(rx, ry, blk, nblk):
    w = lambda v: np.bincount(blk, weights=v, minlength=nblk)
    return np.stack([np.bincount(blk, minlength=nblk).astype(float), w(rx), w(ry),
                     w(rx * rx), w(ry * ry), w(rx * ry)], axis=1)


def corr_from(S):
    n, sx, sy, sxx, syy, sxy = S.T
    cov = sxy - sx * sy / n
    return cov / np.sqrt((sxx - sx ** 2 / n) * (syy - sy ** 2 / n))


def boot(S_in, S_out, units, rng):
    """units: integer array mapping each block to a resampling unit (0..U-1)."""
    U = units.max() + 1
    Ui = np.zeros((U, 6)); Uo = np.zeros((U, 6))
    np.add.at(Ui, units, S_in); np.add.at(Uo, units, S_out)
    keep = (Ui[:, 0] > 0) | (Uo[:, 0] > 0)
    Ui, Uo = Ui[keep], Uo[keep]
    U = len(Ui)
    counts = np.stack([np.bincount(rng.integers(0, U, U), minlength=U) for _ in range(B)])  # (B, U)
    d = corr_from(counts @ Ui) - corr_from(counts @ Uo)
    d = d[np.isfinite(d)]
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"n_units": int(U), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "boot_sd": float(d.std()), "p": float(max(p, 1 / B))}


def main():
    stacks, meta = c.load_stacks()
    A = np.asarray(stacks["cdom_412"], dtype=np.float32)
    C = np.asarray(stacks["chlorophyll"], dtype=np.float32)
    deep = c.deep_mask(meta)
    T, H, W = A.shape
    d = meta["dates"]
    my = ((d.year - c.YEARS[0]) * 12 + (d.month - 1)).to_numpy()
    mon = d.month.to_numpy()
    tile = (np.arange(H)[:, None] // TILE) * (W // TILE) + (np.arange(W)[None, :] // TILE)
    ntile = (H // TILE) * (W // TILE)
    valid = np.isfinite(A) & np.isfinite(C) & (A > 0) & (C > 0) & deep[None]
    rng = np.random.default_rng(42)
    out = {"B": B, "tile_px": TILE, "min_px_per_month_year": MIN_PX}

    for method in METHODS:
        by_month, _ = c.load_zones(method)
        L = c.daily_labels(by_month, meta)
        res = {}
        # ---- b1 inflation factors (all months) ----
        factors = {}
        for zname, zval in (("inside", 1), ("outside", 0)):
            sel = valid & (L == zval)
            Ra = np.full(A.shape, np.nan, np.float32); Rc = np.full(A.shape, np.nan, np.float32)
            Ra[sel] = ss.rankdata(A[sel]); Rc[sel] = ss.rankdata(C[sel])
            r = {ax: (lag1(Ra, i), lag1(Rc, i)) for ax, i in (("time", 0), ("north_south", 1), ("east_west", 2))}
            f = np.prod([(1 - ra * rc) / (1 + ra * rc) for ra, rc in r.values()])
            factors[zname] = {"lag1": {k: {"a412": v[0], "chl": v[1]} for k, v in r.items()}, "factor": float(f)}
            del Ra, Rc
        res["neff_factors"] = factors

        t_idx, r_idx, c_idx = np.nonzero(valid & (L >= 0))
        lab = L[t_idx, r_idx, c_idx]
        a, ch = A[t_idx, r_idx, c_idx], C[t_idx, r_idx, c_idx]
        myv, monv = my[t_idx], mon[t_idx]
        blk = myv * ntile + tile[r_idx, c_idx]
        nblk = 120 * ntile
        unit_my = np.arange(nblk) // ntile           # block -> month-year
        unit_tile = np.arange(nblk)                  # block -> itself

        scopes = {"all": np.ones(len(a), bool), "May-Sep": np.isin(monv, [5, 6, 7, 8, 9])}
        scopes.update({c.MONTH_NAMES[m - 1]: monv == m for m in c.MONTHS})
        for sname, ssel in scopes.items():
            row = {}
            S = {}
            ok = True
            for zname, zval in (("in", 1), ("out", 0)):
                s = ssel & (lab == zval)
                n = int(s.sum())
                if n < 100:
                    ok = False
                    break
                rx, ry = ss.rankdata(a[s]), ss.rankdata(ch[s])
                S[zname] = block_sums(rx, ry, blk[s], nblk)
                row[f"rho_{zname}"] = float(corr_from(S[zname].sum(0, keepdims=True))[0])
                row[f"n_{zname}"] = n
                row[f"n_month_years_{zname}"] = int((np.bincount(myv[s], minlength=120) >= MIN_PX).sum())
            if not ok:
                res[sname] = {"skipped": "fewer than 100 pixel-days in a zone"}
                continue
            row["diff"] = row["rho_in"] - row["rho_out"]
            row["a_naive_fisher"] = fisher(row["rho_in"], row["n_in"], row["rho_out"], row["n_out"])
            row["b1_fisher_bretherton"] = fisher(row["rho_in"], row["n_in"] * factors["inside"]["factor"],
                                                 row["rho_out"], row["n_out"] * factors["outside"]["factor"])
            row["b2_fisher_monthyear_units"] = fisher(row["rho_in"], row["n_month_years_in"],
                                                      row["rho_out"], row["n_month_years_out"])
            row["c1_boot_monthyear"] = boot(S["in"], S["out"], unit_my, rng)
            row["c2_boot_monthyear_x_tile"] = boot(S["in"], S["out"], unit_tile, rng)
            # d. paired Wilcoxon over month-years
            diffs = []
            for k in np.unique(myv[ssel]):
                si = ssel & (myv == k) & (lab == 1)
                so = ssel & (myv == k) & (lab == 0)
                if si.sum() >= MIN_PX and so.sum() >= MIN_PX:
                    ri = ss.spearmanr(a[si], ch[si]).statistic
                    ro = ss.spearmanr(a[so], ch[so]).statistic
                    if np.isfinite(ri) and np.isfinite(ro):
                        diffs.append(ri - ro)
            diffs = np.array(diffs)
            row["d_wilcoxon_monthyears"] = {
                "n": int(len(diffs)), "median_diff": float(np.median(diffs)) if len(diffs) else None,
                "frac_negative": float((diffs < 0).mean()) if len(diffs) else None,
                "p": float(ss.wilcoxon(diffs).pvalue) if len(diffs) >= 6 else None}
            res[sname] = row
            print(method, sname, f"diff={row['diff']:.3f}", "CI", row["c1_boot_monthyear"]["ci95"], flush=True)
        out[method] = res
    json.dump(out, open(c.CACHE_DIR / "corr_diff.json", "w"), indent=1)
    write_md(out)
    print("corr_diff done")


def write_md(out):
    def pf(x):
        return "-" if x is None else ("<1e-300" if x == 0 else f"{x:.1e}")
    L = ["", "## 9. Significance of rho_in - rho_out (a_CDOM(412) vs Chl-a, Spearman)", "",
         "Script `experiments/redo/corr_diff.py`, output `outputs/redo/cache/corr_diff.json` (copied into results.json). "
         "Definitions are in the script docstring. a = naive Fisher z (anti-conservative); b1 = Fisher z with Bretherton-style "
         "n_eff = n * prod_d (1 - r_a,d r_c,d)/(1 + r_a,d r_c,d) over east-west, north-south and day-to-day lag-1 rank autocorrelations; "
         "b2 = Fisher z with n = month-years; c1/c2 = paired block bootstrap over month-years / month-year x 25x25 tiles (B=2000); "
         "d = Wilcoxon signed-rank over month-years with >= 30 px in each zone.", ""]
    for m in METHODS:
        f = out[m]["neff_factors"]
        L.append(f"### {m}\n")
        L.append(f"n_eff factor: inside {f['inside']['factor']:.2e}, outside {f['outside']['factor']:.2e} "
                 "(lag-1 rank autocorrelations a412/chl: " +
                 "; ".join(f"{z} " + ", ".join(f"{ax} {v['a412']:.2f}/{v['chl']:.2f}" for ax, v in f[z]['lag1'].items())
                           for z in ("inside", "outside")) + ")\n")
        L.append("| Scope | rho in | rho out | diff | n in / out | a naive p | b1 n_eff in/out, p | b2 n MY in/out, p | c1 MY boot 95% CI, p | c2 tile boot 95% CI, p | d Wilcoxon n, median diff, p |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for s, r in out[m].items():
            if s == "neff_factors":
                continue
            if "skipped" in r:
                L.append(f"| {s} | - | - | - | skipped (zone < 100 px) | | | | | | |")
                continue
            b1, b2, c1, c2, dd = r["b1_fisher_bretherton"], r["b2_fisher_monthyear_units"], r["c1_boot_monthyear"], r["c2_boot_monthyear_x_tile"], r["d_wilcoxon_monthyears"]
            L.append(f"| {s} | {r['rho_in']:.3f} | {r['rho_out']:.3f} | {r['diff']:+.3f} | {r['n_in']:,} / {r['n_out']:,} | "
                     f"{pf(r['a_naive_fisher']['p'])} | {(f"{b1['n_in']:.0f}/{b1['n_out']:.0f}, {pf(b1['p'])}") if b1 else 'n_eff<=3'} | "
                     f"{r['n_month_years_in']}/{r['n_month_years_out']}, {pf(b2['p']) if b2 else '-'} | "
                     f"[{c1['ci95'][0]:+.3f}, {c1['ci95'][1]:+.3f}], {pf(c1['p'])} | [{c2['ci95'][0]:+.3f}, {c2['ci95'][1]:+.3f}], {pf(c2['p'])} | "
                     f"{dd['n']}, {dd['median_diff'] if dd['median_diff'] is None else round(dd['median_diff'], 3)}, {pf(dd['p'])} |")
        L.append("")
    L += ["Notes: scopes where a zone has fewer than 5 month-years (IF Jan: 1, Feb: 2) are not testable; with so few resampling "
          "units the bootstrap CI need not even contain the point estimate. The Bretherton n_eff is extremely conservative here "
          "(spatial lag-1 rank autocorrelation ~0.99 gives n_eff of tens out of 10^5-10^7) and is only usable for the pooled scopes.",
          "",
          "**Recommendation for the paper:** report c1 (block bootstrap over whole month-years, 95% CI of rho_in - rho_out), "
          "supported by d (Wilcoxon over month-years). Both use the month-year, the unit at which zones are defined, as the "
          "independent sample, and neither needs a parametric autocorrelation model. Do not report naive p-values. "
          "Result: IF, rho is lower inside than outside, all months -0.035 [95% CI -0.049, -0.023], May-Sep -0.056 "
          "[-0.073, -0.040], p < 0.001 (Wilcoxon p = 2e-14, 3e-12; still p = 0.014 / 0.037 under the very conservative n_eff). "
          "-1 sigma, no difference pooled (-0.002 [-0.006, +0.001]); May-Sep +0.009 [+0.004, +0.014], negligible. The IF difference is "
          "statistically robust but small (rho 0.92 vs 0.97). It is consistent with the IF zone sitting in the high-a_CDOM range, "
          "where the algorithmic a_CDOM-Chl relationship bends (section 5), so it should not be read as a distinct biogeochemical regime.", ""]
    p = c.OUT_DIR / "results.md"
    txt = p.read_text(encoding="utf-8")
    marker = "\n## 9. Significance"
    if marker in txt:
        txt = txt[:txt.index(marker)]
    p.write_text(txt + "\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import sys
    if "--md-only" in sys.argv:   # re-append section 9 after report.py rewrote results.md
        write_md(json.load(open(c.CACHE_DIR / "corr_diff.json")))
    else:
        main()
