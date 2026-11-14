"""Summary statistics for the 2026-11-10 paper draft -> outputs/redo/paper_stats.{md,json}.

Setup (current figures): v4 data, 2010-2019, pipeline5 monthly IF zones (90th pct, SST z < -0.5,
SSL z < 1.0, depth >= 10 m; outputs/redo/cache/zones/iforest.npz == figs/bayth_preclassified.pkl,
0/120 month-years differ) and the redo -1 sigma rule (redo/sigma.py). Daily pixels take the zone
map of their own month-year.

Every inside/outside comparison is given two ways:
  (a) outside = every grid pixel with data that is not inside (what the main figures do: the
      scatterplots/violins bbox covers the whole 250x250 grid, incl. land/shallow < 10 m pixels)
  (b) outside = deep (>= 10 m) classified non-upwelling pixels only (redo convention)

Distribution statistics (medians, IQR, Cliff's delta, bootstrap CIs) are computed from per
month-year histograms with NB bins (a_CDOM binned in log10, Chl already log10, slopes linear),
so quantiles are exact to within one bin (< 0.2% of the range). Cliff's delta treats values in
the same bin as ties. Mann-Whitney p-values are not reported: with 10^5-10^7 pixel-days per
group every p is < 1e-300 and pixel-days are strongly autocorrelated, so the uncertainty is
given as a block-bootstrap 95% CI of the median difference instead (blocks = years for a single
calendar month pooled over 2010-2019, month-years for May-Sep pooled; a single month-year has
no CI). B = 2000.

Correlations (a_CDOM(412) vs Chl-a): pixel-days with finite a_CDOM >= 10^-3.5 and Chl > 0.
Pearson is on log10(a_CDOM) vs log10(Chl); Spearman on the raw values (rank-invariant to log).
CIs are a paired month-year block bootstrap of r_in - r_out (B = 2000); for Spearman, ranks are
computed once per scope and group and the bootstrap uses the Pearson correlation of those ranks
(exact for the sample, as in redo/corr_diff.py).
"""

import json
import sys
import warnings

import numpy as np
from scipy import stats as ss

import common as c
import sigma_variants as sv

sys.path.insert(0, str(c.YUCATAN_DIR / "experiments" / "visualization"))
warnings.filterwarnings("ignore")
import bayth_classification as bc  # noqa: E402

NB = 4000
B = 2000
RNG = np.random.default_rng(42)
A412_CLIP = 10 ** -3.5
METHODS = ["sigma", "iforest"]
GROUPS = ["in", "out_a", "out_b"]          # out_a: all not-inside; out_b: deep non-upwelling only
VARS = {"a412": "cdom_412", "s275": "cdom_slope_275_295", "s300": "cdom_slope_300_600",
        "log10_chl": "chlorophyll"}
MAYSEP = [5, 6, 7, 8, 9]
NK = len(c.YEARS) * 12
KEYS = [(y, m) for y in c.YEARS for m in c.MONTHS]
OUT_OF = {"out_a": "(a) all not-inside", "out_b": "(b) deep non-upwelling"}


def transform(key, a):
    """Value used for binning: log10 for a412 and Chl (Chl > 0 only), raw for slopes."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if key == "a412":
            return np.where(a > 0, np.log10(a), np.nan)
        if key == "log10_chl":
            return np.where(a > 0, np.log10(a), np.nan)
    return a


def back(key, t):
    return 10 ** t if key == "a412" else t


# ---------------------------------------------------------------------------
# 1. one pass over the data: histograms per (method, var, group, month-year) + paired corr arrays
# ---------------------------------------------------------------------------
def build(stacks, meta, zones):
    idx = c.month_index(meta)
    edges = {}
    for key, var in VARS.items():
        s = np.asarray(stacks[var][::30], dtype=np.float64)
        t = transform(key, s)
        lo, hi = np.nanpercentile(t, [1e-4, 100 - 1e-4])
        pad = 0.01 * (hi - lo)
        edges[key] = (lo - pad, hi + pad)
    H = {m: {k: np.zeros((3, NK, NB), np.int64) for k in VARS} for m in METHODS}
    pair = {"x": [], "y": [], "my": [], **{m: [] for m in METHODS}}
    n_clipped = {"deep": 0, "all": 0}
    deep = c.deep_mask(meta)
    for k, key_ym in enumerate(KEYS):
        days = idx[key_ym]
        if len(days) == 0:
            continue
        Z = {m: np.broadcast_to(zones[m][key_ym], (len(days),) + deep.shape) for m in METHODS}
        tvals = {}
        for key, var in VARS.items():
            t = transform(key, np.asarray(stacks[var][days], dtype=np.float64))
            ok = np.isfinite(t)
            lo, hi = edges[key]
            b = np.clip(((t[ok] - lo) / (hi - lo) * NB).astype(np.int64), 0, NB - 1)
            for m in METHODS:
                z = Z[m][ok]
                for g, sel in enumerate((z == 1, z != 1, z == 0)):
                    H[m][key][g, k] = np.bincount(b[sel], minlength=NB)
            tvals[key] = t
        la, lc = tvals["a412"], tvals["log10_chl"]
        ok = np.isfinite(la) & np.isfinite(lc)
        clip = ok & (la < np.log10(A412_CLIP))
        n_clipped["all"] += int(clip.sum())
        n_clipped["deep"] += int((clip & deep[None]).sum())
        ok &= ~clip
        pair["x"].append(la[ok].astype(np.float32))
        pair["y"].append(lc[ok].astype(np.float32))
        pair["my"].append(np.full(int(ok.sum()), k, np.int16))
        for m in METHODS:
            pair[m].append(Z[m][ok].astype(np.int8))
        print("month-year", key_ym, flush=True)
    pair = {k: np.concatenate(v) for k, v in pair.items()}
    return H, edges, pair, n_clipped


# ---------------------------------------------------------------------------
# histogram helpers
# ---------------------------------------------------------------------------
def quant(h, q, key, edges):
    """Quantiles from histogram(s). h: (..., NB)."""
    lo, hi = edges[key]
    w = (hi - lo) / NB
    cum = np.cumsum(h, axis=-1, dtype=np.float64)
    n = cum[..., -1:]
    out = []
    for qq in q:
        tgt = qq * n
        i = np.minimum((cum < tgt).sum(axis=-1, keepdims=True), NB - 1)
        prev = np.where(i > 0, np.take_along_axis(cum, np.maximum(i - 1, 0), -1), 0.0)
        hb = np.take_along_axis(h.astype(np.float64), i, -1)
        frac = np.where(hb > 0, (tgt - prev) / np.where(hb > 0, hb, 1), 0.5)
        val = lo + (i + frac) * w
        out.append(np.where(n > 0, back(key, val), np.nan)[..., 0])
    return out


def cliffs(hx, hy):
    nx, ny = hx.sum(), hy.sum()
    if nx == 0 or ny == 0:
        return None
    below = np.concatenate([[0], np.cumsum(hy)[:-1]]).astype(np.float64)
    above = ny - below - hy
    return float((hx * below).sum() - (hx * above).sum()) / (float(nx) * float(ny))


def boot_weights(U):
    ii = RNG.integers(0, U, (B, U))
    W = np.zeros((B, U))
    np.add.at(W, (np.repeat(np.arange(B), U), ii.ravel()), 1)
    return W


def compare(Hm, key, edges, ks, units):
    """Hm: (3, NK, NB) for one method/var. ks: month-year indices in scope; units: list of
    arrays of month-year indices forming bootstrap blocks (None -> no CI)."""
    h = Hm[:, ks].sum(axis=1)                                       # (3, NB)
    res = {}
    q1, med, q3 = quant(h, (0.25, 0.5, 0.75), key, edges)
    for g, name in enumerate(GROUPS):
        res[name] = {"n": int(h[g].sum()), "median": float(med[g]), "q1": float(q1[g]), "q3": float(q3[g])}
    W = boot_weights(len(units)) if units is not None and len(units) >= 3 else None
    if W is not None:
        Hu = np.stack([Hm[:, u].sum(axis=1) for u in units], axis=1).astype(np.float64)  # (3, U, NB)
        bmed = {g: quant(W @ Hu[g], (0.5,), key, edges)[0] for g in range(3)}
    for g, name in ((1, "out_a"), (2, "out_b")):
        d = {"median_diff_in_minus_out": res["in"]["median"] - res[name]["median"],
             "cliffs_delta_in_vs_out": cliffs(h[0], h[g])}
        if W is not None and res["in"]["n"] and res[name]["n"]:
            bd = bmed[0] - bmed[g]
            bd = bd[np.isfinite(bd)]
            d["median_diff_ci95"] = [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]
            d["ci_blocks"] = len(units)
        res["diff_" + name] = d
    return res


def violin_tables(H, edges, method, years=None, by_month=True):
    out = {}
    yrs = years or c.YEARS
    for key in VARS:
        Hm = H[method][key]
        rows = {}
        if by_month:
            for m in c.MONTHS:
                ks = [(y - c.YEARS[0]) * 12 + m - 1 for y in yrs]
                units = [np.array([k]) for k in ks] if len(ks) > 1 else None
                rows[m] = compare(Hm, key, edges, ks, units)
        ks = [(y - c.YEARS[0]) * 12 + m - 1 for y in yrs for m in MAYSEP]
        rows["May-Sep"] = compare(Hm, key, edges, ks, [np.array([k]) for k in ks])
        out[key] = rows
    return out


# ---------------------------------------------------------------------------
# correlations
# ---------------------------------------------------------------------------
def suff(x, y, my):
    return np.stack([np.bincount(my, minlength=NK).astype(np.float64)] +
                    [np.bincount(my, weights=w, minlength=NK) for w in (x, y, x * x, y * y, x * y)], axis=1)


def r_from(S):
    n, sx, sy, sxx, syy, sxy = np.moveaxis(S, -1, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sy / n
        return cov / np.sqrt((sxx - sx ** 2 / n) * (syy - sy ** 2 / n))


def corr_scope(pair, method, ks):
    my = pair["my"].astype(np.int64)
    scope = np.isin(my, ks)
    z = pair[method]
    res = {}
    stats_ = {}
    for name, sel in (("in", z == 1), ("out_a", z != 1), ("out_b", z == 0)):
        s = scope & sel
        x, y, m = pair["x"][s].astype(np.float64), pair["y"][s].astype(np.float64), my[s]
        if len(x) < 10:
            res[name] = {"n": int(len(x))}
            continue
        rx, ry = ss.rankdata(x), ss.rankdata(y)
        Sp, Ss = suff(x, y, m), suff(rx, ry, m)
        stats_[name] = (Sp, Ss)
        n_my = int((np.bincount(m, minlength=NK) >= 30).sum())
        res[name] = {"n": int(len(x)), "n_month_years_ge30": n_my,
                     "pearson_log10": float(r_from(Sp.sum(0))), "spearman": float(r_from(Ss.sum(0)))}
    if "in" in stats_ and len(ks) >= 3:
        W = boot_weights(len(ks))
        for out in ("out_a", "out_b"):
            if out not in stats_:
                continue
            d = {}
            for j, lab in enumerate(("pearson_log10", "spearman")):
                Si, So = stats_["in"][j][ks], stats_[out][j][ks]
                bd = r_from(W @ Si) - r_from(W @ So)
                bd = bd[np.isfinite(bd)]
                d[lab] = {"diff": res["in"][lab] - res[out][lab],
                          "ci95": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
                          "p_two_sided": float(max(1 / B, 2 * min((bd <= 0).mean(), (bd >= 0).mean())))}
            d["blocks"] = len(ks)
            res["diff_" + out] = d
    elif "in" in stats_:
        for out in ("out_a", "out_b"):
            if out in res and "spearman" in res[out]:
                res["diff_" + out] = {lab: {"diff": res["in"][lab] - res[out][lab]}
                                      for lab in ("pearson_log10", "spearman")}
    return res


# ---------------------------------------------------------------------------
# zone counts, slope, depth cutoffs
# ---------------------------------------------------------------------------
def zone_counts(meta):
    deep_n = int(c.deep_mask(meta).sum())
    by_month, clim = c.load_zones("iforest")
    out = {"deep_pixels": deep_n, "by_calendar_month": {}, "years_with_class": {}, "climatology": {}}
    for m in c.MONTHS:
        areas = {y: int((by_month[(y, m)] == 1).sum()) for y in c.YEARS}
        yrs = [y for y, a in areas.items() if a > 0]
        nz = [areas[y] for y in yrs]
        out["by_calendar_month"][m] = {
            "n_years_with_class": len(yrs), "years": yrs,
            "mean_px_all_years": float(np.mean(list(areas.values()))),
            "mean_pct_deep_all_years": 100 * float(np.mean(list(areas.values()))) / deep_n,
            "mean_px_years_with_class": float(np.mean(nz)) if nz else 0.0,
            "mean_pct_deep_years_with_class": 100 * float(np.mean(nz)) / deep_n if nz else 0.0,
            "area_by_year_px": areas}
        out["climatology"][m] = int((clim[m] == 1).sum())
    for y in c.YEARS:
        out["years_with_class"][y] = [m for m in c.MONTHS if (by_month[(y, m)] == 1).any()]
    return out


def slope_stats(meta):
    lon, lat = meta["lon"], meta["lat"]
    z = meta["bathymetry"]
    R = 6_371_000.0
    dy = np.gradient(lat) * np.pi / 180 * R
    dzdy = np.gradient(z, axis=0) / dy[:, None]
    lat2 = np.meshgrid(lon, lat)[1]
    dx = np.gradient(lon)[None, :] * np.pi / 180 * R * np.cos(np.deg2rad(lat2))
    dzdx = np.gradient(z, axis=1) / dx
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    by_month, clim = c.load_zones("iforest")
    clim_any = np.any(np.stack([clim[m] == 1 for m in c.MONTHS]), axis=0)
    ever = np.any(np.stack([g == 1 for g in by_month.values()]), axis=0)
    out = {"grid_spacing_deg": [float(np.median(np.diff(lon))), float(np.median(np.diff(lat)))],
           "grid_spacing_km_approx": [float(np.median(dx) / 1000), float(np.median(dy) / 1000)]}
    for name, mask in (("climatology_class_union", clim_any), ("any_month_year_class", ever),
                       ("all_deep_pixels", c.deep_mask(meta))):
        v = slope[mask & np.isfinite(slope)]
        out[name] = {"n_px": int(v.size), "max_deg": float(v.max()), "p95_deg": float(np.percentile(v, 95)),
                     "median_deg": float(np.median(v)), "pct_below_1deg": 100 * float((v < 1).mean())}
    return out


def depth_cutoffs(stacks, meta):
    idx = c.month_index(meta)
    preprocess, iforest, features = bc._load_model(c.MODEL_PATH)
    bathy = meta["bathymetry"]
    cached, _ = c.load_zones("iforest")
    out = {}
    orig = bc.BAYTH_THRESHOLD
    try:
        for m in c.MONTHS:
            an = {}
            for v in ("sst", "ssl"):
                s = np.asarray(stacks[v][idx[(2018, m)]], dtype=float)
                med = np.nanmedian(s, axis=(1, 2), keepdims=True)
                an[v] = c.nanmean0(s - med)
            row = {}
            for d in (5, 10, 20):
                bc.BAYTH_THRESHOLD = -d
                res, _ = bc._classify(an, preprocess, iforest, features, bathy)
                row[f"{d}m"] = 0 if res is None else int((res["classes"] == 1).sum())
            row["10m_matches_cache"] = row["10m"] == int((cached[(2018, m)] == 1).sum())
            row["deep_px"] = {f"{d}m": int((~np.isnan(bathy) & (bathy <= -d)).sum()) for d in (5, 10, 20)}
            out[m] = row
    finally:
        bc.BAYTH_THRESHOLD = orig
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
UNIT = {"a412": "m^-1", "s275": "nm^-1", "s300": "nm^-1", "log10_chl": "log10 mg m^-3"}
FMT = {"a412": "{:.4f}", "s275": "{:.5f}", "s300": "{:.5f}", "log10_chl": "{:.3f}"}


def fmt_ci(d):
    ci = d.get("median_diff_ci95")
    return "" if ci is None else f" [{ci[0]:+.4g}, {ci[1]:+.4g}]"


def violin_md(L, T, title, row_label):
    for key, rows in T.items():
        f = FMT[key]
        L.append(f"\n**{title}: {key} ({UNIT[key]})**\n")
        L.append(f"| {row_label} | n in | median in [IQR] | n out (a) | median out (a) [IQR] | diff (a) [95% CI] | delta (a) "
                 "| n out (b) | median out (b) [IQR] | diff (b) [95% CI] | delta (b) |")
        L.append("|" + "---|" * 11)
        for r, v in rows.items():
            name = c.MONTH_NAMES[r - 1] if isinstance(r, int) else r
            cells = [name]
            iq = lambda g: f"{f.format(v[g]['median'])} [{f.format(v[g]['q1'])}, {f.format(v[g]['q3'])}]" if v[g]["n"] else "-"
            cells += [f"{v['in']['n']:,}", iq("in")]
            for o in ("out_a", "out_b"):
                d = v["diff_" + o]
                dd = "-" if not v["in"]["n"] else f"{d['median_diff_in_minus_out']:+.4g}{fmt_ci(d)}"
                cd = "-" if d["cliffs_delta_in_vs_out"] is None else f"{d['cliffs_delta_in_vs_out']:+.3f}"
                cells += [f"{v[o]['n']:,}", iq(o), dd, cd]
            L.append("| " + " | ".join(cells) + " |")


def corr_md(L, C):
    L.append("| Method | Scope | group | n | month-years >= 30 px | Pearson log10 | Spearman | diff vs in: Pearson [95% CI] | Spearman [95% CI] |")
    L.append("|" + "---|" * 9)
    for meth, scopes in C.items():
        for sc, r in scopes.items():
            for g in GROUPS:
                x = r.get(g, {})
                if "spearman" not in x:
                    L.append(f"| {meth} | {sc} | {g} | {x.get('n', 0):,} | - | - | - | - | - |")
                    continue
                dcol = ["", ""]
                if g != "in" and "diff_" + g in r:
                    d = r["diff_" + g]
                    dcol = [f"{d[k]['diff']:+.3f}" + (f" [{d[k]['ci95'][0]:+.3f}, {d[k]['ci95'][1]:+.3f}]" if "ci95" in d[k] else "")
                            for k in ("pearson_log10", "spearman")]
                L.append(f"| {meth} | {sc} | {g} | {x['n']:,} | {x['n_month_years_ge30']} | {x['pearson_log10']:.3f} "
                         f"| {x['spearman']:.3f} | {dcol[0]} | {dcol[1]} |")


def main(out_dir=c.OUT_DIR, extra_zones=None, write_markdown=True):
    """extra_zones: {name: {(y, m): zone grid}} appended to METHODS (used by rerun_10m for comparisons)."""
    global METHODS
    stacks, meta = c.load_stacks()
    zones = {m: c.load_zones(m)[0] for m in METHODS}
    if extra_zones:
        zones.update(extra_zones)
        METHODS = METHODS + [k for k in extra_zones if k not in METHODS]
    out ={"setup": __doc__.split("\n\n")[1].replace("\n", " "), "bins": NB, "bootstrap_B": B}

    # 1. -1 sigma shares
    deep = c.deep_mask(meta)
    sig = zones["sigma"]
    out["sigma_share"] = {
        "redo_monthly_rule_pct_of_classified_deep_pixel_months":
            100 * sum(int((g == 1).sum()) for g in sig.values()) / sum(int((g >= 0).sum()) for g in sig.values()),
        "n_classified_pixel_months": sum(int((g >= 0).sum()) for g in sig.values()),
        **json.load(open(c.CACHE_DIR / "sigma_variants.json"))}
    print("sigma shares done", flush=True)

    # 5, 7
    out["if_zone_counts"] = zone_counts(meta)
    out["slope"] = slope_stats(meta)
    out["depth_cutoff_2018"] = depth_cutoffs(stacks, meta)
    print("zones/slope/depth done", flush=True)

    # 6
    km = json.load(open(c.CACHE_DIR / "kmeans_compare.json"))
    out["kmeans_sq_sep_share_pct"] = {v: {k: {f: round(100 * s, 1) for f, s in km[v]["by_k"][k]["sq_sep_share"].items()}
                                          for k in km[v]["by_k"]} for v in km if isinstance(km[v], dict) and "by_k" in km[v]}
    out["kmeans_raw_variance_share_pct"] = {v: {f: round(100 * s, 1) for f, s in km[v]["raw_variance_share"].items()}
                                            for v in km if isinstance(km[v], dict) and "raw_variance_share" in km[v]}

    # 2, 3, 4
    H, edges, pair, n_clipped = build(stacks, meta, zones)
    out["violins_sigma_pooled"] = violin_tables(H, edges, "sigma")
    out["violins_iforest_pooled"] = violin_tables(H, edges, "iforest")
    out["violins_iforest_2018"] = violin_tables(H, edges, "iforest", years=[2018])
    out["violins_sigma_2018"] = violin_tables(H, edges, "sigma", years=[2018])
    print("violin tables done", flush=True)
    allk = list(range(NK))
    scopes = {"all months (Fig 18)": allk,
              "May-Sep (Fig 19)": [k for k in allk if k % 12 + 1 in MAYSEP],
              "Oct-Apr (Fig 20)": [k for k in allk if k % 12 + 1 not in MAYSEP],
              "May 2018 (Fig 17)": [(2018 - c.YEARS[0]) * 12 + 4]}
    out["corr_a412_chl"] = {m: {s: corr_scope(pair, m, ks) for s, ks in scopes.items()} for m in METHODS}
    out["corr_a412_clip"] = {"threshold": A412_CLIP, "n_dropped_all_pixels": n_clipped["all"],
                             "n_dropped_deep_pixels": n_clipped["deep"]}
    for m in extra_zones or {}:
        out[f"violins_{m}_pooled"] = violin_tables(H, edges, m)
    if not write_markdown:
        return out
    json.dump(out, open(out_dir / "paper_stats.json", "w"), indent=1, default=float)
    write_md(out)
    print("paper_stats done", flush=True)
    return out


def write_md(o):
    L = ["# Paper summary statistics (for the 2026-11-10 draft)", "",
         "Generated by `yucatan/experiments/redo/paper_stats.py`; full numbers in `paper_stats.json`.", "",
         o["setup"], "",
         "Outside conventions: **(a)** every grid pixel with data that is not inside (main figures: violins/all_years.py, "
         "scatterplots/common.py; includes land-adjacent and < 10 m pixels); **(b)** deep (>= 10 m) classified "
         "non-upwelling pixels only (redo). n = pixel-days unless stated. delta = Cliff's delta (in vs out). "
         f"CIs: block bootstrap, B = {o['bootstrap_B']}.", ""]
    s = o["sigma_share"]
    L += ["## 1. Share of pixels flagged by -1 sigma", "",
          f"- Redo monthly rule (per month-year monthly-mean MUR SST < spatial mean - 1 sd, deep pixels; redo/sigma.py): "
          f"**{s['redo_monthly_rule_pct_of_classified_deep_pixel_months']:.2f}%** of {s['n_classified_pixel_months']:,} deep pixel-months.",
          f"- Old week-of-year climatology rule (52-53 climatological weekly maps, all pixels, no depth mask): "
          f"**{s['old_weekly_climatology_nodepth']:.2f}%** on v4 (13.20% on v3); with depth >= 10 m {s['old_weekly_climatology_depth10']:.2f}%.",
          f"- Other variants: monthly per month-year without depth mask {s['redo_monthly_per_month_year_nodepth']:.2f}%; "
          f"monthly climatology (12 maps), depth >= 10 m {s['monthly_climatology_depth10']:.2f}%.", ""]
    L += ["## 2. -1 sigma violins (pooled 2010-2019), by calendar month and May-Sep", "",
          "CI of the median difference: per calendar month, blocks = the 10 years; May-Sep, blocks = 50 month-years."]
    violin_md(L, o["violins_sigma_pooled"], "-1 sigma, pooled 2010-2019", "Month")
    L += ["", "## 3. IF violins: 2018 by month, and pooled 2010-2019", "",
          "2018 by month: a single month-year has no block CI. Pooled table: per calendar month (blocks = years) and May-Sep (blocks = month-years)."]
    violin_md(L, o["violins_iforest_2018"], "IF, 2018", "Month")
    violin_md(L, o["violins_iforest_pooled"], "IF, pooled 2010-2019", "Month")
    L += ["", "## 4. a_CDOM(412) vs Chl-a correlation, inside vs outside", "",
          f"Pixel-days with a_CDOM >= 10^-3.5 and Chl > 0 (clip drops {o['corr_a412_clip']['n_dropped_all_pixels']:,} pixel-days on the "
          f"whole grid, {o['corr_a412_clip']['n_dropped_deep_pixels']:,} of them deep). CI: paired month-year block bootstrap of r_in - r_out.", ""]
    corr_md(L, o["corr_a412_chl"])
    z = o["if_zone_counts"]
    L += ["", "## 5. IF class by calendar month (Table 2)", "",
          f"Deep (>= 10 m) pixels: {z['deep_pixels']:,}.", "",
          "| Month | years with class | years | mean area, all 10 yrs (px, % deep) | mean area, years with class (px, % deep) | climatology class (px) |",
          "|---|---|---|---|---|---|"]
    for m, r in z["by_calendar_month"].items():
        L.append(f"| {c.MONTH_NAMES[int(m) - 1]} | {r['n_years_with_class']} | {', '.join(map(str, r['years']))} | "
                 f"{r['mean_px_all_years']:,.0f} ({r['mean_pct_deep_all_years']:.2f}%) | "
                 f"{r['mean_px_years_with_class']:,.0f} ({r['mean_pct_deep_years_with_class']:.2f}%) | {z['climatology'][m]:,} |")
    L += ["", "Months with a class, by year: " + "; ".join(
        f"{y}: {', '.join(c.MONTH_NAMES[m - 1] for m in ms)}" for y, ms in z["years_with_class"].items()), ""]
    L += ["## 6. K-means: share of squared centroid separation per feature (%)", "",
          "From outputs/redo/cache/kmeans_compare.json (redo/kmeans_compare.py, v4, depth >= 10 m except a0).", "",
          "| Variant | k | log10 Chl | SST | SSL |", "|---|---|---|---|---|"]
    for v, ks in o["kmeans_sq_sep_share_pct"].items():
        for k, sh in ks.items():
            L.append(f"| {v} | {k} | {sh['log10_chl']} | {sh['sst']} | {sh['ssl']} |")
    L += ["", "Raw (unscaled) feature variance share (%): " + "; ".join(
        f"{v}: Chl {d['log10_chl']}, SST {d['sst']}, SSL {d['ssl']}" for v, d in o["kmeans_raw_variance_share_pct"].items()), ""]
    sl = o["slope"]
    L += ["## 7. Bathymetric slope (GEBCO on the 250 x 250 grid) and 2018 depth cutoffs", "",
          f"Grid spacing ~{sl['grid_spacing_km_approx'][0]:.2f} km (E-W) x {sl['grid_spacing_km_approx'][1]:.2f} km (N-S); slopes are "
          "for that resolution (finer GEBCO would give steeper local maxima).", "",
          "| Region | n px | max (deg) | 95th pct (deg) | median (deg) | % < 1 deg |", "|---|---|---|---|---|---|"]
    for k in ("climatology_class_union", "any_month_year_class", "all_deep_pixels"):
        r = sl[k]
        L.append(f"| {k} | {r['n_px']:,} | {r['max_deg']:.3f} | {r['p95_deg']:.3f} | {r['median_deg']:.3f} | {r['pct_below_1deg']:.1f} |")
    L += ["", "2018 IF class area (pixels) by depth cutoff (score threshold and z-scores unchanged; only the depth mask differs):", "",
          "| Month | >= 5 m | >= 10 m | >= 20 m | 10 m = current zones |", "|---|---|---|---|---|"]
    for m, r in o["depth_cutoff_2018"].items():
        L.append(f"| {c.MONTH_NAMES[int(m) - 1]} | {r['5m']:,} | {r['10m']:,} | {r['20m']:,} | {r['10m_matches_cache']} |")
    dp = next(iter(o["depth_cutoff_2018"].values()))["deep_px"]
    L += ["", f"Deep pixels per cutoff: 5 m {dp['5m']:,}, 10 m {dp['10m']:,}, 20 m {dp['20m']:,}.", ""]
    L += ["## 8. Methods claims checked against the code", "",
          "- Ocean colour: AQUA MODIS **L2 OC swath** files (E:\\satdata\\cdom\\...\\AQUA_MODIS.<date>T<time>.L2.OC.nc), "
          "variables read from `geophysical_data`, binned to the 250 x 250 grid as a **daily** mean of all swath pixels "
          "(data/preprocessing/pickler_processors.py:197-265, pickler_utils.py:61-77). Not L3 monthly/weekly composites. "
          "Weekly/monthly values in the analysis are means of these daily grids.",
          "- Quality flags: **no code applies l2_flags** (no reference to l2_flags/flags anywhere in data/ or experiments/). "
          "Cloud/land/glint removal is only what the L2 files already contain as fill values (decoded to NaN by xarray), "
          "plus range limits: Chl 0-100 mg m^-3 (pickler_config.py:48-49), a_CDOM(412) <= 1 m^-1 clip.",
          "- SST: **MUR L4** (MUR-JPL-L4-GLOB-v4.1, `analysed_sst`; pickler_config.py:16, 29) in all violins, maps, "
          "IF/sigma classification and K-means. No MODIS SST is used.",
          "- Figure 6 and the 2018 -1 sigma maps (Fig. 7) in the draft were made in nasa-murep-local (w_hist notebooks) "
          "with the week-of-year climatology rule, which classifies 13.2% of pixels (13.21% on v4 data); the redo monthly "
          "rule gives 12.87%.", ""]
    (c.OUT_DIR / "paper_stats.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
