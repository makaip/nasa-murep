"""Paper figures with the 10 m depth cutoff applied in ALL cases -> outputs/rerun_10m/figs/.

Every inside/outside split uses deep pixels only (GEBCO depth >= 10 m, experiments/depth_cutoff.py):
pixels shallower than 10 m are in neither the upwelling nor the non-upwelling group. Maps draw them in
light gray ("Shallow (< 10 m)"), land in darker gray.

Classes: IF = pipeline5 monthly zones (outputs/redo/cache/zones/iforest.npz, identical to
figs/bayth_preclassified.pkl); -1 sigma = monthly rule on deep pixels (redo/sigma.py ->
zones/sigma.npz); Fig. 6 = week-of-year climatology rule on deep pixels. Daily data come from the
redo .npy caches of the v4 pickle (redo/common.py load_stacks). Plot styling follows
experiments/visualization (violins/all_years.py, scatterplots/*.py, yearly_class/*.py) and the
nasa-murep-local notebooks the -1 sigma figures came from (w_hist/histograms_no_fitting.ipynb,
feature_space/sst_upwelling_{video,pooled,all_years}.ipynb).

Violin KDEs use a random subsample of at most VIOLIN_MAX values per violin (after the 3-sd trim the
old scripts apply); exact medians are in paper_stats.md.

Usage: .venv\\Scripts\\python.exe yucatan\\experiments\\rerun_10m\\figures_10m.py [fig06 fig07 ... | all]
"""

import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "redo"))
sys.path.insert(0, str(HERE.parent / "visualization"))
import common as c  # noqa: E402
import depth_cutoff as dc  # noqa: E402
from plot_style import apply_nice_yaxis, nice_log_limits  # noqa: E402

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
import bayth_classification as bc  # noqa: E402

OUT = c.YUCATAN_DIR / "outputs" / "rerun_10m" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 300
VIOLIN_MAX = 200_000
SCATTER_MAX = 50_000
EXTENT = [c.BBOX_LON[0], c.BBOX_LON[1], c.BBOX_LAT[0], c.BBOX_LAT[1]]
MONTH_FULL = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
              "October", "November", "December"]
SHALLOW_COLOR = "#d9d9d9"
LAND_COLOR = "#8f8f8f"
MAYSEP = (5, 6, 7, 8, 9)

VARS = {  # key: (stack, axis label, file tag)
    "a412": ("cdom_412", r"$a_{\text{CDOM}}$ 412 $\text{ (m}^{-1}\text{)}$", "CDOM a(412)"),
    "s275": ("cdom_slope_275_295", r"$S_{275-295}$ $\text{ (nm}^{-1}\text{)}$", "CDOM S275:295"),
    "s300": ("cdom_slope_300_600", r"$S_{300-600}$ $\text{ (nm}^{-1}\text{)}$", "CDOM S300:600"),
    "chl": ("chlorophyll", r"$\log_{10}(\text{mg/m}^3)$", "Log10 Chlorophyll"),
}
VAR_ORDER = ["a412", "s275", "s300", "chl"]          # Figs 13-16 / S2-S5 order
SIGMA_LABELS = ("Other (≥ -1σ SST)", "Upwelling (< -1σ SST)")
IF_LABELS = ("Non-upwelling", "Upwelling")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
_D = {}


def data():
    if not _D:
        stacks, meta = c.load_stacks()
        _D["stacks"], _D["meta"] = stacks, meta
        _D["deep"] = c.deep_mask(meta)
        b = meta["bathymetry"]
        _D["shallow"] = ~_D["deep"] & np.isfinite(b)
        _D["lon2"], _D["lat2"] = np.meshgrid(meta["lon"], meta["lat"])
        _D["idx"] = c.month_index(meta)
        _D["zones"] = {m: c.load_zones(m) for m in ("iforest", "sigma")}
    return _D


def month_values(y, m, method):
    """Deep pixel-day values of month-year (y, m) split by the method's zone map.
    Returns {key: (inside values, outside values)} and the scatter pairs, or None if no data."""
    D = data()
    days = D["idx"][(y, m)]
    if len(days) == 0:
        return None
    zone = D["zones"][method][0][(y, m)]
    deep = D["deep"]
    inside = np.broadcast_to((zone == 1) & deep, (len(days),) + deep.shape)
    outside = np.broadcast_to((zone == 0) & deep, (len(days),) + deep.shape)
    a = {k: np.asarray(D["stacks"][VARS[k][0]][days], dtype=np.float64) for k in VARS}
    cd_ok = np.isfinite(a["a412"]) & np.isfinite(a["s275"]) & np.isfinite(a["s300"])
    with np.errstate(divide="ignore", invalid="ignore"):
        lchl = np.where(a["chl"] > 0, np.log10(a["chl"]), np.nan)
    chl_ok = np.isfinite(lchl)
    res = {}
    for k in ("a412", "s275", "s300"):
        res[k] = (a[k][cd_ok & inside], a[k][cd_ok & outside])
    res["chl"] = (lchl[chl_ok & inside], lchl[chl_ok & outside])
    sc_ok = np.isfinite(a["a412"]) & (a["a412"] > 0) & np.isfinite(a["chl"]) & (a["chl"] > 0)
    res["scatter"] = tuple((a["a412"][sc_ok & g].astype(np.float32), a["chl"][sc_ok & g].astype(np.float32))
                           for g in (inside, outside))
    res["has_data"] = bool((cd_ok & (inside | outside)).any() or (chl_ok & (inside | outside)).any())
    return res


# ---------------------------------------------------------------------------
# maps
# ---------------------------------------------------------------------------
def _map_axes(ax):
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, edgecolor="black", linewidth=0.8, zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=4)


def class_map_figure(zones_by_month, title, path, labels, colors=("#1f5fa8", "#c0392b")):
    """zones_by_month: {m: int8 zone grid (-1 not classified, 0 outside, 1 inside)}."""
    D = data()
    cmap = mcolors.ListedColormap([colors[0], colors[1], SHALLOW_COLOR])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, axes = plt.subplots(3, 4, figsize=(28, 10), subplot_kw={"projection": ccrs.PlateCarree()})
    any_none = False
    for m, ax in enumerate(axes.ravel(), 1):
        _map_axes(ax)
        z = zones_by_month[m]
        if not (z >= 0).any():
            ax.set_title(f"{MONTH_FULL[m - 1]}\nNo Data")
            continue
        disp = np.full(z.shape, np.nan)
        disp[z == 0] = 0
        disp[z == 1] = 1
        disp[D["shallow"]] = 2
        ax.pcolormesh(D["lon2"], D["lat2"], disp, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading="auto")
        if (z == 1).any():
            ax.set_title(MONTH_FULL[m - 1])
        else:
            ax.set_title(f"{MONTH_FULL[m - 1]} *")
            any_none = True
    if title:
        fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(left=0.03, right=0.86, bottom=0.04, top=0.88, wspace=0.06, hspace=0.15)
    if any_none:
        fig.text(0.03, 0.01, "* No upwelling zone classified this month", fontsize=12)
    handles = [Patch(facecolor=colors[1], edgecolor="black", label=labels[1]),
               Patch(facecolor=colors[0], edgecolor="black", label=labels[0]),
               Patch(facecolor=SHALLOW_COLOR, edgecolor="black", label=dc.SHALLOW_LABEL),
               Patch(facecolor=LAND_COLOR, edgecolor="black", label="Land")]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.865, 0.5), fontsize=14, frameon=False)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


_SCORES = {}


def if_scores():
    """{(y, m): IF anomaly score grid} (same scoring as the zones; scores exist for every valid pixel)."""
    if not _SCORES:
        D = data()
        preprocess, iforest, features = bc._load_model(c.MODEL_PATH)
        anom = {v: c.monthly_mean_anomaly(D["stacks"], D["meta"], v) for v in ("sst", "ssl")}
        for k in anom["sst"]:
            res, _ = bc._classify({"sst": anom["sst"][k], "ssl": anom["ssl"][k]}, preprocess, iforest, features,
                                  D["meta"]["bathymetry"])
            _SCORES[k] = None if res is None else res["scores"]
            if res is not None:  # consistency with the cached zones
                assert np.array_equal(res["classes"] == 1, D["zones"]["iforest"][0][k] == 1), k
    return _SCORES


def score_map_figure(year, path, vlim):
    D = data()
    S = if_scores()
    fig, axes = plt.subplots(3, 4, figsize=(28, 10), subplot_kw={"projection": ccrs.PlateCarree()})
    gray = mcolors.ListedColormap([SHALLOW_COLOR])
    mesh = None
    for m, ax in enumerate(axes.ravel(), 1):
        _map_axes(ax)
        s = S[(year, m)]
        if s is None:
            ax.set_title(f"{MONTH_FULL[m - 1]}\nNo Data")
            continue
        sd = np.where(D["deep"], s, np.nan)
        mesh = ax.pcolormesh(D["lon2"], D["lat2"], sd, cmap="magma", vmin=vlim[0], vmax=vlim[1],
                             transform=ccrs.PlateCarree(), shading="auto")
        ax.pcolormesh(D["lon2"], D["lat2"], np.where(D["shallow"], 1.0, np.nan), cmap=gray,
                      transform=ccrs.PlateCarree(), shading="auto")
        z = D["zones"]["iforest"][0][(year, m)]
        if (z == 1).any():
            ax.contour(D["lon2"], D["lat2"], (z == 1).astype(float), levels=[0.5], colors="cyan", linewidths=1.0,
                       transform=ccrs.PlateCarree(), zorder=2)
        ax.set_title(MONTH_FULL[m - 1])
    fig.suptitle(str(year), fontsize=16)
    fig.subplots_adjust(left=0.03, right=0.88, bottom=0.04, top=0.88, wspace=0.06, hspace=0.15)
    if mesh is not None:
        cax = fig.add_axes([0.895, 0.30, 0.012, 0.55])
        cb = fig.colorbar(mesh, cax=cax)
        cb.set_label("Isolation-forest anomaly score", fontsize=12)
    handles = [Line2D([], [], color="cyan", lw=1.5, label="IF upwelling class"),
               Patch(facecolor=SHALLOW_COLOR, edgecolor="black", label=dc.SHALLOW_LABEL),
               Patch(facecolor=LAND_COLOR, edgecolor="black", label="Land")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.885, 0.04), fontsize=11, frameon=False)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# violins
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)


def trim3(v):
    v = v[np.isfinite(v)]
    if v.size > 1:
        mu, sd = v.mean(), v.std()
        v = v[(v >= mu - 3 * sd) & (v <= mu + 3 * sd)]
    return v


def sub(v, n=VIOLIN_MAX):
    return v if v.size <= n else RNG.choice(v, n, replace=False)


def draw_violins(ax, per_month, y_lims, ylabel, labels, legend=True, no_data_wording=True):
    """per_month: {m: (inside, outside) trimmed arrays} or {m: None} for no data."""
    up_d, up_p, non_d, non_p, no_up, no_data = [], [], [], [], [], []
    for m in range(1, 13):
        pm = per_month.get(m)
        if pm is None:
            no_data.append(m)
            continue
        u, n = pm
        if u.size > 1:
            up_d.append(sub(u)); up_p.append(m - 0.2)
        else:
            no_up.append(m - 0.2)
        if n.size > 1:
            non_d.append(sub(n)); non_p.append(m + 0.2)
    for ds, pos, col in ((up_d, up_p, "red"), (non_d, non_p, "blue")):
        if not ds:
            continue
        vp = ax.violinplot(ds, positions=pos, widths=0.35, showmeans=False, showmedians=True, showextrema=True)
        for pc in vp["bodies"]:
            pc.set_facecolor(col); pc.set_edgecolor("black"); pc.set_alpha(0.60)
        for part in ("cbars", "cmins", "cmaxes", "cmedians"):
            vp[part].set_edgecolor("black"); vp[part].set_linewidth(1.2)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([n for n in c.MONTH_NAMES])
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.5, 12.5)
    if y_lims is not None:
        apply_nice_yaxis(ax, *y_lims)
    star = dict(marker="*", linestyle="none", color="black", markeredgecolor="black", markersize=14)
    if no_up:
        ax.plot(no_up, [0.05] * len(no_up), transform=ax.get_xaxis_transform(), clip_on=False, **star)
    for m in no_data:
        ax.text(m, 0.5, "No data", transform=ax.get_xaxis_transform(), ha="center", va="center", color="gray", fontsize=10)
    if legend:
        h = [Patch(facecolor="red", edgecolor="black", alpha=0.60, label=labels[1]),
             Patch(facecolor="blue", edgecolor="black", alpha=0.60, label=labels[0])]
        if no_up:
            h.append(Line2D([], [], label="No data", **star))
        ax.legend(handles=h, loc="upper right")


def year_violin_data(year, method):
    """{key: {m: (inside, outside) trimmed} or None}, plus scatter pairs per month."""
    out = {k: {} for k in VARS}
    scat = {}
    for m in c.MONTHS:
        r = month_values(year, m, method)
        if r is None or not r["has_data"]:
            for k in VARS:
                out[k][m] = None
            continue
        for k in VARS:
            out[k][m] = (trim3(r[k][0]), trim3(r[k][1]))
        scat[m] = r["scatter"]
    return out, scat


def limits_from(vdatas):
    lo = {k: np.inf for k in VARS}
    hi = {k: -np.inf for k in VARS}
    for vd in vdatas:
        for k in VARS:
            for pm in vd[k].values():
                if pm is None:
                    continue
                for v in pm:
                    if v.size > 1:
                        lo[k] = min(lo[k], v.min()); hi[k] = max(hi[k], v.max())
    return {k: (lo[k], hi[k]) for k in VARS}


def single_violin_figure(per_month, y_lims, ylabel, labels, path):
    fig, ax = plt.subplots(figsize=(16, 6))
    draw_violins(ax, per_month, y_lims, ylabel, labels)
    plt.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# scatters
# ---------------------------------------------------------------------------
def scatter_figure(left, right, titles, path):
    """left/right: (a412, chl) arrays."""
    def samp(p):
        n = len(p[0])
        if n > SCATTER_MAX:
            i = np.random.default_rng(42).choice(n, SCATTER_MAX, replace=False)
            return p[0][i], p[1][i]
        return p
    L, R = samp(left), samp(right)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for ax, (x, y), col, t in ((axes[0], L, "red", titles[0]), (axes[1], R, "blue", titles[1])):
        ax.scatter(x, y, c=col, alpha=0.8, marker="o", s=20, linewidths=0.2, edgecolors="black")
        ax.set_title(t)
        ax.set_xlabel(r"$a_{\text{CDOM}}$ 412 $(m^{-1})$")
        ax.set_yscale("log"); ax.set_xscale("log")
    axes[0].set_ylabel("Chlorophyll-a")
    xs = np.concatenate([L[0], R[0]])
    ys = np.concatenate([L[1], R[1]])
    if xs.size:
        xl = nice_log_limits(*np.percentile(xs, [0.5, 99.5]))
        yl = nice_log_limits(ys.min(), ys.max())
        if xl:
            axes[0].set_xlim(*xl)
        if yl:
            axes[0].set_ylim(*yl)
    plt.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def cat_pairs(pairs):
    pairs = [p for p in pairs if len(p[0])]
    if not pairs:
        return np.array([]), np.array([])
    return np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])


# ---------------------------------------------------------------------------
# individual figures
# ---------------------------------------------------------------------------
def fig06():
    """Weekly SST climatology histogram, -1 sigma (week-of-year rule) highlighted, deep pixels only."""
    D = data()
    sst = np.asarray(D["stacks"]["sst"])
    week = D["meta"]["dates"].isocalendar().week.to_numpy()
    rows = []
    thr = {}
    for w in range(1, 53):  # x axis 0.5-52.5, as the original (ISO week 53 falls outside)
        g = c.nanmean0(sst[week == w])
        v = g[D["deep"] & np.isfinite(g)]
        rows.append(pd.DataFrame({"week": w, "value": v}))
        thr[w] = v.mean() - v.std(ddof=1)
    df = pd.concat(rows)
    xb = np.arange(1, 54) - 0.5
    rng_y = [df["value"].min(), df["value"].max()]
    h, xe, ye = np.histogram2d(df["week"], df["value"], bins=[xb, 100], range=[[0.5, 52.5], rng_y])
    yc = (ye[:-1] + ye[1:]) / 2
    hl = np.zeros_like(h)
    for i, w in enumerate(range(1, 53)):
        mask = yc <= thr[w]
        hl[i, mask] = h[i, mask]
    fig = plt.figure(figsize=(10, 6))
    plt.imshow(h.T, origin="lower", aspect="auto", extent=[0.5, 52.5, ye[0], ye[-1]], cmap="bone_r")
    cax = plt.imshow(np.ma.masked_where(hl == 0, hl).T, origin="lower", aspect="auto",
                     extent=[0.5, 52.5, ye[0], ye[-1]], cmap="jet")
    plt.colorbar(cax, label=r"Frequency (Region $\mathbf{\leq} -1\sigma$)")
    plt.xlabel("Week of Year")
    plt.ylabel("SST (°C)")
    plt.tight_layout()
    fig.savefig(OUT / "fig06.png", dpi=DPI)
    plt.close(fig)
    flagged = sum(int((df.loc[df.week == w, "value"] < thr[w]).sum()) for w in thr)
    return {"fig06_pct_deep_pixel_weeks_below_threshold_weeks1to52": 100 * flagged / len(df)}


def fig07():
    z = data()["zones"]["sigma"][0]
    class_map_figure({m: z[(2018, m)] for m in c.MONTHS}, None, OUT / "fig07.png", SIGMA_LABELS,
                     colors=("blue", "red"))


def if_class_map(year, path):
    z = data()["zones"]["iforest"][0]
    class_map_figure({m: z[(year, m)] for m in c.MONTHS}, str(year), path, IF_LABELS)


def fig08():
    if_class_map(2018, OUT / "fig08.png")


def fig10():
    clim = data()["zones"]["iforest"][1]
    class_map_figure(clim, f"Climatology {c.YEARS[0]}-{c.YEARS[-1]}", OUT / "fig10.png", IF_LABELS)


def fig12():
    """-1 sigma violins pooled 2010-2019, 4 panels (as sst_upwelling_pooled.ipynb), deep pixels only."""
    per = {k: {} for k in VARS}
    for m in c.MONTHS:
        acc = {k: ([], []) for k in VARS}
        for y in c.YEARS:
            r = month_values(y, m, "sigma")
            if r is None:
                continue
            for k in VARS:
                acc[k][0].append(r[k][0]); acc[k][1].append(r[k][1])
        for k in VARS:
            per[k][m] = tuple(sub(trim3(np.concatenate(a)), 50_000) for a in acc[k])
    fig, axes = plt.subplots(4, 1, figsize=(16, 20), sharex=True)
    for ax, k in zip(axes, VAR_ORDER):
        draw_violins(ax, per[k], None, VARS[k][1], SIGMA_LABELS, legend=(k == "a412"))
        ax.grid(True, alpha=0.25, linestyle="--")
    axes[3].set_xlabel("Month (2010-2019 pooled)", fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT / "fig12.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def sigma_yearly(limits_only=False):
    """S2-S5: -1 sigma violins per year, one figure per variable with a panel per year (shared y limits)."""
    vds = {y: year_violin_data(y, "sigma")[0] for y in c.YEARS}
    lim = limits_from(vds.values())
    for j, k in enumerate(VAR_ORDER):
        fig, axes = plt.subplots(len(c.YEARS), 1, figsize=(16, 3.6 * len(c.YEARS)), sharex=True)
        for ax, y in zip(axes, c.YEARS):
            draw_violins(ax, vds[y][k], lim[k], VARS[k][1], SIGMA_LABELS, legend=(y == c.YEARS[0]))
            ax.text(0.005, 0.95, str(y), transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
        plt.tight_layout()
        fig.savefig(OUT / f"S{2 + j:02d}.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)


def _year_job(args):
    """Per-year IF violins/scatters (workers). Returns trimmed violin data and scatter pairs."""
    year = args
    vd, scat = year_violin_data(year, "iforest")
    # calendar-season scatter (all deep pixels, no zone)
    season = {True: [], False: []}
    for m, (ins, outs) in scat.items():
        season[m in MAYSEP].append(cat_pairs([ins, outs]))
    return year, vd, scat, {k: cat_pairs(v) for k, v in season.items()}


def per_year_and_pooled(years_for_s=True):
    with ProcessPoolExecutor(max_workers=10) as ex:
        res = {y: (vd, scat, season) for y, vd, scat, season in ex.map(_year_job, c.YEARS)}
    lim = limits_from(r[0] for r in res.values())
    json.dump({k: [float(a), float(b)] for k, (a, b) in lim.items()}, open(OUT.parent / "logs" / "violin_limits_if.json", "w"))
    # Figs 13-16: 2018 IF violins
    for j, k in enumerate(VAR_ORDER):
        single_violin_figure(res[2018][0][k], lim[k], VARS[k][1], IF_LABELS, OUT / f"fig{13 + j}.png")
    # Figs 17-20: scatters
    s18 = res[2018][1]
    scatter_figure(s18[5][0], s18[5][1], ("Inside Upwelling", "Outside Upwelling"), OUT / "fig17.png")
    for fig_n, months in ((18, MAYSEP), (19, tuple(m for m in c.MONTHS if m not in MAYSEP)), (20, tuple(c.MONTHS))):
        ins = cat_pairs([r[1][m][0] for r in res.values() for m in months if m in r[1]])
        outs = cat_pairs([r[1][m][1] for r in res.values() for m in months if m in r[1]])
        scatter_figure(ins, outs, ("Inside Upwelling", "Outside Upwelling"), OUT / f"fig{fig_n}.png")
    if not years_for_s:
        return
    # S6-S85: per year: class map, score map, 4 violins, season scatter, inside/outside scatter (May-Sep)
    S = if_scores()
    allv = np.concatenate([s[data()["deep"] & np.isfinite(s)] for s in S.values() if s is not None])
    vlim = tuple(np.percentile(allv, [1, 99]))
    for i, y in enumerate(c.YEARS):
        base = 6 + 8 * i
        if_class_map(y, OUT / f"S{base:02d}.png")
        score_map_figure(y, OUT / f"S{base + 1:02d}.png", vlim)
        vd, scat, season = res[y]
        for j, k in enumerate(VAR_ORDER):
            single_violin_figure(vd[k], lim[k], VARS[k][1], IF_LABELS, OUT / f"S{base + 2 + j:02d}.png")
        scatter_figure(season[True], season[False], ("Upwelling Months (May-Sep)", "Non-upwelling Months (Oct-Apr)"),
                       OUT / f"S{base + 6:02d}.png")
        ins = cat_pairs([scat[m][0] for m in MAYSEP if m in scat])
        outs = cat_pairs([scat[m][1] for m in MAYSEP if m in scat])
        scatter_figure(ins, outs, ("Inside Upwelling", "Outside Upwelling"), OUT / f"S{base + 7:02d}.png")
        print("S figs", y, flush=True)
    # Fig. 9 = IF anomaly score 2018 (same figure as its supplemental copy)
    import shutil
    shutil.copyfile(OUT / f"S{6 + 8 * (2018 - c.YEARS[0]) + 1:02d}.png", OUT / "fig09.png")


def main(which):
    info = {}
    steps = {"fig06": fig06, "fig07": fig07, "fig08": fig08, "fig10": fig10, "fig12": fig12,
             "S02-S05": sigma_yearly, "if_violins_scatters_S06-S85": per_year_and_pooled}
    for name, fn in steps.items():
        if which and name not in which and "all" not in which:
            continue
        print("==", name, flush=True)
        r = fn()
        if isinstance(r, dict):
            info.update(r)
    if info:
        p = OUT.parent / "logs" / "figures_info.json"
        old = json.load(open(p)) if p.exists() else {}
        old.update(info)
        json.dump(old, open(p, "w"), indent=1)
    print("figures done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["all"])
