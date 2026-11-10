"""Figures for the redo analysis -> outputs/redo/figs (PNG + PDF, 600 dpi).

  * violins_<method>.png : inside vs outside by calendar month (2010-2019 pooled) for
    a_CDOM(412), S275-295, S300-600, log10 Chl-a, styled like the original per-year
    violins (visualization/violins/all_years.py): red/blue bodies, black median and
    extrema bars, 3-sigma trim, black star = no data inside the zone that month.
    Display only: each violin is drawn from a random subsample of <= 20k values (trimmed
    min/max kept); statistics use all values.
  * scatter_a412_chl_loglog.png : a_CDOM(412) vs Chl-a, log-log, inside vs outside, for
    -1 sigma and IF, pooled and May 2018.
  * zonemap_<year>_<month>.png : -1 sigma vs IF zone maps side by side on monthly-mean
    SST (cmocean 'ice', so the red zone stands out); no bathymetry contour.
"""

import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import common as c  # noqa: E402
import stats as st  # noqa: E402

sys.path.insert(0, str(c.YUCATAN_DIR / "experiments" / "visualization"))
from plot_style import apply_nice_yaxis  # noqa: E402

warnings.filterwarnings("ignore")
plt.rcParams.update({"axes.grid": False, "font.size": 10})
RNG = np.random.default_rng(0)
METHOD_LABEL = {"sigma": r"$-1\sigma$ SST", "iforest": "Isolation forest",
                "kmeans": "K-means (k=3, global)", "kmeans_month": "K-means (k=3, per month)"}
VAR_LABEL = {"a412": r"$a_{\mathrm{CDOM}}(412)$ (m$^{-1}$)", "s275": r"$S_{275-295}$ (nm$^{-1}$)",
             "s300": r"$S_{300-600}$ (nm$^{-1}$)", "log10_chl": r"$\log_{10}$ Chl-a (mg m$^{-3}$)"}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(c.FIG_DIR / f"{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def _prep(v):
    """3-sigma trim (as in visualization/violins/all_years.py), then subsample for drawing.

    The subsample keeps the trimmed min and max so the extrema bars are exact."""
    if v.size > 1:
        mu, sd = v.mean(), v.std()
        v = v[(v >= mu - 3 * sd) & (v <= mu + 3 * sd)]
    if v.size > 20_000:
        keep = RNG.choice(v.size, 20_000, replace=False)
        v = np.concatenate([v[keep], [v.min(), v.max()]])
    return v


# Same look as the original per-year violins (visualization/violins/all_years.py).
UP_COLOR, NON_COLOR = "red", "blue"
STAR = dict(marker="*", linestyle="none", color="black", markeredgecolor="black", markersize=14)


def violins(method):
    P = st.pixel_table()
    L = st.labels_for(method)
    month = np.broadcast_to(P["month"][:, None], L.shape)
    fig, axes = plt.subplots(4, 1, figsize=(16, 20), sharex=True)
    any_star = False
    for ax, (key, A) in zip(axes, P["vals"].items()):
        ok = np.isfinite(A) & (L >= 0)
        v, ins, mon = A[ok], L[ok] == 1, month[ok]
        lo_all, hi_all = np.inf, -np.inf
        star_pos = []
        for inside, dx, color in ((True, -0.2, UP_COLOR), (False, 0.2, NON_COLOR)):
            data, pos = [], []
            for m in c.MONTHS:
                vv = v[(mon == m) & (ins == inside)]
                vv = _prep(vv) if vv.size > 1 else vv
                if vv.size < 2:
                    if inside:
                        star_pos.append(m + dx)
                    continue
                data.append(vv)
                pos.append(m + dx)
                lo_all, hi_all = min(lo_all, vv.min()), max(hi_all, vv.max())
            if data:
                vp = ax.violinplot(data, positions=pos, widths=0.35, showmeans=False,
                                   showmedians=True, showextrema=True)
                for b in vp["bodies"]:
                    b.set_facecolor(color); b.set_edgecolor("black"); b.set_alpha(0.60)
                for part in ("cbars", "cmins", "cmaxes", "cmedians"):
                    vp[part].set_edgecolor("black"); vp[part].set_linewidth(1.2)
        apply_nice_yaxis(ax, lo_all, hi_all)
        ax.set_ylabel(VAR_LABEL[key])
        if star_pos:
            any_star = True
            ax.plot(star_pos, [0.05] * len(star_pos), transform=ax.get_xaxis_transform(),
                    clip_on=False, **STAR)
    axes[-1].set_xticks(c.MONTHS)
    axes[-1].set_xticklabels(c.MONTH_NAMES)
    axes[-1].set_xlim(0.5, 12.5)
    handles = [Patch(facecolor=UP_COLOR, edgecolor="black", alpha=0.60, label="Upwelling"),
               Patch(facecolor=NON_COLOR, edgecolor="black", alpha=0.60, label="Non-upwelling")]
    if any_star:
        handles.append(Line2D([], [], label="No data", **STAR))
    axes[0].legend(handles=handles, loc="upper right")
    axes[0].set_title(f"{METHOD_LABEL[method]}: 2010-2019 pooled, depth $\\geq$ 10 m")
    fig.tight_layout()
    save(fig, f"violins_{method}")


A412_CLIP = 10 ** -3.5   # scatter only: drop a_CDOM(412) below this (also excluded from n and rho)


def scatter():
    P = st.pixel_table()
    A, C = P["vals"]["a412"], P["vals"]["log10_chl"]
    my = np.broadcast_to(P["my"][:, None], A.shape)
    may2018 = (2018 - c.YEARS[0]) * 12 + 4
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    for row, method in enumerate(("sigma", "iforest")):
        L = st.labels_for(method)
        ok = np.isfinite(A) & np.isfinite(C) & (A >= A412_CLIP) & (L >= 0)
        a, chl, ins, mm = A[ok], 10 ** C[ok], L[ok] == 1, my[ok]
        for col, (title, sel) in enumerate((("Pooled 2010-2019", np.ones(len(a), bool)),
                                            ("May 2018", mm == may2018))):
            ax = axes[row, col]
            txt = []
            for inside, color, lab in ((False, "#2e6db4", "Outside"), (True, "#c0392b", "Inside")):
                s = sel & (ins == inside)
                idx = np.where(s)[0]
                if idx.size == 0:
                    txt.append(f"{lab}: n=0")
                    continue
                pick = RNG.choice(idx, min(15_000, idx.size), replace=False)
                ax.scatter(chl[pick], a[pick], s=2, alpha=0.25, color=color, linewidths=0, rasterized=True)
                rho = st.ss.spearmanr(a[idx], chl[idx]).statistic if idx.size > 10 else np.nan
                txt.append(f"{lab}: n={idx.size:,}, $\\rho$={rho:.2f}")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title(f"{METHOD_LABEL[method]} - {title}")
            ax.text(0.03, 0.97, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("Chl-a (mg m$^{-3}$)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$a_{\mathrm{CDOM}}(412)$ (m$^{-1}$)")
    # Zoom on the bulk of the points; keep the automatic a_CDOM upper bound.
    axes[0, 0].set_xlim(10 ** -1.75, 10 ** 1.25)
    axes[0, 0].set_ylim(10 ** -2.75, axes[0, 0].get_ylim()[1])
    axes[0, 1].legend(handles=[Patch(color="#c0392b", label="Inside zone"),
                               Patch(color="#2e6db4", label="Outside zone")], loc="lower right", frameon=False)
    fig.tight_layout()
    save(fig, "scatter_a412_chl_loglog")


def zonemap(year=2018, month=7):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cmocean
    stacks, meta = c.load_stacks()
    lon2, lat2 = np.meshgrid(meta["lon"], meta["lat"])
    idx = c.month_index(meta)[(year, month)]
    sst = c.nanmean0(np.asarray(stacks["sst"][idx], dtype=float))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={"projection": ccrs.PlateCarree()})
    for ax, method in zip(axes, ("sigma", "iforest")):
        z = c.load_zones(method)[0][(year, month)]
        ax.set_extent([lon2.min(), lon2.max(), lat2.min(), lat2.max()], crs=ccrs.PlateCarree())
        mesh = ax.pcolormesh(lon2, lat2, sst, cmap=cmocean.cm.ice, shading="auto", transform=ccrs.PlateCarree())
        ax.contour(lon2, lat2, (z == 1).astype(float), levels=[0.5], colors="#d62728", linewidths=1.2,
                   transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", linewidth=0.6, zorder=3)
        gl = ax.gridlines(draw_labels=True, linewidth=0)
        gl.top_labels = gl.right_labels = False
        ax.set_title(f"{METHOD_LABEL[method]}: {c.MONTH_NAMES[month-1]} {year} ({int((z == 1).sum()):,} px)")
    cb = fig.colorbar(mesh, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("Monthly mean SST (°C)")
    save(fig, f"zonemap_{year}_{month:02d}")


if __name__ == "__main__":
    zonemap(2018, 7)
    zonemap(2018, 5)
    scatter()
    for m in ("sigma", "iforest", "kmeans", "kmeans_month"):
        violins(m)
    print("figures done")
