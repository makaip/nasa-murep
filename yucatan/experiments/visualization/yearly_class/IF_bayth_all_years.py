"""Plot one Bayth classification map per year from the shared pickle."""

import os
from pathlib import Path
import sys

import numpy as np

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bayth_classification import PICKLE_PATH, deep_mask, load_preclassified

# YUC_FIGS_ROOT redirects output (default yucatan/figs). YUC_DEPTH10=1 (depth10-everywhere): < 10 m pixels are in
# neither group, so they are left blank instead of drawn as non-upwelling, and *_IF_anom.png score maps are drawn too.
OUT_DIR = Path(os.environ.get("YUC_FIGS_ROOT", Path(__file__).resolve().parents[3] / "figs")) / "bayth"
DEPTH10 = os.environ.get("YUC_DEPTH10") == "1"
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def plot_year(data, year):
    fig, axes = plt.subplots(3, 4, figsize=(28, 10), subplot_kw={"projection": ccrs.PlateCarree()}, dpi=400)
    cmap = mcolors.ListedColormap(["#1f5fa8", "#c0392b"])
    norm = mcolors.BoundaryNorm([-1.5, 0, 1.5], cmap.N)
    deep = deep_mask(data["bathymetry"])
    mesh = None
    any_no_upwelling = False
    for month, ax in enumerate(axes.ravel(), 1):
        record = data["by_year"][year].get(month)
        ax.set_extent([-92.20216, -85.61256, 19.57871, 22.86906], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", linewidth=0.8, zorder=3)
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=4)
        if record is None:
            ax.set_title(f"{MONTH_NAMES[month - 1]}\nNo Data")
            continue
        classes = np.where(deep, record["classes"], np.nan) if DEPTH10 else record["classes"]
        mesh = ax.pcolormesh(data["lon_grid"], data["lat_grid"], classes, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading="auto")
        if (record["classes"] == 1).any():
            ax.set_title(MONTH_NAMES[month - 1])
        else:
            # Asterisk: month has data but no upwelling zone was classified.
            ax.set_title(f"{MONTH_NAMES[month - 1]} *")
            any_no_upwelling = True
    fig.suptitle(str(year), fontsize=16)
    fig.subplots_adjust(left=0.03, right=0.90, bottom=0.04, top=0.88, wspace=0.06, hspace=0.15)
    if any_no_upwelling:
        fig.text(0.03, 0.01, "* No upwelling zone classified this month", fontsize=12)
    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), ticks=[-1, 1], fraction=0.02, pad=0.02)
        cbar.ax.set_yticklabels(["Non-upwelling", "Upwelling"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{year}_IF_class.png")
    plt.close(fig)


def score_limits(data):
    """Shared colour limits for the score maps: 1st-99th percentile of deep-pixel scores, all years."""
    deep = deep_mask(data["bathymetry"])
    v = np.concatenate([r["scores"][deep & np.isfinite(r["scores"])] for y in data["years"]
                        for r in data["by_year"][y].values() if r is not None])
    return np.percentile(v, [1, 99])


def plot_year_scores(data, year, vlim):
    """IF anomaly-score map in the class-map layout (no legend, no zone outline); < 10 m left blank."""
    fig, axes = plt.subplots(3, 4, figsize=(28, 10), subplot_kw={"projection": ccrs.PlateCarree()}, dpi=400)
    deep = deep_mask(data["bathymetry"])
    mesh = None
    for month, ax in enumerate(axes.ravel(), 1):
        record = data["by_year"][year].get(month)
        ax.set_extent([-92.20216, -85.61256, 19.57871, 22.86906], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", linewidth=0.8, zorder=3)
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=4)
        if record is None:
            ax.set_title(f"{MONTH_NAMES[month - 1]}\nNo Data")
            continue
        scores = np.where(deep, record["scores"], np.nan)
        mesh = ax.pcolormesh(data["lon_grid"], data["lat_grid"], scores, cmap="inferno", vmin=vlim[0], vmax=vlim[1],
                             transform=ccrs.PlateCarree(), shading="auto")
        ax.set_title(MONTH_NAMES[month - 1])
    fig.suptitle(str(year), fontsize=16)
    fig.subplots_adjust(left=0.03, right=0.90, bottom=0.04, top=0.88, wspace=0.06, hspace=0.15)
    if mesh is not None:
        fig.colorbar(mesh, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="Anomaly score")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{year}_IF_anom.png")
    plt.close(fig)


if __name__ == "__main__":
    data = load_preclassified(PICKLE_PATH)
    for year in data["years"]:
        plot_year(data, year)
    if DEPTH10:
        vlim = score_limits(data)
        for year in data["years"]:
            plot_year_scores(data, year, vlim)
