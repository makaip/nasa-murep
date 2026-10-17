"""Plot the preclassified Bayth data for one year."""

from pathlib import Path
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bayth_classification import PICKLE_PATH, load_preclassified

OUT_DIR = Path(__file__).resolve().parents[3] / "figs" / "bayth"
TARGET_YEAR = 2018
LON_MIN, LON_MAX = -92.20216, -85.61256
LAT_MIN, LAT_MAX = 19.57871, 22.86906
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def plot_year(data, year):
    records = data["by_year"][year]
    fig, axes = plt.subplots(3, 4, figsize=(28, 10), subplot_kw={"projection": ccrs.PlateCarree()}, dpi=400)
    cmap = mcolors.ListedColormap(["#1f5fa8", "#c0392b"])
    norm = mcolors.BoundaryNorm([-1.5, 0, 1.5], cmap.N)
    mesh = None
    any_no_upwelling = False
    for month, ax in enumerate(axes.ravel(), 1):
        record = records.get(month)
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", linewidth=0.8)
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
        if record is None:
            ax.set_title(f"{MONTH_NAMES[month - 1]}\nNo Data")
            continue
        mesh = ax.pcolormesh(data["lon_grid"], data["lat_grid"], record["classes"], cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading="auto")
        ax.contour(data["lon_grid"], data["lat_grid"], data["bathymetry"], levels=[data["bayth_threshold"]], colors="limegreen", linewidths=1.2, transform=ccrs.PlateCarree())
        if (record["classes"] == 1).any():
            ax.set_title(MONTH_NAMES[month - 1])
        else:
            # Asterisk: month has data but no upwelling zone was classified.
            ax.set_title(f"{MONTH_NAMES[month - 1]} *")
            any_no_upwelling = True
    fig.subplots_adjust(left=0.03, right=0.90, bottom=0.04, top=0.90, wspace=0.06, hspace=0.10)
    if any_no_upwelling:
        fig.text(0.03, 0.01, "* No upwelling zone classified this month", fontsize=12)
    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), ticks=[-1, 1], fraction=0.02, pad=0.02)
        cbar.ax.set_yticklabels(["Non-upwelling", "Upwelling"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{year}_class.png")
    plt.close(fig)


if __name__ == "__main__":
    plot_year(load_preclassified(PICKLE_PATH), TARGET_YEAR)
