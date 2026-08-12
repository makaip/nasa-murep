import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

plt.style.use('default')

DATA_PATH = Path(r"E:\satdata\Custom\yucatan_sst_chlor_cdomslope_ssl_bathy_2010-2019_v3.pkl")
OUT_DIR = Path("./yucatan/experiments/visualization/raw/figs")

TARGET_YEAR = 2018
MONTH_NAMES = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

RAW_VARS = ['sst', 'chlorophyll', 'ssl', 'cdom_412',
            'cdom_slope_275_295', 'cdom_slope_300_600']

MIN_VALID_DAYS = 0

LAT_MIN, LON_MIN = 19.57871, -92.20216
LAT_MAX, LON_MAX = 22.86906, -85.61256

PLOT_CONFIGS = [
    # (variable, colorbar label, colormap, vmin, vmax)
    ('log_chlor', 'log10(mg/m³)', 'viridis', None, None),
    ('sst', '°C', 'RdYlBu_r', None, None),
    ('ssl', 'm', 'magma', None, None),
    ('cdom_412', 'CDOM (412 nm)', 'cividis', None, None),
    ('cdom_slope_275_295', 'slope (275-295 nm)', 'plasma', None, None),
    ('cdom_slope_300_600', 'slope (300-600 nm)', 'plasma', None, None),
]


def load_data(path):
    data = pd.read_pickle(path)
    df = data['data']
    metadata = data['metadata']
    print("Data loaded successfully.")

    lon_centers = np.array(metadata['lon_centers'])
    lat_centers = np.array(metadata['lat_centers'])
    if lon_centers.ndim == 1 and lat_centers.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
    else:
        lon_grid, lat_grid = lon_centers, lat_centers

    return df, lon_grid, lat_grid


def _is_missing_day(v):
    if v is None:
        return True
    if np.isscalar(v):
        return bool(np.isnan(v))
    arr = np.asarray(v)
    return arr.ndim == 0  # 0-d array (e.g. np.array(nan)) is also "missing", not a grid


def stack_month_grids(df, year, var, month):
    mask = (df.index.year == year) & (df.index.month == month)
    grids = [
        np.asarray(v, dtype=float)
        for v in df.loc[mask, var]
        if not _is_missing_day(v)
    ]
    if not grids:
        return None

    shapes = {g.shape for g in grids}
    if len(shapes) > 1:
        raise ValueError(
            f"Inconsistent grid shapes for var={var!r}, {year}-{month:02d}: {shapes}"
        )
    return np.stack(grids, axis=0)


def monthly_mean(df, year, var, month, min_valid_days=0):
    stacked = stack_month_grids(df, year, var, month)
    if stacked is None:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        mean_grid = np.nanmean(stacked, axis=0)

        if min_valid_days > 1:
            valid_counts = np.sum(~np.isnan(stacked), axis=0)
            mean_grid = np.where(valid_counts >= min_valid_days, mean_grid, np.nan)

    return mean_grid


def build_monthly_means(df, year, raw_vars, min_valid_days=0):
    monthly = {
        m: {v: monthly_mean(df, year, v, m, min_valid_days) for v in raw_vars}
        for m in range(1, 13)
    }
    for m in monthly:
        chlor = monthly[m].get('chlorophyll')
        monthly[m]['log_chlor'] = np.log10(chlor) if chlor is not None else None
    return monthly


def plot_monthly_variable(monthly_means, lon_grid, lat_grid, var_key, unit_label,
                           cmap, vmin, vmax, out_dir):
    grids = [monthly_means[m].get(var_key) for m in range(1, 13)
              if monthly_means[m].get(var_key) is not None]

    if vmin is None or vmax is None:
        all_vals = np.concatenate([g.flatten() for g in grids])
        vmin = vmin if vmin is not None else np.nanpercentile(all_vals, 1)
        vmax = vmax if vmax is not None else np.nanpercentile(all_vals, 99)

    print(f"Plotting {var_key}: vmin={vmin:.2f}, vmax={vmax:.2f}")

    fig, axes = plt.subplots(3, 4, figsize=(28, 10),
                              subplot_kw={'projection': ccrs.PlateCarree()}, dpi=400)

    mesh_for_cbar = None
    for month_num, ax in enumerate(axes.ravel(), start=1):
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.set_aspect('auto')
        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.8, zorder=3)
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=4)
        ax.gridlines(draw_labels=False, linewidth=0.45, color='white', alpha=0.55, linestyle='--', zorder=2)

        data_grid = monthly_means[month_num].get(var_key)
        if data_grid is None:
            ax.set_title(f"{MONTH_NAMES[month_num - 1]}\nNo Data", fontsize=10)
            continue

        mesh = ax.pcolormesh(
            lon_grid, lat_grid, data_grid,
            cmap=cmap, transform=ccrs.PlateCarree(),
            vmin=vmin, vmax=vmax, shading='auto', zorder=1
        )
        mesh_for_cbar = mesh_for_cbar or mesh
        ax.set_title(MONTH_NAMES[month_num - 1], fontsize=10)

    fig.subplots_adjust(left=0.03, right=0.90, bottom=0.04, top=0.9, wspace=0.06, hspace=0.1)
    if mesh_for_cbar is not None:
        cax = fig.add_axes([0.915, 0.04, 0.012, 0.88])
        cbar = fig.colorbar(mesh_for_cbar, cax=cax)
        cbar.set_label(unit_label, fontsize=12)

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / var_key)
    plt.close(fig)


def main():
    df, lon_grid, lat_grid = load_data(DATA_PATH)

    print(f"Processing data for {TARGET_YEAR}...")
    monthly_means = build_monthly_means(df, TARGET_YEAR, RAW_VARS, MIN_VALID_DAYS)

    for var_key, unit_label, cmap, vmin, vmax in PLOT_CONFIGS:
        plot_monthly_variable(monthly_means, lon_grid, lat_grid,
                               var_key, unit_label, cmap, vmin, vmax, OUT_DIR)


if __name__ == "__main__":
    main()