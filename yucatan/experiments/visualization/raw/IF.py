"""
Monthly anomaly-score and upwelling-classification plots.

Pipeline (per calendar month of TARGET_YEAR):
  1. Stack daily sst / ssl grids for the month.
  2. Compute same-day (spatial) anomalies: value - daily median.
  3. Average the daily anomalies over the month -> mean_sst_anom, mean_ssl_anom grids.
  4. Run [sst_anom, ssl_anom] through the trained isolation-forest pipeline
     (preprocess -> iforest) to get a per-pixel anomaly score
     (score = -decision_function, higher = more anomalous).
  5. Classify each pixel using two percentile thresholds computed from that
     month's score distribution:
       score >= upper percentile  -> upwelling      (red)
       score <  lower percentile  -> non-upwelling   (blue)
       otherwise                  -> buffer          (white)

Outputs:
  figs/anom.png   monthly anomaly-score maps (continuous colormap)
  figs/class.png  monthly upwelling/non-upwelling/buffer classification maps
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

plt.style.use('default')


DATA_PATH = Path(r"E:\satdata\Custom\yucatan_sst_chlor_cdomslope_ssl_bathy_2010-2019_v4.pkl")
MODEL_PATH = Path(r"F:\Programming\GitHub\nasa-murep\yucatan\experiments\isolation_forest\models\iforest_pipeline4.joblib")
OUT_DIR = Path("./yucatan/experiments/visualization/raw/figs")

TARGET_YEAR = 2018
MONTH_NAMES = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

# Features the anomaly model was trained on (must match bundle["features"] order).
ANOM_VARS = ['sst', 'ssl']

# Percentile thresholds for classification (computed per month, over that
# month's valid-pixel anomaly-score distribution).
PERCENTILE_UPWELLING = 90      # score >= this percentile -> upwelling
PERCENTILE_NON_UPWELLING = 80  # score <  this percentile -> non-upwelling

LAT_MIN, LON_MIN = 19.57871, -92.20216
LAT_MAX, LON_MAX = 22.86906, -85.61256


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


def load_model(path):
    bundle = joblib.load(path)
    iforest_pipeline = bundle["iforest_pipeline"]
    features = bundle.get("features", ["sst_anom", "ssl_anom"])
    preprocess = iforest_pipeline.named_steps["preprocess"]
    iforest = iforest_pipeline.named_steps["iforest"]
    print(f"Loaded pipeline from {path}")
    print(f"Features: {features}")
    return preprocess, iforest, features


def _is_missing_day(v):
    if v is None:
        return True
    if np.isscalar(v):
        return bool(np.isnan(v))
    arr = np.asarray(v)
    return arr.ndim == 0


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


def same_day_anomaly(stack):
    """Subtract each day's spatial median from that day's grid."""
    daily_median = np.nanmedian(stack, axis=(1, 2), keepdims=True)
    return stack - daily_median


def monthly_mean_anomaly(df, year, var, month):
    """Mean, over the month's days, of the same-day anomaly grid."""
    stack = stack_month_grids(df, year, var, month)
    if stack is None:
        return None
    anom_stack = same_day_anomaly(stack)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.nanmean(anom_stack, axis=0)


def compute_monthly_scores(df, year, preprocess, iforest, features):
    """Returns (monthly_scores, monthly_thresholds), both dicts keyed by month.

    monthly_scores[m]      -> 2D grid of anomaly scores (nan where invalid), or None
    monthly_thresholds[m]  -> (threshold_upwelling, threshold_non_upwelling), or None
    """
    monthly_scores = {}
    monthly_thresholds = {}

    for m in range(1, 13):
        anom_grids = {v: monthly_mean_anomaly(df, year, v, m) for v in ANOM_VARS}

        if any(g is None for g in anom_grids.values()):
            monthly_scores[m] = None
            monthly_thresholds[m] = None
            continue

        valid = np.ones_like(anom_grids[ANOM_VARS[0]], dtype=bool)
        for g in anom_grids.values():
            valid &= ~np.isnan(g)

        if not valid.any():
            monthly_scores[m] = None
            monthly_thresholds[m] = None
            continue

        X_raw = pd.DataFrame({f"{v}_anom": anom_grids[v][valid] for v in ANOM_VARS})
        X = X_raw[features]
        X_t = preprocess.transform(X)
        scores_valid = -iforest.decision_function(X_t)

        score_grid = np.full(anom_grids[ANOM_VARS[0]].shape, np.nan)
        score_grid[valid] = scores_valid
        monthly_scores[m] = score_grid

        threshold_u = np.percentile(scores_valid, PERCENTILE_UPWELLING)
        threshold_n = np.percentile(scores_valid, PERCENTILE_NON_UPWELLING)
        monthly_thresholds[m] = (threshold_u, threshold_n)

        print(f"{MONTH_NAMES[m - 1]}: score range [{np.nanmin(scores_valid):.3f}, "
              f"{np.nanmax(scores_valid):.3f}], upwelling>={threshold_u:.3f}, "
              f"non-upwelling<{threshold_n:.3f}")

    return monthly_scores, monthly_thresholds


def classify_grid(score_grid, threshold_u, threshold_n):
    """-1 = non-upwelling, 0 = buffer, 1 = upwelling, nan = invalid pixel."""
    class_grid = np.full(score_grid.shape, np.nan)
    valid = ~np.isnan(score_grid)
    class_grid[valid & (score_grid >= threshold_u)] = 1
    class_grid[valid & (score_grid < threshold_n)] = -1
    class_grid[valid & (score_grid >= threshold_n) & (score_grid < threshold_u)] = 0
    return class_grid


def _new_month_axes():
    fig, axes = plt.subplots(3, 4, figsize=(28, 10),
                              subplot_kw={'projection': ccrs.PlateCarree()}, dpi=400)
    for ax in axes.ravel():
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.set_aspect('auto')
        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.8, zorder=3)
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=4)
        ax.gridlines(draw_labels=False, linewidth=0.45, color='white', alpha=0.55, linestyle='--', zorder=2)
    return fig, axes


def plot_monthly_anomaly_scores(monthly_scores, lon_grid, lat_grid, out_dir,
                                 filename='anom.png', cmap='inferno'):
    grids = [g for g in monthly_scores.values() if g is not None]
    if not grids:
        print("No anomaly score data to plot.")
        return

    all_vals = np.concatenate([g[~np.isnan(g)] for g in grids])
    vmin = np.percentile(all_vals, 1)
    vmax = np.percentile(all_vals, 99)
    print(f"Plotting anomaly_score: vmin={vmin:.3f}, vmax={vmax:.3f}")

    fig, axes = _new_month_axes()
    mesh_for_cbar = None
    for month_num, ax in enumerate(axes.ravel(), start=1):
        score_grid = monthly_scores.get(month_num)
        if score_grid is None:
            ax.set_title(f"{MONTH_NAMES[month_num - 1]}\nNo Data", fontsize=10)
            continue

        mesh = ax.pcolormesh(
            lon_grid, lat_grid, score_grid,
            cmap=cmap, transform=ccrs.PlateCarree(),
            vmin=vmin, vmax=vmax, shading='auto', zorder=1
        )
        mesh_for_cbar = mesh_for_cbar or mesh
        ax.set_title(MONTH_NAMES[month_num - 1], fontsize=10)

    fig.subplots_adjust(left=0.03, right=0.90, bottom=0.04, top=0.9, wspace=0.06, hspace=0.1)
    if mesh_for_cbar is not None:
        cax = fig.add_axes([0.915, 0.04, 0.012, 0.88])
        cbar = fig.colorbar(mesh_for_cbar, cax=cax)
        cbar.set_label('Anomaly score', fontsize=12)

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / filename)
    plt.close(fig)


def plot_monthly_classification(monthly_scores, monthly_thresholds, lon_grid, lat_grid,
                                 out_dir, filename='class.png'):
    cmap = mcolors.ListedColormap(['#1f5fa8', '#ffffff', '#c0392b'])  # blue, white, red
    norm = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    print("Plotting classification (upwelling / buffer / non-upwelling)")

    fig, axes = _new_month_axes()
    mesh_for_cbar = None
    for month_num, ax in enumerate(axes.ravel(), start=1):
        score_grid = monthly_scores.get(month_num)
        thresholds = monthly_thresholds.get(month_num)
        if score_grid is None or thresholds is None:
            ax.set_title(f"{MONTH_NAMES[month_num - 1]}\nNo Data", fontsize=10)
            continue

        threshold_u, threshold_n = thresholds
        class_grid = classify_grid(score_grid, threshold_u, threshold_n)

        mesh = ax.pcolormesh(
            lon_grid, lat_grid, class_grid,
            cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
            shading='auto', zorder=1
        )
        mesh_for_cbar = mesh_for_cbar or mesh
        ax.set_title(MONTH_NAMES[month_num - 1], fontsize=10)

    fig.subplots_adjust(left=0.03, right=0.90, bottom=0.04, top=0.9, wspace=0.06, hspace=0.1)
    if mesh_for_cbar is not None:
        cax = fig.add_axes([0.915, 0.04, 0.012, 0.88])
        cbar = fig.colorbar(mesh_for_cbar, cax=cax, ticks=[-1, 0, 1])
        cbar.ax.set_yticklabels(['Non-upwelling', 'Buffer', 'Upwelling'])

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / filename)
    plt.close(fig)


def main():
    df, lon_grid, lat_grid = load_data(DATA_PATH)
    preprocess, iforest, features = load_model(MODEL_PATH)

    print(f"Processing data for {TARGET_YEAR}...")
    monthly_scores, monthly_thresholds = compute_monthly_scores(
        df, TARGET_YEAR, preprocess, iforest, features
    )

    plot_monthly_anomaly_scores(monthly_scores, lon_grid, lat_grid, OUT_DIR)
    plot_monthly_classification(monthly_scores, monthly_thresholds, lon_grid, lat_grid, OUT_DIR)


if __name__ == "__main__":
    main()