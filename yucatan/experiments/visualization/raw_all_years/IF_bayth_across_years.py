"""
Monthly-CLIMATOLOGY anomaly-score and upwelling-classification plots.

This is the multi-year companion to monthly_anomaly_by_year.py. Instead of one
figure per year, it produces a single figure where each panel is a calendar
month (Jan-Dec) averaged across every year in YEARS (2010-2019).

Pipeline (per calendar month, across all years in YEARS):
  1. For each year, stack daily sst / ssl grids for that calendar month.
  2. Compute same-day (spatial) anomalies: value - daily median.
  3. Average the daily anomalies over the month -> that year's mean_sst_anom /
     mean_ssl_anom grid for that month (identical to the per-year script).
  4. Average those per-year monthly-mean-anomaly grids ACROSS ALL YEARS
     (nanmean, pixel-wise) -> a single climatological mean_sst_anom /
     mean_ssl_anom grid for that calendar month.
  5. Run [sst_anom, ssl_anom] through the trained isolation-forest pipeline
     (preprocess -> iforest) ONCE per calendar month, on the climatological
     grids, to get a per-pixel anomaly score (score = -decision_function,
     higher = more anomalous).
  6. Classify each pixel as a binary upwelling / non-upwelling call using a
     single percentile threshold on that month's climatological score
     distribution, PLUS a physical clip:
       score >= percentile threshold
         AND sst_anom z-score < -0.5   (cold water)
         AND ssl_anom z-score < 1      (not elevated sea level)
         AND bathymetry <= BAYTH_THRESHOLD (at least 10 m deep)
                                    -> upwelling
       otherwise                  -> non-upwelling

     The z-scores are computed per month from that month's own valid-pixel
     mean/std of the CLIMATOLOGICAL sst_anom and ssl_anom grids (same
     spatial-stats approach as the percentile threshold). The bathymetry
     check uses a static "deep enough" mask built once from the bathymetry
     grid (same units/sign convention as metadata["bathymetry"], i.e.
     negative = below sea level), so any upwelling candidate sitting in
     water shallower than 10 m gets dropped back to non-upwelling.

Outputs (saved to ./bayth_figs/):
  bayth_figs/climatology_IF_anom.png   monthly climatological anomaly-score maps
  bayth_figs/climatology_IF_class.png  monthly climatological classification maps
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
OUT_DIR = Path("./bayth_figs")

YEARS = list(range(2010, 2020))  # 2010-2019 inclusive, averaged together
MONTH_NAMES = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

# Features the anomaly model was trained on (must match bundle["features"] order).
ANOM_VARS = ['sst', 'ssl']

# Percentile threshold for classification (computed per month, over that
# month's valid-pixel climatological anomaly-score distribution).
PERCENTILE_UPWELLING = 90      # score >= this percentile -> upwelling (candidate)

# Physical clip applied on top of the score threshold before a pixel can be
# labeled upwelling. Z-scores are computed per month from that month's own
# valid-pixel climatological sst_anom / ssl_anom distributions.
SST_ANOM_Z_MAX = -0.5   # sst_anom z-score must be < this (cold anomaly)
SSL_ANOM_Z_MAX = 1.0    # ssl_anom z-score must be < this

# Bathymetry depth cutoff (same units/sign convention as metadata["bathymetry"],
# i.e. negative = below sea level). Pixels shallower than this (bathymetry >
# BAYTH_THRESHOLD) are excluded from the upwelling class regardless of score.
# -10 => at least 10 m deep required to qualify as upwelling.
BAYTH_THRESHOLD = -10

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

    bathymetry = np.array(metadata['bathymetry'])

    return df, lon_grid, lat_grid, bathymetry


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
    """Mean, over the month's days (for one year), of the same-day anomaly grid."""
    stack = stack_month_grids(df, year, var, month)
    if stack is None:
        return None
    anom_stack = same_day_anomaly(stack)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.nanmean(anom_stack, axis=0)


def climatological_monthly_mean_anomaly(df, years, var, month):
    """Average of each year's monthly-mean-anomaly grid, across all `years`.

    Returns None only if NO year has data for this (var, month) combination.
    """
    per_year_grids = []
    for year in years:
        g = monthly_mean_anomaly(df, year, var, month)
        if g is not None:
            per_year_grids.append(g)

    if not per_year_grids:
        return None

    shapes = {g.shape for g in per_year_grids}
    if len(shapes) > 1:
        raise ValueError(
            f"Inconsistent grid shapes across years for var={var!r}, month={month:02d}: {shapes}"
        )

    stacked = np.stack(per_year_grids, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.nanmean(stacked, axis=0)


def zscore_grid(grid, valid):
    """Spatial z-score of `grid` computed from its own valid-pixel mean/std."""
    z = np.full(grid.shape, np.nan)
    vals = grid[valid]
    mean = np.nanmean(vals)
    std = np.nanstd(vals)
    if std == 0 or np.isnan(std):
        return z  # all-nan / degenerate month, leave as nan (fails clip safely)
    z[valid] = (grid[valid] - mean) / std
    return z


def compute_climatological_monthly_scores(df, years, preprocess, iforest, features):
    """Same shape of output as the per-year script's compute_monthly_scores,
    but each month's anomaly grids are averaged across `years` BEFORE being
    run through the isolation-forest pipeline once.

    monthly_scores[m]      -> 2D grid of anomaly scores (nan where invalid), or None
    monthly_thresholds[m]  -> threshold_upwelling (scalar), or None
    monthly_zscores[m]     -> (sst_anom_z_grid, ssl_anom_z_grid), or None
    """
    monthly_scores = {}
    monthly_thresholds = {}
    monthly_zscores = {}

    for m in range(1, 13):
        anom_grids = {v: climatological_monthly_mean_anomaly(df, years, v, m) for v in ANOM_VARS}

        if any(g is None for g in anom_grids.values()):
            monthly_scores[m] = None
            monthly_thresholds[m] = None
            monthly_zscores[m] = None
            continue

        valid = np.ones_like(anom_grids[ANOM_VARS[0]], dtype=bool)
        for g in anom_grids.values():
            valid &= ~np.isnan(g)

        if not valid.any():
            monthly_scores[m] = None
            monthly_thresholds[m] = None
            monthly_zscores[m] = None
            continue

        X_raw = pd.DataFrame({f"{v}_anom": anom_grids[v][valid] for v in ANOM_VARS})
        X = X_raw[features]
        X_t = preprocess.transform(X)
        scores_valid = -iforest.decision_function(X_t)

        score_grid = np.full(anom_grids[ANOM_VARS[0]].shape, np.nan)
        score_grid[valid] = scores_valid
        monthly_scores[m] = score_grid

        threshold_u = np.percentile(scores_valid, PERCENTILE_UPWELLING)
        monthly_thresholds[m] = threshold_u

        sst_z = zscore_grid(anom_grids['sst'], valid)
        ssl_z = zscore_grid(anom_grids['ssl'], valid)
        monthly_zscores[m] = (sst_z, ssl_z)

        print(f"{MONTH_NAMES[m - 1]} (climatology {years[0]}-{years[-1]}): "
              f"score range [{np.nanmin(scores_valid):.3f}, {np.nanmax(scores_valid):.3f}], "
              f"upwelling>={threshold_u:.3f}")

    return monthly_scores, monthly_thresholds, monthly_zscores


def build_deep_enough_mask(bathymetry, bayth_threshold):
    """Binary depth mask: 1 where bathymetry <= threshold (deep water),
    0 where bathymetry > threshold (shallow water, incl. land/nan)."""
    deep_enough = np.zeros(bathymetry.shape, dtype=float)
    valid_bathy = ~np.isnan(bathymetry)
    deep_enough[valid_bathy & (bathymetry <= bayth_threshold)] = 1.0
    return deep_enough


def classify_grid(score_grid, threshold_u, sst_z_grid, ssl_z_grid, deep_enough_mask):
    """Binary classification: -1 = non-upwelling, 1 = upwelling, nan = invalid pixel.

    Upwelling requires high anomaly score AND a physical clip:
    sst_anom z-score < SST_ANOM_Z_MAX and ssl_anom z-score < SSL_ANOM_Z_MAX,
    AND water at least as deep as BAYTH_THRESHOLD (deep_enough_mask == 1).
    Every other valid pixel is non-upwelling (no buffer class).
    """
    class_grid = np.full(score_grid.shape, np.nan)
    valid = ~np.isnan(score_grid)

    clip_ok = (sst_z_grid < SST_ANOM_Z_MAX) & (ssl_z_grid < SSL_ANOM_Z_MAX)
    upwelling_candidate = (valid & (score_grid >= threshold_u) & clip_ok).astype(float)

    # zero out candidate upwelling pixels sitting in water that's too shallow
    upwelling_final = upwelling_candidate * deep_enough_mask

    class_grid[valid] = -1
    class_grid[upwelling_final == 1] = 1
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
                                 filename='climatology_IF_anom.png', suptitle=None, cmap='inferno'):
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

    if suptitle:
        fig.suptitle(suptitle, fontsize=16)
    fig.subplots_adjust(left=0.03, right=0.90, bottom=0.04, top=0.88, wspace=0.06, hspace=0.15)
    if mesh_for_cbar is not None:
        cax = fig.add_axes([0.915, 0.04, 0.012, 0.84])
        cbar = fig.colorbar(mesh_for_cbar, cax=cax)
        cbar.set_label('Anomaly score', fontsize=12)

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / filename)
    plt.close(fig)


def plot_monthly_classification(monthly_scores, monthly_thresholds, monthly_zscores,
                                 lon_grid, lat_grid, deep_enough_mask, out_dir,
                                 filename='climatology_IF_class.png', suptitle=None):
    cmap = mcolors.ListedColormap(['#1f5fa8', '#c0392b'])  # blue, red
    norm = mcolors.BoundaryNorm([-1.5, 0, 1.5], cmap.N)

    print("Plotting classification (upwelling / non-upwelling)")

    fig, axes = _new_month_axes()
    mesh_for_cbar = None
    for month_num, ax in enumerate(axes.ravel(), start=1):
        score_grid = monthly_scores.get(month_num)
        thresholds = monthly_thresholds.get(month_num)
        zscores = monthly_zscores.get(month_num)
        if score_grid is None or thresholds is None or zscores is None:
            ax.set_title(f"{MONTH_NAMES[month_num - 1]}\nNo Data", fontsize=10)
            continue

        threshold_u = thresholds
        sst_z_grid, ssl_z_grid = zscores
        class_grid = classify_grid(score_grid, threshold_u, sst_z_grid, ssl_z_grid,
                                    deep_enough_mask)

        mesh = ax.pcolormesh(
            lon_grid, lat_grid, class_grid,
            cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
            shading='auto', zorder=1
        )
        mesh_for_cbar = mesh_for_cbar or mesh
        ax.set_title(MONTH_NAMES[month_num - 1], fontsize=10)

    if suptitle:
        fig.suptitle(suptitle, fontsize=16)
    fig.subplots_adjust(left=0.03, right=0.90, bottom=0.04, top=0.88, wspace=0.06, hspace=0.15)
    if mesh_for_cbar is not None:
        cax = fig.add_axes([0.915, 0.04, 0.012, 0.84])
        cbar = fig.colorbar(mesh_for_cbar, cax=cax, ticks=[-1, 1])
        cbar.ax.set_yticklabels(['Non-upwelling', 'Upwelling'])

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / filename)
    plt.close(fig)


def main():
    df, lon_grid, lat_grid, bathymetry = load_data(DATA_PATH)
    preprocess, iforest, features = load_model(MODEL_PATH)

    deep_enough_mask = build_deep_enough_mask(bathymetry, BAYTH_THRESHOLD)

    print(f"Processing climatology across {YEARS[0]}-{YEARS[-1]}...")
    monthly_scores, monthly_thresholds, monthly_zscores = compute_climatological_monthly_scores(
        df, YEARS, preprocess, iforest, features
    )

    suptitle = f"Climatology {YEARS[0]}-{YEARS[-1]}"
    plot_monthly_anomaly_scores(monthly_scores, lon_grid, lat_grid, OUT_DIR, suptitle=suptitle)
    plot_monthly_classification(monthly_scores, monthly_thresholds, monthly_zscores,
                                 lon_grid, lat_grid, deep_enough_mask, OUT_DIR, suptitle=suptitle)


if __name__ == "__main__":
    main()