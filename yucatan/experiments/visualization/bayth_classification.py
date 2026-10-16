"""Generate and load shared Bayth upwelling classifications."""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd


YUCATAN_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = Path(r"E:\satdata\Custom\yucatan_sst_chlor_cdomslope_ssl_bathy_2010-2019_v4.pkl")
MODEL_PATH = YUCATAN_DIR / "experiments" / "isolation_forest" / "models" / "iforest_pipeline5.joblib"
PICKLE_PATH = YUCATAN_DIR / "figs" / "bayth_preclassified.pkl"
YEARS = list(range(2010, 2020))
ANOM_VARS = ["sst", "ssl"]
PERCENTILE_UPWELLING = 90
SST_ANOM_Z_MAX = -0.5
SSL_ANOM_Z_MAX = 1.0
BAYTH_THRESHOLD = -10


def _is_missing_day(value):
    if value is None:
        return True
    if np.isscalar(value):
        return bool(np.isnan(value))
    return np.asarray(value).ndim == 0


def _load_source(path):
    bundle = pd.read_pickle(path)
    df = bundle["data"]
    metadata = bundle["metadata"]
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    lon = np.asarray(metadata["lon_centers"])
    lat = np.asarray(metadata["lat_centers"])
    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    return df, metadata, lon, lat


def _load_model(path):
    bundle = joblib.load(path)
    pipeline = bundle["iforest_pipeline"]
    features = bundle.get("features", ["sst_anom", "ssl_anom"])
    return pipeline.named_steps["preprocess"], pipeline.named_steps["iforest"], features


def _stack_month(df, year, variable, month):
    rows = df[(df.index.year == year) & (df.index.month == month)]
    grids = [np.asarray(value, dtype=float) for value in rows[variable] if not _is_missing_day(value)]
    if not grids:
        return None
    shapes = {grid.shape for grid in grids}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent grid shapes for {variable!r}, {year}-{month:02d}: {shapes}")
    return np.stack(grids)


def _monthly_mean_anomaly(df, year, variable, month):
    stack = _stack_month(df, year, variable, month)
    if stack is None:
        return None
    daily_median = np.nanmedian(stack, axis=(1, 2), keepdims=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(stack - daily_median, axis=0)


def _climatological_anomaly(df, years, variable, month):
    grids = [_monthly_mean_anomaly(df, year, variable, month) for year in years]
    grids = [grid for grid in grids if grid is not None]
    if not grids:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.stack(grids), axis=0)


def _zscore(grid, valid):
    result = np.full(grid.shape, np.nan)
    values = grid[valid]
    std = np.nanstd(values)
    if std == 0 or np.isnan(std):
        return result
    result[valid] = (values - np.nanmean(values)) / std
    return result


def _classify(anomalies, preprocess, iforest, features, bathymetry):
    valid = np.ones_like(anomalies["sst"], dtype=bool)
    for grid in anomalies.values():
        valid &= ~np.isnan(grid)
    if not valid.any():
        return None, None

    raw = pd.DataFrame({f"{name}_anom": anomalies[name][valid] for name in ANOM_VARS})
    scores = -iforest.decision_function(preprocess.transform(raw[features]))
    score_grid = np.full(anomalies["sst"].shape, np.nan)
    score_grid[valid] = scores
    threshold = np.percentile(scores, PERCENTILE_UPWELLING)
    sst_z = _zscore(anomalies["sst"], valid)
    ssl_z = _zscore(anomalies["ssl"], valid)
    deep = ~np.isnan(bathymetry) & (bathymetry <= BAYTH_THRESHOLD)
    upwelling = valid & (score_grid >= threshold) & (sst_z < SST_ANOM_Z_MAX) & (ssl_z < SSL_ANOM_Z_MAX) & deep
    class_grid = np.full(score_grid.shape, np.nan)
    class_grid[valid] = -1
    class_grid[upwelling] = 1
    return {"scores": score_grid, "classes": class_grid, "threshold": threshold}, (sst_z, ssl_z)


def _classify_scope(df, years, preprocess, iforest, features, bathymetry, climatology=False):
    result = {}
    for month in range(1, 13):
        if climatology:
            anomalies = {name: _climatological_anomaly(df, years, name, month) for name in ANOM_VARS}
        else:
            anomalies = {name: [_monthly_mean_anomaly(df, year, name, month) for year in years] for name in ANOM_VARS}
        if climatology:
            if any(value is None for value in anomalies.values()):
                result[month] = None
            else:
                classified, zscores = _classify(anomalies, preprocess, iforest, features, bathymetry)
                result[month] = None if classified is None else {**classified, "zscores": zscores}
            continue
        for index, year in enumerate(years):
            year_result = result.setdefault(year, {})
            year_anomalies = {name: anomalies[name][index] for name in ANOM_VARS}
            if any(value is None for value in year_anomalies.values()):
                year_result[month] = None
                continue
            classified, zscores = _classify(year_anomalies, preprocess, iforest, features, bathymetry)
            year_result[month] = None if classified is None else {**classified, "zscores": zscores}
    return result


def generate_preclassified_data(data_path=DATA_PATH, model_path=MODEL_PATH, output_path=PICKLE_PATH):
    df, metadata, lon_grid, lat_grid = _load_source(data_path)
    preprocess, iforest, features = _load_model(model_path)
    bathymetry = np.asarray(metadata["bathymetry"], dtype=float)
    data = {
        "version": 1,
        "years": YEARS,
        "months": list(range(1, 13)),
        "lon_grid": lon_grid,
        "lat_grid": lat_grid,
        "bathymetry": bathymetry,
        "bayth_threshold": BAYTH_THRESHOLD,
        "by_year": _classify_scope(df, YEARS, preprocess, iforest, features, bathymetry),
        "climatology": _classify_scope(df, YEARS, preprocess, iforest, features, bathymetry, climatology=True),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(data, output_path)
    print(f"Saved preclassified Bayth data to {output_path}")
    return data


def load_preclassified(path=PICKLE_PATH):
    return pd.read_pickle(path)


if __name__ == "__main__":
    generate_preclassified_data()
