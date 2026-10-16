import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path

# ----------------------------------------------------------------------------
# Absolute paths - update these if your environment differs
# ----------------------------------------------------------------------------
DATA_PATH = r"E:\satdata\Custom\yucatan_sst_chlor_cdomslope_ssl_bathy_2010-2019_v4.pkl"
MODEL_PATH = r"F:\Programming\GitHub\nasa-murep\yucatan\experiments\isolation_forest\models\iforest_pipeline5.joblib"
FIGS_ROOT = Path(__file__).resolve().parents[3] / "figs" / "scatterplots"

# Upwelling zone bounding box (Yucatan shelf)
UPWELLING_LON_MIN, UPWELLING_LON_MAX = -92.20216, -85.61256
UPWELLING_LAT_MIN, UPWELLING_LAT_MAX = 19.57871, 22.86906

# Percentile of the daily anomaly-score distribution used as the isolation-forest
# "anomalous" cutoff (matches the value used in the window-based experiment script).
ANOMALY_PERCENTILE = 95

# Bathymetry cutoff (same sign convention as metadata["bathymetry"]: negative = below
# sea level). Pixels shallower than this (bathy > BATHY_THRESHOLD), plus land / NaN,
# can never be labeled upwelling. -20 => at least 20 m deep required.
BATHY_THRESHOLD = -20


def _same_day_anomaly(arr2d):
    """Subtract the day's own spatial median (matches training-time definition)."""
    return arr2d - np.nanmedian(arr2d)


def load_dataset():
    with open(DATA_PATH, "rb") as f:
        blob = pickle.load(f)
    df = blob["data"]
    metadata = blob["metadata"]
    df = df[(df.index.year >= 2010) & (df.index.year <= 2019)]

    grid_shape = np.asarray(df["sst"].iloc[0]).shape

    def shape_ok(a):
        return a is not None and np.asarray(a).shape == grid_shape

    keep = (
        df["sst"].apply(shape_ok)
        & df["chlorophyll"].apply(shape_ok)
        & df["ssl"].apply(shape_ok)
        & df["cdom_412"].apply(shape_ok)
    )
    df = df[keep]
    return df, metadata, grid_shape


def load_model():
    bundle = joblib.load(MODEL_PATH)
    iforest_pipeline = bundle["iforest_pipeline"]
    features = bundle.get("features", ["sst_anom", "ssl_anom"])
    preprocess = iforest_pipeline.named_steps["preprocess"]
    iforest = iforest_pipeline.named_steps["iforest"]
    return preprocess, iforest, features


def build_bbox_mask(metadata):
    lon_centers = np.array(metadata["lon_centers"])
    lat_centers = np.array(metadata["lat_centers"])
    lon_grid_2d, lat_grid_2d = np.meshgrid(lon_centers, lat_centers)
    in_bbox = (
        (lon_grid_2d >= UPWELLING_LON_MIN) & (lon_grid_2d <= UPWELLING_LON_MAX)
        & (lat_grid_2d >= UPWELLING_LAT_MIN) & (lat_grid_2d <= UPWELLING_LAT_MAX)
    )
    return in_bbox, lon_grid_2d, lat_grid_2d


def build_deep_mask(metadata, grid_shape, threshold=BATHY_THRESHOLD):
    """
    Static boolean mask from the (fixed) bathymetry grid:
    True where bathymetry <= threshold (deep enough), False for shallow / land / NaN.
    Built once and reused for every day.
    """
    bathy = np.asarray(metadata["bathymetry"], dtype=float)
    if bathy.shape != grid_shape:
        raise ValueError(f"Bathymetry shape {bathy.shape} != data grid shape {grid_shape}")
    return ~np.isnan(bathy) & (bathy <= threshold)


def classify_day_upwelling(sst_2d, ssl_2d, preprocess, iforest, features, deep_mask,
                            percentile=ANOMALY_PERCENTILE):
    """
    Run the isolation-forest pipeline on a single day's SST/SSL fields and return a
    boolean upwelling mask over the full grid (False for NaN / non-flagged pixels).

    A pixel is upwelling only if it is in the top `percentile` of that day's anomaly
    scores, passes the cold/low-sea-level clip, AND lies in water at least as deep as
    BATHY_THRESHOLD (deep_mask == True).
    """
    grid_shape = sst_2d.shape
    sst_anom = _same_day_anomaly(sst_2d)
    ssl_anom = _same_day_anomaly(ssl_2d)

    valid = ~np.isnan(sst_anom) & ~np.isnan(ssl_anom)
    upwelling_mask = np.zeros(grid_shape, dtype=bool)
    if not np.any(valid):
        return upwelling_mask

    X_day = pd.DataFrame({"sst_anom": sst_anom[valid], "ssl_anom": ssl_anom[valid]})[features]
    X_transformed = preprocess.transform(X_day)

    anomaly_score = -iforest.decision_function(X_transformed)
    if len(anomaly_score) == 0:
        return upwelling_mask

    raw_anomaly = anomaly_score >= np.percentile(anomaly_score, percentile)
    sst_anom_z = X_transformed["anom__sst_anom"].to_numpy()
    ssl_anom_z = X_transformed["anom__ssl_anom"].to_numpy()

    is_upwelling = raw_anomaly & (sst_anom_z < -0.5) & (ssl_anom_z < 1)
    is_upwelling &= deep_mask[valid]   # shallow candidates -> non-upwelling

    rows, cols = np.where(valid)
    upwelling_mask[rows[is_upwelling], cols[is_upwelling]] = True
    return upwelling_mask


def build_all_data(use_iforest=True, max_days=None):
    """
    Loop over every day in the dataset and assemble a long-format DataFrame of
    chlorophyll / CDOM / month / year / Season (and, if use_iforest=True, the
    isolation-forest-derived Zone label) restricted to the upwelling bounding box.

    Set use_iforest=False to skip the (slower) model step for plots that don't
    need the Zone label (e.g. the calendar-season-only comparison).
    """
    df, metadata, grid_shape = load_dataset()
    in_bbox, _, _ = build_bbox_mask(metadata)
    bbox_flat = in_bbox.flatten()

    preprocess = iforest = features = deep_mask = None
    if use_iforest:
        preprocess, iforest, features = load_model()
        deep_mask = build_deep_mask(metadata, grid_shape)

    dates = df.index if max_days is None else df.index[:max_days]
    rows = []

    for date_idx in dates:
        row = df.loc[date_idx]
        chlor = row.get("chlorophyll")
        cdom = row.get("cdom_412")
        sst = row.get("sst")
        ssl = row.get("ssl")
        if chlor is None or cdom is None or sst is None:
            continue
        if use_iforest and ssl is None:
            continue

        chlor_arr = np.asarray(chlor, dtype=float).flatten()
        cdom_arr = np.asarray(cdom, dtype=float).flatten()
        valid = ~(np.isnan(chlor_arr) | np.isnan(cdom_arr)) & bbox_flat
        if not np.any(valid):
            continue

        record = {
            "chlor": chlor_arr[valid],
            "cdom412": cdom_arr[valid],
            "month": date_idx.month,
            "year": date_idx.year,
        }

        if use_iforest:
            sst_2d = np.asarray(sst, dtype=float)
            ssl_2d = np.asarray(ssl, dtype=float)
            upwelling_mask = classify_day_upwelling(sst_2d, ssl_2d, preprocess, iforest, features, deep_mask)
            record["upwelling"] = upwelling_mask.flatten()[valid]

        rows.append(pd.DataFrame(record))

    all_data = pd.concat(rows, ignore_index=True)
    if use_iforest:
        all_data["Zone"] = all_data["upwelling"].map({True: "Inside Upwelling", False: "Outside Upwelling"})
    all_data["Season"] = all_data["month"].apply(
        lambda m: "Upwelling Months (May-Sep)" if 5 <= m <= 9 else "Non-upwelling Months (Oct-Apr)"
    )
    return all_data