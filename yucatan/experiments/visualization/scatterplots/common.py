import numpy as np
import pandas as pd
import pickle
import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# Absolute paths - update these if your environment differs
# ----------------------------------------------------------------------------
DATA_PATH = r"E:\satdata\Custom\yucatan_sst_chlor_cdomslope_ssl_bathy_2010-2019_v4.pkl"
FIGS_ROOT = Path(__file__).resolve().parents[3] / "figs" / "scatterplots"

# Upwelling zone bounding box (Yucatan shelf)
UPWELLING_LON_MIN, UPWELLING_LON_MAX = -92.20216, -85.61256
UPWELLING_LAT_MIN, UPWELLING_LAT_MAX = 19.57871, 22.86906

# Zone labels come from the shared monthly Bayth classification pickle
# (figs/bayth_preclassified.pkl, built by ../bayth_classification.py with
# iforest_pipeline5, 90th percentile, SST z < -0.5, SSL z < 1.0, depth >= 10 m),
# the same source as the violins and maps. A daily pixel is "inside" if its
# month-year map has class 1 at that pixel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bayth_classification import PICKLE_PATH, load_preclassified  # noqa: E402


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


def build_bbox_mask(metadata):
    lon_centers = np.array(metadata["lon_centers"])
    lat_centers = np.array(metadata["lat_centers"])
    lon_grid_2d, lat_grid_2d = np.meshgrid(lon_centers, lat_centers)
    in_bbox = (
        (lon_grid_2d >= UPWELLING_LON_MIN) & (lon_grid_2d <= UPWELLING_LON_MAX)
        & (lat_grid_2d >= UPWELLING_LAT_MIN) & (lat_grid_2d <= UPWELLING_LAT_MAX)
    )
    return in_bbox, lon_grid_2d, lat_grid_2d


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

    preclassified = load_preclassified(PICKLE_PATH) if use_iforest else None

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
        # Log-log plots: keep strictly positive values only.
        valid = ~(np.isnan(chlor_arr) | np.isnan(cdom_arr)) & (chlor_arr > 0) & (cdom_arr > 0) & bbox_flat
        if not np.any(valid):
            continue

        record = {
            "chlor": chlor_arr[valid],
            "cdom412": cdom_arr[valid],
            "month": date_idx.month,
            "year": date_idx.year,
        }

        if use_iforest:
            month_class = preclassified["by_year"].get(int(date_idx.year), {}).get(int(date_idx.month))
            if month_class is None:
                upwelling_flat = np.zeros(grid_shape, dtype=bool).flatten()
            else:
                upwelling_flat = (month_class["classes"] == 1).flatten()
            record["upwelling"] = upwelling_flat[valid]

        rows.append(pd.DataFrame(record))

    all_data = pd.concat(rows, ignore_index=True)
    if use_iforest:
        all_data["Zone"] = all_data["upwelling"].map({True: "Inside Upwelling", False: "Outside Upwelling"})
    all_data["Season"] = all_data["month"].apply(
        lambda m: "Upwelling Months (May-Sep)" if 5 <= m <= 9 else "Non-upwelling Months (Oct-Apr)"
    )
    return all_data