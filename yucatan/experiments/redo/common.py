"""Shared setup for the redo analysis (one consistent configuration for every method).

Setup (applies to every method):
  * data: v4 pickle (Mannino 2014 MLR CDOM, MODIS L2 ocean colour, MUR SST, GCOOS SSL,
    GEBCO 2024 bathymetry), 250x250 grid, 2010-01-01 .. 2019-12-31
  * depth mask: GEBCO bathymetry <= -10 m (depth >= 10 m); shallower pixels / land are
    never classified and are excluded from both "inside" and "outside" populations
  * classification: one zone map per month-year (from that month's mean fields) plus a
    2010-2019 monthly climatology (12 maps)
  * daily pixels are labelled by the zone map of their own month-year
  * statistics domain: Yucatan-shelf bounding box (same box as the existing figure
    scripts) AND depth >= 10 m. Classification itself runs on the full deep grid.
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

YUCATAN_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(YUCATAN_DIR / "experiments"))
import depth_cutoff  # noqa: E402  (shared 10 m cutoff)
DATA_PATH = Path(r"E:\satdata\Custom\yucatan_sst_chlor_cdomslope_ssl_bathy_2010-2019_v4.pkl")
MODEL_PATH = YUCATAN_DIR / "experiments" / "isolation_forest" / "models" / "iforest_pipeline5.joblib"
OUT_DIR = YUCATAN_DIR / "outputs" / "redo"
FIG_DIR = OUT_DIR / "figs"
CACHE_DIR = OUT_DIR / "cache"          # git-ignored
ZONE_DIR = CACHE_DIR / "zones"

YEARS = list(range(2010, 2020))
MONTHS = list(range(1, 13))
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DEPTH_THRESHOLD = depth_cutoff.BATHY_THRESHOLD   # bathymetry <= -10 m  <=> depth >= 10 m

# Upwelling-zone bounding box used by the existing violin / scatter scripts
BBOX_LON = (-92.20216, -85.61256)
BBOX_LAT = (19.57871, 22.86906)

VARS = ["sst", "ssl", "chlorophyll", "cdom_412", "cdom_slope_275_295", "cdom_slope_300_600"]

for d in (OUT_DIR, FIG_DIR, CACHE_DIR, ZONE_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _grid(value, shape):
    if value is None:
        return None
    a = np.asarray(value, dtype=float)
    return a if a.shape == shape else None


def load_stacks(force=False):
    """Daily (T, 250, 250) float32 stacks for every variable, cached as .npy."""
    meta_path = CACHE_DIR / "meta.npz"
    if not force and meta_path.exists() and all((CACHE_DIR / f"{v}.npy").exists() for v in VARS):
        meta = dict(np.load(meta_path, allow_pickle=True))
        stacks = {v: np.load(CACHE_DIR / f"{v}.npy", mmap_mode="r") for v in VARS}
        meta["dates"] = pd.DatetimeIndex(meta["dates"])
        return stacks, meta

    blob = pd.read_pickle(DATA_PATH)
    df, md = blob["data"], blob["metadata"]
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df[(df.index.year >= YEARS[0]) & (df.index.year <= YEARS[-1])].sort_index()
    shape = tuple(md["grid_shape"])
    stacks = {}
    for v in VARS:
        out = np.full((len(df),) + shape, np.nan, dtype=np.float32)
        for i, val in enumerate(df[v].to_numpy()):
            g = _grid(val, shape)
            if g is not None:
                out[i] = g
        np.save(CACHE_DIR / f"{v}.npy", out)
        stacks[v] = out
    lon, lat = np.asarray(md["lon_centers"]), np.asarray(md["lat_centers"])
    meta = {
        "dates": np.asarray(df.index.values),
        "lon": lon, "lat": lat,
        "bathymetry": np.asarray(md["bathymetry"], dtype=float),
    }
    np.savez(meta_path, **meta)
    meta["dates"] = pd.DatetimeIndex(meta["dates"])
    return stacks, meta


def deep_mask(meta):
    return depth_cutoff.deep_mask(meta["bathymetry"])


def bbox_mask(meta):
    lon2, lat2 = np.meshgrid(meta["lon"], meta["lat"])
    return ((lon2 >= BBOX_LON[0]) & (lon2 <= BBOX_LON[1])
            & (lat2 >= BBOX_LAT[0]) & (lat2 <= BBOX_LAT[1]))


def month_index(meta):
    """{(year, month): integer index array of days}."""
    d = meta["dates"]
    return {(y, m): np.where((d.year == y) & (d.month == m))[0] for y in YEARS for m in MONTHS}


def nanmean0(a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=0)


def monthly_means(stacks, meta, var):
    """{(y, m): mean map} of the raw field."""
    idx = month_index(meta)
    return {k: nanmean0(np.asarray(stacks[var][i], dtype=float)) for k, i in idx.items()}


def monthly_mean_anomaly(stacks, meta, var):
    """{(y, m): mean over days of (field - that day's whole-domain median)} (IF definition)."""
    idx = month_index(meta)
    out = {}
    for k, i in idx.items():
        s = np.asarray(stacks[var][i], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(s, axis=(1, 2), keepdims=True)
        out[k] = nanmean0(s - med)
    return out


def climatology(month_maps):
    """Mean over years of month-year maps -> {m: map}."""
    return {m: nanmean0(np.stack([month_maps[(y, m)] for y in YEARS])) for m in MONTHS}


# ---------------------------------------------------------------------------
# Zone maps: int8 arrays, 1 = inside zone, 0 = outside (classified), -1 = not classified
# ---------------------------------------------------------------------------
def save_zones(method, by_month, clim, extra=None):
    keys = [(y, m) for y in YEARS for m in MONTHS]
    arr = np.stack([by_month[k] for k in keys]).astype(np.int8)
    carr = np.stack([clim[m] for m in MONTHS]).astype(np.int8)
    np.savez_compressed(ZONE_DIR / f"{method}.npz", by_month=arr, clim=carr, **(extra or {}))


def load_zones(method):
    z = np.load(ZONE_DIR / f"{method}.npz")
    keys = [(y, m) for y in YEARS for m in MONTHS]
    by_month = {k: z["by_month"][i] for i, k in enumerate(keys)}
    clim = {m: z["clim"][i] for i, m in enumerate(MONTHS)}
    return by_month, clim


def zone_grid(inside, classified):
    g = np.full(classified.shape, -1, dtype=np.int8)
    g[classified] = 0
    g[classified & inside] = 1
    return g


def daily_labels(zones_by_month, meta):
    """(T, H, W) int8 labels: each day gets its month-year zone map."""
    d = meta["dates"]
    out = np.empty((len(d),) + meta["bathymetry"].shape, dtype=np.int8)
    for i, (y, m) in enumerate(zip(d.year, d.month)):
        out[i] = zones_by_month[(y, m)]
    return out
