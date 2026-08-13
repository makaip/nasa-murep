"""Pipeline orchestration for multi-variable satellite data pickling."""

import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd

from pickler_config import (
    BATHYMETRY_FILE,
    BATHYMETRY_VAR,
    END_YEAR,
    LAT_BINS,
    LAT_MAX,
    LAT_MIN,
    LON_BINS,
    LON_MAX,
    LON_MIN,
    OUTPUT_DIR,
    OUTPUT_FILENAME,
    START_YEAR,
)
from pickler_processors import (
    process_bathymetry_data,
    process_cdom_data,
    process_chlorophyll_data,
    process_sst_data,
    process_ssl_data,
)
from pickler_utils import create_spatial_grid


def _set_field(
    day_data: Dict, date, field: str, value, n_points_field: str = None, n_points=None
) -> None:
    """Assign one variable's data for one date, warning if it overwrites an existing entry.

    A warning here means two different source files mapped to the same calendar
    date for the same variable (e.g. a duplicate/reprocessed granule, or a stray
    file for one day sitting in another day's directory). That's a real data
    question worth investigating -- it should never be silently absorbed.
    """
    entry = day_data.setdefault(date, {})
    if field in entry:
        print(f"  Warning: multiple source files map to {date.date()} for '{field}' -- using the last one")
    entry[field] = value
    if n_points_field is not None:
        entry[n_points_field] = n_points


def _merge_records(
    sst_results: List[Dict],
    chlor_results: List[Dict],
    cdom_results: List[Dict],
    ssl_results: List[Dict],
) -> List[Dict]:
    """Merge per-product records into a single list of per-date records.

    This must be called ONCE on the complete result lists spanning every year,
    not per year. Merging per year was the source of the duplicate-date bug:
    each year's pass flushed its own date-keyed dict into `all_data`
    independently, so a boundary file (e.g. a stray Dec-31 granule sitting in
    the next year's directory) could produce a second, separate row for a date
    already emitted by the previous year's pass -- same date, two rows, which
    is what showed up as nested/duplicate-index data downstream.
    """
    day_data: Dict = {}

    for record in sst_results:
        _set_field(day_data, record["date"], "sst", record["sst"], "sst_n_points", record["n_points"])

    for record in chlor_results:
        _set_field(
            day_data, record["date"], "chlorophyll", record["chlorophyll"],
            "chlor_n_points", record["n_points"],
        )

    for record in cdom_results:
        date = record["date"]

        if "cdom_412" in record:
            _set_field(
                day_data, date, "cdom_412", record["cdom_412"],
                "cdom_412_n_points", record.get("cdom_412_n_points", np.nan),
            )

        if "cdom_slope_275_295" in record:
            _set_field(
                day_data, date, "cdom_slope_275_295", record["cdom_slope_275_295"],
                "cdom_slope_275_295_n_points", record.get("cdom_slope_275_295_n_points", np.nan),
            )

        if "cdom_slope_300_600" in record:
            _set_field(
                day_data, date, "cdom_slope_300_600", record["cdom_slope_300_600"],
                "cdom_slope_300_600_n_points", record.get("cdom_slope_300_600_n_points", np.nan),
            )

    for record in ssl_results:
        _set_field(day_data, record["date"], "ssl", record["ssl"], "ssl_n_points", record["n_points"])

    merged_records = []
    for date, data in day_data.items():
        merged = {"date": date}
        merged.update(data)
        merged_records.append(merged)
    return merged_records


def run_pipeline() -> None:
    """Main processing pipeline."""
    print("\n" + "=" * 70)
    print("Multi-Variable Satellite Data Pickler")
    print("=" * 70)
    print(f"Time range: {START_YEAR}-{END_YEAR}")
    print(f"Region: [{LON_MIN}, {LAT_MIN}] to [{LON_MAX}, {LAT_MAX}]")
    print(f"Grid resolution: {LAT_BINS} x {LON_BINS}")
    print(f"Output: {os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)}")
    print("=" * 70)

    lat_edges, lon_edges, lat_centers, lon_centers = create_spatial_grid()
    print("\nSpatial grid created:")
    print(f"  Latitude bins: {len(lat_centers)} ({LAT_MIN}° to {LAT_MAX}°)")
    print(f"  Longitude bins: {len(lon_centers)} ({LON_MIN}° to {LON_MAX}°)")

    bathymetry_grid = process_bathymetry_data(lat_edges, lon_edges)

    all_sst_results: List[Dict] = []
    all_chlor_results: List[Dict] = []
    all_cdom_results: List[Dict] = []
    all_ssl_results: List[Dict] = []

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\n{'#' * 70}")
        print(f"# PROCESSING YEAR {year}")
        print(f"{'#' * 70}")

        sst_results = process_sst_data(year, lat_edges, lon_edges)
        chlor_results = process_chlorophyll_data(year, lat_edges, lon_edges)
        cdom_results = process_cdom_data(year, lat_edges, lon_edges)
        ssl_results = process_ssl_data(year, lat_edges, lon_edges)

        all_sst_results.extend(sst_results)
        all_chlor_results.extend(chlor_results)
        all_cdom_results.extend(cdom_results)
        all_ssl_results.extend(ssl_results)

        print(
            f"\nYear {year} summary: {len(sst_results)} SST, {len(chlor_results)} chlorophyll, "
            f"{len(cdom_results)} CDOM, {len(ssl_results)} SSL records collected"
        )

    print(f"\n{'=' * 70}")
    print("Merging all years into unified per-date records...")
    print(f"{'=' * 70}")

    all_data = _merge_records(all_sst_results, all_chlor_results, all_cdom_results, all_ssl_results)

    if not all_data:
        print("ERROR: No data collected! Check file paths and data availability.")
        return

    all_data.sort(key=lambda x: x["date"])

    df = pd.DataFrame(all_data)
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    if df.index.has_duplicates:
        dupes = sorted(set(df.index[df.index.duplicated()]))
        raise ValueError(
            f"Duplicate dates remain in the final index after merging: {dupes}. "
            "This should not happen after a single global merge -- check "
            "extract_date_from_filepath output for these dates before proceeding."
        )

    print("\nDataFrame created:")
    print(f"  Total records: {len(df)}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Columns: {list(df.columns)}")
    print("\nData availability:")

    for col, label in [
        ("sst", "SST"),
        ("chlorophyll", "Chlorophyll"),
        ("cdom_412", "CDOM 412"),
        ("cdom_slope_275_295", "CDOM slope S275:295"),
        ("cdom_slope_300_600", "CDOM slope S300:600"),
        ("ssl", "SSL"),
    ]:
        if col in df.columns:
            print(f"  {label}: {df[col].notna().sum()} days")
        else:
            print(f"  {label}: 0 days (no data collected)")

    print("\nStatic layer availability:")
    if bathymetry_grid is not None:
        print(f"  Bathymetry: available ({bathymetry_grid.shape[0]} x {bathymetry_grid.shape[1]})")
    else:
        print("  Bathymetry: unavailable")

    metadata = {
        "lat_centers": lat_centers,
        "lon_centers": lon_centers,
        "lat_edges": lat_edges,
        "lon_edges": lon_edges,
        "bbox": (LON_MIN, LON_MAX, LAT_MIN, LAT_MAX),
        "grid_shape": (LAT_BINS, LON_BINS),
        "bathymetry": bathymetry_grid,
        "bathymetry_file": BATHYMETRY_FILE,
        "bathymetry_var": BATHYMETRY_VAR,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    print(f"\n{'=' * 70}")
    print(f"Saving pickle file to: {output_path}")
    print(f"{'=' * 70}")

    with open(output_path, "wb") as f:
        pickle.dump({"data": df, "metadata": metadata}, f)

    print("\n✓ SUCCESS! Pickle file saved.")
    print(f"  File size: {os.path.getsize(output_path) / (1024 ** 2):.2f} MB")
    print("\nTo load the data:")
    print("  import pickle")
    print(f"  with open('{output_path}', 'rb') as f:")
    print("      data = pickle.load(f)")
    print("  df = data['data']")
    print("  metadata = data['metadata']")
    print("\n" + "=" * 70)