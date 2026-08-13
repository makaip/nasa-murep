"""Shared helper utilities for the multi-variable satellite pickler."""

import os
import re
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
from scipy.stats import binned_statistic_2d

from pickler_config import (
    CDOM_412_COEFFS,
    LAT_BINS,
    LAT_MAX,
    LAT_MIN,
    LON_BINS,
    LON_MAX,
    LON_MIN,
    MAX_CDOM,
    MIN_CDOM,
    S275_295_COEFFS,
    S300_600_COEFFS,
)


def create_spatial_grid() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create consistent spatial grid for binning."""
    lat_edges = np.linspace(LAT_MIN, LAT_MAX, LAT_BINS + 1)
    lon_edges = np.linspace(LON_MIN, LON_MAX, LON_BINS + 1)

    lat_centers = lat_edges[:-1] + np.diff(lat_edges) / 2
    lon_centers = lon_edges[:-1] + np.diff(lon_edges) / 2
    return lat_edges, lon_edges, lat_centers, lon_centers


def extract_date_from_filepath(filepath: str) -> Optional[datetime]:
    """Extract date from satellite filename at normalized midnight time."""
    basename = os.path.basename(filepath)
    patterns = [
        r"(\d{4})(\d{2})(\d{2})",  # YYYYMMDD
        r"(\d{4})-(\d{2})-(\d{2})",  # YYYY-MM-DD
        r"(\d{4})(\d{3})",  # YYYYDDD
    ]

    for pattern in patterns:
        match = re.search(pattern, basename)
        if not match:
            continue
        try:
            if len(match.groups()) == 3:
                year, month, day = map(int, match.groups())
                return datetime(year, month, day)
            if len(match.groups()) == 2:
                year, day_of_year = map(int, match.groups())
                return datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
        except ValueError:
            continue
    return None


def bin_data_to_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    lat_edges: np.ndarray,
    lon_edges: np.ndarray,
) -> np.ndarray:
    """Bin scattered data to 2D grid using mean statistic."""
    binned_data, _, _, _ = binned_statistic_2d(
        lat,
        lon,
        values,
        statistic="mean",
        bins=[lat_edges, lon_edges],
        range=[[LAT_MIN, LAT_MAX], [LON_MIN, LON_MAX]],
    )
    return binned_data


def calculate_cdom(rrs_443: np.ndarray, rrs_547: np.ndarray) -> np.ndarray:
    """Calculate CDOM(412) using the MLR algorithm (Mannino et al., MODIS bands).

    Ln[aCDOM(412)] = B0 + B1*Ln[Rrs_443] + B2*Ln[Rrs_547]
    """
    cdom_values = np.full_like(rrs_443, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        valid_mask = (
            ~np.isnan(rrs_443)
            & ~np.isnan(rrs_547)
            & (rrs_443 > 0)
            & (rrs_547 > 0)
        )

        if np.any(valid_mask):
            ln_rrs_443 = np.log(rrs_443[valid_mask])
            ln_rrs_547 = np.log(rrs_547[valid_mask])

            ln_cdom_412 = (
                CDOM_412_COEFFS["B0"]
                + CDOM_412_COEFFS["B1"] * ln_rrs_443
                + CDOM_412_COEFFS["B2"] * ln_rrs_547
            )
            cdom_412 = np.exp(ln_cdom_412)
            cdom_412 = np.where(
                (cdom_412 >= MIN_CDOM) & (cdom_412 <= MAX_CDOM),
                cdom_412,
                np.nan,
            )
            cdom_values[valid_mask] = cdom_412

    return cdom_values


def calculate_cdom_slopes(rrs_443: np.ndarray, rrs_547: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate CDOM spectral slopes S_275:295 and S_300:600 (Mannino et al.)."""
    s275_295 = np.full_like(rrs_443, np.nan)
    s300_600 = np.full_like(rrs_443, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        valid_mask = (
            ~np.isnan(rrs_443)
            & ~np.isnan(rrs_547)
            & (rrs_443 > 0)
            & (rrs_547 > 0)
        )

        if np.any(valid_mask):
            ln_rrs_443 = np.log(rrs_443[valid_mask])
            ln_rrs_547 = np.log(rrs_547[valid_mask])

            ln_s275_295 = (
                S275_295_COEFFS["B0"]
                + S275_295_COEFFS["B1"] * ln_rrs_443
                + S275_295_COEFFS["B2"] * ln_rrs_547
            )
            ln_s300_600 = (
                S300_600_COEFFS["B0"]
                + S300_600_COEFFS["B1"] * ln_rrs_443
                + S300_600_COEFFS["B2"] * ln_rrs_547
            )

            s275_295[valid_mask] = np.exp(ln_s275_295)
            s300_600[valid_mask] = np.exp(ln_s300_600)

    return s275_295, s300_600