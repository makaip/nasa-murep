"""Single depth cutoff shared by every Yucatan analysis and figure script.

Pixels with GEBCO depth < MIN_DEPTH_M (bathymetry > -MIN_DEPTH_M), land and pixels without
bathymetry are excluded from BOTH the inside (upwelling) and the outside (non-upwelling)
group in every statistic and figure ("deep-only" convention, rerun 2026-11-13).
"""

import numpy as np

MIN_DEPTH_M = 10.0
BATHY_THRESHOLD = -MIN_DEPTH_M          # bathymetry <= -10 m  <=>  depth >= 10 m
SHALLOW_LABEL = f"Shallow (< {MIN_DEPTH_M:g} m)"


def deep_mask(bathymetry, min_depth_m=MIN_DEPTH_M):
    b = np.asarray(bathymetry, dtype=float)
    return ~np.isnan(b) & (b <= -min_depth_m)
