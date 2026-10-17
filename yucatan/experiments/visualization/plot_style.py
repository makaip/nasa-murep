"""Shared axis helpers so every figure uses clean, round axis limits."""

import numpy as np
from matplotlib.ticker import MaxNLocator

# Allowed tick spacings (times a power of ten), e.g. 0.25 -> limits like 6.25 or 6.5.
NICE_STEPS = [1, 2, 2.5, 5, 10]
NICE_NBINS = 6


def nice_limits(lo, hi, nbins=NICE_NBINS):
    """Expand (lo, hi) outward to the nearest clean tick multiples."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    ticks = MaxNLocator(nbins=nbins, steps=NICE_STEPS).tick_values(lo, hi)
    return float(ticks[0]), float(ticks[-1])


def nice_log_limits(lo, hi):
    """Expand (lo, hi) outward to whole decades for log-scaled axes."""
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0 or hi <= 0:
        return None
    return 10.0 ** np.floor(np.log10(lo)), 10.0 ** np.ceil(np.log10(hi))


def apply_nice_yaxis(ax, lo, hi, nbins=NICE_NBINS):
    limits = nice_limits(lo, hi, nbins)
    if limits is None:
        return None
    ax.set_ylim(*limits)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, steps=NICE_STEPS))
    return limits


def apply_nice_xaxis(ax, lo, hi, nbins=NICE_NBINS):
    limits = nice_limits(lo, hi, nbins)
    if limits is None:
        return None
    ax.set_xlim(*limits)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, steps=NICE_STEPS))
    return limits
